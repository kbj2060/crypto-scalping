//! 레벨 5분 워커의 «메시지 하나 처리» -- scripts/live_sr_lean_20261118.py `run()` 루프 몸통(:278-330)과 `refill`(:248-270)·`Filler`(:218-229).
//! 소켓·HTTP 는 모른다: 부르는 쪽(src/bin/rt_sr_lean.rs · 시험)이 aggTrade 글자, REST 호출 함수, 시계, 잠을 넘긴다 -- 같은 입력이면 같은 줄.
use crate::sr_lean::{Kline, Live};
use crate::SrLean;
use serde_json::Value;
use std::collections::VecDeque;

/// 번호 메움 REST 호출 상한(분당 · 가중치 20 × 30 = 600 ≪ IP 한도 2,400).
pub const FILL_MAX: usize = 30;
/// 1분봉 요청 간격(성공·실패와 무관하게 1분에 한 번 · 가중치 10). 시험만 줄인다.
pub const KL_GAP_S: f64 = 60.0;

/// REST: `경로?쿼리`(예: `klines?symbol=ETHUSDT&interval=1m&limit=1500`) → JSON. Err = 원본에서 aiohttp 오류·시간초과로 잡히는 것.
pub type Rest<'a> = dyn FnMut(&str) -> Result<Value, String> + 'a;

pub struct Runner<'m> {
    pub live: Live<'m>,
    last_a: Option<i64>,
    calls: VecDeque<f64>,
    kl_next: f64,
    pub kl_gap: f64,
    own_chk: Option<i64>,
    last_state: f64,
}

/// 원본이 잡지 않는 예외(깨진 JSON · 빠진 키) -- 워커가 죽고 감시 스크립트가 다시 띄운다.
#[derive(Debug)]
pub struct Fatal(pub String);

fn int(v: &Value, k: &str) -> Result<i64, Fatal> {
    v.get(k).and_then(Value::as_i64).ok_or_else(|| Fatal(format!("KeyError/TypeError: {k}")))
}
/// 파이썬 float(x): 문자열·숫자 둘 다.
fn num(v: &Value, k: &str) -> Result<f64, Fatal> {
    let x = v.get(k).ok_or_else(|| Fatal(format!("KeyError: {k}")))?;
    x.as_f64().or_else(|| x.as_str().and_then(|s| s.trim().parse().ok())).ok_or_else(|| Fatal(format!("ValueError: float({x})")))
}
fn boolean(v: &Value, k: &str) -> Result<bool, Fatal> {
    v.get(k).and_then(Value::as_bool).ok_or_else(|| Fatal(format!("KeyError/TypeError: {k}")))
}

fn klines_of(v: &Value) -> Result<Vec<Kline>, String> {
    let rows = v.as_array().ok_or("1분봉이 목록이 아님")?;
    rows.iter().map(|r| r.as_array().and_then(|r| Kline::from_rest(r)).ok_or_else(|| format!("1분봉 줄을 못 읽음: {r}"))).collect()
}

impl<'m> Runner<'m> {
    pub fn new(art: &'m SrLean) -> Self {
        Runner { live: Live::new(art), last_a: None, calls: VecDeque::new(), kl_next: 0.0, kl_gap: KL_GAP_S, own_chk: None, last_state: 0.0 }
    }

    /// `Filler.check`: 1씩 늘면 ok · 건너뛰면 빠진 구간 · 옛 번호는 dup.
    fn check(&mut self, a: i64) -> (bool, Option<(i64, i64)>) {
        if self.last_a.is_some_and(|l| a <= l) {
            return (true, None);
        }
        let gap = self.last_a.filter(|&l| a > l + 1).map(|l| (l + 1, a - 1));
        self.last_a = Some(a);
        (false, gap)
    }

    /// 빠진 묶음 번호 lo..=hi 를 REST 로 메운다. 분당 FILL_MAX 번을 넘기면 포기(구멍). Err = 원본에서 예외(→ 구멍).
    fn refill(&mut self, mut lo: i64, hi: i64, rest: &mut Rest, clock: &mut dyn FnMut() -> f64, sleep: &mut dyn FnMut(f64)) -> Result<bool, String> {
        let mut miss = 0;
        while lo <= hi {
            let now = clock();
            while self.calls.front().is_some_and(|&c| c < now - 60.0) {
                self.calls.pop_front();
            }
            if self.calls.len() >= FILL_MAX {
                return Ok(false);
            }
            self.calls.push_back(now);
            let raw = rest(&format!("aggTrades?symbol=ETHUSDT&fromId={lo}&limit=1000"))?;
            let raw = raw.as_array().ok_or("aggTrades 가 목록이 아님")?;
            let mut last_got = None;
            for r in raw {
                let a = int(r, "a").map_err(|e| e.0)?;
                if lo <= a && a <= hi {
                    let f = |e: Fatal| e.0;
                    self.live.trade(int(r, "T").map_err(f)?, num(r, "p").map_err(f)?, num(r, "q").map_err(f)?, int(r, "f").map_err(f)?, int(r, "l").map_err(f)?, boolean(r, "m").map_err(f)?);
                    last_got = Some(a);
                }
            }
            if raw.last().is_some_and(|r| r.get("a").and_then(Value::as_i64).is_some_and(|a| a >= hi)) {
                return Ok(true); // REST 가 hi 까지 봤다 -- 그 사이에 안 온 번호는 원래 없는 번호
            }
            match last_got {
                Some(a) => lo = a + 1,
                None => {
                    // REST 가 아직 못 따라왔다(재접속 직후) -- 1초씩 세 번까지 기다리고 안 되면 구멍
                    miss += 1;
                    if miss > 3 {
                        return Ok(false);
                    }
                    sleep(1.0);
                }
            }
        }
        Ok(true)
    }

    /// 메시지 하나(None = 2초 동안 수신 없음). 돌려주는 것 = 새로 쓸 생존 표시 JSON(10초마다).
    /// 원본 순서: 번호 확인 → 빠진 번호 메움(실패면 구멍) → 체결 → 1분봉 → 2분 뒤 다시 확인(실패면 이번 메시지는 여기서 끝) → 닫힌 초 → 생존 표시.
    pub fn on_msg(&mut self, msg: Option<&str>, rest: &mut Rest, clock: &mut dyn FnMut() -> f64, sleep: &mut dyn FnMut(f64)) -> Result<Option<String>, Fatal> {
        let mut top = None; // 초는 «그 뒤 초의 체결이 왔을 때»만 닫는다(벽시계로 닫으면 끊긴 연결의 빈 초로 채점한다)
        if let Some(text) = msg {
            let d: Value = serde_json::from_str(text).map_err(|e| Fatal(format!("JSONDecodeError: {e}")))?;
            let (dup, gap) = self.check(int(&d, "a")?);
            if let Some((lo, hi)) = gap {
                // 빠진 번호를 지금 체결보다 먼저 넣는다(그 초의 마지막 체결가 순서)
                let ok = self.refill(lo, hi, rest, clock, sleep).unwrap_or_else(|e| {
                    println!("번호 메움 실패 {e}");
                    false
                });
                println!("묶음 번호 빠짐 {lo}~{hi}({}건) → {}", hi - lo + 1, if ok { "메움" } else { "구멍 처리(1시간 뒤부터 다시 채점)" });
                if !ok {
                    self.live.hole(int(&d, "T")?.div_euclid(1000));
                }
            }
            if !dup {
                let t = int(&d, "T")?;
                self.live.trade(t, num(&d, "p")?, num(&d, "q")?, int(&d, "f")?, int(&d, "l")?, boolean(&d, "m")?);
                top = Some(t.div_euclid(1000) - 1);
                if self.live.t0.is_none() {
                    (self.live.t0, self.live.done) = (Some(t.div_euclid(1000)), Some(t.div_euclid(1000)));
                    println!("첫 체결 {}", t.div_euclid(1000));
                }
            }
        }
        let now = clock();
        let lv = &mut self.live;
        // 거래소 시각으로 t0 가 든 분이 닫힌 뒤에
        if let (true, Some(done), Some(t0)) = (lv.need_klines, lv.done, lv.t0) {
            if now >= self.kl_next && done % 60 >= 3 && done.div_euclid(60) * 60 > t0 {
                self.kl_next = now + self.kl_gap;
                match rest("klines?symbol=ETHUSDT&interval=1m&limit=1500").and_then(|v| klines_of(&v)) {
                    Ok(rows) if !rows.is_empty() => {
                        lv.klines(&rows, done.div_euclid(60));
                        self.own_chk = Some(done.div_euclid(60) + 1);
                        println!("1분봉 받음");
                    }
                    Ok(_) => {}
                    Err(e) => println!("1분봉 실패(1분 뒤 다시) {e}"),
                }
            }
        }
        if let (Some(oc), Some(done)) = (self.own_chk, lv.done) {
            if done.div_euclid(60) > oc && done % 60 >= 3 && !lv.need_klines {
                // 한 번(받은 지 2분 뒤): 처음 받은 1분봉의 마지막 몇 분을 다시 받아 바꾼다 -- REST 가 몇 초 늦어 직전 분이 덜 찬 값일 수 있다(11-18 실측)
                match rest("klines?symbol=ETHUSDT&interval=1m&limit=10").and_then(|v| klines_of(&v)) {
                    Ok(rows) => {
                        lv.recheck(&rows);
                        self.own_chk = None;
                    }
                    Err(e) => {
                        println!("분봉 대조 실패 {e}"); // 다음 분에 다시 -- 원본은 `continue`: 이번 메시지의 닫힌 초 처리·생존 표시를 건너뛴다
                        self.own_chk = Some(oc + 1);
                        return Ok(None);
                    }
                }
            }
        }
        if let (Some(top), Some(done)) = (top, lv.done) {
            for s in done + 1..=top {
                lv.second(s);
            }
        }
        let now = clock();
        if now - self.last_state > 10.0 {
            self.last_state = now;
            return Ok(Some(lv.state_json(now as i64)));
        }
        Ok(None)
    }
}
