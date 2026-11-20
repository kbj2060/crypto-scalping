//! 바이낸스 체결 테이프의 REST 검증·복구(scripts/live_trade_tape_collector_20260916.py, 줄 번호는 그 파일):
//! `verify_recent`(:707-736) · `backfill_loop`(:739-797) · `binance_minute_fetcher`(:827-863) · `_bn_pace`·`_bn_note_weight`(:800-824) ·
//! `TapeStore` 의 `backfill_candidates`·`live_rows_outside_gaps`·`_mark_backfill`·`replace_minute`·`mark_backfill_failed`·`verify`·
//! `unverified_minutes`(:587-705).
//! 🔴기본은 꺼져 있다(main.rs `RTCOLLECT_REST=1`). 그림자 기간엔 파이썬이 같은 REST 를 부르고 있어 켜면 거래소 호출이 두 배다.
//! 심볼마다 스레드 하나(원본 = 심볼마다 asyncio 태스크)가 자기 SQLite 연결로(WAL -- 수집 연결과 서로 안 막는다; 원본은 연결 하나를 락으로 나눴다).
//! 속도 제한은 프로세스 전체 하나(`Pace` = 원본 `_BN_PACE`). 시계는 `Clock`(시험은 가짜 시계 -- 정답지도 원본을 가짜 시계로 돌렸다).
//! 원본에 없는 것 하나: 418·429 를 받으면 scripts/binance_ban_guard.py 의 공용 파일에 해제 시각을 적고 그때까지 요청을 **보내지 않는다**
//!   (원본 테이프 수집기는 aiohttp 라 requests 를 감싸는 그 가드를 안 탄다 -- 2026-09-26 IP 밴 사고: 차단 중 두드리면 해제가 늦어진다).
use crate::tape::{bucket, Cell, TapeBuffer};
use crate::{py_float, py_int, py_truthy};
use rusqlite::{params, Connection};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// 복구 주기. 한 주기에 몇 분을 받을지는 `per_cycle`(ETH 10 · 나머지 4).
pub const BACKFILL_SECONDS: f64 = 60.0;
pub const VERIFY_SECONDS: f64 = 300.0;
pub const MAX_ATTEMPTS: i64 = 3;
/// 🔴aggTrades 시각 검색은 «최근 2일»만(-4166, 2026-09-24 실측).
pub const LOOKBACK_S: i64 = 47 * 3600;
/// `unverified_minutes` 기본 창(원본 BACKFILL_LOOKBACK_S = 7일).
pub const VERIFY_LOOKBACK_S: i64 = 7 * 86400;
pub const TOL: f64 = 1e-4;
/// 2026-11-01: 심볼별 per_cycle 로만 묶으면 합이 IP 가중치(봇·대시보드·수집기가 한 한도 2,400/분)를 모른다 -- aggTrades(가중치 20)는
/// 프로세스 전체가 2초 간격(분당 ≤ 30회 = 600)으로, 응답 헤더의 IP 1분 사용량이 1500 이상이면 다음 분까지 양보(복구 없이도 ~1,200 이 나간다).
pub const AGG_GAP_S: f64 = 2.0;
pub const WEIGHT_YIELD: i64 = 1500;
/// 🔴세션이 시한 없음(WS 때문)이라 반쯤 열린 REST 연결 하나가 수신 루프를 멈췄다(2026-10-31 23:01, 선물 5심볼 14분) -- 요청마다 20초.
pub const TIMEOUT: Duration = Duration::from_secs(20);
pub const SOURCE: &str = "binance rest aggTrades";
pub const NOTE: &str = "buy_max/sell_max NULL(REST 에 개별 체결 최대가 없다) · 주문=aggTrade";

/// 단조 초·벽시계 초·잠. 가짜면 잠이 두 시계를 같이 민다(원본 정답지의 가짜 `time`·`asyncio.sleep` 과 같은 덧셈).
pub struct Clock(Option<Mutex<(f64, f64)>>);

impl Clock {
    pub fn real() -> Self {
        Clock(None)
    }
    pub fn fake(mono: f64, wall: f64) -> Self {
        Clock(Some(Mutex::new((mono, wall))))
    }
    pub fn is_fake(&self) -> bool {
        self.0.is_some()
    }
    pub fn mono(&self) -> f64 {
        self.0.as_ref().map_or_else(crate::net::mono_s, |m| m.lock().unwrap().0)
    }
    pub fn wall(&self) -> f64 {
        self.0.as_ref().map_or_else(wall_s, |m| m.lock().unwrap().1)
    }
    pub fn sleep(&self, s: f64) {
        match &self.0 {
            Some(m) => {
                let mut g = m.lock().unwrap();
                g.0 += s;
                g.1 += s;
            }
            None => std::thread::sleep(Duration::from_secs_f64(s.max(0.0))),
        }
    }
}

pub fn wall_s() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// 프로세스 전체 «다음 aggTrades 슬롯»(단조 초).
pub type Pace = Arc<Mutex<f64>>;

/// 테이프 행(초, 가격빈, 16칸). NaN = NULL.
pub type Row = (i64, i64, Cell);

/// 실패: 상태 코드가 있는 것(raise_for_status) · 그 밖(시간초과·연결·JSON·밴 중이라 안 보냄 -- 원본 `status` 없는 예외).
#[derive(Debug)]
pub enum Fail {
    Status(u16),
    Other(String),
}

struct Resp {
    status: u16,
    weight: Option<String>,
    body: String,
}

fn at<'a>(v: &'a Value, i: usize) -> Result<&'a Value, Fail> {
    v.as_array().and_then(|a| a.get(i)).ok_or_else(|| Fail::Other("IndexError/TypeError".into()))
}

fn key<'a>(v: &'a Value, k: &str) -> Result<&'a Value, Fail> {
    v.get(k).ok_or_else(|| Fail::Other(format!("KeyError: '{k}'")))
}

fn int(v: &Value) -> Result<i64, Fail> {
    py_int(v).map_err(|e| Fail::Other(e.0))
}

fn float(v: &Value) -> Result<f64, Fail> {
    py_float(v).map_err(|e| Fail::Other(e.0))
}

/// 밴 해제까지 남은 초(binance_ban_guard.ban_remaining).
pub fn ban_left(file: &Path, now: f64) -> f64 {
    std::fs::read_to_string(file).ok().and_then(|s| s.trim().parse::<f64>().ok()).map_or(0.0, |u| (u - now).max(0.0))
}

/// binance_ban_guard.note: 418 = 본문 «banned until <13자리 ms>», 429 = Retry-After(없으면 30초 · 418 은 120초). 더 긴 기록은 안 덮는다.
pub fn ban_note(file: &Path, status: u16, body: &str, retry_after: Option<&str>, now: f64) {
    if status != 418 && status != 429 {
        return;
    }
    let banned = body.find("banned until ").and_then(|i| {
        let d = &body[i + 13..];
        (d.len() >= 13 && d[..13].bytes().all(|b| b.is_ascii_digit())).then(|| d[..13].parse::<f64>().unwrap() / 1000.0)
    });
    let until = banned.unwrap_or_else(|| match retry_after.filter(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit())) {
        Some(r) => now + r.parse::<f64>().unwrap(),
        None => now + if status == 418 { 120.0 } else { 30.0 },
    });
    if until <= now + ban_left(file, now) {
        return;
    }
    if let Some(d) = file.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, format!("{until:.3}\n")).and_then(|_| std::fs::rename(&tmp, file)).is_err() {
        eprintln!("⚠️밴 기록 실패: {}", file.display());
    }
}

/// 한 심볼의 REST 검증·복구.
pub struct Rest {
    con: Connection,
    pub symbol: String,
    bucket: f64,
    per_cycle: usize,
    klines_url: String,
    agg_url: String,
    agent: ureq::Agent,
    clock: Arc<Clock>,
    pace: Pace,
    ban_file: PathBuf,
    /// 가짜 시계일 때만: 나간 요청 (단조, 벽, 경로?쿼리) -- 정답지의 «요청 열»과 대조.
    pub sent: Vec<(f64, f64, String)>,
}

impl Rest {
    /// db = 수집기가 이미 연(표가 있는) hot 파일. 연결은 따로 연다.
    #[allow(clippy::too_many_arguments)]
    pub fn new(db: &Path, symbol: &str, klines_url: &str, agg_url: &str, clock: Arc<Clock>, pace: Pace, ban_file: PathBuf, timeout: Duration) -> rusqlite::Result<Self> {
        let con = Connection::open(db)?;
        con.busy_timeout(Duration::from_secs(30))?;
        con.execute_batch("PRAGMA synchronous = NORMAL")?;
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(timeout)).http_status_as_error(false).proxy(None).build().into();
        Ok(Rest {
            con,
            symbol: symbol.to_string(),
            bucket: bucket(symbol),
            // 🔴ETH 외는 한 주기 4분: 망이 끊겼다 붙으면 모든 심볼이 동시에 복구를 시작하는데 BTC 한 분은 aggTrades 여러 쪽(쪽당 20)이다.
            per_cycle: if symbol == "ethusdt" { 10 } else { 4 },
            klines_url: klines_url.into(),
            agg_url: agg_url.into(),
            agent,
            clock,
            pace,
            ban_file,
            sent: vec![],
        })
    }

    /// 영원히: 60초마다 복구 한 주기, 300초마다 1분봉 대조(원본은 대조를 WS 수신 루프 안에서 기다린다 -- 여기는 이 스레드에서).
    pub fn run(mut self) {
        let mut verified = self.clock.mono();
        loop {
            self.clock.sleep(BACKFILL_SECONDS);
            if self.clock.mono() - verified >= VERIFY_SECONDS {
                verified = self.clock.mono();
                if let Err(e) = self.verify_recent() {
                    eprintln!("[{}] 1분봉 대조 실패(수집은 계속): {e:?}", self.symbol);
                }
            }
            self.backfill_cycle();
        }
    }

    fn get(&mut self, url: &str, q: &str) -> Result<Resp, Fail> {
        let (mono, wall) = (self.clock.mono(), self.clock.wall());
        let left = ban_left(&self.ban_file, wall);
        if left > 0.0 {
            return Err(Fail::Other(format!("binance IP ban: {left:.0}s left -- request not sent")));
        }
        let target = format!("{url}?{q}");
        if self.clock.is_fake() {
            let path = target.splitn(4, '/').nth(3).map_or(String::new(), |p| format!("/{p}"));
            self.sent.push((mono, wall, path));
        }
        let mut r = match self.agent.get(&target).call() {
            Ok(r) => r,
            Err(ureq::Error::Timeout(_)) => return Err(Fail::Other("TimeoutError".into())),
            Err(e) => return Err(Fail::Other(format!("ClientError: {e}"))),
        };
        let h = |n: &str| r.headers().get(n).and_then(|v| v.to_str().ok()).map(str::to_string);
        let (status, weight, retry_after) = (r.status().as_u16(), h("x-mbx-used-weight-1m"), h("retry-after"));
        let body = r.body_mut().read_to_string().map_err(|e| Fail::Other(format!("ClientPayloadError: {e}")))?;
        ban_note(&self.ban_file, status, &body, retry_after.as_deref(), self.clock.wall());
        Ok(Resp { status, weight, body })
    }

    /// `_bn_note_weight`: IP 1분 사용량이 문턱 이상이면 다음 분(+1초)까지 슬롯을 민다.
    fn note_weight(&self, h: Option<&str>) {
        let Some(used) = h.and_then(|v| v.trim().parse::<i64>().ok()) else { return };
        if used >= WEIGHT_YIELD {
            let mut n = self.pace.lock().unwrap();
            *n = n.max(self.clock.mono() + 61.0 - self.clock.wall() % 60.0);
        }
    }

    /// `_bn_pace`: 슬롯을 먼저 예약하고 기다린다 -- 동시에 깬 심볼이 같은 슬롯을 안 잡는다.
    fn wait_slot(&self) {
        let wait = {
            let mut n = self.pace.lock().unwrap();
            let now = self.clock.mono();
            let w = *n - now;
            *n = n.max(now) + AGG_GAP_S;
            w
        };
        if wait > 0.0 {
            self.clock.sleep(wait);
        }
    }

    fn checked(&self, r: Resp) -> Result<Value, Fail> {
        self.note_weight(r.weight.as_deref());
        if r.status >= 400 {
            return Err(Fail::Status(r.status));
        }
        serde_json::from_str(&r.body).map_err(|e| Fail::Other(format!("JSONDecodeError: {e}")))
    }

    /// `binance_minute_fetcher.fetch`: 1분봉이 확정됐으면 aggTrades 로 그 분을 다시 만든다. Ok(None) = 아직 확정 전(다음 주기로).
    /// aggTrade = 테이커 주문 조각 · 🔴buy_max/sell_max 는 NULL(«그 초의 최대 개별 체결»은 REST 에 없다 -- 0 이 아니다: 「없었다」≠「모른다」).
    pub fn fetch(&mut self, m: i64) -> Result<Option<(Vec<Row>, f64)>, Fail> {
        let (start, end) = (m * 1000, m * 1000 + 59_999);
        let sym = self.symbol.to_uppercase();
        let (ku, au) = (self.klines_url.clone(), self.agg_url.clone());
        let r = self.get(&ku, &format!("symbol={sym}&interval=1m&startTime={start}&limit=1"))?;
        let k = self.checked(r)?;
        // if not k or int(k[0][0]) != start or int(k[0][6]) >= time.time() * 1000
        if !py_truthy(&k) || int(at(at(&k, 0)?, 0)?)? != start || int(at(at(&k, 0)?, 6)?)? as f64 >= self.clock.wall() * 1000.0 {
            return Ok(None);
        }
        let mut buf = TapeBuffer::new(self.bucket, &self.symbol);
        let mut q = format!("symbol={sym}&startTime={start}&endTime={end}&limit=1000");
        loop {
            self.wait_slot();
            let r = self.get(&au, &q)?;
            let batch = self.checked(r)?;
            let Some(batch) = batch.as_array() else { return Err(Fail::Other("TypeError: aggTrades 가 목록이 아님".into())) };
            for a in batch {
                let (ts, price, qty) = (int(key(a, "T")?)?, float(key(a, "p")?)?, float(key(a, "q")?)?);
                if start <= ts && ts <= end && price > 0.0 && qty > 0.0 {
                    let sell = py_truthy(key(a, "m")?);
                    let fills = int(key(a, "l")?)? - int(key(a, "f")?)? + 1;
                    buf.add_agg(ts, price, qty, sell, fills);
                }
            }
            let last = batch.last();
            if batch.len() < 1000 || int(key(last.unwrap(), "T")?)? > end {
                break;
            }
            q = format!("symbol={sym}&fromId={}&limit=1000", int(key(last.unwrap(), "a")?)? + 1);
        }
        let mut rows = buf.take_closed(true);
        for r in &mut rows {
            (r.2[4], r.2[5]) = (f64::NAN, f64::NAN);
        }
        Ok(Some((rows, float(at(at(&k, 0)?, 5)?)?)))
    }

    /// `backfill_loop` 한 주기(잠 뒤). 공백·불일치 분을 받아 **1분봉과 1e-4 안에서 맞을 때만** 분을 통째로 교체한다(2026-09-24).
    /// 맞지 않으면 교체하지 않고 실패로 적는다(3번이면 포기 -- 그 분은 verify_1m 에 불일치로 남아 연구 쿼리가 뺀다).
    pub fn backfill_cycle(&mut self) {
        let minutes = match self.candidates() {
            Ok(m) => m,
            Err(e) => return eprintln!("[{}] 복구 대상 조회 실패: {e}", self.symbol),
        };
        let mut fixed = 0;
        for m in minutes {
            let (rows, kline_qty) = match self.fetch(m) {
                // 🔴영구 거부(예: -4166 «최근 2일만»)를 일시 오류처럼 다루면 같은 분을 1분마다 영원히 다시 받는다(2026-09-24 서버 실측).
                Err(Fail::Status(s)) if (400..500).contains(&s) && s != 418 && s != 429 => {
                    if let Err(e) = self.mark(m, SOURCE, 0, 0.0, 0.0, 0.0, false, &format!("거부 {s} · {NOTE}")) {
                        eprintln!("[{}] 복구 {m} 실패 기록 못 함: {e}", self.symbol); // ponytail: 원본은 여기서 복구 태스크가 죽는다
                    }
                    eprintln!("[{}] 복구 {m} 거부됨 {s} -- 실패로 기록(3번이면 포기)", self.symbol);
                    continue;
                }
                Err(e) => {
                    eprintln!("[{}] 복구 {m} 받기 실패(다음 주기 재시도): {e:?}", self.symbol);
                    break;
                }
                Ok(None) => continue,
                Ok(Some(x)) => x,
            };
            match self.settle(m, rows, kline_qty) {
                Ok(true) => fixed += 1,
                Ok(false) => {}
                Err(e) => eprintln!("[{}] 복구 {m} 쓰기 실패(다음 주기 재시도): {e}", self.symbol),
            }
        }
        if fixed > 0 {
            eprintln!("[{}] 복구 {fixed}분 교체 (1분봉과 일치)", self.symbol);
        }
    }

    /// ⭐라이브가 온전히 받은 초는 라이브 그대로, 공백에 걸린 초만 REST 로. 그래도 안 맞으면(공백 없이 조용히 빠진 분 등) REST 만으로 한 번 더.
    fn settle(&mut self, m: i64, rows: Vec<Row>, kline_qty: f64) -> rusqlite::Result<bool> {
        let live = self.live_rows_outside_gaps(m)?;
        let secs: HashSet<i64> = live.iter().map(|r| r.0).collect();
        let mut hybrid: Vec<Row> = live.into_iter().chain(rows.iter().filter(|r| !secs.contains(&r.0)).cloned()).collect();
        hybrid.sort_by_key(|r| (r.0, r.1));
        let (mut got, mut rel) = (0.0, 0.0);
        for (cand, how) in [(&hybrid, "라이브+REST"), (&rows, "REST")] {
            (got, rel) = qty_rel(cand, kline_qty);
            if rel.abs() <= TOL {
                self.replace_minute(m, cand, kline_qty, SOURCE, &format!("{how} · {NOTE}"))?;
                return Ok(true);
            }
        }
        self.mark(m, SOURCE, rows.len() as i64, got, kline_qty, rel, false, NOTE)?;
        eprintln!("[{}] ⚠️복구 {m} REST 합이 1분봉과 안 맞는다 rel {rel:+.4e} -- 교체 안 함", self.symbol);
        Ok(false)
    }

    /// 다시 받을 분: 공백과 겹치거나 1분봉과 안 맞은 분. 이미 맞는 분·교체한 분·3번 실패한 분, 끝난 지 3분 안 된 분(아직 flush·검사 전),
    /// 거래소가 더는 안 주는 옛 분(47시간)은 뺀다. 최근 분부터 `per_cycle` 개.
    pub fn candidates(&self) -> rusqlite::Result<Vec<i64>> {
        let now = self.clock.wall() as i64; // int(time.time())
        let lo = now - LOOKBACK_S;
        let mut st = self.con.prepare("SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ?")?;
        let gaps: Vec<(i64, i64)> = st.query_map(params![self.symbol, lo * 1000], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        let mut st = self.con.prepare("SELECT ts_min, rel_err FROM verify_1m WHERE symbol = ? AND ts_min >= ?")?;
        let ver: Vec<(i64, f64)> = st.query_map(params![self.symbol, lo], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
        let mut st = self.con.prepare("SELECT ts_min, ok, attempts FROM backfill_1m WHERE symbol = ? AND ts_min >= ?")?;
        let done: HashSet<i64> = st
            .query_map(params![self.symbol, lo], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))?
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|&(_, ok, att)| ok != 0 || att >= MAX_ATTEMPTS)
            .map(|r| r.0)
            .collect();
        let bad: HashSet<i64> = ver.iter().filter(|r| r.1.abs() > TOL).map(|r| r.0).collect();
        let good: HashSet<i64> = ver.iter().filter(|r| r.1.abs() <= TOL).map(|r| r.0).collect();
        let mut g: HashSet<i64> = HashSet::new();
        for &(a, b) in &gaps {
            g.extend((a.div_euclid(60_000) * 60..(b - 1).div_euclid(60_000) * 60 + 60).step_by(60));
        }
        let mut c: Vec<i64> = g
            .union(&bad)
            .copied()
            .filter(|m| lo <= *m && *m <= now - 180 && !done.contains(m) && (bad.contains(m) || !good.contains(m)))
            .collect();
        c.sort_unstable_by(|a, b| b.cmp(a));
        c.truncate(self.per_cycle);
        Ok(c)
    }

    /// 그 분에서 **공백과 안 겹치는 초**의 라이브 행. 🔴바이낸스 aggTrades 는 **시장 체결만** 준다(2026-09-24: 16:52 분이 1분봉보다
    /// 0.793 ETH(−0.064%) 적었는데 라이브 @trade 는 그 체결을 0초에 갖고 있었다) -- 라이브가 온전히 받은 초까지 덮으면 그 체결이 사라진다.
    pub fn live_rows_outside_gaps(&self, m: i64) -> rusqlite::Result<Vec<Row>> {
        let mut st = self.con.prepare_cached(
            "SELECT ts_sec, price_bin, buy_qty, sell_qty, buy_n, sell_n, buy_max, sell_max, retail_buy_qty, retail_sell_qty, \
             whale_buy_qty, whale_sell_qty, whale_buy_n, whale_sell_n, retail_buy_n, retail_sell_n, order_buy_n, order_sell_n \
             FROM trade_tape_1s t WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? \
             AND NOT EXISTS (SELECT 1 FROM gaps g WHERE g.symbol = t.symbol AND g.from_ms < (t.ts_sec + 1) * 1000 AND g.to_ms > t.ts_sec * 1000) \
             ORDER BY 1, 2",
        )?;
        let rows = st.query_map(params![self.symbol, m, m + 60], |r| {
            let mut c = [0.0; 16];
            for (j, x) in c.iter_mut().enumerate() {
                *x = r.get::<_, Option<f64>>(j + 2)?.unwrap_or(f64::NAN);
            }
            Ok((r.get(0)?, r.get(1)?, c))
        })?;
        rows.collect()
    }

    /// `_mark_backfill`: 그 분의 기록 한 줄을 갈아 끼운다(시도 횟수는 이어 센다). 자동커밋(원본 mark_backfill_failed 와 같다).
    #[allow(clippy::too_many_arguments)]
    fn mark(&self, m: i64, source: &str, n_rows: i64, tape_qty: f64, kline_qty: f64, rel: f64, ok: bool, note: &str) -> rusqlite::Result<()> {
        let prev: i64 = self.con.query_row(
            "SELECT coalesce(max(attempts), 0) FROM backfill_1m WHERE symbol = ? AND ts_min = ?",
            params![self.symbol, m],
            |r| r.get(0),
        )?;
        self.con.execute("DELETE FROM backfill_1m WHERE symbol = ? AND ts_min = ?", params![self.symbol, m])?;
        self.con.execute(
            "INSERT INTO backfill_1m VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
            params![self.symbol, m, source, n_rows, tape_qty, kline_qty, rel, ok as i64, prev + 1, note],
        )?;
        Ok(())
    }

    /// 그 분의 행을 **통째로** 갈아 끼운다(한 트랜잭션). 분 단위인 이유: 끊긴 경계의 초는 라이브 반쪽이 이미 들어 있어 초 단위로 메우면 겹친다.
    fn replace_minute(&self, m: i64, rows: &[Row], kline_qty: f64, source: &str, note: &str) -> rusqlite::Result<()> {
        let (tape_qty, rel) = qty_rel(rows, kline_qty);
        self.con.execute_batch("BEGIN")?;
        let r = (|| {
            self.con.execute("DELETE FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ?", params![self.symbol, m, m + 60])?;
            crate::store::insert_rows(&self.con, &self.symbol, rows)?;
            self.con.execute("DELETE FROM verify_1m WHERE symbol = ? AND ts_min = ?", params![self.symbol, m])?;
            self.con.execute(
                "INSERT INTO verify_1m VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                params![self.symbol, m, tape_qty, kline_qty, rel],
            )?;
            self.mark(m, source, rows.len() as i64, tape_qty, kline_qty, rel, true, note)?;
            self.con.execute_batch("COMMIT")
        })();
        if r.is_err() {
            let _ = self.con.execute_batch("ROLLBACK");
        }
        r
    }

    /// 검사 기록이 없는 끝난 분(최근 먼저 30개). 🔴30 이다(예전 5): 서버 시계가 0.917배라 «5분마다 최근 5분»은 11분마다 한 분을 건너뛰어
    /// 12시간 718분 중 80분이 검사된 적 없었다(2026-09-24). 🔴`- 120`: 끝난 지 몇 초 안 된 분은 마지막 초들이 아직 버퍼/flush 대기라
    /// «유실»로 영구히 찍힌다(2026-09-23 OKX 06:13 등 −2~−14% 거짓 경보). 최근 7일만 훑는다(전체를 5분마다 훑으면 느리다).
    pub fn unverified_minutes(&self) -> rusqlite::Result<Vec<i64>> {
        let now = self.clock.wall() as i64;
        let mut st = self.con.prepare_cached(
            "SELECT DISTINCT ts_sec - ts_sec % 60 AS m FROM trade_tape_1s t \
             WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? - 120 \
               AND NOT EXISTS (SELECT 1 FROM verify_1m v WHERE v.symbol = t.symbol AND v.ts_min = t.ts_sec - t.ts_sec % 60) \
             ORDER BY m DESC LIMIT ?",
        )?;
        let rows = st.query_map(params![self.symbol, now - VERIFY_LOOKBACK_S, now, 30], |r| r.get(0))?;
        rows.collect()
    }

    /// 분별 테이프 합을 1분봉과 대조해 verify_1m 에 적는다. 반환 = (분, 상대오차, 공백과 겹침).
    pub fn verify(&self, kvol: &HashMap<i64, f64>, minutes: &[i64]) -> rusqlite::Result<Vec<(i64, f64, bool)>> {
        let mut out = vec![];
        for &m in minutes {
            let Some(&kline_qty) = kvol.get(&m) else { continue };
            let tape_qty: f64 = self.con.query_row(
                "SELECT coalesce(sum(buy_qty + sell_qty), 0) FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? + 60",
                params![self.symbol, m, m],
                |r| r.get(0),
            )?;
            let rel = if kline_qty != 0.0 { (tape_qty - kline_qty) / kline_qty } else { 0.0 };
            self.con.execute("INSERT INTO verify_1m VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)", params![self.symbol, m, tape_qty, kline_qty, rel])?;
            let gapped: i64 = self.con.query_row(
                "SELECT count(*) FROM gaps WHERE symbol = ? AND from_ms < ? AND to_ms > ?",
                params![self.symbol, (m + 60) * 1000, m * 1000],
                |r| r.get(0),
            )?;
            out.push((m, rel, gapped != 0));
        }
        Ok(out)
    }

    /// `verify_recent`: 직전 분들을 1분봉 거래량과 대조. 200 이 아니면 조용히 끝(원본). 실패해도 수집은 계속.
    pub fn verify_recent(&mut self) -> Result<(), Fail> {
        let minutes = self.unverified_minutes().map_err(|e| Fail::Other(e.to_string()))?;
        let (Some(&hi), Some(&lo)) = (minutes.first(), minutes.last()) else { return Ok(()) };
        let q = format!("symbol={}&interval=1m&startTime={}&limit={}", self.symbol.to_uppercase(), lo * 1000, 1500.min((hi - lo).div_euclid(60) + 2));
        let ku = self.klines_url.clone();
        let r = self.get(&ku, &q)?;
        if r.status != 200 {
            return Ok(());
        }
        let k: Value = serde_json::from_str(&r.body).map_err(|e| Fail::Other(format!("JSONDecodeError: {e}")))?;
        let mut kvol = HashMap::new();
        for row in k.as_array().ok_or_else(|| Fail::Other("TypeError".into()))? {
            kvol.insert(int(at(row, 0)?)?.div_euclid(1000), float(at(row, 5)?)?);
        }
        let res = self.verify(&kvol, &minutes).map_err(|e| Fail::Other(e.to_string()))?;
        for (m, rel, gapped) in res.iter().filter(|r| r.1.abs() > TOL) {
            eprintln!("[{}] {}완전성 {m} rel_err {rel:+.4e}{}", self.symbol, if *gapped { "" } else { "⚠️" },
                if *gapped { " -- 기록된 공백과 겹친다(예상된 부족)" } else { " -- 체결 유실" });
        }
        Ok(())
    }
}

/// (합, 상대오차): 원본 `sum(r[2] + r[3] for r in rows)` -- 왼쪽부터 더한다. ponytail: 파이썬 3.12+ 의 sum 은 보정 합(Neumaier)이라
/// 끝자리가 다를 수 있다(개발기 3.11 · 서버 버전 미확인) -- 다르면 verify_1m·backfill_1m 의 tape_qty·rel_err 만 1ulp 갈린다.
fn qty_rel(rows: &[Row], kline_qty: f64) -> (f64, f64) {
    let got = rows.iter().fold(0.0, |s, r| s + (r.2[0] + r.2[1]));
    (got, if kline_qty != 0.0 { (got - kline_qty) / kline_qty } else { 0.0 })
}
