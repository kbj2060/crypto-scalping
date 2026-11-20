//! OKX·Bybit 체결 테이프의 REST -- 🔴기본 꺼짐(main.rs `RTCOLLECT_REST=1`). 그림자 동안 켜면 같은 거래소 호출이 두 배다.
//! · OKX(scripts/live_okx_trade_tape_collector_20260923.py, 줄 번호는 그 파일):
//!   `assert_ct_val`(:166-186) 시작할 때 instruments 의 ctVal 이 하드코딩과 갈라졌으면 **뜬다**(종료코드 1). REST 가 안 되면 통과
//!   («다르다는 증거»가 있을 때만 멈춘다 -- 거래소 API 가 잠깐 죽었다고 수집기가 안 뜨면 그 시간은 영구 손실).
//!   `verify_recent`(:189-237) history-candles 로 1분봉 대조 · `okx_minute_fetcher`(:273-311) history-trades 로 분 복구 --
//!   복구 뼈대는 바이낸스와 같은 원본 `backfill_loop`(rest.rs `backfill_cycle`) · per_cycle 15 · 창 7일(원본 기본 BACKFILL_LOOKBACK_S).
//! · Bybit(scripts/live_bybit_trade_tape_collector_20261006.py `verify_recent` :138-155): v5 kline 1분봉 대조만(복구 없음).
//! 🔴User-Agent 가 없으면 OKX REST 가 403(2026-09-23 실측) · 요청마다 20초 시한(반쯤 열린 연결이 수신 루프를 멈췄다, :124-126).
//! 🔴원본과 다른 한 곳: 원본은 대조(verify_recent)를 **WS 수신 루프 안에서** 기다려, 대조 응답을 읽다 예외가 나면(이상한 1분봉 모양·
//!   바이낸스는 시간초과까지) 그 WS 연결이 끊기고 재연결 공백이 적힌다. 여기는 대조가 따로 스레드라 로그만 남는다(앞 조각 rest.rs 와 같다).
use crate::rest::{agent, aio_json, at, float, int, Clock, Fail, Pace, Rest, Row, Venue};
use crate::tape::TapeBuffer;
use crate::{py_str, py_truthy};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub const BASE: &str = "https://www.okx.com";
pub const BYBIT_KLINE_URL: &str = "https://api.bybit.com/v5/market/kline";
pub const UA: &str = "Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)";
pub const TIMEOUT: Duration = Duration::from_secs(20);
/// 2026-11-01 심볼 통합 뒤 복구 요청 7심볼이 동시에 나가 한도(20회/2초, IP 단위)를 넘겨 매 분 429(18시간 공백 복구가 막힘) --
/// 프로세스 전체가 요청 사이 최소 간격을 지키고, 429 를 받으면 10초 쉰다(:259-270).
pub const OKX_MIN_GAP_S: f64 = 0.15;
pub const OKX_429_PENALTY_S: f64 = 10.0;
pub const OKX_SOURCE: &str = "okx rest history-trades(type=1)";
pub const OKX_NOTE: &str = "주문 = 연속 (ts,side,px) 묶음 -- 라이브보다 주문수 ~7% 적음, 총량·체결수 정확";

/// GET → (상태, 본문 JSON). 본문은 aiohttp `r.json()` 규칙(`rest::aio_json`).
fn get_json(agent: &ureq::Agent, url: &str) -> Result<(u16, Value), Fail> {
    let mut r = match agent.get(url).header("User-Agent", UA).call() {
        Ok(r) => r,
        Err(ureq::Error::Timeout(_)) => return Err(Fail::Other("TimeoutError".into())),
        Err(e) => return Err(Fail::Other(format!("ClientError: {e}"))),
    };
    let status = r.status().as_u16();
    let ctype = r.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let body = r.body_mut().read_to_string().map_err(|e| Fail::Other(format!("ClientPayloadError: {e}")))?;
    Ok((status, aio_json(&ctype, &body)?))
}

/// `assert_ct_val`. 반환 = 거래소 값(대조 못 했으면 None). 갈라졌으면 종료코드 1(원본 SystemExit) -- 이미 쌓인 행을 재환산해야 한다.
pub fn assert_ct_val(base: &str, inst: &str) -> Option<f64> {
    let expected = crate::okx::ct_val(inst)?;
    match ct_val_of(&agent(TIMEOUT), &format!("{base}/api/v5/public/instruments?instType=SWAP&instId={inst}")) {
        Err(e) => {
            eprintln!("⚠️ctVal 대조 실패({e:?}) -- 하드코딩 {expected} 로 계속한다");
            None
        }
        Ok((got, ccy)) if (got - expected).abs() > 1e-12 => {
            eprintln!("🔴{inst} 의 ctVal 이 갈라졌다: 코드 {expected} vs 거래소 {got} ({ccy}). 이대로 쌓으면 물량이 통째로 틀린다 -- CT_VALS 를 고치고 이미 쌓인 행을 재환산할 것.");
            std::process::exit(1)
        }
        Ok((got, ccy)) => {
            eprintln!("ctVal 확인 {inst} = {got} {ccy}");
            Some(got)
        }
    }
}

/// float(body["data"][0]["ctVal"]) · str(body["data"][0].get("ctValCcy", "")) -- 상태 코드는 보지 않는다(원본 그대로).
pub fn ct_val_of(agent: &ureq::Agent, url: &str) -> Result<(f64, String), Fail> {
    let (_, body) = get_json(agent, url)?;
    let d0 = body.get("data").and_then(|d| d.get(0)).ok_or_else(|| Fail::Other("KeyError/IndexError: data[0]".into()))?;
    let got = float(d0.get("ctVal").ok_or_else(|| Fail::Other("KeyError: 'ctVal'".into()))?)?;
    let Value::Object(o) = d0 else { return Err(Fail::Other("AttributeError: .get".into())) };
    Ok((got, py_str(o.get("ctValCcy"), "")))
}

/// 열쇠 하나 꺼내기(파이썬 `x["k"]` -- 사전이 아니거나 없으면 예외).
fn k<'a>(v: &'a Value, key: &str) -> Result<&'a Value, Fail> {
    crate::rest::key(v, key)
}

/// `group_fills`(:240-256): REST 개별 체결(tradeId 순) → 라이브 `trades` 와 같은 묶음 (ts_ms, px, sz계약, sell, 체결수).
/// 2026-09-24 같은 분 WS/REST 동시 실측: 연속된 (ts, side, px) 묶음 191개가 WS 메시지와 수량까지 191/191 일치 -- 다만 WS 는 같은 키를
/// 다시 쪼개 «주문 수»는 복구한 분이 라이브보다 ~7% 적다(총량·체결수는 정확, backfill_1m.note).
pub fn group_fills(fills: &[&Value]) -> Result<Vec<(i64, f64, f64, bool, i64)>, Fail> {
    let mut out: Vec<(i64, f64, f64, bool, i64)> = vec![];
    let mut last: Option<(&Value, &Value, &Value)> = None;
    for f in fills {
        let key = (k(f, "ts")?, k(f, "side")?, k(f, "px")?);
        if Some(key) == last {
            let o = out.last_mut().unwrap();
            o.2 += float(k(f, "sz")?)?;
            o.4 += 1;
        } else {
            out.push((int(key.0)?, float(key.2)?, float(k(f, "sz")?)?, key.1.as_str() == Some("sell"), 1));
            last = Some(key);
        }
    }
    Ok(out)
}

impl Rest {
    /// OKX 체결 테이프(okx_tape.sqlite)의 REST. `pace` = 프로세스 전체 OKX 슬롯(원본 `_PACE`).
    pub fn okx(db: &Path, inst: &str, base: &str, clock: Arc<Clock>, pace: Pace, timeout: Duration) -> rusqlite::Result<Self> {
        let ct = crate::okx::ct_val(inst).expect("ctVal");
        let mut r = Rest::new(db, inst, &format!("{base}/api/v5/market/history-candles"), &format!("{base}/api/v5/market/history-trades"),
                              clock, pace, PathBuf::new(), timeout)?;
        (r.venue, r.bucket, r.per_cycle, r.lookback, r.source, r.note) =
            (Venue::Okx { ct }, crate::okx::bucket(inst), 15, crate::rest::VERIFY_LOOKBACK_S, OKX_SOURCE, OKX_NOTE);
        Ok(r)
    }

    /// Bybit 체결 테이프(bybit_tape.sqlite)의 1분봉 대조. 원본 세션 시한 20초(ClientTimeout(total=20)).
    pub fn bybit(db: &Path, symbol: &str, kline_url: &str, clock: Arc<Clock>, timeout: Duration) -> rusqlite::Result<Self> {
        let mut r = Rest::new(db, symbol, kline_url, "", clock, Arc::new(std::sync::Mutex::new(0.0)), PathBuf::new(), timeout)?;
        r.venue = Venue::Bybit;
        Ok(r)
    }

    /// OKX `get`(:277-286): 슬롯 → 요청 → 429 면 슬롯 10초 뒤로 → 400 이상 = 상태 예외 → code != "0" = 예외 → data(없으면 []).
    fn okx_get(&mut self, url: &str, q: &str) -> Result<Vec<Value>, Fail> {
        self.wait_slot();
        let r = self.get(url, q)?;
        if r.status == 429 {
            let mut n = self.pace.lock().unwrap();
            *n = n.max(self.clock.mono()) + OKX_429_PENALTY_S;
        }
        if r.status >= 400 {
            return Err(Fail::Status(r.status));
        }
        let body = aio_json(&r.ctype, &r.body)?;
        let Value::Object(b) = &body else { return Err(Fail::Other("AttributeError: .get".into())) };
        if py_str(b.get("code"), "None") != "0" {
            return Err(Fail::Other(format!("RuntimeError: okx {} {}", py_str(b.get("code"), "None"), py_str(b.get("msg"), "None"))));
        }
        match b.get("data").filter(|v| py_truthy(v)) {
            None => Ok(vec![]),
            Some(Value::Array(a)) => Ok(a.clone()),
            Some(_) => Err(Fail::Other("TypeError: data 가 목록이 아님".into())), // ponytail: 원본은 사전·글자도 일단 받는다(다음 색인에서 죽는다) -- 결과는 같다(예외)
        }
    }

    /// `okx_minute_fetcher.fetch`(:288-310): 1분봉이 확정(confirm == "1")이면 history-trades 를 tradeId(type=1)로 거꾸로 넘겨 그 분을 다시 만든다.
    /// 🔴ts(type=2)로 넘기면 같은 ms 체결이 경계에서 빠져 2026-09-24 실측 −9.4% 였다(tradeId 로는 1,853건 연속, 1분봉 volCcy 와 오차 0).
    /// 칸의 최대(buy_max·sell_max)는 묶은 주문 기준 그대로 둔다(라이브와 같은 단위 -- 원본도 NULL 로 안 바꾼다).
    pub(crate) fn fetch_okx(&mut self, m: i64, ct: f64) -> Result<Option<(Vec<Row>, f64)>, Fail> {
        let (start, end) = (m * 1000, (m + 60) * 1000);
        let (cu, tu, inst) = (self.klines_url.clone(), self.agg_url.clone(), self.symbol.clone());
        let c = self.okx_get(&cu, &format!("instId={inst}&bar=1m&after={end}&limit=1"))?;
        if c.is_empty() || int(at(&c[0], 0)?)? != start || py_str(Some(at(&c[0], 8)?), "") != "1" {
            return Ok(None);
        }
        let edge = self.okx_get(&tu, &format!("instId={inst}&type=2&after={}&limit=1", end + 1))?;
        let mut fills: Vec<Value> = vec![];
        let mut after = match edge.first() {
            Some(e) => Some(int(k(e, "tradeId")?)? + 1),
            None => None,
        };
        while let Some(a) = after {
            self.clock.sleep(0.12);
            let page = self.okx_get(&tu, &format!("instId={inst}&type=1&after={a}&limit=100"))?;
            let Some(last) = page.last() else { break };
            for f in &page {
                let ts = int(k(f, "ts")?)?;
                if start <= ts && ts < end {
                    fills.push(f.clone());
                }
            }
            after = Some(int(k(last, "tradeId")?)?);
            if int(k(last, "ts")?)? < start || page.len() < 100 {
                break;
            }
        }
        let mut keyed = fills.iter().map(|f| Ok((int(k(f, "tradeId")?)?, f))).collect::<Result<Vec<_>, Fail>>()?;
        keyed.sort_by_key(|x| x.0); // 안정 정렬(파이썬 sort 와 같다)
        let mut buf = TapeBuffer::new(self.bucket, &self.symbol);
        for (ts, px, sz, sell, n) in group_fills(&keyed.iter().map(|x| x.1).collect::<Vec<_>>())? {
            buf.add_agg(ts, px, sz * ct, sell, n);
        }
        Ok(Some((buf.take_closed(true), float(at(&c[0], 6)?)?)))
    }

    /// OKX `verify_recent`(:189-237): 묶음의 최근 분부터 **구간마다** history-candles 100분씩(최대 6번 · 사이 0.12초).
    /// 🔴`candles` 는 최근 100분뿐이라 밀린 분을 영영 못 봤다 → history-candles · 🔴confirm == "0"(형성 중인 봉)은 뺀다 · 물량은 volCcy(row[6]).
    pub(crate) fn verify_okx(&mut self) -> Result<(), Fail> {
        let minutes = self.unverified_minutes().map_err(|e| Fail::Other(e.to_string()))?;
        if minutes.is_empty() {
            return Ok(());
        }
        let mut kvol: HashMap<i64, f64> = HashMap::new();
        let mut left: Vec<i64> = minutes.clone();
        left.sort_unstable_by(|a, b| b.cmp(a));
        let (cu, inst) = (self.klines_url.clone(), self.symbol.clone());
        for _ in 0..6 {
            let Some(&anchor) = left.first() else { break };
            let body = match self.get(&cu, &format!("instId={inst}&bar=1m&limit=100&after={}", (anchor + 60) * 1000)) {
                Ok(r) if r.status != 200 => break,
                Ok(r) => match aio_json(&r.ctype, &r.body) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("[{inst}] kline 조회 실패(수집은 계속): {e:?}");
                        break;
                    }
                },
                Err(e) => {
                    eprintln!("[{inst}] kline 조회 실패(수집은 계속): {e:?}");
                    break;
                }
            };
            // body.get("data", []) -- 여기서 나는 예외는 원본에선 WS 연결을 끊는다(모듈 머리말)
            let Value::Object(b) = &body else { return Err(Fail::Other("AttributeError: .get".into())) };
            let rows: &[Value] = match b.get("data") {
                None => &[],
                Some(Value::Array(a)) => a,
                Some(_) => return Err(Fail::Other("TypeError: data".into())),
            };
            for row in rows {
                if py_str(Some(at(row, 8)?), "") == "1" {
                    kvol.insert(int(at(row, 0)?)?.div_euclid(1000), float(at(row, 6)?)?);
                }
            }
            left.retain(|m| *m < anchor - 99 * 60);
            self.clock.sleep(0.12);
        }
        if kvol.is_empty() {
            return Ok(());
        }
        let res = self.verify(&kvol, &minutes).map_err(|e| Fail::Other(e.to_string()))?;
        for (m, rel, gapped) in res.iter().filter(|r| r.1.abs() > crate::rest::TOL) {
            eprintln!("[{inst}] {}완전성 {m} rel_err {rel:+.4e}{}", if *gapped { "" } else { "⚠️" },
                if *gapped { " -- 기록된 공백과 겹친다(예상된 부족)" } else { " -- 체결 유실. OKX 일별 덤프(🔴파일 경계 UTC+8)로 덮어써야 한다" });
        }
        Ok(())
    }

    /// Bybit `verify_recent`(:138-155) 한 종목: 상태 코드를 보지 않는다 · 무엇이 틀려도 로그만(그 종목 건너뜀).
    pub(crate) fn verify_bybit(&mut self) -> Result<(), Fail> {
        let minutes = self.unverified_minutes().map_err(|e| Fail::Other(e.to_string()))?;
        let (Some(&hi), Some(&lo)) = (minutes.first(), minutes.last()) else { return Ok(()) };
        let (ku, s) = (self.klines_url.clone(), self.symbol.clone());
        let kvol = (|| -> Result<HashMap<i64, f64>, Fail> {
            let r = self.get(&ku, &format!("category=linear&symbol={s}&interval=1&start={}&end={}&limit=1000", lo * 1000, hi * 1000))?;
            let body = aio_json(&r.ctype, &r.body)?;
            let list = k(k(&body, "result")?, "list")?;
            let rows = list.as_array().ok_or_else(|| Fail::Other("TypeError: list".into()))?; // ponytail: 사전·글자 list 는 원본도 결국 예외
            rows.iter().map(|x| Ok((int(at(x, 0)?)?.div_euclid(1000), float(at(x, 5)?)?))).collect()
        })();
        let kvol = match kvol {
            Ok(k) => k,
            Err(e) => {
                eprintln!("{s} 1분봉 조회 실패(수집은 계속): {e:?}");
                return Ok(());
            }
        };
        for (m, rel, _) in self.verify(&kvol, &minutes).map_err(|e| Fail::Other(e.to_string()))?.iter().filter(|r| r.1.abs() > crate::rest::TOL && !r.2) {
            eprintln!("⚠️{s} 완전성 {m} rel_err {rel:+.4e} -- 체결 유실");
        }
        Ok(())
    }
}
