//! 바이낸스 체결 테이프(`scripts/live_trade_tape_collector_20260916.py`, 줄 번호는 그 파일) 수집 루프 한 심볼
//! (`collect()` :923-1000)에서 소켓·REST 를 뺀 것. 시계(`now`)는 부르는 쪽이 준다(단조 초 -- 원본 time.monotonic).
use crate::store::{open_db, TapeStore};
use crate::tape::{bucket, TakerOrderAggregator, TapeBuffer};
use crate::{need, py_float, py_int, py_truthy, Reconnect};
use rusqlite::Connection;
use serde_json::{Map, Value};
use std::path::Path;
use std::rc::Rc;

pub const FLUSH_SECONDS: f64 = 5.0; // 완결된 초만 쓴다 -- 한 초는 정확히 한 번 기록된다(:104)

pub struct Collector {
    pub store: TapeStore,
    pub buffer: TapeBuffer,
    orders: TakerOrderAggregator,
    spot: bool,
    last_ms: i64,
    first: bool,
    flushed_at: f64,
}

/// 파이썬 int(trade.get(k) or 0)
fn py_int_or0(o: &Map<String, Value>, k: &str) -> Result<i64, Reconnect> {
    match o.get(k) {
        Some(v) if py_truthy(v) => py_int(v),
        _ => Ok(0),
    }
}

impl Collector {
    /// market = "futures"(@trade, 되묶기) | "spot"(@aggTrade, 이미 주문 단위 -- 2026-10-01 저장 재설계 4d).
    pub fn new(con: Rc<Connection>, symbol: &str, market: &str, now: f64) -> rusqlite::Result<Self> {
        let b = bucket(symbol);
        let store = TapeStore::new(con, symbol, b)?;
        let last_ms = store.last_ts_ms()?;
        Ok(Self {
            store,
            buffer: TapeBuffer::new(b, symbol),
            orders: TakerOrderAggregator::default(),
            spot: market == "spot",
            last_ms,
            first: true,
            flushed_at: now,
        })
    }

    pub fn on_connect(&mut self) {
        self.first = true;
    }

    /// 연결이 끝났다(정상 끝·글자 아닌 메시지·예외·접속 실패 전부, :999). 진행 중인 초는 버퍼에 남아 다음 연결로 간다.
    pub fn on_disconnect(&mut self) -> usize {
        let rows = self.buffer.take_closed(false);
        self.store.write(rows)
    }

    /// 글자 메시지 하나(:955-993). 평가 순서까지 원본과 같다 -- 어디서 예외가 나느냐에 따라 버퍼·공백에 남는 것이 다르다.
    /// Ok(Some(n)) = 이번에 플러시해 n행 썼다.
    pub fn on_text(&mut self, now: f64, text: &str) -> Result<Option<usize>, Reconnect> {
        let v: Value = serde_json::from_str(text).map_err(|e| Reconnect(format!("JSONDecodeError: {e}")))?;
        let Value::Object(o) = &v else { return Err(Reconnect("AttributeError: .get".into())) };
        let want = if self.spot { "aggTrade" } else { "trade" };
        if o.get("e").and_then(Value::as_str) != Some(want) {
            return Ok(None);
        }
        let price = py_float(need(o, "p")?)?;
        let qty = py_float(need(o, "q")?)?;
        if !(price > 0.0 && qty > 0.0) {
            // 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다(2026-09-16 실측 12,312건 중 37건=0.3%).
            // 합계는 안 틀리지만 가격 0 자리에 빈이 생긴다 -- 받는 쪽에서 막는다.
            return Ok(None);
        }
        let ts_ms = py_int(need(o, "T")?)?;
        if self.first {
            self.first = false;
            // 처음 켠 판이면 그 «분의 시작부터 여기까지»가 안 받은 구간이다 -- 안 적으면 완전성 검사가 유실로 오해한다.
            self.orders.reset(); // 끊김 전에 열려 있던 묶음은 버린다
            let (from, why) =
                if self.last_ms != 0 { (self.last_ms, "ws_reconnect") } else { (ts_ms.div_euclid(60_000) * 60_000, "startup") };
            self.store.record_gap(from, ts_ms, why);
        }
        let sell = py_truthy(need(o, "m")?);
        if self.spot {
            // aggTrade = 이미 테이커 주문 하나(대시보드 현물 레인과 같은 규칙). 체결 수 = l-f+1
            let fills = (py_int_or0(o, "l")? - py_int_or0(o, "f")? + 1).max(1);
            self.buffer.add_agg(ts_ms, price, qty, sell, fills);
        } else {
            self.buffer.add(ts_ms, price, qty, sell);
            let tid = match o.get("t") {
                None | Some(Value::Null) => None,
                Some(t) => Some(py_int(t)?), // 칸엔 이미 들어갔다 -- 원본도 여기서 예외면 묶음만 빠진다
            };
            if let Some((p, q, ms, s)) = self.orders.add(price, qty, ts_ms, sell, tid) {
                self.buffer.add_order(ms, p, q, s);
            }
        }
        self.last_ms = ts_ms;
        if now - self.flushed_at >= FLUSH_SECONDS {
            self.flushed_at = now;
            let rows = self.buffer.take_closed(false);
            return Ok(Some(self.store.write(rows)));
        }
        // 원본은 여기서 VERIFY_SECONDS(300초)마다 verify_recent(REST 1분봉 대조)를 부른다 -- 이번 범위 밖.
        Ok(None)
    }
}

/// 정답지 입력을 원본과 같은 연결 규칙으로 먹인다(tools/gen_tape_fixture.py 의 가짜 웹소켓과 같은 뜻).
/// 사건 = [recv, "t"|"c"|"f", 본문]: 't' 글자 메시지 · 'c' 글자 아닌 메시지(연결 끝) · 'f' 접속 실패(연결 사이에만).
/// 입력이 끝나면 다음 접속에서 멈춘다(마지막 플러시 없음 -- 파이썬 쪽도 CancelledError 로 끝난다).
/// `on_flush(recv, n)` = 플러시마다(속도 측정용).
pub fn replay(runs: &[Value], dir: &Path, mut on_flush: impl FnMut(f64, usize)) -> rusqlite::Result<()> {
    let mut dbs: std::collections::HashMap<&str, Rc<Connection>> = Default::default();
    for run in runs {
        let market = run["market"].as_str().unwrap();
        let file = if market == "spot" { "binance_spot_tape.sqlite" } else { "binance_tape.sqlite" };
        let con = match dbs.get(file) {
            Some(c) => c.clone(),
            None => dbs.entry(file).or_insert(open_db(&dir.join(file))?).clone(),
        };
        let mut c = Collector::new(con, run["symbol"].as_str().unwrap(), market, run["t0"].as_f64().unwrap())?;
        let ev = run["events"].as_array().unwrap();
        let mut i = 0;
        while i < ev.len() {
            if ev[i][1] == "f" {
                i += 1;
                c.on_disconnect();
                continue;
            }
            c.on_connect();
            while i < ev.len() {
                match ev[i][1].as_str().unwrap() {
                    "f" => break,
                    "c" => {
                        i += 1;
                        break;
                    }
                    _ => {
                        let now = ev[i][0].as_f64().unwrap();
                        i += 1;
                        match c.on_text(now, ev[i - 1][2].as_str().unwrap()) {
                            Ok(Some(n)) => on_flush(now, n),
                            Ok(None) => {}
                            Err(_) => break,
                        }
                    }
                }
            }
            c.on_disconnect();
        }
    }
    Ok(())
}
