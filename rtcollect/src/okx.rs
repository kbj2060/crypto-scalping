//! OKX 체결 테이프(`scripts/live_okx_trade_tape_collector_20260923.py`, 줄 번호는 그 파일) 수집 루프 한 종목(`collect()` :314-392)
//! 에서 소켓·REST(ctVal 대조·1분봉 대조·history-trades 복구)를 뺀 것. 시계 `now` = 단조 초.
//! ⭐OKX `trades` 는 **이미 테이커 주문 단위**다(`count` = 개별 체결 수). 바이낸스 되묶기를 이식하면 이중 집계(2026-09-23 실측
//!   trades-all 4,810건 vs trades 1,780건, sz 합 동일) -- 그래서 주문 하나가 `add_agg` 로 모든 칸을 한 번에 채운다.
//! 🔴`sz` 는 계약 수다(ctVal 곱해야 기초자산) · `side` 는 테이커 측면 그대로(뒤집지 않는다).
//! ⚠️`buy_max`/`sell_max` = «가장 큰 테이커 주문»(바이낸스 표는 «가장 큰 개별 체결») -- meta `max_unit` 에 박는다.
use crate::store::TapeStore;
use crate::{py_float, py_int, py_iter, py_repr, py_truthy, Reconnect};
use rusqlite::Connection;
use serde_json::{Map, Value};
use std::rc::Rc;

pub const FLUSH_SECONDS: f64 = 5.0;
pub const RECV_TIMEOUT_S: f64 = 25.0; // 이 시간 조용하면 재연결(:123 · OKX 는 30초 무응답이면 끊는다)

/// 가격빈(:106-109) -- 바이낸스 표와 같은 뜻(대략 가격의 0.4bp). 모르는 종목 0.01.
pub fn bucket(inst: &str) -> f64 {
    match inst {
        "ETH-USDT-SWAP" | "SKHYNIX-USDT-SWAP" => 0.1,
        "BTC-USDT-SWAP" => 1.0,
        "SOL-USDT-SWAP" | "SOXL-USDT-SWAP" => 0.01,
        "XRP-USDT-SWAP" => 0.0001,
        "HYPE-USDT-SWAP" => 0.001,
        "ZEC-USDT-SWAP" | "SNDK-USDT-SWAP" | "MU-USDT-SWAP" => 0.05,
        _ => 0.01,
    }
}

/// 계약당 기초자산 수량(:115-118). 하드코딩 이유는 경계와 같다(쌓인 행의 뜻이 고정돼야 한다). 2026-11-01 OKX REST 실측 · SOXL 11-02.
/// 🔴모르는 종목은 None = 수집하지 않는다(원본 SystemExit). 원본의 «시작할 때 REST 로 대조해 다르면 뜬다»는 옮기지 않았다 --
///   원본도 REST 가 안 되면 통과시키고, 그림자 기간엔 파이썬이 같은 표로 대조한다.
pub fn ct_val(inst: &str) -> Option<f64> {
    Some(match inst {
        "ETH-USDT-SWAP" | "HYPE-USDT-SWAP" => 0.1,
        "BTC-USDT-SWAP" | "ZEC-USDT-SWAP" => 0.01,
        "SOL-USDT-SWAP" | "SNDK-USDT-SWAP" | "SOXL-USDT-SWAP" | "SKHYNIX-USDT-SWAP" | "MU-USDT-SWAP" => 1.0,
        "XRP-USDT-SWAP" => 100.0,
        _ => return None,
    })
}

/// `trades` 한 행 → (ts_ms, price, 기초자산 수량, sell, 체결수). 못 믿을 행은 None(:144-163).
pub fn parse_trade(t: &Value, ct: f64) -> Option<(i64, f64, f64, bool, i64)> {
    let t = t.as_object()?; // 사전 아님 = t["px"] TypeError -> None
    let price = py_float(t.get("px")?).ok()?;
    let contracts = py_float(t.get("sz")?).ok()?;
    let ts_ms = py_int(t.get("ts")?).ok()?;
    if !(price > 0.0 && contracts > 0.0 && ts_ms > 0) {
        return None;
    }
    let side = py_str_lower(t.get("side"));
    if side != "buy" && side != "sell" {
        return None;
    }
    // int(t.get("count") or 1) -- 못 읽으면 1
    let fills = match t.get("count").filter(|v| py_truthy(v)) {
        Some(v) => py_int(v).unwrap_or(1),
        None => 1,
    };
    Some((ts_ms, price, contracts * ct, side == "sell", fills.max(1)))
}

/// str(t.get("side", "")).lower() -- 문자열이 아니면 "buy"/"sell" 이 될 수 없다(None → "none" 등).
fn py_str_lower(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).map_or(String::new(), str::to_lowercase)
}

pub struct Collector {
    pub store: TapeStore,
    pub buffer: crate::tape::TapeBuffer,
    inst: String,
    ct: f64,
    last_ms: i64,
    first: bool,
    flushed_at: f64,
}

impl Collector {
    pub fn new(con: Rc<Connection>, inst: &str, now: f64) -> rusqlite::Result<Option<Self>> {
        let Some(ct) = ct_val(inst) else { return Ok(None) };
        let b = bucket(inst);
        let store = TapeStore::new(con, inst, b)?;
        // 단위를 데이터와 함께 남긴다(:324-331)
        store.put_meta(&format!("ct_val:{inst}"), &py_repr(ct))?;
        store.put_meta(&format!("max_unit:{inst}"), "taker_order")?;
        store.put_meta(&format!("source:{inst}"), "okx ws v5 public trades (already order-aggregated)")?;
        let last_ms = store.last_ts_ms()?;
        Ok(Some(Self { buffer: crate::tape::TapeBuffer::new(b, inst), store, inst: inst.to_string(), ct, last_ms, first: true, flushed_at: now }))
    }

    pub fn subscribe(&self) -> String {
        serde_json::json!({"op": "subscribe", "args": [{"channel": "trades", "instId": self.inst}]}).to_string()
    }

    pub fn on_connect(&mut self) {
        self.first = true;
    }

    /// :391. 진행 중인 초는 버퍼에 남는다.
    pub fn on_disconnect(&mut self) -> usize {
        let rows = self.buffer.take_closed(false);
        self.store.write(rows)
    }

    /// 글자 메시지 하나(:352-381). Err = 그 연결을 접는다(구독 거부 `break` 포함). Ok(Some(n)) = 플러시해 n행 썼다.
    /// 🔴플러시는 «체결 채널 메시지를 다 처리한 뒤»에만 본다 -- pong·깨진 JSON·이벤트·다른 채널은 `continue` 로 건너뛴다(원본 그대로).
    pub fn on_text(&mut self, now: f64, text: &str) -> Result<Option<usize>, Reconnect> {
        if text == "pong" {
            return Ok(None);
        }
        let Ok(payload) = serde_json::from_str::<Value>(text) else { return Ok(None) }; // ValueError -> continue
        let Value::Object(p) = &payload else { return Err(Reconnect("AttributeError".into())) };
        let event = p.get("event");
        if event.and_then(Value::as_str) == Some("error") {
            eprintln!("[{}] 🔴구독 거부: {:?} (code {:?})", self.inst, p.get("msg"), p.get("code"));
            return Err(Reconnect("구독 거부".into()));
        }
        if event.is_some_and(py_truthy) {
            return Ok(None); // subscribe ack 등
        }
        if channel(p)? != Some("trades") {
            return Ok(None);
        }
        for row in py_iter(p, "data")? {
            let Some((ts_ms, price, qty, sell, fills)) = parse_trade(row, self.ct) else { continue };
            if self.first {
                self.first = false;
                let (from, why) =
                    if self.last_ms != 0 { (self.last_ms, "ws_reconnect") } else { (ts_ms.div_euclid(60_000) * 60_000, "startup") };
                self.store.record_gap(from, ts_ms, why);
            }
            self.buffer.add_agg(ts_ms, price, qty, sell, fills);
            self.last_ms = ts_ms;
        }
        if now - self.flushed_at >= FLUSH_SECONDS {
            self.flushed_at = now;
            let rows = self.buffer.take_closed(false);
            return Ok(Some(self.store.write(rows)));
        }
        // 원본은 여기서 VERIFY_SECONDS(300초)마다 verify_recent(REST 1분봉 대조)를 부른다 -- 범위 밖.
        Ok(None)
    }
}

/// (payload.get("arg") or {}).get("channel") -- 참인데 사전이 아니면 AttributeError.
fn channel(p: &Map<String, Value>) -> Result<Option<&str>, Reconnect> {
    match p.get("arg").filter(|v| py_truthy(v)) {
        None => Ok(None),
        Some(Value::Object(a)) => Ok(a.get("channel").and_then(Value::as_str)),
        Some(_) => Err(Reconnect("AttributeError".into())),
    }
}

/// 정답지 입력을 원본과 같은 규칙으로 먹인다. 사건 = [단조 초, 종류, 본문]:
/// 'o' 접속 · 'f' 접속 실패 · 't' 글자 · 'w' 25초 수신 시한(= 재연결) · 'c' 글자 아닌 메시지(= 연결 끝).
/// 종목(실행)은 차례로 -- 원본도 종목마다 상태가 따로이고 파일·연결만 나눈다. 입력이 끝나면 다음 접속에서 멈춘다.
pub fn replay(runs: &[Value], path: &std::path::Path) -> rusqlite::Result<()> {
    let con = crate::store::open_db(path)?;
    for run in runs {
        let mut c = Collector::new(con.clone(), run["inst"].as_str().unwrap(), run["t0"].as_f64().unwrap())?.expect("ctVal");
        // 연결 안에서 예외가 나면 그 연결은 끝 -- 다음 사건은 'o'·'f' 다(생성기가 그렇게 만든다).
        for e in run["events"].as_array().unwrap() {
            let (t, kind, body) = (e[0].as_f64().unwrap(), e[1].as_str().unwrap(), e[2].as_str().unwrap_or(""));
            match kind {
                "o" => c.on_connect(),
                "t" => {
                    if c.on_text(t, body).is_err() {
                        c.on_disconnect();
                    }
                }
                _ => {
                    c.on_disconnect(); // 'w' · 'c' · 'f'
                }
            }
        }
    }
    Ok(())
}
