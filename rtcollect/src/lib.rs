//! 파이썬 수집기를 «같은 입력 → 디스크에 같은 행»으로 옮긴 것. 수집기마다 모듈 하나, 소켓은 `net`(실행 파일만 쓴다).
//! - `binance`   = scripts/live_trade_tape_collector_20260916.py   (binance_tape.sqlite · binance_spot_tape.sqlite)
//! - `bybit_liq` = scripts/live_bybit_liquidation_collector_20261003.py (bybit_liq.sqlite)
//! - `bybit_tape`= scripts/live_bybit_trade_tape_collector_20261006.py  (bybit_tape.sqlite + bybit_ctx.sqlite)
//! - `okx`       = scripts/live_okx_trade_tape_collector_20260923.py    (okx_tape.sqlite)
//! 세 체결 테이프는 원본처럼 `tape`(칸 규칙)·`store`(표·쓰기)를 나눠 쓴다. 정답지 대조 = tests/*_parity.rs.
//!
//! 옮기지 않은 것(REST): 1분봉 대조(verify_1m 쓰기) · aggTrades/history-trades 복구(backfill_1m) · OKX ctVal REST 대조.
//! 표는 원본과 같이 만들되 쓰지 않는다 -- 그림자 대조 도구(tools/shadow_tape.py)가 파이썬 쪽 두 표로 «REST 가 채운 분»을 뺀다.
pub mod binance;
pub mod bybit_liq;
pub mod bybit_tape;
pub mod net;
pub mod okx;
pub mod store;
pub mod tape;

use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// 원본에서 예외가 났다 = 그 연결을 접고 다시 붙는다(`except Exception`). 글자 = 파이썬 `type(exc).__name__`(+ 뜻).
#[derive(Debug)]
pub struct Reconnect(pub String);

/// 마지막으로 쓰기에 성공한 단조 시각(ms, `net::mono_ms`) -- 원본 `_WD["write"]`(정지 감시가 본다). 0 = 아직 없음.
pub static LAST_WRITE_MS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn wrote() {
    LAST_WRITE_MS.store(net::mono_ms(), Relaxed);
}

pub(crate) fn need<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a Value, Reconnect> {
    o.get(k).ok_or_else(|| Reconnect(format!("KeyError: '{k}'")))
}

/// 원본 repr(float). ponytail: 1e-5 미만·1e16 이상에서 파이썬은 `1e-05` 꼴, Rust 는 `1e-5` -- 칸·경계·ctVal 엔 그런 수가 없다.
pub(crate) fn py_repr(x: f64) -> String {
    format!("{x:?}")
}

/// 파이썬 float(x). ponytail: 문자열의 밑줄(`1_0`)·JSON 의 NaN/Infinity 낱말은 파이썬만 받는다 -- 거래소는 안 보낸다.
pub(crate) fn py_float(v: &Value) -> Result<f64, Reconnect> {
    match v {
        Value::Number(n) => n.as_f64().ok_or_else(|| Reconnect("ValueError".into())),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| Reconnect(format!("ValueError: float {s:?}"))),
        Value::Bool(b) => Ok(*b as i64 as f64),
        _ => Err(Reconnect("TypeError: float".into())),
    }
}

/// 파이썬 int(x): 정수 · 실수는 0 쪽으로 자름 · 정수 문자열.
pub(crate) fn py_int(v: &Value) -> Result<i64, Reconnect> {
    match v {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64))
            .ok_or_else(|| Reconnect("ValueError: int".into())),
        Value::String(s) => s.trim().parse::<i64>().map_err(|_| Reconnect(format!("ValueError: int {s:?}"))),
        Value::Bool(b) => Ok(*b as i64),
        _ => Err(Reconnect("TypeError: int".into())),
    }
}

/// 파이썬 bool(x)
pub(crate) fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// 파이썬 `for x in (msg.get(k) or [])`: 거짓이면 빈 목록 · 목록은 원소 · 사전은 열쇠(문자열) · 문자열은 글자 --
/// 열쇠·글자는 원소 처리에서 TypeError 로 걸러지므로 «원소 없음»과 같다. 참인 숫자·True 는 `for` 가 TypeError = 재연결.
pub(crate) fn py_iter<'a>(o: &'a Map<String, Value>, k: &str) -> Result<&'a [Value], Reconnect> {
    match o.get(k) {
        Some(Value::Array(a)) => Ok(a),
        Some(v @ (Value::Number(_) | Value::Bool(_))) if py_truthy(v) => Err(Reconnect("TypeError: not iterable".into())),
        _ => Ok(&[]),
    }
}
