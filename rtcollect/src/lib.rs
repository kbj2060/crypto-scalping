//! 파이썬 수집기를 «같은 입력 → 디스크에 같은 행»으로 옮긴 것. 수집기마다 모듈 하나, 소켓은 `net`(실행 파일만 쓴다).
//! - `binance`   = scripts/live_trade_tape_collector_20260916.py   (binance_tape.sqlite · binance_spot_tape.sqlite)
//! - `bybit_liq` = scripts/live_bybit_liquidation_collector_20261003.py (bybit_liq.sqlite)
//! - `bybit_tape`= scripts/live_bybit_trade_tape_collector_20261006.py  (bybit_tape.sqlite + bybit_ctx.sqlite)
//! - `okx`       = scripts/live_okx_trade_tape_collector_20260923.py    (okx_tape.sqlite)
//! - `okx_ctx`   = scripts/live_okx_context_collector_20260923.py   (okx_ctx.sqlite -- OI·마크·펀딩·청산)
//! - `binance_ctx` = scripts/live_binance_ctx_collector_20261001.py (binance_ctx.sqlite -- 마크·강제청산 WS · OI REST 는 기본 꺼짐)
//! - `xyz`       = scripts/live_xyz100_collector_20261014.py         (hl_xyz.sqlite -- 접속 때 REST 메움은 기본 꺼짐)
//! 세 체결 테이프는 원본처럼 `tape`(칸 규칙)·`store`(표·쓰기)를 나눠 쓴다. 맥락 셋은 `store::open_hot`·`in_tx`. 정답지 대조 = tests/*_parity.rs.
//!
//! - `rest`      = 바이낸스 테이프의 REST 1분봉 대조(verify_1m)·aggTrades 복구(backfill_1m) -- 기본 꺼짐(main.rs `RTCOLLECT_REST=1`).
//! - `rest_venue` = OKX ctVal 대조·history-candles 대조·history-trades 복구 · Bybit 1분봉 대조 -- 같은 `rest::Rest` 뼈대, 기본 꺼짐.
//! 아직 안 옮긴 수집기(저장 형식이 SQLite 가 아닌 것들) = rtcollect/REMAINING.md.
pub mod binance;
pub mod binance_ctx;
pub mod bookdepth;
pub mod bt;
pub mod bybit_liq;
pub mod bybit_tape;
pub mod depth;
pub mod net;
pub mod hl_ctx;
pub mod ob5m;
pub mod okx;
pub mod okx_ctx;
pub mod raster;
pub mod rest;
pub mod rest_venue;
pub mod store;
pub mod tape;
pub mod xyz;

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

/// 원본 repr(float) -- `py_float_repr`(지수 꼴까지 파이썬과 같다).
pub(crate) fn py_repr(x: f64) -> String {
    py_float_repr(x)
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

/// 구독 글자 = 원본 `json.dumps({"op": "subscribe", "args": [...]})` 기본 모양(", " · ": "). `args` = 이미 JSON 글자인 원소들.
pub fn py_subscribe(args: &[String]) -> String {
    format!(r#"{{"op": "subscribe", "args": [{}]}}"#, args.join(", "))
}

/// 파이썬 int(time.time() * 1000) -- 양수라 0 쪽 자르기 = 내림.
pub(crate) fn ms(t: f64) -> i64 {
    (t * 1000.0) as i64
}

/// 메시지 값을 파이썬 sqlite3 가 묶는 대로(문자열·None·참거짓=정수·정수·실수). 목록·사전은 원본에서 바인딩 오류다.
pub(crate) fn py_sql(v: &Value) -> Result<rusqlite::types::Value, Reconnect> {
    use rusqlite::types::Value as Sql;
    Ok(match v {
        Value::String(s) => Sql::Text(s.clone()),
        Value::Null => Sql::Null,
        Value::Bool(b) => Sql::Integer(*b as i64),
        Value::Number(n) => n.as_i64().map_or_else(|| Sql::Real(n.as_f64().unwrap_or(f64::NAN)), Sql::Integer),
        _ => return Err(Reconnect("InterfaceError: 바인딩할 수 없는 값".into())),
    })
}

/// 파이썬 str(x)(없으면 `default`). ponytail: 목록·사전은 repr 흉내(작은따옴표 · 글자 안 따옴표·이스케이프는 안 맞춘다) -- 거래소는 그 칸에 글자만 보낸다.
pub(crate) fn py_str(v: Option<&Value>, default: &str) -> String {
    fn repr(v: &Value) -> String {
        match v {
            Value::String(s) => format!("'{s}'"),
            Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
            Value::Object(o) => format!("{{{}}}", o.iter().map(|(k, x)| format!("'{k}': {}", repr(x))).collect::<Vec<_>>().join(", ")),
            v => py_str(Some(v), ""),
        }
    }
    match v {
        None => default.to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".into(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(Value::Number(n)) => n.as_i64().map_or_else(|| n.as_u64().map_or_else(|| py_repr(n.as_f64().unwrap()), |u| u.to_string()), |i| i.to_string()),
        Some(v) => repr(v),
    }
}

/// 원본 `_f(v)`(float(v), TypeError·ValueError 면 None) -- 없는 열쇠도 None.
pub(crate) fn py_f(v: Option<&Value>) -> Option<f64> {
    v.and_then(|v| py_float(v).ok())
}

/// `int(_f(v, 0) or 0)`: 못 읽으면 0 · 무한대는 OverflowError, NaN 은 ValueError(둘 다 잡지 않는다 = 재연결).
pub(crate) fn py_f_int(v: Option<&Value>) -> Result<i64, Reconnect> {
    match py_f(v) {
        None => Ok(0),
        Some(f) if f.is_finite() => Ok(f.trunc() as i64),
        Some(f) => Err(Reconnect(format!("{}: int({f})", if f.is_nan() { "ValueError" } else { "OverflowError" }))),
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

/// 파이썬 float.__repr__(json.dumps 가 쓰는 것): 가장 짧은 왕복 숫자 · 지수가 -4 이상 16 미만이면 고정 소수(`.0` 붙음) · 아니면 `1e-05`·`1.5e+16`.
pub fn py_float_repr(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() { "NaN".into() } else if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }; // json.dumps 낱말
    }
    let e = format!("{x:e}"); // 예: "-1.2345e-5" -- 가장 짧은 왕복 자리
    let (m, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let (neg, m) = m.strip_prefix('-').map_or((false, m), |r| (true, r));
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    let s = if (-4..16).contains(&exp) {
        let point = exp + 1; // 소수점 앞 자리 수
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat((-point) as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        }
    } else {
        let mant = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    if neg { format!("-{s}") } else { s }
}

/// 파이썬 `json.dumps(v, separators=(",", ":"))`(ensure_ascii=True): 열쇠 순서 그대로(serde_json preserve_order) · 비 ASCII 는 \uXXXX(대리쌍).
pub(crate) fn py_dumps(v: &Value) -> String {
    fn s(out: &mut String, t: &str) {
        out.push('"');
        for c in t.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                    let mut b = [0u16; 2];
                    for u in c.encode_utf16(&mut b) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
    fn go(out: &mut String, v: &Value) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => out.push_str(&i.to_string()),
                (_, Some(u)) => out.push_str(&u.to_string()),
                _ => out.push_str(&py_float_repr(n.as_f64().unwrap())),
            },
            Value::String(t) => s(out, t),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    go(out, x);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    s(out, k);
                    out.push(':');
                    go(out, x);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    go(&mut out, v);
    out
}

/// 파이썬 `a == b`(JSON 값): 숫자끼리는 값으로(1 == 1.0 · True == 1), 나머지는 모양 그대로.
pub(crate) fn py_eq(a: &Value, b: &Value) -> bool {
    let num = |v: &Value| match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(x) => Some(*x as i64 as f64),
        _ => None,
    };
    match (a, b) {
        (Value::Number(x), Value::Number(y)) if x.as_i64().is_some() && y.as_i64().is_some() => x.as_i64() == y.as_i64(),
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x == y,
            (None, None) => a == b,
            _ => false,
        },
    }
}

#[cfg(test)]
mod py_tests {
    use super::*;
    #[test]
    fn repr_and_dumps_match_python() {
        // 파이썬 3.11 repr 로 확인한 값
        for (x, want) in [(1e16, "1e+16"), (1e15, "1000000000000000.0"), (0.0001, "0.0001"), (0.00001, "1e-05"), (2500.1, "2500.1"),
                          (-1.5e-7, "-1.5e-07"), (123456789012345680.0, "1.2345678901234568e+17"), (0.0, "0.0"), (5.0, "5.0"), (1.25e-4, "0.000125")] {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
        let v: Value = serde_json::from_str(r#"{"b":[["1.0","2"]],"a":1,"é":"가\n","f":1.0e16,"n":null,"t":true}"#).unwrap();
        let want = concat!(r#"{"b":[["1.0","2"]],"a":1,""#, "\\u00e9", r#"":""#, "\\uac00", r#"\n","f":1e+16,"n":null,"t":true}"#);
        assert_eq!(py_dumps(&v), want);
        assert!(py_eq(&Value::from(1), &serde_json::from_str("1.0").unwrap()) && py_eq(&Value::from(true), &Value::from(1)));
    }
}
