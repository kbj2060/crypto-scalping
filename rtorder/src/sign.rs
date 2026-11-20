//! 서명 · 쿼리 문자열 · 오류 문자열 -- `scripts/live_binance_account_20260910.py::_sign` 과 `live_manual_peg_execute_20260912.py::signed` 의
//! 문자열 규칙을 그대로(2026-11-20 둘째 조각). 네트워크 없음 -- 보내기는 `http.rs`.
//! 정답지: tests/fixtures/http_fixture.json(원본 파이썬 + 진짜 aiohttp 를 로컬 가짜 서버에 붙여 만든 것).

use crate::pyfmt;
use ring::hmac;
use serde_json::Value;

pub const RECV_WINDOW_MS: i64 = 5000;

/// 파이썬 dict 값 하나. `urlencode` 는 값마다 `str()` 을 부른다 -- float 은 repr("2500.0" · "1e-05"), None 은 "None", bool 은 "True".
#[derive(Debug, Clone, PartialEq)]
pub enum P {
    S(String),
    I(i64),
    F(f64),
    B(bool),
    None,
}

impl P {
    pub fn s(v: &str) -> P {
        P::S(v.into())
    }
    /// JSON 값(거래소 응답에서 꺼낸 orderId 등)을 그 값의 파이썬 `str()` 그대로.
    pub fn of(v: Option<&Value>) -> P {
        match v {
            None | Some(Value::Null) => P::None,
            Some(Value::Bool(b)) => P::B(*b),
            Some(Value::String(s)) => P::S(s.clone()),
            Some(Value::Number(n)) => n.as_i64().map(P::I).unwrap_or_else(|| P::F(n.as_f64().unwrap_or(f64::NAN))),
            Some(v) => P::S(py_repr(v)),
        }
    }
    pub fn py_str(&self) -> String {
        match self {
            P::S(s) => s.clone(),
            P::I(i) => i.to_string(),
            P::F(f) => pyfmt::repr(*f),
            P::B(b) => (if *b { "True" } else { "False" }).into(),
            P::None => "None".into(),
        }
    }
}

pub type Params = Vec<(&'static str, P)>;

/// `urllib.parse.quote_plus(s)`(safe='') -- 늘 안전한 글자 = 영숫자 `_.-~` · 빈칸은 '+' · 나머지는 UTF-8 바이트마다 %XX(대문자).
pub fn quote_plus(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => o.push(b as char),
            b' ' => o.push('+'),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

/// `urlencode(dict)` -- 넣은 순서 그대로(파이썬 dict 순서), 키·값 둘 다 quote_plus.
pub fn urlencode(params: &[(&str, P)]) -> String {
    params.iter().map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(&v.py_str()))).collect::<Vec<_>>().join("&")
}

/// `_sign(params, secret, offset)`: 쿼리 + recvWindow + timestamp 를 HMAC-SHA256(16진) 으로 서명. `ts = int(time.time()*1000) + offset`(호출부가 계산).
/// 🔴시계 보정은 /fapi/v1/time 앵커 하나 -- 두 벌로 두면 한쪽만 조용히 -1021(2026-09-10 WSL 시계 1초 드리프트).
pub fn sign_query(params: &[(&str, P)], secret: &str, ts: i64) -> String {
    let mut all: Vec<(&str, P)> = params.to_vec();
    all.push(("recvWindow", P::I(RECV_WINDOW_MS)));
    all.push(("timestamp", P::I(ts)));
    let q = urlencode(&all);
    let tag = hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes()), q.as_bytes());
    format!("{q}&signature={}", hex::encode(tag.as_ref()))
}

/// 파이썬 `int(time.time() * 1000)`(양수에서 0 쪽으로 자름).
pub fn ms(epoch_sec: f64) -> i64 {
    (epoch_sec * 1000.0) as i64
}

fn py_str_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::from(q);
    for ch in s.chars() {
        match ch {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => {
                o.push('\\');
                o.push(c)
            }
            // ponytail: 파이썬 isprintable 표를 다 옮기지 않았다 -- 제어 글자·흔한 공백/서식 글자만 이스케이프. 거래소 msg 는 ASCII 라 충분, 다른 글자가 문제 되면 표를 늘린다.
            c if c.is_control() || matches!(c, '\u{a0}' | '\u{ad}' | '\u{1680}' | '\u{2000}'..='\u{200f}' | '\u{2028}'..='\u{202f}' | '\u{205f}'..='\u{2064}' | '\u{3000}' | '\u{feff}') => {
                let n = c as u32;
                o.push_str(&if n < 0x100 { format!("\\x{n:02x}") } else if n < 0x10000 { format!("\\u{n:04x}") } else { format!("\\U{n:08x}") });
            }
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

/// 파이썬 `repr()` -- json.loads 한 값(dict 순서는 응답 순서: serde_json preserve_order).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => (if *b { "True" } else { "False" }).into(),
        Value::Number(n) => n.as_i64().map(|i| i.to_string()).or_else(|| n.as_u64().map(|u| u.to_string())).unwrap_or_else(|| pyfmt::repr(n.as_f64().unwrap())),
        Value::String(s) => py_str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr(v))).collect::<Vec<_>>().join(", ")),
    }
}

/// 파이썬 `str()` -- 문자열이면 그대로, 아니면 repr.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => py_repr(v),
    }
}

/// 파이썬 진릿값.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(m)) => !m.is_empty(),
    }
}

/// `str(o.get(k) or "")`.
pub fn str_or_empty(o: &Value, k: &str) -> String {
    let v = o.get(k);
    if truthy(v) { py_str(v.unwrap()) } else { String::new() }
}

fn py_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `signed()` 가 200 이 아닌 응답을 오류 문자열로: `"<상태> [<code>] <msg>"`.
/// 🔴code 를 버리면 안 된다 -- 2026-09-20 «400 Due to ...» 에 "5022" 가 없어 리페그가 죽었다(is_post_only_reject 가 문구도 같이 보는 까닭).
/// payload 가 dict 가 아니면 파이썬은 `.get` 에서 AttributeError 를 내고 그것이 오류 문자열이 된다(숫자로 시작 안 함 = «나갔는지 모름»).
pub fn http_error(status: u16, payload: &Value) -> String {
    let Value::Object(m) = payload else {
        return format!("AttributeError: '{}' object has no attribute 'get'", py_type(payload));
    };
    let tag = match m.get("code") {
        None | Some(Value::Null) => String::new(),
        Some(c) => format!("[{}] ", py_str(c)),
    };
    let msg = m.get("msg").map(py_str).unwrap_or_else(|| py_repr(payload));
    format!("{status} {tag}{msg}")
}

/// `{"__error__": s}`.
pub fn err_value(s: String) -> Value {
    serde_json::json!({ "__error__": s })
}

/// `r.get("__error__")`(dict 일 때만 -- `"__error__" in list` 는 원소 비교라 거의 늘 False).
pub fn error_of(r: &Value) -> Option<&str> {
    r.get("__error__").and_then(Value::as_str)
}

/// `float(x)` -- 문자열·숫자. 파이썬은 실패하면 ValueError.
pub fn py_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap()),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| format!("ValueError: could not convert string to float: {}", py_str_repr(s))),
        Value::Bool(b) => Ok(*b as i64 as f64),
        v => Err(format!("TypeError: float() argument must be a string or a real number, not '{}'", py_type(v))),
    }
}

/// `float(order.get("executedQty") or 0.0)`.
pub fn executed_qty(o: &Value) -> Result<f64, String> {
    let v = o.get("executedQty");
    if truthy(v) { py_float(v.unwrap()) } else { Ok(0.0) }
}

// ── 밴 기록(scripts/binance_ban_guard.py::note) ────────────────────────────────
// 418/429 는 기록만 -- 주문·취소는 막지 않는다: 손절 청산을 가짜 밴 기록으로 멈추면 더 위험하다(signed 주석). 그래서 이 크레이트는 ban_remaining 으로 요청을 막지 않는다.

/// 해제까지 남은 초(파일 없음·깨짐 = 0).
pub fn ban_remaining(path: &std::path::Path, now: f64) -> f64 {
    std::fs::read_to_string(path).ok().and_then(|s| s.trim().parse::<f64>().ok()).map_or(0.0, |u| (u - now).max(0.0))
}

/// 418 은 본문의 «banned until <13자리 ms>», 429 는 Retry-After(숫자만 · 없으면 30초, 418 은 120초). 더 긴 기록이 있으면 안 덮는다. 돌려주기 = 새로 적은 남은 초(안 적었으면 0).
pub fn ban_note(path: &std::path::Path, status: u16, body: &str, retry_after: Option<&str>, now: f64) -> f64 {
    if status != 418 && status != 429 {
        return 0.0;
    }
    let banned = body.find("banned until ").and_then(|i| {
        let d = &body[i + 13..];
        (d.len() >= 13 && d[..13].bytes().all(|b| b.is_ascii_digit())).then(|| d[..13].parse::<f64>().unwrap() / 1000.0)
    });
    let until = banned.unwrap_or_else(|| match retry_after.filter(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit())) {
        Some(r) => now + r.parse::<f64>().unwrap(),
        None => now + if status == 418 { 120.0 } else { 30.0 },
    });
    if until <= now + ban_remaining(path, now) {
        return 0.0; // 이미 더 긴 기록이 있다
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // 원자적 교체(읽는 쪽이 반쪽을 못 본다) -- 원본과 같은 «.<pid>.tmp»
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    if std::fs::write(&tmp, format!("{until:.3}\n")).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
    until - now
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quote_plus_and_errors_match_python() {
        assert_eq!(quote_plus("a b+c&d=e/f?g~h_i.j-k"), "a+b%2Bc%26d%3De%2Ff%3Fg~h_i.j-k");
        assert_eq!(quote_plus("1e+16"), "1e%2B16");
        assert_eq!(http_error(400, &serde_json::json!({"code": -5022, "msg": "x"})), "400 [-5022] x");
        assert_eq!(http_error(400, &serde_json::json!([1])), "AttributeError: 'list' object has no attribute 'get'");
        let d = std::env::temp_dir().join(format!("rtorder_ban_{}", std::process::id()));
        let f = d.join("ban");
        let _ = std::fs::remove_file(&f);
        let left = ban_note(&f, 418, "banned until 1800000600000. x", None, 1_800_000_000.0);
        assert_eq!(left, 600.0);
        assert_eq!(ban_note(&f, 429, "", Some("5"), 1_800_000_000.0), 0.0, "더 짧은 기록이 긴 차단을 덮으면 안 된다");
        assert_eq!(ban_note(&f, 400, "", None, 0.0), 0.0);
        let _ = std::fs::remove_dir_all(&d);
    }
}
