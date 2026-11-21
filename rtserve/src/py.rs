//! 원본 payload 함수들이 기대는 파이썬 의미 몇 가지(JSON 값 위에서): 비교·참거짓·산술·round·splitlines·집합 열쇠.
//! 값은 serde_json::Value(preserve_order = dict 순서) -- 정수는 i64, 실수는 f64 로 갈라 둔다(json.dumps 가 `1` 과 `1.0` 을 다르게 쓴다).

use serde_json::{Map, Number, Value};

pub type V = Value;
pub type Obj = Map<String, Value>;

/// 파이썬 예외 한 줄(`TypeError: ...`) -- 원본 핸들러가 `f"{type(exc).__name__}: {exc}"` 로 싣는 꼴.
#[derive(Debug, Clone, PartialEq)]
pub struct PyErr(pub String);

pub type R<T> = Result<T, PyErr>;

pub fn err<T>(kind: &str, msg: impl AsRef<str>) -> R<T> {
    Err(PyErr(format!("{kind}: {}", msg.as_ref())))
}

pub fn f(x: f64) -> V {
    // ponytail: NaN·무한대는 serde_json 이 못 담아 null 이 된다(원본은 NaN 글자) -- 지금 엔드포인트 입력에서 나올 길이 없다.
    Number::from_f64(x).map_or(V::Null, V::Number)
}

pub fn i(x: i64) -> V {
    V::from(x)
}

pub fn is_int(v: &V) -> bool {
    matches!(v, V::Number(n) if n.is_i64() || n.is_u64())
}

/// 숫자(bool 포함 -- 파이썬에서 True == 1)
pub fn num(v: &V) -> Option<f64> {
    match v {
        V::Number(n) => n.as_f64(),
        V::Bool(b) => Some(*b as i64 as f64),
        _ => None,
    }
}

fn int_of(v: &V) -> Option<i64> {
    match v {
        V::Number(n) => n.as_i64(),
        V::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

pub fn type_name(v: &V) -> &'static str {
    match v {
        V::Null => "NoneType",
        V::Bool(_) => "bool",
        V::Number(_) if is_int(v) => "int",
        V::Number(_) => "float",
        V::String(_) => "str",
        V::Array(_) => "list",
        V::Object(_) => "dict",
    }
}

pub fn truthy(v: &V) -> bool {
    match v {
        V::Null => false,
        V::Bool(b) => *b,
        V::Number(n) => n.as_f64() != Some(0.0),
        V::String(s) => !s.is_empty(),
        V::Array(a) => !a.is_empty(),
        V::Object(o) => !o.is_empty(),
    }
}

/// 파이썬 `a == b`
pub fn eq(a: &V, b: &V) -> bool {
    match (int_of(a), int_of(b)) {
        (Some(x), Some(y)) => x == y,
        _ => match (num(a), num(b)) {
            (Some(x), Some(y)) => x == y,
            (None, None) => match (a, b) {
                (V::Array(x), V::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q)),
                (V::Object(x), V::Object(y)) => x.len() == y.len() && x.iter().all(|(k, p)| y.get(k).is_some_and(|q| eq(p, q))),
                _ => a == b,
            },
            _ => false,
        },
    }
}

/// 파이썬 `a < b`(숫자끼리·문자열끼리 · 목록은 사전식 · 그 밖은 TypeError)
pub fn lt(a: &V, b: &V) -> R<bool> {
    match (a, b) {
        (V::String(x), V::String(y)) => Ok(x < y), // 코드 포인트 순 = UTF-8 바이트 순
        (V::Array(x), V::Array(y)) => {
            for (p, q) in x.iter().zip(y) {
                if !eq(p, q) {
                    return lt(p, q);
                }
            }
            Ok(x.len() < y.len())
        }
        _ => match (int_of(a), int_of(b)) {
            (Some(x), Some(y)) => Ok(x < y),
            _ => match (num(a), num(b)) {
                (Some(x), Some(y)) => Ok(x < y),
                _ => err("TypeError", format!("'<' not supported between instances of '{}' and '{}'", type_name(a), type_name(b))),
            },
        },
    }
}

pub fn ge(a: &V, b: &V) -> R<bool> {
    // 파이썬 a >= b (숫자) -- NaN 이 없으니 not (a < b)
    match (num(a), num(b)) {
        (Some(_), Some(_)) => Ok(!lt(a, b)?),
        _ => match (a, b) {
            (V::String(x), V::String(y)) => Ok(x >= y),
            _ => err("TypeError", format!("'>=' not supported between instances of '{}' and '{}'", type_name(a), type_name(b))),
        },
    }
}

pub fn gt(a: &V, b: &V) -> R<bool> {
    match (num(a), num(b)) {
        (Some(_), Some(_)) => lt(b, a),
        _ => match (a, b) {
            (V::String(x), V::String(y)) => Ok(x > y),
            _ => err("TypeError", format!("'>' not supported between instances of '{}' and '{}'", type_name(a), type_name(b))),
        },
    }
}

fn arith(a: &V, b: &V, op: char) -> R<V> {
    if let (Some(x), Some(y)) = (int_of(a), int_of(b)) {
        let r = match op {
            '+' => x.checked_add(y),
            '-' => x.checked_sub(y),
            _ => x.checked_mul(y),
        };
        if let Some(r) = r {
            return Ok(i(r));
        }
    }
    match (num(a), num(b)) {
        (Some(x), Some(y)) => Ok(f(match op {
            '+' => x + y,
            '-' => x - y,
            _ => x * y,
        })),
        _ => err("TypeError", format!("unsupported operand type(s) for {op}: '{}' and '{}'", type_name(a), type_name(b))),
    }
}

pub fn add(a: &V, b: &V) -> R<V> {
    arith(a, b, '+')
}
pub fn sub(a: &V, b: &V) -> R<V> {
    arith(a, b, '-')
}
pub fn mul(a: &V, b: &V) -> R<V> {
    arith(a, b, '*')
}

/// 파이썬 `x / y`(참 나눗셈 · 늘 실수)
pub fn div(a: &V, b: &V) -> R<f64> {
    match (num(a), num(b)) {
        (Some(_), Some(y)) if y == 0.0 => err("ZeroDivisionError", "division by zero"),
        (Some(x), Some(y)) => Ok(match (int_of(a), int_of(b)) {
            // int / int 는 파이썬이 정확한 몫을 반올림한다 -- 2^53 안의 정수면 f64 나눗셈과 같다
            (Some(p), Some(q)) if p.unsigned_abs() < 1 << 53 && q.unsigned_abs() < 1 << 53 => p as f64 / q as f64,
            _ => x / y,
        }),
        _ => err("TypeError", format!("unsupported operand type(s) for /: '{}' and '{}'", type_name(a), type_name(b))),
    }
}

/// 파이썬 `round(x, nd)`(float): 이진 값을 십진 nd 자리로 정확히 반올림(짝수) -- Rust 의 `{:.N}` 는 정확한 값을 짝수로 반올림한다.
pub fn round_nd(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let r: f64 = format!("{x:.nd$}").parse().unwrap();
    if r == 0.0 { 0.0f64.copysign(x) } else { r }
}

/// 파이썬 `round(x)`(float → int, 짝수 반올림)
pub fn round0(x: f64) -> R<i64> {
    if x.is_nan() {
        return err("ValueError", "cannot convert float NaN to integer");
    }
    if x.is_infinite() {
        return err("OverflowError", "cannot convert float infinity to integer");
    }
    Ok(x.round_ties_even() as i64)
}

/// 파이썬 `round(v, nd)` 를 JSON 값에: int 는 그대로 int, float 는 float, 그 밖은 TypeError
pub fn round_v(v: &V, nd: usize) -> R<V> {
    if is_int(v) {
        return Ok(v.clone());
    }
    match v {
        V::Bool(b) => Ok(i(*b as i64)),
        V::Number(n) => Ok(f(round_nd(n.as_f64().unwrap(), nd))),
        _ => err("TypeError", format!("type {} doesn't define __round__ method", type_name(v))),
    }
}

/// 파이썬 `round(v)` 를 JSON 값에(결과 int)
pub fn round0_v(v: &V) -> R<V> {
    if let Some(x) = int_of(v) {
        return Ok(i(x));
    }
    match v {
        V::Number(n) => Ok(i(round0(n.as_f64().unwrap())?)),
        _ => err("TypeError", format!("type {} doesn't define __round__ method", type_name(v))),
    }
}

/// 파이썬 `float(v)`(JSON 값 · 문자열은 앞뒤 공백 허용)
pub fn to_float(v: &V) -> R<f64> {
    match v {
        V::String(s) => s.trim().replace('_', "").parse::<f64>().or_else(|_| err("ValueError", format!("could not convert string to float: '{s}'"))),
        _ => num(v).map_or_else(|| err("TypeError", format!("float() argument must be a string or a real number, not '{}'", type_name(v))), Ok),
    }
}

/// 파이썬 `str.splitlines()` -- \n·\r·\r\n 말고도 \v \f \x1c \x1d \x1e \x85     에서 끊는다
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((idx, c)) = it.next() {
        let brk = matches!(c, '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}');
        if brk {
            out.push(&s[start..idx]);
            let mut end = idx + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// 파이썬 집합·사전 열쇠(해시 같음): 1 == 1.0 == True 가 같은 열쇠
pub fn hkey(v: &V) -> String {
    match v {
        V::Bool(b) => (*b as i64).to_string(),
        V::Number(_) => {
            let x = num(v).unwrap();
            match int_of(v) {
                Some(n) => n.to_string(),
                None if x.fract() == 0.0 && x.abs() < 9.2e18 => (x as i64).to_string(),
                None => rtcollect::py_float_repr(x),
            }
        }
        V::String(s) => format!("s{s}"),
        V::Array(a) => format!("({})", a.iter().map(hkey).collect::<Vec<_>>().join(",")),
        V::Null => "None".into(),
        V::Object(_) => "{unhashable}".into(),
    }
}

/// 파이썬 `str(v)`(JSON 값 · 열쇠 만들 때)
pub fn to_str(v: &V) -> String {
    match v {
        V::String(s) => s.clone(),
        V::Null => "None".into(),
        V::Bool(b) => if *b { "True" } else { "False" }.into(),
        V::Number(n) if is_int(v) => n.to_string(),
        V::Number(n) => rtcollect::py_float_repr(n.as_f64().unwrap()),
        _ => rtcollect::py_dumps_default(v), // ponytail: 목록·사전의 str() 은 따옴표가 ' 라 다르다 -- 코인 이름 자리에 올 일이 없다
    }
}

/// `d[k]`(사전 아님 → TypeError · 없음 → KeyError)
pub fn item<'a>(d: &'a V, k: &str) -> R<&'a V> {
    match d {
        V::Object(o) => o.get(k).map_or_else(|| err("KeyError", format!("'{k}'")), Ok),
        V::Array(_) | V::String(_) => err("TypeError", format!("{} indices must be integers or slices, not str", type_name(d))),
        _ => err("TypeError", format!("'{}' object is not subscriptable", type_name(d))),
    }
}

/// `d.get(k)` -- 사전이 아니면 AttributeError
pub fn get<'a>(d: &'a V, k: &str) -> R<Option<&'a V>> {
    match d {
        V::Object(o) => Ok(o.get(k)),
        _ => err("AttributeError", format!("'{}' object has no attribute 'get'", type_name(d))),
    }
}

/// 파이썬 `json.loads`(serde_json · 순서 보존). 실패 문구는 원본(JSONDecodeError)과 다르다 -- 원본도 그 문구를 화면에 쓰지 않는다.
pub fn loads(s: &str) -> R<V> {
    serde_json::from_str(s.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r'))).or_else(|e| err("JSONDecodeError", e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn splitlines_like_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd\u{2028}e\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(splitlines("a\n\nb"), vec!["a", "", "b"]);
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n"), vec![""]);
    }

    #[test]
    fn compare_and_round() {
        assert!(eq(&json!(1), &json!(1.0)) && eq(&json!(true), &json!(1)) && !eq(&json!("1"), &json!(1)));
        assert!(lt(&json!(1), &json!(1.5)).unwrap() && lt(&json!("a"), &json!(1)).is_err());
        assert_eq!(round_nd(2.675, 2), 2.67); // 이진 값이 2.67499.. 라 내림(파이썬과 같음)
        assert_eq!(round_nd(0.125, 2), 0.12); // 정확한 반값은 짝수
        assert_eq!(round0(2.5).unwrap(), 2);
        assert_eq!(round0(-0.5).unwrap(), 0);
        assert_eq!(mul(&json!(3), &json!(1000)).unwrap(), json!(3000));
        assert_eq!(hkey(&json!(1.0)), hkey(&json!(1)));
    }
}
