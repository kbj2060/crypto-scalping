//! 파이썬 float 동작을 비트·문자열 그대로 재현한다. 주문가·수량은 파이썬에서 `urlencode` → `str(float)`(= repr)로
//! 거래소에 나간다 -- 그래서 «같은 값»의 기준은 f64 가 아니라 그 문자열이다.
//! 10진 고정소수 크레이트를 안 쓰는 이유: 원본이 float 연산이라 같은 IEEE 연산을 같은 순서로 하는 편이 끝자리까지 같다.

/// `round(x, n)`(n ≥ 1). CPython 은 정확한 10진 전개를 올바르게 반올림한 뒤 다시 float 로 읽는다.
/// n ≥ 1 이면 동점(정확히 ...5)이 float 로 표현될 수 없어(분모에 5 가 남는다) 반올림 방향 규칙이 결과를 바꾸지 않는다.
pub fn round_n(x: f64, n: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{:.*}", n, x).parse().unwrap()
}

/// `round(x, 0)`(float 를 돌려줌)·`int(round(x))` -- 동점은 짝수 쪽(파이썬 규칙).
pub fn round0(x: f64) -> f64 {
    x.round_ties_even()
}

/// 파이썬 `max(a, b)`: 같으면 **앞의 것**(0.0 과 -0.0 이 갈린다 -- repr 이 "0.0"/"-0.0").
pub fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// 파이썬 `min(a, b)`.
pub fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// 파이썬 `repr(float)` / `str(float)`. 가장 짧은 왕복 자릿수(Rust `{:e}` 와 같은 자릿수) ·
/// 지수 e 가 -4 ≤ e < 16 이면 고정 소수(끝에 ".0"), 아니면 "1e-05"·"1.5e+16" 모양.
pub fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let s = format!("{:e}", x); // 예: "-1.2345e3", "1e-5", "0e0"
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let point = exp + 1; // 소수점 앞 자릿수
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// f-string `{x:.Nf}` · `{x:,.Nf}`(comma = 천 단위 쉼표). 정확한 10진 전개에서 반올림(동점 짝수) -- Rust `{:.N}` 과 같다.
pub fn fixed(x: f64, n: usize, comma: bool) -> String {
    let s = format!("{:.*}", n, x);
    if !comma || !x.is_finite() {
        return s;
    }
    let (sign, rest) = s.strip_prefix('-').map(|r| ("-", r)).unwrap_or(("", &s));
    let (int, frac) = rest.split_once('.').map(|(a, b)| (a, Some(b))).unwrap_or((rest, None));
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    match frac {
        Some(f) => format!("{sign}{out}.{f}"),
        None => format!("{sign}{out}"),
    }
}

/// f-string `{x:g}`(유효 6자리 · 끝 0 지움 · 지수 < -4 또는 ≥ 6 이면 지수 모양).
/// ponytail: 6자리 동점(예: 1234565.0)에서 Rust `{:.5e}` 와 파이썬의 동점 처리가 같은지 확인 안 함 -- 쓰는 곳은 `100*stop_pct` 하나(작은 값). 큰 값에 쓰게 되면 대조 시험부터.
pub fn g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let s = format!("{:.5e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let strip = |t: String| -> String {
        if t.contains('.') { t.trim_end_matches('0').trim_end_matches('.').to_string() } else { t }
    };
    if (-4..6).contains(&exp) {
        strip(format!("{:.*}", (5 - exp) as usize, x))
    } else {
        format!("{}e{}{:02}", strip(mant.to_string()), if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repr_matches_python() {
        for (x, want) in [
            (2470.0, "2470.0"), (0.1 + 0.2, "0.30000000000000004"), (1e-5, "1e-05"), (0.0001, "0.0001"),
            (1e16, "1e+16"), (1e15, "1000000000000000.0"), (-0.0, "-0.0"), (0.0, "0.0"), (6.154, "6.154"),
            (123456789012345680.0, "1.2345678901234568e+17"), (-1.5e-7, "-1.5e-07"), (2.5e300, "2.5e+300"),
        ] {
            assert_eq!(repr(x), want, "{x}");
        }
    }
    #[test]
    fn fixed_and_g_match_python() {
        assert_eq!(fixed(1234567.5, 0, true), "1,234,568");
        assert_eq!(fixed(2.5, 0, true), "2"); // 동점 짝수
        assert_eq!(fixed(-1234.5, 2, true), "-1,234.50");
        assert_eq!(fixed(-0.04, 1, false), "-0.0");
        assert_eq!(fixed(999.0, 0, true), "999");
        assert_eq!(g(5.000000000000001), "5");
        assert_eq!(g(0.75), "0.75");
        assert_eq!(g(1e-5), "1e-05");
        assert_eq!(g(1234567.0), "1.23457e+06");
        assert_eq!(round_n(2.675, 2), 2.67); // 2.675 는 실제로 2.67499999...
        assert_eq!(round0(0.5), 0.0);
        assert_eq!(round0(1.5), 2.0);
    }
}

/// 파이썬 json.dumps(ensure_ascii=True) 의 문자열 한 개.
pub fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            ' '..='~' => o.push(c),
            _ => {
                for u in c.encode_utf16(&mut [0; 2]) {
                    o.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    o.push('"');
    o
}

/// 파이썬 json.dumps(v) 기본 모양(", " · ": " · ensure_ascii · 실수 = repr) -- aiohttp web.json_response 본문 글자.
pub fn json_dumps(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::Number(n) => match (n.as_i64(), n.as_u64()) {
            (Some(i), _) => i.to_string(),
            (_, Some(u)) => u.to_string(),
            _ => repr(n.as_f64().unwrap()),
        },
        Value::String(s) => json_str(s),
        Value::Array(a) => format!("[{}]", a.iter().map(json_dumps).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!("{{{}}}", m.iter().map(|(k, x)| format!("{}: {}", json_str(k), json_dumps(x))).collect::<Vec<_>>().join(", ")),
    }
}
