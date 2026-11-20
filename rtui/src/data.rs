//! /api/footprint · /api/market-history 응답 모양(읽는 칸만) + 숫자 글자(app.js 와 같은 반올림).
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Footprint {
    pub bucket: f64,
    pub bars: Vec<FpBar>,
}

#[derive(Deserialize)]
pub struct FpBar {
    pub time: i64,
    /// [가격, 매수, 매도, 고래 매수, 고래 매도, 리테일 매수, 리테일 매도] -- 차트는 앞 셋만 읽는다
    pub levels: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct MarketHistory {
    pub candles: Vec<Candle>,
}

#[derive(Deserialize, Clone, Copy)]
pub struct Candle {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// JS Number.prototype.toFixed: 정확히 반인 값은 0 에서 먼 쪽으로(Rust `{:.N}` 은 짝수 쪽 -- 42.25 → "42.2" 대 JS "42.3")
pub fn to_fixed(v: f64, d: usize) -> String {
    let exact = format!("{:.1100}", v.abs());
    let frac = &exact[exact.find('.').unwrap() + 1..];
    if frac[d..].starts_with('5') && frac[d + 1..].bytes().all(|b| b == b'0') {
        let m = 10f64.powi(d as i32);
        return format!("{}{:.*}", if v < 0.0 { "-" } else { "" }, d, ((v.abs() * m).floor() + 1.0) / m);
    }
    format!("{v:.d$}")
}

/// app.js 6840 fmtFootprintQty
pub fn fmt_qty(v: f64) -> String {
    if !(v > 0.0) { return String::new(); }
    if v >= 1e6 { return to_fixed(v / 1e6, if v >= 1e7 { 0 } else { 1 }) + "M"; }
    if v >= 1000.0 { return to_fixed(v / 1000.0, if v >= 10000.0 { 0 } else { 1 }) + "k"; }
    if v >= 100.0 { return to_fixed(v, 0); }
    to_fixed(v, 1)
}

#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn fmt() {
        assert_eq!(to_fixed(42.25, 1), "42.3");
        assert_eq!(to_fixed(1777.45, 1), "1777.5"); // 1777.45 는 이진수로 1777.4500000000000455 → 위로
        assert_eq!(fmt_qty(0.0), "");
        assert_eq!(fmt_qty(99.95), "100.0"); // 이진 99.95000000000000284 → JS (99.95).toFixed(1) = "100.0"
        assert_eq!(fmt_qty(1234.0), "1.2k");
        assert_eq!(fmt_qty(12345.0), "12k");
        assert_eq!(fmt_qty(2.5e6), "2.5M");
    }
}
