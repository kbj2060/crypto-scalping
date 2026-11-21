//! /api/footprint · /api/market-history · /api/footprint-week 응답 모양(읽는 칸만) + 숫자 글자(app.js 와 같은 반올림).
use serde::Deserialize;

/// /api/footprint 응답(app.js refreshFootprint 가 읽는 칸). 숫자 칸이 없으면 app.js 의 `Number(x) || 기본값` 과 같게 0.
#[derive(Deserialize, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Footprint {
    #[serde(default)]
    pub bucket: f64,
    pub bars: Vec<FpBar>,
    #[serde(default)]
    pub bar_seconds: f64,
    #[serde(default)]
    pub whale_min_usd: f64,
    #[serde(default)]
    pub retail_max_usd: f64,
    #[serde(default)]
    pub full: bool,
}

#[derive(Deserialize, Clone, PartialEq, Debug)]
pub struct FpBar {
    pub time: i64,
    /// [가격, 매수, 매도, 고래 매수, 고래 매도, 리테일 매수, 리테일 매도] -- 차트는 앞 셋 · 레인·프로파일은 일곱 다
    pub levels: Vec<Vec<f64>>,
    /// 봉 안 누적 델타 [최저, 최고, 마감](서버 1초 체결 순서 -- 레인 꼬리)
    #[serde(default)]
    pub dhl: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct MarketHistory {
    #[serde(default)]
    pub candles: Vec<Candle>,
}

/// /api/footprint-week (app.js maybeFetchFpWeek 가 읽는 칸 중 이 카드가 쓰는 것 -- oi·volPre 는 범위 밖 레인)
#[derive(Deserialize)]
pub struct Week {
    #[serde(default)]
    pub candles: Vec<Candle>,
    #[serde(default)]
    pub bars: Vec<FpBar>,
    #[serde(default)]
    pub liq: Vec<LiqBar>,
}

/// /api/liquidation-5m-history 응답(app.js refreshLiquidation5mSignal 3257 · 꼬리 3215) -- warmed_up 이 아니면 빈 목록
#[derive(Deserialize, Default)]
pub struct LiqHist {
    #[serde(default)]
    pub warmed_up: bool,
    #[serde(default)]
    pub bars: Vec<LiqBar>,
}

/// 5분 청산 한 봉(ts = ISO 시각 글자 · 금액 = 거래소 합 · okx = 그 봉을 OKX 가 봤나 · okx_detail·bybit·hl = 거래소 몫(있으면))
#[derive(Deserialize, Clone, Debug, PartialEq, Default)]
pub struct LiqBar {
    pub ts: String,
    #[serde(default, deserialize_with = "num")]
    pub long_usd: f64,
    #[serde(default, deserialize_with = "num")]
    pub short_usd: f64,
    #[serde(default, deserialize_with = "num")]
    pub events: f64,
    #[serde(default, deserialize_with = "truthy")]
    pub partial: bool,
    #[serde(default, deserialize_with = "truthy")]
    pub okx: bool,
    #[serde(default)]
    pub okx_detail: Option<LiqVenue>,
    #[serde(default)]
    pub bybit: Option<LiqVenue>,
    #[serde(default)]
    pub hl: Option<LiqVenue>,
}

#[derive(Deserialize, Clone, Debug, PartialEq, Default)]
pub struct LiqVenue {
    #[serde(default, deserialize_with = "num")]
    pub long_usd: f64,
    #[serde(default, deserialize_with = "num")]
    pub short_usd: f64,
    #[serde(default, deserialize_with = "num")]
    pub n: f64,
}

/// `Number(x) || 0`(null·글자·없음 → 0)
fn num<'de, D: serde::Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v { serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0), serde_json::Value::String(s) => s.trim().parse().unwrap_or(0.0), serde_json::Value::Bool(b) => b as u8 as f64, _ => 0.0 })
}
/// JS `!!x`
fn truthy<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v { serde_json::Value::Null => false, serde_json::Value::Bool(b) => b, serde_json::Value::Number(n) => n.as_f64() != Some(0.0), serde_json::Value::String(s) => !s.is_empty(), _ => true })
}

/// Date.parse(ISO 글자) → 초(내림). 받는 꼴: YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM] · 시간대 없으면 UTC 로 본다
// ponytail: JS 는 시간대 없는 날짜+시각을 지역 시각으로 읽는다 -- 서버 ts 는 늘 +00:00/Z 를 단다(fixture 도). 아니면 여기서 tz 를 받는다
pub fn iso_sec(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let n = |a: usize, l: usize| s.get(a..a + l)?.parse::<i64>().ok();
    if b.len() < 19 { return None; }
    let (y, mo, d, h, mi, se) = (n(0, 4)?, n(5, 2)?, n(8, 2)?, n(11, 2)?, n(14, 2)?, n(17, 2)?);
    let mut i = 19;
    let mut ms = 0.0;
    if b.get(i) == Some(&b'.') { let j = i + 1 + b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count(); ms = format!("0{}", &s[i..j]).parse::<f64>().ok()? * 1000.0; i = j; }
    let off = match b.get(i) { None | Some(b'Z') => 0, Some(&c) if c == b'+' || c == b'-' => { let v = n(i + 1, 2)? * 3600 + n(i + 4, 2)? * 60; if c == b'+' { v } else { -v } } _ => return None };
    let ms_total = (days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se - off) as f64 * 1000.0 + ms;
    Some((ms_total / 1000.0).floor() as i64)
}
/// new Date(sec*1000).toISOString()
pub fn iso_of(sec: i64) -> String {
    let (d, t) = (sec.div_euclid(86_400), sec.rem_euclid(86_400));
    let (y, m, dd) = civil_from_days(d);
    format!("{y:04}-{m:02}-{dd:02}T{:02}:{:02}:{:02}.000Z", t / 3600, t % 3600 / 60, t % 60)
}
// Howard Hinnant 날짜 식
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    (yoe + era * 400 + if m <= 2 { 1 } else { 0 }, m, d)
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Candle {
    pub time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// 형성 중 봉(미완 경매가 건너뛴다). None = 키 없음(app.js Object.assign 이 덮지 않는다 -- view::tf_candles)
    #[serde(default)]
    pub partial: Option<bool>,
    #[serde(default)]
    pub volume: f64,
    /// 서버가 마감봉에 붙이는 SMA144(없으면 «아직 안 붙었다» -- feed::history_retry_due 의 증상)
    #[serde(default)]
    pub sma: Option<f64>,
}

impl Candle {
    pub fn partial(&self) -> bool { self.partial == Some(true) }
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

/// `Number(l[i]) || 0` -- 칸 배열이 짧거나 NaN 이면 0
pub fn lv(l: &[f64], i: usize) -> f64 {
    l.get(i).copied().filter(|v| !v.is_nan()).unwrap_or(0.0)
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
    fn iso() {
        assert_eq!(iso_sec("2026-11-20T06:35:00+00:00"), Some(1_795_156_500));
        assert_eq!(iso_sec("2026-11-20T15:35:00.999+09:00"), Some(1_795_156_500));
        assert_eq!(iso_sec("2026-11-20T06:35:00Z"), Some(1_795_156_500));
        assert_eq!(iso_of(1_795_156_500), "2026-11-20T06:35:00.000Z");
        assert_eq!(iso_sec(&iso_of(951_782_400)), Some(951_782_400)); // 2000-02-29
    }
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
