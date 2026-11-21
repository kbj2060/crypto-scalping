//! CBOE 지연 옵션 체인 → 대시보드 Option 카드 요약(`scripts/cboe_option_summary_20261101.py`, 줄 번호는 그 파일) → `cboe_option_state.json`.
//! 계산 도구는 원본이 가져다 쓰는 Deribit 쪽(`collect_deribit_option_gex_20260815.py` 의 _skew_interp·_smile·_surface_stats·_atm_spread_iv·_const_maturity)을 같은 식으로.
//! numpy 식: interp(binary search · 정확히 같으면 그 값) · gradient(가운데 2차 · 끝 1차) · linspace(i·step + start · 끝 = stop) · 짝 더하기 sum · median.
//! pandas groupby sum = Kahan 보정 합(ladder) · Series.sum = numpy 짝 더하기. erf·exp·log·pow = libm(파이썬 math·numpy(X86_V4 끔)과 같은 함수).
//! 실현변동성 rv7 = 바이낸스 1시간봉 REST -- 부르는 쪽 klines(url) 가 준다(기본 꺼짐 -- RTCOLLECT_REST 규칙).
use crate::ob5m::np_sum;
use serde_json::{json, Map, Value};
use std::path::Path;

extern "C" {
    fn erf(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64;
}

fn erf_(x: f64) -> f64 {
    unsafe { erf(x) }
}
fn pow_(x: f64, y: f64) -> f64 {
    unsafe { pow(x, y) }
}

pub const MULT: f64 = 100.0;
/// 바이낸스 선물 REST 기본 주소(klines -- 실현변동성).
pub const BINANCE: &str = "https://fapi.binance.com";

// ── 시각(미 동부) ─────────────────────────────────────────────────────────────

pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub(crate) fn civil(days: i64) -> (i64, i64, i64) {
    let k = crate::depth::hour_key(days * 86_400_000).unwrap();
    (k[..4].parse().unwrap(), k[5..7].parse().unwrap(), k[8..10].parse().unwrap())
}

/// 그 해 m 월 n 번째 일요일(일 번호).
fn nth_sunday(y: i64, m: i64, n: i64) -> i64 {
    let d1 = days_from_civil(y, m, 1);
    let wd = (d1 + 4).rem_euclid(7); // 1970-01-01 = 목(4) · 0 = 일
    d1 + (7 - wd) % 7 + 7 * (n - 1)
}

/// 미 동부 지역 시각 → UTC 초(zoneinfo fold=0: 없는 시각(3월 2시대)은 EST · 겹치는 시각(11월 1시대)은 EDT).
pub fn et_to_utc(y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64) -> i64 {
    let local = days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s;
    let (a, b) = (nth_sunday(y, 3, 2) * 86_400 + 3 * 3600, nth_sunday(y, 11, 1) * 86_400 + 2 * 3600);
    let off = if local >= a && local < b { 4 } else { 5 };
    local + off * 3600
}

/// UTC 초 → 미 동부 지역 (일 번호, 그날 초).
pub fn utc_to_et(t: i64) -> (i64, i64) {
    let y = civil(t.div_euclid(86_400)).0;
    let (a, b) = (nth_sunday(y, 3, 2) * 86_400 + 7 * 3600, nth_sunday(y, 11, 1) * 86_400 + 6 * 3600);
    let local = t - if t >= a && t < b { 4 } else { 5 } * 3600;
    (local.div_euclid(86_400), local.rem_euclid(86_400))
}

/// "YYYY-MM-DDTHH:MM:SS"(미 동부) → UTC 초.
fn parse_et(s: &str) -> Option<i64> {
    let n = |a: usize, b: usize| s.get(a..b)?.parse::<i64>().ok();
    (s.len() == 19).then_some(())?;
    Some(et_to_utc(n(0, 4)?, n(5, 7)?, n(8, 10)?, n(11, 13)?, n(14, 16)?, n(17, 19)?))
}

/// 옵션 기호 `<SYM><yymmdd><C|P><행사가×1000 8자리>` → (만기 UTC 초 = 미 동부 16:00, C?, 행사가).
fn parse_opt(o: &str) -> Option<(i64, bool, f64)> {
    let b = o.as_bytes();
    if b.len() < 15 {
        return None;
    }
    let tail = &o[o.len() - 15..];
    let head = &o[..o.len() - 15];
    if head.is_empty() || !head.bytes().all(|c| c.is_ascii_uppercase() || c == b'.') || !tail[..6].bytes().all(|c| c.is_ascii_digit())
        || !(tail.as_bytes()[6] == b'C' || tail.as_bytes()[6] == b'P') || !tail[7..].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let n = |a: usize, b: usize| tail[a..b].parse::<i64>().unwrap();
    Some((et_to_utc(2000 + n(0, 2), n(2, 4), n(4, 6), 16, 0, 0), tail.as_bytes()[6] == b'C', n(7, 15) as f64 / 1000.0))
}

fn utc_iso(t: i64, us: i64) -> String {
    let k = crate::depth::hour_key(t * 1000).unwrap();
    let b = format!("{k}:{:02}:{:02}", t.rem_euclid(3600) / 60, t.rem_euclid(60));
    if us == 0 { format!("{b}+00:00") } else { format!("{b}.{us:06}+00:00") }
}

// ── numpy ────────────────────────────────────────────────────────────────────

/// np.interp(x0, xp, fp) -- numpy compiled_base.c arr_interp(끝 밖 = 끝값).
pub fn interp(x0: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 1 {
        return fp[0];
    }
    if x0.is_nan() {
        return x0;
    }
    if x0 < xp[0] {
        return fp[0];
    }
    if x0 > xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|v| *v <= x0) - 1; // xp[j] <= x0 < xp[j+1]
    if j == n - 1 || xp[j] == x0 {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let r = slope * (x0 - xp[j]) + fp[j];
    if r.is_nan() {
        let r2 = slope * (x0 - xp[j + 1]) + fp[j + 1];
        if r2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return r2;
    }
    r
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    let mut y: Vec<f64> = (0..n).map(|i| i as f64 * step + a).collect();
    y[n - 1] = b;
    y
}

/// np.gradient(f) -- 간격 1.
fn gradient1(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut o = vec![0.0; n];
    for i in 1..n - 1 {
        o[i] = (f[i + 1] - f[i - 1]) / 2.0;
    }
    o[0] = (f[1] - f[0]) / 1.0;
    o[n - 1] = (f[n - 1] - f[n - 2]) / 1.0;
    o
}

/// np.gradient(f, x) -- 고르지 않은 간격(가운데 2차 · 끝 1차). 간격이 다 같으면 numpy 는 스칼라 길로 간다.
fn gradient_x(f: &[f64], x: &[f64]) -> Vec<f64> {
    let n = f.len();
    let dx: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
    if dx.iter().all(|d| *d == dx[0]) {
        let h = dx[0];
        let mut o = vec![0.0; n];
        for i in 1..n - 1 {
            o[i] = (f[i + 1] - f[i - 1]) / (2.0 * h);
        }
        o[0] = (f[1] - f[0]) / h;
        o[n - 1] = (f[n - 1] - f[n - 2]) / h;
        return o;
    }
    let mut o = vec![0.0; n];
    for i in 1..n - 1 {
        let (d1, d2) = (dx[i - 1], dx[i]);
        let a = -(d2) / (d1 * (d1 + d2));
        let b = (d2 - d1) / (d1 * d2);
        let c = d1 / (d2 * (d1 + d2));
        o[i] = a * f[i - 1] + b * f[i] + c * f[i + 1];
    }
    o[0] = (f[1] - f[0]) / dx[0];
    o[n - 1] = (f[n - 1] - f[n - 2]) / dx[n - 2];
    o
}

fn median(v: &mut [f64]) -> f64 {
    crate::ob5m::np_median(v)
}

fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn pyround_n(x: f64, n: usize) -> f64 {
    format!("{x:.n$}").parse().unwrap()
}

fn fnum(v: Option<&Value>) -> f64 {
    // float(o.get(k) or 0)
    match v {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::String(s)) if !s.is_empty() => s.trim().parse().unwrap_or(f64::NAN),
        Some(Value::Bool(b)) => *b as u8 as f64,
        _ => 0.0,
    }
}

fn freq(v: &Value) -> Result<f64, String> {
    // float(o[k]) -- 없거나 못 읽으면 예외
    crate::py_float(v).map_err(|_| "ValueError: float".to_string())
}

// ── 체인 ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Row {
    exp: i64,
    call: bool,
    strike: f64,
    iv: f64,
    bid: f64,
    ask: f64,
    oi: f64,
    days: f64,
}

/// `chain_frame`(:37-50).
fn chain_frame(options: &[Value], now: f64) -> Result<Vec<Row>, String> {
    let mut out = vec![];
    for o in options {
        let Some((exp, call, strike)) = o["option"].as_str().and_then(parse_opt) else { continue };
        let days = (exp as f64 - now) / 86400.0;
        if days <= 0.0 {
            continue;
        }
        out.push(Row { exp, call, strike, iv: freq(&o["iv"])? * 100.0, bid: freq(&o["bid"])?, ask: freq(&o["ask"])?, oi: freq(&o["open_interest"])?, days });
    }
    Ok(out)
}

fn bs_delta(fwd: f64, k: f64, iv_pct: f64, yrs: f64, call: bool) -> f64 {
    let s = iv_pct / 100.0;
    if fwd <= 0.0 || k <= 0.0 || s <= 0.0 || yrs <= 0.0 {
        return 0.0;
    }
    let d1 = ((fwd / k).ln() + 0.5 * s * s * yrs) / (s * yrs.sqrt());
    let nd1 = 0.5 * (1.0 + erf_(d1 / 2f64.sqrt()));
    if call { nd1 } else { nd1 - 1.0 }
}

fn sorted_by_strike(mut v: Vec<&Row>) -> Vec<&Row> {
    v.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap());
    v
}

fn argsort_interp(x: &[f64], y: &[f64], x0: f64) -> Option<f64> {
    let mut o: Vec<usize> = (0..x.len()).collect();
    o.sort_by(|a, b| x[*a].partial_cmp(&x[*b]).unwrap());
    let (xs, ys): (Vec<f64>, Vec<f64>) = (o.iter().map(|i| x[*i]).collect(), o.iter().map(|i| y[*i]).collect());
    (xs[0] <= x0 && x0 <= xs[xs.len() - 1]).then(|| interp(x0, &xs, &ys))
}

/// `_skew_interp`(D :215-237).
fn skew_interp(g: &[&Row], fwd: f64, yrs: f64) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("hours".into(), json!(yrs * 8760.0));
    for k in ["rr25i", "bf25i", "atm_i"] {
        out.insert(k.into(), Value::Null);
    }
    let oc = sorted_by_strike(g.iter().copied().filter(|r| r.call && r.strike >= fwd).collect());
    let op = sorted_by_strike(g.iter().copied().filter(|r| !r.call && r.strike <= fwd).collect());
    if oc.len() < 2 || op.len() < 2 || yrs <= 0.0 {
        return out;
    }
    let otm: Vec<&Row> = op.iter().copied().filter(|r| r.strike < fwd).chain(oc.iter().copied()).collect();
    let atm = argsort_interp(&otm.iter().map(|r| (r.strike / fwd).ln()).collect::<Vec<_>>(), &otm.iter().map(|r| r.iv).collect::<Vec<_>>(), 0.0);
    let dc: Vec<f64> = oc.iter().map(|r| bs_delta(fwd, r.strike, r.iv, yrs, true)).collect();
    let dp: Vec<f64> = op.iter().map(|r| bs_delta(fwd, r.strike, r.iv, yrs, false)).collect();
    let c25 = argsort_interp(&dc, &oc.iter().map(|r| r.iv).collect::<Vec<_>>(), 0.25);
    let p25 = argsort_interp(&dp, &op.iter().map(|r| r.iv).collect::<Vec<_>>(), -0.25);
    out.insert("atm_i".into(), json!(atm));
    if let (Some(c), Some(p)) = (c25, p25) {
        out.insert("rr25i".into(), json!(c - p));
        if let Some(a) = atm {
            out.insert("bf25i".into(), json!((c + p) / 2.0 - a));
        }
    }
    out
}

/// `_surface_stats`(D :251-289).
fn surface_stats(g: &[&Row], fwd: f64, yrs: f64) -> Map<String, Value> {
    let mut out = Map::new();
    for k in ["mfiv", "tail", "rn_skew", "rn_kurt"] {
        out.insert(k.into(), Value::Null);
    }
    let otm = sorted_by_strike(g.iter().copied().filter(|r| (!r.call && r.strike < fwd) || (r.call && r.strike >= fwd)).filter(|r| r.iv > 0.0).collect());
    if otm.len() < 3 || yrs <= 0.0 {
        return out;
    }
    let (kx, vy): (Vec<f64>, Vec<f64>) = (otm.iter().map(|r| (r.strike / fwd).ln()).collect(), otm.iter().map(|r| r.iv / 100.0).collect());
    let sm = |x: f64| interp(x, &kx, &vy);
    let sig0 = sm(0.0);
    let w = (4.5 * sig0 * yrs.sqrt()).max(0.06);
    let kk: Vec<f64> = linspace(-w, w, 801).into_iter().map(|x| fwd * x.exp()).collect(); // K
    let k_ = &kk;
    let lk: Vec<f64> = k_.iter().map(|k| (k / fwd).ln()).collect();
    let sq: Vec<f64> = lk.iter().map(|x| sm(*x) * yrs.sqrt()).collect();
    let n = |x: f64| 0.5 * (1.0 + erf_(x / 2f64.sqrt()));
    let mut call = vec![0.0; 801];
    let mut put = vec![0.0; 801];
    for i in 0..801 {
        let d1 = (-lk[i] + 0.5 * sq[i] * sq[i]) / sq[i];
        let d2 = d1 - sq[i];
        call[i] = fwd * n(d1) - k_[i] * n(d2);
        put[i] = call[i] - (fwd - k_[i]);
    }
    let q: Vec<f64> = (0..801).map(|i| if k_[i] < fwd { put[i] } else { call[i] }).collect();
    let dk = gradient1(k_);
    let var = 2.0 / yrs * np_sum(&(0..801).map(|i| q[i] / (k_[i] * k_[i]) * dk[i]).collect::<Vec<_>>());
    out.insert("mfiv".into(), if var > 0.0 { json!(var.sqrt() * 100.0) } else { Value::Null });
    let cdf: Vec<f64> = gradient_x(&call, k_).into_iter().map(|g| (1.0 + g).max(0.0).min(1.0)).collect();
    let mut tail = Map::new();
    for x in [2.0f64, 3.0, 5.0] {
        let lo = interp(fwd * (1.0 - x / 100.0), k_, &cdf);
        let hi = interp(fwd * (1.0 + x / 100.0), k_, &cdf);
        tail.insert(format!("{}", x as i64), json!(pyround_n((lo + (1.0 - hi)).max(0.0).min(1.0), 4)));
    }
    out.insert("tail".into(), Value::Object(tail));
    let sel = |f: &dyn Fn(usize) -> f64, up: bool| np_sum(&(0..801).filter(|i| (k_[*i] >= fwd) == up).map(f).collect::<Vec<_>>());
    let k2 = |i: usize| k_[i] * k_[i];
    let l2 = |i: usize| lk[i] * lk[i];
    let l3 = |i: usize| pow_(lk[i], 3.0);
    let v = sel(&|i| 2.0 * (1.0 - lk[i]) / k2(i) * call[i] * dk[i], true) + sel(&|i| 2.0 * (1.0 - lk[i]) / k2(i) * put[i] * dk[i], false);
    let w_ = sel(&|i| (6.0 * lk[i] - 3.0 * l2(i)) / k2(i) * call[i] * dk[i], true) + sel(&|i| (6.0 * lk[i] - 3.0 * l2(i)) / k2(i) * put[i] * dk[i], false);
    let x_ = sel(&|i| (12.0 * l2(i) - 4.0 * l3(i)) / k2(i) * call[i] * dk[i], true) + sel(&|i| (12.0 * l2(i) - 4.0 * l3(i)) / k2(i) * put[i] * dk[i], false);
    let mu = -v / 2.0 - w_ / 6.0 - x_ / 24.0;
    let den = v - mu * mu;
    if den > 0.0 {
        out.insert("rn_skew".into(), json!((w_ - 3.0 * mu * v + 2.0 * pow_(mu, 3.0)) / pow_(den, 1.5)));
        out.insert("rn_kurt".into(), json!((x_ - 4.0 * mu * w_ + 6.0 * mu * mu * v - 3.0 * pow_(mu, 4.0)) / pow_(den, 2.0)));
    }
    out
}

/// `_atm_spread_iv`(D :292-308) -- linear=True(USD 호가).
fn atm_spread_iv(g: &[&Row], fwd: f64, yrs: f64) -> Value {
    let x: Vec<&&Row> = g.iter().filter(|r| (r.strike / fwd - 1.0).abs() <= 0.02 && r.bid > 0.0 && r.ask > r.bid).collect();
    if x.is_empty() || yrs <= 0.0 {
        return Value::Null;
    }
    let mut out = vec![];
    for r in x {
        let sig = r.iv / 100.0;
        if sig <= 0.0 {
            continue;
        }
        let d1 = ((fwd / r.strike).ln() + 0.5 * sig * sig * yrs) / (sig * yrs.sqrt());
        let vega = fwd * (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt() * yrs.sqrt() / 100.0;
        let spread = (r.ask - r.bid) * 1.0;
        if vega > 0.0 {
            out.push(spread / vega);
        }
    }
    if out.is_empty() { Value::Null } else { json!(median(&mut out)) }
}

/// `_const_maturity`(D :311-322).
fn const_maturity(exps: &[Map<String, Value>], days: f64) -> Value {
    let tgt = days * 24.0;
    let f = |e: &Map<String, Value>, k: &str| e.get(k).and_then(Value::as_f64);
    let ok: Vec<&Map<String, Value>> = exps.iter().filter(|e| f(e, "rr25i").is_some() && f(e, "bf25i").is_some() && f(e, "atm_i").is_some()).collect();
    let mut a: Option<&Map<String, Value>> = None;
    for e in ok.iter().filter(|e| f(e, "hours").unwrap() <= tgt) {
        if a.is_none_or(|x| f(e, "hours").unwrap() > f(x, "hours").unwrap()) {
            a = Some(e);
        }
    }
    let mut b: Option<&Map<String, Value>> = None;
    for e in ok.iter().filter(|e| f(e, "hours").unwrap() >= tgt) {
        if b.is_none_or(|x| f(e, "hours").unwrap() < f(x, "hours").unwrap()) {
            b = Some(e);
        }
    }
    let (Some(a), Some(b)) = (a, b) else { return Value::Null };
    let (ha, hb) = (f(a, "hours").unwrap(), f(b, "hours").unwrap());
    let w = if hb == ha { 0.0 } else { (tgt - ha) / (hb - ha) };
    let var = (1.0 - w) * pow_(f(a, "atm_i").unwrap(), 2.0) * ha + w * pow_(f(b, "atm_i").unwrap(), 2.0) * hb;
    json!({"atm": if var > 0.0 { json!((var / tgt).sqrt()) } else { Value::Null },
           "rr": (1.0 - w) * f(a, "rr25i").unwrap() + w * f(b, "rr25i").unwrap(), "bf": (1.0 - w) * f(a, "bf25i").unwrap() + w * f(b, "bf25i").unwrap()})
}

/// `flow_today`(:53-72).
fn flow_today(options: &[Value], s: f64) -> Value {
    let mut rows: Vec<Vec<Value>> = vec![];
    let mut key: Vec<(f64, i64)> = vec![];
    let (mut tc, mut tp) = ([0.0f64, 0.0], [0.0f64, 0.0]);
    for o in options {
        let Some((exp, call, strike)) = o["option"].as_str().and_then(parse_opt) else { continue };
        let vol = fnum(o.get("volume"));
        if vol <= 0.0 {
            continue;
        }
        let ltp = fnum(o.get("last_trade_price"));
        let px = if ltp != 0.0 { ltp } else { (fnum(o.get("bid")) + fnum(o.get("ask"))) / 2.0 };
        let usd = vol * px * MULT;
        let t = if call { &mut tc } else { &mut tp };
        t[0] += vol;
        t[1] += usd;
        let tm = o.get("last_trade_time").and_then(Value::as_str).filter(|s| !s.is_empty()).and_then(parse_et).map(|x| x * 1000);
        let dl = fnum(o.get("delta"));
        let expo = pyround(vol * if dl == dl { dl.abs() } else { 0.0 } * MULT * s);
        key.push((-(expo as f64), -pyround(usd)));
        rows.push(vec![json!(tm), json!(pyround(usd)), json!(exp * 1000), json!(if call { "C" } else { "P" }), json!(strike), json!(vol), json!(px), json!(expo)]);
    }
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|a, b| key[*a].partial_cmp(&key[*b]).unwrap()); // 안정 정렬(-노출, -명목)
    let top: Vec<Value> = idx.iter().take(6).map(|i| Value::from(rows[*i].clone())).collect();
    json!({"call": [pyround(tc[0]), pyround(tc[1])], "put": [pyround(tp[0]), pyround(tp[1])], "top": top})
}

/// `gex_levels`(:75-89).
fn gex_levels(options: &[Value], s: f64, now: f64) -> Value {
    let mut by: Vec<(f64, f64)> = vec![];
    for o in options {
        let Some((exp, call, k)) = o["option"].as_str().and_then(parse_opt) else { continue };
        if exp as f64 <= now {
            continue;
        }
        let g = fnum(o.get("gamma")) * fnum(o.get("open_interest")) * MULT * s * s * 0.01;
        let add = if call { g } else { -g };
        match by.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += add,
            None => by.push((k, 0.0 + add)),
        }
    }
    let net = pyround(by.iter().fold(0.0, |a, x| a + x.1));
    let mut lv: Vec<(f64, f64)> = by.iter().copied().filter(|(k, _)| (k / s - 1.0).abs() <= 0.30).collect();
    lv.sort_by(|a, b| (-a.1.abs()).partial_cmp(&(-b.1.abs())).unwrap());
    let pick = |pos: bool| lv.iter().filter(|x| if pos { x.1 > 0.0 } else { x.1 < 0.0 }).take(3).map(|(k, v)| json!([k, pyround(*v)])).collect::<Vec<_>>();
    json!({"net": net, "pos": pick(true), "neg": pick(false)})
}

/// `gamma_flip`(:92-124).
fn gamma_flip(options: &[Value], s: f64, now: f64) -> Value {
    let mut rows = vec![];
    for o in options {
        let Some((exp, call, k)) = o["option"].as_str().and_then(parse_opt) else { continue };
        let t = (exp as f64 - now) / (365.0 * 86400.0);
        let (iv, oi) = (fnum(o.get("iv")), fnum(o.get("open_interest")));
        if t > 0.0 && iv > 0.0 && oi > 0.0 {
            rows.push((k, iv, t, oi, if call { 1.0 } else { -1.0 }));
        }
    }
    if rows.is_empty() {
        return Value::Null;
    }
    let sd: Vec<f64> = rows.iter().map(|r| r.1 * r.2.sqrt()).collect();
    let net = |x: f64| -> f64 {
        let terms: Vec<f64> = rows.iter().zip(&sd).map(|(r, sd)| {
            let d1 = ((x / r.0).ln() + 0.5 * sd * sd) / sd;
            let g = (-0.5 * d1 * d1).exp() / ((2.0 * std::f64::consts::PI).sqrt() * x * sd);
            r.4 * r.3 * g
        }).collect();
        np_sum(&terms) * MULT * (x * x) * 0.01
    };
    let grid: Vec<f64> = linspace(1.0 - 0.30, 1.0 + 0.30, 241).into_iter().map(|g| s * g).collect();
    let v: Vec<f64> = grid.iter().map(|x| net(*x)).collect();
    let now_v = net(s);
    let sg = |x: f64| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 };
    let flips: Vec<f64> = (0..v.len() - 1).filter(|i| sg(v[*i]) * sg(v[i + 1]) < 0.0).map(|i| grid[i] - v[i] * (grid[i + 1] - grid[i]) / (v[i + 1] - v[i])).collect();
    let mut flip: Option<f64> = None;
    for p in &flips {
        if flip.is_none_or(|f| (p - s).abs() < (f - s).abs()) {
            flip = Some(*p);
        }
    }
    let flip = flip.filter(|f| *f != 0.0);
    json!({"net": pyround(now_v), "regime": if now_v > 0.0 { "long" } else { "short" }, "flip": flip.map(|f| pyround_n(f, 2)),
           "flip_pct": flip.map(|f| pyround_n((f / s - 1.0) * 100.0, 2)), "n": rows.len()})
}

/// pandas groupby(strike)[[c, p]].sum() -- 무리마다 Kahan 보정 합(pandas _libs group_sum) · 행사가 오름차순.
fn ladder(rows: &[(f64, f64, f64)]) -> Vec<Value> {
    let mut ks: Vec<f64> = rows.iter().map(|r| r.0).collect();
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ks.dedup();
    ks.iter().map(|k| {
        let mut acc = [(0.0f64, 0.0f64); 2];
        for r in rows.iter().filter(|r| r.0 == *k) {
            for (j, v) in [r.1, r.2].into_iter().enumerate() {
                let (sum, comp) = acc[j];
                let y = v - comp;
                let t = sum + y;
                acc[j] = (t, t - sum - y);
            }
        }
        json!([k, pyround(acc[0].0), pyround(acc[1].0)])
    }).collect()
}

/// `summarize`(:187-238). Err = 원본 예외(빈 체인 등).
pub fn summarize(data: &Value, now: f64) -> Result<Map<String, Value>, String> {
    let s = freq(&data["current_price"])?;
    let options: Vec<Value> = data["options"].as_array().cloned().ok_or("KeyError: 'options'")?;
    let ch = chain_frame(&options, now)?;
    if ch.is_empty() {
        return Err("KeyError: 'expiration_ts'".into()); // 원본: 빈 DataFrame 의 groupby
    }
    let last = parse_et(data["last_trade_time"].as_str().ok_or("KeyError: 'last_trade_time'")?).ok_or("ValueError: strptime")?;
    let iv30 = freq(&data["iv30"])?;
    let mut out = Map::new();
    out.insert("recorded_at_utc".into(), json!(utc_iso(last, 0)));
    out.insert("currency".into(), data["symbol"].clone());
    out.insert("src".into(), json!("CBOE 지연"));
    out.insert("dvol".into(), Value::Null);
    out.insert("index".into(), json!(s));
    out.insert("rv7".into(), Value::Null);
    out.insert("iv30".into(), if iv30 != 0.0 { json!(iv30) } else { Value::Null });
    out.insert("hedge".into(), json!([]));
    let mut expiries: Vec<i64> = ch.iter().map(|r| r.exp).collect();
    expiries.sort();
    expiries.dedup();
    let mut exps: Vec<(Map<String, Value>, Vec<&Row>, f64)> = vec![];
    let usd = MULT * s;
    for e in &expiries {
        let g: Vec<&Row> = ch.iter().filter(|r| r.exp == *e).collect();
        let gi: Vec<&Row> = g.iter().copied().filter(|r| r.iv > 0.0).collect();
        if !gi.iter().any(|r| r.call) || !gi.iter().any(|r| !r.call) {
            continue;
        }
        let yrs = g[0].days / 365.0;
        let calls: Vec<&Row> = g.iter().copied().filter(|r| r.call).collect();
        let puts: Vec<&Row> = g.iter().copied().filter(|r| !r.call).collect();
        // gi.iloc[(strike - S).abs().argsort()[:1]] -- 가장 가까운 행사가(같은 거리면 앞 줄 -- 같은 행사가 콜·풋이면 값이 같다)
        let mut best = gi[0];
        for r in &gi {
            if (r.strike - s).abs() < (best.strike - s).abs() {
                best = r;
            }
        }
        let atm_k = best.strike;
        let atm_rows: Vec<f64> = gi.iter().filter(|r| r.strike == atm_k).map(|r| r.iv).collect();
        let coi = np_sum(&calls.iter().map(|r| r.oi).collect::<Vec<_>>());
        let poi = np_sum(&puts.iter().map(|r| r.oi).collect::<Vec<_>>());
        let mut uk: Vec<f64> = g.iter().map(|r| r.strike).collect();
        uk.sort_by(|a, b| a.partial_cmp(b).unwrap());
        uk.dedup();
        let pain_of = |p: f64| np_sum(&calls.iter().map(|r| (p - r.strike).max(0.0) * r.oi).collect::<Vec<_>>()) + np_sum(&puts.iter().map(|r| (r.strike - p).max(0.0) * r.oi).collect::<Vec<_>>());
        let mut pain = uk[0];
        for p in &uk {
            if pain_of(*p) < pain_of(pain) {
                pain = *p;
            }
        }
        let mut m = Map::new();
        m.insert("exp_ms".into(), json!(e * 1000));
        m.insert("atm_iv".into(), json!(np_sum(&atm_rows) / atm_rows.len() as f64));
        m.insert("call_oi_usd".into(), json!(coi * usd));
        m.insert("put_oi_usd".into(), json!(poi * usd));
        m.insert("pc".into(), if coi != 0.0 { json!(poi / coi) } else { Value::Null });
        m.insert("pain".into(), json!(pain));
        m.extend(skew_interp(&gi, s, yrs));
        m.insert("fwd".into(), json!(s));
        m.insert("atm_oi_usd".into(), json!(np_sum(&g.iter().filter(|r| (r.strike / s - 1.0).abs() <= 0.02).map(|r| r.oi).collect::<Vec<_>>()) * usd));
        exps.push((m, gi, yrs));
    }
    let tot = exps.iter().fold(0.0, |a, e| a + (e.0["call_oi_usd"].as_f64().unwrap() + e.0["put_oi_usd"].as_f64().unwrap()));
    for e in exps.iter_mut() {
        let v = e.0["call_oi_usd"].as_f64().unwrap() + e.0["put_oi_usd"].as_f64().unwrap();
        e.0.insert("oi_share".into(), if tot != 0.0 { json!(v / tot) } else { Value::Null });
    }
    let hours = |e: &(Map<String, Value>, Vec<&Row>, f64)| e.0["hours"].as_f64().unwrap();
    let live: Vec<usize> = (0..exps.len()).filter(|i| hours(&exps[*i]) >= 3.0).collect();
    let near = |d: f64| -> Option<usize> {
        let mut b: Option<usize> = None;
        for i in &live {
            if b.is_none_or(|x| (hours(&exps[*i]) - d * 24.0).abs() < (hours(&exps[x]) - d * 24.0).abs()) {
                b = Some(*i);
            }
        }
        b
    };
    let pick = [("front", live.first().copied()), ("7", near(7.0)), ("30", near(30.0))];
    let mut surf = Map::new();
    for (k, e) in pick {
        surf.insert(k.into(), match e {
            Some(i) => {
                let mut m = Map::new();
                m.insert("exp_ms".into(), exps[i].0["exp_ms"].clone());
                m.insert("hours".into(), json!(pyround_n(hours(&exps[i]), 2)));
                m.insert("fwd".into(), json!(s));
                m.extend(surface_stats(&exps[i].1, s, exps[i].2));
                Value::Object(m)
            }
            None => Value::Null,
        });
    }
    out.insert("surface".into(), Value::Object(surf));
    out.insert("atm_spread_iv".into(), pick[1].1.map_or(Value::Null, |i| atm_spread_iv(&exps[i].1, s, exps[i].2)));
    out.insert("front_atm_oi_usd".into(), exps.first().map_or(Value::Null, |e| e.0["atm_oi_usd"].clone()));
    let em: Vec<Map<String, Value>> = exps.iter().map(|e| e.0.clone()).collect();
    out.insert("expiries".into(), Value::from(em.iter().take(8).map(|m| Value::Object(m.clone())).collect::<Vec<_>>()));
    out.insert("flow_today".into(), flow_today(&options, s));
    out.insert("gex".into(), gex_levels(&options, s, now));
    out.insert("gamma".into(), gamma_flip(&options, s, now));
    let mut cm = Map::new();
    for d in [1.0f64, 7.0, 30.0, 60.0] {
        cm.insert(format!("{}", d as i64), const_maturity(&em, d));
    }
    out.insert("cm".into(), Value::Object(cm));
    let band: Vec<&Row> = ch.iter().filter(|r| r.strike >= s * 0.92 && r.strike <= s * 1.08).collect();
    let first = ch.iter().map(|r| r.exp).min();
    let lad = |f: &dyn Fn(&Row) -> bool| ladder(&band.iter().filter(|r| f(r)).map(|r| {
        let u = r.oi * MULT * s;
        (r.strike, if r.call { u } else { 0.0 }, if r.call { 0.0 } else { u })
    }).collect::<Vec<_>>());
    let mut st = Map::new();
    st.insert("front".into(), Value::from(first.map_or(vec![], |f| lad(&|r: &Row| r.exp == f))));
    st.insert("week".into(), Value::from(lad(&|r: &Row| r.days <= 7.0)));
    st.insert("all".into(), Value::from(lad(&|r: &Row| r.days > 0.0)));
    st.insert("front_exp_ms".into(), first.map_or(Value::Null, |f| json!(f * 1000)));
    out.insert("strikes".into(), Value::Object(st));
    Ok(out)
}

/// `volume_vs_normal`(:127-152): 지난 거래일마다 «같은 미 동부 시각 이전의 마지막 스냅숏» 거래량 합의 평균(최근 10일).
fn volume_vs_normal(sym_dir: &Path, now_day: i64, now_sec: i64, today_n: f64) -> Value {
    let clock = now_sec / 60;
    let mut pick: std::collections::BTreeMap<i64, (i64, std::path::PathBuf)> = std::collections::BTreeMap::new();
    for d in std::fs::read_dir(sym_dir).into_iter().flatten().flatten() {
        let dp = d.path();
        let Some(dn) = dp.file_name().and_then(|x| x.to_str()).map(str::to_string) else { continue };
        for f in std::fs::read_dir(&dp).into_iter().flatten().flatten() {
            let fp = f.path();
            let name = fp.file_name().and_then(|x| x.to_str()).unwrap_or("").to_string();
            if !name.ends_with(".json.gz") || dn.len() != 8 || name.len() < 6 {
                continue;
            }
            let n = |s: &str| s.parse::<i64>().ok();
            let (Some(y), Some(mo), Some(da), Some(h), Some(mi), Some(se)) = (n(&dn[..4]), n(&dn[4..6]), n(&dn[6..8]), n(&name[..2]), n(&name[2..4]), n(&name[4..6])) else { continue };
            if !(1..=12).contains(&mo) || !(1..=31).contains(&da) || h > 23 || mi > 59 || se > 59 {
                continue;
            }
            let t = days_from_civil(y, mo, da) * 86_400 + h * 3600 + mi * 60 + se;
            let (day, sec) = utc_to_et(t);
            if day >= now_day || sec / 60 > clock {
                continue;
            }
            if pick.get(&day).is_none_or(|x| t > x.0) {
                pick.insert(day, (t, fp.clone()));
            }
        }
    }
    let days: Vec<&(i64, std::path::PathBuf)> = pick.values().collect();
    let mut vols = vec![];
    for (_, p) in days.iter().skip(days.len().saturating_sub(10)) {
        let Ok(raw) = std::fs::read(p) else { continue };
        let mut txt = String::new();
        if std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&raw[..]), &mut txt).is_err() {
            continue;
        }
        let Ok(j) = parse_lenient(&txt) else { continue };
        let Some(ops) = j.get("data").and_then(|d| d.get("options")).and_then(Value::as_array) else { continue };
        vols.push(ops.iter().fold(0.0, |a, o| a + fnum(o.get("volume"))));
    }
    let vols: Vec<f64> = vols.into_iter().filter(|v| *v > 0.0).collect();
    if vols.is_empty() {
        return Value::Null;
    }
    let avg = vols.iter().fold(0.0, |a, v| a + v) / vols.len() as f64;
    json!({"today": pyround(today_n), "avg": pyround(avg), "ratio": pyround_n(today_n / avg, 3), "days": vols.len()})
}

/// `realized_vol_7d`(:158-184). klines(url) = 응답 본문(Err = 실패 -- REST 꺼짐 포함).
pub fn realized_vol_7d(sym: &str, cache: &Path, now: f64, klines: &mut dyn FnMut(&str) -> Result<Vec<u8>, String>) -> Option<f64> {
    let hit: Option<(f64, Option<f64>)> = std::fs::read_to_string(cache).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|j| j.get(sym).cloned()).and_then(|h| Some((h.get(0)?.as_f64()?, h.get(1).and_then(Value::as_f64))));
    if let Some((t, v)) = hit {
        if now - t < 6.0 * 3600.0 {
            return v;
        }
    }
    let got = (|| -> Result<Option<f64>, String> {
        let body = klines(&format!("{BINANCE}/fapi/v1/klines?symbol={sym}USDT&interval=1h&limit=169"))?;
        let j: Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
        let cl: Vec<f64> = j.as_array().ok_or("not list")?.iter().map(|k| k.get(4).and_then(|x| crate::py_float(x).ok()).ok_or("bad")).collect::<Result<_, _>>()?;
        let rets: Vec<f64> = cl.windows(2).map(|w| w[1] / w[0] - 1.0).collect();
        Ok(if rets.len() > 24 { Some(pow_(rets.iter().fold(0.0, |a, x| a + x * x) / rets.len() as f64 * 24.0 * 365.0, 0.5) * 100.0) } else { None })
    })();
    match got {
        Err(_) => hit.filter(|h| now - h.0 < 24.0 * 3600.0).and_then(|h| h.1),
        Ok(v) => {
            if let Some(x) = v {
                let mut cur: Map<String, Value> = std::fs::read_to_string(cache).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
                cur.insert(sym.into(), json!([now, x]));
                let _ = std::fs::write(cache, crate::py_dumps_default(&Value::Object(cur)));
            }
            v
        }
    }
}

/// 파이썬 json 은 NaN·Infinity 낱말을 받는다(CBOE 델타에 실제로 왔다 -- 원본 주석) · serde_json 은 못 받는다 → null 로(그 칸은 원본도 0 으로 친다).
/// ponytail: 글자 안의 «NaN» 까지 바꾼다 -- 옵션 기호·시각 글자에는 없다.
fn parse_lenient(txt: &str) -> serde_json::Result<Value> {
    serde_json::from_str::<Value>(txt).or_else(|_| serde_json::from_str::<Value>(&txt.replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null")))
}

/// `_clean`: 유한하지 않은 실수 → None(serde_json 은 NaN 을 Null 로 이미 바꾼다 -- 그대로 둔다).
fn clean(v: Value) -> Value {
    v
}

/// `write_state`(:250-278): 종목 하나를 상태 파일에 합친다. now = 시계(벽 · 시험 = 가짜) · rv7 = 실현변동성(None 이면 칸 None).
pub fn write_state(snapshot: &Path, state: &Path, now: f64, rv7: &mut dyn FnMut(&str) -> Option<f64>) -> Result<(), String> {
    let raw = std::fs::read(snapshot).map_err(|e| format!("OSError: {e}"))?;
    let mut txt = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&raw[..]), &mut txt).map_err(|e| format!("OSError: {e}"))?;
    let data = parse_lenient(&txt).map_err(|e| format!("JSONDecodeError: {e}"))?["data"].clone();
    let mut opt = summarize(&data, now)?;
    let sym = data["symbol"].as_str().unwrap_or("").to_string();
    opt.insert("rv7".into(), json!(rv7(&sym)));
    let ft0 = opt["flow_today"].clone();
    // 시계 = 수집 시각(파일 이름 <UTC날짜>/<HHMMSS>) -- 이름이 시각이 아니면 데이터 시각
    let name = |p: &Path| p.file_name().and_then(|x| x.to_str()).unwrap_or("").to_string();
    let (dn, fname) = (snapshot.parent().map(name).unwrap_or_default(), name(snapshot));
    let parsed = (|| {
        let n = |s: &str| s.parse::<i64>().ok();
        if dn.len() != 8 || fname.len() < 6 {
            return None;
        }
        let (y, mo, d, h, mi, s) = (n(&dn[..4])?, n(&dn[4..6])?, n(&dn[6..8])?, n(&fname[..2])?, n(&fname[2..4])?, n(&fname[4..6])?);
        ((1..=12).contains(&mo) && (1..=31).contains(&d) && h < 24 && mi < 60 && s < 60).then(|| days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s)
    })();
    let last = parse_et(data["last_trade_time"].as_str().unwrap_or("")).ok_or("ValueError: strptime")?;
    let (cday, csec) = utc_to_et(parsed.unwrap_or(last));
    let today = ft0["call"][0].as_f64().unwrap_or(0.0) + ft0["put"][0].as_f64().unwrap_or(0.0);
    opt.insert("vol_norm".into(), volume_vs_normal(snapshot.parent().and_then(Path::parent).unwrap_or(Path::new(".")), cday, csec, today));
    let opt = clean(Value::Object(opt));
    let mut cur: Map<String, Value> = std::fs::read_to_string(state).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|j| j.get("currencies").and_then(Value::as_object).cloned()).unwrap_or_default();
    let prev: Vec<Value> = cur.get(&sym).and_then(|c| c.get("options")).and_then(|o| o.get("flow_series")).and_then(Value::as_array).cloned().unwrap_or_default();
    let tm = last * 1000;
    let ft = &opt["flow_today"];
    let mut pts: Vec<Value> = prev.into_iter().filter(|p| p[0].as_f64().is_some_and(|t| t > tm as f64 - 24.0 * 3600e3 && t != tm as f64)).collect();
    pts.push(json!([tm, ft["call"][1], ft["put"][1]]));
    pts.sort_by(|a, b| {
        let k = |v: &Value| (v[0].as_f64().unwrap_or(0.0), v[1].as_f64().unwrap_or(0.0), v[2].as_f64().unwrap_or(0.0));
        k(a).partial_cmp(&k(b)).unwrap()
    });
    let mut opt = opt;
    opt.as_object_mut().unwrap().insert("flow_series".into(), Value::from(pts));
    let rec = opt["recorded_at_utc"].clone();
    let spot = opt["index"].clone();
    cur.insert(sym.clone(), json!({"opt_hist": {}, "options": opt, "recorded_at_utc": rec, "spot_price": spot}));
    let (s, us) = (now.floor() as i64, ((now - now.floor()) * 1e6).round_ties_even() as i64);
    let (s, us) = if us >= 1_000_000 { (s + 1, us - 1_000_000) } else { (s, us) };
    let body = json!({"generated_at": utc_iso(s, us), "currencies": cur});
    let tmp = state.with_file_name(format!("{}.{}.tmp", state.file_name().unwrap().to_string_lossy(), std::process::id()));
    if let Some(d) = state.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(&tmp, crate::py_dumps_default_utf8(&body)).and_then(|_| std::fs::rename(&tmp, state)).map_err(|e| format!("OSError: {e}"))
}
