//! 방향 전진 기록기 · 개장 되돌림 · 4h 변동성 HGB(지평 사다리 · QX4 실패 때 대체) · 7일/4~13주 신호 (2026-11-21 · R5).
//! 원본: scripts/live_eth_direction_forward_recorder_20261010.py `decide`·`fund_resid`·`record` (전진 판정 dir4h_p2 2027-03-14 · dir3asset 2027-04-08 이 읽는 줄) ·
//! dashboard/dir_signals.py `open_rev_from_klines`·`vol4h_from_klines`·`dir_signals_payload`(`_coinbase_btc`·`_binance_btc`·`_fred`·`asof`·`liq_score`·`cb_premium_z`·`_sig`) ·
//! scripts/live_eth_sizing_vol_model_20260912.py `build_features`(= qx4::svm_features)·`predict_vol` · 서버 api_vol_forecast 의 HGB 쪽 klines 10쪽.
//! HTTP 는 모른다: 부르는 쪽이 `fetch(전체 URL)` 을 넘긴다(URL 글자 = 원본이 만드는 글자).
//! 순서 규칙: np.std(ddof=1) = 짝 합 평균 → (x−평균)² 짝 합 ÷ (n−1) · np.mean(시드, axis=0) = 시드 차례 합 ÷ 8 · Python sum = 차례 합(3.11).
use crate::box_hold::{npsum, py_round};
use crate::risk_radar::{days_from_civil, iso};
use crate::sr_lean::{jf, roll_std};
use crate::{predict_proba, Model};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

const NAN: f64 = f64::NAN;
const DAY: i64 = 86_400_000;
const SYMS: [&str; 3] = ["ETHUSDT", "BTCUSDT", "SOLUSDT"];
const FLOW: [(&str, f64, f64); 3] = [("ETHUSDT", 2.547, 264.279), ("BTCUSDT", 1.627, 175.668), ("SOLUSDT", 5.711, 411.739)];
const WARMUP: usize = 400;
const VOL_P: (f64, f64) = (-1.4950, 5.1671);

pub type Fetch<'a> = dyn FnMut(&str) -> Result<Value, String> + 'a;

fn num(v: &Value) -> f64 {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(NAN)
}
fn int(v: &Value) -> i64 {
    v.as_i64().or_else(|| v.as_f64().map(|x| x as i64)).or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(0)
}
fn sign(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { x } // np.sign(±0) = ±0 · NaN = NaN
}
fn isign(x: f64) -> i64 {
    sign(x) as i64
}
/// json.dumps(ensure_ascii=True) 의 문자열.
pub fn jstr_ascii(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    o.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => o.push(c),
        }
    }
    o + "\""
}
fn day_str(ms: i64) -> String {
    iso(ms)[..10].to_string()
}

/// np.std(x, ddof=1)
fn np_std1(x: &[f64]) -> f64 {
    let m = npsum(x) / x.len() as f64;
    let d: Vec<f64> = x.iter().map(|v| (v - m) * (v - m)).collect();
    (npsum(&d) / (x.len() as f64 - 1.0).max(0.0)).sqrt()
}

/// 결정 시각 기준값(원본 decide). day0 = 정수 ms 또는(서버 시계 = 실수 ms) 실수 -- 실수면 exit_due_ms 가 실수로 적힌다.
pub struct Decision {
    pub date: String,
    votes: f64,
    trend_pos: i64,
    p0: f64,
    f30: f64,
    sig30: f64,
    pub open_pos: i64,
    p30: f64,
    exit_due: f64,
    float_clock: bool,
    hour: i64,
    fund: Option<(f64, f64, i64)>,
}

/// decide(daily, m5, now_ms, sym, hour). 23:55·00:25 봉이 없으면 Err("KeyError").
pub fn decide(daily: &[Value], m5: &[Value], now_ms: f64, float_clock: bool, sym: &str, hour: i64) -> Result<Decision, String> {
    let dc: Vec<f64> = daily.iter().map(|k| num(&k[4])).collect();
    let n = dc.len();
    if n < 91 {
        return Err("IndexError: index out of bounds".into());
    }
    let votes = npsum(&[7, 14, 28, 56, 90].map(|l| sign(dc[n - 1] - dc[n - 1 - l]))) / 5.0;
    let day0 = (now_ms / DAY as f64).floor() * DAY as f64 + (hour * 3_600_000) as f64;
    let mut by_open: BTreeMap<i64, f64> = BTreeMap::new();
    for k in m5 {
        by_open.insert(int(&k[0]), num(&k[4]));
    }
    let d0 = day0 as i64;
    let p0 = *by_open.get(&(d0 - 300_000)).ok_or("KeyError")?;
    let p30 = *by_open.get(&(d0 + 1_500_000)).ok_or("KeyError")?;
    let opens: Vec<f64> = by_open.range(..=d0 - 300_000).map(|x| *x.1).collect();
    let lc: Vec<f64> = opens[opens.len().saturating_sub(289)..].iter().map(|x| x.ln()).collect();
    let dif: Vec<f64> = lc.windows(2).map(|w| w[1] - w[0]).collect();
    let sig30 = np_std1(&dif) * 6f64.sqrt() * 1e4;
    let f30 = (p30 / p0).ln() * 1e4;
    // fund_resid
    let rows: Vec<&Value> = m5.iter().filter(|k| (d0..d0 + 1_800_000).contains(&int(&k[0]))).collect();
    let fund = if rows.len() == 6 && rows[0].as_array().is_some_and(|a| a.len() >= 10) {
        let (mut v, mut tb) = (0.0, 0.0);
        for k in &rows {
            v += num(&k[5]);
        }
        for k in &rows {
            tb += num(&k[9]);
        }
        let imb = if v > 0.0 { (2.0 * tb - v) / v } else { 0.0 };
        let (_, b0, b1) = FLOW.iter().find(|x| x.0 == sym).copied().ok_or("KeyError")?;
        let res = f30 - (b0 + b1 * imb);
        Some((imb, res, if res.abs() > 0.5 * sig30 { -isign(res) } else { 0 }))
    } else {
        None
    };
    Ok(Decision {
        date: day_str(d0),
        votes,
        trend_pos: if votes.abs() >= 0.5 { isign(votes) } else { 0 },
        p0,
        f30,
        sig30,
        open_pos: if f30.abs() > 0.5 * sig30 { -isign(f30) } else { 0 },
        p30,
        exit_due: day0 + (4 * 3_600_000 + 1_800_000) as f64,
        float_clock,
        hour,
        fund,
    })
}

impl Decision {
    fn exit_s(&self) -> String {
        if self.float_clock { jf(self.exit_due) } else { (self.exit_due as i64).to_string() }
    }
    /// 기록 줄의 decide 몫(json.dumps 기본 구분자).
    fn fields(&self) -> String {
        let mut s = format!("\"date\": \"{}\", \"votes\": {}, \"trend_pos\": {}, \"trend_entry_px\": {}, \"f30_bp\": {}, \"sigma30_bp\": {}, \"open_pos\": {}, \"entry_px\": {}, \"exit_due_ms\": {}, \"hour\": {}",
                            self.date, jf(self.votes), self.trend_pos, jf(self.p0), jf(py_round(self.f30, 2)), jf(py_round(self.sig30, 2)), self.open_pos, jf(self.p30), self.exit_s(), self.hour);
        if let Some((imb, res, fp)) = self.fund {
            s += &format!(", \"imb30\": {}, \"resid_bp\": {}, \"fund_pos\": {fp}", jf(py_round(imb, 5)), jf(py_round(res, 2)));
        }
        s
    }
}

fn klines_url(base: &str, sym: &str, iv: &str, limit: usize) -> String {
    format!("{base}/fapi/v1/klines?symbol={sym}&interval={iv}&limit={limit}")
}
fn arr(v: Value) -> Result<Vec<Value>, String> {
    match v {
        Value::Array(a) => Ok(a),
        o => Err(format!("klines: {o}")),
    }
}

/// record(): 00·08·16 UTC 31분 뒤 그 시간 안에서만 · 3자산 줄을 하나씩 덧붙인다(뒤 자산이 실패하면 앞 줄은 남는다 -- 원본과 같다). 반환 = 쓴 줄 수.
pub fn record(fetch: &mut Fetch, base: &str, now: i64, out: &Path) -> Result<usize, String> {
    let hour = now.rem_euclid(DAY) / 3_600_000;
    if ![0, 8, 16].contains(&hour) || now % 3_600_000 < 1_860_000 {
        return Err("00:31·08:31·16:31 UTC 이후 그 시간 안에서 실행".into());
    }
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let mut n = 0;
    for sym in SYMS {
        let mut daily = arr(fetch(&klines_url(base, sym, "1d", 93))?)?;
        daily.pop();
        let m5: Vec<Value> = arr(fetch(&klines_url(base, sym, "5m", 320))?)?.into_iter().filter(|k| int(&k[0]) + 300_000 <= now).collect();
        let d = decide(&daily, &m5, (now - hour * 3_600_000) as f64, false, sym, hour)?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).map_err(|e| format!("{}: {e}", out.display()))?;
        writeln!(f, "{{\"sym\": \"{sym}\", \"recorded_ms\": {now}, {}}}", d.fields()).map_err(|e| e.to_string())?;
        n += 1;
    }
    Ok(n)
}

/// open_rev_from_klines(daily, m5, now_ms) → JSON 글자. float_clock = 서버(time.time()*1000 실수) 시계.
pub fn open_rev(daily: &[Value], m5: &[Value], now_ms: f64, float_clock: bool) -> Result<String, String> {
    let day0 = (now_ms as i64).div_euclid(DAY) * DAY;
    let (entry, exit) = (day0 + 1_800_000, day0 + 16_200_000);
    if now_ms < entry as f64 {
        return Ok(format!("{{\"ok\": true, \"pending\": true, \"date\": \"{}\", \"open_pos\": 0, \"entry_ms\": {entry}, \"exit_due_ms\": {exit}}}", day_str(day0)));
    }
    let dc: Vec<Value> = daily.iter().filter(|k| (int(&k[6]) as f64) < now_ms).cloned().collect();
    let mc: Vec<Value> = m5.iter().filter(|k| (int(&k[6]) as f64) < now_ms).cloned().collect();
    let d = decide(&dc, &mc, now_ms, float_clock, "ETHUSDT", 0)?;
    Ok(format!("{{\"ok\": true, \"pending\": false, \"date\": \"{}\", \"f30_bp\": {}, \"sigma30_bp\": {}, \"open_pos\": {}, \"entry_px\": {}, \"exit_due_ms\": {}, \"entry_ms\": {entry}}}",
               d.date, jf(py_round(d.f30, 2)), jf(py_round(d.sig30, 2)), d.open_pos, jf(d.p30), d.exit_s()))
}

// ── 4h 변동성 HGB(배포 22열 · 8시드)
pub struct Svm {
    features: Vec<String>,
    models: Vec<Model>,
    train_end: Option<String>,
    expand_logit: Option<(f64, f64)>,
}
impl Svm {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let e = |x: serde_json::Error| x.to_string();
        let el: Option<Vec<f64>> = serde_json::from_value(v["expand_logit"].clone()).map_err(e)?;
        Ok(Svm {
            features: serde_json::from_value(v["features"].clone()).map_err(e)?,
            models: serde_json::from_value(v["models"].clone()).map_err(e)?,
            train_end: v["train_end"].as_str().map(str::to_string),
            expand_logit: el.filter(|x| x.len() == 2).map(|x| (x[0], x[1])), // `art.get("expand_logit") or VOL_P`
        })
    }
    pub fn with_expand(mut self, e: Option<(f64, f64)>) -> Self {
        self.expand_logit = e;
        self
    }
}

/// vol4h_from_klines(rows, art, now_ms) → JSON 글자.
pub fn vol4h(rows: &[Value], art: &Svm, now_ms: f64) -> String {
    let rows: Vec<&Value> = rows.iter().filter(|r| (int(&r[6]) as f64) < now_ms).collect();
    if rows.len() < WARMUP + 288 {
        return format!("{{\"ok\": false, \"reason\": {}}}", jstr_ascii(&format!("봉 부족({})", rows.len())));
    }
    let col = |i: usize| rows.iter().map(|r| num(&r[i])).collect::<Vec<f64>>();
    let ts: Vec<i64> = rows.iter().map(|r| int(&r[0])).collect();
    let c = col(4);
    let f = crate::qx4::svm_features(&ts, &col(2), &col(3), &c, &col(7), &col(8));
    let cols: Vec<&Vec<f64>> = art.features.iter().map(|k| &f[k.as_str()]).collect();
    let keep: Vec<usize> = (WARMUP..c.len()).filter(|&i| cols.iter().all(|v| v[i].is_finite())).collect();
    let keep = &keep[keep.len().saturating_sub(30 * 288)..];
    let p: Vec<f64> = keep.iter().map(|&i| {
        let x: Vec<f64> = cols.iter().map(|v| v[i]).collect();
        let mut s = 0.0;
        for (k, m) in art.models.iter().enumerate() {
            let y = predict_proba(m, &x).unwrap();
            s = if k == 0 { y } else { s + y };
        }
        (s / art.models.len() as f64).exp()
    }).collect();
    let mut srt = p.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = srt.len() / 2;
    let refv = if srt.len() % 2 == 1 { srt[m] } else { (srt[m - 1] + srt[m]) / 2.0 };
    let last = *keep.last().unwrap();
    let (pred, rv48) = (*p.last().unwrap(), f["rv48"][last].exp());
    let s4 = pred * 48f64.sqrt();
    let (a0, a1) = art.expand_logit.unwrap_or(VOL_P);
    let px = *c.last().unwrap();
    format!("{{\"ok\": true, \"asof_ms\": {}, \"px\": {}, \"sig4h_pct\": {}, \"sig4h_usd\": {}, \"lvl\": {}, \"x_now\": {}, \"p_expand\": {}, \"ref_days\": {}, \"model_train_end\": {}}}",
            int(&rows.last().unwrap()[6]) + 1, jf(px), jf(py_round(100.0 * s4, 3)), jf(py_round(px * s4, if px >= 10.0 { 2 } else { 4 })),
            jf(py_round(pred / refv, 3)), jf(py_round(pred / rv48, 3)), jf(py_round(1.0 / (1.0 + (-(a0 + a1 * (pred / rv48).ln())).exp()), 3)),
            jf(py_round(p.len() as f64 / 288.0, 1)), art.train_end.as_deref().map_or("null".into(), jstr_ascii))
}

/// 서버 api_vol_forecast 의 HGB 쪽: 5분봉 1000 × 최대 10쪽(endTime 으로 거슬러 · 빈 쪽이면 멈춤).
/// 서버 원문(2026-11-21 코디네이터 확인): 쿼리 symbol·interval·limit=1000·(endTime) · 첫 쪽 endTime 없음 · 빈 쪽이면 멈춤 · rows = got + rows.
pub fn vol4h_rows(fetch: &mut Fetch, base: &str, sym: &str) -> Result<Vec<Value>, String> {
    let mut rows: Vec<Value> = vec![];
    let mut end: Option<i64> = None;
    for _ in 0..10 {
        let got = arr(fetch(&(klines_url(base, sym, "5m", 1000) + &end.map_or(String::new(), |e| format!("&endTime={e}"))))?)?;
        if got.is_empty() {
            break;
        }
        end = Some(int(&got[0][0]) - 1);
        rows = got.into_iter().chain(rows).collect();
    }
    Ok(rows)
}

// ── 7일 코인베이스 프리미엄 · 4~13주 유동성
type Ser = Vec<(i64, f64)>; // (UTC 날 시작 초, 값) 오름차순

fn iso_s(t: i64) -> String {
    iso(t * 1000)
}

/// _sig(z): 마지막 상태(±1 · |v| ≥ 1)·연속 일수·최근 21일.
fn sig(z: &Ser) -> String {
    let z: Vec<&(i64, f64)> = z.iter().filter(|x| !x.1.is_nan()).collect();
    let Some(last) = z.last() else { return "{\"ok\": false, \"reason\": \"no_data\"}".into() };
    let st = |v: f64| if v >= 1.0 { 1 } else if v <= -1.0 { -1 } else { 0 };
    let s = st(last.1);
    let run = z.iter().rev().take_while(|x| st(x.1) == s).count();
    let hist: Vec<String> = z[z.len().saturating_sub(21)..].iter().map(|x| format!("[\"{}\", {}]", day_str(x.0 * 1000), jf(py_round(x.1, 3)))).collect();
    format!("{{\"ok\": true, \"date\": \"{}\", \"v\": {}, \"state\": {s}, \"run\": {run}, \"hist\": [{}]}}", day_str(last.0 * 1000), jf(py_round(last.1, 3)), hist.join(", "))
}

fn prep(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| if v.is_infinite() { NAN } else { *v }).collect()
}

/// cb_premium_z(코인베이스, 바이낸스): 날짜 합집합에서 ln(비) · NaN 뺌 → 7일 평균(최소 5) → 7일 변화 ÷ 365일 표준편차(최소 180).
fn cb_z(cb: &Ser, bn: &Ser) -> Ser {
    let (a, b): (BTreeMap<i64, f64>, BTreeMap<i64, f64>) = (cb.iter().copied().collect(), bn.iter().copied().collect());
    let keys: std::collections::BTreeSet<i64> = a.keys().chain(b.keys()).copied().collect();
    let p: Ser = keys.into_iter().filter_map(|k| Some((k, (a.get(&k)? / b.get(&k)?).ln())).filter(|x| !x.1.is_nan())).collect();
    let v: Vec<f64> = p.iter().map(|x| x.1).collect();
    let p7 = crate::dir4h::roll_mean_pub(&v, 7, 5);
    let d7: Vec<f64> = (0..v.len()).map(|i| if i >= 7 { p7[i] - p7[i - 7] } else { NAN }).collect();
    let sd = roll_std(&prep(&d7), 365, 180);
    p.iter().enumerate().map(|(i, x)| (x.0, d7[i] / sd[i])).collect()
}

/// asof(s, idx, lag): 날짜 d 값을 d+lag 부터 쓰도록 밀고 idx(날 시작 초)에 앞채움.
fn asof(s: &Ser, idx: &[i64], lag: i64) -> Vec<f64> {
    let mut sh: Vec<(i64, f64)> = s.iter().filter(|x| !x.1.is_nan()).map(|&(t, v)| (t + lag * 86_400, v)).collect();
    sh.sort_by_key(|x| x.0);
    let mut j = 0;
    let mut cur = NAN;
    idx.iter().map(|&d| {
        while j < sh.len() && sh[j].0 <= d {
            cur = sh[j].1;
            j += 1;
        }
        cur
    }).collect()
}

fn liq_score(dxy: &[f64], walcl: &[f64], rrp: &[f64], tga: &[f64]) -> Vec<f64> {
    let n = dxy.len();
    let nl: Vec<f64> = (0..n).map(|i| walcl[i] / 1e3 - if rrp[i].is_nan() { 0.0 } else { rrp[i] } - tga[i] / 1e3).collect();
    let sh = |x: &[f64], i: usize| if i >= 91 { x[i - 91] } else { NAN };
    let a: Vec<f64> = (0..n).map(|i| -(dxy[i] / sh(dxy, i)).ln()).collect();
    let b: Vec<f64> = (0..n).map(|i| (nl[i] / sh(&nl, i)).ln()).collect();
    let (sa, sb) = (roll_std(&prep(&a), 730, 365), roll_std(&prep(&b), 730, 365));
    (0..n).map(|i| a[i] / sa[i] + b[i] / sb[i]).collect()
}

fn coinbase(fetch: &mut Fetch, days: i64, now: f64) -> Result<Ser, String> {
    let today = (now / 86_400.0).floor() as i64 * 86_400;
    let (mut rows, mut end, mut days) = (vec![], today, days);
    while days > 0 {
        let n = days.min(300);
        let start = end - n * 86_400;
        let url = format!("https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400&start={}&end={}", iso_s(start), iso_s(end - 1));
        rows.extend(arr(fetch(&url)?)?);
        (end, days) = (start, days - n);
    }
    let m: BTreeMap<i64, f64> = rows.iter().filter(|r| int(&r[0]) < today).map(|r| (int(&r[0]), num(&r[4]))).collect(); // 같은 날이면 뒤 행(딕셔너리)
    Ok(m.into_iter().collect())
}

fn binance(fetch: &mut Fetch, now: f64) -> Result<Ser, String> {
    let raw = arr(fetch("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000")?)?;
    let m: BTreeMap<i64, f64> = raw.iter().filter(|r| (int(&r[6]) as f64) < now * 1000.0).map(|r| (int(&r[0]).div_euclid(1000), num(&r[4]))).collect();
    Ok(m.into_iter().collect())
}

fn fred(fetch: &mut Fetch, sid: &str, key: &str) -> Result<Ser, String> {
    let v = fetch(&format!("https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json&observation_start=2015-01-01"))?;
    let obs = v["observations"].as_array().ok_or("KeyError: 'observations'")?;
    let mut m: BTreeMap<i64, f64> = BTreeMap::new();
    for o in obs {
        let (d, val) = (o["date"].as_str().unwrap_or(""), o["value"].as_str().unwrap_or(""));
        if val == "." || val.is_empty() {
            continue;
        }
        let t = ts_day(d).ok_or_else(|| format!("ValueError: {d}"))?;
        m.insert(t, val.parse::<f64>().map_err(|_| format!("ValueError: could not convert string to float: '{val}'"))?);
    }
    Ok(m.into_iter().collect())
}

fn ts_day(d: &str) -> Option<i64> {
    let n = |a: usize, b: usize| d.get(a..b).and_then(|x| x.parse::<i64>().ok());
    Some(days_from_civil(n(0, 4)?, n(5, 7)?, n(8, 10)?) * 86_400)
}

/// dir_signals_payload(fred_key, now) → JSON 글자. 원천 하나가 실패하면 그 칸만 {"ok": false, "reason"}(원본 예외 문구와 글자가 다를 수 있다 -- 키 없음만 같은 글자).
pub fn dir_signals(fetch: &mut Fetch, fred_key: Option<&str>, now: f64) -> String {
    let cb = coinbase(fetch, 420, now).and_then(|c| binance(fetch, now).map(|b| sig(&cb_z(&c, &b))));
    let cb = cb.unwrap_or_else(|e| format!("{{\"ok\": false, \"reason\": {}}}", jstr_ascii(&e.chars().take(160).collect::<String>())));
    let z91 = (|| -> Result<String, String> {
        let key = fred_key.filter(|k| !k.is_empty()).ok_or("RuntimeError: FRED_API_KEY 없음")?;
        let end = (now / 86_400.0).floor() as i64 * 86_400 - 86_400;
        let idx: Vec<i64> = (ts_day("2016-01-01").unwrap()..=end).step_by(86_400).collect();
        let mut ins = vec![];
        for (sid, lag) in [("DTWEXBGS", 10), ("WALCL", 2), ("RRPONTSYD", 1), ("WTREGEN", 2)] {
            ins.push(asof(&fred(fetch, sid, key)?, &idx, lag));
        }
        let z = liq_score(&ins[0], &ins[1], &ins[2], &ins[3]);
        Ok(sig(&idx.iter().copied().zip(z).collect()))
    })();
    let z91 = z91.unwrap_or_else(|e| {
        let r: String = e.chars().take(160).collect();
        format!("{{\"ok\": false, \"reason\": {}}}", jstr_ascii(&fred_key.filter(|k| !k.is_empty()).map_or(r.clone(), |k| r.replace(k, "***"))))
    });
    format!("{{\"ok\": true, \"k\": 1.0, \"generated_at\": {}, \"cb\": {cb}, \"z91\": {z91}}}", now as i64)
}
