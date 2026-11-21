//! 실시간 위험 레이더 -- 닫힌 1분봉 캐시(+ Deribit DVOL 시간봉) → 다음 15분 σ̂·80/95% 띠·앞 1시간 조용/보통/격함·급변 경고 (2026-11-21 · R5).
//! 원본 scripts/live_risk_radar_20261115.py `grid`·`dvol_series`·`compute`·`main` 의 조립(:221-240)과 연구
//! scripts/research_realtime_risk_radar_20261115.py `features`·`vol_features`·`iv_hourly`·`iv_features`·`wsum`·`design` 과 같은 값을 낸다.
//! 화면 = ⑤ 예측 줄 «15분 폭»·«앞 1시간»·손절/익절 꼬리표(`/api/risk-radar` 30초). 계수 = dashboard/risk_radar_params_20261116.json(B2, 동결).
//! 🔴SOXL(주식 세션 HAR-S · `compute_session`)은 아직 안 옮겼다(rtmodel/REMAINING.md #1). REST 받기(update_cache·update_dvol)도 부르는 쪽 몫.
//!
//! 끝자리: `X @ beta`(numpy 행렬·벡터 곱)는 BLAS 다(개발기 MKL · 서버 미확인) -- 여기는 열 차례 합이라 끝자리가 다를 수 있다(시험이 잰다).
//! rolling std(시간봉 168)·median(720)은 sr_lean·dir4h 와 같은 pandas 식, rolling rank(pct) = (작은 수 + (같은 수+1)/2) / 개수(tools 실측).
use crate::box_hold::py_round;
use crate::sr_lean::{jf, roll_std};
use serde::Deserialize;
use std::collections::HashMap;

const MIN: i64 = 60_000;
pub const STALE_DATA_S: f64 = 180.0;
const WIN: [(&str, usize); 7] = [("lrv1", 1), ("lrv5", 5), ("lrv15", 15), ("lrv60", 60), ("lrv240", 240), ("lrv1d", 1440), ("lrv7d", 10080)];
const VWIN: [(&str, usize); 5] = [("lqv1", 1), ("lqv5", 5), ("lqv15", 15), ("lqv60", 60), ("lqv1d", 1440)];
const IV_STALE_S: i64 = 7200;
const NAN: f64 = f64::NAN;

#[derive(Deserialize)]
pub struct Params {
    pub cols: Vec<String>,
    pub quiet: f64,
    pub loud: f64,
    pub alert_k: f64,
    pub alert_k5: f64,
    pub alert_dedupe_min: i64,
    pub coins: HashMap<String, CoinP>,
}

#[derive(Deserialize)]
pub struct CoinP {
    /// "session" = 주식 세션 HAR-S(SOXL · risk_radar_params_soxl_*.json) -- 그 파일에만 있다
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub touch: Option<Vec<f64>>,
    pub beta15: Vec<f64>,
    pub c15: f64,
    pub beta60: Vec<f64>,
    pub c60: f64,
    pub q80: f64,
    pub q95: f64,
    pub cb0: f64,
    pub vclip: Option<HashMap<String, f64>>,
    pub cols60_iv: Option<Vec<String>>,
    pub beta60_iv: Option<Vec<f64>>,
    pub c60_iv: Option<f64>,
}

/// 워커 1분봉 캐시(t = 봉 시작 ms · 오름차순 · 겹침 없음 -- update_cache 가 보장).
pub struct Bars {
    pub t: Vec<i64>,
    pub h: Vec<f64>,
    pub l: Vec<f64>,
    pub c: Vec<f64>,
    pub v: Vec<f64>,
}

/// DVOL 캐시(봉 시작 ms · 종가 · 받은 시각 ms).
pub struct Dvol {
    pub ts: Vec<i64>,
    pub close: Vec<f64>,
    pub got: Vec<i64>,
}

/// 파이썬 json.dumps 와 같은 글자로 쓰는 값(키 순서 보존 · 실수 = repr · ensure_ascii=False).
#[derive(Clone, Debug)]
pub enum J {
    Null,
    B(bool),
    I(i64),
    F(f64),
    S(String),
    O(Vec<(&'static str, J)>),
    A(Vec<J>),
}

impl J {
    pub fn dump(&self) -> String {
        match self {
            J::Null => "null".into(),
            J::B(b) => b.to_string(),
            J::I(i) => i.to_string(),
            J::F(f) => {
                assert!(f.is_finite(), "allow_nan=False"); // 원본은 NaN 을 None 으로 바꿔 둔다
                jf(*f)
            }
            J::S(s) => {
                let mut o = String::from("\"");
                for ch in s.chars() {
                    match ch {
                        '"' => o += "\\\"",
                        '\\' => o += "\\\\",
                        '\n' => o += "\\n",
                        '\r' => o += "\\r",
                        '\t' => o += "\\t",
                        '\u{8}' => o += "\\b",
                        '\u{c}' => o += "\\f",
                        c if (c as u32) < 0x20 => o += &format!("\\u{:04x}", c as u32),
                        c => o.push(c),
                    }
                }
                o + "\""
            }
            J::O(kv) => format!("{{{}}}", kv.iter().map(|(k, v)| format!("\"{k}\": {}", v.dump())).collect::<Vec<_>>().join(", ")),
            J::A(v) => format!("[{}]", v.iter().map(J::dump).collect::<Vec<_>>().join(", ")),
        }
    }
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::O(kv) => kv.iter().find(|(a, _)| *a == k).map(|x| &x.1),
            _ => None,
        }
    }
}

fn fin(x: f64) -> J {
    if x.is_finite() { J::F(x) } else { J::Null }
}

/// 초 → "%Y-%m-%dT%H:%M:%SZ"(UTC).
pub fn iso(ms: i64) -> String {
    let (d, s) = (ms.div_euclid(86_400_000), ms.rem_euclid(86_400_000) / 1000);
    // civil_from_days(Howard Hinnant)
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (mon <= 2) as i64;
    format!("{y:04}-{mon:02}-{day:02}T{:02}:{:02}:{:02}Z", s / 3600, s / 60 % 60, s % 60)
}

/// wsum(x, w): x[t−w+1..t] 합(NaN = 0 · ±inf = ±최대)과 NaN 개수 -- 누적합(차례) 차이.
fn wsum(x: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let (mut cs, mut cn) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    for i in 0..n {
        let v = if x[i].is_nan() { 0.0 } else if x[i].is_infinite() { f64::MAX.copysign(x[i]) } else { x[i] };
        cs[i + 1] = if i == 0 { v } else { cs[i] + v };
        let k = x[i].is_nan() as u8 as f64;
        cn[i + 1] = if i == 0 { k } else { cn[i] + k };
    }
    let (mut s, mut k) = (vec![NAN; n], vec![NAN; n]);
    for i in w.saturating_sub(1)..n {
        if i + 1 >= w {
            s[i] = cs[i + 1] - cs[i + 1 - w];
            k[i] = cn[i + 1] - cn[i + 1 - w];
        }
    }
    (s, k)
}

fn logwin(s: &[f64], k: &[f64], w: usize, floor: f64, half: bool) -> Vec<f64> {
    let wf = w as f64;
    (0..s.len()).map(|i| {
        let ok = if w <= 240 { k[i] == 0.0 } else { k[i] <= 0.05 * wf };
        if !ok {
            return NAN;
        }
        let den = if wf - k[i] > 1.0 { wf - k[i] } else { 1.0 }; // np.maximum(w − k, 1)
        let a = s[i] * wf / den;
        let m = if a.is_nan() || floor.is_nan() { NAN } else { a.max(floor) };
        if half { 0.5 * m.ln() } else { m.ln() }
    }).collect()
}

/// 원본 compute() 가 내는 코인 한 칸(+ main 이 붙이는 data_age_s · 지연이면 valid False·error).
pub fn coin_out(b: &Bars, p: &CoinP, pp: &Params, dvol: Option<&Dvol>, now_ms: i64) -> J {
    let (m0, m1) = (b.t[0].div_euclid(MIN), b.t[b.t.len() - 1].div_euclid(MIN));
    let n = (m1 - m0 + 1) as usize;
    let (mut cl, mut qv) = (vec![NAN; n], vec![NAN; n]);
    for (j, &t) in b.t.iter().enumerate() {
        let i = (t.div_euclid(MIN) - m0) as usize;
        cl[i] = b.c[j];
        qv[i] = b.v[j] * b.c[j]; // 거래대금 = v×c(연구 load_qv)
    }
    let mins: Vec<i64> = (0..n as i64).map(|i| i + m0).collect();
    // features
    let lc: Vec<f64> = cl.iter().map(|x| x.ln()).collect();
    let r: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { lc[i] - lc[i - 1] }).collect();
    let r2: Vec<f64> = r.iter().map(|x| x * x).collect();
    let mut f: HashMap<&str, Vec<f64>> = HashMap::new();
    for (nm, w) in WIN {
        let (s, k) = wsum(&r2, w);
        f.insert(nm, logwin(&s, &k, w, 1e-10 * w as f64, true));
    }
    let cell: Vec<usize> = mins.iter().map(|m| {
        let ms = (m + 1) * MIN; // 표적 창 첫 분
        let (dow, hour) = ((ms.div_euclid(86_400_000) + 3).rem_euclid(7), ms.rem_euclid(86_400_000) / 3_600_000);
        (hour + 24 * (dow >= 5) as i64) as usize
    }).collect();
    // B0: 직전 완결 168개 1시간봉 로그수익 표준편차
    let hidx: Vec<usize> = (0..n).filter(|&i| (mins[i] + 1) % 60 == 0).collect();
    let hc: Vec<f64> = hidx.iter().map(|&i| lc[i]).collect();
    let hr: Vec<f64> = (0..hc.len()).map(|j| if j == 0 { NAN } else { hc[j] - hc[j - 1] }).collect();
    let sd = roll_std(&hr, 168, 150);
    let last_h = |i: usize| hidx.partition_point(|&h| h <= i) as i64 - 1;
    let b0sd: Vec<f64> = (0..n).map(|i| { let l = last_h(i); if l >= 0 { sd[l as usize] } else { NAN } }).collect();
    for (nm, w) in VWIN {
        let (s, k) = wsum(&qv, w);
        let mut x = logwin(&s, &k, w, 1.0, false);
        if let Some(vc) = &p.vclip {
            let c = vc[nm];
            for v in x.iter_mut() {
                if v.is_finite() {
                    *v = v.max(c);
                } else {
                    *v = NAN;
                }
            }
        }
        f.insert(nm, x);
    }
    // X @ beta: [1, 열들, 시각 칸 원핫(칸 0 뺌)]
    let lin = |cv: &[&[f64]], beta: &[f64], i: usize| -> f64 {
        let mut z = beta[0];
        for (j, c) in cv.iter().enumerate() {
            z += c[i] * beta[1 + j];
        }
        // ponytail: 원핫은 0·1 이라 해당 칸 하나만 더한다(numpy BLAS 는 0×β 도 더한다 -- 값은 같고 합 순서만 다르다)
        if cell[i] > 0 {
            z += beta[cv.len() + cell[i]];
        }
        z
    };
    let cols_of = |f: &HashMap<&str, Vec<f64>>, cols: &[String]| -> Vec<Vec<f64>> { cols.iter().map(|c| f[c.as_str()].clone()).collect() };
    let cv = cols_of(&f, &pp.cols);
    let cv: Vec<&[f64]> = cv.iter().map(|v| v.as_slice()).collect();
    let f15: Vec<f64> = (0..n).map(|i| p.c15 * (2.0 * lin(&cv, &p.beta15, i)).exp()).collect();
    let mut f60: Vec<f64> = (0..n).map(|i| p.c60 * (2.0 * lin(&cv, &p.beta60, i)).exp()).collect();
    let mut f60i = vec![NAN; n];
    if let (Some(dv), Some(cols), Some(beta), Some(c)) = (dvol, &p.cols60_iv, &p.beta60_iv, p.c60_iv) {
        // dvol_series: 끝나기 전에 받은 봉은 뺀다 → iv_hourly: 양수만 · 같은 시각은 뒤 것 · 정렬 · 720시간 분위(240 이상)
        let mut rows: Vec<(i64, f64)> = vec![];
        for j in 0..dv.ts.len() {
            if dv.got[j] >= dv.ts[j] + 3_600_000 && dv.close[j] > 0.0 {
                let t = dv.ts[j].div_euclid(1000);
                if let Some(k) = rows.iter().position(|r| r.0 == t) {
                    rows.remove(k);
                }
                rows.push((t, dv.close[j]));
            }
        }
        rows.sort_by_key(|r| r.0);
        if !rows.is_empty() {
            let x: Vec<f64> = rows.iter().map(|r| r.1).collect();
            let pct: Vec<f64> = (0..x.len()).map(|i| {
                let w = &x[i.saturating_sub(719)..=i];
                if w.len() < 240 {
                    return NAN;
                }
                let (less, eq) = (w.iter().filter(|&&v| v < x[i]).count(), w.iter().filter(|&&v| v == x[i]).count());
                (less as f64 + (eq + 1) as f64 / 2.0) / w.len() as f64 * 100.0
            }).collect();
            let tk: Vec<i64> = rows.iter().map(|r| r.0 + 3600).collect();
            let (mut il, mut ip) = (vec![NAN; n], vec![NAN; n]);
            for i in 0..n {
                let now = (mins[i] + 1) * 60;
                let j = tk.partition_point(|&t| t <= now) as i64 - 1;
                if j >= 0 && now - tk[j as usize] <= IV_STALE_S {
                    il[i] = x[j as usize].ln();
                    ip[i] = pct[j as usize];
                }
            }
            f.insert("I_l", il);
            f.insert("I_p", ip);
            let ci = cols_of(&f, cols);
            let ci: Vec<&[f64]> = ci.iter().map(|v| v.as_slice()).collect();
            for i in 0..n {
                f60i[i] = c * (2.0 * lin(&ci, beta, i)).exp();
                if f60i[i].is_finite() {
                    f60[i] = f60i[i];
                }
            }
        }
    }
    // usual60: 직전 완결 60분 RV 의 720개 굴림 중앙값(600 이상)
    let (s60, k60) = wsum(&r2, 60);
    let pidx: Vec<usize> = (0..n).filter(|&i| k60[i] == 0.0 && (mins[i] + 1) % 60 == 0 && s60[i].sqrt().is_finite()).collect();
    let past: Vec<f64> = pidx.iter().map(|&i| s60[i].sqrt()).collect();
    let uh: Vec<f64> = (0..past.len()).map(|j| {
        let mut w: Vec<f64> = past[j.saturating_sub(719)..=j].to_vec();
        if w.len() < 600 {
            return NAN;
        }
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = w.len();
        if m % 2 == 1 { w[m / 2] } else { (w[m / 2] + w[m / 2 - 1]) / 2.0 }
    }).collect();
    let usual = |i: usize| -> f64 {
        let l = pidx.partition_point(|&p| p <= i) as i64 - 1;
        if l >= 0 && !uh.is_empty() { uh[l as usize] } else { NAN }
    };
    // 급변: 1분 |r| ≥ k·σ̂(직전 분 예측) 또는 5분 RV ≥ k5·σ̂(5분 전 예측)·√(5/15) · 30분 중복 제거(캐시 전체를 앞에서부터)
    let (mut last_alert, mut prev) = (None::<i64>, -1_000_000_000i64);
    let lrv5 = &f["lrv5"];
    for i in 0..n {
        let sig1 = if i == 0 { NAN } else { f15[i - 1].sqrt() / 15f64.sqrt() };
        let f5 = if i < 5 { NAN } else { f15[i - 5] };
        let raw = (r[i].abs() >= pp.alert_k * sig1 || lrv5[i].exp() >= pp.alert_k5 * f5.sqrt() * (5.0f64 / 15.0).sqrt()) && f15[i].is_finite();
        if raw && i as i64 - prev > pp.alert_dedupe_min {
            (last_alert, prev) = (Some(i as i64), i as i64);
        }
    }
    let i = n - 1;
    let s15 = f15[i].sqrt();
    let s60 = f60[i].sqrt();
    let us = usual(i);
    let ratio = if s60.is_finite() && us.is_finite() && us > 0.0 { s60 / us } else { NAN };
    let state = if !ratio.is_finite() { J::Null } else { J::S((if ratio <= pp.quiet { "조용" } else if ratio >= pp.loud { "격함" } else { "보통" }).into()) };
    let ago = last_alert.map(|a| i as i64 - a);
    let pc = |x: f64, k: f64| if x.is_finite() { J::F(x * k * 100.0) } else { J::Null };
    let b0 = b0sd[i] * (15.0 / 60.0 * p.cb0).sqrt();
    let age = py_round((now_ms - b.t[b.t.len() - 1] - MIN) as f64 / 1000.0, 1);
    let mut kv: Vec<(&'static str, J)> = vec![
        ("ts", J::S(iso((m0 + i as i64 + 1) * MIN))),
        ("price", fin(cl[i])),
        ("sigma15_pct", pc(s15, 1.0)),
        ("band80_pct", pc(s15, p.q80)),
        ("band95_pct", pc(s15, p.q95)),
        ("baseline7d_sigma15_pct", pc(b0, 1.0)),
        ("sigma60_pct", pc(s60, 1.0)),
        ("usual60_pct", pc(us, 1.0)),
        ("ratio60", fin(ratio)),
        ("state60", state),
        ("alert", J::O(vec![("active", J::B(ago.is_some_and(|a| a < 15))), ("minutes_ago", ago.map_or(J::Null, J::I))])),
        ("valid", J::B(s15.is_finite() && ratio.is_finite())),
        ("iv60", J::B(f60i[i].is_finite())),
        ("data_age_s", J::F(age)),
    ];
    if age > STALE_DATA_S {
        // REST 가 새 봉을 안 줘도 옛 값이 지금 값처럼 보이던 것(검증 11-16 ②) -- 화면은 24h 로 떨어진다
        kv.iter_mut().find(|x| x.0 == "valid").unwrap().1 = J::B(false);
        kv.push(("error", J::S(format!("데이터 지연 {}초", format!("{age:.0}")))));
    }
    J::O(kv)
}

/// main() 의 조립: latest.json 글자 + forward.jsonl 줄(오류 없는 코인만). coins = 원본 계수 파일의 코인 순서.
pub fn assemble(now_ms: i64, params_name: &str, coins: Vec<(&'static str, J)>) -> (String, Vec<String>) {
    let mut log = vec![];
    for (c, o) in &coins {
        if o.get("error").is_none() {
            let mut kv: Vec<(&'static str, J)> = vec![("coin", J::S((*c).into()))];
            for k in ["ts", "price", "sigma15_pct", "baseline7d_sigma15_pct", "sigma60_pct", "usual60_pct", "state60", "data_age_s"] {
                kv.push((k, o.get(k).unwrap().clone()));
            }
            kv.push(("alert", o.get("alert").unwrap().get("active").unwrap().clone()));
            log.push(J::O(kv).dump());
        }
    }
    let latest = J::O(vec![("updated", J::S(iso(now_ms.div_euclid(1000) * 1000))), ("params", J::S(params_name.into())), ("coins", J::O(coins))]).dump();
    (latest, log)
}

// ── 주식 세션 HAR-S(SOXL) -- 원본 compute_session(:183-197) + 연구 research_soxl_session_vol_20261112 frame·clock·har_x ─────────────────
/// NYSE 휴장일(연구 파일 HOLIDAYS 그대로 · 2027 까지). ponytail: 손으로 적은 표 -- 해가 바뀌면 원본과 같이 늘린다.
const HOLIDAYS: [&str; 16] = ["2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"];
pub const SESS: [&str; 6] = ["정규장", "프리장", "애프터", "야간", "주말", "휴장"];
const SWIN: [(&str, usize); 4] = [("lrv15", 15), ("lrv60", 60), ("lrv1d", 1440), ("lrv7d", 10080)];

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// 미국 동부 시각의 UTC 오프셋(초): 2007 이후 규칙 -- 3월 둘째 일요일 02:00 ~ 11월 첫 일요일 02:00(현지) 서머타임.
fn ny_offset(utc_s: i64) -> i64 {
    let y = iso(utc_s * 1000)[..4].parse::<i64>().unwrap();
    let sunday = |m: i64, nth: i64| {
        let d1 = days_from_civil(y, m, 1);
        let dow = (d1 + 4).rem_euclid(7); // 일=0
        d1 + (7 - dow) % 7 + 7 * (nth - 1)
    };
    let start = sunday(3, 2) * 86400 + 7 * 3600; // 02:00 EST = 07:00 UTC
    let end = sunday(11, 1) * 86400 + 6 * 3600; // 02:00 EDT = 06:00 UTC
    if utc_s >= start && utc_s < end { -4 * 3600 } else { -5 * 3600 }
}

/// clock(τ, H) 한 칸: (세션, open_in, first60, last30).
#[doc(hidden)]
pub fn clock(tau: i64, h: f64) -> (usize, f64, f64, f64) {
    let loc = tau + ny_offset(tau);
    let (day, sod) = (loc.div_euclid(86400), loc.rem_euclid(86400));
    let m = sod / 60;
    let w = (day + 3).rem_euclid(7); // 월=0
    let date = &iso(day * 86_400_000)[..10];
    let hol = HOLIDAYS.contains(&date);
    let sess = if w == 5 || (w == 4 && m >= 1200) || (w == 6 && m < 1200) {
        4
    } else if hol && (240..1200).contains(&m) {
        5
    } else if (570..960).contains(&m) {
        0
    } else if (240..570).contains(&m) {
        1
    } else if (960..1200).contains(&m) {
        2
    } else {
        3
    };
    // 개장(동부 09:30)·마감(+390분) UTC 초: 현지 날짜 −5 ~ +5일의 평일·비휴장일
    let mut opens = vec![];
    for dd in day - 6..=day + 6 {
        let (wd, ds) = ((dd + 3).rem_euclid(7), iso(dd * 86_400_000));
        if wd < 5 && !HOLIDAYS.contains(&&ds[..10]) {
            let local = dd * 86400 + 570 * 60;
            let off = ny_offset(local + 5 * 3600); // 그 날 09:30 의 오프셋(전환은 02:00 이라 같은 날 안에서 안 바뀐다)
            opens.push(local - off);
        }
    }
    let clip = |x: f64| x.clamp(0.0, 1440.0);
    let i = opens.partition_point(|&o| o <= tau) as i64 - 1;
    let since_open = clip((tau - opens[i.max(0) as usize]) as f64 / 60.0);
    let nx = opens.partition_point(|&o| o < tau).min(opens.len() - 1);
    let to_open = clip((opens[nx] - tau) as f64 / 60.0);
    let closes: Vec<i64> = opens.iter().map(|o| o + 390 * 60).collect();
    let nc = closes.partition_point(|&c| c < tau).min(closes.len() - 1);
    let to_close = clip((closes[nc] - tau) as f64 / 60.0);
    (sess, (to_open < h) as u8 as f64, (sess == 0 && since_open < 60.0) as u8 as f64, (sess == 0 && to_close <= 30.0) as u8 as f64)
}

/// compute_session: 마지막 닫힌 봉 다음 분 τ 의 15분·1시간 σ̂(세션 HAR-S) + 세션 이름 + 꼬리표 표. «앞 1시간»·급변 경고는 없다(코인에서만 검정).
pub fn session_out(b: &Bars, p: &CoinP, now_ms: i64) -> J {
    // df.tail(9 × 1440) + 형성 중 봉 자리(빈 행) → 분 격자
    let a = b.t.len().saturating_sub(9 * 1440);
    let (t, c) = (&b.t[a..], &b.c[a..]);
    let (m0, m1) = (t[0].div_euclid(MIN), t[t.len() - 1].div_euclid(MIN) + 1);
    let n = (m1 - m0 + 1) as usize;
    let mut cl = vec![NAN; n];
    for (j, &tt) in t.iter().enumerate() {
        cl[(tt.div_euclid(MIN) - m0) as usize] = c[j];
    }
    let lc: Vec<f64> = cl.iter().map(|x| x.ln()).collect();
    let r2: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { let r = lc[i] - lc[i - 1]; r * r }).collect();
    let j = n - 1; // τ = 분 j 의 시작 · 피쳐 = 분 j−1
    let mut f: HashMap<&str, f64> = HashMap::new();
    for (nm, w) in SWIN {
        let mean = crate::dir4h::roll_mean_pub(&r2, w, (w as f64 * 0.95) as usize);
        let v = mean[j - 1].sqrt();
        f.insert(nm, if v.is_nan() { NAN } else { v.max(1e-6) }.ln());
    }
    let tau = (j as i64 + m0) * 60;
    let x = |h: f64| -> (Vec<f64>, usize) {
        let (sess, oi, f60, l30) = clock(tau, h);
        let mut v = vec![1.0, f["lrv15"], f["lrv60"], f["lrv1d"], f["lrv7d"]];
        for k in 1..6 {
            v.push((sess == k) as u8 as f64);
        }
        v.extend([oi, f60, l30]);
        (v, sess)
    };
    // b0: 시간봉 종가(그 시의 59분 · NaN 은 뺌)의 로그수익 168개 std × √(H/60) → 분 격자로 앞 값 채움
    let hidx: Vec<usize> = (0..n).filter(|&i| (i as i64 + m0 + 1) % 60 == 0 && !cl[i].is_nan()).collect();
    let hl: Vec<f64> = hidx.iter().map(|&i| lc[i]).collect();
    let hr: Vec<f64> = (0..hl.len()).map(|k| if k == 0 { NAN } else { hl[k] - hl[k - 1] }).collect();
    let sd = roll_std(&hr, 168, 150);
    let b0 = |h: f64| -> f64 {
        let mut v = NAN;
        for (k, &i) in hidx.iter().enumerate() {
            if i > j - 1 {
                break;
            }
            let y = sd[k] * (h / 60.0).sqrt();
            if !y.is_nan() {
                v = y; // reindex 뒤 ffill: NaN 은 앞 값으로
            }
        }
        v
    };
    let dot = |x: &[f64], beta: &[f64]| x.iter().zip(beta).fold(0.0, |s, (a, b)| s + a * b); // ponytail: numpy 는 BLAS(1행 gemv) -- 끝자리
    let (x15, sess) = x(15.0);
    let (x60, _) = x(60.0);
    let v15 = p.c15 * (2.0 * dot(&x15, &p.beta15)).exp();
    let v60 = p.c60 * (2.0 * dot(&x60, &p.beta60)).exp();
    let pc = |x: f64, m: f64| if x.is_finite() { J::F(x * m * 100.0) } else { J::Null };
    let age = py_round((now_ms - b.t[b.t.len() - 1] - MIN) as f64 / 1000.0, 1);
    let mut kv: Vec<(&'static str, J)> = vec![
        ("ts", J::S(iso(tau * 1000))),
        ("price", fin(b.c[b.c.len() - 1])),
        ("sigma15_pct", pc(v15.sqrt(), 1.0)),
        ("band80_pct", pc(v15.sqrt(), p.q80)),
        ("band95_pct", pc(v15.sqrt(), p.q95)),
        ("baseline7d_sigma15_pct", pc(b0(15.0) * p.cb0.sqrt(), 1.0)),
        ("sigma60_pct", pc(v60.sqrt(), 1.0)),
        ("usual60_pct", J::Null),
        ("ratio60", J::Null),
        ("state60", J::Null),
        ("session", J::S(SESS[sess].into())),
        ("alert", J::O(vec![("active", J::B(false)), ("minutes_ago", J::Null)])),
        ("valid", J::B(v15.is_finite() && v60.is_finite())),
        ("touch", J::A(p.touch.clone().unwrap_or_default().into_iter().map(J::F).collect())),
        ("data_age_s", J::F(age)),
    ];
    if age > STALE_DATA_S {
        kv.iter_mut().find(|x| x.0 == "valid").unwrap().1 = J::B(false);
        kv.push(("error", J::S(format!("데이터 지연 {}초", format!("{age:.0}")))));
    }
    J::O(kv)
}
