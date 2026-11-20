//! 박스 유지 확률 -- 닫힌 1분봉 → 피쳐 → HGB → (1h Platt90 · ETH 4h QX4I) → 화면 payload (2026-11-20 · R5 둘째 조각).
//! 원본 dashboard/box_hold.py(2026-10-16 «HGB로 배포» · 10-20 XRP · 10-31 BTC·SOL · 11-18 SOXL)와 같은 값을 낸다.
//! 박스 = 지금가 ± 0.5×(지난 W분 고저폭) · 값 = 다음 W분 안 안 벗어날 확률. 🔴방향 정보 아님(어느 쪽으로 깨질지는 모른다).
//!
//! 끝자리까지 맞추려고 따른 계산(numpy 2.4.3 · pandas 2.3.3, 개발기에서 실측 대조):
//! - `np.mean`·`.sum()` = numpy 짝 합(pairwise_sum: 8칸 미만 차례 합 · 128칸 이하 8갈래 펼침 · 넘으면 8의 배수로 반 나눔) / n.
//! - `rolling(W).max()/min()` = 창 안에 NaN 이 하나라도 있으면 NaN(min_periods = W) · 아니면 최대/최소.
//! - `np.nanmedian` = NaN 뺀 값의 중앙(짝수면 가운데 둘 합 / 2) · 전부 NaN 이면 NaN.
//! - `np.log` 는 numpy SIMD 구현이라 libm(여기 `ln`)과 가끔 1ulp 다르다(개발기 무작위 20만 개 중 24개) -- 시험이 그 폭을 잰다.
//!   `np.sin`·`np.cos` 는 libm 과 비트 같음(같은 실측).
//! 화면(app.js boxNow·boxState·미래 상자)·LLM 요약(scripts/live_llm_brief_20261017.py)이 읽는 필드만 낸다 -- 뺀 필드는 보고서.
use crate::{platt90, predict_proba, BoxBundle, Qx4iIso};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::Path;

/// 닫힌 1분봉 한 개(오름차순). t_ms = 봉 시작(UTC ms).
#[derive(Clone, Copy)]
pub struct Bar {
    pub t_ms: i64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub v: f64,
    pub tb: f64,
}

pub const NEED: usize = 1499; // 1h 박스 24h 창폭 중앙값 = 1440 + 59 · σ24 = 1441
pub const WIN: [(&str, usize); 3] = [("30m", 30), ("1h", 60), ("4h", 240)];
const MAJOR_IDS: [i64; 6] = [10, 50, 53, 54, 46, 9]; // FRED release_id: CPI·고용·GDP·PCE·PPI·소매판매 (+ FOMC 금리결정)
const CLAIMS_ID: i64 = 180; // 실업수당(주간)
/// 1h Platt90 이 있는 코인(2026-10-26 ETH · 10-31 XRP·BTC 통과 · SOL 은 2025 창 ΔBrier +.000003 불통과 → 표에 없음 = 보정 없음).
pub const PLATT90_ASSETS: [&str; 3] = ["eth", "xrp", "btc"];
const VOL_STALE_MS: f64 = 15.0 * 60_000.0;

// ── numpy·pandas 와 같은 순서의 기본 연산
fn npsum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = 0.0;
        for v in a {
            s += v;
        }
        return s;
    }
    if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let m = n - n % 8;
        for i in (8..m).step_by(8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[m..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    npsum(&a[..n2]) + npsum(&a[n2..])
}
fn npmean(a: &[f64]) -> f64 {
    npsum(a) / a.len() as f64
}
fn tail(a: &[f64], w: usize) -> &[f64] {
    &a[a.len() - w..]
}
/// np.max / np.min: NaN 이 있으면 NaN.
fn npext(a: &[f64], max: bool) -> f64 {
    let mut m = a[0];
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if (max && v > m) || (!max && v < m) {
            m = v;
        }
    }
    m
}
/// pandas rolling(w).max()/min() 전체 열.
fn rolling_ext(x: &[f64], w: usize, max: bool) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    let mut dq = std::collections::VecDeque::new();
    let mut last_nan: Option<usize> = None;
    for i in 0..x.len() {
        if x[i].is_nan() {
            last_nan = Some(i);
        } else {
            while let Some(&j) = dq.back() {
                if (max && x[j] <= x[i]) || (!max && x[j] >= x[i]) {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(i);
        }
        while let Some(&j) = dq.front() {
            if j + w <= i {
                dq.pop_front();
            } else {
                break;
            }
        }
        if i + 1 >= w && last_nan.map_or(true, |k| k + w <= i) {
            out[i] = x[dq[0]];
        }
    }
    out
}
fn nanmedian(a: &[f64]) -> f64 {
    let mut v: Vec<f64> = a.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let k = v.len() / 2;
    if v.len() % 2 == 1 {
        v[k]
    } else {
        (v[k - 1] + v[k]) / 2.0
    }
}
/// 파이썬 round(x, nd): 이진값 그대로를 십진 nd 자리로(정확히 반이면 짝수 쪽) -- Rust 서식도 같은 규칙(시험이 대조).
pub fn py_round(x: f64, nd: usize) -> f64 {
    format!("{x:.nd$}").parse().unwrap()
}

// ── 시각
/// (요일 월=0, 시, 분) -- 1970-01-01 = 목요일(3).
fn dhm(ms: i64) -> (i64, i64, i64) {
    let d = ms.div_euclid(86_400_000);
    let s = ms.rem_euclid(86_400_000) / 1000;
    ((d + 3).rem_euclid(7), s / 3600, s / 60 % 60)
}
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}
/// ISO 시각 → (UTC ns, 시간대 표기 있음). 받는 꼴 = `YYYY-MM-DD[(T| )HH:MM[:SS[.f]]][Z|±HH:MM|±HHMM]`.
// ponytail: pd.Timestamp 는 이 밖의 꼴도 읽는다 -- 일정 워커·Platt90 적합 스크립트는 isoformat() 만 쓴다(소스 확인). 다른 꼴이 오면 «못 읽음 = 건너뜀»이라 그때 꼴을 더한다.
pub fn parse_iso_ns(s: &str) -> Option<(i64, bool)> {
    let b = s.as_bytes();
    let num = |i: usize, n: usize| -> Option<i64> {
        let t = b.get(i..i + n)?;
        t.iter().all(u8::is_ascii_digit).then(|| t.iter().fold(0i64, |a, c| a * 10 + (c - b'0') as i64))
    };
    let (y, mo, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    if b.get(4) != Some(&b'-') || b.get(7) != Some(&b'-') || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let mut i = 10;
    let (mut hh, mut mm, mut ss, mut ns) = (0, 0, 0, 0i64);
    if matches!(b.get(i), Some(b'T') | Some(b' ')) {
        hh = num(i + 1, 2)?;
        if b.get(i + 3) != Some(&b':') {
            return None;
        }
        mm = num(i + 4, 2)?;
        i += 6;
        if b.get(i) == Some(&b':') {
            ss = num(i + 1, 2)?;
            i += 3;
            if b.get(i) == Some(&b'.') {
                let st = i + 1;
                i = st;
                while b.get(i).is_some_and(u8::is_ascii_digit) {
                    i += 1;
                }
                if i == st {
                    return None;
                }
                let f = &s[st..i.min(st + 9)];
                ns = f.parse::<i64>().ok()? * 10i64.pow(9 - f.len() as u32);
            }
        }
        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }
    }
    let mut off = 0;
    let tz = match b.get(i) {
        None => false,
        Some(b'Z') if i + 1 == b.len() => true,
        Some(&c) if c == b'+' || c == b'-' => {
            let oh = num(i + 1, 2)?;
            let om = if b.get(i + 3) == Some(&b':') { num(i + 4, 2)? } else { num(i + 3, 2)? };
            let end = if b.get(i + 3) == Some(&b':') { i + 6 } else { i + 5 };
            if end != b.len() {
                return None;
            }
            off = (oh * 60 + om) * if c == b'+' { 1 } else { -1 };
            true
        }
        _ => return None,
    };
    let secs = days_from_civil(y, mo, d) * 86_400 + hh * 3600 + mm * 60 + ss - off * 60;
    Some((secs * 1_000_000_000 + ns, tz))
}

/// box_hold.event_times: 거시 일정 → (주요 발표 ms, 실업수당 ms) 정렬.
fn event_times(events: &[Value]) -> (Vec<i64>, Vec<i64>) {
    let (mut major, mut claims) = (vec![], vec![]);
    for e in events {
        let Some(t) = e.get("time_utc").and_then(Value::as_str).and_then(parse_iso_ns) else { continue };
        let t = t.0.div_euclid(1_000_000);
        let s = |k: &str| e.get(k).and_then(Value::as_str).unwrap_or("");
        let det = s("detail");
        let mut rid = None; // re.search(r"release_id=(\d+)") -- 숫자가 붙은 첫 자리
        for (p, _) in det.match_indices("release_id=") {
            let d: String = det[p + 11..].chars().take_while(char::is_ascii_digit).collect();
            if !d.is_empty() {
                rid = d.parse::<i64>().ok();
                break;
            }
        }
        // 10-19 리뷰: 일정의 category fomc 에는 기자회견(결정 30분 뒤)·의사록도 있다 -- 학습 major 는 금리결정(14:00 ET)만
        if (s("category") == "fomc" && s("title_ko").contains("금리결정")) || rid.is_some_and(|r| MAJOR_IDS.contains(&r)) {
            major.push(t);
        } else if rid == Some(CLAIMS_ID) {
            claims.push(t);
        }
    }
    major.sort();
    claims.sort();
    (major, claims)
}
fn next_min(tc: i64, ev: &[i64]) -> f64 {
    let j = ev.partition_point(|&x| x <= tc);
    if j < ev.len() { (ev[j] - tc) as f64 / 60000.0 } else { 1e12 }
}
fn prev_min(tc: i64, ev: &[i64]) -> f64 {
    let j = ev.partition_point(|&x| x <= tc);
    if j > 0 { (tc - ev[j - 1]) as f64 / 60000.0 } else { 1e12 }
}

pub type Feat = HashMap<&'static str, f64>;

/// box_hold.features: 마지막 닫힌 봉 마감 시점의 척도별 피쳐(+ `_rng` = 창 고저폭). wins = 계산할 창(분).
/// Err = 파이썬이 예외를 내는 입력(지난 24h 수익이 전부 0 → log_vr 의 1e-6 / 0.0 이 ZeroDivisionError · 서버는 payload 전체 ok False).
pub fn features(k: &[Bar], events: &[Value], prof: &[f64], wins: &[usize]) -> Result<Vec<Feat>, String> {
    let col = |f: fn(&Bar) -> f64| k.iter().map(f).collect::<Vec<f64>>();
    let (h, l, c, v, tb) = (col(|b| b.h), col(|b| b.l), col(|b| b.c), col(|b| b.v), col(|b| b.tb));
    let lc: Vec<f64> = c.iter().map(|x| x.ln()).collect();
    let r: Vec<f64> = lc.windows(2).map(|w| w[1] - w[0]).collect();
    let r2: Vec<f64> = r.iter().map(|x| x * x).collect();
    let s24 = npmean(tail(&r2, 1440)).sqrt();
    let ln2x4 = 4.0 * 2f64.ln();
    let pk: Vec<f64> = h.iter().zip(&l).map(|(a, b)| (a / b).ln()).map(|x| x * x / ln2x4).collect();
    let t_ms = k[k.len() - 1].t_ms;
    let tc = t_ms + 60_000;
    let (dow, hour, minute) = dhm(t_ms);
    let hr = hour as f64 + minute as f64 / 60.0;
    let how = |ms: i64| {
        let (d, h, _) = dhm(ms);
        (d * 24 + h) as usize
    };
    let ls24 = s24.ln();
    let lsig = |w: usize| (npmean(tail(&r2, w)).sqrt() + 1e-9).ln() - ls24;
    let imb = |w: usize| {
        let x: Vec<f64> = tail(&tb, w).iter().zip(tail(&v, w)).map(|(t, v)| 2.0 * t - v).collect();
        (npsum(&x) / npsum(tail(&v, w))).abs()
    };
    let season = |ks: &mut dyn Iterator<Item = i64>| npmean(&ks.map(|ms| prof[how(ms)]).collect::<Vec<_>>()).ln();
    let sv = npmean(tail(&r2, 60)).sqrt();
    // 파이썬 max(np 값, 1e-6) / s24: 1e-6 쪽이 고르면 파이썬 float 나눗셈 -- s24 = 0 이면 예외
    let vr = if sv.is_nan() { sv } else if sv < 1e-6 { 1e-6 } else { sv };
    if sv < 1e-6 && s24 == 0.0 {
        return Err("ZeroDivisionError: float division by zero".into());
    }
    let base: [(&'static str, f64); 10] = [
        ("V_lsig5m", lsig(5)),
        ("V_lsig15m", lsig(15)),
        ("V_lsig1h", lsig(60)),
        ("V_lsig4h", lsig(240)),
        ("V_lsig24", ls24),
        ("V_park1h", (npmean(tail(&pk, 60)).sqrt() + 1e-9).ln() - ls24),
        ("V_park24", (npmean(tail(&pk, 1440)).sqrt() + 1e-9).ln() - ls24),
        ("V_jump1h", npext(&tail(&r, 60).iter().map(|x| x.abs()).collect::<Vec<_>>(), true) / s24),
        ("Q_imb1h", imb(60)),
        ("S_season_now", season(&mut (0..60).map(|j| t_ms - j * 60_000))),
    ];
    let (major, claims) = event_times(events);
    let px = c[c.len() - 1];
    let mut out = vec![];
    for &w in wins {
        let (hi, lo) = (rolling_ext(&h, w, true), rolling_ext(&l, w, false));
        let rngs: Vec<f64> = hi.iter().zip(&lo).map(|(a, b)| a - b).collect();
        let n = rngs.len();
        let (rng, prng) = (rngs[n - 1], rngs[n - 1 - w]);
        let wlo = npext(tail(&l, w), false);
        let p = (px - wlo) / rng;
        let nm = next_min(tc, &major);
        let wf = w as f64;
        let mut f: Feat = base.iter().copied().collect();
        f.extend([
            ("log_ds", (0.5 * rng / px / (s24 * wf.sqrt())).ln()),
            ("log_vr", (vr / s24).ln()),
            ("log_rng", (rng / px * 1e4).ln()),
            ("hs", (2.0 * std::f64::consts::PI * hr / 24.0).sin()),
            ("hc", (2.0 * std::f64::consts::PI * hr / 24.0).cos()),
            ("wkd", (dow >= 5) as u8 as f64),
            ("R_edge", if 1.0 - p < p { 1.0 - p } else { p }), // 파이썬 min(p, 1-p): NaN 이면 p
            ("R_trend", (px - c[c.len() - 1 - w]).abs() / rng),
            // 2026-10-24: prng = 0(직전 창이 평평) 이면 inf -- 아래 유한값 검사가 그 척도만 ok False 로 뺀다
            ("R_compress", (rng / prng).ln()),
            ("R_squeeze", (rng / nanmedian(tail(&rngs, 1440))).ln()),
            ("Q_vol5_vs_W", (npmean(tail(&v, 5)) / npmean(tail(&v, w))).ln()),
            ("Q_imbW", imb(w)),
            ("S_season_fut", season(&mut (1..=w as i64).map(|j| t_ms + j * 60_000))),
            ("M_major_in", (nm <= wf) as u8 as f64),
            ("M_major_pos", if nm <= wf { nm / wf } else { 1.0 }),
            ("M_major_recent", (prev_min(tc, &major) <= 60.0) as u8 as f64),
            ("M_claims_in", (next_min(tc, &claims) <= wf) as u8 as f64),
            ("_rng", rng),
        ]);
        out.push(f);
    }
    Ok(out)
}

pub struct Platt {
    pub a: f64,
    pub b: f64,
}

/// box_hold.platt90: `<dir>/box_hold_platt90_<asset>_1h.json`(scripts/fit_box_hold_platt90_20261026.py 가 매월 1일 쓴다).
/// 없음·깨짐·40일 넘음·다른 모델용·a/b 비유한·created_utc 에 시간대 없음(파이썬은 빼기에서 예외) = None(보정 없음).
pub fn platt_load(asset: &str, dir: &Path, model_file: &str, now_ms: i64) -> Option<Platt> {
    if !PLATT90_ASSETS.contains(&asset) {
        return None;
    }
    let c: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("box_hold_platt90_{asset}_1h.json"))).ok()?).ok()?;
    let (created, tz) = parse_iso_ns(c.get("created_utc")?.as_str()?)?;
    let fresh = tz && (now_ms as i128) * 1_000_000 - created as i128 <= 40 * 86_400 * 1_000_000_000i128;
    let (a, b) = (c.get("a")?.as_f64()?, c.get("b")?.as_f64()?);
    // 10-26 리뷰: 키가 빠지면 payload 전체가 죽었다 → 검사. str(x) 는 문자열이 아니면 늘 참
    let filled = |k: &str| c.get(k).map(|v| v.as_str().map_or(true, |s| !s.is_empty()));
    let ok = a.is_finite() && b.is_finite() && filled("fit_start")? && filled("fit_end")?;
    (fresh && ok && c.get("model_file")?.as_str()? == model_file).then_some(Platt { a, b })
}

/// 서버 QX4 결과(qx4_state["last"]) 중 등조가 쓰는 칸.
pub struct Vol {
    pub ok: bool,
    pub asof_ms: f64,
    pub log_rv: f64,
    pub log_rv_iv: Option<f64>,
}

fn build(k: &[Bar], events: &[Value], b: &BoxBundle, asset: &str, cal: Option<&Platt>, qx: Option<(&Qx4iIso, &Vol)>, rnd: fn(f64, usize) -> f64) -> Value {
    if k.len() < NEED {
        return json!({"ok": false});
    }
    let k = &k[k.len().saturating_sub(NEED + 60)..];
    let ok_scales: Vec<(&str, usize)> = WIN.iter().copied().filter(|(n, _)| b.scales.get(*n).is_some_and(|s| s.deployable)).collect();
    let fs = match features(k, events, &b.prof, &ok_scales.iter().map(|x| x.1).collect::<Vec<_>>()) {
        Ok(f) => f,
        Err(_) => return json!({"ok": false}),
    };
    let px = k[k.len() - 1].c;
    let dp = if px >= 10.0 { 2 } else { 6 }; // XRP $1.3 -- 2자리면 박스 위·아래가 1.30/1.29
    let asof = k[k.len() - 1].t_ms + 60_000;
    let mut scales = Map::new();
    for (name, w) in WIN {
        let Some(i) = ok_scales.iter().position(|x| x.0 == name) else {
            scales.insert(name.into(), json!({"ok": false})); // 검정 불통과(보정) -- 계산하지 않는다
            continue;
        };
        let (f, cf) = (&fs[i], &b.scales[name]);
        let x: Vec<f64> = cf.cols.iter().map(|c| f[c.as_str()]).collect();
        let rng = f["_rng"];
        if !(rng > 0.0 && x.iter().all(|v| v.is_finite())) {
            scales.insert(name.into(), json!({"ok": false}));
            continue;
        }
        let mut p = predict_proba(&cf.model, &x).expect("HGB 는 예외가 없다");
        if name == "1h" {
            if let Some(c) = cal {
                p = platt90(p, c.a, c.b); // 화면 q20·q80 은 그대로(연구 채점도 원 분위 · 라벨 바뀜 2.6%)
            }
        }
        let mut s = json!({"ok": true, "p": rnd(p, 4), "base": rnd(cf.train_base, 4), "q20": rnd(cf.q20, 4), "q80": rnd(cf.q80, 4),
                           "up": rnd(px + 0.5 * rng, dp), "dn": rnd(px - 0.5 * rng, dp), "half_pct": rnd(0.5 * rng / px * 100.0, 3),
                           "until_ms": asof + w as i64 * 60_000});
        // 2026-10-29 ETH 4h = QX4I σ → 두 벽 무접촉 확률 → 등조 지도(ΔBrier −.00732 단측 90% 통과). 변동성이 15분 넘게 묵었거나 없으면 HGB 그대로.
        //   half_frac 은 반올림 전 값(반올림하면 등조 계단에서 최대 5pp 튐)
        if let (Some((iso, vol)), "4h") = (qx, name) {
            if vol.ok && (asof as f64 - vol.asof_ms).abs() <= VOL_STALE_MS {
                let q = iso.p(vol.log_rv, vol.log_rv_iv, 0.5 * rng / px);
                s["p"] = json!(rnd(q, 4));
                s["src"] = json!(if vol.log_rv_iv.is_some() { "qx4i" } else { "qx4" });
            }
        }
        scales.insert(name.into(), s);
    }
    json!({"ok": true, "asof_ms": asof, "asset": asset, "scales": scales})
}

/// `/api/box-hold` 본문(화면이 읽는 필드). qx = ETH 4h 등조(서버는 ETH 만 넘긴다).
pub fn payload(k: &[Bar], events: &[Value], b: &BoxBundle, asset: &str, cal: Option<&Platt>, qx: Option<(&Qx4iIso, &Vol)>) -> Value {
    build(k, events, b, asset, cal, qx, py_round)
}
/// 시험용: 반올림 전 값.
#[doc(hidden)]
pub fn payload_unrounded(k: &[Bar], events: &[Value], b: &BoxBundle, asset: &str, cal: Option<&Platt>, qx: Option<(&Qx4iIso, &Vol)>) -> Value {
    build(k, events, b, asset, cal, qx, |x, _| x)
}
