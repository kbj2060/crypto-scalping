//! «띠 안·이탈(regime band)» -- 앞으로 4시간 안에 종가가 가장 가까운 지지~저항 띠를 벗어날 확률 (2026-11-21 · R5).
//! 원본: scripts/live_regime_band_20261102.py `from_klines`·`level_at`·`predict_latest` · 청산맵 scripts/live_liquidation_map_20260824.py
//! `_bar_arrays`·`_window_at`·`_sub_bars`·`_own_bar_alive`·`_prepare_common`·`_bins_from_common`·`levels_from_bins`·`compute_spliced_levels` ·
//! 피쳐 research_regime_acc_20261102 `frame_for` + research_regime_f_20261102 `features` · 기록 live_regime_band_recorder_20261102 `record`·`score`·
//! `score_new`·`summarize` · 크론 live_regime_band_cron_20261102 `panel`·`main`(매시 02분 · 화면 /api/regime-band 는 기록 파일 꼬리만 읽는다).
//! 순서 규칙: 청산 칸 합 = 파이썬 딕셔너리 넣은 순서대로 차례 합 · 1시간 거래량 = pandas groupby sum(Kahan) · 5분칸 행 합 = numpy 짝 합(12칸) ·
//!   rolling sum/mean/var = pandas(Kahan · 같은 값 연속) · np.round = 짝수 쪽 반올림 · round(x, 4) = 파이썬 round.
use crate::box_hold::{npsum, py_round};
use crate::qx4::roll_sum;
use crate::risk_radar::{days_from_civil, iso};
use crate::sr_lean::{jf, roll_ext, roll_var};
use crate::{predict_proba, Model};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

const NAN: f64 = f64::NAN;
pub const COINS: [&str; 4] = ["BTC", "ETH", "SOL", "XRP"];
const NEED: usize = 2700;
const LOOKBACK_H: usize = 24;
const TIERS: [f64; 6] = [10.0, 20.0, 25.0, 50.0, 75.0, 100.0];
const MMR: f64 = 0.005;
const BINS: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01];

/// 닫힌 5분봉(패널 한 줄). t = 시가 시각(초 · UTC).
#[derive(Clone, Copy)]
pub struct Bar {
    pub t: i64,
    pub h: f64,
    pub l: f64,
    pub c: f64,
    pub qv: f64,
    pub tb: f64,
}

pub struct Bundle {
    pub features: Vec<String>,
    model: Model,
}
impl Bundle {
    pub fn from_json(v: &Value) -> Result<Self, String> {
        Ok(Bundle {
            features: serde_json::from_value(v["features"].clone()).map_err(|e| e.to_string())?,
            model: serde_json::from_value(v["model"].clone()).map_err(|e| e.to_string())?,
        })
    }
}

// ── 청산맵 레벨
/// 봉 창(시작 초 · 고 · 저 · 종 · 거래량) -- 청산맵 입력(2026-11-22 pub: rtserve ctx_rest 가 REST 닫힌 1시간봉으로 같은 계산을 쓴다).
pub struct Win {
    pub t: Vec<i64>,
    pub h: Vec<f64>,
    pub l: Vec<f64>,
    pub c: Vec<f64>,
    pub v: Vec<f64>,
}
fn ss(a: &[i64], x: i64) -> usize {
    a.partition_point(|&v| v < x) // np.searchsorted(side=left)
}

/// _sub_bars: 창의 1시간봉마다 그 시간 5분봉 (고, 저, 거래량) 12칸 -- 모자란 줄은 고·저 NaN · 거래량 0.
fn sub_bars(ts: &[i64], m: &Win) -> Option<Vec<[[f64; 12]; 3]>> {
    if m.t.is_empty() {
        return None;
    }
    let last = *m.t.last().unwrap();
    Some(ts.iter().map(|&t0| {
        let mut o = [[NAN; 12]; 3];
        for j in ss(&m.t, t0)..ss(&m.t, t0 + 3600) {
            let k = ((m.t[j] - t0).div_euclid(300)) as usize;
            (o[0][k], o[1][k], o[2][k]) = (m.h[j], m.l[j], m.v[j]);
        }
        let exp = (last + 300 - t0).div_euclid(300).clamp(0, 12) as usize;
        if o[1].iter().filter(|x| x.is_finite()).count() < exp {
            o = [[NAN; 12]; 3];
        }
        for x in o[2].iter_mut() {
            if x.is_nan() {
                *x = 0.0; // nan_to_num
            }
        }
        o
    }).collect())
}

/// _own_bar_alive: 봉 자신의 청산가가 같은 시간 «그 뒤 5분봉»에 안 닿은 거래량 몫(5분봉 없으면 1).
pub fn own_alive(sub: &Option<Vec<[[f64; 12]; 3]>>, e: &[f64], liq: &[f64], long: bool) -> Vec<f64> {
    let Some(sub) = sub else { return vec![1.0; liq.len()] };
    sub.iter().enumerate().map(|(i, [h5, l5, v5])| {
        let x: Vec<f64> = (0..12).map(|j| match (long, if long { l5[j] } else { h5[j] }) {
            (true, v) if v.is_nan() => f64::INFINITY,
            (false, v) if v.is_nan() => f64::NEG_INFINITY,
            (_, v) => v,
        }).collect();
        let (mut after, mut acc) = ([0.0; 12], if long { f64::INFINITY } else { f64::NEG_INFINITY });
        for j in (0..12).rev() {
            after[j] = acc; // min/max(x[j+1:]) · 끝 칸 = ±inf
            acc = if long { acc.min(x[j]) } else { acc.max(x[j]) };
        }
        let wv: Vec<f64> = (0..12).map(|j| v5[j] * ((l5[j] <= e[i] && h5[j] >= e[i]) as u8 as f64)).collect();
        let ok: Vec<f64> = (0..12).map(|j| wv[j] * ((if long { after[j] > liq[i] } else { after[j] < liq[i] }) as u8 as f64)).collect();
        let den = npsum(&wv);
        if den > 0.0 { npsum(&ok) / den } else { 1.0 }
    }).collect()
}

pub struct Common {
    pub base: Vec<f64>,
    pub fmin: Vec<f64>,
    pub fmax: Vec<f64>,
    pub bw: f64,
    pub close: Vec<f64>,
    pub sub: Option<Vec<[[f64; 12]; 3]>>,
}

/// _prepare_common(가중·생존 필터 · 5분봉 자기 봉 판정) -- 창이 20줄 미만이거나 가격이 0 이하면 None.
pub fn prepare(w: &Win, cp: f64, m5: &Win) -> Option<Common> {
    prepare_hl(w, cp, m5, Some(240.0))
}

/// prepare 에 halflife_h(None = 감쇠 끔 · 원본 2026-11-15 60·90·180일 지도) -- rtserve 청산 막대(compute_tier_profile)가 쓴다.
pub fn prepare_hl(w: &Win, cp: f64, m5: &Win, hl: Option<f64>) -> Option<Common> {
    let n = w.t.len();
    if n < 20 || !(cp > 0.0) {
        return None;
    }
    let now = w.t[n - 1];
    let base = (0..n).map(|i| w.v[i] * hl.map_or(1.0, |hl| (-((now - w.t[i]) as f64 / 3600.0) / hl).exp())).collect();
    let (mut fmin, mut fmax) = (vec![f64::INFINITY; n], vec![f64::NEG_INFINITY; n]);
    for i in (0..n - 1).rev() {
        fmin[i] = fmin[i + 1].min(w.l[i + 1]);
        fmax[i] = fmax[i + 1].max(w.h[i + 1]);
    }
    Some(Common { base, fmin, fmax, bw: (cp * 0.001).max(1e-9), close: w.c.clone(), sub: sub_bars(&w.t, m5) })
}

/// _bins_from_common: 칸 번호 → 무게(넣은 순서 유지 · 차례 합).
pub fn bins(cm: &Common, e: &[f64]) -> Option<Bins> {
    let mut out: Vec<(i64, f64)> = vec![];
    let mut at: HashMap<i64, usize> = HashMap::new();
    let mut add = |px: &[f64], w: &[f64], alive: &dyn Fn(usize) -> bool| {
        for i in 0..px.len() {
            if alive(i) && px[i] > 0.0 {
                let b = (px[i] / cm.bw).round_ties_even() as i64;
                let k = *at.entry(b).or_insert_with(|| {
                    out.push((b, 0.0));
                    out.len() - 1
                });
                out[k].1 += w[i];
            }
        }
    };
    for lev in TIERS {
        let (lf, sf) = (1.0 - 1.0 / lev + MMR, 1.0 + 1.0 / lev - MMR);
        let ll: Vec<f64> = e.iter().map(|x| x * lf).collect();
        let sl: Vec<f64> = e.iter().map(|x| x * sf).collect();
        let (fl, fs) = (own_alive(&cm.sub, e, &ll, true), own_alive(&cm.sub, e, &sl, false));
        let wl: Vec<f64> = (0..e.len()).map(|i| cm.base[i] / 6.0 * fl[i]).collect();
        let wsh: Vec<f64> = (0..e.len()).map(|i| cm.base[i] / 6.0 * fs[i]).collect();
        add(&ll, &wl, &|i| cm.fmin[i] > ll[i] && fl[i] > 0.0);
        add(&sl, &wsh, &|i| cm.fmax[i] < sl[i] && fs[i] > 0.0);
    }
    let mx = out.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    if out.is_empty() || !(mx > 0.0) { None } else { Some(out) }
}

/// levels_from_bins 의 한쪽(scripts/live_liquidation_map_20260824.py 238~265): 가까운 순 (price, weight_pct, distance_pct) = fmt 그대로
/// (round 4 · round 4 · round 3). 2026-11-22 넓힘: 화면 청산맵 레벨(rtserve)이 무게%·거리%까지 쓴다.
pub fn side_levels(b: &[(i64, f64)], bw: f64, cp: f64, below: bool) -> Vec<(f64, f64, f64)> {
    let mx = b.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max); // 파이썬 max(값) -- NaN 없음
    let mut lv: Vec<(f64, f64)> = b.iter().map(|&(k, w)| (k as f64 * bw, w))
        .filter(|&(p, w)| w / mx >= 0.05 && (p - cp).abs() / cp <= 0.05 && if below { p < cp } else { p > cp }).collect();
    lv.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // 무게 큰 순(안정)
    lv.truncate(6);
    lv.sort_by(|a, b| if below { b.0.partial_cmp(&a.0) } else { a.0.partial_cmp(&b.0) }.unwrap());
    lv.into_iter().map(|(p, w)| (py_round(p, 4), py_round(w / mx, 4), py_round((p - cp) / cp * 100.0, 3))).collect()
}

/// levels_from_bins 의 한쪽: 가까운 순 가격(round 4).
fn side(b: &[(i64, f64)], bw: f64, cp: f64, below: bool) -> Vec<f64> {
    side_levels(b, bw, cp, below).into_iter().map(|x| x.0).collect()
}

/// level_at(df, t): 경계 t(초)에 알 수 있던 (가격, 지지[], 저항[]) -- 가까운 순.
pub fn level_at(df: &[Bar], t: i64) -> (f64, Vec<f64>, Vec<f64>) {
    let (cp, b) = spliced_bins(df, t);
    match b {
        Some((Some(bm), Some(bc), bw)) => (cp, side(&bm, bw, cp, true), side(&bc, bw, cp, false)),
        _ => (cp, vec![], vec![]),
    }
}

/// 칸 번호 → 무게 한 벌(넣은 순서).
pub type Bins = Vec<(i64, f64)>;

/// level_at 의 속: (가격, (중간가 칸 · 종가 칸 · 칸 폭)) -- 창이 모자라면(20줄 미만) 칸 없음.
pub fn spliced_bins(df: &[Bar], t: i64) -> (f64, Option<(Option<Bins>, Option<Bins>, f64)>) {
    let d = Win { t: df.iter().map(|b| b.t).collect(), h: df.iter().map(|b| b.h).collect(), l: df.iter().map(|b| b.l).collect(),
                  c: df.iter().map(|b| b.c).collect(), v: df.iter().map(|b| b.qv / b.c).collect() };
    // 1시간봉 = resample("1h") 중 12봉 다 찬 시간(고 max · 저 min · 마지막 종가 · 거래량 Kahan 합)
    let mut h1 = Win { t: vec![], h: vec![], l: vec![], c: vec![], v: vec![] };
    let mut i = 0;
    while i < d.t.len() {
        let hr = d.t[i].div_euclid(3600) * 3600;
        let j = i + d.t[i..].partition_point(|&x| x < hr + 3600);
        if j - i == 12 {
            let (mut s, mut comp) = (0.0f64, 0.0f64);
            for &v in &d.v[i..j] {
                let y = v - comp;
                let tt = s + y;
                comp = tt - s - y;
                if comp.is_nan() {
                    comp = 0.0;
                }
                s = tt;
            }
            h1.t.push(hr);
            h1.h.push(d.h[i..j].iter().copied().fold(f64::NEG_INFINITY, f64::max));
            h1.l.push(d.l[i..j].iter().copied().fold(f64::INFINITY, f64::min));
            h1.c.push(d.c[j - 1]);
            h1.v.push(s);
        }
        i = j;
    }
    let k = ss(&d.t, t);
    // _window_at
    let hour = t.div_euclid(3600) * 3600;
    let (p0, p1) = (ss(&d.t, hour), ss(&d.t, t));
    let i1 = ss(&h1.t, hour);
    let i0 = (i1 + (p1 > p0) as usize).saturating_sub(LOOKBACK_H);
    let mut w = Win { t: h1.t[i0..i1].to_vec(), h: h1.h[i0..i1].to_vec(), l: h1.l[i0..i1].to_vec(), c: h1.c[i0..i1].to_vec(), v: h1.v[i0..i1].to_vec() };
    if p1 > p0 {
        w.t.push(hour);
        w.h.push(d.h[p0..p1].iter().copied().fold(f64::NEG_INFINITY, f64::max)); // np.max -- NaN 없음
        w.l.push(d.l[p0..p1].iter().copied().fold(f64::INFINITY, f64::min));
        w.c.push(d.c[p1 - 1]);
        w.v.push(npsum(&d.v[p0..p1]));
    }
    let cp = d.c[k - 1];
    let a = k.saturating_sub(2200);
    let m5 = Win { t: d.t[a..k].to_vec(), h: d.h[a..k].to_vec(), l: d.l[a..k].to_vec(), c: vec![], v: d.v[a..k].to_vec() };
    let Some(cm) = prepare(&w, cp, &m5) else { return (cp, None) };
    let mid: Vec<f64> = (0..w.t.len()).map(|i| (w.h[i] + w.l[i]) / 2.0).collect();
    (cp, Some((bins(&cm, &mid), bins(&cm, &cm.close), cm.bw)))
}

// ── 피쳐
fn ln(x: f64) -> f64 {
    x.ln()
}
fn diff(x: &[f64], k: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if i < k { NAN } else { x[i] - x[i - k] }).collect()
}

/// frame_for(df, [i], btc, {t: level_at}) 의 학습 열 순서 피쳐(무한대 → NaN) + (R1, S1). i = 패널 행.
pub fn features(df: &[Bar], btc_c: &[f64], i: usize, lvl: &(f64, Vec<f64>, Vec<f64>), cols: &[String]) -> (Vec<f64>, f64, f64) {
    let (c, h, l, qv, tb): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
        (df.iter().map(|b| b.c).collect(), df.iter().map(|b| b.h).collect(), df.iter().map(|b| b.l).collect(), df.iter().map(|b| b.qv).collect(), df.iter().map(|b| b.tb).collect());
    let lc: Vec<f64> = c.iter().map(|x| ln(*x)).collect();
    let r1 = diff(&lc, 1);
    let r2: Vec<f64> = r1.iter().map(|x| x * x).collect();
    let rv = |w| roll_sum(&r2, w, w)[i].sqrt();
    let ad: Vec<f64> = diff(&c, 1).iter().map(|x| x.abs()).collect();
    let er = |cc: &[f64], a: &[f64], w: usize| (cc[i] - if i >= w { cc[i - w] } else { NAN }).abs() / roll_sum(a, w, w)[i];
    let (hi, lo) = (roll_ext(&h, 288, 288, true)[i], roll_ext(&l, 288, 288, false)[i]);
    let s12 = roll_sum(&qv, 12, 12);
    let flow: Vec<f64> = (0..c.len()).map(|k| 2.0 * tb[k] - qv[k]).collect();
    let (r12, r48, r288, r2016) = (rv(12), rv(48), rv(288), rv(2016));
    let wh = &h[i - 287..=i];
    let wl = &l[i - 287..=i];
    let amax = wh.iter().enumerate().fold(0, |b, (k, x)| if *x > wh[b] { k } else { b }); // np.argmax 첫 자리
    let amin = wl.iter().enumerate().fold(0, |b, (k, x)| if *x < wl[b] { k } else { b });
    let hr = ((df[i].t + 300).rem_euclid(86_400) / 3600) as f64;
    let blc: Vec<f64> = btc_c.iter().map(|x| ln(*x)).collect();
    let bd2: Vec<f64> = diff(&blc, 1).iter().map(|x| x * x).collect();
    let bad: Vec<f64> = diff(btc_c, 1).iter().map(|x| x.abs()).collect();
    let (_, sup, res) = lvl;
    let p = c[i];
    let mn = |v: &[f64]| v.iter().copied().fold(NAN, |a, b| if a.is_nan() || b < a { b } else { a });
    let mx = |v: &[f64]| v.iter().copied().fold(NAN, |a, b| if a.is_nan() || b > a { b } else { a });
    let (r1_, r2_, s1_, s2_) = (mn(res), mx(res), mx(sup), mn(sup));
    let sig = r48;
    let mut f: HashMap<&str, f64> = HashMap::from([
        ("rv12_288", ln(r12 / r288)), ("rv48_288", ln(r48 / r288)), ("rv288_2016", ln(r288 / r2016)), ("lrv288", ln(r288)),
        ("er48", er(&c, &ad, 48)), ("er288", er(&c, &ad, 288)),
        ("vr", roll_var(&diff(&lc, 12), 288, 288)[i] / (12.0 * roll_var(&r1, 288, 288)[i])),
        ("pos", (p - lo) / (hi - lo)), ("rng_rv", ln((hi - lo) / p) - ln(r288)),
        ("qv_ratio", ln(s12[i] / crate::dir4h::roll_mean_pub(&s12, 2016, 2016)[i])),
        ("tk12", roll_sum(&flow, 12, 12)[i] / s12[i]), ("tk48", roll_sum(&flow, 48, 48)[i] / roll_sum(&qv, 48, 48)[i]),
        ("since_hi", (287 - amax) as f64), ("since_lo", (287 - amin) as f64),
        ("hs", (2.0 * std::f64::consts::PI * hr / 24.0).sin()), ("hc", (2.0 * std::f64::consts::PI * hr / 24.0).cos()),
        ("btc_er48", er(btc_c, &bad, 48)), ("btc_rv", ln(roll_sum(&bd2, 48, 48)[i].sqrt() / roll_sum(&bd2, 288, 288)[i].sqrt())),
        ("lw", ln((r1_ - s1_) / p)), ("lw_sig", ln((r1_ - s1_) / p / sig)), ("dR", ln(r1_ / p) - ln(sig)), ("dS", ln(p / s1_) - ln(sig)),
        ("bpos", (p - s1_) / (r1_ - s1_)), ("nR", res.len() as f64), ("nS", sup.len() as f64),
        ("wide_R", ln(r2_ / p) - ln(sig)), ("wide_S", ln(p / s2_) - ln(sig)),
    ]);
    let x = cols.iter().map(|k| f.remove(k.as_str()).map_or(NAN, |v| if v.is_infinite() { NAN } else { v })).collect();
    (x, r1_, s1_)
}

/// panels["BTC"].reindex(df.index).c
pub fn btc_on(df: &[Bar], btc: &[Bar]) -> Vec<f64> {
    let m: HashMap<i64, f64> = btc.iter().map(|b| (b.t, b.c)).collect();
    df.iter().map(|b| m.get(&b.t).copied().unwrap_or(NAN)).collect()
}

pub struct Pred {
    pub ts: i64,
    pub p: f64,
    pub support: f64,
    pub resistance: f64,
    pub x: Vec<f64>,
    pub level: (f64, Vec<f64>, Vec<f64>),
}

/// predict_latest(panels, coin)
pub fn predict_latest(panels: &HashMap<String, Vec<Bar>>, coin: &str, b: &Bundle) -> Result<Pred, String> {
    let df = &panels[coin];
    let last = df.last().ok_or("빈 패널")?;
    if last.t.rem_euclid(3600) / 60 != 55 || df.len() < 2500 {
        return Err("AssertionError: 마지막 봉은 정시 마감 봉(시가 55분)이고 길이 ≥ 2500 이어야 한다".into());
    }
    let t = last.t + 300;
    let level = level_at(df, t);
    let (x, r1, s1) = features(df, &btc_on(df, &panels["BTC"]), df.len() - 1, &level, &b.features);
    let p = predict_proba(&b.model, &x).ok_or("predict_proba")?;
    Ok(Pred { ts: t, p, support: s1, resistance: r1, x, level })
}

pub fn proba(b: &Bundle, x: &[f64]) -> f64 {
    predict_proba(&b.model, x).unwrap()
}

// ── 크론·기록기
fn ts_str(t: i64) -> String {
    let s = iso(t * 1000);
    format!("{} {}", &s[..10], &s[11..19])
}
fn ts_parse(s: &str) -> Option<i64> {
    let n = |a: usize, b: usize| s.get(a..b).and_then(|x| x.parse::<i64>().ok());
    Some(days_from_civil(n(0, 4)?, n(5, 7)?, n(8, 10)?) * 86_400 + n(11, 13)? * 3600 + n(14, 16)? * 60 + n(17, 19)?)
}
fn jstr(s: &str) -> String {
    Value::String(s.into()).to_string() // ensure_ascii=False 와 같은 꼴(한글 그대로)
}

pub type Get<'a> = crate::dirrest::Get<'a>;

/// cron.panel: 1500봉 + 그 앞 1500봉 → 시가 중복 제거·정렬 · 닫힌 봉만 · 꼬리 2700.
pub fn panel(get: &mut Get, sym: &str, now_ms: i64) -> Result<Vec<Bar>, String> {
    let q = |end: Option<i64>| {
        let mut p: Vec<(&'static str, String)> = vec![("symbol", sym.into()), ("interval", "5m".into()), ("limit", "1500".into())];
        if let Some(e) = end {
            p.push(("endTime", e.to_string()));
        }
        p
    };
    let a = get("/fapi/v1/klines", &q(None))?;
    let a = a.as_array().ok_or("klines")?;
    let t0 = a.first().and_then(|r| r[0].as_i64()).ok_or("IndexError: list index out of range")?;
    let b = get("/fapi/v1/klines", &q(Some(t0 - 1)))?;
    let num = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).ok_or_else(|| format!("float({v})"));
    let mut seen = HashSet::new();
    let mut rows = vec![];
    for r in b.as_array().ok_or("klines")?.iter().chain(a) {
        let t = r[0].as_i64().ok_or("open_time")?;
        if seen.insert(t) && r[6].as_i64().ok_or("close_time")? < now_ms {
            rows.push(Bar { t: t.div_euclid(1000), h: num(&r[2])?, l: num(&r[3])?, c: num(&r[4])?, qv: num(&r[7])?, tb: num(&r[10])? });
        }
    }
    rows.sort_by_key(|b| b.t);
    Ok(rows.split_off(rows.len().saturating_sub(NEED)))
}

fn key(r: &Value) -> (String, String) {
    (r["ts"].as_str().unwrap_or("").into(), r["coin"].as_str().unwrap_or("").into())
}
fn fnum(v: &Value) -> f64 {
    v.as_f64().unwrap_or(NAN) // null(= 적을 때 NaN) → NaN
}

/// 원본 json.loads 는 NaN 을 읽는다 -- serde 가 읽게 NaN 을 null 로 바꿔 읽는다(쓴 글자는 그대로 둔다).
fn read_rows(p: &Path) -> Vec<Value> {
    std::fs::read_to_string(p).map(|s| s.lines().filter_map(|l| serde_json::from_str(&l.replace("NaN", "null")).ok()).collect()).unwrap_or_default()
}

fn append(p: &Path, rows: &[String]) -> Result<(), String> {
    use std::io::Write;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    for r in rows {
        writeln!(f, "{r}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// recorder.record: 4코인 예측 줄(이미 있는 (ts, coin) 은 뺌)을 붙인다. 반환 = 새 줄 수.
pub fn record(panels: &HashMap<String, Vec<Bar>>, b: &Bundle, log: &Path) -> Result<usize, String> {
    let seen: HashSet<(String, String)> = read_rows(log).iter().map(key).collect();
    let mut out = vec![];
    for coin in COINS {
        let o = predict_latest(panels, coin, b)?;
        let ts = ts_str(o.ts);
        if !seen.contains(&(ts.clone(), coin.to_string())) {
            let reg = if o.p >= 0.5 { "추세" } else { "횡보" };
            out.push(format!("{{\"ts\": {}, \"coin\": {}, \"p_trend\": {}, \"regime\": {}, \"confidence\": {}, \"support\": {}, \"resistance\": {}}}",
                             jstr(&ts), jstr(coin), jf(o.p), jstr(reg), jf((o.p - 0.5).abs() * 2.0), jf(o.support), jf(o.resistance)));
        }
    }
    append(log, &out)?;
    Ok(out.len())
}

struct Scored {
    ts: String,
    coin: String,
    p: f64,
    conf: f64,
    y: i64,
    correct: i64,
}

/// recorder.score: 결정 시각부터 48봉(t ~ t+235분 · 양끝 포함) 종가가 기록한 띠 밖으로 나간 적 있나. 4시간 안 지난 줄은 뺌.
fn score(panels: &HashMap<String, Vec<Bar>>, log: &Path) -> Vec<Scored> {
    let mut res = vec![];
    for r in read_rows(log) {
        let (ts, coin) = key(&r);
        let (Some(df), Some(t)) = (panels.get(&coin), ts_parse(&ts)) else { continue };
        let w: Vec<f64> = df.iter().filter(|b| b.t >= t && b.t <= t + 235 * 60).map(|b| b.c).collect();
        if w.len() < 48 {
            continue;
        }
        let (rs, sp, p) = (fnum(&r["resistance"]), fnum(&r["support"]), fnum(&r["p_trend"]));
        let y = w.iter().any(|&c| c > rs || c < sp) as i64;
        res.push(Scored { ts, coin, p, conf: fnum(&r["confidence"]), y, correct: ((p >= 0.5) == (y == 1)) as i64 });
    }
    res
}

/// recorder.score_new: 새로 채점된 줄을 scored 에 붙인다. 반환 = 새 줄 수.
pub fn score_new(panels: &HashMap<String, Vec<Bar>>, log: &Path, scored: &Path) -> Result<usize, String> {
    let d = score(panels, log);
    if d.is_empty() {
        return Ok(0);
    }
    let done: HashSet<(String, String)> = read_rows(scored).iter().map(key).collect();
    let new: Vec<String> = d.iter().filter(|r| !done.contains(&(r.ts.clone(), r.coin.clone())))
        .map(|r| format!("{{\"ts\": {}, \"coin\": {}, \"p_trend\": {}, \"confidence\": {}, \"y\": {}, \"correct\": {}}}", jstr(&r.ts), jstr(&r.coin), jf(r.p), jf(r.conf), r.y, r.correct))
        .collect();
    append(scored, &new)?;
    Ok(new.len())
}

/// str(pd.Interval) -- "[0.0, 0.2)"
fn bin_label(k: usize) -> String {
    format!("[{}, {})", jf(BINS[k]), jf(BINS[k + 1]))
}

/// recorder.summarize → json.dumps(…, ensure_ascii=False, indent=1) 글자. 반환 (글자, n_scored).
pub fn summarize(scored: &Path) -> (String, usize) {
    let rows = read_rows(scored);
    if rows.is_empty() {
        return ("{\n \"n_scored\": 0\n}".into(), 0);
    }
    let mean = |v: &[i64]| v.iter().sum::<i64>() as f64 / v.len() as f64;
    let (mut by_bin, mut by_coin): (BTreeMap<usize, Vec<i64>>, BTreeMap<String, Vec<i64>>) = Default::default();
    let (mut ys, mut cs, mut tss) = (vec![], vec![], vec![]);
    for r in &rows {
        let (ts, coin) = key(r);
        let (conf, y, c) = (fnum(&r["confidence"]), r["y"].as_i64().unwrap_or(0), r["correct"].as_i64().unwrap_or(0));
        if let Some(k) = (0..5).find(|&k| conf >= BINS[k] && conf < BINS[k + 1]) {
            by_bin.entry(k).or_default().push(c); // pd.cut(right=False) · 밖이면 NaN(빠짐)
        }
        by_coin.entry(coin).or_default().push(c);
        (ys.push(y), cs.push(c), tss.push(ts));
    }
    let grp = |items: Vec<(String, &Vec<i64>)>| {
        let s: Vec<String> = items.iter().map(|(k, v)| format!("  {}: {{\n   \"n\": {},\n   \"acc\": {}\n  }}", jstr(k), v.len(), jf(mean(v)))).collect();
        if s.is_empty() { "{}".to_string() } else { format!("{{\n{}\n }}", s.join(",\n")) }
    };
    let coin_acc: Vec<String> = by_coin.iter().map(|(k, v)| format!("  {}: {{\n   \"n\": {},\n   \"acc\": {}\n  }}", jstr(k), v.len(), jf(mean(v)))).collect();
    let txt = format!("{{\n \"n_scored\": {},\n \"first\": {},\n \"last\": {},\n \"acc\": {},\n \"trend_rate\": {},\n \"by_conf\": {},\n \"by_coin\": {}\n}}",
                      rows.len(), jstr(tss.iter().min().unwrap()), jstr(tss.iter().max().unwrap()), jf(mean(&cs)), jf(mean(&ys)),
                      grp(by_bin.iter().map(|(k, v)| (bin_label(*k), v)).collect()), if coin_acc.is_empty() { "{}".into() } else { format!("{{\n{}\n }}", coin_acc.join(",\n")) });
    (txt, rows.len())
}

/// cron.main: 4코인 패널 → 마지막 정시 마감 봉까지 자름 → 같은 시각·70분 안 확인 → 기록·채점·요약. 반환 = 출력 JSON 글자.
pub fn cron(get: &mut Get, now_ms: i64, b: &Bundle, log: &Path, scored: &Path, summary: &Path) -> Result<String, String> {
    let mut panels: HashMap<String, Vec<Bar>> = HashMap::new();
    for c in COINS {
        let mut p = panel(get, &format!("{c}USDT"), now_ms)?;
        let k = p.iter().rposition(|x| x.t.rem_euclid(3600) / 60 == 55).ok_or("IndexError: index -1 is out of bounds")?;
        p.truncate(k + 1);
        panels.insert(c.into(), p);
    }
    let last: HashSet<i64> = panels.values().map(|p| p.last().unwrap().t).collect();
    if last.len() != 1 {
        return Err("AssertionError: 코인별 마지막 정시 봉이 다름".into());
    }
    let dt = panels["XRP"].last().unwrap().t + 300;
    let age = (now_ms as f64 / 1000.0 - dt as f64) / 60.0;
    if !(age <= 70.0) {
        return Err(format!("AssertionError: 데이터가 오래됨({age:.0}분 전 결정 시각)"));
    }
    let new = record(&panels, b, log)?;
    let n = score_new(&panels, log, scored)?;
    let (s, tot) = summarize(scored);
    if let Some(d) = summary.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(summary, s).map_err(|e| format!("{}: {e}", summary.display()))?;
    Ok(format!("{{\"recorded\": {new}, \"newly_scored\": {n}, \"n_scored_total\": {tot}, \"decision_ts\": {}}}", jstr(&ts_str(dt))))
}
