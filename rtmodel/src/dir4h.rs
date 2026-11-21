//! 4시간 방향 -- REST 꼴 5분봉·metrics → 피쳐 65 → HGB 회귀 → 화면 payload (2026-11-21 · R5).
//! 원본 scripts/live_eth_dir4h_20261015.py `build_features`(:89-165)·`frames_for`(:167)·`predict`(:262)·`payload_from_features`(:379)과 같은 값을 낸다.
//! 🔴화면 «4h 예측» 칸 · 참고용(전진 판정 전). 결정 = 5분봉 마감, 보유 4h. REST 수집(rest_frames·daily_closes)은 안 옮겼다 -- 입력은 부르는 쪽이 준다.
//!
//! 끝자리까지 맞추려고 따른 계산(numpy 2.4.3 · pandas 2.3.3, tools/check_pandas_ops.py 가 실제 값과 비트 대조):
//! - rolling 은 전부 입력의 ±inf 를 NaN 으로 바꾼 뒤(_prep_values) · mean = Kahan(더하기·빼기 따로)+음수 수 부호 보정+같은 값 연속 ·
//!   std = sr_lean::roll_std · median = 가운데(짝수면 (위+아래)/2) · quantile(linear) = lo + (hi−lo)·(q(n−1) − ⌊q(n−1)⌋) · max/min = sr_lean::roll_ext.
//! - ewm(alpha, adjust=True).mean = 가중 평균 점화식(값이 같으면 안 바꿈).
//! - np.nanmean·nanstd(axis 0, 코인 19개) = 코인 차례 합(축 0 줄이기는 짝 합이 아니다).
//! - 🔴P_ac144(np.corrcoef) 는 numpy 가 BLAS(개발기 MKL · 서버 미확인)로 내적한다 -- 여기는 차례 합이라 끝자리가 다를 수 있다(시험이 폭을 잰다).
use crate::box_hold::{npsum, py_round};
use crate::sr_lean::{roll_ext, roll_std};
use crate::{predict_proba, read_json, Model};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

pub const COINS: [&str; 19] = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BNB", "BTC", "DOGE", "DOT", "FIL", "INJ", "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TRX", "XRP"];
/// 연구 feat_cols 순서(M·D·B·S 뺌) -- 모델 입력 순서 계약.
pub const FEATS: [&str; 65] = ["P_r5m", "P_r15m", "P_r1h", "P_r4h", "P_r1d", "P_r7d", "P_r28d", "P_rv1h", "P_rv1d", "P_vratio", "P_rsi14", "P_pctb",
    "P_pos1d", "P_veto_z", "P_ac144", "P_trend5", "T_imb5m", "T_imb15m", "T_imb1h", "T_imb4h", "T_cvd1h_z", "T_vol_z",
    "T_trades_z", "T_size", "O_doi5m", "O_doi1h", "O_doi4h", "O_doi1d", "O_doi1h_z", "O_doi1h_x_r", "O_toppos",
    "O_toppos_d1h", "O_toppos_z", "O_topacct", "O_topacct_d1h", "O_topacct_z", "O_glob", "O_glob_d1h", "O_glob_z",
    "O_takerls", "O_takerls_d1h", "O_takerls_z", "O_top_minus_glob", "X_btc_r5m", "X_btc_r15m", "X_btc_r1h", "X_btc_r4h",
    "X_btc_imb1h", "X_btc_doi1h", "X_breadth1h", "X_breadth1d", "X_eth_rel1h", "X_eth_rel1d", "X_eth_rank1d", "X_disp1h",
    "C_hsin", "C_hcos", "C_dow", "C_to_funding", "C_weekend", "U_eth_since_open", "U_eth_f30_day", "U_btc_since_open",
    "U_btc_f30_day", "U_min_since_open"];
const NAN: f64 = f64::NAN;
const BAR: i64 = 300_000;
const DAY: i64 = 86_400_000;

/// vision 꼴 5분봉 한 코인(timestamp = 봉 시작 ms). metrics 가 없는 코인은 NaN.
#[derive(Clone, Default)]
pub struct Frame {
    pub ts: Vec<i64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub trades: Vec<f64>,
    pub tbb: Vec<f64>,
    /// [sum_open_interest, count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio]
    pub m: [Vec<f64>; 5],
}

/// 그리드에 얹은 한 코인(열 = 그리드 길이).
struct G {
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
    trades: Vec<f64>,
    tbb: Vec<f64>,
    m: [Vec<f64>; 5],
}

/// grid(): 시각 정렬 · 같은 시각은 뒤 줄 · ts 격자에 왼쪽 결합(없는 봉 = NaN). ts None = 그 코인 처음~끝 5분 격자.
// ponytail: 정렬은 안정 정렬(같은 시각이면 입력 순서) -- 원본 sort_values 는 quicksort(불안정)라 같은 시각이 여럿이면 남는 줄이 다를 수 있다.
//   REST 원천은 _merge 가 이미 중복을 지운다(겹친 시각은 합성 시험에서만) -- 시험이 그 경우의 결과를 잰다.
fn grid(f: &Frame, ts: Option<&[i64]>) -> (Vec<i64>, G) {
    let mut idx: Vec<usize> = (0..f.ts.len()).collect();
    idx.sort_by_key(|&i| f.ts[i]);
    let mut last: HashMap<i64, usize> = HashMap::new();
    for &i in &idx {
        last.insert(f.ts[i], i);
    }
    let own;
    let ts = match ts {
        Some(t) => t,
        None => {
            let (a, b) = (f.ts[idx[0]], f.ts[idx[idx.len() - 1]]);
            own = (0..=(b - a) / BAR).map(|k| a + k * BAR).collect::<Vec<_>>();
            &own
        }
    };
    let pick = |c: &Vec<f64>| ts.iter().map(|t| last.get(t).map_or(NAN, |&i| c[i])).collect::<Vec<f64>>();
    let g = G { high: pick(&f.high), low: pick(&f.low), close: pick(&f.close), volume: pick(&f.volume), trades: pick(&f.trades), tbb: pick(&f.tbb),
                m: [pick(&f.m[0]), pick(&f.m[1]), pick(&f.m[2]), pick(&f.m[3]), pick(&f.m[4])] };
    (ts.to_vec(), g)
}

fn ffill(x: &[f64]) -> Vec<f64> {
    let mut o = x.to_vec();
    for i in 1..o.len() {
        if o[i].is_nan() {
            o[i] = o[i - 1];
        }
    }
    o
}

/// lag(x, k): 원본은 k > n 이면 길이가 안 맞아 예외.
fn lag(x: &[f64], k: usize) -> Result<Vec<f64>, String> {
    if k > x.len() {
        return Err(format!("operands could not be broadcast together with shapes ({},) ({k},)", x.len()));
    }
    Ok((0..x.len()).map(|i| if i >= k { x[i - k] } else { NAN }).collect())
}

/// np.nan_to_num: NaN → 0 · ±inf → ±f64::MAX.
fn n2n(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else if v.is_infinite() { f64::MAX.copysign(v) } else { v }
}

/// rsum(x, k): 누적합(차례) 차이 -- 앞 k−1 칸 NaN.
fn rsum(x: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.0; x.len() + 1];
    for i in 0..x.len() {
        c[i + 1] = if i == 0 { n2n(x[0]) } else { c[i] + n2n(x[i]) };
    }
    (0..x.len()).map(|i| if i + 1 >= k { c[i + 1] - c[i + 1 - k] } else { NAN }).collect()
}

fn prep(x: &[f64]) -> Vec<f64> {
    x.iter().map(|v| if v.is_infinite() { NAN } else { *v }).collect()
}

fn shift1(x: &[f64]) -> Vec<f64> {
    (0..x.len()).map(|i| if i == 0 { NAN } else { x[i - 1] }).collect()
}

/// pandas rolling(w, min_periods).mean().
fn roll_mean(x: &[f64], w: usize, minp: usize) -> Vec<f64> {
    let x = prep(x);
    let (mut nobs, mut sx, mut ca, mut cr, mut neg, mut same) = (0i64, 0.0f64, 0.0f64, 0.0f64, 0i64, 0i64);
    let mut prev = x.first().copied().unwrap_or(NAN);
    let mut out = vec![NAN; x.len()];
    for i in 0..x.len() {
        if i >= w && !x[i - w].is_nan() {
            let v = x[i - w];
            nobs -= 1;
            let y = -v - cr;
            let t = sx + y;
            cr = t - sx - y;
            sx = t;
            neg -= v.is_sign_negative() as i64;
        }
        let v = x[i];
        if !v.is_nan() {
            nobs += 1;
            let y = v - ca;
            let t = sx + y;
            ca = t - sx - y;
            sx = t;
            neg += v.is_sign_negative() as i64;
            same = if v == prev { same + 1 } else { 1 };
            prev = v;
        }
        if nobs >= minp.max(1) as i64 && nobs > 0 {
            let r = sx / nobs as f64;
            out[i] = if same >= nobs { prev } else if neg == 0 && r < 0.0 { 0.0 } else if neg == nobs && r > 0.0 { 0.0 } else { r };
        }
    }
    out
}

#[doc(hidden)]
pub fn roll_mean_pub(x: &[f64], w: usize, minp: usize) -> Vec<f64> {
    roll_mean(x, w, minp)
}

/// rz(x, w, mp): (x − 중앙값) / ((q75 − q25)/1.349) · 창 = 한 칸 밀린 x · 폭 0 → NaN. 정렬된 창 하나로 셋을 같이 낸다.
fn rz(x: &[f64], w: usize, mp: usize) -> Vec<f64> {
    let s = prep(&shift1(x));
    let mut win: Vec<f64> = vec![];
    let mut out = vec![NAN; x.len()];
    let q = |v: &[f64], q: f64| {
        let f = q * (v.len() - 1) as f64;
        let k = f as usize;
        if k as f64 == f { v[k] } else { v[k] + (v[k + 1] - v[k]) * (f - k as f64) }
    };
    for i in 0..x.len() {
        if i >= w && !s[i - w].is_nan() {
            let j = win.partition_point(|&v| v < s[i - w]);
            win.remove(j);
        }
        if !s[i].is_nan() {
            let j = win.partition_point(|&v| v < s[i]);
            win.insert(j, s[i]);
        }
        let n = win.len();
        if n >= mp.max(1) {
            let med = if n % 2 == 1 { win[n / 2] } else { (win[n / 2] + win[n / 2 - 1]) / 2.0 };
            let iqr = (q(&win, 0.75) - q(&win, 0.25)) / 1.349;
            out[i] = (x[i] - med) / if iqr == 0.0 { NAN } else { iqr };
        }
    }
    out
}

/// ewm(alpha).mean() (adjust=True · ignore_na=False · min_periods=0).
fn ewm(x: &[f64], alpha: f64) -> Vec<f64> {
    let x = prep(x);
    let mut out = vec![NAN; x.len()];
    let (mut w, mut old, f) = (x[0], 1.0f64, 1.0 - alpha);
    let mut nobs = !x[0].is_nan() as i64;
    out[0] = if nobs > 0 { w } else { NAN };
    for i in 1..x.len() {
        let v = x[i];
        let obs = !v.is_nan();
        nobs += obs as i64;
        if !w.is_nan() {
            old *= f;
            if obs {
                if w != v {
                    w = (old * w + 1.0 * v) / (old + 1.0);
                }
                old += 1.0;
            }
        } else if obs {
            w = v;
        }
        out[i] = if nobs >= 1 { w } else { NAN };
    }
    out
}

/// rolling(144).apply(corrcoef(v[:-1], v[1:])[0, 1]) -- 창에 NaN 이 있으면 NaN.
fn roll_ac(r: &[f64], w: usize) -> Vec<f64> {
    let r = prep(r);
    let mut out = vec![NAN; r.len()];
    for i in w - 1..r.len() {
        let v = &r[i + 1 - w..=i];
        if v.iter().any(|x| x.is_nan()) {
            continue;
        }
        let (a, b) = (&v[..w - 1], &v[1..]);
        let n = (w - 1) as f64;
        let (ma, mb) = (npsum(a) / n, npsum(b) / n);
        let (xa, xb): (Vec<f64>, Vec<f64>) = (a.iter().map(|x| x - ma).collect(), b.iter().map(|x| x - mb).collect());
        let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).fold(0.0, |s, (x, y)| s + x * y); // ponytail: numpy 는 BLAS 내적 -- 끝자리 다를 수 있음
        let f = 1.0 / (n - 1.0);
        let (c00, c01, c11) = (dot(&xa, &xa) * f, dot(&xa, &xb) * f, dot(&xb, &xb) * f);
        let c = c01 / c00.sqrt() / c11.sqrt();
        out[i] = if c.is_nan() { c } else { c.clamp(-1.0, 1.0) };
    }
    out
}

fn sign(v: f64) -> f64 {
    if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { v }
}

/// frames_for: 대상 코인을 "ETH" 자리에, ETH 를 그 코인 자리에.
pub fn frames_for(frames: &HashMap<String, Frame>, coin: &str) -> HashMap<String, Frame> {
    let mut f = frames.clone();
    if coin != "ETH" {
        f.insert("ETH".into(), frames[coin].clone());
        f.insert(coin.into(), frames["ETH"].clone());
    }
    f
}

/// 피쳐 표: 행 = 대상(ETH 자리) 5분 격자 · 열 = FEATS. Err = 원본이 예외를 내는 입력(대상 패널 < 8,064봉).
pub struct Features {
    pub ts: Vec<i64>,
    pub cols: HashMap<&'static str, Vec<f64>>,
}

/// build_features(frames, daily_close). daily = (일 시작 ms, 그날 종가) 오름차순.
pub fn build_features(frames: &HashMap<String, Frame>, daily: &[(i64, f64)]) -> Result<Features, String> {
    let (ts, p) = grid(&frames["ETH"], None);
    let n = ts.len();
    let c = ffill(&p.close);
    let lc: Vec<f64> = c.iter().map(|v| v.ln()).collect();
    let mut ft: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let sub = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<f64>>();
    let div = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x / y).collect::<Vec<f64>>();
    let scale = |a: Vec<f64>, k: f64| a.into_iter().map(|x| x * k).collect::<Vec<f64>>();
    let r1: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { (lc[i] - lc[i - 1]) * 1e4 }).collect();
    for (k, nm) in [(1, "P_r5m"), (3, "P_r15m"), (12, "P_r1h"), (48, "P_r4h"), (288, "P_r1d"), (2016, "P_r7d"), (8064, "P_r28d")] {
        ft.insert(nm, scale(sub(&lc, &lag(&lc, k)?), 1e4));
    }
    ft.insert("P_rv1h", roll_std(&prep(&r1), 12, 12));
    ft.insert("P_rv1d", roll_std(&prep(&r1), 288, 288));
    ft.insert("P_vratio", div(&ft["P_rv1h"], &ft["P_rv1d"]));
    let d: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { c[i] - c[i - 1] }).collect();
    let up: Vec<f64> = d.iter().map(|&v| if v < 0.0 { 0.0 } else { v }).collect(); // clip(lower=0)
    let dn: Vec<f64> = d.iter().map(|&v| -(if v > 0.0 { 0.0 } else { v })).collect(); // -clip(upper=0)
    let (eu, ed) = (ewm(&up, 1.0 / 14.0), ewm(&dn, 1.0 / 14.0));
    ft.insert("P_rsi14", (0..n).map(|i| 100.0 - 100.0 / (1.0 + eu[i] / ed[i])).collect());
    let (m20, s20) = (roll_mean(&c, 20, 20), roll_std(&prep(&c), 20, 20));
    ft.insert("P_pctb", (0..n).map(|i| (c[i] - m20[i]) / (2.0 * s20[i])).collect());
    let (hi, lo) = (&p.high, &p.low);
    let (h288, l288) = (roll_ext(&prep(hi), 288, 288, true), roll_ext(&prep(lo), 288, 288, false));
    ft.insert("P_pos1d", (0..n).map(|i| (c[i] - l288[i]) / (h288[i] - l288[i])).collect());
    let c1 = lag(&c, 1)?;
    let mx = |a: f64, b: f64| if a.is_nan() || b.is_nan() { NAN } else { a.max(b) }; // np.maximum: NaN 이면 NaN
    let tr: Vec<f64> = (0..n).map(|i| mx(hi[i] - lo[i], mx((hi[i] - c1[i]).abs(), (lo[i] - c1[i]).abs()))).collect();
    let (m144, etr) = (roll_mean(&c, 144, 144), ewm(&tr, 1.0 / 144.0));
    ft.insert("P_veto_z", (0..n).map(|i| (c[i] - m144[i]) / etr[i]).collect());
    ft.insert("P_ac144", roll_ac(&r1, 144));
    // 일봉 종가 = 다음날 00:00 에 앎 · 7·14·28·56·90일 추세 부호 평균
    ft.insert("P_trend5", (0..n).map(|i| {
        let t = ts[i] + BAR;
        let dk = daily.partition_point(|&(d0, _)| d0 + DAY <= t) as i64 - 1;
        let mut s = 0.0;
        for (j, l) in [7i64, 14, 28, 56, 90].into_iter().enumerate() {
            let v = if dk - l >= 0 { sign(daily[dk.max(0) as usize].1 - daily[(dk - l).max(0) as usize].1) } else { NAN };
            s = if j == 0 { v } else { s + v };
        }
        s / 5.0
    }).collect());
    let (v, tb, trd) = (&p.volume, &p.tbb, &p.trades);
    let dl: Vec<f64> = (0..n).map(|i| 2.0 * tb[i] - v[i]).collect();
    for (k, nm) in [(1, "T_imb5m"), (3, "T_imb15m"), (12, "T_imb1h"), (48, "T_imb4h")] {
        ft.insert(nm, div(&rsum(&dl, k), &rsum(v, k)));
    }
    let (dl12, v12, t12) = (rsum(&dl, 12), rsum(v, 12), rsum(trd, 12));
    ft.insert("T_cvd1h_z", div(&dl12, &roll_std(&prep(&shift1(&dl12)), 2016, 288)));
    ft.insert("T_vol_z", rz(&v12.iter().map(|x| x.ln()).collect::<Vec<_>>(), 2016, 288));
    ft.insert("T_trades_z", rz(&t12.iter().map(|x| x.ln()).collect::<Vec<_>>(), 2016, 288));
    ft.insert("T_size", (0..n).map(|i| (v12[i] / t12[i]).ln()).collect());
    let oi = lag(&p.m[0], 1)?;
    let loi: Vec<f64> = oi.iter().map(|x| x.ln()).collect();
    for (k, nm) in [(1, "O_doi5m"), (12, "O_doi1h"), (48, "O_doi4h"), (288, "O_doi1d")] {
        ft.insert(nm, scale(sub(&loi, &lag(&oi, k)?.iter().map(|x| x.ln()).collect::<Vec<_>>()), 1e4));
    }
    ft.insert("O_doi1h_z", rz(&ft["O_doi1h"], 2016, 288));
    ft.insert("O_doi1h_x_r", (0..n).map(|i| ft["O_doi1h"][i] * sign(ft["P_r1h"][i])).collect());
    for (col, nm) in [(2, ["O_toppos", "O_toppos_d1h", "O_toppos_z"]), (1, ["O_topacct", "O_topacct_d1h", "O_topacct_z"]), (3, ["O_glob", "O_glob_d1h", "O_glob_z"]),
                      (4, ["O_takerls", "O_takerls_d1h", "O_takerls_z"])] {
        let x: Vec<f64> = lag(&p.m[col], 1)?.iter().map(|v| v.ln()).collect();
        ft.insert(nm[1], sub(&x, &lag(&x, 12)?));
        ft.insert(nm[2], rz(&x, 8064, 2016));
        ft.insert(nm[0], x);
    }
    ft.insert("O_top_minus_glob", sub(&ft["O_toppos"], &ft["O_glob"]));
    let (mut r12, mut r288) = (vec![], vec![]);
    for other in COINS {
        let q = frames.get(other).map(|f| grid(f, Some(&ts)).1);
        let lq: Vec<f64> = match &q {
            Some(q) => ffill(&q.close).iter().map(|x| x.ln()).collect(),
            None => vec![NAN; n],
        };
        r12.push(scale(sub(&lq, &lag(&lq, 12)?), 1e4));
        r288.push(scale(sub(&lq, &lag(&lq, 288)?), 1e4));
        if other == "BTC" {
            let q = q.ok_or("AttributeError: BTC 5분봉 없음(원본도 q.taker_buy_base 에서 예외)")?;
            for (k, nm) in [(1, "X_btc_r5m"), (3, "X_btc_r15m"), (12, "X_btc_r1h"), (48, "X_btc_r4h")] {
                ft.insert(nm, scale(sub(&lq, &lag(&lq, k)?), 1e4));
            }
            let bdl: Vec<f64> = (0..n).map(|i| 2.0 * q.tbb[i] - q.volume[i]).collect();
            ft.insert("X_btc_imb1h", div(&rsum(&bdl, 12), &rsum(&q.volume, 12)));
            let boi = lag(&q.m[0], 1)?;
            ft.insert("X_btc_doi1h", scale(sub(&boi.iter().map(|x| x.ln()).collect::<Vec<_>>(), &lag(&boi, 12)?.iter().map(|x| x.ln()).collect::<Vec<_>>()), 1e4));
        }
    }
    // np.nanmean·nanstd(axis 0): 코인 차례 합
    let nanmean = |r: &[Vec<f64>], i: usize| {
        let (mut s, mut k) = (0.0, 0usize);
        for (j, row) in r.iter().enumerate() {
            let x = if row[i].is_nan() { 0.0 } else { row[i] };
            s = if j == 0 { x } else { s + x };
            k += !row[i].is_nan() as usize;
        }
        (s / k as f64, k)
    };
    let (mut b1h, mut b1d, mut disp, mut rank) = (vec![NAN; n], vec![NAN; n], vec![NAN; n], vec![NAN; n]);
    for i in 0..n {
        let (m12, k12) = nanmean(&r12, i);
        b1h[i] = m12;
        b1d[i] = nanmean(&r288, i).0;
        let mut ss = 0.0;
        for (j, row) in r12.iter().enumerate() {
            let x = if row[i].is_nan() { 0.0 } else { row[i] - m12 };
            ss = if j == 0 { x * x } else { ss + x * x };
        }
        disp[i] = if k12 == 0 { NAN } else { (ss / k12 as f64).sqrt() };
        let pr = ft["P_r1d"][i];
        rank[i] = r288.iter().filter(|row| row[i] < pr).count() as f64 / COINS.len() as f64;
    }
    ft.insert("X_eth_rel1h", sub(&ft["P_r1h"], &b1h));
    ft.insert("X_eth_rel1d", sub(&ft["P_r1d"], &b1d));
    ft.insert("X_breadth1h", b1h);
    ft.insert("X_breadth1d", b1d);
    ft.insert("X_disp1h", disp);
    ft.insert("X_eth_rank1d", rank);
    // 시각 = 봉 마감(ts + 5분)
    let tt: Vec<i64> = ts.iter().map(|t| t + BAR).collect();
    let hr: Vec<f64> = tt.iter().map(|t| { let s = t.rem_euclid(DAY) / 1000; (s / 3600) as f64 + (s / 60 % 60) as f64 / 60.0 }).collect();
    let dow: Vec<i64> = tt.iter().map(|t| (t.div_euclid(DAY) + 3).rem_euclid(7)).collect(); // 1970-01-01 = 목요일(3)
    let tau = 2.0 * std::f64::consts::PI;
    ft.insert("C_hsin", hr.iter().map(|h| (tau * h / 24.0).sin()).collect());
    ft.insert("C_hcos", hr.iter().map(|h| (tau * h / 24.0).cos()).collect());
    ft.insert("C_dow", dow.iter().map(|&d| d as f64).collect());
    ft.insert("C_to_funding", hr.iter().map(|h| (8.0 - h % 8.0) % 8.0).collect());
    ft.insert("C_weekend", dow.iter().map(|&d| (d >= 5) as u8 as f64).collect());
    for (sym, pre) in [("ETH", ["U_eth_since_open", "U_eth_f30_day"]), ("BTC", ["U_btc_since_open", "U_btc_f30_day"])] {
        // 시각 t 의 가격 = t−5분 봉 종가(그 코인 자기 격자 · 앞 값 채움)
        let (sts, sg) = grid(&frames[sym], None);
        let sc = ffill(&sg.close);
        let at: HashMap<i64, f64> = sts.iter().copied().zip(sc).collect();
        let px = |t: i64| at.get(&(t - BAR)).copied().unwrap_or(NAN);
        let (mut a, mut b) = (vec![NAN; n], vec![NAN; n]);
        for i in 0..n {
            let d0 = tt[i].div_euclid(DAY) * DAY;
            let (now, o, o30) = (px(tt[i]), px(d0), px(d0 + 1_800_000));
            a[i] = (now / o).ln() * 1e4;
            b[i] = if tt[i] - d0 >= 1_800_000 { (o30 / o).ln() * 1e4 } else { NAN };
        }
        ft.insert(pre[0], a);
        ft.insert(pre[1], b);
    }
    ft.insert("U_min_since_open", tt.iter().map(|t| (t - t.div_euclid(DAY) * DAY) as f64 / 1000.0 / 60.0).collect());
    for v in ft.values_mut() {
        for x in v.iter_mut() {
            if x.is_infinite() {
                *x = NAN; // ft.replace([inf, -inf], nan)
            }
        }
    }
    Ok(Features { ts, cols: ft })
}

/// 내보낸 모델(tools/export_dir4h.py).
#[derive(Deserialize)]
pub struct Art {
    pub coin: String,
    pub features: Vec<String>,
    #[serde(deserialize_with = "crate::bit")]
    pub lo: f64,
    #[serde(deserialize_with = "crate::bit")]
    pub hi: f64,
    #[serde(deserialize_with = "crate::bit")]
    pub bp_scale: f64,
    pub month: String,
    pub train_window: Vec<String>,
    pub models: Vec<Model>,
}

impl Art {
    pub fn load(p: &Path) -> Result<Art, String> {
        read_json(p)
    }

    /// predict(art, F 의 행 i): 피쳐를 float32 로 → 모델 평균.
    pub fn predict(&self, f: &Features, i: usize) -> f64 {
        let x: Vec<f64> = self.features.iter().map(|c| f.cols[c.as_str()][i] as f32 as f64).collect();
        let mut s = 0.0;
        for (k, m) in self.models.iter().enumerate() {
            let v = predict_proba(m, &x).expect("HGB 는 예외가 없다");
            s = if k == 0 { v } else { s + v };
        }
        s / self.models.len() as f64
    }
}

/// payload_from_features: 마지막 봉 예측·문턱·신호 + 최근 48봉 이력(bp 환산은 표시만 -- 신호는 모형 단위로).
pub fn payload(art: &Art, f: &Features) -> Value {
    let arm = match art.coin.as_str() { "ETH" => "S", "SOL" => "E", "XRP" => "J", _ => "?" }; // 원본 ARM 표
    let (k, n) = (art.bp_scale, f.ts.len());
    let p = art.predict(f, n - 1);
    let nan: Vec<&str> = FEATS.iter().copied().filter(|c| !f.cols[c][n - 1].is_finite()).collect();
    let sig = if p >= art.hi { 1 } else if p <= art.lo { -1 } else { 0 };
    let hist: Vec<Value> = (n.saturating_sub(48)..n).map(|i| json!([f.ts[i] + BAR, py_round(art.predict(f, i) * k, 2)])).collect();
    json!({"ok": true, "asset": art.coin.to_lowercase(), "arm": arm, "decided_at_ms": f.ts[n - 1] + BAR, "pred_bp": py_round(p * k, 2), "lo": py_round(art.lo * k, 2),
           "hi": py_round(art.hi * k, 2), "signal": sig, "model_month": art.month, "train_window": art.train_window, "nan_feats": nan, "hist": hist})
}
