//! 4h 변동성 QX4(+QX4I) -- REST 5분봉 4코인 → 피쳐(배포 22열 + HAR 13 + 시각 2 + 4코인 평균 4) → HGB 셋·MLP 둘·4코인 HGB·4코인 MLP 의 로그 RV 평균 → 화면 payload (2026-11-21 · R5).
//! 원본 scripts/live_vol4h_qx4_20261025.py `frames_from_klines`·`_mlp_np`·`_pred`·`predict_log`·`qx4_from_klines`·`atm_iv`·`qx4i_log`(+ 서버 qx4_payload 의 QX4I 덧붙이기)와
//! 연구 research_vol4h_forecast_horserace_20261025 `features`·`add_mkt` · live_eth_sizing_vol_model_20260912 `build_features`.
//! 화면 «4h 변동성» 칸(`/api/vol-forecast` · 5분) · 박스 4h 등조 입력. 전진 fwd_vol(2026-12-23)은 동결 파일로 다시 계산한다(서빙 기록 아님).
//! 🔴MLP 는 numpy float32 행렬곱(`h @ W.T + b` · BLAS sgemm)이다. 여기는 k 차례 f32 FMA -- 서버 OpenBLAS(Haswell 커널)의 가운데 타일은 이 순서와 같고
//!   가장자리 타일(출력 0~7·56~63 열 · 일부 행)은 다른 순서라 1ulp(f32 · 상대 ~2e-7) 갈린다(2026-11-21 실측). 그 몫이 로그 RV 에 상대 ~6e-8 로 남는다 --
//!   화면 값(3자리 반올림)은 시험 8벌 × 4코인에서 전부 같았다. 가장자리 커널 순서는 OpenBLAS 소스 없이 못 찾았다.
use crate::box_hold::{npsum, py_round};
use crate::sr_lean::{roll_ext, roll_std};
use crate::{predict_proba, Model};
use serde_json::{json, Value};
use std::collections::HashMap;

pub const COINS: [&str; 4] = ["ETH", "XRP", "BTC", "SOL"];
const BAR: i64 = 300_000;
const REF_BARS: usize = 30 * 288;
const WARM: usize = 8640;
pub const NEED_BARS: usize = REF_BARS + WARM + 288;
const STALE_MS: f64 = 600_000.0;
const MAX_FILLED: usize = 36;
const NAN: f64 = f64::NAN;
const WS: [usize; 5] = [12, 48, 288, 2016, 8640];
pub const MKT: [&str; 4] = ["mkt_lrv12", "mkt_lrv48", "mkt_lrv288", "mkt_qvz"];
/// 연구 BETA_I(12라운드 · 12b) -- QX4I 보정 계수(동결)
const BETA_I: [(&str, f64, f64); 2] = [("ETH", -0.04708836659130656, 0.17635794080180575), ("BTC", -0.025686466054476508, 0.11858982566109531)];

fn bits(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}
fn bits_vec(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(bits).collect()
}

pub struct Hgbs {
    models: Vec<Model>,
    cols: Vec<String>,
}
pub struct Mlp {
    /// 시드마다 [(W 행 우선 out×in, out, in, b)]
    layers: Vec<Vec<(Vec<f32>, usize, usize, Vec<f32>)>>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    ym: f64,
    ys: f64,
    cols: Vec<String>,
}
pub struct CoinArt {
    m3: Hgbs,
    m6: Mlp,
    c: f64,
    expand: (f64, f64),
}
pub struct Art {
    pub coins: HashMap<String, CoinArt>,
    px3: Hgbs,
    px6: Mlp,
    pub train_end: Option<String>,
}

fn cols(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect()
}
fn hgbs(v: &Value) -> Hgbs {
    Hgbs { models: v["models"].as_array().unwrap().iter().map(|m| serde_json::from_value(m.clone()).unwrap()).collect(), cols: cols(&v["cols"]) }
}
fn f32s(v: &Value) -> (Vec<f32>, Vec<usize>) {
    (v["b"].as_array().unwrap().iter().map(|b| f32::from_bits(b.as_u64().unwrap() as u32)).collect(), v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
}
fn mlp(v: &Value) -> Mlp {
    let layers = v["layers"].as_array().unwrap().iter().map(|seed| seed.as_array().unwrap().iter().map(|l| {
        let ((w, sh), (b, _)) = (f32s(&l["W"]), f32s(&l["b"]));
        (w, sh[0], sh[1], b)
    }).collect()).collect();
    Mlp { layers, mu: bits_vec(&v["mu"]), sd: bits_vec(&v["sd"]), ym: bits(&v["ym"]), ys: bits(&v["ys"]), cols: cols(&v["cols"]) }
}

impl Art {
    /// tools/export_qx4.py 가 쓴 JSON.
    pub fn from_json(v: &Value) -> Art {
        let coins = v["coins"].as_object().unwrap().iter().map(|(k, c)| {
            let e = bits_vec(&c["expand_logit"]);
            (k.clone(), CoinArt { m3: hgbs(&c["M3"]), m6: mlp(&c["M6"]), c: bits(&c["c"]), expand: (e[0], e[1]) })
        }).collect();
        Art { coins, px3: hgbs(&v["PX3"]), px6: mlp(&v["PX6"]), train_end: v["train_end"].as_str().map(str::to_string) }
    }
}

fn n2n(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else if v.is_infinite() { f64::MAX.copysign(v) } else { v }
}

/// _pred(HGB 셋): 시드 평균(np.mean 축 0 = 차례 합 / 개수) · 입력 nan_to_num.
fn pred_hgb(m: &Hgbs, f: &HashMap<&'static str, Vec<f64>>, i: usize) -> f64 {
    let x: Vec<f64> = m.cols.iter().map(|c| n2n(f[c.as_str()][i])).collect();
    let mut s = 0.0;
    for (k, h) in m.models.iter().enumerate() {
        let p = predict_proba(h, &x).expect("HGB");
        s = if k == 0 { p } else { s + p };
    }
    s / m.models.len() as f64
}

/// _pred(MLP): (nan_to_num(x) − mu)/sd → float32 → [W·h + b · ReLU] → 출력 × ys + ym(float32) · 시드 평균(float32).
fn pred_mlp(m: &Mlp, f: &HashMap<&'static str, Vec<f64>>, i: usize) -> f32 {
    let x: Vec<f32> = m.cols.iter().enumerate().map(|(j, c)| ((n2n(f[c.as_str()][i]) - m.mu[j]) / m.sd[j]) as f32).collect();
    let (ys, ym) = (m.ys as f32, m.ym as f32); // numpy 2: f32 배열 × 파이썬 실수 = f32(약한 형)
    let mut tot = 0f32;
    for (k, layers) in m.layers.iter().enumerate() {
        let mut h = x.clone();
        for (li, (w, out, inn, b)) in layers.iter().enumerate() {
            let mut nh = vec![0f32; *out];
            for o in 0..*out {
                let row = &w[o * inn..(o + 1) * inn];
                let mut s = 0f32;
                for (a, wv) in h.iter().zip(row) {
                    s = a.mul_add(*wv, s); // ponytail: OpenBLAS sgemm(Haswell) 커널의 k 차례 FMA 라고 가정 -- 시험이 잰다
                }
                nh[o] = s + b[o];
                if li + 1 < layers.len() && nh[o] < 0.0 {
                    nh[o] = 0.0; // np.maximum(h, 0)
                }
            }
            h = nh;
        }
        let y = h[0] * ys + ym;
        tot = if k == 0 { y } else { tot + y };
    }
    tot / m.layers.len() as f32
}

/// predict_log: 4 구성원(M3 · M6 · PX3 · PX6) 평균(로그 RV · f64).
fn predict_log(art: &Art, coin: &str, f: &HashMap<&'static str, Vec<f64>>, i: usize) -> f64 {
    let a = &art.coins[coin];
    let v = [pred_hgb(&a.m3, f, i), pred_mlp(&a.m6, f, i) as f64, pred_hgb(&art.px3, f, i), pred_mlp(&art.px6, f, i) as f64];
    (((v[0] + v[1]) + v[2]) + v[3]) / 4.0
}

/// pandas rolling(w, min_periods).sum(): Kahan(더하기·빼기 따로) · 같은 값 연속이면 값×개수.
fn roll_sum(x: &[f64], w: usize, minp: usize) -> Vec<f64> {
    let x: Vec<f64> = x.iter().map(|v| if v.is_infinite() { NAN } else { *v }).collect();
    let (mut nobs, mut sx, mut ca, mut cr, mut same) = (0i64, 0.0f64, 0.0f64, 0.0f64, 0i64);
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
        }
        let v = x[i];
        if !v.is_nan() {
            nobs += 1;
            let y = v - ca;
            let t = sx + y;
            ca = t - sx - y;
            sx = t;
            same = if v == prev { same + 1 } else { 1 };
            prev = v;
        }
        if nobs >= minp.max(1) as i64 {
            out[i] = if same >= nobs { prev * nobs as f64 } else { sx };
        }
    }
    out
}

/// 한 코인 5분 격자(h·l·c·v·n) → 피쳐(배포 22 + HAR + hod·wkend). ts = 봉 시작 ms.
fn features(ts: &[i64], h: &[f64], l: &[f64], c: &[f64], v: &[f64], nt: &[f64]) -> HashMap<&'static str, Vec<f64>> {
    let n = c.len();
    let mut f: HashMap<&'static str, Vec<f64>> = HashMap::new();
    let mx = |a: f64, b: f64| if a.is_nan() || b.is_nan() { NAN } else { a.max(b) }; // np.maximum
    // ── live_eth_sizing_vol_model.build_features
    let lcm: Vec<f64> = c.iter().map(|x| mx(*x, 1e-12).ln()).collect();
    let lr: Vec<f64> = (0..n).map(|i| if i == 0 { lcm[0] - 0.0 } else { lcm[i] - lcm[i - 1] }).collect(); // np.diff(prepend=0.0)
    let tr: Vec<f64> = (0..n).map(|i| if i == 0 { (c[0] - c[0]).abs() } else { (c[i] - c[i - 1]).abs() }).collect();
    let m288 = crate::dir4h::roll_mean_pub(&tr, 288, 72);
    f.insert("atr288", (0..n).map(|i| mx(m288[i] / mx(c[i], 1e-9), 1e-12).ln()).collect());
    for (w, nm) in [(12usize, "rv12"), (48, "rv48"), (288, "rv288")] {
        let sd = roll_std(&lr.iter().map(|x| if x.is_infinite() { NAN } else { *x }).collect::<Vec<_>>(), w, 8.max(w / 4));
        f.insert(nm, sd.iter().map(|s| mx(*s, 1e-12).ln()).collect());
    }
    f.insert("volexp", (0..n).map(|i| f["rv12"][i].exp() / mx(f["rv288"][i].exp(), 1e-12)).collect());
    let hour: Vec<f64> = ts.iter().map(|t| { let s = t.rem_euclid(86_400_000) / 1000; (s / 3600) as f64 + (s / 60 % 60) as f64 / 60.0 }).collect();
    let tau = 2.0 * std::f64::consts::PI;
    f.insert("hour_sin", hour.iter().map(|h| (tau * h / 24.0).sin()).collect());
    f.insert("hour_cos", hour.iter().map(|h| (tau * h / 24.0).cos()).collect());
    f.insert("dow", ts.iter().map(|t| (t.div_euclid(86_400_000) + 3).rem_euclid(7) as f64).collect());
    let qv: Vec<f64> = (0..n).map(|i| v[i] * c[i]).collect();
    for (arr, nm) in [(&qv, "qv_z"), (&nt.to_vec(), "nt_z")] {
        let (m, s) = (crate::dir4h::roll_mean_pub(arr, 288, 288), roll_std(&arr.iter().map(|x| if x.is_infinite() { NAN } else { *x }).collect::<Vec<_>>(), 288, 288));
        f.insert(nm, (0..n).map(|i| (arr[i] - m[i]) / s[i]).collect());
    }
    for (w, nms) in [(12usize, ["ret12", "pos_in_range12", "dist_lo12", "dist_hi12"]), (48, ["ret48", "pos_in_range48", "dist_lo48", "dist_hi48"]),
                     (144, ["ret144", "pos_in_range144", "dist_lo144", "dist_hi144"])] {
        let (rmin, rmax) = (roll_ext(l, w, w, false), roll_ext(h, w, w, true));
        f.insert(nms[0], (0..n).map(|i| if i >= w { c[i] / c[i - w] - 1.0 } else { NAN }).collect());
        f.insert(nms[1], (0..n).map(|i| (c[i] - rmin[i]) / mx(rmax[i] - rmin[i], 1e-9)).collect());
        f.insert(nms[2], (0..n).map(|i| (c[i] - rmin[i]) / mx(c[i], 1e-9)).collect());
        f.insert(nms[3], (0..n).map(|i| (rmax[i] - c[i]) / mx(c[i], 1e-9)).collect());
    }
    // ── 연구 features: HAR 원 피쳐
    let lc: Vec<f64> = c.iter().map(|x| x.ln()).collect();
    let r: Vec<f64> = (0..n).map(|i| if i == 0 { lc[0] - lc[0] } else { lc[i] - lc[i - 1] }).collect(); // prepend=log(c0)
    let r2: Vec<f64> = r.iter().map(|x| x * x).collect();
    let ar: Vec<f64> = r.iter().map(|x| x.abs()).collect();
    for w in WS {
        let s = roll_sum(&r2, w, w);
        f.insert(["lrv12", "lrv48", "lrv288", "lrv2016", "lrv8640"][WS.iter().position(|x| *x == w).unwrap()], s.iter().map(|x| mx(*x, 1e-10).ln()).collect());
    }
    for (w, nm) in [(48usize, ["rsn48", "j48", "q48", "lrvq48", "lrv48"]), (288, ["rsn288", "j288", "q288", "lrvq288", "lrv288"])] {
        let rv: Vec<f64> = roll_sum(&r2, w, w).iter().map(|x| mx(*x, 1e-12)).collect();
        let neg: Vec<f64> = (0..n).map(|i| r2[i] * (r[i] < 0.0) as u8 as f64).collect();
        let sn = roll_sum(&neg, w, w);
        let pr: Vec<f64> = (0..n).map(|i| if i == 0 { NAN } else { ar[i] * ar[i - 1] }).collect();
        let bv0 = roll_sum(&pr, w, w);
        let r4: Vec<f64> = r.iter().map(|x| x.powf(4.0)).collect(); // r ** 4 = libm pow
        let s4 = roll_sum(&r4, w, w);
        let rsn: Vec<f64> = (0..n).map(|i| sn[i] / rv[i]).collect();
        let j: Vec<f64> = (0..n).map(|i| mx(rv[i] - std::f64::consts::PI / 2.0 * bv0[i], 0.0) / rv[i]).collect();
        let q: Vec<f64> = (0..n).map(|i| (w as f64 / 3.0 * s4[i]).sqrt() / rv[i]).collect();
        let lq: Vec<f64> = (0..n).map(|i| f[nm[4]][i] * q[i]).collect();
        f.insert(nm[0], rsn);
        f.insert(nm[1], j);
        f.insert(nm[2], q);
        f.insert(nm[3], lq);
    }
    f.insert("hod", ts.iter().map(|t| ((t + BAR).rem_euclid(86_400_000) / 3_600_000) as f64).collect());
    f.insert("wkend", ts.iter().map(|t| (((t + BAR).div_euclid(86_400_000) + 3).rem_euclid(7) >= 5) as u8 as f64).collect());
    f
}

/// frames_from_klines: 완결봉만 · 4코인 공통 5분 격자(가장 늦게 시작 ~ 가장 일찍 끝) · 빈 봉 = 종가 이어 붙임·고저 = 종가·거래 0 · + 4코인 평균.
/// 반환 (격자, 코인 → 피쳐, 코인 → 종가, 코인 → 최근 60일 빈 봉 수).
#[allow(clippy::type_complexity)]
pub fn frames(rows: &HashMap<String, Vec<Value>>, now_ms: f64) -> Result<(Vec<i64>, HashMap<String, HashMap<&'static str, Vec<f64>>>, HashMap<String, Vec<f64>>, Vec<(String, usize)>), String> {
    let num = |v: &Value| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())).unwrap_or(NAN);
    let mut per: Vec<(String, std::collections::BTreeMap<i64, [f64; 5]>)> = vec![];
    for s in COINS {
        let mut m = std::collections::BTreeMap::new();
        for r in &rows[s] {
            if (r[6].as_i64().unwrap() as f64) < now_ms {
                m.insert(r[0].as_i64().unwrap(), [num(&r[2]), num(&r[3]), num(&r[4]), num(&r[5]), num(&r[8])]); // 같은 시각은 뒤 줄
            }
        }
        if m.is_empty() {
            return Err("IndexError: index 0 is out of bounds".into());
        }
        per.push((s.to_string(), m));
    }
    let a = per.iter().map(|(_, m)| *m.keys().next().unwrap()).max().unwrap();
    let b = per.iter().map(|(_, m)| *m.keys().next_back().unwrap()).min().unwrap();
    let grid: Vec<i64> = if b >= a { (0..=(b - a) / BAR).map(|k| a + k * BAR).collect() } else { vec![] };
    let (mut feats, mut closes, mut filled) = (HashMap::new(), HashMap::new(), vec![]);
    for (s, m) in &per {
        let n = grid.len();
        let (mut h, mut l, mut c, mut v, mut nt) = (vec![NAN; n], vec![NAN; n], vec![NAN; n], vec![0.0; n], vec![0.0; n]);
        let mut miss = vec![false; n];
        for (i, t) in grid.iter().enumerate() {
            match m.get(t) {
                Some(r) => (h[i], l[i], c[i], v[i], nt[i]) = (r[0], r[1], r[2], r[3], r[4]),
                None => miss[i] = true,
            }
        }
        for i in 0..n {
            if c[i].is_nan() && i > 0 {
                c[i] = c[i - 1];
            }
            if h[i].is_nan() {
                h[i] = c[i];
            }
            if l[i].is_nan() {
                l[i] = c[i];
            }
            if v[i].is_nan() {
                v[i] = 0.0;
            }
            if nt[i].is_nan() {
                nt[i] = 0.0;
            }
        }
        filled.push((s.clone(), miss[n.saturating_sub(60 * 288)..].iter().filter(|x| **x).count()));
        feats.insert(s.clone(), features(&grid, &h, &l, &c, &v, &nt));
        closes.insert(s.clone(), c);
    }
    for (k, src) in [("mkt_lrv12", "lrv12"), ("mkt_lrv48", "lrv48"), ("mkt_lrv288", "lrv288"), ("mkt_qvz", "qv_z")] {
        let avg: Vec<f64> = (0..grid.len()).map(|i| {
            let mut s = 0.0;
            for (j, c) in COINS.iter().enumerate() {
                let x = feats[*c][src][i];
                s = if j == 0 { x } else { s + x };
            }
            s / 4.0
        }).collect();
        for c in COINS {
            feats.get_mut(c).unwrap().insert(k, avg.clone());
        }
    }
    let _ = npsum; // (차례 합만 쓴다)
    Ok((grid, feats, closes, filled))
}

/// qx4_from_klines: 화면 payload(대시보드 vol4h_from_klines 와 같은 키 · 중앙값형 눈금).
pub fn payload(rows: &HashMap<String, Vec<Value>>, art: &Art, now_ms: f64, coin: &str) -> Value {
    let miss: Vec<&str> = COINS.iter().copied().filter(|s| rows.get(*s).is_none_or(|r| r.is_empty())).collect();
    if !miss.is_empty() {
        return json!({"ok": false, "reason": format!("코인 klines 빠짐 {}", miss.join(","))});
    }
    let (grid, fs, closes, filled) = match frames(rows, now_ms) {
        Ok(x) => x,
        Err(e) => return json!({"ok": false, "reason": e.chars().take(160).collect::<String>()}),
    };
    let f = &fs[coin];
    let need: Vec<&str> = M4_COLS.iter().copied().chain(MKT).collect();
    let good: Vec<usize> = (WARM.min(grid.len())..grid.len()).filter(|&i| need.iter().all(|c| f[*c][i].is_finite())).collect();
    if good.len() < REF_BARS {
        return json!({"ok": false, "reason": format!("봉 부족({} < {REF_BARS}, 코인마다 {NEED_BARS}봉 필요)", good.len())});
    }
    if filled.iter().any(|x| x.1 > MAX_FILLED) {
        let d: Vec<String> = filled.iter().map(|(s, k)| format!("'{s}': {k}")).collect();
        return json!({"ok": false, "reason": format!("빈 봉 많음 {{{}}}", d.join(", "))});
    }
    let last = *good.last().unwrap();
    let asof = grid[last] + BAR;
    if now_ms - asof as f64 > STALE_MS {
        return json!({"ok": false, "reason": format!("지연 {}분", ((now_ms - asof as f64) / 60000.0) as i64)});
    }
    let a = &art.coins[coin];
    let tail = &good[good.len() - REF_BARS..];
    let p: Vec<f64> = tail.iter().map(|&i| predict_log(art, coin, f, i)).collect();
    let sig: Vec<f64> = p.iter().map(|x| (x / 2.0).exp()).collect();
    let s4 = sig[sig.len() - 1];
    let mut so = sig.clone();
    so.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let m = so.len();
    let refm = if m % 2 == 1 { so[m / 2] } else { (so[m / 2 - 1] + so[m / 2]) / 2.0 }; // np.median
    let now4 = (f["lrv48"][last] / 2.0).exp();
    let px = closes[coin][last];
    let (a0, a1) = a.expand;
    let sm = s4 * (a.c / 2.0).exp();
    json!({"ok": true, "asof_ms": asof, "px": px, "sig4h_pct": py_round(100.0 * s4, 3), "sig4h_usd": py_round(px * s4, if px >= 10.0 { 2 } else { 4 }),
           "lvl": py_round(s4 / refm, 3), "x_now": py_round(s4 / now4, 3), "p_expand": py_round(1.0 / (1.0 + (-(a0 + a1 * (s4 / now4).ln())).exp()), 3),
           "ref_days": py_round(tail.len() as f64 / 288.0, 1), "sig4h_mean_pct": py_round(100.0 * sm, 3), "filled_bars_60d": filled.iter().find(|x| x.0 == coin).unwrap().1,
           "model": "QX4", "model_train_end": art.train_end, "log_rv": p[p.len() - 1]})
}

/// 연구 SETS["M4"] = 배포 22 + HAR 13 + hod·wkend.
const M4_COLS: [&str; 37] = ["atr288", "rv12", "rv48", "rv288", "volexp", "hour_sin", "hour_cos", "dow", "qv_z", "nt_z",
    "ret12", "pos_in_range12", "dist_lo12", "dist_hi12", "ret48", "pos_in_range48", "dist_lo48", "dist_hi48", "ret144", "pos_in_range144", "dist_lo144", "dist_hi144",
    "lrv12", "lrv48", "lrv288", "lrv2016", "lrv8640", "rsn48", "rsn288", "j48", "j288", "q48", "q288", "lrvq48", "lrvq288", "hod", "wkend"];

extern "C" {
    fn erf(x: f64) -> f64; // glibc libm(파이썬 math.erf 와 같은 함수)
}

/// atm_iv: [d−1h, d) 옵션 체결 · 블록·청산 빼고 · BS 델타(r=0 · 만기 08:00 UTC · 연 365.25일) |0.35~0.65| · 만기까지 1시간~10일 → iv 중앙값.
pub fn atm_iv(trades: &[Value], d_ms: i64, coin: &str) -> Option<f64> {
    let mut ivs = vec![];
    for t in trades {
        let ts = t["timestamp"].as_i64()?;
        let truthy = |k: &str| t.get(k).is_some_and(|v| !(v.is_null() || v == &json!(false) || v == &json!("") || v == &json!(0)));
        let Some(iv) = t.get("iv").and_then(Value::as_f64) else { continue };
        let name = t["instrument_name"].as_str().unwrap_or("");
        let Some((exp_s, strike, cp)) = parse_inst(name, coin) else { continue };
        if !(d_ms - 3_600_000 <= ts && ts < d_ms) || truthy("block_trade_id") || truthy("liquidation") || iv.is_nan() || iv <= 0.0 {
            continue;
        }
        let days = ((exp_s * 1000 + 8 * 3_600_000) - ts) as f64 / 86_400_000.0;
        if !(1.0 / 24.0 <= days && days <= 10.0) {
            continue;
        }
        let (tt, sig) = (days / 365.25, iv / 100.0);
        let idx = t["index_price"].as_f64()?;
        let d = 0.5 * (1.0 + unsafe { erf(((idx / strike).ln() + 0.5 * sig.powf(2.0) * tt) / (sig * tt.sqrt()) / 2f64.sqrt()) }) - (cp == 'P') as u8 as f64;
        if (0.35..=0.65).contains(&d.abs()) {
            ivs.push(iv);
        }
    }
    if ivs.is_empty() {
        return None;
    }
    ivs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ivs.len();
    Some(if n % 2 == 1 { ivs[n / 2] } else { (ivs[n / 2 - 1] + ivs[n / 2]) / 2.0 })
}

/// ^{coin}-(\d+[A-Z]{3}\d{2})-(\d+)-([CP])$ → (만기일 00:00 UTC 초, 행사가, C|P). strptime("%d%b%y")(달 이름은 대소문자 무시 · %y 00~68 = 20xx).
fn parse_inst(s: &str, coin: &str) -> Option<(i64, f64, char)> {
    let rest = s.strip_prefix(coin)?.strip_prefix('-')?;
    let mut it = rest.split('-');
    let (e, k, cp) = (it.next()?, it.next()?, it.next()?);
    if it.next().is_some() || cp.len() != 1 || !matches!(cp, "C" | "P") || k.is_empty() || !k.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let nd = e.bytes().take_while(u8::is_ascii_digit).count();
    if nd == 0 || e.len() != nd + 5 || !e[nd..nd + 3].bytes().all(|b| b.is_ascii_uppercase()) || !e[nd + 3..].bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let day: i64 = e[..nd].parse().ok()?;
    let mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"].iter().position(|m| *m == &e[nd..nd + 3])? as i64 + 1;
    let yy: i64 = e[nd + 3..].parse().ok()?;
    let y = if yy <= 68 { 2000 + yy } else { 1900 + yy };
    if nd > 2 || !(1..=31).contains(&day) {
        return None; // ponytail: strptime 은 없는 날(2월 30일)을 예외로 낸다 -- 원본은 그 체결 하나에서 죽는다(여기선 건너뜀)
    }
    let dfc = |y: i64, m: i64, d: i64| {
        let y = if m <= 2 { y - 1 } else { y };
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
        era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
    };
    Some((dfc(y, mon, day) * 86400, k.parse().ok()?, cp.chars().next()?))
}

/// qx4i_log + 서버 qx4_payload 의 덧붙이기: ŷ + a + b·(ln(IV²·4/8766) − ŷ) → iv_atm·iv_hour_ms·sig4h_iv_pct·sig4h_iv_usd·log_rv_iv.
pub fn add_iv(res: &mut Value, coin: &str, iv: f64, hour_ms: i64) {
    let Some((_, a, b)) = BETA_I.iter().find(|x| x.0 == coin) else { return };
    let p = res["log_rv"].as_f64().unwrap();
    let x1 = ((iv / 100.0).powf(2.0) * 4.0 / 8766.0).ln() - p;
    let lq = p + a + b * x1;
    let s = (lq / 2.0).exp();
    let px = res["px"].as_f64().unwrap();
    let o = res.as_object_mut().unwrap();
    o.insert("iv_atm".into(), json!(iv));
    o.insert("iv_hour_ms".into(), json!(hour_ms));
    o.insert("sig4h_iv_pct".into(), json!(py_round(100.0 * s, 3)));
    o.insert("sig4h_iv_usd".into(), json!(py_round(px * s, 2)));
    o.insert("log_rv_iv".into(), json!(lq));
}

pub fn has_iv(coin: &str) -> bool {
    BETA_I.iter().any(|x| x.0 == coin)
}

#[doc(hidden)]
pub fn debug_members(art: &Art, coin: &str, f: &HashMap<&'static str, Vec<f64>>, i: usize) -> [f64; 4] {
    let a = &art.coins[coin];
    [pred_hgb(&a.m3, f, i), pred_mlp(&a.m6, f, i) as f64, pred_hgb(&art.px3, f, i), pred_mlp(&art.px6, f, i) as f64]
}

// ── REST 이어 받기(원본 refresh_rows·merge_klines · 서버 qx4_payload 가 요청마다 부름) ─────────────────────────────────────────
/// 원본 SYMS 순서(ETH·XRP·BTC·SOL).
pub const SYMS: [(&str, &str); 4] = [("ETH", "ETHUSDT"), ("XRP", "XRPUSDT"), ("BTC", "BTCUSDT"), ("SOL", "SOLUSDT")];

/// merge_klines: 최근 1000봉(got)을 기존 행 뒤에 잇는다(겹침은 got 쪽) · 이어지지 않으면 None · 끝 keep 개만.
fn merge(have: &[Value], got: &[Value], keep: usize) -> Option<Vec<Value>> {
    if have.is_empty() || got.is_empty() {
        return None;
    }
    let first = got[0][0].as_i64()?;
    if first > have[have.len() - 1][0].as_i64()? + BAR {
        return None;
    }
    let mut out: Vec<Value> = have.iter().filter(|r| r[0].as_i64().is_some_and(|t| t < first)).cloned().collect();
    out.extend(got.iter().cloned());
    let n = out.len();
    Some(out.split_off(n.saturating_sub(keep)))
}

/// refresh_rows: 코인마다 최근 1000봉을 이어 붙이고, 처음이거나 끊겼으면 NEED_BARS(+300)까지 endTime 으로 거슬러 받는다. 하나라도 빈 응답이면 false.
/// fetch(쿼리) → 행 목록(실패·빈 응답 = 빈 목록). 쿼리 순서 = 원본 dict(symbol·interval·limit·endTime).
pub fn refresh(fetch: &mut dyn FnMut(&[(&'static str, String)]) -> Vec<Value>, rows: &mut HashMap<String, Vec<Value>>) -> bool {
    for (coin, sym) in SYMS {
        let got = fetch(&[("symbol", sym.into()), ("interval", "5m".into()), ("limit", "1000".into())]);
        if got.is_empty() {
            return false;
        }
        let merged = merge(rows.get(coin).map_or(&[][..], |v| v.as_slice()), &got, NEED_BARS + 2000);
        let merged = match merged {
            Some(m) if m.len() >= NEED_BARS => m,
            _ => {
                let mut out = got.clone();
                let mut end = got[0][0].as_i64().unwrap() - 1;
                while out.len() < NEED_BARS + 300 {
                    let part = fetch(&[("symbol", sym.into()), ("interval", "5m".into()), ("limit", "1000".into()), ("endTime", end.to_string())]);
                    if part.is_empty() {
                        return false;
                    }
                    end = part[0][0].as_i64().unwrap() - 1;
                    out = part.into_iter().chain(out).collect();
                }
                out
            }
        };
        rows.insert(coin.into(), merged);
    }
    true
}
