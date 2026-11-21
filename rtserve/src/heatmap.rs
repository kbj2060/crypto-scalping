//! /api/flow/heatmap 계산(2026-11-22 · 여섯째 조각). 원본(전부 Read 로 확인한 줄):
//!   scripts/live_orderflow_raster_collector_20260914.py row_stats(185~224) · approach_ratio(227~276) · vol_scores(279~324) · read_window(→ raster_read.rs)
//!   dashboard/server.py heatmap_obi(1136~1147) · _heatmap_read(4482~4521) · _heatmap_rows(4523~4627) · _volsig_ref(4629) · _volsig_read(4649) · _approach_read(4672) · api_flow_heatmap(api3.rs)
//! numpy 2.2.5 규칙(서버 판 · 정답지로 비트 확인): 축 0 줄이기(mean·sum·min·max·cumsum) = 행 차례 f32 · 1차원 연속 np.sum = 8192칸 묶음 짝 합 ·
//! np.percentile(f32) = q·지표·가중치 전부 f32(q = 97f32/100f32 · vi = (n−1)·q · 선형 보간 b−a 를 f32 로) · NEP 50(f32 배열 ⊕ 파이썬 실수 = f32).

use serde_json::{json, Map, Value};

use crate::raster_read::{read_window, Window, N_BINS};

/// numpy pairwise_sum(f32 · 8192칸 묶음) -- rtcollect::ob5m::np_sum 의 f32 판
pub fn np_sum_f32(a: &[f32]) -> f32 {
    fn pw(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            return a.iter().fold(0f32, |s, v| s + v);
        }
        if n <= 128 {
            let mut r = [0f32; 8];
            r.copy_from_slice(&a[..8]);
            let m = n - n % 8;
            let mut i = 8;
            while i < m {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for v in &a[m..] {
                res += v;
            }
            return res;
        }
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pw(&a[..n2]) + pw(&a[n2..])
    }
    if a.len() <= 8192 {
        return pw(a);
    }
    a.chunks(8192).fold(0f32, |s, c| s + pw(c))
}

/// np.percentile(f32 배열, 97) -- 빈 배열은 부르지 않는다
pub fn percentile_f32(v: &mut [f32], pct: f32) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let vi = (n as f32 - 1.0) * (pct / 100f32);
    let lo = (vi.floor() as i64).clamp(0, n as i64 - 1) as usize;
    let hi = (lo + 1).min(n - 1);
    let g = vi - lo as f32;
    let (a, b) = (v[lo], v[hi]);
    let d = b - a;
    if g >= 0.5 { b - d * (1f32 - g) } else { a + d * g }
}

/// 파이썬 round(x)(짝수 쪽) -- 작은 양수만
fn pyround(x: f64) -> i64 {
    x.round_ties_even() as i64
}

pub struct Stats {
    pub cols: [(&'static str, Vec<f32>); 7],
}

/// row_stats(a, dt_s): a = 절대값 잔량(시간 t × 칸 b · 행 우선)
pub fn row_stats(a: &[f32], t: usize, b: usize, dt_s: i64) -> Result<Stats, String> {
    if t == 0 {
        return Err("ValueError: a must be (time, bins) with time > 0".into());
    }
    if t == 1 {
        return Err("IndexError: index -1 is out of bounds for axis 0 with size 0".into()); // csum[-1] (d 가 빈 배열)
    }
    let at = |i: usize, j: usize| a[i * b + j];
    let n60 = pyround(60.0 / dt_s.max(1) as f64).max(1) as usize;
    let mean = |r0: usize, r1: usize, j: usize| {
        let mut s = 0f32;
        for i in r0..r1 {
            s += at(i, j);
        }
        s / (r1 - r0) as f32
    };
    let mut out: [Vec<f32>; 7] = Default::default();
    for j in 0..b {
        let d60 = if t >= 6 * n60 { mean(t - n60, t, j) - mean(t - 6 * n60, t - n60, j) } else { 0.0 };
        let mut up: Vec<f32> = (0..t - 1).map(|i| (at(i + 1, j) - at(i, j)).max(0.0)).collect();
        let refill = up.iter().fold(0f32, |s, v| s + v);
        let n_up = up.iter().filter(|v| **v > 0.0).count() as f32;
        up.sort_by(|x, y| x.total_cmp(y));
        let mut cs = 0f32;
        let csum: Vec<f32> = up.iter().map(|v| {
            cs += v;
            cs
        }).collect();
        let thr = csum[csum.len() - 1] * 0.5;
        let idx = csum.iter().position(|c| *c >= thr).unwrap_or(0);
        let (mut mn, mut mx) = (at(0, j), at(0, j));
        for i in 1..t {
            mn = mn.min(at(i, j));
            mx = mx.max(at(i, j));
        }
        for (k, v) in [at(t - 1, j), mn, mx, refill, d60, up[idx], n_up].into_iter().enumerate() {
            out[k].push(v);
        }
    }
    let [i, p, pk, r, d, bl, nu] = out;
    Ok(Stats { cols: [("inst", i), ("pers", p), ("peak", pk), ("refill", r), ("d60", d), ("blk", bl), ("n_up", nu)] })
}

/// approach_ratio(w) → (ratio(칸마다 · 자격 없으면 NaN), 자격 칸 수)
pub fn approach_ratio(w: &Window) -> (Vec<f32>, i64) {
    let n = w.n_bins;
    let mut out = vec![f32::NAN; n];
    let okr: Vec<usize> = (0..w.mid.len()).filter(|&i| w.mid[i].is_finite() && w.mid[i] > 0.0).collect();
    if okr.len() < 10 || n == 0 {
        return (out, 0);
    }
    let price: Vec<f32> = (0..n).map(|j| ((w.bin_lo + j as i64) as f64 * w.bin_size) as f32).collect();
    let edges: Vec<f32> = (0..51).map(|i| i as f32 * 0.05f32).collect(); // np.arange(0, 2.5001, 0.05, f32)
    let k = 50usize;
    let t = okr.len();
    let mut dm = vec![0f32; t * n];
    let mut bkt = vec![0usize; t * n];
    let mut sums = vec![0f64; t * k];
    let mut cnts = vec![0i64; t * k];
    for (r, &i) in okr.iter().enumerate() {
        let md = w.mid[i];
        for j in 0..n {
            let x = (price[j] - md).abs() / md * 100.0f32;
            dm[r * n + j] = x;
            let dg = edges.partition_point(|e| *e <= x); // np.digitize(右 = False) = searchsorted side right
            let bk = (dg as i64 - 1).clamp(0, k as i64 - 1) as usize;
            bkt[r * n + j] = bk;
            sums[r * k + bk] += w.qty[i * n + j].abs() as f64;
            cnts[r * k + bk] += 1;
        }
    }
    let need = pyround(60.0 / (w.dt_s as f64).max(1.0)).max(1);
    let (mut num, mut den) = (vec![0f64; n], vec![0f64; n]);
    let (mut nn, mut nf) = (vec![0i64; n], vec![0i64; n]);
    let mut amax = vec![f32::NEG_INFINITY; n];
    for (r, &i) in okr.iter().enumerate() {
        for j in 0..n {
            let a = w.qty[i * n + j].abs();
            amax[j] = amax[j].max(a);
            let bk = bkt[r * n + j];
            let norm = sums[r * k + bk] / (cnts[r * k + bk].max(1) as f64);
            let rel = a as f64 / norm.max(1e-9);
            let x = dm[r * n + j];
            let near = x < 0.35f32;
            let far = x >= 0.35f32 && x < 1.2f32;
            num[j] += if near { rel } else { 0.0 };
            den[j] += if far { rel } else { 0.0 };
            nn[j] += near as i64;
            nf[j] += far as i64;
        }
    }
    let mut q = 0;
    for j in 0..n {
        let (a, b) = (num[j] / nn[j].max(1) as f64, den[j] / nf[j].max(1) as f64);
        if nn[j] >= need && nf[j] >= need && amax[j] > 0.0 && b > 0.0 {
            out[j] = (a / b) as f32;
            q += 1;
        }
    }
    (out, q)
}

/// vol_scores(w, idx) → (past_bp, refill) -- idx None = 마지막 초 하나
pub fn vol_scores(w: &Window, idx: Option<Vec<usize>>) -> (Vec<f32>, Vec<f32>) {
    let dt = w.dt_s.max(1) as usize;
    let (lb, pb) = (600 / dt, 300 / dt);
    let n = w.n_bins;
    let len = w.mid.len();
    if n == 0 || len <= lb {
        return (vec![], vec![]);
    }
    let price: Vec<f64> = (0..n).map(|j| (w.bin_lo + j as i64) as f64 * w.bin_size).collect();
    let ok = |i: usize| w.mid[i].is_finite() && w.mid[i] > 0.0;
    let (mut past, mut refill) = (vec![], vec![]);
    for i in idx.unwrap_or_else(|| vec![len - 1]) {
        if i < lb || i >= len || !ok(i) || !ok(i - pb) {
            continue;
        }
        let m = w.mid[i] as f64;
        let rows = i - lb..i + 1;
        let mut peak = vec![f32::NEG_INFINITY; n];
        for r in rows.clone() {
            for j in 0..n {
                peak[j] = peak[j].max(w.qty[r * n + j].abs());
            }
        }
        let sel: Vec<usize> = (0..n).filter(|&j| ((price[j] - m) / m * 100.0).abs() <= 0.5 && peak[j] > 0.0).collect();
        if sel.len() < 5 {
            continue;
        }
        // win[:, sel] 은 numpy 가 열 우선(F)으로 만든다 → up.sum() = 열 우선 차례로 8192칸 묶음 짝 합(서버 판 numpy 60/60 확인)
        let mut up = Vec::with_capacity(lb * sel.len());
        for &j in &sel {
            for r in i - lb..i {
                up.push((w.qty[(r + 1) * n + j].abs() - w.qty[r * n + j].abs()).max(0.0));
            }
        }
        let ps = np_sum_f32(&sel.iter().map(|&j| peak[j]).collect::<Vec<_>>()) as f64;
        past.push(((m - w.mid[i - pb] as f64).abs() / m * 1e4) as f32);
        refill.push((np_sum_f32(&up) / ps.max(1e-9) as f32) as f64 as f32);
    }
    (past, refill)
}

/// heatmap_obi(last_qty, price, center)
pub fn heatmap_obi(last: &[f32], price: &[f64], center: f64) -> f64 {
    let (mut b, mut k) = (vec![], vec![]);
    for (q, p) in last.iter().zip(price) {
        if ((p - center) / center * 100.0).abs() <= 0.5 {
            b.push((*q as f64).max(0.0));
            k.push((-*q as f64).max(0.0));
        }
    }
    let (b, k) = (rtcollect::ob5m::np_sum(&b), rtcollect::ob5m::np_sum(&k));
    crate::py::round_nd((b - k) / (b + k).max(1e-9), 3)
}

pub fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            s.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

pub fn f4(a: &[f32]) -> String {
    b64(&a.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

fn f(x: f64) -> Value {
    crate::py::f(x)
}

/// 풋프린트 칸(체결 대조): 봉 시작 초 → [(가격, 테이커 매수, 테이커 매도)] -- 허브 풋프린트 칸(가격 = round(키 × 칸 폭, dp))
pub type Fp = Vec<(i64, Vec<(f64, f64, f64)>)>;

/// _heatmap_rows(w, qty, mid, symbol) -- fp None = 그 코인 흐름 엔진 없음(원본 LookupError → fill_source None)
pub fn heatmap_rows(w: &Window, fp: Option<&Fp>) -> Result<(Value, Value), String> {
    let okr: Vec<usize> = (0..w.mid.len()).filter(|&i| w.mid[i].is_finite()).collect();
    if okr.is_empty() {
        return Ok((Value::Null, Value::Null));
    }
    let n = w.n_bins;
    let a: Vec<f32> = okr.iter().flat_map(|&i| w.qty[i * n..(i + 1) * n].iter().map(|x| x.abs())).collect();
    let st = row_stats(&a, okr.len(), n, w.dt_s)?;
    let mut mids: Vec<f32> = okr.iter().map(|&i| w.mid[i]).collect();
    mids.sort_by(|x, y| x.total_cmp(y));
    let h = mids.len() / 2;
    let spot = if mids.len() % 2 == 1 { mids[h] } else { (mids[h - 1] + mids[h]) / 2f32 } as f64;
    let price: Vec<f64> = (0..n).map(|j| (w.bin_lo + j as i64) as f64 * w.bin_size).collect();
    let (m0, m1) = (w.mid[okr[0]] as f64, w.mid[*okr.last().unwrap()] as f64);
    let cov_half = 0.4 * N_BINS as f64 * w.bin_size;
    let cov: Vec<bool> = price.iter().map(|p| (p - m0).abs() < cov_half && (p - m1).abs() < cov_half).collect();
    let same_bid: Vec<bool> = (0..n).map(|j| cov[j] && price[j] < m0 && price[j] < m1).collect();
    let same_ask: Vec<bool> = (0..n).map(|j| cov[j] && price[j] > m0 && price[j] > m1).collect();
    let (r0, r1) = (okr[0], *okr.last().unwrap());
    let bid_drop: Vec<f32> = (0..n).map(|j| (w.qty[r0 * n + j].max(0.0) - w.qty[r1 * n + j].max(0.0)).max(0.0)).collect();
    let ask_drop: Vec<f32> = (0..n).map(|j| ((-w.qty[r0 * n + j]).max(0.0) - (-w.qty[r1 * n + j]).max(0.0)).max(0.0)).collect();
    let (mut fill_bid, mut fill_ask) = (vec![0f64; n], vec![0f64; n]);
    if let Some(fp) = fp {
        let t_start = w.t0_ms as f64 / 1000.0;
        let t_end = t_start + (w.cols as i64 * w.dt_s) as f64;
        for (bar, cells) in fp {
            if ((bar + 300) as f64) < t_start || (*bar as f64) > t_end {
                continue;
            }
            for (px, buy, sell) in cells {
                let i = (px / w.bin_size).round_ties_even() as i64 - w.bin_lo;
                if 0 <= i && (i as usize) < n {
                    fill_ask[i as usize] += buy;
                    fill_bid[i as usize] += sell;
                }
            }
        }
    }
    let pick = |v: &[f32], m: &[bool]| (0..n).filter(|&j| m[j]).map(|j| v[j]).collect::<Vec<f32>>();
    let drop = (np_sum_f32(&pick(&bid_drop, &same_bid)) + np_sum_f32(&pick(&ask_drop, &same_ask))) as f64;
    let mn = |d: &[f32], fl: &[f64], m: &[bool]| (0..n).filter(|&j| m[j]).map(|j| (d[j] as f64).min(fl[j])).collect::<Vec<f64>>();
    let filled = rtcollect::ob5m::np_sum(&mn(&bid_drop, &fill_bid, &same_bid)) + rtcollect::ob5m::np_sum(&mn(&ask_drop, &fill_ask, &same_ask));
    let leave = if fp.is_some() && drop > 0.0 { f(crate::py::round_nd(1.0 - filled / drop, 3)) } else { Value::Null };
    let (inst, pers) = (&st.cols[0].1, &st.cols[1].1);
    let isum = np_sum_f32(inst);
    let ps = np_sum_f32(pers) / if (isum as f64) > 1e-9 { isum } else { 1e-9f32 };
    let mut rows = Map::new();
    rows.insert("bin_lo".into(), json!(w.bin_lo));
    rows.insert("bin_size".into(), f(w.bin_size));
    for (k, v) in &st.cols {
        rows.insert(format!("{k}_f4"), json!(f4(v)));
    }
    let last = &w.qty[r1 * n..(r1 + 1) * n];
    let summary = json!({"spot": f(crate::py::round_nd(spot, 6)), "persist_share": f(crate::py::round_nd(ps as f64, 3)), "offtouch_leave_share": leave,
        "fill_source": if fp.is_some() { json!("footprint") } else { Value::Null },
        "offtouch_bins": same_bid.iter().filter(|x| **x).count() + same_ask.iter().filter(|x| **x).count(),
        "obi": f(heatmap_obi(last, &price, m1)), "obi_band_pct": 0.5, "window_s": w.cols as i64 * w.dt_s});
    Ok((Value::Object(rows), summary))
}

/// _heatmap_read(symbol, cols, agg, rows_only) -- 원본처럼 dict 순서: read_window 칸(qty·mid 뺌) · scale · rows · summary · (book | qty_i8·mid_b64)
pub fn heatmap_read(root: &std::path::Path, symbol: &str, now_ms: i64, cols: usize, agg: usize, rows_only: bool, fp: Option<&Fp>) -> Result<Value, String> {
    let w = read_window(root, symbol, now_ms, cols, agg);
    let mut nz: Vec<f32> = w.qty.iter().filter(|x| **x != 0.0).map(|x| x.abs()).collect();
    let mut scale = if nz.is_empty() { 1.0 } else { percentile_f32(&mut nz, 97.0) as f64 };
    if !(scale > 0.0) {
        scale = 1.0;
    }
    let (rows, summary) = heatmap_rows(&w, fp)?;
    let mut out = Map::new();
    out.insert("symbol".into(), json!(w.symbol));
    out.insert("t0_ms".into(), json!(w.t0_ms));
    out.insert("dt_s".into(), json!(w.dt_s));
    out.insert("cols".into(), json!(w.cols));
    out.insert("bin_size".into(), f(w.bin_size));
    out.insert("bin_lo".into(), json!(w.bin_lo));
    out.insert("n_bins".into(), json!(w.n_bins));
    out.insert("valid_ratio".into(), f(w.valid_ratio));
    out.insert("scale".into(), f(scale));
    out.insert("rows".into(), rows);
    out.insert("summary".into(), summary);
    if rows_only {
        let b = read_window(root, symbol, now_ms, 3, 1);
        if let Some(i) = (0..b.mid.len()).rev().find(|&i| b.mid[i].is_finite()).filter(|_| b.n_bins > 0) {
            out.insert("book".into(), json!({"t_ms": b.t0_ms + i as i64 * 1000, "bin_lo": b.bin_lo, "bin_size": f(b.bin_size), "mid": f(b.mid[i] as f64), "q_f4": f4(b.row(i))}));
        }
        return Ok(Value::Object(out));
    }
    let s = scale as f32;
    let q8: Vec<u8> = w.qty.iter().map(|q| ((q / s).clamp(-1.0, 1.0) * 127.0f32).round_ties_even() as i8 as u8).collect();
    out.insert("qty_i8".into(), json!(b64(&q8)));
    out.insert("mid_b64".into(), json!(f4(&w.mid)));
    Ok(Value::Object(out))
}

/// _approach_read: 4시간(2880열 × 5초 접기) → rows 에 덧붙일 칸
pub fn approach_read(root: &std::path::Path, symbol: &str, now_ms: i64) -> Map<String, Value> {
    let w = read_window(root, symbol, now_ms, 2880, 5);
    let (ratio, q) = approach_ratio(&w);
    let mut m = Map::new();
    m.insert("approach_bin_lo".into(), json!(if w.n_bins == 0 { 0 } else { w.bin_lo }));
    m.insert("approach_bin_size".into(), f(w.bin_size));
    m.insert("approach_qualifying".into(), json!(q));
    m.insert("approach_window_s".into(), json!(4 * 3600));
    m.insert("approach_f4".into(), json!(f4(&ratio)));
    m
}

/// _volsig_ref → 정렬한 (past, refill, score) · 60개 미만이면 None(원본 {})
pub fn volsig_ref(root: &std::path::Path, symbol: &str, now_ms: i64) -> Option<(Vec<f32>, Vec<f32>, Vec<f64>)> {
    let w = read_window(root, symbol, now_ms, 4 * 3600, 1);
    let (past, refill) = vol_scores(&w, Some((600..4 * 3600).step_by(60).collect()));
    if past.len() < 60 {
        return None;
    }
    let (mut sp, mut sr) = (past.clone(), refill.clone());
    sp.sort_by(|a, b| a.total_cmp(b));
    sr.sort_by(|a, b| a.total_cmp(b));
    let ss = |s: &[f32], x: f32| s.partition_point(|v| *v < x) as f64;
    let mut sc: Vec<f64> = past.iter().zip(&refill).map(|(p, r)| ss(&sp, *p) / sp.len() as f64 + ss(&sr, *r) / sr.len() as f64).collect();
    sc.sort_by(|a, b| a.total_cmp(b));
    Some((sp, sr, sc))
}

/// _volsig_read(symbol, ref) → summary 에 덧붙일 칸(없으면 빈 맵)
pub fn volsig_read(root: &std::path::Path, symbol: &str, now_ms: i64, rf: &(Vec<f32>, Vec<f32>, Vec<f64>)) -> Map<String, Value> {
    let w = read_window(root, symbol, now_ms, 660, 1);
    let (past, refill) = vol_scores(&w, None);
    let mut m = Map::new();
    if past.is_empty() {
        return m;
    }
    let pp = rf.0.partition_point(|v| *v < past[0]) as f64 / rf.0.len() as f64;
    let rp = rf.1.partition_point(|v| *v < refill[0]) as f64 / rf.1.len() as f64;
    let sc = rf.2.partition_point(|v| *v < pp + rp) as f64 / rf.2.len() as f64;
    m.insert("vol_pct".into(), json!(pyround(100.0 * sc)));
    m.insert("vol_past_pct".into(), json!(pyround(100.0 * pp)));
    m.insert("vol_refill_pct".into(), json!(pyround(100.0 * rp)));
    m.insert("vol_ref_n".into(), json!(rf.2.len()));
    m
}
