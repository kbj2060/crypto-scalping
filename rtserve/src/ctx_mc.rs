//! /api/market-context(시장 맥락 카드) -- 2026-11-22 옮김. 원본(Read 로 본 실제 줄): server.py market_context_payload 2601~2708 ·
//! api_market_context 2717~2732(3초 · 120초 안 묵은 값) · _mc_collectors 2534~2556 · _mc_history 2558~2583 · _mc_liq_profile 2585~2595 ·
//! _mc_profile 2597~2599 → tape_levels 931~969 · _px_scale 2532 · MC_LIQ_PROFILE_S 2525 · HL_CTX_DB_PATH·OI_LSRATIO_DB_PATH 2523~2524 ·
//! LIQ_BURST_60S_USD 369 · own_db_rows 1022~1030 · dashboard/market_ctx.py pct_rank 30~35 · quad_1h 64~77 · oi_stats 83~104 · _oi_change_z 107~126 ·
//! oi_series_live 129~134 · oi_sum_series 137~156 · ring_at 159~166 · lev_state 169~185 · hl_liq_levels 188~207 · liq_profile 210~217 ·
//! value_area 262~277 · profile_nodes 280~301 · 상수 20~27 · micro_ref.deriv_from_ring 263~279(= rtcore::situation::deriv_from_ring).
//! 재료: 메모리 몫(미시 payload · x · evidence · 마크 링 · 미드 링 · depth25 링 · oi_1s 링 · mp_state.next_funding_ms) + 파일 몫(아래 읽기 함수) + REST 몫
//! (펀딩 이력·DVOL -- `ctx_rest::Fetch` 로 · `RTSERVE_REST` 있을 때만 · 없으면 원본 실패 갈래 = 빈 목록). 캐시 수명은 `TTL` 표.
use crate::py::{round0, round_nd};
use serde_json::{json, Map, Value as V};
use std::collections::BTreeMap;
use std::path::Path;

pub const SWEEP_BPS: [i64; 3] = [25, 50, 100];
pub const HL_LIQ_BIN_USD: f64 = 5.0;
pub const HL_LIQ_MAX_PCT: f64 = 0.10;
pub const LIQ_PROFILE_BIN_USD: f64 = 2.5;
pub const PROFILE_BIN_FRAC: f64 = 5e-4;
pub const OI_Z_MAD_SCALE: f64 = 1.55;
pub const MC_LIQ_PROFILE_S: f64 = 12.0 * 3600.0;
/// (키, ttl, max_stale) -- 2617~2629.
pub const TTL: [(&str, f64, f64); 6] = [("mc_collectors", 15.0, 120.0), ("mc_history", 300.0, 1800.0), ("mc_liq_profile", 60.0, 600.0),
    ("mc_funding_hist", 3600.0, 21600.0), ("mc_dvol_year", 3600.0, 21600.0), ("mc_profile", 300.0, 1800.0)];
pub const MC_RESPONSE_S: f64 = 3.0;
pub const MC_STALE_MAX_S: f64 = 120.0;

/// LIQ_BURST_60S_USD(369) = (round(expm1(12.732)), round(expm1(13.21))).
pub fn liq_burst() -> [i64; 2] {
    [12.732f64.exp_m1().round_ties_even() as i64, 13.21f64.exp_m1().round_ties_even() as i64]
}

fn fin(x: f64) -> bool {
    x.is_finite()
}
fn r0(x: f64) -> i64 {
    round0(x).unwrap_or(0)
}

/// market_ctx.pct_rank(30~35).
pub fn pct_rank(x: Option<f64>, hist: &[f64], min_n: usize) -> Option<f64> {
    let a: Vec<f64> = hist.iter().copied().filter(|v| fin(*v)).collect();
    let x = x.filter(|x| fin(*x))?;
    (a.len() >= min_n).then(|| a.iter().filter(|v| **v <= x).count() as f64 / a.len() as f64)
}

/// quad_1h(64~77).
pub fn quad_1h(m: Option<f64>, doi: Option<f64>, p75: Option<f64>) -> V {
    let (Some(m), Some(doi)) = (m, doi) else { return V::Null };
    let q = |k: &str, l: &str, n: &str| json!({"key": k, "label": l, "note": n});
    if p75.is_none_or(|p| m.abs() < p) {
        q("small", "이동이 작다 — 판정 보류", "1시간 이동이 하루 상위 25% 밖")
    } else if m < 0.0 && doi > 0.0 {
        q("dn_up", "하락 + OI↑ · 새 숏 유입", "숏은 서두르지 않는 쪽(근거)")
    } else if m < 0.0 {
        q("dn_dn", "하락 + OI↓ · 롱 정리", "숏은 거둘 쪽(근거)")
    } else if doi > 0.0 {
        q("up_up", "상승 + OI↑ · 새 롱 유입", "미측정 · 서술")
    } else {
        q("up_dn", "상승 + OI↓ · 숏 커버", "미측정 · 서술")
    }
}

fn median(v: &[f64]) -> f64 {
    rtcollect::ob5m::np_median(&mut v.to_vec())
}
fn np_std(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let m = rtcollect::ob5m::np_sum(a) / n;
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    (rtcollect::ob5m::np_sum(&sq) / n).sqrt()
}

/// _oi_change_z(107~126).
fn oi_change_z(grid: &[f64], k: usize, t0: i64, step: i64) -> Option<f64> {
    let d: Vec<f64> = if grid.len() > k { (k..grid.len()).map(|i| grid[i] / grid[i - k] - 1.0).collect() } else { vec![] };
    if d.is_empty() {
        return None;
    }
    let wk: Vec<bool> = (0..d.len()).map(|i| ((t0 + (i + k) as i64 * step).div_euclid(86400) + 3).rem_euclid(7) >= 5).collect();
    let last_wk = wk[wk.len() - 1];
    let head = &d[..d.len() - 1];
    let same: Vec<f64> = head.iter().zip(&wk).filter(|(x, w)| fin(**x) && **w == last_wk).map(|(x, _)| *x).collect();
    let past: Vec<f64> = if same.len() >= 72 { same } else { head.iter().copied().filter(|x| fin(*x)).collect() };
    let med = if past.is_empty() { 0.0 } else { median(&past) };
    let mut mad = if past.is_empty() { 0.0 } else { 1.4826 * median(&past.iter().map(|x| (x - med).abs()).collect::<Vec<_>>()) };
    if !past.is_empty() {
        mad = mad.max(0.25 * np_std(&past));
    }
    let l = d[d.len() - 1];
    (fin(l) && past.len() >= 72 && mad > 0.0).then(|| (l - med) / (OI_Z_MAD_SCALE * mad))
}

/// oi_stats(83~104): 5분 격자 OI → {oi, d1h_pct, d24h_pct, z1h, d30m_pct, z30m}.
pub fn oi_stats(series: &[(i64, f64)]) -> Map<String, V> {
    let step = 300;
    let mut out: Map<String, V> = ["oi", "d1h_pct", "d24h_pct", "z1h", "d30m_pct", "z30m"].iter().map(|k| (k.to_string(), V::Null)).collect();
    let (Some(f), Some(l)) = (series.first(), series.last()) else { return out };
    let (t0, t1) = (f.0, l.0);
    let n = ((t1 - t0).div_euclid(step) + 1) as usize;
    if n > 10_000_000 {
        return out; // ponytail: 원본엔 없는 상한(깨진 시각이 큰 할당을 만들지 않게) -- 7일 격자는 2,016칸
    }
    let mut grid = vec![f64::NAN; n];
    for &(t, v) in series {
        grid[(t - t0).div_euclid(step) as usize] = v;
    }
    let g = |i: usize| grid[grid.len() - 1 - i];
    out["oi"] = json!(g(0));
    let (k1, k24) = (12usize, 288usize);
    if grid.len() > k1 && fin(g(k1)) {
        out["d1h_pct"] = json!((g(0) / g(k1) - 1.0) * 100.0);
    }
    if grid.len() > k24 && fin(g(k24)) {
        out["d24h_pct"] = json!((g(0) / g(k24) - 1.0) * 100.0);
    }
    for (k, dk, zk) in [(k1, "d1h_pct", "z1h"), (6usize, "d30m_pct", "z30m")] {
        if k == 6 && grid.len() > k && fin(g(k)) {
            out[dk] = json!((g(0) / g(k) - 1.0) * 100.0);
        }
        out[zk] = json!(oi_change_z(&grid, k, t0, step));
    }
    out
}

/// oi_series_live(129~134).
pub fn oi_series_live(series: &[(i64, f64)], sec: i64, value: f64) -> Vec<(i64, f64)> {
    let b = sec.div_euclid(300) * 300;
    series.iter().copied().filter(|r| r.0 < b).chain([(b, value)]).collect()
}

/// oi_sum_series(137~156). `per` = 원본 dict 차례(binance, okx, hl, bybit).
pub fn oi_sum_series(per: &[(String, Vec<(i64, f64)>)], since: i64) -> (Vec<(i64, f64)>, Vec<String>) {
    let (tol, step, ffill) = (3600i64, 300i64, 2i64);
    let use_: Vec<&(String, Vec<(i64, f64)>)> = per.iter().filter(|(_, r)| r.first().is_some_and(|f| f.0 <= since + tol)).collect();
    if !use_.iter().any(|(k, _)| k == "binance") {
        return (vec![], vec![]);
    }
    let maps: Vec<BTreeMap<i64, f64>> = use_
        .iter()
        .map(|(_, rows)| {
            let mut m = BTreeMap::new();
            let mut prev: Option<(i64, f64)> = None;
            for &(t, v) in rows {
                if let Some(p) = prev {
                    for j in 1..((t - p.0).div_euclid(step)).min(ffill + 1) {
                        m.insert(p.0 + j * step, p.1);
                    }
                }
                m.insert(t, v);
                prev = Some((t, v));
            }
            m
        })
        .collect();
    let ts: Vec<i64> = maps[0].keys().copied().filter(|t| maps.iter().all(|m| m.contains_key(t))).collect();
    (ts.iter().map(|t| (*t, maps.iter().fold(0.0, |s, m| s + m[t]))).collect(), use_.iter().map(|(k, _)| k.clone()).collect())
}

/// ring_at(159~166): 가까운 순 0, −1, +1, −2, +2.
pub fn ring_at<T: Clone>(ring: &BTreeMap<i64, T>, sec: i64, fallback: T) -> T {
    for d in [0, -1, 1, -2, 2] {
        if let Some(v) = ring.get(&(sec + d)) {
            return v.clone();
        }
    }
    fallback
}

/// lev_state(169~185).
pub fn lev_state(basis_pct: Option<f64>, oi_z: Option<f64>) -> V {
    let hot = oi_z.is_some_and(|z| z >= 1.0);
    let l = |k: &str, s: &str| json!({"key": k, "label": s});
    let Some(b) = basis_pct else { return l("na", "베이시스 이력 쌓는 중") };
    if b >= 0.9 && hot {
        l("long_crowd", "롱 쏠림 — 프리미엄 높고 OI 급증")
    } else if b <= 0.1 && hot {
        l("short_crowd", "숏 쏠림 — 할인 깊고 OI 급증")
    } else if oi_z.is_some_and(|z| z <= -1.5) {
        l("deleverage", "레버리지 빠지는 중 — OI 급감")
    } else if b >= 0.9 {
        l("premium", "선물 프리미엄 높음")
    } else if b <= 0.1 {
        l("discount", "선물 할인 깊음")
    } else {
        l("normal", "보통")
    }
}

/// hl_liq_levels(188~207). rows = (szi, liq_px)(None 은 0 = 거짓).
pub fn hl_liq_levels(rows: &[(Option<f64>, Option<f64>)], mid: f64, bin: f64) -> V {
    let (top, max_pct) = (3usize, HL_LIQ_MAX_PCT);
    let mut agg: Vec<((&str, f64), [f64; 2])> = vec![]; // 원본 dict 차례
    for &(szi, liq) in rows {
        let (Some(szi), Some(liq)) = (szi.filter(|x| *x != 0.0), liq.filter(|x| *x != 0.0)) else { continue };
        if !(liq > 0.0) || (liq / mid - 1.0).abs() > max_pct {
            continue;
        }
        let side = if szi > 0.0 { "below" } else { "above" };
        if (side == "below") != (liq < mid) {
            continue;
        }
        let k = (side, round_nd(r0(liq / bin) as f64 * bin, 10));
        match agg.iter_mut().find(|(kk, _)| kk.0 == k.0 && kk.1 == k.1) {
            Some((_, a)) => {
                a[0] += szi.abs() * mid;
                a[1] += 1.0;
            }
            None => agg.push((k, [szi.abs() * mid, 1.0])),
        }
    }
    let mut out = Map::new();
    for side in ["below", "above"] {
        let mut lv: Vec<(f64, [f64; 2])> = agg.iter().filter(|(k, _)| k.0 == side).map(|(k, a)| (k.1, *a)).collect();
        lv.sort_by(|a, b| (-a.1[0]).partial_cmp(&-b.1[0]).unwrap()); // 안정 정렬(파이썬 sorted)
        lv.truncate(top);
        lv.sort_by(|a, b| (a.0 - mid).abs().partial_cmp(&(b.0 - mid).abs()).unwrap());
        out.insert(side.into(), V::Array(lv.iter().map(|(px, a)| json!({"px": px, "usd": r0(a[0]), "n": a[1] as i64})).collect()));
    }
    V::Object(out)
}

/// liq_profile(210~217): [(가격, USD, 롱?)] → [[가격, 롱USD, 숏USD]] 가격 오름차순.
pub fn liq_profile(events: &[(f64, f64, bool)], bin: f64) -> V {
    let mut agg: BTreeMap<u64, (f64, [f64; 2])> = BTreeMap::new();
    for &(px, usd, long) in events {
        if px > 0.0 && usd > 0.0 {
            let k = round_nd(r0(px / bin) as f64 * bin, 10);
            agg.entry(k.to_bits()).or_insert((k, [0.0, 0.0])).1[if long { 0 } else { 1 }] += usd;
        }
    }
    let mut v: Vec<&(f64, [f64; 2])> = agg.values().collect();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    V::Array(v.iter().map(|(k, a)| json!([k, r0(a[0]), r0(a[1])])).collect())
}

fn bincount(k: &[i64], v: &[f64]) -> (i64, Vec<f64>) {
    let k0 = *k.iter().min().unwrap();
    let n = (k.iter().max().unwrap() - k0 + 1) as usize;
    let mut h = vec![0.0; n];
    for (kk, vv) in k.iter().zip(v) {
        h[(kk - k0) as usize] += vv;
    }
    (k0, h)
}

/// value_area(262~277).
pub fn value_area(k: &[i64], v: &[f64], bw: f64) -> V {
    if k.is_empty() || rtcollect::ob5m::np_sum(v) <= 0.0 {
        return V::Null;
    }
    let (k0, h) = bincount(k, v);
    let poc = h.iter().enumerate().fold(0, |b, (i, x)| if *x > h[b] { i } else { b });
    let (mut lo, mut hi, mut tot) = (poc, poc, h[poc]);
    let target = 0.7 * rtcollect::ob5m::np_sum(&h);
    while tot < target && (lo > 0 || hi < h.len() - 1) {
        let up = if hi < h.len() - 1 { h[hi + 1] } else { -1.0 };
        let dn = if lo > 0 { h[lo - 1] } else { -1.0 };
        if up >= dn {
            hi += 1;
            tot += up;
        } else {
            lo -= 1;
            tot += dn;
        }
    }
    json!({"val": (k0 + lo as i64) as f64 * bw, "poc": ((k0 + poc as i64) as f64 + 0.5) * bw, "vah": (k0 + hi as i64 + 1) as f64 * bw})
}

/// np.convolve(a, v, "same") -- 원본 _pyarray_correlate(mode 1)의 세 갈래(왼쪽 n/2 · 가운데 · 오른쪽) · 내적은 16칸 미만이라 차례 합.
fn convolve_same(a: &[f64], v: &[f64]) -> Vec<f64> {
    let (a, v) = if v.len() > a.len() { (v, a) } else { (a, v) };
    let v: Vec<f64> = v.iter().rev().copied().collect(); // convolve = correlate(a, v[::-1])
    let (n1, n2) = (a.len(), v.len());
    let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).fold(0.0, |s, (p, q)| s + p * q);
    let n_left = n2 / 2;
    let n_right = n2 - n_left - 1;
    let mut out = vec![];
    let mut n = n2 - n_left;
    for i in 0..n_left {
        out.push(dot(&a[..n], &v[n_left - i..]));
        n += 1;
    }
    for i in 0..(n1 - n2 + 1) {
        out.push(dot(&a[i..i + n2], &v));
    }
    let mut start = n1 - n2 + 1;
    let mut n = n2;
    for _ in 0..n_right {
        n -= 1;
        out.push(dot(&a[start..start + n], &v[..n]));
        start += 1;
    }
    out
}

/// profile_nodes(280~301). NODE_WIN 6 · NODE_SIG 2.0 · NODE_MAX 8(27).
pub fn profile_nodes(k: &[i64], v: &[f64], bw: f64) -> V {
    let (win, sig, nmax) = (6usize, 2.0f64, 8usize);
    if k.len() < 2 * win || rtcollect::ob5m::np_sum(v) <= 0.0 {
        return json!({"hvn": [], "lvn": []});
    }
    let (k0, h) = bincount(k, v);
    let x: Vec<f64> = (0..13).map(|i| -3.0 * sig + i as f64).collect();
    let mut ker: Vec<f64> = x.iter().map(|x| (-0.5 * (x / sig).powi(2)).exp()).collect();
    let ks = rtcollect::ob5m::np_sum(&ker);
    for q in ker.iter_mut() {
        *q /= ks;
    }
    let s = convolve_same(&h, &ker);
    let n = s.len();
    let at = |i: i64, pad: f64| if i < 0 || i >= n as i64 { pad } else { s[i as usize] };
    let mx: Vec<f64> = (0..n as i64).map(|i| (i - win as i64..=i + win as i64).map(|j| at(j, f64::NEG_INFINITY)).fold(f64::NEG_INFINITY, f64::max)).collect();
    let mn: Vec<f64> = (0..n as i64).map(|i| (i - win as i64..=i + win as i64).map(|j| at(j, f64::INFINITY)).fold(f64::INFINITY, f64::min)).collect();
    let mut left = s.clone();
    for i in 1..n {
        left[i] = left[i].max(left[i - 1]);
    }
    let mut right = s.clone();
    for i in (0..n - 1).rev() {
        right[i] = right[i].max(right[i + 1]);
    }
    let smax = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = Map::new();
    for name in ["hvn", "lvn"] {
        let isx: Vec<bool> = (0..n).map(|i| if name == "hvn" { s[i] >= mx[i] && s[i] >= 0.25 * smax } else { s[i] <= mn[i] && s[i] <= 0.5 * left[i].min(right[i]) }).collect();
        let mut starts = vec![];
        let mut ends = vec![];
        let mut prev = false;
        for i in 0..=n {
            let cur = i < n && isx[i];
            if cur && !prev {
                starts.push(i);
            }
            if !cur && prev {
                ends.push(i);
            }
            prev = cur;
        }
        let mut idx: Vec<usize> = starts.iter().zip(&ends).map(|(a, b)| (a + b - 1) / 2).collect();
        let key = |i: usize| if name == "hvn" { -s[i] } else { s[i] };
        idx.sort_by(|a, b| key(*a).partial_cmp(&key(*b)).unwrap());
        idx.truncate(nmax);
        idx.sort();
        out.insert(name.into(), V::Array(idx.iter().map(|&i| json!({"px": ((k0 + i as i64) as f64 + 0.5) * bw, "rel": round_nd(s[i] / smax, 3)})).collect()));
    }
    V::Object(out)
}

// ── 파일 몫 ──
fn ro(db: &Path) -> rusqlite::Result<rusqlite::Connection> {
    rusqlite::Connection::open_with_flags(db, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX)
}

fn val(v: rusqlite::types::ValueRef) -> V {
    use rusqlite::types::ValueRef::*;
    match v {
        Null => V::Null,
        Integer(i) => json!(i),
        Real(f) => json!(f),
        Text(t) => json!(String::from_utf8_lossy(t)),
        Blob(_) => V::Null,
    }
}

/// _read_only_rows(SQLite) → 행(값 모양 그대로: 정수는 정수 · 실수는 실수).
pub fn sqlite_rows(db: &Path, sql: &str, params: &[&dyn rusqlite::ToSql]) -> Result<Vec<Vec<V>>, String> {
    let c = ro(db).map_err(|e| e.to_string())?;
    let mut q = c.prepare(sql).map_err(|e| e.to_string())?;
    let n = q.column_count();
    let rows = q.query_map(params, |r| (0..n).map(|i| r.get_ref(i).map(val)).collect::<Result<Vec<_>, _>>()).map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// 수집기 파일 자리(코인마다) -- 원본 OKX_CTX_DB_PATH·BYBIT_CTX_DB·HL_CTX_DB_PATH·HL_LIQ_BY_ASSET·OI_LSRATIO_DB_PATH·OI_1S_DB_PATH·MARK_PRICE_DB_PATH·MICRO_TAPE_DB.
pub struct Paths<'a> {
    pub okx_ctx: &'a Path,
    pub bybit_ctx: &'a Path,
    pub hl_ctx: &'a Path,
    pub hl_pos: &'a Path,
    pub lsratio: &'a Path,
    pub bn_ctx: &'a Path,
    pub tape: &'a Path,
}

/// _mc_collectors(2534~2556). `last_good` = 원본 mc_last_good(원천마다 마지막 성공값). 돌려줌: {키: 행 목록 또는 null, …, "errors": {…}}.
/// 원본과 다른 곳: errors 글자 = Rust 오류(원본 repr(exc)[:100]).
pub fn mc_collectors(p: &Paths, asset: &str, okx_inst: &str, symbol: &str, last_good: &mut Map<String, V>) -> Map<String, V> {
    let coin = asset.to_uppercase();
    let ls_table = if asset == "eth" { "oi_lsratio_5m".to_string() } else { format!("oi_lsratio_5m_{asset}") };
    let mut out = Map::new();
    out.insert("errors".into(), json!({})); // 원본 dict 첫 키
    let mut errors = Map::new();
    let mut q = |key: &str, path: &Path, r: &dyn Fn() -> Result<Vec<Vec<V>>, String>| {
        let got = if path.exists() { r().map(|v| json!(v)) } else { Ok(V::Null) };
        match got {
            Ok(v) => {
                out.insert(key.into(), v.clone());
                last_good.insert(key.into(), v);
            }
            Err(e) => {
                out.insert(key.into(), last_good.get(key).cloned().unwrap_or(V::Null));
                errors.insert(key.into(), json!(e.chars().take(100).collect::<String>()));
            }
        }
    };
    q("okx_fund", p.okx_ctx, &|| sqlite_rows(p.okx_ctx, "SELECT funding_rate, funding_time FROM okx_funding WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", &[&okx_inst]));
    q("okx_mark", p.okx_ctx, &|| sqlite_rows(p.okx_ctx, "SELECT mark_px, ts_ms FROM okx_mark WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", &[&okx_inst]));
    q("okx_oi", p.okx_ctx, &|| sqlite_rows(p.okx_ctx, "SELECT oi_base FROM okx_oi WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", &[&okx_inst]));
    q("by_oi", p.bybit_ctx, &|| sqlite_rows(p.bybit_ctx, "SELECT oi_base FROM bybit_oi WHERE inst = ? ORDER BY ts_ms DESC LIMIT 1", &[&symbol]));
    q("hl_ctx", p.hl_ctx, &|| sqlite_rows(p.hl_ctx, "SELECT funding, open_interest, premium, mark_px, mid_px, recv_ms FROM hl_asset_ctx WHERE coin = ? ORDER BY recv_ms DESC LIMIT 1", &[&coin]));
    q("hl_pos", p.hl_pos, &|| sqlite_rows(p.hl_pos, "SELECT szi, liq_px FROM hl_positions WHERE coin = ? AND ts_ms >= (SELECT max(ts_ms) FROM hl_cycles)", &[&coin]));
    q("ls", p.lsratio, &|| lsratio_rows(p.lsratio, &ls_table));
    out.insert("errors".into(), V::Object(errors));
    out
}

/// 롱숏비(duckdb · 읽기 전용) -- 원본 SQL 그대로(25시간 · ts 차례).
fn lsratio_rows(db: &Path, table: &str) -> Result<Vec<Vec<V>>, String> {
    let cfg = duckdb::Config::default().access_mode(duckdb::AccessMode::ReadOnly).map_err(|e| e.to_string())?;
    let c = duckdb::Connection::open_with_flags(db, cfg).map_err(|e| e.to_string())?;
    let mut q = c
        .prepare(&format!("SELECT epoch(ts), global_ls_ratio, top_pos_ls_ratio, taker_ls_ratio FROM {table} WHERE epoch(ts) >= ? ORDER BY ts"))
        .map_err(|e| e.to_string())?;
    // 원본 `ts >= now() - INTERVAL 25 HOUR`(표의 ts = TIMESTAMPTZ · oi_lsratio_collector.py 78) -- 이 크레이트의 duckdb 는 ICU 가 없어
    //   TIMESTAMPTZ 빼기를 못 한다(Binder Error) → 같은 뜻을 epoch 초로: 지금 벽시계 − 25시간.
    let cut = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64()) - 25.0 * 3600.0;
    let rows = q.query_map([cut], |r| {
        let g = |i: usize| -> duckdb::Result<V> { Ok(r.get::<_, Option<f64>>(i)?.map_or(V::Null, |x| json!(x))) };
        Ok(vec![g(0)?, g(1)?, g(2)?, g(3)?])
    });
    rows.map_err(|e| e.to_string())?.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

/// _mc_history(2558~2583) 의 값.
#[derive(Clone, Debug, Default)]
pub struct History {
    pub oi: Vec<(i64, f64)>,
    pub oi_per: Vec<(String, Vec<(i64, f64)>)>,
    pub basis: Vec<f64>,
}

fn pairs(rows: Vec<Vec<V>>) -> Vec<(i64, f64)> {
    rows.into_iter().filter_map(|r| Some((r[0].as_i64().or_else(|| r[0].as_f64().map(|x| x as i64))?, r[1].as_f64()?))).collect()
}

/// _mc_history(2558~2583): 7일 OI 5분 끝값(바이낸스 · OKX · HL · Bybit) · 1분 베이시스. 원천 하나가 없거나 실패하면 그 몫만 빈다.
pub fn mc_history(p: &Paths, now_ms: i64, symbol_lower: &str, okx_inst: &str, coin: &str, symbol: &str) -> History {
    let since = now_ms - 7 * 86400 * 1000;
    let own = |db: &Path, sql: &str, params: &[&dyn rusqlite::ToSql]| -> Vec<Vec<V>> {
        if !db.exists() {
            return vec![];
        }
        sqlite_rows(db, sql, params).unwrap_or_else(|e| {
            println!("own db read failed ({}): {e}", db.display());
            vec![]
        })
    };
    let oi = pairs(own(p.bn_ctx, "SELECT ts_ms / 300000 * 300 AS b, open_interest, max(ts_ms) FROM oi_1s WHERE symbol = ? AND ts_ms >= ? GROUP BY b ORDER BY b", &[&symbol_lower, &since]));
    let mut per = vec![("binance".to_string(), oi.clone())];
    for (k, path, tbl, ts, v, kc, key) in [("okx", p.okx_ctx, "okx_oi", "ts_ms", "oi_base", "inst", okx_inst), ("hl", p.hl_ctx, "hl_asset_ctx", "recv_ms", "open_interest", "coin", coin),
        ("bybit", p.bybit_ctx, "bybit_oi", "ts_ms", "oi_base", "inst", symbol)] {
        let sql = format!("SELECT {ts} / 300000 * 300 AS b, {v}, max({ts}) FROM {tbl} WHERE {kc} = ? AND {ts} >= ? AND {v} IS NOT NULL GROUP BY b ORDER BY b");
        let rows = if path.exists() { sqlite_rows(path, &sql, &[&key, &since]) } else { Ok(vec![]) };
        per.push((k.to_string(), pairs(rows.unwrap_or_else(|e| {
            println!("mc_history oi {k}: {e}");
            vec![]
        }))));
    }
    let basis = own(p.bn_ctx, "SELECT (mark - index_px) / index_px * 1e4 FROM mark_price_1s WHERE symbol = ? AND ts_ms >= ? AND index_px > 0 AND ts_ms % 60000 < 1000", &[&symbol_lower, &since]);
    History { oi, oi_per: per, basis: basis.into_iter().filter_map(|r| r[0].as_f64()).collect() }
}

/// _mc_liq_profile(2585~2595): 바이낸스 청산 12시간 → liq_profile. 실패면 [].
pub fn mc_liq_profile(p: &Paths, symbol: &str, now_s: f64, px_scale: f64) -> V {
    let cut = ((now_s - MC_LIQ_PROFILE_S) * 1000.0) as i64;
    let Ok(rows) = sqlite_rows(p.bn_ctx, "SELECT price, usd, side FROM liquidations WHERE symbol = ? AND ts_ms >= ?", &[&symbol, &cut]) else { return json!([]) };
    let ev: Vec<(f64, f64, bool)> = rows.iter().map(|r| (r[0].as_f64().unwrap_or(0.0), r[1].as_f64().unwrap_or(0.0), r[2].as_str() == Some("long"))).collect();
    liq_profile(&ev, LIQ_PROFILE_BIN_USD * px_scale)
}

/// tape_levels(931~969): 전일 가치영역 · 24시간 HVN/LVN · 주간 VWAP · 앵커드 VWAP.
pub fn tape_levels(db: &Path, sym: &str, bk: f64, now: i64) -> Result<V, String> {
    let d0 = now.div_euclid(86400) * 86400;
    let w0 = d0 - ((d0.div_euclid(86400) + 3).rem_euclid(7)) * 86400;
    let vw = sqlite_rows(db, "
        WITH t AS (SELECT ts_sec, price_bin, buy_qty + sell_qty AS q FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?),
             pd AS (SELECT ts_sec, price_bin FROM t WHERE ts_sec >= ? AND ts_sec < ?),
             ah AS (SELECT ts_sec FROM pd ORDER BY price_bin DESC, ts_sec LIMIT 1),
             al AS (SELECT ts_sec FROM pd ORDER BY price_bin, ts_sec LIMIT 1)
        SELECT (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= ?),
               (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= (SELECT ts_sec FROM ah)),
               (SELECT sum(price_bin * q) / nullif(sum(q), 0) FROM t WHERE ts_sec >= (SELECT ts_sec FROM al)),
               (SELECT ts_sec FROM ah), (SELECT ts_sec FROM al),
               (SELECT price_bin FROM t ORDER BY ts_sec DESC, price_bin DESC LIMIT 1),
               (SELECT price_bin FROM pd ORDER BY ts_sec DESC, price_bin DESC LIMIT 1)",
        &[&sym, &w0.min(d0 - 86400), &(d0 - 86400), &d0, &w0])?
        .into_iter()
        .next()
        .ok_or("IndexError")?;
    let px = |b: &V| b.as_f64().map(|b| (b + 0.5) * bk);
    let Some(last) = px(&vw[5]).filter(|x| *x != 0.0) else { return Ok(json!({"available": false})) };
    let bw = bk.max(r0(px(&vw[6]).filter(|x| *x != 0.0).unwrap_or(last) * PROFILE_BIN_FRAC / bk) as f64 * bk);
    // 원본 SQL 의 floor() 는 SQLite 수학 함수(파이썬 sqlite3 는 켜져 있다) -- 이 크레이트의 묶음 SQLite 에는 없다 → 같은 값을 CAST 로(음수면 한 칸 아래).
    let fl = |x: &str| format!("(CASE WHEN {x} < CAST({x} AS INTEGER) THEN CAST({x} AS INTEGER) - 1 ELSE CAST({x} AS INTEGER) END)");
    let x = "((price_bin + 0.5) * ?1 / ?2)";
    let sql = format!("
        SELECT 0, {f}, sum(buy_qty + sell_qty) FROM trade_tape_1s
        WHERE symbol = ?3 AND ts_sec >= ?4 AND ts_sec < ?5 GROUP BY 2
        UNION ALL
        SELECT 1, {f}, sum(buy_qty + sell_qty) FROM trade_tape_1s
        WHERE symbol = ?3 AND ts_sec >= ?6 GROUP BY 2", f = fl(x));
    let rows = sqlite_rows(db, &sql, &[&bk, &bw, &sym, &(d0 - 86400), &d0, &(now - 86400)])?;
    let split = |g: i64| -> (Vec<i64>, Vec<f64>) {
        rows.iter().filter(|r| r[0].as_i64() == Some(g)).map(|r| (r[1].as_i64().unwrap_or(0), r[2].as_f64().unwrap_or(0.0))).unzip()
    };
    let (pk, pv) = split(0);
    let (hk, hv) = split(1);
    let mut out = Map::new();
    out.insert("available".into(), json!(true));
    out.insert("last".into(), json!(last));
    out.insert("bin".into(), json!(bw));
    out.insert("prev_day".into(), value_area(&pk, &pv, bw));
    if let V::Object(n) = profile_nodes(&hk, &hv, bw) {
        out.extend(n);
    }
    out.insert("vwap_week".into(), json!(px(&vw[0])));
    out.insert("week_start".into(), json!(w0));
    out.insert("avwap_hi".into(), json!(px(&vw[1])));
    out.insert("avwap_hi_ts".into(), vw[3].clone());
    out.insert("avwap_lo".into(), json!(px(&vw[2])));
    out.insert("avwap_lo_ts".into(), vw[4].clone());
    Ok(V::Object(out))
}

// ── REST 몫(부르는 쪽이 RTSERVE_REST 있을 때만 · 실패면 원본처럼 빈 목록) ──
pub fn funding_hist_url(base: &str, symbol: &str) -> String {
    format!("{base}/fapi/v1/fundingRate?symbol={symbol}&limit=1000")
}
/// funding_hist(2613~2616): fundingRate 마지막 540개.
pub fn funding_hist(raw: &V) -> Result<Vec<f64>, String> {
    let a = raw.as_array().ok_or("TypeError")?;
    let v: Vec<f64> = a.iter().map(|r| r["fundingRate"].as_str().and_then(|s| s.parse().ok()).or_else(|| r["fundingRate"].as_f64()).ok_or("ValueError".to_string())).collect::<Result<_, _>>()?;
    Ok(v[v.len().saturating_sub(540)..].to_vec())
}
pub fn dvol_url(now_ms: i64) -> String {
    format!("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=ETH&start_timestamp={}&end_timestamp={now_ms}&resolution=1D", now_ms - 366 * 86_400_000)
}
/// dvol_year(2605~2611): sorted(result.data)(행 사전식) 의 종가(r[4]).
pub fn dvol_year(j: &V) -> Result<Vec<f64>, String> {
    let rows = j["result"]["data"].as_array().ok_or("KeyError")?;
    let mut r: Vec<Vec<f64>> = rows.iter().map(|x| x.as_array().map(|a| a.iter().filter_map(V::as_f64).collect()).ok_or("TypeError".to_string())).collect::<Result<_, _>>()?;
    r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    r.iter().map(|x| x.get(4).copied().ok_or("IndexError".to_string())).collect()
}

/// market_context_payload(2632~2708)의 메모리 몫.
pub struct McIn<'a> {
    pub asset: &'a str,
    pub now: f64,
    /// size_bands(sym) = (리테일 상한, 고래 하한).
    pub bands_usd: (f64, f64),
    /// micro_state["payload"](available 거짓이면 {}).
    pub micro: &'a V,
    /// situation_state["ctx"](x) · now.evidence.
    pub x: &'a Map<String, V>,
    pub ev: &'a V,
    pub mark_ring: &'a BTreeMap<i64, (f64, f64, f64)>,
    pub mid_ring: &'a BTreeMap<i64, f64>,
    pub depth25: &'a [(f64, f64)],
    pub oi_1s: &'a BTreeMap<i64, f64>,
    pub next_funding_ms: Option<i64>,
    pub px_scale: f64,
}

/// market_context_payload(2601~2708). col = mc_collectors · hist = mc_history · prof = mc_liq_profile · fh·dvy = REST(실패면 []) · vprof = tape_levels(실패면 {available:false,error}).
pub fn market_context_payload(i: &McIn, col: &Map<String, V>, hist: &History, prof: &V, fh: &[f64], dvy: &[f64], vprof: &V) -> V {
    let one = |k: &str| -> Option<Vec<V>> { col.get(k).and_then(V::as_array).and_then(|a| a.first()).and_then(|r| r.as_array().cloned()) };
    let empty = Map::new();
    let mp = i.micro.as_object().filter(|m| m.get("available").and_then(V::as_bool).unwrap_or(false)).unwrap_or(&empty);
    let mx = |k: &str| i.x.get(k).cloned().unwrap_or(V::Null);
    let evg = |k: &str| i.ev.get(k).cloned().unwrap_or(V::Null);
    let dv = rtcore::situation::deriv_from_ring(i.mark_ring, 1800, rtcore::situation::BASIS_PCT);
    let bn_mark = i.mark_ring.iter().next_back().map(|(_, v)| v.0);
    let mid = mp.get("mid").and_then(V::as_f64).filter(|m| *m != 0.0).or(bn_mark);
    let oi = oi_stats(&match i.oi_1s.iter().next_back() {
        Some((&s, &v)) => oi_series_live(&hist.oi, s, v),
        None => hist.oi.clone(),
    });
    let (okx_oi, hl, by_oi) = (one("okx_oi"), one("hl_ctx"), one("by_oi").and_then(|r| r.first().cloned()).unwrap_or(V::Null));
    let now_s = i.now as i64;
    let (oi_ser, oi_ven) = oi_sum_series(&hist.oi_per, (now_s - 7 * 86400).div_euclid(300) * 300);
    let latest = |k: &str| -> V {
        match k {
            "binance" => i.oi_1s.iter().next_back().map_or(V::Null, |(_, v)| json!(v)),
            "okx" => okx_oi.as_ref().map_or(V::Null, |r| r[0].clone()),
            "hl" => hl.as_ref().map_or(V::Null, |r| r[1].clone()),
            _ => by_oi.clone(),
        }
    };
    let truthy = |v: &V| v.as_f64().is_some_and(|x| x != 0.0) || v.as_str().is_some_and(|s| !s.is_empty());
    let mut oi_all = V::Null;
    if !oi_ser.is_empty() && oi_ven.iter().all(|k| truthy(&latest(k))) {
        let tot = oi_ven.iter().fold(0.0, |s, k| s + latest(k).as_f64().unwrap_or(0.0));
        let mut m = oi_stats(&oi_series_live(&oi_ser, now_s, tot));
        m.insert("venues".into(), json!(oi_ven));
        let by: Map<String, V> = oi_ven
            .iter()
            .map(|k| (k.clone(), if k == "binance" { oi["d1h_pct"].clone() } else { oi_stats(hist.oi_per.iter().find(|(n, _)| n == k).map_or(&[][..], |(_, r)| r))["d1h_pct"].clone() }))
            .collect();
        m.insert("by_venue".into(), V::Object(by));
        let d1h = match (m["oi"].as_f64(), m["d1h_pct"].as_f64()) {
            (Some(o), Some(p)) => json!(o * (1.0 - 1.0 / (1.0 + p / 100.0))),
            _ => V::Null,
        };
        m.insert("d1h".into(), d1h);
        oi_all = V::Object(m);
    }
    let (okx_mark, okx_fund) = (one("okx_mark"), one("okx_fund"));
    let bp = |a: Option<f64>, b: Option<f64>| match (a.filter(|x| *x != 0.0), b.filter(|x| *x != 0.0)) {
        (Some(a), Some(b)) => json!((a / b - 1.0) * 1e4),
        _ => V::Null,
    };
    let ls: Vec<Vec<V>> = col.get("ls").and_then(V::as_array).map(|a| a.iter().filter_map(|r| r.as_array().cloned()).collect()).unwrap_or_default();
    let lsv = |j: usize| ls.iter().rev().find(|r| !r[j].is_null()).map_or(V::Null, |r| r[j].clone());
    let ls24 = |j: usize| -> V {
        let target = i.now - 86400.0;
        let best = ls.iter().filter(|r| !r[j].is_null()).map(|r| ((r[0].as_f64().unwrap_or(f64::NAN) - target).abs(), r[j].as_f64().unwrap_or(f64::NAN)))
            .fold((1e9, None::<f64>), |b, c| if c.0 < b.0 || (c.0 == b.0 && b.1.is_some_and(|v| c.1 < v)) { (c.0, Some(c.1)) } else { b });
        if best.0 <= 7200.0 { json!(best.1) } else { V::Null }
    };
    let sw = mp.get("sweep").filter(|v| !v.is_null());
    let basis_pct = pct_rank(dv.basis_bp, &hist.basis, 600);
    let liq60 = mp.get("liq60").filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    let oiz = oi_all.get("z1h").and_then(V::as_f64).or_else(|| oi["z1h"].as_f64());
    let mut oi_out = oi.clone();
    oi_out.insert("okx".into(), okx_oi.as_ref().map_or(V::Null, |r| r[0].clone()));
    oi_out.insert("hl".into(), hl.as_ref().map_or(V::Null, |r| r[1].clone()));
    oi_out.insert("bybit".into(), by_oi.clone());
    oi_out.insert("all".into(), oi_all.clone());
    let pct25 = |side: usize, key: &str| -> V {
        match (i.depth25.len() >= 600, sw.and_then(|s| s[key][0].as_f64())) {
            (true, Some(x)) => json!(i.depth25.iter().filter(|d| (if side == 0 { d.0 } else { d.1 }) <= x).count() as f64 / i.depth25.len() as f64),
            _ => V::Null,
        }
    };
    let hl3 = hl.as_ref().and_then(|r| r[3].as_f64()).filter(|x| *x != 0.0);
    let hl_rows: Vec<(Option<f64>, Option<f64>)> = col.get("hl_pos").and_then(V::as_array).map(|a| a.iter().map(|r| (r[0].as_f64(), r[1].as_f64())).collect()).unwrap_or_default();
    let hl_liq = match hl3.or(mid) {
        Some(m) => hl_liq_levels(&hl_rows, m, HL_LIQ_BIN_USD * i.px_scale),
        None => json!({"below": [], "above": []}),
    };
    let okx_ref = okx_mark.as_ref().map(|r| ring_at(i.mark_ring, r[1].as_i64().unwrap_or(0).div_euclid(1000), (bn_mark.unwrap_or(f64::NAN), 0.0, 0.0)).0).filter(|x| !x.is_nan());
    let hl_ref = hl.as_ref().and_then(|r| ring_at(i.mid_ring, r[5].as_i64().unwrap_or(0).div_euclid(1000), mp.get("mid").and_then(V::as_f64).unwrap_or(f64::NAN)).into()).filter(|x: &f64| !x.is_nan());
    json!({
        "available": true, "asset": i.asset, "ts": i.now, "mid": mid,
        "bands_usd": [i.bands_usd.0, i.bands_usd.1],
        "funding": {"bn": dv.funding, "bn_pct180": pct_rank(dv.funding, fh, 30), "next_ms": i.next_funding_ms,
                    "bn_at_base": dv.funding.is_some_and(|f| (f - 0.0001).abs() < 1e-9),
                    "okx": okx_fund.as_ref().map_or(V::Null, |r| r[0].clone()), "okx_next_ms": okx_fund.as_ref().map_or(V::Null, |r| r[1].clone()),
                    "hl_8h": hl.as_ref().and_then(|r| r[0].as_f64()).map_or(V::Null, |f| json!(f * 8.0))},
        "basis": {"bp": dv.basis_bp, "d30_bp": dv.basis_d_bp, "pct7d": basis_pct,
                  "hl_premium_bp": hl.as_ref().and_then(|r| r[2].as_f64()).map_or(V::Null, |p| json!(p * 1e4))},
        "lev": lev_state(basis_pct, oiz),
        "oi": V::Object(oi_out),
        "ls": if ls.is_empty() { V::Null } else { json!({"global": lsv(1), "top_pos": lsv(2), "taker": lsv(3), "global_24h": ls24(1), "top_pos_24h": ls24(2),
                                                         "age_s": i.now - ls[ls.len() - 1][0].as_f64().unwrap_or(f64::NAN)}) },
        "quad": quad_1h(mx("move60").as_f64(), oi_all.get("d1h").and_then(V::as_f64).or_else(|| mx("oi60").as_f64()), mx("move60_p75").as_f64()),
        "move60": mx("move60"), "oi60": mx("oi60"),
        "flow": {"z60": mx("z60"), "z60h": mx("z60h"), "net60": mx("net60"), "cvd30_z": mx("cvd30_z"),
                 "venues": mx("venues"), "z60h_all": mx("z60h_all"), "net60_all": mx("net60_all"),
                 "z30h": mx("z30h"), "z30h_all": mx("z30h_all"), "net30_all": mx("net30_all"),
                 "cvd30_all_z": mx("cvd30_all_z"), "taker_all": mx("taker_all"),
                 "bn30": evg("cvd"), "okx30": mx("okx30")},
        "btc": {"move_bp": evg("btc_move_bp"), "rel": evg("btc_rel")},
        "venues": {"okx_bp": bp(okx_mark.as_ref().and_then(|r| r[0].as_f64()), okx_ref), "hl_bp": bp(hl.as_ref().and_then(|r| r[4].as_f64()), hl_ref)},
        "book": {"spread": mp.get("spread").cloned().unwrap_or(V::Null), "sweep": sw.cloned().unwrap_or(V::Null), "bps": SWEEP_BPS,
                 "bid25_pct": pct25(0, "bid"), "ask25_pct": pct25(1, "ask")},
        "hl_liq": hl_liq,
        "liq_profile": prof,
        "profile": vprof,
        "iv_rank": if dvy.is_empty() { V::Null } else { json!({"dvol": dvy[dvy.len() - 1], "rank365": pct_rank(Some(dvy[dvy.len() - 1]), &dvy[..dvy.len() - 1], 200), "n": dvy.len() - 1}) },
        "burst": match (i.asset == "eth", liq60) {
            (true, Some(l)) => json!({"long_usd_60s": l.get("long").cloned().unwrap_or(V::Null), "short_usd_60s": l.get("short").cloned().unwrap_or(V::Null), "thr": liq_burst(), "ts": mp.get("ts").cloned().unwrap_or(V::Null)}),
            _ => V::Null,
        },
        "errors": col.get("errors").cloned().unwrap_or(json!({})),
    })
}

/// api_market_context(2717~2732): 성공이면 그 값을 «마지막 성공»으로 · 실패(3초 넘김 포함)면 120초 안이면 묵은 값 + stale·stale_age_s·error, 넘으면 {available:false, error}.
pub fn mc_response(r: Result<V, String>, last: &mut Option<(V, f64)>, now: f64) -> V {
    match r {
        Ok(p) => {
            *last = Some((p.clone(), now));
            p
        }
        Err(e) => {
            let err: String = e.chars().take(200).collect();
            match last {
                Some((p, at)) if now - *at < MC_STALE_MAX_S => {
                    let mut m = p.as_object().cloned().unwrap_or_default();
                    m.insert("stale".into(), json!(true));
                    m.insert("stale_age_s".into(), json!(now - *at));
                    m.insert("error".into(), json!(err));
                    V::Object(m)
                }
                _ => json!({"available": false, "error": err}),
            }
        }
    }
}
