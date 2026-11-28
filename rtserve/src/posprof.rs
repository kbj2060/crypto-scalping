//! 포지션 프로파일(2026-11-28): /api/position-profile -- 바깥 요청은 model::fetch_binance(→ binance_get 한 문 · RTSERVE_REST 있을 때만).
//! 원본(Read 로 읽은 줄): server.py pos_build·pos_bars_rest·pos_offset·pos_ensure·pos_forming 3303~3388 · api_position_profile 4738~4771 ·
//!   POSITION_PROFILE_CACHE_SECONDS 251(1초) · STALE_GRACE_SECONDS 1041 · dashboard/position_profile.py 전부(Book·_step·bins·payload·
//!   oi_hist_map·bars_from_klines·forming_from_cells·live_at·side_share) · coin_flow.py collect_oi_1s 680~702(1초 OI 링 = hot oi_1s 꼬리).
//! 흐름 엔진 몫: 형성 봉 칸(고저·거래량) = 허브 풋프린트 봉 · 마지막 체결가 = 허브 Supply1s 초 행 · 1초 OI = hot SQLite(원본과 같은 원천).
//! numpy 규칙: .sum() = rtcollect::ob5m::np_sum(짝 합) · np.exp = libm(f64::exp · 서버 판 = AVX-512 없음) · np.round(·,1) = ×10 짝수 반올림 ÷10.
//! 시험: tests/routes7_parity.rs(정답지 = tools/gen_posprof_fixture.py · 원본 모듈 dashboard/position_profile.py 를 그대로 불러 씀).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use rtcollect::ob5m::np_sum;
use serde_json::{json, Map, Value as V};

use crate::model::fetch_binance;
use crate::py::{round_nd, PyErr, R};
use crate::{http_exc, server_error, Config, Handler, Req, Resp};

const K: f64 = 0.057;
const TAU: f64 = 576.0;
const RHO: f64 = 0.02;
const RING: usize = 8640;
const WINDOWS: [i64; 3] = [1, 7, 30];
const SEED_BACKOFF: i64 = 60;
const REASON: &str = "position_profile_upstream_error";

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/position-profile" => vec![("GET", api_position_profile)],
        _ => Vec::new(),
    }
}

fn pymax(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}
fn pymin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// side_share: 손바뀜 중 롱 몫(직전 12·288봉 수익 bp) · np.clip(NaN 은 그대로)
fn side_share(cl: &[f64]) -> f64 {
    let (n, c) = (cl.len(), cl[cl.len() - 1]);
    let r1 = if n >= 13 { (c / cl[n - 13] - 1.0) * 1e4 } else { 0.0 };
    let r24 = if n >= 289 { (c / cl[n - 289] - 1.0) * 1e4 } else { 0.0 };
    let s = 0.497 - 0.027 * r1 / 41.7 - 0.036 * r24 / 125.6;
    if s.is_nan() { s } else { s.max(0.05).min(0.95) }
}

#[derive(Default, Clone)]
pub struct Book {
    t: Vec<i64>,
    h: Vec<f64>,
    l: Vec<f64>,
    x: Vec<f64>,
    xl: Vec<f64>,
    xs: Vec<f64>,
    cl: Vec<f64>,
    old: f64,
    old_l: f64,
    old_s: f64,
    pub oi: Option<f64>,
}

/// _step(복사본) → (x', old', 새 봉 양)
fn step(x: &[f64], mut old: f64, oi_prev: f64, oi: f64, v: f64) -> (Vec<f64>, f64, f64) {
    let d = oi - oi_prev;
    let rem = pymax(-d, 0.0) + K * v;
    let mut x = x.to_vec();
    if rem > 0.0 {
        let n = x.len();
        let w: Vec<f64> = (0..n).map(|i| (-((n - i) as i64) as f64 / TAU).exp() + RHO).collect();
        let wx: Vec<f64> = (0..n).map(|i| x[i] * w[i]).collect();
        let tw = np_sum(&wx) + old * RHO;
        if tw > 0.0 {
            let wmax = if n > 0 { w.iter().copied().fold(f64::NEG_INFINITY, f64::max) } else { 0.0 };
            let f = pymin(rem / tw, 1.0 / pymax(wmax, RHO));
            for i in 0..n {
                x[i] -= x[i].min(wx[i] * f);
            }
            old = pymax(old - old * RHO * f, 0.0);
        }
    }
    let mut new = pymax(d, 0.0) + K * v;
    let tot = np_sum(&x) + new + old;
    if tot > 0.0 {
        let r = oi / tot;
        x.iter_mut().for_each(|v| *v *= r);
        old *= r;
        new *= r;
    }
    (x, old, new)
}

impl Book {
    pub fn push(&mut self, t: i64, h: f64, l: f64, v: f64, oi: f64, c: f64) {
        if !(oi.is_finite() && oi > 0.0) {
            return;
        }
        self.cl.push(c);
        if self.cl.len() > 289 {
            self.cl.remove(0);
        }
        let Some(prev) = self.oi else {
            (self.oi, self.old, self.old_l, self.old_s) = (Some(oi), oi, oi, oi);
            return;
        };
        let v = pymax(v, 0.0);
        let s = side_share(&self.cl);
        let (mut x, old, new) = step(&self.x, self.old, prev, oi, v);
        let (mut xl, old_l, nl) = step(&self.xl, self.old_l, prev, oi, 2.0 * s * v);
        let (mut xs, old_s, ns) = step(&self.xs, self.old_s, prev, oi, 2.0 * (1.0 - s) * v);
        (x.push(new), xl.push(nl), xs.push(ns));
        (self.x, self.xl, self.xs, self.old, self.old_l, self.old_s) = (x, xl, xs, old, old_l, old_s);
        (self.t.push(t), self.h.push(h), self.l.push(l));
        self.oi = Some(oi);
        if self.x.len() > RING {
            let k = self.x.len() - RING;
            self.old += np_sum(&self.x[..k]);
            self.old_l += np_sum(&self.xl[..k]);
            self.old_s += np_sum(&self.xs[..k]);
            for a in [&mut self.h, &mut self.l, &mut self.x, &mut self.xl, &mut self.xs] {
                a.drain(..k);
            }
            self.t.drain(..k);
        }
    }

    /// snapshot(forming): 형성 봉을 임시 한 걸음으로 얹은 (t, h, l, x, xl, xs) -- 상태는 안 바뀜
    fn snapshot(&self, forming: Option<(i64, f64, f64, f64, f64)>) -> (Vec<i64>, [Vec<f64>; 5]) {
        let mut out = (self.t.clone(), [self.h.clone(), self.l.clone(), self.x.clone(), self.xl.clone(), self.xs.clone()]);
        let (Some((ft, fh, fl, fv, foi)), Some(prev)) = (forming, self.oi) else { return out };
        let fv = pymax(fv, 0.0);
        let s = if self.cl.is_empty() { 0.5 } else { side_share(&self.cl) };
        let (x, _, new) = step(&self.x, self.old, prev, foi, fv);
        let (xl, _, nl) = step(&self.xl, self.old_l, prev, foi, 2.0 * s * fv);
        let (xs, _, ns) = step(&self.xs, self.old_s, prev, foi, 2.0 * (1.0 - s) * fv);
        out.0.push(ft);
        let a = &mut out.1;
        (a[0].push(fh), a[1].push(fl));
        (a[2], a[3], a[4]) = (x, xl, xs);
        (a[2].push(new), a[3].push(nl), a[4].push(ns));
        out
    }
}

/// bins: 봉마다 남은 양을 고~저 사이 bw 칸에 고르게 → (첫 칸 번호, 칸 값) · np.add.at = 봉·칸 차례로 더함
fn bins(h: &[f64], l: &[f64], x: &[f64], bw: f64) -> (i64, Vec<f64>) {
    let m: Vec<usize> = (0..x.len()).filter(|&i| x[i] > 0.0).collect();
    if m.is_empty() {
        return (0, Vec::new());
    }
    let hi: Vec<f64> = m.iter().map(|&i| { let b = l[i] + 1e-9; if h[i] >= b || h[i].is_nan() { h[i] } else { b } }).collect();
    let k0: Vec<i64> = m.iter().map(|&i| (l[i] / bw).floor() as i64).collect();
    let k1: Vec<i64> = hi.iter().map(|v| (v / bw).floor() as i64).collect();
    let lo = *k0.iter().min().unwrap();
    let mut out = vec![0.0; (*k1.iter().max().unwrap() - lo + 1) as usize];
    for (j, &i) in m.iter().enumerate() {
        for k in k0[j]..=k1[j] {
            let (a, b) = (((k + 1) as f64 * bw), (k as f64 * bw));
            let ov = (if a < hi[j] { a } else { hi[j] }) - (if b > l[i] { b } else { l[i] });
            let ov = (if ov >= 0.0 { ov } else { 0.0 }) / (hi[j] - l[i]);
            out[(k - lo) as usize] += x[i] * ov;
        }
    }
    (lo, out)
}

fn dist(h: &[f64], l: &[f64], q: &[f64], m: &[bool], bw: f64) -> Map<String, V> {
    let pick = |a: &[f64]| a.iter().zip(m).filter(|p| *p.1).map(|p| *p.0).collect::<Vec<f64>>();
    let (lo, vals) = bins(&pick(h), &pick(l), &pick(q), bw);
    let tot = np_sum(&vals);
    let avg = if tot > 0.0 {
        let p: Vec<f64> = vals.iter().enumerate().map(|(i, v)| ((i as i64 + lo) as f64 + 0.5) * bw * v).collect();
        json!(round_nd(np_sum(&p) / tot, 6))
    } else {
        V::Null
    };
    let mut d = Map::new();
    d.insert("lo".into(), json!(lo));
    d.insert("vals".into(), json!(vals.iter().map(|v| (v * 10.0).round_ties_even() / 10.0).collect::<Vec<f64>>()));
    d.insert("tot".into(), json!(round_nd(tot, 1)));
    d.insert("avg".into(), avg);
    d
}

pub fn payload(book: &Book, now_s: i64, price: Option<f64>, forming: Option<(i64, f64, f64, f64, f64)>, bw: f64, windows: &[i64], rng: Option<(i64, i64)>) -> V {
    let (t, [h, l, x, xl, xs]) = book.snapshot(forming);
    let mut masks: Vec<(String, Vec<bool>)> = windows.iter().map(|d| (d.to_string(), t.iter().map(|&s| s >= now_s - d * 86400).collect())).collect();
    if let Some((a, b)) = rng {
        masks.push(("range".into(), t.iter().map(|&s| s >= a && s < b).collect()));
    }
    let mut win = Map::new();
    for (key, m) in masks {
        let mut w = dist(&h, &l, &x, &m, bw);
        w.insert("hours".into(), json!(round_nd(m.iter().filter(|b| **b).count() as f64 / 12.0, 1)));
        w.insert("long".into(), V::Object(dist(&h, &l, &xl, &m, bw)));
        w.insert("short".into(), V::Object(dist(&h, &l, &xs, &m, bw)));
        win.insert(key, V::Object(w));
    }
    if rng.is_some() {
        win["range"].as_object_mut().unwrap().insert("first_s".into(), t.first().map_or(V::Null, |v| json!(v)));
    }
    json!({"asof": now_s, "price": price, "oi": book.oi, "bw": bw, "win": win,
           "basis": "추정(5분 OI·거래량 · 회전+나이 가중 닫기 · 롱·숏 = 손바뀜을 직전 1h·24h 수익으로 가른 근사) -- 실측 포지션 아님, 지도로만"})
}

// ───────────────────────── 서버 몫(pos_*) ─────────────────────────

pub struct St {
    pub book: Book,
    last: i64,
    tried: i64,
    off: Option<f64>,
}

/// 원본 pos_books · pos_seed_fail(코인마다) -- Config 에 하나
#[derive(Default)]
pub struct State {
    books: HashMap<String, St>,
    seed_fail: HashMap<String, i64>,
}

type Bar5 = (i64, f64, f64, f64, f64, f64);

fn get(cfg: &Config, url: String) -> R<V> {
    Ok(fetch_binance(cfg, &url, Some(REASON)).map_err(PyErr)?.unwrap_or(V::Null))
}

fn fl(v: &V) -> R<f64> {
    crate::py::to_float(v)
}

fn int(v: &V) -> R<i64> {
    v.as_i64().or_else(|| v.as_str().and_then(|s| crate::api::py_int_str(s)).map(|x| x as i64)).ok_or_else(|| PyErr("TypeError: int".into()))
}

/// pos_bars_rest: 닫힌 5분봉 [start_s, end_s) -- OI 이력(500쪽) 다 받고 klines(1500쪽) · 끝 스냅샷 없는 봉은 뺀다
fn bars_rest(cfg: &Config, sym: &str, start_s: i64, end_s: i64) -> R<Vec<Bar5>> {
    let step = 300_000i64;
    let mut oi: HashMap<i64, f64> = HashMap::new();
    let mut t = start_s * 1000;
    while t < end_s * 1000 {
        let rows = get(cfg, format!("{}/futures/data/openInterestHist?symbol={sym}&period=5m&limit=500&startTime={t}&endTime={}", cfg.rest, (t + 499 * step).min(end_s * 1000)))?;
        for r in rows.as_array().map_or(&[][..], |a| a.as_slice()) {
            oi.insert(int(&r["timestamp"])?.div_euclid(1000), fl(&r["sumOpenInterest"])?);
        }
        t += 500 * step;
    }
    let mut out = Vec::new();
    t = start_s * 1000;
    while t < end_s * 1000 {
        let rows = get(cfg, format!("{}/fapi/v1/klines?symbol={sym}&interval=5m&limit=1500&startTime={t}&endTime={}", cfg.rest, end_s * 1000 - 1))?;
        let Some(rows) = rows.as_array().filter(|a| !a.is_empty()) else { break };
        for r in rows {
            let b = int(&r[0])?.div_euclid(1000);
            if let Some(&o) = oi.get(&(b + 300)) {
                if b < end_s {
                    out.push((b, fl(&r[2])?, fl(&r[3])?, fl(&r[5])?, o, fl(&r[4])?));
                }
            }
        }
        t = int(&rows[rows.len() - 1][0])? + step;
    }
    Ok(out)
}

/// 흐름 엔진의 1초 OI 링(coin_flow collect_oi_1s): hot oi_1s 의 (지금 − 660초 − 30초, 지금] → 초 → 마지막 값 · 지금 − 660초 앞은 버림
/// ponytail: 부를 때마다 hot 을 읽는다(1초 캐시 뒤라 초에 한 번) -- 위 끝(지금)은 원본 링에 «아직 안 온» 행을 넣지 않으려는 것
fn oi_ring(cfg: &Config, sym: &str) -> BTreeMap<i64, f64> {
    let now = (cfg.now)();
    let (lo, hi) = (((now - 660.0) * 1000.0) as i64 - 30_000, (now * 1000.0) as i64);
    let Ok(rows) = crate::api::rows(&cfg.root.join("data/hot/binance_ctx.sqlite"), "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms > ? AND ts_ms <= ? ORDER BY ts_ms",
        &[rusqlite::types::Value::Text(sym.into()), rusqlite::types::Value::Integer(lo), rusqlite::types::Value::Integer(hi)]) else { return BTreeMap::new() };
    let mut ring = BTreeMap::new();
    for r in &rows {
        if let (Some(t), Some(v)) = (r[0].as_i64(), r[1].as_f64()) {
            ring.insert(t.div_euclid(1000), v);
        }
    }
    if !rows.is_empty() {
        let cut = now.trunc() as i64 - 660;
        ring.retain(|k, _| *k >= cut);
    }
    ring
}

/// live_at: t 이하 가장 가까운 값(30초 안)
fn live_at(ring: &BTreeMap<i64, f64>, t: i64) -> Option<f64> {
    ring.range(t - 30..=t).next_back().map(|p| *p.1)
}

fn offset(cfg: &Config, sym: &str, st: &mut St) {
    if let (Some(live), Some(oi)) = (live_at(&oi_ring(cfg, sym), st.last + 300), st.book.oi) {
        st.off = Some(oi - live);
    }
}

/// pos_ensure: 처음 = 30일 적재(실패 = 60초 백오프 · 502) · 새 5분봉이 닫혔으면 1분에 한 번 밀기(실패 = 로그만)
fn ensure(cfg: &Config, asset: &str, ps: &mut State) -> R<()> {
    let sym = format!("{}USDT", asset.to_uppercase());
    let low = sym.to_lowercase();
    let now = (cfg.now)().trunc() as i64;
    let cur = now.div_euclid(300) * 300;
    let Some(st) = ps.books.get_mut(asset) else {
        if now - ps.seed_fail.get(asset).copied().unwrap_or(0) < SEED_BACKOFF {
            return Err(PyErr("HTTPServiceUnavailable: position_profile_seed_backoff".into()));
        }
        let bars = match bars_rest(cfg, &sym, cur - 30 * 86400 + 300, cur) {
            Ok(b) => b,
            Err(PyErr(m)) => {
                ps.seed_fail.insert(asset.into(), now);
                (cfg.log)(&format!("position profile seed failed (retry in {SEED_BACKOFF}s): {m}"));
                return Err(PyErr(format!("HTTPBadGateway: {REASON}")));
            }
        };
        let mut book = Book::default();
        for b in &bars {
            book.push(b.0, b.1, b.2, b.3, b.4, b.5);
        }
        let mut st = St { book, last: bars.last().map_or(cur - 300, |b| b.0), tried: now, off: None };
        offset(cfg, &low, &mut st);
        ps.books.insert(asset.into(), st);
        return Ok(());
    };
    if cur - st.last > 300 && now - st.tried >= 60 {
        st.tried = now;
        match bars_rest(cfg, &sym, st.last + 300, cur) {
            Ok(bs) => {
                for b in bs {
                    if b.0 > st.last {
                        st.book.push(b.0, b.1, b.2, b.3, b.4, b.5);
                        st.last = b.0;
                    }
                }
                offset(cfg, &low, st);
            }
            Err(PyErr(m)) => (cfg.log)(&format!("position profile advance failed (retry next minute): {m}")),
        }
    } else if st.off.is_none() {
        offset(cfg, &low, st);
    }
    Ok(())
}

/// pos_forming: (형성 봉, 마지막 체결가) -- 칸 = 허브 풋프린트 봉(since 이후) · 가격 = 허브 Supply1s 의 가장 늦은 초(값 있는 것)
fn forming(cfg: &Config, sym: &str, st: &St, bucket: f64) -> (Option<(i64, f64, f64, f64, f64)>, Option<f64>) {
    let g = cfg.live.syms.lock().unwrap();
    let s = g.get(sym);
    let px = s.and_then(|s| s.supply.as_ref()).and_then(|sp| {
        sp.seconds.iter().chain(sp.partial.bn.iter()).filter(|r| r.7 != 0.0).max_by_key(|r| r.0).map(|r| r.7)
    });
    let since = st.last + 300;
    let (mut ks, mut v) = (Vec::new(), 0.0);
    for (_, (lv, _)) in s.map(|s| s.fp_bars.range(since..)).into_iter().flatten() {
        for l in lv.values() {
            ks.push((l.0 / bucket).round() as i64);
            v += l.1 + l.2;
        }
    }
    drop(g);
    let ring = oi_ring(cfg, sym);
    let (Some(&kmax), Some(&kmin), Some(off), Some((_, &last))) = (ks.iter().max(), ks.iter().min(), st.off, ring.iter().next_back()) else { return (None, px) };
    let now = (cfg.now)().trunc() as i64;
    (Some((now.div_euclid(300) * 300, (kmax as f64 + 0.5) * bucket, (kmin as f64 - 0.5) * bucket, v, last + off)), px)
}

fn api_position_profile(cfg: &Config, req: &Req) -> Resp {
    let asset = match crate::api::coin_asset(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    if !cfg.flows().contains(&asset.as_str()) {
        return http_exc(404, Some("flow_off"), &[]);
    }
    let q = req.q("days");
    let mut wins: Vec<i64> = match q {
        Some(d @ ("1" | "7" | "30")) => vec![d.parse().unwrap()],
        _ => WINDOWS.to_vec(),
    };
    let mut rng = None;
    if req.q("from_s").is_some() {
        let g = |k: &str| crate::api::py_int_str(req.q(k).unwrap_or("")).map(|v| v.clamp(i64::MIN as i128, i64::MAX as i128) as i64);
        let (Some(f_s), Some(t_s)) = (g("from_s"), g("to_s")) else { return http_exc(400, Some("bad_range"), &[]) };
        let g4 = |v: i64| v.div_euclid(14400) * 14400;
        let now_i = (cfg.now)().trunc() as i64;
        let (f_s, t_s) = (g4(f_s).max(g4(now_i - 31 * 86400)), g4(t_s).min(g4(now_i + 86400)));
        if f_s >= t_s {
            return http_exc(400, Some("bad_range"), &[]);
        }
        (rng, wins) = (Some((f_s, t_s)), Vec::new());
    }
    let key = format!("position_profile:{asset}:{}:{}", wins.iter().map(i64::to_string).collect::<Vec<_>>().join(","),
                      rng.map_or("None".into(), |(a, b)| format!("({a}, {b})")));
    let c = cfg.clone();
    let produce = Arc::new(move || {
        let st_all = c.posprof.clone();
        let mut ps = st_all.lock().unwrap(); // ponytail: 코인 전부 잠금 하나(원본은 코인마다) -- 첫 적재(REST 24회) 동안 다른 코인도 기다린다
        ensure(&c, &asset, &mut ps)?;
        let st = &ps.books[&asset];
        let sym = format!("{asset}usdt");
        let bw = if asset == "eth" { 1.0 } else { crate::hub::fp_bucket(&sym) };
        let (fm, px) = forming(&c, &sym, st, crate::hub::fp_bucket(&sym));
        Ok(payload(&st.book, (c.now)().trunc() as i64, px, fm, bw, &wins, rng))
    });
    match cfg.swr.get(&key, 1.0, 600.0, produce) {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) if m.starts_with("HTTP") => {
            let mut r = Resp::json(&json!({"error": REASON}));
            r.status = 502;
            r
        }
        Err(_) => server_error(),
    }
}

/// 자체점검(원본 selftest ①②③④): 양 합 = OI · 칸 합 보존 · 형성 봉 무해 · 30일 접힘
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_invariants() {
        let mut b = Book::default();
        let n = 9000;
        for i in 0..n {
            let (oi, px) = (1e6 + (i as f64 * 0.37).sin() * 5e4, 2500.0 + (i as f64 * 0.11).cos() * 30.0);
            b.push(i * 300, px + 2.0, px - 2.0, 1e3 + (i % 7) as f64 * 500.0, oi, px);
        }
        assert!((np_sum(&b.x) + b.old - b.oi.unwrap()).abs() < 1e-3 && b.x.len() == RING && b.t.len() == RING);
        assert!((np_sum(&b.xl) + b.old_l - b.oi.unwrap()).abs() < 1e-3);
        let (_, vals) = bins(&b.h, &b.l, &b.x, 1.0);
        assert!((np_sum(&vals) - np_sum(&b.x)).abs() < 1e-6 * np_sum(&b.x));
        let x0 = b.x.clone();
        b.snapshot(Some((n * 300, 2501.0, 2499.0, 2e3, 1.01e6)));
        assert_eq!(x0, b.x);
    }
}
