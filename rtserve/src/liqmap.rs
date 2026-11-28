//! 청산맵(2026-11-28): /api/liquidation-map · /api/liquidation-map/tiers -- 바깥 요청은 전부 model::fetch_binance(→ binance_get 한 문 · RTSERVE_REST 있을 때만).
//! 원본(Read 로 읽은 줄): server.py liq_klines_df 3322~3345 · liq_oi_usd 3347~3365 · load_liquidation_tiers 3367~3378 · liq_hist_memo·load_liquidation_map 3467~3516 ·
//!   api_liquidation_map 4804~4813 · api_liquidation_tiers 4877~4883 · 상수 197~223·272~273 · scripts/live_liquidation_map_20260824.py
//!   compute_spliced_levels 303~362 · _spliced_raw 463~476 · _spliced_history 479~501 · _bar_arrays·_window_at·latest_window_5m·compute_spliced_heatmap_history_5m 507~568 ·
//!   compute_tier_profile 574~616 · _m5_until 132~135. 가중·생존 필터·칸 = rtmodel::regime_band(prepare_hl·bins·side_levels·own_alive -- 같은 원본을 한 곳만).
//! 시험: tests/liqmap_parity.rs(정답지 = tools/gen_liqmap_fixture.py · 127.0.0.1 가짜 REST).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rtmodel::regime_band::{self as rb, Bins, Win};
use rusqlite::types::Value as Sql;
use serde_json::{json, Value as V};

use crate::ctx_rest::{parse_klines, Kline};
use crate::model::fetch_binance;
use crate::py::{round_nd, PyErr, R};
use crate::{http_exc, server_error, Config, Handler, Req, Resp};

const FETCH_1H: usize = 48;
const FETCH_5M: usize = 460;
const TTL: f64 = 60.0;
const STALE: f64 = 600.0;
const LOOKBACK_H: usize = 24;
const SNAPSHOTS: i64 = 144;
const MMR: f64 = 0.005;
const TIER_GROUPS: [(&str, &[f64]); 4] = [("10", &[10.0]), ("20~25", &[20.0, 25.0]), ("50", &[50.0]), ("75~100", &[75.0, 100.0])];
const REASON: &str = "liquidation_map_upstream_error";

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    let h: Handler = match p {
        "/api/liquidation-map" => api_liquidation_map,
        "/api/liquidation-map/tiers" => api_liquidation_tiers,
        _ => return Vec::new(),
    };
    vec![("GET", h)]
}

// ───────────────────────── 입력 ─────────────────────────

fn klines(cfg: &Config, q: &str) -> R<Vec<V>> {
    let v = fetch_binance(cfg, &format!("{}/fapi/v1/klines?{q}", cfg.rest), Some(REASON)).map_err(PyErr)?;
    v.and_then(|v| v.as_array().cloned()).ok_or_else(|| PyErr("TypeError: klines".into()))
}

/// liq_klines_df: 한 번 1,500 · 더 많으면 endTime 으로 앞쪽 · 정렬·같은 시각 마지막 · 형성 봉 버림(keep_forming 거짓).
pub fn liq_klines(cfg: &Config, sym: &str, interval: &str, limit: usize, keep_forming: bool) -> R<Vec<Kline>> {
    let base = format!("symbol={sym}&interval={interval}");
    let mut raw = klines(cfg, &format!("{base}&limit={}", limit.min(1500)))?;
    while !raw.is_empty() && raw.len() < limit && raw.len() % 1500 == 0 {
        let end = raw[0][0].as_i64().unwrap_or(0) - 1;
        let more = klines(cfg, &format!("{base}&limit={}&endTime={end}", (limit - raw.len()).min(1500)))?;
        if more.is_empty() {
            break;
        }
        let short = more.len() < 1500;
        raw = more.into_iter().chain(raw).collect();
        if short {
            break;
        }
    }
    let mut by = std::collections::BTreeMap::new();
    for k in parse_klines(&V::Array(raw)).map_err(PyErr)? {
        by.insert(k.open_ms, k);
    }
    let mut v: Vec<Kline> = by.into_values().collect();
    let now_ms = ((cfg.now)() * 1000.0) as i64;
    if !keep_forming && v.last().is_some_and(|k| k.close_ms >= now_ms) {
        v.pop();
    }
    Ok(v)
}

fn win(k: &[Kline]) -> Win {
    Win { t: k.iter().map(|x| x.open_ms.div_euclid(1000)).collect(), h: k.iter().map(|x| x.high).collect(), l: k.iter().map(|x| x.low).collect(),
          c: k.iter().map(|x| x.close).collect(), v: k.iter().map(|x| x.volume).collect() }
}

fn ss(a: &[i64], x: i64) -> usize {
    a.partition_point(|&v| v < x)
}

fn slice(w: &Win, a: usize, b: usize) -> Win {
    Win { t: w.t[a..b].to_vec(), h: w.h[a..b].to_vec(), l: w.l[a..b].to_vec(), c: w.c[a..b].to_vec(), v: w.v[a..b].to_vec() }
}

/// _window_at: 시작 < floor_hour(t) 인 1시간봉 + 그 시간 닫힌 5분봉 [floor_hour(t), t) 부분봉 한 줄(고 max · 저 min · 마지막 종가 · 거래량 np.sum).
fn window_at(h: &Win, m: &Win, t: i64, lb: usize) -> Win {
    let hour = t.div_euclid(3600) * 3600;
    let (p0, p1) = (ss(&m.t, hour), ss(&m.t, t));
    let i1 = ss(&h.t, hour);
    let i0 = (i1 + (p1 > p0) as usize).saturating_sub(lb);
    let mut w = slice(h, i0, i1);
    if p1 > p0 {
        w.t.push(hour);
        w.h.push(m.h[p0..p1].iter().copied().fold(f64::NEG_INFINITY, f64::max));
        w.l.push(m.l[p0..p1].iter().copied().fold(f64::INFINITY, f64::min));
        w.c.push(m.c[p1 - 1]);
        w.v.push(rtcollect::ob5m::np_sum(&m.v[p0..p1]));
    }
    w
}

fn mid(w: &Win) -> Vec<f64> {
    (0..w.t.len()).map(|i| (w.h[i] + w.l[i]) / 2.0).collect()
}

/// pd.Timestamp(초, tz=UTC).isoformat()
fn iso_s(t: i64) -> String {
    rtmodel::risk_radar::iso(t * 1000).replace('Z', "+00:00")
}

/// datetime.now(timezone.utc).isoformat()(마이크로초가 0 이면 안 붙음)
fn now_iso(cfg: &Config) -> String {
    let now = (cfg.now)();
    let (mut s, mut us) = (now.floor() as i64, ((now - now.floor()) * 1e6).round_ties_even() as i64); // datetime.fromtimestamp: round(frac × 1e6) 짝수 쪽
    if us >= 1_000_000 {
        (s, us) = (s + 1, us - 1_000_000);
    }
    let b = iso_s(s);
    if us == 0 { b } else { b.replace("+00:00", &format!(".{us:06}+00:00")) }
}

// ───────────────────────── 계산 ─────────────────────────

fn heat_side(b: &Bins, bw: f64, max: f64) -> Vec<V> {
    b.iter().map(|&(k, w)| json!({"price": round_nd(k as f64 * bw, 4), "weight_pct": round_nd(w / max, 4)})).collect()
}

fn by_price(mut v: Vec<V>) -> V {
    v.sort_by(|a, b| a["price"].as_f64().unwrap().partial_cmp(&b["price"].as_f64().unwrap()).unwrap()); // sorted(key=price) 안정
    V::Array(v)
}

fn maxw(b: &Bins) -> f64 {
    b.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max)
}

/// compute_spliced_levels(303~362) 반환 dict.
pub fn spliced_levels(w: &Win, cp: f64, m5: &Win) -> V {
    let n = w.t.len();
    let empty = |reason: &str| json!({"warmed_up": false, "error": reason, "current_price": null, "bars_used": n, "lookback_hours": 0.0, "support_levels": [], "resistance_levels": []});
    let Some(cm) = rb::prepare(w, cp, m5) else {
        return empty(if n < 20 || !(cp > 0.0) { "insufficient_data" } else { "no_surviving_levels" });
    };
    let (Some(bm), Some(bc)) = (rb::bins(&cm, &mid(w)), rb::bins(&cm, &cm.close)) else { return empty("no_surviving_levels") };
    let bw = cm.bw;
    let lv = |l: Vec<(f64, f64, f64)>| V::Array(l.into_iter().map(|(p, wp, d)| json!({"price": p, "weight_pct": wp, "distance_pct": d})).collect());
    let side = |b: &Bins, below: bool| -> Vec<V> {
        let s: Bins = b.iter().copied().filter(|&(k, _)| (k as f64 * bw < cp) == below).collect();
        let mx = if s.is_empty() { 0.0 } else { maxw(&s) };
        if mx > 0.0 { heat_side(&s, bw, mx) } else { vec![] }
    };
    let heat = by_price(side(&bm, true).into_iter().chain(side(&bc, false)).collect());
    json!({"warmed_up": true, "error": null, "current_price": cp, "bars_used": n, "lookback_hours": (w.t[n - 1] - w.t[0]) as f64 / 3600.0,
           "bin_width": round_nd(bw, 4), "heatmap_bins": heat,
           "support_levels": lv(rb::side_levels(&bm, bw, cp, true)), "resistance_levels": lv(rb::side_levels(&bc, bw, cp, false))})
}

/// (지지 칸, 저항 칸, 칸 폭) -- _spliced_raw
type Raw = (Bins, Bins, f64);

fn spliced_raw(w: &Win, cp: f64, m5: &Win) -> Option<Raw> {
    let cm = rb::prepare(w, cp, m5)?;
    let (bm, bc) = (rb::bins(&cm, &mid(w)), rb::bins(&cm, &cm.close));
    if bm.is_none() && bc.is_none() {
        return None;
    }
    let bw = cm.bw;
    Some((bm.unwrap_or_default().into_iter().filter(|x| (x.0 as f64) * bw < cp).collect(),
          bc.unwrap_or_default().into_iter().filter(|x| (x.0 as f64) * bw > cp).collect(), bw))
}

fn spliced_history(raws: &[(i64, Raw)]) -> V {
    let gm = |f: fn(&Raw) -> &Bins| raws.iter().map(|(_, r)| f(r)).filter(|b| !b.is_empty()).map(maxw).fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x)))).unwrap_or(0.0);
    let (gs, gr) = (gm(|r| &r.0), gm(|r| &r.1));
    if raws.is_empty() || !(gs > 0.0 || gr > 0.0) {
        return json!([]);
    }
    let fmt = |b: &Bins, bw: f64, g: f64| if g > 0.0 { heat_side(b, bw, g) } else { vec![] };
    V::Array(raws.iter().map(|(t, (s, r, bw))| json!({"ts_utc": iso_s(*t), "bins": by_price(fmt(s, *bw, gs).into_iter().chain(fmt(r, *bw, gr)).collect())})).collect())
}

/// _m5_until: 시각 t 전에 시작한 5분봉만.
fn m5_until(m: &Win, t: i64) -> Win {
    slice(m, 0, ss(&m.t, t))
}

type Memo = HashMap<(i64, u64), Option<Raw>>;
/// 원본 liq_hist_memo(코인마다 · 프로세스 수명) -- 여기는 (뿌리, 코인)마다(시험이 Config 를 여럿 만든다).
static MEMO: LazyLock<Mutex<HashMap<(PathBuf, String), Memo>>> = LazyLock::new(Default::default);

/// compute_spliced_heatmap_history_5m(539~568) -- memo 는 마지막 경계를 안 넣고 창 밖·옛 기준가를 버린다.
pub fn heatmap_history_5m(h: &Win, m: &Win, cp: f64, memo: &mut Memo) -> V {
    if m.t.is_empty() || !(cp > 0.0) {
        return json!([]);
    }
    let t_last = m.t[m.t.len() - 1] + 300;
    let first = t_last - (SNAPSHOTS - 1) * 300;
    let mut raws = vec![];
    for t in (first..=t_last).step_by(300) {
        let key = (t, cp.to_bits());
        let raw = match memo.get(&key) {
            Some(r) => r.clone(),
            None => {
                let r = spliced_raw(&window_at(h, m, t, LOOKBACK_H), cp, &m5_until(m, t));
                if t < t_last {
                    memo.insert(key, r.clone());
                }
                r
            }
        };
        if let Some(r) = raw {
            raws.push((t, r));
        }
    }
    memo.retain(|k, _| k.1 == cp.to_bits() && k.0 >= first);
    spliced_history(&raws)
}

/// np.round(x, d)(d > 0): x × 10^d → rint → ÷ 10^d
fn np_round(x: f64, d: i32) -> f64 {
    let f = 10f64.powi(d);
    (x * f).round_ties_even() / f
}

/// compute_tier_profile(577~616). hl = None 이면 감쇠 끔.
pub fn tier_profile(w: &Win, cp: f64, oi_usd: Option<f64>, m5: &Win, hl: Option<f64>) -> V {
    let Some(cm) = rb::prepare_hl(w, cp, m5, hl) else { return V::Null };
    let (md, close, bw) = (mid(w), &cm.close, cm.bw);
    let wt: Vec<f64> = cm.base.iter().map(|b| b / 6.0).collect();
    let mut per: Vec<(&str, Vec<i64>, Vec<f64>)> = vec![];
    for (name, levs) in TIER_GROUPS {
        let (mut ks, mut vs) = (vec![], vec![]);
        for &lev in levs {
            let ll: Vec<f64> = md.iter().map(|x| x * (1.0 - 1.0 / lev + MMR)).collect();
            let sl: Vec<f64> = close.iter().map(|x| x * (1.0 + 1.0 / lev - MMR)).collect();
            for (entry, liq, long) in [(&md, &ll, true), (close, &sl, false)] {
                let own = rb::own_alive(&cm.sub, entry, liq, long);
                for i in 0..liq.len() {
                    let alive = if long { cm.fmin[i] > liq[i] } else { cm.fmax[i] < liq[i] };
                    if alive && liq[i] > 0.0 && own[i] > 0.0 {
                        ks.push((liq[i] / bw).round_ties_even() as i64);
                        vs.push(wt[i] * own[i] * entry[i]);
                    }
                }
            }
        }
        per.push((name, ks, vs));
    }
    let all: Vec<i64> = per.iter().flat_map(|p| p.1.iter().copied()).collect();
    if all.is_empty() {
        return V::Null;
    }
    let lo = *all.iter().min().unwrap();
    let n = (*all.iter().max().unwrap() - lo + 1) as usize;
    let total = per.iter().fold(0.0, |a, p| a + rtcollect::ob5m::np_sum(&p.2));
    if !(total > 0.0) {
        return V::Null;
    }
    let oi = oi_usd.filter(|x| *x > 0.0);
    let (scale, unit) = match oi { Some(o) => (2.0 * o / total, "oi_usd"), None => (1.0 / total, "share") };
    let tiers: Vec<V> = per.iter().map(|(name, k, v)| {
        let mut arr = vec![0.0; n];
        for (k, v) in k.iter().zip(v) {
            arr[(k - lo) as usize] += v * scale;
        }
        json!({"name": name, "values": arr.into_iter().map(|x| np_round(x, if unit == "oi_usd" { 2 } else { 8 })).collect::<Vec<_>>()})
    }).collect();
    json!({"bin_width": round_nd(bw, 6), "lo": lo, "current_price": cp, "tiers": tiers, "unit": unit,
           "oi_usd": oi.map(|o| round_nd(o, 0)), "basis": "추정(지난 1시간봉들 = 가상 진입 · 거래량×최근성 가중) -- 실측 포지션 아님"})
}

fn first_f64(db: &Path, sql: &str, p: &[Sql]) -> Option<f64> {
    crate::api::rows(db, sql, p).ok()?.first()?.first()?.as_f64().filter(|x| *x != 0.0)
}

/// liq_oi_usd: 바이낸스 hot oi_1s 마지막 + OKX·바이비트(10분 안 값만) × 가격 -- 바이낸스가 없으면 None.
pub fn liq_oi_usd(root: &Path, sym: &str, cp: f64, now: f64) -> Option<f64> {
    let hot = root.join("data/hot");
    let mut coins = first_f64(&hot.join("binance_ctx.sqlite"), "SELECT open_interest FROM oi_1s WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1", &[Sql::Text(sym.to_lowercase())])?;
    let since = (now * 1000.0) as i64 - 600_000;
    let base = sym.strip_suffix("USDT").unwrap_or(sym);
    for (f, tbl, key) in [("okx_ctx.sqlite", "okx_oi", format!("{base}-USDT-SWAP")), ("bybit_ctx.sqlite", "bybit_oi", sym.to_string())] {
        let p = hot.join(f);
        if p.exists() {
            coins += first_f64(&p, &format!("SELECT oi_base FROM {tbl} WHERE inst = ? AND ts_ms >= ? ORDER BY ts_ms DESC LIMIT 1"), &[Sql::Text(key), Sql::Integer(since)]).unwrap_or(0.0);
        }
    }
    Some(coins * cp)
}

fn produce_map(cfg: &Config, asset: &str) -> R<V> {
    let sym = format!("{}USDT", asset.to_uppercase());
    let df = liq_klines(cfg, &sym, "1h", FETCH_1H, false)?;
    let d5 = liq_klines(cfg, &sym, "5m", FETCH_5M, false)?;
    let cp = df.last().map_or(0.0, |k| k.close);
    let (h, m) = (win(&df), win(&d5));
    let mut p = spliced_levels(&slice(&h, h.t.len().saturating_sub(LOOKBACK_H), h.t.len()), cp, &m);
    let hist = {
        let mut all = MEMO.lock().unwrap();
        heatmap_history_5m(&h, &m, cp, all.entry((cfg.root.clone(), asset.to_string())).or_default())
    };
    let o = p.as_object_mut().unwrap();
    o.insert("heatmap_history".into(), hist);
    let mut tp = V::Null;
    if !m.t.is_empty() {
        let t_last = m.t[m.t.len() - 1] + 300;
        let w = window_at(&h, &m, t_last, LOOKBACK_H);
        if !w.t.is_empty() {
            let end_px = w.c[w.c.len() - 1];
            let oi = liq_oi_usd(&cfg.root, &sym, end_px, (cfg.now)());
            tp = tier_profile(&w, cp, oi, &m, Some(240.0));
            if let Some(t) = tp.as_object_mut() {
                t.insert("current_price".into(), json!(end_px));
                t.insert("known_until".into(), json!(iso_s(t_last)));
            }
        }
    }
    if let Some(&t) = h.t.last() {
        o.insert("levels_price".into(), json!(cp));
        o.insert("levels_known_until".into(), json!(iso_s(t + 3600)));
    }
    o.insert("tier_profile".into(), tp);
    o.insert("generated_at".into(), json!(now_iso(cfg)));
    Ok(p)
}

/// 원본이 잡지 않는 예외 갈래: 밴·가중치 = aiohttp 가 HTTPServiceUnavailable 을 그대로(503 글자) · 그 밖 = 500.
fn uncaught(m: &str) -> Resp {
    match m.strip_prefix("HTTPServiceUnavailable: ") {
        Some(r) => http_exc(503, Some(r), &[]),
        // RestOff(RTSERVE_REST 없음) = 원본에 없는 갈래 -- 밴과 같은 «안 보냄» 503
        None if m.starts_with("RestOff") => http_exc(503, Some("rest_off"), &[]),
        None => server_error(),
    }
}

fn bad_gateway(v: V) -> Resp {
    let mut r = Resp::json(&v);
    r.status = 502;
    r
}

fn api_liquidation_map(cfg: &Config, req: &Req) -> Resp {
    let asset = match crate::api::coin_asset(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let (c, a) = (cfg.clone(), asset.clone());
    match cfg.swr.get(&format!("liquidation_map:{asset}"), TTL, STALE, Arc::new(move || produce_map(&c, &a))) {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) if m.starts_with("HTTPBadGateway") => bad_gateway(json!({"error": REASON, "detail": "Binance klines fetch failed."})),
        Err(PyErr(m)) => uncaught(&m),
    }
}

fn api_liquidation_tiers(cfg: &Config, req: &Req) -> Resp {
    let q = req.q("days").unwrap_or("1");
    let days: i64 = if ["1", "7", "30", "60", "90", "180", "365"].contains(&q) { q.parse().unwrap() } else { 1 };
    let asset = match crate::api::coin_asset(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let (c, a) = (cfg.clone(), asset.clone());
    let r = cfg.swr.get(&format!("liquidation_tiers:{asset}:{days}"), if days < 365 { TTL } else { 300.0 }, STALE, Arc::new(move || {
        let sym = format!("{}USDT", a.to_uppercase());
        let n = (days * 24) as usize;
        let df = liq_klines(&c, &sym, "1h", n + 2, true)?;
        let w = win(&df[df.len().saturating_sub(n)..]);
        let cp = w.c.last().copied().unwrap_or(0.0);
        let oi = liq_oi_usd(&c.root, &sym, cp, (c.now)());
        let tp = tier_profile(&w, cp, oi, &win(&[]), if days >= 60 { None } else { Some(240.0) });
        Ok(json!({"days": days, "tier_profile": tp, "decay": days < 60, "generated_at": now_iso(&c)}))
    }));
    match r {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) if m.starts_with("HTTPBadGateway") => bad_gateway(json!({"error": REASON})),
        Err(PyErr(m)) => uncaught(&m),
    }
}
