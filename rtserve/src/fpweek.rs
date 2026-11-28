//! 풋프린트 1주일(2026-11-28): /api/footprint-week -- 지난 7일 5분봉 칸(바이낸스 hot 테이프) · OI(바이낸스 + OKX) · 청산(바이낸스 + OKX + Bybit) · 캔들(바이낸스 5분봉).
//! 원본(Read 로 읽은 줄): server.py fp_week_part 4098~4155 · fp_week_sync 4157~4187 · fp_week_candles 4189~4211 · api_footprint_week 4213~4228 ·
//!   coin_flow.py okx_tape_footprint 132~206 · FLOW_SPECS 272~281 · footprint_daily.py week_bars 522~536 · oi_deltas 539~549.
//! 바깥 요청 = 캔들 klines 뿐(model::fetch_binance · RTSERVE_REST 있을 때만 · 못 받으면 원본처럼 캔들 없이 칸·레인만).
//! 시험: tests/routes6_parity.rs(정답지 = tools/gen_fpweek_fixture.py).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

use rusqlite::types::Value as Sql;
use serde_json::{json, Value as V};

use crate::api::{int_v, rows};
use crate::ctx_rest::parse_klines;
use crate::model::fetch_binance;
use crate::py::{round0, round_nd, to_float, PyErr, R};
use crate::{http_exc, server_error, Config, Handler, Req, Resp};

const BAR: i64 = 300;
const DAYS: i64 = 7;
const VWMA_PRE_S: i64 = 100 * 3600;

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/footprint-week" => vec![("GET", api_footprint_week)],
        _ => Vec::new(),
    }
}

/// FLOW_SPECS(coin_flow.py 272~281): (칸 폭, 가격 자릿수)
fn spec(asset: &str) -> (f64, usize) {
    let dp = match asset { "xrp" => 4, "btc" => 0, _ => 2 };
    (crate::hub::fp_bucket(&format!("{asset}usdt")), dp)
}

type Cells = BTreeMap<i64, [f64; 6]>;

struct Tape {
    bars: BTreeMap<i64, Cells>,
    ok_min: BTreeSet<i64>,
}

fn sq(e: rusqlite::Error) -> String {
    format!("sqlite3.Error: {e}")
}

/// okx_tape_footprint 의 칸·온전한 분 부분(봉 300초 · ts_sec < t0)
fn tape(db: &Path, inst: &str, lo: i64, t0: i64, bucket: f64, tape_bucket: f64, now: i64) -> Result<Tape, String> {
    let q = "SELECT ts_sec - ts_sec % ? AS bar, price_bin, sum(buy_qty), sum(sell_qty), sum(coalesce(whale_buy_qty, 0)), sum(coalesce(whale_sell_qty, 0)),
                    sum(coalesce(retail_buy_qty, 0)), sum(coalesce(retail_sell_qty, 0)) FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? GROUP BY 1, 2";
    let s = |x: &str| Sql::Text(x.into());
    let mut bars: BTreeMap<i64, Cells> = BTreeMap::new();
    let mut key: HashMap<i64, i64> = HashMap::new();
    for r in rows(db, q, &[Sql::Integer(BAR), s(inst), Sql::Integer(lo), Sql::Integer(t0)]).map_err(sq)? {
        let pb = int_v(&r[1]).map_err(|e| e.0)?;
        let k = *key.entry(pb).or_insert_with(|| (round_nd(pb as f64 * tape_bucket, 10) / bucket).round_ties_even() as i64);
        let t = bars.entry(int_v(&r[0]).map_err(|e| e.0)?).or_default().entry(k).or_insert([0.0; 6]);
        for j in 0..6 {
            t[j] += if r[2 + j].is_null() { 0.0 } else { to_float(&r[2 + j]).map_err(|e| e.0)? };
        }
    }
    let ver = rows(db, "SELECT ts_min, rel_err FROM verify_1m WHERE symbol = ? AND ts_min >= ?", &[s(inst), Sql::Integer(lo)]).map_err(sq)?;
    let gaps = rows(db, "SELECT from_ms, to_ms FROM gaps WHERE symbol = ? AND to_ms > ?", &[s(inst), Sql::Integer(lo * 1000)]).map_err(sq)?;
    let gaps: Vec<(i64, i64)> = gaps.iter().map(|g| (int_v(&g[0]).unwrap_or(0), int_v(&g[1]).unwrap_or(0))).collect();
    let mut ok_min = BTreeSet::new();
    let mut seen = BTreeSet::new();
    for v in &ver {
        let m = int_v(&v[0]).map_err(|e| e.0)?;
        if to_float(&v[1]).map_err(|e| e.0)?.abs() <= 1e-4 {
            ok_min.insert(m);
        }
        seen.insert(m);
    }
    let mut m = lo;
    while m < t0 + 60 {
        if !seen.contains(&m) && m >= now - 900 && !gaps.iter().any(|&(a, b)| a < (m + 60) * 1000 && b > m * 1000) {
            ok_min.insert(m);
        }
        m += 60;
    }
    Ok(Tape { bars, ok_min })
}

#[derive(Clone)]
struct Part {
    bars: Vec<V>,
    oi_bn: BTreeMap<i64, (f64, f64)>,
    oi_ox: BTreeMap<i64, (f64, f64)>,
    liq: BTreeMap<i64, [f64; 9]>,
    n: usize,
    ok: bool,
}

fn bar(ms: i64) -> i64 {
    ms.div_euclid(1000).div_euclid(BAR) * BAR
}

/// fp_week_part(f, d0, d1)
fn part(root: &Path, asset: &str, now: i64, d0: i64, d1: i64) -> Result<Part, String> {
    let (bucket, dp) = spec(asset);
    let (sym, symu) = (format!("{asset}usdt"), format!("{}USDT", asset.to_uppercase()));
    let inst = format!("{}-USDT-SWAP", asset.to_uppercase());
    let hot = root.join("data/hot");
    let mut fails: Vec<String> = vec![];
    let get = |fails: &mut Vec<String>, p: &Path, sql: &str, prm: &[Sql]| -> Vec<Vec<V>> {
        if !p.exists() {
            return vec![];
        }
        rows(p, sql, prm).unwrap_or_else(|e| {
            fails.push(format!("{}: {e}", p.display()));
            vec![]
        })
    };
    let tb = rtcore::restore::tape_bucket(&sym).ok_or_else(|| format!("KeyError: '{sym}'"))?;
    let bn = tape(&hot.join("binance_tape.sqlite"), &sym, d0, d1, bucket, tb, now)?; // 주 원천 -- 못 읽으면 응답 실패
    let (lo_ms, hi_ms) = (Sql::Integer(d0 * 1000), Sql::Integer(d1 * 1000));
    // OKX: 테이프는 tape_max 만(칸은 바이낸스만 -- 11-02) · OI·청산 = okx_ctx(OKX_LIQ_DB_PATH 도 같은 파일)
    let (otape, octx) = (hot.join("okx_tape.sqlite"), hot.join("okx_ctx.sqlite"));
    let mut oi_ox = BTreeMap::new();
    let mut oliq: Vec<(i64, bool, f64)> = vec![];
    if otape.exists() {
        let r = tape(&otape, &inst, d0, d1, bucket, rtcore::restore::okx_tape_bucket(&inst).unwrap_or(f64::NAN), now)
            .and_then(|_| rows(&otape, "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", &[Sql::Text(inst.clone())]).map_err(sq));
        match r {
            Ok(_) => {
                let oi = rows(&octx, "SELECT ts_ms, oi_base FROM okx_oi WHERE inst = ? AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms", &[Sql::Text(inst.clone()), lo_ms.clone(), hi_ms.clone()]);
                let lq = rows(&octx, "SELECT ts_ms, pos_side, sz_base, bk_px FROM okx_liquidations WHERE inst_id = ? AND ts_ms >= ? AND ts_ms < ? AND sz_base > 0 ORDER BY ts_ms",
                              &[Sql::Text(inst.clone()), lo_ms.clone(), hi_ms.clone()]);
                match oi {
                    Ok(v) => for r in v {
                        let (b, x) = (bar(int_v(&r[0]).map_err(|e| e.0)?), to_float(&r[1]).map_err(|e| e.0)?);
                        oi_ox.entry(b).and_modify(|o: &mut (f64, f64)| o.1 = x).or_insert((x, x));
                    },
                    Err(_) => fails.push("okx oi/liq".into()),
                }
                match lq {
                    Ok(v) => for r in v {
                        let px = if r[3].is_null() { 0.0 } else { to_float(&r[3]).map_err(|e| e.0)? };
                        oliq.push((int_v(&r[0]).map_err(|e| e.0)?, r[1].as_str() == Some("long"), to_float(&r[2]).map_err(|e| e.0)? * px));
                    },
                    Err(_) => if !fails.iter().any(|f| f == "okx oi/liq") { fails.push("okx oi/liq".into()) },
                }
            }
            Err(e) => fails.push(format!("okx tape: {e}")),
        }
    }
    let ctx = hot.join("binance_ctx.sqlite");
    let mut oi_bn = BTreeMap::new();
    for r in get(&mut fails, &ctx, "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms", &[Sql::Text(sym.clone()), lo_ms.clone(), hi_ms.clone()]) {
        let (b, x) = (bar(int_v(&r[0]).map_err(|e| e.0)?), to_float(&r[1]).map_err(|e| e.0)?);
        oi_bn.entry(b).and_modify(|o: &mut (f64, f64)| o.1 = x).or_insert((x, x));
    }
    let mut liq: BTreeMap<i64, [f64; 9]> = BTreeMap::new();
    let mut put = |ms: i64, long: bool, usd: f64, venue: usize| {
        let a = liq.entry(bar(ms)).or_insert([0.0; 9]);
        for j in [0, venue] {
            // 원본 그대로: 바이낸스(venue 0)는 (0, 0) 두 번 -- 합계 칸에 두 번 더한다
            a[j + if long { 0 } else { 1 }] += usd;
            a[j + 2] += 1.0;
        }
    };
    for r in get(&mut fails, &ctx, "SELECT ts_ms, side, usd FROM liquidations WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?", &[Sql::Text(symu.clone()), lo_ms.clone(), hi_ms.clone()]) {
        let usd = if r[2].is_null() { 0.0 } else { to_float(&r[2]).map_err(|e| e.0)? };
        put(int_v(&r[0]).map_err(|e| e.0)?, r[1].as_str() == Some("long"), usd, 0);
    }
    for (t, long, usd) in oliq {
        put(t, long, usd, 3);
    }
    for r in get(&mut fails, &hot.join("bybit_liq.sqlite"), "SELECT ts_ms, side, qty, price FROM bybit_liquidations WHERE symbol = ? AND ts_ms >= ? AND ts_ms < ?", &[Sql::Text(symu), lo_ms, hi_ms]) {
        put(int_v(&r[0]).map_err(|e| e.0)?, r[1].as_str() == Some("Buy"), to_float(&r[2]).map_err(|e| e.0)? * to_float(&r[3]).map_err(|e| e.0)?, 6);
    }
    // week_bars(bn, None, d0, d1): 바이낸스 테이프가 온전한(분 다 통과) 봉만
    let mut out = vec![];
    for b in (d0..d1).step_by(BAR as usize) {
        let Some(c) = bn.bars.get(&b) else { continue };
        if !(b..b + BAR).step_by(60).all(|m| bn.ok_min.contains(&m)) {
            continue;
        }
        let lv: Vec<V> = c.iter().map(|(k, v)| {
            let mut row = vec![json!(round_nd(*k as f64 * bucket, dp))];
            row.extend(v.iter().map(|x| json!(round_nd(*x, 3))));
            V::Array(row)
        }).collect();
        out.push(json!({"time": b, "levels": lv}));
    }
    if !fails.is_empty() {
        eprintln!("fp week {asset} {d0} 읽기 실패 -- 이 날은 안 굳힌다: {fails:?}");
    }
    let n = out.len();
    Ok(Part { bars: out, oi_bn, oi_ox, liq, n, ok: fails.is_empty() })
}

/// 원본 fp_week_days(닫히고 다 찬 날 = 불변) -- (뿌리, 코인, 날)마다(시험이 Config 를 여럿 만든다)
static DAYS_MEMO: LazyLock<Mutex<HashMap<(PathBuf, String, i64), Part>>> = LazyLock::new(Default::default);

/// oi_deltas: {봉: (첫, 끝)} → {봉: (Δ, 끝, 공백)}
fn oi_deltas(oc: &BTreeMap<i64, (f64, f64)>) -> BTreeMap<i64, (f64, f64, i64)> {
    let mut prev: Option<(i64, f64)> = None;
    oc.iter().map(|(&b, &(o, c))| {
        let gap = prev.is_none_or(|p| b - p.0 > BAR);
        let d = if gap { c - o } else { c - prev.unwrap().1 };
        prev = Some((b, c));
        (b, (d, c, gap as i64))
    }).collect()
}

fn iso(t: i64) -> String {
    rtmodel::risk_radar::iso(t * 1000).replace('Z', "+00:00")
}

/// fp_week_sync(f, now)
fn sync(root: &Path, asset: &str, now: f64) -> R<V> {
    let ni = now.floor() as i64;
    let today = ni.div_euclid(86_400) * 86_400;
    let first = today - DAYS * 86_400;
    let mut parts = vec![];
    for d in (first..today).step_by(86_400) {
        let key = (root.to_path_buf(), asset.to_string(), d);
        let memo = DAYS_MEMO.lock().unwrap().get(&key).cloned();
        let p = match memo { Some(p) => p, None => part(root, asset, ni, d, d + 86_400).map_err(PyErr)? };
        if p.n == (86_400 / BAR) as usize && p.ok {
            DAYS_MEMO.lock().unwrap().insert(key, p.clone());
        }
        parts.push(p);
    }
    parts.push(part(root, asset, ni, today, ni.div_euclid(BAR) * BAR).map_err(PyErr)?);
    DAYS_MEMO.lock().unwrap().retain(|k, _| !(k.0 == root && k.1 == asset && k.2 < first));
    let (mut oi_bn, mut oi_ox, mut liq) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for p in &parts {
        oi_bn.extend(p.oi_bn.clone());
        oi_ox.extend(p.oi_ox.clone());
        liq.extend(p.liq.clone());
    }
    let (dbn, dox) = (oi_deltas(&oi_bn), oi_deltas(&oi_ox));
    let oi: Vec<V> = dbn.iter().filter(|(b, _)| dox.is_empty() || dox.contains_key(b)).map(|(b, &(d, c, g))| {
        let (xd, xc) = dox.get(b).map_or((0.0, 0.0), |x| (x.0, x.1));
        json!([b, round_nd(d + xd, 3), round_nd(c + xc, 3), 0, g])
    }).collect();
    let det = |a: &[f64; 9], j: usize| -> R<Option<V>> {
        Ok(if a[j + 2] != 0.0 { Some(json!({"long_usd": round0(a[j])?, "short_usd": round0(a[j + 1])?, "n": a[j + 2] as i64})) } else { None })
    };
    let mut lo = vec![];
    for (b, a) in &liq {
        let mut m = serde_json::Map::new();
        m.insert("ts".into(), json!(iso(*b)));
        m.insert("long_usd".into(), json!(round_nd(a[0], 2)));
        m.insert("short_usd".into(), json!(round_nd(a[1], 2)));
        m.insert("events".into(), json!(a[2] as i64));
        for (k, j) in [("okx_detail", 3), ("bybit", 6)] {
            if let Some(v) = det(a, j)? {
                m.insert(k.into(), v);
            }
        }
        lo.push(V::Object(m));
    }
    Ok(json!({"bars": parts.iter().flat_map(|p| p.bars.clone()).collect::<Vec<_>>(), "oi": oi, "liq": lo}))
}

/// fp_week_candles(asset): 지난 7일 마감 5분봉(+ 밴드 워밍업) · 둘째 = 1주일 앞 100시간 [시각, 종가, 거래량]
fn candles(cfg: &Config, asset: &str) -> R<V> {
    let now = (cfg.now)();
    let week0 = ((now.floor() as i64).div_euclid(86_400) - DAYS) * 86_400;
    let mut start_ms = (week0 - (crate::ctx_rest::VETO_TAIL as i64 * BAR).max(VWMA_PRE_S)) * 1000;
    let sym = format!("{}USDT", asset.to_uppercase());
    let mut raw: Vec<V> = vec![];
    while raw.len() < 4000 {
        let url = format!("{}/fapi/v1/klines?symbol={sym}&interval=5m&startTime={start_ms}&limit=1000", cfg.rest);
        let got = fetch_binance(cfg, &url, Some("fp_week_klines")).map_err(PyErr)?;
        let got = got.and_then(|g| g.as_array().cloned()).unwrap_or_default();
        let n = got.len();
        if let Some(last) = got.last() {
            start_ms = int_v(&last[0])? + BAR * 1000;
        }
        raw.extend(got);
        if n < 1000 {
            break;
        }
    }
    let now_ms = now * 1000.0;
    let ks: Vec<_> = parse_klines(&V::Array(raw)).map_err(PyErr)?.into_iter().filter(|k| (k.close_ms as f64) < now_ms).collect();
    if ks.is_empty() {
        return Ok(json!([[], []]));
    }
    let pre: Vec<V> = ks.iter().filter(|k| { let t = k.open_ms.div_euclid(1000); week0 - VWMA_PRE_S <= t && t < week0 })
        .map(|k| json!([k.open_ms.div_euclid(1000), k.close, k.volume])).collect();
    let c: Vec<V> = crate::mhist::kline_candles(&ks, now_ms, if asset == "eth" { 4 } else { 6 }, ks.len()).into_iter()
        .filter(|c| c["time"].as_i64().unwrap_or(0) >= week0).collect();
    Ok(json!([c, pre]))
}

fn api_footprint_week(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if !cfg.flows().contains(&asset.as_str()) {
        return http_exc(404, Some("flow_off"), &[]);
    }
    let (c, a) = (cfg.clone(), asset.clone());
    let r = cfg.swr.get(&format!("fp_week_{asset}"), 300.0, 1800.0, Arc::new(move || {
        let (c2, a2) = (c.clone(), a.clone());
        let k = match c.swr.get(&format!("fp_week_k5_{a}"), 600.0, 3600.0, Arc::new(move || candles(&c2, &a2))) {
            Ok(v) => v,
            // 원본이 잡는 것 = HTTPException(밴·502)·ClientError·TimeoutError → 캔들 없이 칸·레인만 · RestOff(원본에 없는 갈래)도 같이
            Err(PyErr(m)) if m.starts_with("HTTP") || m.starts_with("ClientError") || m.starts_with("RestOff") => json!([[], []]),
            Err(e) => return Err(e),
        };
        let body = sync(&c.root, &a, (c.now)())?;
        let (bucket, _) = spec(&a);
        let (rm, wm) = crate::hub::band(&format!("{a}usdt"));
        let mut o = serde_json::Map::new();
        for (k2, v) in [("symbol", json!(format!("{}USDT", a.to_uppercase()))), ("bucket", json!(bucket)), ("barSeconds", json!(BAR)),
                        ("retailMaxUsd", json!(rm)), ("whaleMinUsd", json!(wm)), ("days", json!(DAYS)), ("candles", k[0].clone()), ("volPre", k[1].clone())] {
            o.insert(k2.into(), v);
        }
        for (k2, v) in body.as_object().unwrap() {
            o.insert(k2.clone(), v.clone());
        }
        o.insert("at".into(), json!((c.now)()));
        Ok(V::Object(o))
    }));
    match r {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) => {
            (cfg.log)(&format!("footprint-week {asset}: {m}"));
            server_error()
        }
    }
}
