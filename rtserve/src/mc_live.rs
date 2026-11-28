//! GET /api/market-context 붙이기(2026-11-28) -- 계산은 ctx_mc.rs(원본과 글자 대조 끝남) · 여기는 재료 모으기만(CTX.md «붙이는 쪽 사용법»).
//! 원본(Read 로 본 줄): server.py market_context_payload 2607~2714 · api_market_context 2723~2737(3초 · 120초 안 묵은 값) ·
//! collect_mark_price 1879~1916(마크 링 = 초 → (마크, 지수, 펀딩) · MARK_PRICE_RING_S 창 = rtcore::situation::MarkRing 이 같은 규칙).
//! 메모리 몫: 마크 링(허브 Mark) · 미시 payload·미드 링·depth25(hub SitRun.micro -- ETH 만 돈다) · x(situation_tick 셋째 값) · evidence(/api/situation 의 now.evidence) ·
//! oi_1s(hot oi_1s 최근 660초 -- hub oi_rows 와 같은 읽기의 마지막 값만) · next_funding_ms(코어 streams.mp).
//! 🔴알려진 빈 곳: ETH 말고는 rtserve 가 미시·상황 틱을 안 돌려 micro·x·evidence 가 빈 채(원본은 코인마다 돈다) -- mid 는 마크로 물러선다.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::{json, Map, Value as V};

use crate::ctx_mc::{self as mc, History, McIn, Paths};
use crate::py::PyErr;
use crate::{http_exc, Config, Handler, Req, Resp};

#[derive(Default)]
pub struct McLive {
    /// 심볼(소문자) → 마크 링(원본 cctx[asset].mark_ring)
    pub marks: HashMap<String, rtcore::situation::MarkRing>,
    /// 코인 → mc_last_good · mc_last_payload
    pub last_good: HashMap<String, Map<String, V>>,
    pub last: HashMap<String, Option<(V, f64)>>,
}

impl McLive {
    /// 허브 Mark 한 건(스냅숏의 mark 도) -- 원본 1898~1906: 키 = E // 1000 · 다듬기 시각 = 받은 시각
    pub fn put_mark(&mut self, m: &rtcore::proto::Stamped) {
        if let rtcore::proto::Event::Mark { mark, index, funding, .. } = m.ev {
            let now = m.rx_us as f64 / 1e6;
            self.marks.entry(m.s.clone()).or_insert_with(|| rtcore::situation::MarkRing::new(now)).push(m.ex_ms, (mark, index, funding), now);
        }
    }
}

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    if p == "/api/market-context" { vec![("GET", api_market_context)] } else { Vec::new() }
}

fn ttl(k: &str) -> (f64, f64) {
    mc::TTL.iter().find(|t| t.0 == k).map(|t| (t.1, t.2)).unwrap()
}

struct Files {
    hot: std::path::PathBuf,
    live: std::path::PathBuf,
    hl_pos: std::path::PathBuf,
}

impl Files {
    fn new(cfg: &Config, asset: &str) -> Files {
        let hot = cfg.root.join("data/hot");
        // HL_LIQ_BY_ASSET(hub hl_db 와 같은 표) -- 표 밖 코인은 없는 파일(원본 KeyError 자리 · 수집기 몫이 빈다)
        let hl_pos = crate::hub::hl_db(asset).map_or_else(|| hot.join("__none__"), |p| cfg.root.join(p));
        Files { hot, live: cfg.live_dir(), hl_pos }
    }
    fn paths(&self) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf, std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
        let h = &self.hot;
        (h.join("okx_ctx.sqlite"), h.join("bybit_ctx.sqlite"), h.join("hl_ctx.sqlite"), self.live.join("oi_lsratio.duckdb"), h.join("binance_ctx.sqlite"), h.join("binance_tape.sqlite"))
    }
}

macro_rules! with_paths {
    ($f:expr, $p:ident, $body:expr) => {{
        let (okx, by, hl, ls, bn, tape) = $f.paths();
        let $p = Paths { okx_ctx: &okx, bybit_ctx: &by, hl_ctx: &hl, hl_pos: &$f.hl_pos, lsratio: &ls, bn_ctx: &bn, tape: &tape };
        $body
    }};
}

fn okx_inst(asset: &str) -> String {
    if asset == "eth" {
        std::env::var("DASHBOARD_OKX_INST").unwrap_or_else(|_| "ETH-USDT-SWAP".into()).to_uppercase() // hub.rs 상황 틱과 같은 원천
    } else {
        format!("{}-USDT-SWAP", asset.to_uppercase())
    }
}

fn hist_to_v(h: &History) -> V {
    json!({"oi": h.oi, "oi_per": h.oi_per, "basis": h.basis})
}

fn hist_from_v(v: &V) -> History {
    let pairs = |a: &V| -> Vec<(i64, f64)> { serde_json::from_value(a.clone()).unwrap_or_default() };
    History {
        oi: pairs(&v["oi"]),
        oi_per: v["oi_per"].as_array().map(|a| a.iter().map(|r| (r[0].as_str().unwrap_or("").to_string(), pairs(&r[1]))).collect()).unwrap_or_default(),
        basis: v["basis"].as_array().map(|a| a.iter().filter_map(V::as_f64).collect()).unwrap_or_default(),
    }
}

fn swr(cfg: &Config, key: &str, k: &str, f: impl Fn() -> Result<V, String> + Send + Sync + 'static) -> Result<V, String> {
    let (t, s) = ttl(k);
    cfg.swr.get(key, t, s, Arc::new(move || f().map_err(PyErr))).map_err(|e| e.0)
}

/// market_context_payload(2607~2714) -- 재료 모으기 + ctx_mc::market_context_payload
pub fn payload(cfg: &Config, asset: &str) -> Result<V, String> {
    let sym = format!("{asset}usdt");
    let symbol = sym.to_uppercase();
    let now = (cfg.now)();
    let px_scale = crate::hub::fp_bucket(&sym) / crate::hub::fp_bucket("ethusdt");
    let (c, a) = (cfg.clone(), asset.to_string());
    let col = swr(cfg, &format!("mc_collectors_{asset}"), "mc_collectors", move || {
        let f = Files::new(&c, &a);
        let mut lg = c.live.mc.lock().unwrap().last_good.remove(&a).unwrap_or_default();
        let out = with_paths!(f, p, mc::mc_collectors(&p, &a, &okx_inst(&a), &format!("{}USDT", a.to_uppercase()), &mut lg));
        c.live.mc.lock().unwrap().last_good.insert(a.clone(), lg);
        Ok(V::Object(out))
    })?;
    let (c, a) = (cfg.clone(), asset.to_string());
    let hist = hist_from_v(&swr(cfg, &format!("mc_history_{asset}"), "mc_history", move || {
        let f = Files::new(&c, &a);
        let now_ms = ((c.now)() * 1000.0) as i64;
        Ok(hist_to_v(&with_paths!(f, p, mc::mc_history(&p, now_ms, &format!("{a}usdt"), &okx_inst(&a), &a.to_uppercase(), &format!("{}USDT", a.to_uppercase())))))
    })?);
    let (c, a) = (cfg.clone(), asset.to_string());
    let prof = swr(cfg, &format!("mc_liq_profile_{asset}"), "mc_liq_profile", move || {
        let f = Files::new(&c, &a);
        Ok(with_paths!(f, p, mc::mc_liq_profile(&p, &format!("{}USDT", a.to_uppercase()), (c.now)(), px_scale)))
    })?;
    // REST 몫 -- 실패·꺼짐 = [](원본 2626~2633)
    let (c, s2) = (cfg.clone(), symbol.clone());
    let fh: Vec<f64> = swr(cfg, &format!("mc_funding_hist_{asset}"), "mc_funding_hist", move || {
        let raw = crate::model::fetch_binance(&c, &mc::funding_hist_url(&c.rest, &s2), Some("funding_hist_upstream_error"))?.unwrap_or(V::Null);
        mc::funding_hist(&raw).map(|v| json!(v))
    })
    .ok()
    .and_then(|v| serde_json::from_value(v).ok())
    .unwrap_or_default();
    let dvy: Vec<f64> = if asset == "eth" {
        let c = cfg.clone();
        swr(cfg, "mc_dvol_year", "mc_dvol_year", move || {
            let j = crate::model::ext_get(&c, &mc::dvol_url(((c.now)() * 1000.0) as i64))?;
            mc::dvol_year(&j).map(|v| json!(v))
        })
        .ok()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
    } else {
        vec![]
    };
    let (c, s2) = (cfg.clone(), sym.clone());
    let vprof = swr(cfg, &format!("mc_profile_{asset}"), "mc_profile", move || {
        let bk = crate::api2::TAPE_BUCKETS.iter().find(|x| x.0 == s2).ok_or_else(|| format!("KeyError: '{s2}'"))?.1;
        mc::tape_levels(&c.root.join("data/hot/binance_tape.sqlite"), &s2, bk, (c.now)().trunc() as i64)
    })
    .unwrap_or_else(|e| json!({"available": false, "error": e.chars().take(100).collect::<String>()}));
    // 메모리 몫
    let (micro, mid_ring, depth25): (V, BTreeMap<i64, f64>, Vec<(f64, f64)>) = if asset == "eth" {
        let r = cfg.live.run.lock().unwrap();
        (r.micro.payload.clone(), r.micro.mid_ring.clone(), r.micro.depth25_ring.iter().copied().collect())
    } else {
        (json!({}), BTreeMap::new(), vec![])
    };
    let (x, ev, next_funding_ms) = {
        let st = cfg.live.sit.lock().unwrap();
        let (x, ev) = if asset == "eth" {
            (st.x.clone(), st.out.as_ref().map_or(json!({}), |o| o.0["now"]["evidence"].clone()))
        } else {
            (Map::new(), json!({}))
        };
        (x, if ev.is_object() { ev } else { json!({}) }, st.streams.get(&sym).and_then(|s| s.mp.next_funding_ms))
    };
    let mark_ring = cfg.live.mc.lock().unwrap().marks.get(&sym).map(|m| m.ring.clone()).unwrap_or_default();
    let oi_1s = oi_last(cfg, &sym, now);
    let i = McIn { asset, now, bands_usd: crate::hub::band(&sym), micro: &micro, x: &x, ev: &ev, mark_ring: &mark_ring, mid_ring: &mid_ring,
                   depth25: &depth25, oi_1s: &oi_1s, next_funding_ms, px_scale };
    Ok(mc::market_context_payload(&i, col.as_object().unwrap_or(&Map::new()), &hist, &prof, &fh, &dvy, &vprof))
}

/// flows[asset].oi_1s 의 마지막 값(원본 collect_oi_1s 링 = 지금−660초 안 · market_context 는 max 초 하나만 쓴다)
fn oi_last(cfg: &Config, sym: &str, now: f64) -> BTreeMap<i64, f64> {
    let lo = ((now - 660.0) * 1000.0) as i64 - 30_000;
    let rows = crate::api::rows(&cfg.root.join("data/hot/binance_ctx.sqlite"), "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms > ? ORDER BY ts_ms DESC LIMIT 1",
        &[rusqlite::types::Value::Text(sym.into()), rusqlite::types::Value::Integer(lo)]).unwrap_or_default();
    let cut = now.trunc() as i64 - 660;
    rows.iter().filter_map(|r| Some((r[0].as_i64()?.div_euclid(1000), r[1].as_f64()?))).filter(|(k, _)| *k >= cut).collect()
}

/// GET /api/market-context?asset= (2723~2737): 3초 안에 못 끝나면 실패 갈래(계산은 뒤에서 마저 -- 캐시를 채운다)
pub fn api_market_context(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if !cfg.flows().contains(&asset.as_str()) {
        return http_exc(404, Some("flow_off"), &[]);
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let (c, a) = (cfg.clone(), asset.clone());
    std::thread::spawn(move || {
        let _ = tx.send(payload(&c, &a));
    });
    let r = match rx.recv_timeout(std::time::Duration::from_secs_f64(mc::MC_RESPONSE_S)) {
        Ok(r) => r,
        Err(_) => Err("TimeoutError: ".into()),
    };
    let now = (cfg.now)();
    let mut g = cfg.live.mc.lock().unwrap();
    let v = mc::mc_response(r, g.last.entry(asset).or_default(), now);
    Resp::json(&v)
}
