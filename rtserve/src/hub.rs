//! rtcore 허브(`/rt` · rtcore/src/proto.rs) 구독 → 메모리 상태 엔드포인트(/api/supply-1s · /api/footprint)와 SSE 둘(/api/stream · /api/events).
//! 원본은 이 상태를 자기 프로세스의 수집 루프(coin_flow · collect_force_orders)가 채웠다 -- 여기서는 rtcore 가 민 값을 받아 쌓기만 한다.
//! 받는 규칙은 proto.rs 주석 그대로: 스냅숏 = 통째로 · Supply1s = Supply1s::merge(초마다 덮어쓰기) · Footprint = 칸마다 덮어쓰기 + drop + 창 자르기.
//! 응답 모양·창(최근 660초)·커서(since*)는 server.py supply_1s_payload(4181)·api_footprint(3971)·api_stream(3130)·api_events(3651) 그대로.

use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use rtcore::proto::{Event, Footprint, Level, Stamped, Supply1s};
use serde_json::{json, Map, Value};

use crate::py::{self, f};
use crate::{http_exc, Config, Req, Resp, MARKET_ASSETS};

/// 원본 SUPPLY_1S_SECONDS(coin_flow.py:94) · FOOTPRINT_BARS(51) · FOOTPRINT_KEEP_BARS(86) · FOOTPRINT_BAR_SECONDS(50)
const SUPPLY_S: i64 = 660;
const FP_BARS: i64 = 12;
const FP_KEEP: i64 = 288;
const BAR_S: i64 = 300;

#[derive(Default)]
pub struct Sym {
    pub supply: Option<Supply1s>,
    /// 풋프린트 머리(마지막 메시지의 칸 폭·ready·경계) + 봉(시각 → (가격 비트 → 칸, dhl))
    pub fp_head: Option<Footprint>,
    pub fp_bars: BTreeMap<i64, (BTreeMap<u64, Level>, Option<[f64; 3]>)>,
    /// 코어 Micro 의 OKX OI 5분 Δ(봉 → (ΔOI, 봉 끝 OI) · 원본 api_oi_5m okx_d · 반올림 전)와 OKX first_bar
    pub okx_oi5m: BTreeMap<i64, (f64, f64)>,
    pub okx_from: Option<i64>,
    /// 코어 Liq5m(바이낸스 봉 · OKX·Bybit 몫) -- 원본 liq_5m[asset]·liq_5m_state·okx_liq_events·bybit_liq_events 자리
    pub liq: Option<rtcore::proto::Liq5m>,
}

#[derive(Default)]
pub struct Live {
    pub syms: Mutex<HashMap<String, Sym>>,
    /// 종목(대문자) → (가격, 거래소 ms) -- 원본 /api/events tickers 의 원천(REST ticker/price 대신 허브 Prices)
    pub prices: Mutex<BTreeMap<String, (f64, i64)>>,
    pub connected: Mutex<bool>,
}

impl Live {
    /// 허브 메시지 한 건(모르는 type·칸은 건너뛴다)
    pub fn apply(&self, v: &str) {
        let Ok(m) = serde_json::from_str::<Stamped>(v) else { return };
        match m.ev {
            Event::Snapshot { supply, footprint, micro, liq5m, .. } => {
                let mut g = self.syms.lock().unwrap();
                let s = g.entry(m.s.clone()).or_default();
                s.liq = liq5m;
                s.okx_oi5m.clear();
                if let Some(mc) = micro {
                    put_micro(s, mc);
                }
                s.supply = Some(supply);
                s.fp_bars.clear();
                if let Some(fp) = footprint {
                    put_fp(s, fp);
                } else {
                    s.fp_head = None;
                }
            }
            Event::Supply1s(n) => {
                let mut g = self.syms.lock().unwrap();
                let s = g.entry(m.s.clone()).or_default();
                match &mut s.supply {
                    Some(old) => old.merge(n),
                    None => s.supply = Some(n),
                }
                prune(s.supply.as_mut().unwrap());
            }
            Event::Footprint(fp) => {
                let mut g = self.syms.lock().unwrap();
                put_fp(g.entry(m.s.clone()).or_default(), fp);
            }
            Event::Liq5m(l) => {
                let mut g = self.syms.lock().unwrap();
                put_liq(g.entry(m.s.clone()).or_default(), l);
            }
            Event::Micro(mc) => {
                let mut g = self.syms.lock().unwrap();
                put_micro(g.entry(m.s.clone()).or_default(), mc);
            }
            Event::Prices { rows } => {
                let mut p = self.prices.lock().unwrap();
                for (s, px, ms) in rows {
                    p.insert(s.to_uppercase(), (px, ms));
                }
            }
            _ => {}
        }
    }
}

/// Micro 의 okx_oi5m = 바뀐 봉부터 끝까지(받는 쪽은 봉마다 덮어쓴다 · proto.rs) · okx_from = 마지막 값
fn put_micro(s: &mut Sym, mc: rtcore::proto::Micro) {
    for (b, d, c) in mc.okx_oi5m {
        s.okx_oi5m.insert(b, (d, c));
    }
    while s.okx_oi5m.len() > 600 {
        s.okx_oi5m.pop_first(); // ponytail: 원본 okx_oi_5m 링보다 넉넉히(288봉 창) -- 원본 보관 길이와 맞출 필요가 생기면 그 상수로
    }
    if mc.okx_from.is_some() {
        s.okx_from = mc.okx_from;
    }
}

/// Liq5m 밀기 = 바뀐 봉만(봉마다 덮어쓰기) · okx·bybit 는 실리면 통째로 갈아 끼운다(proto.rs) · 288봉 밖은 버린다(원본 liq_5m_add 청소)
fn put_liq(s: &mut Sym, l: rtcore::proto::Liq5m) {
    let Some(old) = &mut s.liq else {
        s.liq = Some(l);
        return;
    };
    old.from_s = l.from_s;
    let mut m: BTreeMap<i64, rtcore::proto::Liq5mRow> = old.bars.drain(..).map(|r| (r.0, r)).collect();
    m.extend(l.bars.into_iter().map(|r| (r.0, r)));
    if let Some(top) = m.keys().next_back().copied() {
        m.retain(|b, _| *b >= top - FP_KEEP * BAR_S);
    }
    old.bars = m.into_values().collect();
    if l.okx.is_some() {
        old.okx = l.okx;
    }
    if l.bybit.is_some() {
        old.bybit = l.bybit;
    }
}

/// 받아 쌓는 칸이 끝없이 자라지 않게: 원본 창(660초)보다 오래된 초를 버린다(원본 링도 그만큼만 든다)
fn prune(s: &mut Supply1s) {
    let keep = |now: i64| move |t: i64| t > now - SUPPLY_S - 5;
    let (bn, okx, spot) = (keep(s.now), keep(s.okx_now), keep(s.spot_now));
    s.seconds.retain(|r| bn(r.0));
    s.liq.retain(|r| bn(r.0));
    s.okx.retain(|r| okx(r.0));
    s.okx_liq.retain(|r| okx(r.0));
    s.spot.retain(|r| spot(r.0));
    let last_oi = s.okx_oi.last().map_or(0, |r| r.0);
    s.okx_oi.retain(|r| r.0 > last_oi - SUPPLY_S - 5);
    let b = s.bybit_liq.last().map_or(0, |r| r.0);
    s.bybit_liq.retain(|r| r.0 > b - SUPPLY_S - 5);
}

/// proto.rs Footprint 받는 규칙: full 이면 통째로 · 칸은 가격으로 덮어쓰고 · dhl 통째로 · drop · barsExpected 넘으면 오래된 쪽부터
fn put_fp(s: &mut Sym, fp: Footprint) {
    if fp.full {
        s.fp_bars.clear();
    }
    for b in &fp.bars {
        let e = s.fp_bars.entry(b.time).or_default();
        for l in &b.levels {
            e.0.insert(l.0.to_bits(), *l);
        }
        e.1 = b.dhl;
    }
    for t in &fp.drop {
        s.fp_bars.remove(t);
    }
    while s.fp_bars.len() > fp.bars_expected {
        s.fp_bars.pop_first();
    }
    s.fp_head = Some(Footprint { bars: Vec::new(), drop: Vec::new(), ..fp });
}

/// 구독 루프: 붙고 → `{"subscribe":[…]}` → 받는 대로 Live 에. 끊기면 1·2·4…30초 뒤 다시(rtorder coreval::subscribe 와 같은 꼴).
pub async fn subscribe(url: String, symbols: Vec<String>, live: Arc<Live>) {
    let mut wait = 1.0f64;
    loop {
        if let Ok((mut ws, _)) = tokio_tungstenite::connect_async(url.as_str()).await {
            let sub = json!({"subscribe": symbols}).to_string();
            if ws.send(tokio_tungstenite::tungstenite::Message::Text(sub.into())).await.is_ok() {
                *live.connected.lock().unwrap() = true;
                wait = 1.0;
                while let Some(Ok(m)) = ws.next().await {
                    if let tokio_tungstenite::tungstenite::Message::Text(t) = m {
                        live.apply(t.as_str());
                    }
                }
            }
            *live.connected.lock().unwrap() = false;
        }
        tokio::time::sleep(Duration::from_secs_f64(wait)).await;
        wait = (wait * 2.0).min(30.0);
    }
}

// ───────────────────────── /api/supply-1s (server.py:4181 supply_1s_payload) ─────────────────────────

fn q_int(q: &HashMap<String, String>, k: &str) -> i64 {
    q.get(k).and_then(|v| crate::api::py_int_str(v)).map_or(0, |v| v.clamp(i64::MIN as i128, i64::MAX as i128) as i64)
}

fn rows8(rows: &[rtcore::proto::SecRow], lo: i64, hi: i64) -> Vec<Value> {
    rows.iter().filter(|r| lo < r.0 && r.0 < hi).map(|r| json!([r.0, f(r.1), f(r.2), f(r.3), f(r.4), f(r.5), f(r.6), f(r.7)])).collect()
}

fn liq5(rows: &[rtcore::proto::LiqRow], ok: impl Fn(i64) -> bool) -> Vec<Value> {
    rows.iter().filter(|r| ok(r.0)).map(|r| json!([r.0, f(r.1), f(r.2), r.3, r.4])).collect()
}

/// 원본 oi_1s 링(coin_flow.py collect_oi_1s 641~665: 수집기 hot oi_1s 꼬리를 0.5초마다 읽어 초 → 마지막 값 · 지금−660초 앞은 버림) +
/// supply_1s_payload 4299~4301(oi_floor = max(sinceOi, 가장 늦은 초 − 660) 뒤 초만). 2026-11-22 병행 대조에서 rtserve 가 늘 [] 를 내던 결함을 고침
/// (전엔 «REST 폴링 = 코어 범위 밖»으로 잘못 알았다 -- 원본도 hot 을 읽는다).
/// ponytail: 부를 때마다 hot 을 읽는다(SSE 0.25초 틱 포함 · 읽기 ~1ms) -- 무거우면 0.5초 메모로
fn oi_rows(cfg: &Config, symbol: &str, since: i64) -> Vec<Value> {
    let wall = (cfg.now)();
    let lo = ((wall - SUPPLY_S as f64) * 1000.0) as i64 - 30_000;
    let Ok(rows) = crate::api::rows(&cfg.root.join("data/hot/binance_ctx.sqlite"), "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms > ? ORDER BY ts_ms",
        &[rusqlite::types::Value::Text(symbol.into()), rusqlite::types::Value::Integer(lo)]) else { return vec![] };
    let mut ring: BTreeMap<i64, f64> = BTreeMap::new();
    for r in rows {
        if let (Some(t), Some(v)) = (r[0].as_i64(), r[1].as_f64()) {
            ring.insert(t.div_euclid(1000), v);
        }
    }
    let cut = wall.trunc() as i64 - SUPPLY_S;
    ring.retain(|k, _| *k >= cut);
    let floor = since.max(ring.keys().next_back().copied().unwrap_or(0) - SUPPLY_S);
    ring.range(floor + 1..).map(|(k, v)| json!([k, f(*v)])).collect()
}

/// 원본 supply_1s_payload 의 모양과 창·커서.
pub fn supply_payload(cfg: &Config, live: &Live, symbol: &str, q: &HashMap<String, String>) -> V {
    let g = live.syms.lock().unwrap();
    let empty = Supply1s {
        now: 0, retail_max_usd: 0.0, whale_min_usd: 0.0, seconds: vec![], partial: rtcore::proto::Partial { bn: None, okx: None, spot: None },
        liq: vec![], spot: vec![], spot_now: 0, spot_meta: rtcore::proto::SpotMeta { connected: false, trade_age: None, errors: 0 },
        okx: vec![], okx_now: 0, okx_meta: rtcore::proto::OkxMeta { connected: false, trade_age: None, oi_age: None, inst: String::new(), errors: 0 },
        okx_liq: vec![], okx_oi: vec![], bybit_liq: vec![],
    };
    let s = g.get(symbol).and_then(|x| x.supply.as_ref()).unwrap_or(&empty);
    let (sm, sw) = band(symbol);
    let (retail, whale) = if s.now == 0 && s.seconds.is_empty() { (sm, sw) } else { (s.retail_max_usd, s.whale_min_usd) };
    let wall = (cfg.now)().trunc() as i64;
    let bnow = wall - 3;
    let bfloor = q_int(q, "sinceBybitLiq").max(bnow - SUPPLY_S);
    let bybit = liq5(&s.bybit_liq, |x| bfloor < x && x <= bnow);
    let spot_floor = q_int(q, "sinceSpot").max(s.spot_now - SUPPLY_S);
    let okx_floor = q_int(q, "sinceOkx").max(s.okx_now - SUPPLY_S);
    let okx_oi_floor = q_int(q, "sinceOkxOi").max(s.okx_oi.last().map_or(0, |r| r.0) - SUPPLY_S);
    let okx_liq_floor = q_int(q, "sinceOkxLiq").max(s.okx_now - SUPPLY_S);
    let part = |p: &Option<rtcore::proto::SecRow>| p.map_or(Value::Null, |r| json!([r.0, f(r.1), f(r.2), f(r.3), f(r.4), f(r.5), f(r.6), f(r.7)]));
    let partial = json!({"okx": part(&s.partial.okx), "spot": part(&s.partial.spot), "bn": part(&s.partial.bn)});
    let okx_meta = json!({"connected": s.okx_meta.connected, "tradeAge": s.okx_meta.trade_age.map_or(Value::Null, f), "oiAge": s.okx_meta.oi_age.map_or(Value::Null, f),
                          "inst": s.okx_meta.inst, "errors": s.okx_meta.errors});
    let spot_meta = json!({"connected": s.spot_meta.connected, "tradeAge": s.spot_meta.trade_age.map_or(Value::Null, f), "errors": s.spot_meta.errors});
    let okx = rows8(&s.okx, okx_floor, s.okx_now);
    let okx_oi: Vec<Value> = s.okx_oi.iter().filter(|r| r.0 > okx_oi_floor).map(|r| json!([r.0, f(r.1)])).collect();
    let okx_liq = liq5(&s.okx_liq, |x| okx_liq_floor < x && x < s.okx_now);
    let spot = rows8(&s.spot, spot_floor, s.spot_now);
    let o = |pairs: Vec<(&str, V)>| Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Map<String, V>>());
    if s.now == 0 && s.partial.bn.is_none() && s.seconds.is_empty() {
        return o(vec![("symbol", json!(symbol.to_uppercase())), ("seconds", json!([])), ("now", json!(0)), ("oi", json!(oi_rows(cfg, symbol, q_int(q, "sinceOi")))), ("liq", json!([])), ("okx", json!(okx)),
            ("okxOi", json!(okx_oi)), ("okxLiq", json!(okx_liq)), ("okxNow", json!(s.okx_now)), ("bybitLiq", json!(bybit)), ("okxMeta", okx_meta),
            ("spot", json!(spot)), ("spotNow", json!(s.spot_now)), ("spotMeta", spot_meta), ("partial", partial), ("retailMaxUsd", f(retail)), ("whaleMinUsd", f(whale))]);
    }
    let newest = s.now;
    let liq_floor = q_int(q, "sinceLiq").max(newest - SUPPLY_S);
    let floor = q_int(q, "since").max(newest - SUPPLY_S);
    o(vec![("symbol", json!(symbol.to_uppercase())), ("now", json!(newest)), ("retailMaxUsd", f(retail)), ("whaleMinUsd", f(whale)), ("oi", json!(oi_rows(cfg, symbol, q_int(q, "sinceOi")))),
        ("liq", json!(liq5(&s.liq, |x| liq_floor < x && x < newest))), ("okx", json!(okx)), ("okxOi", json!(okx_oi)), ("okxLiq", json!(okx_liq)),
        ("okxNow", json!(s.okx_now)), ("bybitLiq", json!(bybit)), ("okxMeta", okx_meta), ("spot", json!(spot)), ("spotNow", json!(s.spot_now)),
        ("spotMeta", spot_meta), ("partial", partial), ("seconds", json!(rows8(&s.seconds, floor, newest)))])
}

type V = Value;

/// 코인별 (리테일 상한, 고래 하한) -- scripts/live_trade_tape_collector size_bands(SIZE_BANDS_USD · 없으면 ETH 1만/10만)
fn band(symbol: &str) -> (f64, f64) {
    const B: [(&str, f64, f64); 8] = [("solusdt", 7_500.0, 55_000.0), ("xrpusdt", 3_400.0, 23_000.0), ("btcusdt", 13_000.0, 160_000.0), ("zecusdt", 1_000.0, 6_000.0),
        ("sndkusdt", 1_000.0, 6_700.0), ("soxlusdt", 3_900.0, 36_000.0), ("skhynixusdt", 700.0, 4_300.0), ("muusdt", 700.0, 5_400.0)];
    B.iter().find(|x| x.0 == symbol).map_or((10_000.0, 100_000.0), |x| (x.1, x.2))
}

/// 원본 flow_for(1725): `?asset=` 소문자 · 흐름 엔진이 꺼진 코인이면 404 flow_off
fn flow_symbol(cfg: &Config, req_asset: Option<&str>) -> Result<String, Resp> {
    let a = req_asset.filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if cfg.flows().contains(&a.as_str()) { Ok(format!("{a}usdt")) } else { Err(http_exc(404, Some("flow_off"), &[])) }
}

fn qmap(req: &Req) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for (k, v) in &req.query {
        m.entry(k.clone()).or_insert_with(|| v.clone()); // request.query.get = 첫 값
    }
    m
}

/// GET /api/supply-1s (server.py:4341)
pub fn api_supply(cfg: &Config, live: &Live, req: &Req) -> Resp {
    match flow_symbol(cfg, req.q("asset")) {
        Ok(sym) => Resp::json(&supply_payload(cfg, live, &sym, &qmap(req))),
        Err(r) => r,
    }
}

// ───────────────────────── /api/footprint (server.py:3971) ─────────────────────────

/// GET /api/footprint -- ?bars(1~288 · 기본 12)·?since(봉 시각 · ready 일 때만 증분)
/// 다른 곳(문서 §의도한 차이): `agg`(aggTrades 로 메운 봉) 는 코어가 싣지 않아 늘 false · 코어 스냅숏은 최근 144봉까지 · dhl 은 코어 판(rtcore footprint.rs 머리 주석).
pub fn api_footprint(cfg: &Config, live: &Live, req: &Req) -> Resp {
    let sym = match flow_symbol(cfg, req.q("asset")) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let want = match req.q("bars") {
        None => FP_BARS,
        Some(b) => crate::api::py_int_str(b).map_or(FP_BARS, |n| n.clamp(1, FP_KEEP as i128) as i64),
    };
    let since = req.q("since").map_or(Some(0), |s| crate::api::py_int_str(s).map(|v| v as i64)).unwrap_or(0);
    let g = live.syms.lock().unwrap();
    let s = g.get(&sym);
    let head = s.and_then(|s| s.fp_head.as_ref());
    let (sm, sw) = band(&sym);
    let ready = head.is_some_and(|h| h.ready);
    let full = since <= 0 || !ready;
    let mut bars = Vec::new();
    if let Some(s) = s {
        let n = s.fp_bars.len();
        for (t, (lv, dhl)) in s.fp_bars.iter().skip(n.saturating_sub(want as usize)) {
            if !full && *t < since {
                continue;
            }
            let levels: Vec<V> = lv.values().map(|l| json!([f(l.0), f(l.1), f(l.2), f(l.3), f(l.4), f(l.5), f(l.6)])).collect();
            let mut b = Map::new();
            b.insert("time".into(), json!(t));
            b.insert("agg".into(), json!(false));
            b.insert("levels".into(), Value::Array(levels));
            if let Some(d) = dhl {
                b.insert("dhl".into(), json!([f(d[0]), f(d[1]), f(d[2])]));
            }
            bars.push(Value::Object(b));
        }
    }
    let bucket = head.map_or_else(|| fp_bucket(&sym), |h| h.bucket);
    let v = json!({"symbol": sym.to_uppercase(), "bucket": f(bucket), "barSeconds": BAR_S, "barsExpected": want, "ready": ready, "full": full,
                   "retailMaxUsd": f(head.map_or(sm, |h| h.retail_max_usd)), "whaleMinUsd": f(head.map_or(sw, |h| h.whale_min_usd)),
                   "venues": ["binance-perp"], "bars": bars}); // 원본은 바이낸스 선물만(11-02 사용자 결정 · server.py 3981~3984)
    Resp::json(&v)
}

/// dashboard/coin_flow.py FLOW_SPECS 칸 폭(코어가 아직 아무것도 안 보냈을 때만 쓴다)
fn fp_bucket(sym: &str) -> f64 {
    match sym {
        "solusdt" | "soxlusdt" => 0.02,
        "xrpusdt" => 0.0003,
        "btcusdt" => 15.0,
        "sndkusdt" | "skhynixusdt" | "muusdt" => 0.2,
        _ => 0.5,
    }
}

// ───────────────────────── SSE ─────────────────────────

pub type Body = http_body_util::combinators::BoxBody<Bytes, Infallible>;

/// SSE 응답 머리(원본 StreamResponse 의 headers 그대로 · 청크 전송)
fn sse(rx: tokio::sync::mpsc::Receiver<Bytes>) -> hyper::Response<Body> {
    use http_body_util::BodyExt;
    let stream = futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|b| (Ok::<_, Infallible>(hyper::body::Frame::data(b)), rx)) });
    let mut r = hyper::Response::new(BodyExt::boxed(http_body_util::StreamBody::new(stream)));
    let h = r.headers_mut();
    for (k, v) in [("content-type", "text/event-stream"), ("cache-control", "no-cache"), ("connection", "keep-alive"), ("x-accel-buffering", "no")] {
        h.insert(k, hyper::header::HeaderValue::from_static(v));
    }
    r
}

/// GET /api/stream (server.py:3130): 0.25초마다 `event: supply`(?supply=1 · 커서는 서버가 든다).
/// 🔴`event: situation` 은 안 낸다 -- 상황 계산(compute_situation)은 아직 파이썬 몫. 화면은 스트림이 살아 있으면 /api/situation 폴링을 끄므로
///   상황이 옮겨지기 전에는 화면을 이 스트림으로 돌리면 안 된다(문서 §다음 조각).
pub fn api_stream(cfg: Arc<Config>, live: Arc<Live>, req: &Req) -> Result<hyper::Response<Body>, Resp> {
    let mut q = qmap(req);
    let want = q.get("supply").map(String::as_str) == Some("1");
    let sym = if want { Some(flow_symbol(&cfg, q.get("asset").map(String::as_str))?) } else { None };
    let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(4);
    tokio::spawn(async move {
        const CURSORS: [(&str, &str); 8] = [("seconds", "since"), ("oi", "sinceOi"), ("liq", "sinceLiq"), ("okx", "sinceOkx"), ("okxOi", "sinceOkxOi"),
            ("okxLiq", "sinceOkxLiq"), ("spot", "sinceSpot"), ("bybitLiq", "sinceBybitLiq")];
        loop {
            if let Some(sym) = &sym {
                let sup = supply_payload(&cfg, &live, sym, &q);
                for (key, name) in CURSORS {
                    if let Some(last) = sup[key].as_array().and_then(|a| a.last()) {
                        let cur = q.get(name).and_then(|v| crate::api::py_int_str(v)).unwrap_or(0) as i64;
                        q.insert(name.into(), cur.max(last[0].as_i64().unwrap_or(0)).to_string());
                    }
                }
                let line = format!("event: supply\ndata: {}\n\n", py::dumps(&sup, (",", ":"), false, None));
                if tx.send(Bytes::from(line)).await.is_err() {
                    return; // 끊김
                }
            } else if tx.is_closed() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    });
    Ok(sse(rx))
}

/// /api/events 발행기 하나(원본 publish_dashboard_events · 1초): 상태 파일이 바뀐 때만 state(세 블록) · tickers · assets · acct_seq.
pub struct Events {
    pub clients: Mutex<Vec<tokio::sync::mpsc::Sender<Bytes>>>,
    /// (마지막 state 보기 · 마지막 tickers) -- 접속 즉시 보내는 첫 프레임의 원천
    pub latest: Mutex<(Value, Value)>,
    /// rtorder /internal/acct 의 seq(마지막으로 읽은 값 · 못 읽으면 그대로)
    pub acct_seq: Mutex<i64>,
}

/// 원본 sse_state_view(1258): 화면이 읽는 세 블록만
fn state_view(st: &Value) -> Value {
    match st.get("state") {
        Some(s) if py::truthy(s) => {
            let mut m = Map::new();
            for k in ["session", "microstructure", "tail_risk"] {
                m.insert(k.into(), s.get(k).cloned().unwrap_or(Value::Null));
            }
            json!({"state": m})
        }
        _ => st.clone(),
    }
}

/// datetime.now(timezone.utc).isoformat()
fn iso_utc(t: f64) -> String {
    let us = crate::api::now_us(t);
    let secs = us.div_euclid(1_000_000);
    let frac = us.rem_euclid(1_000_000);
    let (y, m, d) = crate::api2::civil(secs.div_euclid(86400));
    let s = secs.rem_euclid(86400);
    let base = format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60);
    if frac == 0 { format!("{base}+00:00") } else { format!("{base}.{frac:06}+00:00") }
}

impl Events {
    /// 한 주기(원본 루프 한 바퀴): 상태 서명이 바뀌면 state 를 싣고 · 아니면 null
    pub fn tick(&self, cfg: &Config, live: &Live, last_sig: &mut Option<Option<(i128, u64)>>) -> String {
        let p = cfg.live_dir().join("dashboard_state.json");
        let sig = std::fs::metadata(&p).ok().map(|m| {
            use std::os::unix::fs::MetadataExt;
            (m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128, m.size())
        });
        let changed = last_sig.as_ref() != Some(&sig);
        let mut latest = self.latest.lock().unwrap();
        if changed {
            let st = std::fs::read_to_string(&p).ok().and_then(|s| py::loads(&s).ok()).unwrap_or(Value::Null);
            latest.0 = state_view(&json!({"state": st}));
            *last_sig = Some(sig);
        }
        let now = (cfg.now)();
        let prices = live.prices.lock().unwrap();
        let mut tick = Map::new();
        for a in &cfg.assets {
            let i = MARKET_ASSETS.iter().position(|x| x == a).unwrap();
            let sym = MARKET_SYMBOL[i];
            if let Some((px, _)) = prices.get(sym).filter(|x| x.0 > 0.0) {
                tick.insert(a.clone(), json!({"symbol": sym, "price": f(*px), "ts": iso_utc(now)}));
            }
        }
        latest.1 = Value::Object(tick);
        let payload = json!({"state": if changed { latest.0.clone() } else { Value::Null }, "tickers": latest.1, "assets": cfg.assets,
                             "acct_seq": *self.acct_seq.lock().unwrap()}); // 원본 order_api.acct_seq() = rtorder /internal/acct 의 seq
        py::dumps(&payload, (",", ":"), false, None)
    }

    /// 발행 루프: 1초마다 tick → 클라마다 칸 1개 큐(차 있으면 옛 것을 버리고 새 것)
    pub async fn run(self: Arc<Self>, cfg: Arc<Config>, live: Arc<Live>) {
        let mut last = None;
        loop {
            let c2 = cfg.clone();
            if let Ok(Some(seq)) = tokio::task::spawn_blocking(move || crate::order::acct_seq(&c2)).await {
                *self.acct_seq.lock().unwrap() = seq;
            }
            let enc = self.tick(&cfg, &live, &mut last);
            let frame = Bytes::from(format!("data: {enc}\n\n"));
            self.clients.lock().unwrap().retain(|c| match c.try_send(frame.clone()) {
                Ok(()) => true,
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => true, // ponytail: 원본은 옛 것을 버리고 새 것을 넣는다 -- 여기선 새 것을 버린다(다음 초에 다시 온다)
                Err(_) => false,
            });
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    /// GET /api/events: 접속 즉시 {state, tickers, assets} · 그 뒤 발행 프레임 · 20초 조용하면 `: keepalive`
    pub fn connect(self: &Arc<Self>, cfg: &Config) -> hyper::Response<Body> {
        let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(1);
        let (otx, orx) = tokio::sync::mpsc::channel::<Bytes>(4);
        let first = {
            let l = self.latest.lock().unwrap();
            json!({"state": l.0, "tickers": l.1, "assets": cfg.assets})
        };
        self.clients.lock().unwrap().push(tx);
        tokio::spawn(async move {
            let mut rx = rx;
            if otx.send(Bytes::from(format!("data: {}\n\n", py::dumps(&first, (",", ":"), false, None)))).await.is_err() {
                return;
            }
            loop {
                let b = match tokio::time::timeout(Duration::from_secs(20), rx.recv()).await {
                    Ok(Some(b)) => b,
                    Ok(None) => return,
                    Err(_) => Bytes::from_static(b": keepalive\n\n"),
                };
                if otx.send(b).await.is_err() {
                    return;
                }
            }
        });
        sse(orx)
    }
}

/// orders.py:338 MARKET_SYMBOLS 값(MARKET_ASSETS 와 같은 순서)
pub const MARKET_SYMBOL: [&str; 9] = ["ETHUSDT", "SOLUSDT", "BTCUSDT", "XRPUSDT", "HYPEUSDT", "SNDKUSDT", "SOXLUSDT", "SKHYNIXUSDT", "MUUSDT"];

// ───────────────────────── /api/oi-5m (server.py:4363~4411 · oi_5m_buckets 983~1019) ─────────────────────────

/// 원본 oi_5m_buckets: hot SQLite oi_1s(바이낸스 OI 1초 · 수집기가 씀) → 봉별 [봉, Δ, 봉 끝 OI, 건수, 공백]
pub fn oi_5m_buckets(cfg: &Config, bars: i64, symbol: &str) -> py::R<V> {
    let db = cfg.root.join("data/hot/binance_ctx.sqlite");
    if !db.exists() {
        return Ok(json!([]));
    }
    let floor = ((cfg.now)().trunc() as i64 / 300 - (bars - 1)) * 300;
    let Ok(raw) = crate::api::rows(&db, "SELECT ts_ms, open_interest FROM oi_1s WHERE symbol = ? AND ts_ms >= ? ORDER BY ts_ms",
        &[rusqlite::types::Value::Text(symbol.into()), rusqlite::types::Value::Integer((floor - 300) * 1000)]) else {
        (cfg.log)("oi-5m read failed"); // 원본: 표가 없으면 [] (첫 가동)
        return Ok(json!([]));
    };
    let mut agg: BTreeMap<i64, (f64, f64, i64)> = BTreeMap::new();
    for r in raw {
        let bar = crate::api::int_v(&r[0])?.div_euclid(1000).div_euclid(300) * 300;
        let v = py::to_float(&r[1])?;
        agg.entry(bar).and_modify(|a| {
            a.1 = v;
            a.2 += 1;
        }).or_insert((v, v, 1));
    }
    let mut out = Vec::new();
    let (mut prev_bar, mut prev_close) = (None::<i64>, 0.0);
    for (bar, (o, c, n)) in agg {
        let gap = prev_bar.is_none_or(|p| bar - p > 300);
        let delta = if gap { c - o } else { c - prev_close };
        prev_bar = Some(bar);
        prev_close = c;
        if bar >= floor {
            out.push(json!([bar, f(py::round_nd(delta, 3)), f(py::round_nd(c, 3)), n, if gap { 1 } else { 0 }]));
        }
    }
    Ok(Value::Array(out))
}

/// GET /api/oi-5m -- ?bars(1~288 · 기본 48) · 바이낸스 = hot oi_1s(swr 5초) · OKX = 코어 Micro.okx_oi5m(first_bar «뒤» 봉만 합산)
/// 다른 곳(문서 §5): «현재 OI»(openInterest)는 원본이 수집 루프의 1초 링 마지막 값 · rtserve 는 hot oi_1s 의 마지막 행(수집기 쓰기 주기만큼 늦을 수 있다)
pub fn api_oi_5m(cfg: &Config, req: &Req) -> Resp {
    let sym = match flow_symbol(cfg, req.q("asset")) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let bars = req.q("bars").map_or(Some(48), |b| crate::api::py_int_str(b).map(|v| v.clamp(1, 288) as i64)).unwrap_or(48);
    let (c, s2) = (cfg.clone(), sym.clone());
    let buckets = match cfg.swr.get(&format!("oi_5m:{sym}:{bars}"), 5.0, 0.0, Arc::new(move || oi_5m_buckets(&c, bars, &s2))) {
        Ok(Value::Array(b)) => b,
        _ => Vec::new(),
    };
    let last_oi = crate::api::rows(&cfg.root.join("data/hot/binance_ctx.sqlite"), "SELECT open_interest FROM oi_1s WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1",
        &[rusqlite::types::Value::Text(sym.clone())]).ok().and_then(|r| r.first().and_then(|x| x[0].as_f64()));
    let now_oi = last_oi.or_else(|| buckets.last().and_then(|b| b[2].as_f64())).unwrap_or(0.0);
    let g = cfg.live.syms.lock().unwrap();
    let (okx_d, okx_from) = g.get(&sym).map_or((BTreeMap::new(), 0), |s| (s.okx_oi5m.clone(), s.okx_from.unwrap_or(0)));
    drop(g);
    let bars_out: Vec<V> = if okx_d.is_empty() {
        buckets
    } else {
        buckets.into_iter().filter_map(|b| {
            let t = b[0].as_i64()?;
            let (d, c) = okx_d.get(&t).filter(|_| t > okx_from)?;
            let mut row = vec![json!(t), f(py::round_nd(b[1].as_f64()? + d, 3)), f(py::round_nd(b[2].as_f64()? + c, 3))];
            row.extend(b.as_array()?[3..].iter().cloned());
            Some(Value::Array(row))
        }).collect()
    };
    Resp::json(&json!({"symbol": sym.to_uppercase(), "barSeconds": 300, "openInterest": f(py::round_nd(now_oi, 3)),
        "venues": if okx_d.is_empty() { json!(["binance-perp"]) } else { json!(["binance-perp", "okx-swap"]) }, "bars": bars_out}))
}

// ───────────────── /api/liquidation-5m-history (server.py:4413~4473 · liq_5m_payload 537 · merge_okx_liq 878 · okx_liq_hot_bars 856 · merge_bybit_liq 833 · merge_hl_liq 907) ─────────────────

/// datetime.fromtimestamp(b, timezone.utc).isoformat() (정수 초)
fn iso_s(b: i64) -> String {
    iso_utc(b as f64)
}

/// 원본 liq_5m_payload: 기록 시작 봉과 그 앞은 안 싣고 · 빈 봉은 0
fn liq_payload(l: Option<&rtcore::proto::Liq5m>, bars: i64, now: f64) -> Map<String, V> {
    let now_bar = (now.trunc() as i64).div_euclid(BAR_S) * BAR_S;
    let from_bar = l.map_or(now.trunc() as i64, |l| l.from_s.trunc() as i64).div_euclid(BAR_S) * BAR_S; // 코어에 없음 = 원본 «기록 시작 = 지금»
    let rows: BTreeMap<i64, &rtcore::proto::Liq5mRow> = l.map(|l| l.bars.iter().map(|r| (r.0, r)).collect()).unwrap_or_default();
    let out: Vec<V> = (0..bars).rev().map(|k| now_bar - k * BAR_S).filter(|b| *b > from_bar).map(|b| {
        let (lo, sh, n, lp, sp) = rows.get(&b).map_or((0.0, 0.0, 0, None, None), |r| (r.1, r.2, r.3, r.4, r.5));
        json!({"ts": iso_s(b), "long_usd": f(py::round_nd(lo, 2)), "short_usd": f(py::round_nd(sh, 2)), "events": n,
               "long_min_px": lp.map(f), "short_max_px": sp.map(f), "partial": b == now_bar})
    }).collect();
    let mut m = Map::new();
    m.insert("warmed_up".into(), json!(true));
    m.insert("bars".into(), Value::Array(out));
    m.insert("error".into(), Value::Null);
    m.insert("source".into(), json!("dashboard-forceorder"));
    m
}

fn bar_t(b: &V) -> i64 {
    crate::api::iso_parse(b["ts"].as_str().unwrap_or("")).map_or(0, |(us, _)| us.div_euclid(1_000_000))
}

/// 봉에 거래소 몫을 더한다(원본 {**b, "long_usd": b + a0, …, key: {...}} -- 있는 칸은 제자리, 새 칸은 끝에)
fn add_venue(b: &mut V, lo: f64, sh: f64, n: i64, extra: &[(&str, V)]) {
    let o = b.as_object_mut().unwrap();
    for (k, a) in [("long_usd", lo), ("short_usd", sh)] {
        let v = o[k].as_f64().unwrap_or(0.0) + a;
        o.insert(k.into(), f(v));
    }
    let e = o["events"].as_i64().unwrap_or(0) + n;
    o.insert("events".into(), json!(e));
    for (k, v) in extra {
        o.insert((*k).into(), v.clone());
    }
}

fn rnd(x: f64) -> V {
    json!(py::round0(x).unwrap_or(0))
}

fn detail(lo: f64, sh: f64, n: i64) -> V {
    json!({"long_usd": rnd(lo), "short_usd": rnd(sh), "n": n})
}

/// 원본 okx_liq_hot_bars: hot okx_liquidations → ({봉: [롱USD, 숏USD, 건수]}, 이 종목 기록 시작 ms) · 실패 = ({}, None)
fn okx_liq_hot_bars(cfg: &Config, inst: &str) -> (BTreeMap<i64, (f64, f64, i64)>, Option<i64>) {
    use rusqlite::types::Value as S;
    let db = cfg.root.join("data/hot/okx_ctx.sqlite");
    let since = ((cfg.now)().trunc() as i64).div_euclid(BAR_S) - FP_KEEP;
    let start = match crate::api::rows(&db, "SELECT min(ts_ms) FROM okx_liquidations WHERE inst_id = ?", &[S::Text(inst.into())]) {
        Ok(r) => r.first().and_then(|x| x[0].as_i64()),
        Err(_) => return (BTreeMap::new(), None),
    };
    let Ok(rows) = crate::api::rows(&db, "SELECT ts_ms - ts_ms % ?, pos_side, sum(sz_base * coalesce(bk_px, 0)), count(*) \
                                          FROM okx_liquidations WHERE inst_id = ? AND ts_ms >= ? AND sz_base > 0 GROUP BY 1, 2",
        &[S::Integer(BAR_S * 1000), S::Text(inst.into()), S::Integer(since * BAR_S * 1000)]) else { return (BTreeMap::new(), None) };
    let mut out: BTreeMap<i64, (f64, f64, i64)> = BTreeMap::new();
    for r in rows {
        let a = out.entry(r[0].as_i64().unwrap_or(0).div_euclid(1000)).or_insert((0.0, 0.0, 0));
        let usd = r[2].as_f64().unwrap_or(0.0);
        if r[1].as_str() == Some("long") { a.0 += usd } else { a.1 += usd }
        a.2 += r[3].as_i64().unwrap_or(0);
    }
    (out, start)
}

/// 원본 hl_whale_liq_events: [(체결 ms, USD, 롱청산?, 주소)] · 없거나 실패 = []
fn hl_whale_liq_events(cfg: &Config, since_ms: i64, db: &str, coin: &str) -> Vec<(i64, f64, bool, String)> {
    use rusqlite::types::Value as S;
    let Ok(rows) = crate::api::rows(&cfg.root.join(db), "SELECT fill_ms, px * sz, dir, side, user FROM hl_liquidations\n                                      WHERE coin = ? AND fill_ms >= ?",
        &[S::Text(coin.into()), S::Integer(since_ms)]) else { return Vec::new() };
    rows.into_iter().map(|r| {
        let long = match r[2].as_str() {
            Some(d) if !d.is_empty() => d.contains("Long"),
            _ => r[3].as_str() == Some("A"),
        };
        let user = match &r[4] {
            Value::String(s) => s.clone(),
            Value::Null => "None".into(),
            v => v.to_string(),
        };
        (r[0].as_i64().unwrap_or(0), r[1].as_f64().unwrap_or(0.0), long, user)
    }).collect()
}

/// 원본 HL_LIQ_BY_ASSET(server.py:497)
fn hl_db(asset: &str) -> Option<String> {
    match asset {
        "eth" => Some("data/hot/hl_positions.sqlite".into()),
        "sol" | "xrp" => Some("data/hot/hl_positions_sol_xrp.sqlite".into()),
        "btc" | "sndk" | "soxl" | "skhynix" | "mu" => Some(format!("data/hot/hl_positions_{asset}.sqlite")),
        _ => None,
    }
}

/// GET /api/liquidation-5m-history -- 바이낸스·OKX(first_bar «뒤»)·Bybit = 코어 Liq5m · OKX first_bar «앞» = hot(swr 60) · HL = hot(swr 30)
pub fn api_liquidation_5m_history(cfg: &Config, req: &Req) -> Resp {
    let asset = match crate::api::coin_asset(req) {
        Ok(a) => a,
        Err(r) => return r,
    };
    let bars = match req.q("bars") {
        None => FP_BARS,
        Some(b) => crate::api::py_int_str(b).map_or(FP_BARS, |n| n.clamp(1, FP_KEEP as i128) as i64),
    };
    let now = (cfg.now)();
    let sym = format!("{asset}usdt");
    let l = cfg.live.syms.lock().unwrap().get(&sym).and_then(|s| s.liq.clone());
    let mut payload = liq_payload(l.as_ref(), bars, now);
    let empty = payload["bars"].as_array().is_none_or(|b| b.is_empty());
    if !cfg.flows().contains(&asset.as_str()) || empty {
        return Resp::json(&Value::Object(payload));
    }
    let (okx, bybit) = l.map_or((None, None), |l| (l.okx, l.bybit));
    let okx_from = okx.as_ref().and_then(|o| o.from).unwrap_or(0);
    let inst = if asset == "eth" { std::env::var("DASHBOARD_OKX_INST").unwrap_or_else(|_| "ETH-USDT-SWAP".into()).to_uppercase() } else { format!("{}-USDT-SWAP", asset.to_uppercase()) };
    let (c, i2) = (cfg.clone(), inst.clone());
    let hot = cfg.swr.get(&format!("okx_liq_hot:{inst}"), 60.0, 0.0, Arc::new(move || {
        let (b, s) = okx_liq_hot_bars(&c, &i2);
        Ok(json!([b.into_iter().map(|(t, a)| json!([t, f(a.0), f(a.1), a.2])).collect::<Vec<_>>(), s]))
    })).unwrap_or(json!([[], null]));
    let hot_start = hot[1].as_i64();
    let hot_bars: BTreeMap<i64, (f64, f64, i64)> = hot[0].as_array().map(|a| a.iter().map(|r| (r[0].as_i64().unwrap_or(0), (r[1].as_f64().unwrap_or(0.0), r[2].as_f64().unwrap_or(0.0), r[3].as_i64().unwrap_or(0)))).collect()).unwrap_or_default();
    let mut out: Vec<V> = match payload.get_mut("bars") {
        Some(Value::Array(b)) => std::mem::take(b), // 자리는 그대로(원본 {**payload, "bars": …} 는 키 순서를 안 바꾼다)
        _ => Vec::new(),
    };
    let mut venues = vec![json!("binance-perp")];
    if okx_from != 0 || hot_start.is_some() {
        venues.push(json!("okx-swap"));
        let live: BTreeMap<i64, (f64, f64, i64)> = okx.map(|o| o.bars.into_iter().map(|r| (r.0, (r.1, r.2, r.3))).collect()).unwrap_or_default();
        let hot_first = hot_start.map(|s| s.div_euclid(1000).div_euclid(BAR_S) * BAR_S);
        for b in &mut out {
            let t = bar_t(b);
            let a = if okx_from != 0 && t > okx_from { Some(live.get(&t).copied().unwrap_or((0.0, 0.0, 0))) } else if hot_first.is_some_and(|h| t > h) {
                Some(hot_bars.get(&t).copied().unwrap_or((0.0, 0.0, 0)))
            } else { None };
            if let Some((lo, sh, n)) = a {
                add_venue(b, lo, sh, n, &[("okx", json!(true)), ("okx_detail", detail(lo, sh, n))]);
            }
        }
    }
    if let Some(first) = bybit.as_ref().and_then(|b| b.from) {
        venues.push(json!("bybit-perp"));
        let add: BTreeMap<i64, (f64, f64, i64)> = bybit.map(|b| b.bars.into_iter().map(|r| (r.0, (r.1, r.2, r.3))).collect()).unwrap_or_default();
        for b in &mut out {
            let t = bar_t(b);
            if t > first {
                let (lo, sh, n) = add.get(&t).copied().unwrap_or((0.0, 0.0, 0));
                add_venue(b, lo, sh, n, &[("bybit", detail(lo, sh, n))]);
            }
        }
    }
    if let Some(db) = hl_db(&asset) {
        let since = ((now - (FP_KEEP * BAR_S) as f64) * 1000.0).trunc() as i64;
        let (c, a2) = (cfg.clone(), asset.clone());
        let ev = cfg.swr.get(&format!("hl_whale_liq_events:{asset}"), 30.0, 0.0, Arc::new(move || {
            Ok(json!(hl_whale_liq_events(&c, since, &db, &a2.to_uppercase()).into_iter().map(|(t, u, l, w)| json!([t, f(u), l, w])).collect::<Vec<_>>()))
        })).unwrap_or(json!([]));
        let mut add: BTreeMap<i64, (f64, f64, i64, Vec<String>)> = BTreeMap::new();
        for e in ev.as_array().into_iter().flatten() {
            let a = add.entry(e[0].as_i64().unwrap_or(0).div_euclid(1000).div_euclid(BAR_S) * BAR_S).or_insert((0.0, 0.0, 0, Vec::new()));
            if e[2].as_bool() == Some(true) { a.0 += e[1].as_f64().unwrap_or(0.0) } else { a.1 += e[1].as_f64().unwrap_or(0.0) }
            a.2 += 1;
            let u = e[3].as_str().unwrap_or("").to_string();
            if !a.3.contains(&u) {
                a.3.push(u);
            }
        }
        let mut any = false;
        for b in &mut out {
            if let Some((lo, sh, n, users)) = add.get(&bar_t(b)) {
                any = true;
                add_venue(b, *lo, *sh, *n, &[("hl", json!({"long_usd": rnd(*lo), "short_usd": rnd(*sh), "n": n, "users": users[..users.len().min(5)]}))]);
            }
        }
        if any {
            venues.push(json!("hyperliquid-whales"));
        }
    }
    payload.insert("bars".into(), Value::Array(out));
    payload.insert("venues".into(), Value::Array(venues));
    Resp::json(&Value::Object(payload))
}
