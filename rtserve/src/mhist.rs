//! 차트 과거 봉(2026-11-28): /api/market-history -- 바깥 요청은 model::fetch_binance(→ binance_get 한 문 · RTSERVE_REST 있을 때만).
//! 원본(Read 로 읽은 줄): server.py forming_kline 404~415 · EVIDENCE_ASSETS 195 · trend_veto_rows 1619~1652 · kline_candles 1654~1667 ·
//!   load_market_history_from_evidence_cache 1671~1722 · load_market_history 1724~1742 · load_chart_klines_frames 3193~3281(swr "evidence_signal" 60초 · 묵음 0) ·
//!   api_market_history 3709~3720 · dashboard/market_ctx.py market_session·session_starts 223~237 · session_vwap 240~258 · vwap_rows 305~332 · orders.py MARKET_SYMBOLS 340.
//! 시험: tests/mhist_parity.rs(정답지 = tools/gen_mhist_fixture.py · 127.0.0.1 가짜 REST).

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Map, Value as V};

use crate::api::days_from_civil;
use crate::ctx_rest::{evidence_frame, ewm_mean, parse_klines, rolling_mean, Kline, TREND_VETO_K, TREND_VETO_N, VETO_TAIL};
use crate::model::fetch_binance;
use crate::py::{round_nd, PyErr, R};
use crate::{http_exc, server_error, Config, Handler, Req, Resp};

const TTL: f64 = 60.0;
const STALE: f64 = 600.0;

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/market-history" => vec![("GET", api_market_history)],
        _ => Vec::new(),
    }
}

/// trend_veto_rows: 봉 시각(초) → {sma, atr, veto[, drop, vn, vk]}. `rows` = 시간순(마감봉).
pub fn trend_veto_rows(rows: &[Kline], tail: usize) -> BTreeMap<i64, Map<String, V>> {
    let w = &rows[rows.len().saturating_sub(tail)..];
    let mut out = BTreeMap::new();
    if w.len() < TREND_VETO_N + 1 {
        return out;
    }
    let c: Vec<f64> = w.iter().map(|k| k.close).collect();
    let sma = rolling_mean(&c, TREND_VETO_N);
    let tr: Vec<f64> = (0..w.len())
        .map(|k| { let hl = w[k].high - w[k].low; if k == 0 { hl } else { hl.max((w[k].high - c[k - 1]).abs()).max((w[k].low - c[k - 1]).abs()) } })
        .collect();
    let atr = ewm_mean(&tr, 1.0 / TREND_VETO_N as f64);
    let mut cur = 0;
    for k in 0..w.len() {
        let (dev, eps) = ((c[k] - sma[k]) / sma[k], TREND_VETO_K * atr[k] / c[k]);
        if !(dev.is_finite() && eps.is_finite()) {
            continue;
        }
        if dev > eps {
            cur = 1;
        } else if dev < -eps {
            cur = -1;
        }
        let mut r = Map::new();
        r.insert("sma".into(), json!(round_nd(sma[k], 6)));
        r.insert("atr".into(), json!(round_nd(atr[k], 6)));
        r.insert("veto".into(), json!(cur));
        if k + 1 >= TREND_VETO_N {
            r.insert("drop".into(), json!(round_nd(c[k + 1 - TREND_VETO_N], 6)));
            r.insert("vn".into(), json!(TREND_VETO_N));
            r.insert("vk".into(), json!(TREND_VETO_K));
        }
        out.insert(w[k].open_ms.div_euclid(1000), r);
    }
    out
}

/// 그레고리력 날 수 → (년, 월, 일)
fn civil(d: i64) -> (i64, i64, i64) {
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as i64, m, doy - (153 * mp + 2) / 5 + 1)
}

/// y 년 m 월 n 째 일요일(n = 0 이면 마지막)의 날 수 -- 1970-01-01 = 목요일
fn sunday(y: i64, m: i64, n: i64) -> i64 {
    let dow = |d: i64| (d + 4).rem_euclid(7); // 0 = 일요일
    if n == 0 {
        let last = if m == 12 { days_from_civil(y + 1, 1, 1) } else { days_from_civil(y, m + 1, 1) } - 1;
        last - dow(last)
    } else {
        let first = days_from_civil(y, m, 1);
        first + (7 - dow(first)) % 7 + 7 * (n - 1)
    }
}

const SESSIONS: [&str; 3] = ["아시아", "유럽", "미국"];

/// session_starts(day): 도쿄 09:00(+9 · 서머타임 없음) · 런던 08:00(BST = 3월 마지막 일요일 ~ 10월 마지막 일요일 · 전환 01:00 UTC 가 08:00 앞) ·
/// 뉴욕 09:30(EDT = 3월 둘째 일요일 ~ 11월 첫 일요일 · 전환 02:00 현지가 09:30 앞) -- ZoneInfo 와 같은 규칙(2007~ 미국 · 1996~ 유럽).
pub fn session_starts(day: i64) -> [i64; 3] {
    let (y, _, _) = civil(day);
    let bst = day >= sunday(y, 3, 0) && day < sunday(y, 10, 0);
    let edt = day >= sunday(y, 3, 2) && day < sunday(y, 11, 1);
    let d0 = day * 86_400;
    [d0, d0 + if bst { 7 } else { 8 } * 3600, d0 + if edt { 13 * 3600 + 1800 } else { 14 * 3600 + 1800 }]
}

fn market_session(t: i64) -> (i64, usize) {
    let st = session_starts(t.div_euclid(86_400));
    let i = st.partition_point(|&s| s <= t);
    if i == 0 { (st[0], 0) } else { (st[i - 1], i - 1) }
}

fn session_vwap(k: &[Kline], key: impl Fn(i64) -> i64) -> Vec<Option<(f64, f64)>> {
    let (mut day, mut sv, mut spv, mut sp2v) = (None, 0.0, 0.0, 0.0);
    k.iter().map(|b| {
        let t = b.open_ms.div_euclid(1000);
        if day != Some(key(t)) {
            (day, sv, spv, sp2v) = (Some(key(t)), 0.0, 0.0, 0.0);
        }
        let tp = (b.high + b.low + b.close) / 3.0;
        sv += b.volume;
        spv += tp * b.volume;
        sp2v += tp * tp * b.volume;
        (sv > 0.0).then(|| { let m = spv / sv; (m, (sp2v / sv - m * m).max(0.0).sqrt()) })
    }).collect()
}

fn cum(k: &[Kline], key: impl Fn(i64) -> i64) -> Vec<f64> {
    let (mut k0, mut s) = (None, 0.0);
    k.iter().map(|b| {
        let t = b.open_ms.div_euclid(1000);
        if k0 != Some(key(t)) {
            (k0, s) = (Some(key(t)), 0.0);
        }
        s += b.volume;
        s
    }).collect()
}

/// mctx.vwap_rows(ts, high, low, close, vol, nd)
pub fn vwap_rows(k: &[Kline], nd: usize) -> BTreeMap<i64, Map<String, V>> {
    let day = |t: i64| t.div_euclid(86_400);
    let ses = |t: i64| market_session(t).0;
    let (vw, sw, vc, svc) = (session_vwap(k, day), session_vwap(k, ses), cum(k, day), cum(k, ses));
    let mut out = BTreeMap::new();
    for (i, b) in k.iter().enumerate() {
        let t = b.open_ms.div_euclid(1000);
        let mut r = Map::new();
        if let Some((a, s)) = vw[i] {
            r.insert("vwap".into(), json!(round_nd(a, nd)));
            r.insert("vsd".into(), json!(round_nd(s, nd)));
            r.insert("vcum".into(), json!(round_nd(vc[i], 3)));
        }
        if let Some((a, s)) = sw[i] {
            let (st, nm) = market_session(t);
            r.insert("svwap".into(), json!(round_nd(a, nd)));
            r.insert("svsd".into(), json!(round_nd(s, nd)));
            r.insert("svcum".into(), json!(round_nd(svc[i], 3)));
            r.insert("sstart".into(), json!(st));
            r.insert("sname".into(), json!(SESSIONS[nm]));
        }
        if !r.is_empty() {
            out.insert(t, r);
        }
    }
    out
}

fn candle(k: &Kline, vol: bool, extra: &[Option<&Map<String, V>>], partial: bool) -> V {
    let mut r = Map::new();
    r.insert("time".into(), json!(k.open_ms.div_euclid(1000)));
    for (n, x) in [("open", k.open), ("high", k.high), ("low", k.low), ("close", k.close)] {
        r.insert(n.into(), json!(x));
    }
    if vol {
        r.insert("volume".into(), json!(k.volume));
    }
    for m in extra.iter().flatten() {
        r.extend(m.iter().map(|(a, b)| (a.clone(), b.clone())));
    }
    if partial {
        r.insert("partial".into(), json!(true));
    }
    V::Object(r)
}

/// kline_candles(rows, now_ms, nd, veto_tail): 추세 밴드 = 마감봉(close_time < 지금)만 · VWAP = 전부 · 형성 봉 partial.
pub fn kline_candles(rows: &[Kline], now_ms: f64, nd: usize, veto_tail: usize) -> Vec<V> {
    let closed: Vec<Kline> = rows.iter().copied().filter(|k| (k.close_ms as f64) < now_ms).collect();
    let (veto, vwap) = (trend_veto_rows(&closed, veto_tail), vwap_rows(rows, nd));
    rows.iter().map(|k| { let t = k.open_ms.div_euclid(1000); candle(k, true, &[veto.get(&t), vwap.get(&t)], k.close_ms as f64 >= now_ms) }).collect()
}

/// load_market_history_from_evidence_cache 의 행 조립: tail 200 · veto(프레임 tail 720) · VWAP(tail 600 · 거래량 있을 때만)
fn frame_candles(src: &[Kline], vol: bool) -> Vec<V> {
    let veto = trend_veto_rows(src, VETO_TAIL);
    let vwap = if vol { vwap_rows(&src[src.len().saturating_sub(600)..], 4) } else { BTreeMap::new() };
    src[src.len().saturating_sub(200)..].iter().map(|k| { let t = k.open_ms.div_euclid(1000); candle(k, vol, &[veto.get(&t), vwap.get(&t)], false) }).collect()
}

/// forming_kline(raw, now_ms): 마지막 행이 아직 안 닫혔으면(시작 + 300 > 지금) partial 캔들
fn forming(raw: &[Kline], now_ms: f64) -> V {
    match raw.last() {
        Some(k) if (k.open_ms.div_euclid(1000) + 300) as f64 > now_ms / 1000.0 => candle(k, true, &[], true),
        _ => V::Null,
    }
}

fn klines_raw(cfg: &Config, sym: &str, limit: usize, reason: Option<&str>) -> Result<Option<V>, String> {
    fetch_binance(cfg, &format!("{}/fapi/v1/klines?symbol={sym}&interval=5m&limit={limit}", cfg.rest), reason)
}

/// load_chart_klines_frames produce: ETH(실패 = 예외) · BTC(실패 = 없음 · 로그) -- 값 = 두 코인의 차트 행 + 형성 봉(원본 evidence_signal_cache).
fn produce_frames(cfg: &Config) -> R<V> {
    let now_ms = ((cfg.now)() * 1000.0) as i64 as f64; // int(time.time() * 1000)
    let raw = klines_raw(cfg, "ETHUSDT", 1500, Some("evidence_signal_upstream_error")).map_err(PyErr)?.unwrap_or(V::Null);
    let eth = parse_klines(&raw).map_err(PyErr)?;
    let frame = evidence_frame(&raw, now_ms as i64).map_err(PyErr)?;
    let btc = match klines_raw(cfg, "BTCUSDT", 1500, None).and_then(|b| b.map(|b| parse_klines(&b).map(|p| (evidence_frame(&b, now_ms as i64), p))).transpose()) {
        Ok(Some((Ok(f), p))) => Some((f, p)),
        Ok(None) => None,
        Ok(Some((Err(m), _))) | Err(m) => {
            (cfg.log)(&format!("evidence-signal BTC leg failed (smt_divergence family will read as not-fired this cycle): {m}"));
            None
        }
    };
    Ok(json!({"eth": frame_candles(&frame, true), "btc": btc.as_ref().map(|b| frame_candles(&b.0, false)),
              "forming": {"eth": forming(&eth, now_ms), "btc": btc.as_ref().map_or(V::Null, |b| forming(&b.1, now_ms))}}))
}

/// 원본이 잡지 않는 예외 = aiohttp 가 HTTPException 을 그대로(«상태: 이유» 글자) · 그 밖 = 500.
fn uncaught(m: &str) -> Resp {
    for (p, s) in [("HTTPBadGateway: ", 502), ("HTTPServiceUnavailable: ", 503)] {
        if let Some(r) = m.strip_prefix(p) {
            return http_exc(s, Some(r), &[]);
        }
    }
    // RestOff(RTSERVE_REST 없음) = 원본에 없는 갈래 -- 밴과 같은 «안 보냄» 503
    if m.starts_with("RestOff") { http_exc(503, Some("rest_off"), &[]) } else { server_error() }
}

fn api_market_history(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").unwrap_or("").to_lowercase();
    let Some(i) = crate::MARKET_ASSETS.iter().position(|a| *a == asset) else {
        return http_exc(400, Some("unsupported_market_history_asset"), &[]);
    };
    let c = cfg.clone();
    let r = if asset == "eth" || asset == "btc" {
        cfg.swr.get("evidence_signal", TTL, 0.0, Arc::new(move || produce_frames(&c))).and_then(|v| {
            let rows = v[&asset].as_array().filter(|a| !a.is_empty()).cloned().ok_or_else(|| PyErr("HTTPBadGateway: market_history_upstream_error".into()))?;
            let f = &v["forming"][&asset];
            let mut rows = rows;
            if f.is_object() && f["time"].as_i64() > rows.last().and_then(|x| x["time"].as_i64()) {
                rows.push(f.clone());
            }
            Ok(V::Array(rows))
        })
    } else {
        let sym = crate::hub::MARKET_SYMBOL[i];
        cfg.swr.get(&format!("market_history:{asset}"), TTL, STALE, Arc::new(move || {
            let raw = klines_raw(&c, sym, 1000, Some("market_history_upstream_error")).map_err(PyErr)?.unwrap_or(V::Null);
            let v = kline_candles(&parse_klines(&raw).map_err(PyErr)?, (c.now)() * 1000.0, 6, VETO_TAIL);
            Ok(V::Array(v[v.len().saturating_sub(200)..].to_vec()))
        }))
    };
    match r {
        Ok(candles) => Resp::json(&json!({"asset": asset, "candles": candles})),
        Err(PyErr(m)) => uncaught(&m),
    }
}
