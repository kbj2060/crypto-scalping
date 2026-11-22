//! 아홉째 조각(2026-11-22): 요청 때 바이낸스 REST 를 부르는 주소 -- 바깥 요청은 전부 model::fetch_binance(→ binance_get 한 문 · RTSERVE_REST 있을 때만).
//! 원본(전부 Read 로 읽은 줄): api_trend(server.py:2743~2760) · trend_rule.trend_payload(dashboard/trend_rule.py:22~68) ·
//!   api_swing_lines(server.py:2762~2794) · swing_lines.swing_state(dashboard/swing_lines.py:17~44) · api_cycle_path(server.py:4820~4845) · fetch_binance_json(1531~1551) · swr_cached(1482~1517) · STALE_GRACE_SECONDS 600(1040).
//! 시험: tests/rest_parity.rs(정답지 = tools/gen_rest_fixture.py · 127.0.0.1 가짜 REST).

use std::sync::Arc;

use serde_json::{json, Value};

use crate::model::fetch_binance;
use crate::py::{round_nd, PyErr, R, V};
use crate::{http_exc, server_error, Config, Handler, Req, Resp, MARKET_ASSETS};

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    let h: Handler = match p {
        "/api/trend" => api_trend,
        "/api/cycle-path" => api_cycle_path,
        "/api/swing-lines" => api_swing_lines,
        _ => return Vec::new(),
    };
    vec![("GET", h)]
}

fn market_symbol(asset: &str) -> Option<&'static str> {
    MARKET_ASSETS.iter().position(|a| *a == asset).map(|i| crate::hub::MARKET_SYMBOL[i])
}

fn klines(cfg: &Config, q: &str, reason: &str) -> R<Vec<Value>> {
    let v = fetch_binance(cfg, &format!("{}/fapi/v1/klines?{q}", cfg.rest), Some(reason)).map_err(PyErr)?;
    v.and_then(|v| v.as_array().cloned()).ok_or_else(|| PyErr(format!("TypeError: '{}' object is not iterable", "NoneType")))
}

fn f64_of(v: &Value) -> f64 {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse().ok())).unwrap_or(f64::NAN)
}

// ───────────────────────── /api/trend ─────────────────────────

const LOOKBACKS: [usize; 5] = [7, 14, 28, 56, 90];
const F1_LOOKBACKS: [usize; 4] = [7, 14, 21, 28];

/// trend_rule.trend_payload(22~68) -- 합은 파이썬 sum(왼쪽부터) · log·sqrt = libm(파이썬 math 와 같은 함수)
pub fn trend_payload(c: &[f64], ct: &[i64]) -> V {
    let n = c.len();
    let need = 90 + 2;
    if n < need || c.iter().any(|x| !(*x > 0.0)) {
        return json!({"ok": false, "reason": format!("일봉이 모자랍니다 ({n}/{need})")});
    }
    let votes: Vec<V> = LOOKBACKS.iter().map(|&l| json!({"L": l, "up": c[n - 1] > c[n - 1 - l], "ref_now": c[n - 1 - l], "flip_price": c[n - l]})).collect();
    let ups = LOOKBACKS.iter().filter(|&&l| c[n - 1] > c[n - 1 - l]).count();
    let signal = LOOKBACKS.iter().map(|&l| if c[n - 1] > c[n - 1 - l] { 1i64 } else { -1 }).sum::<i64>() as f64 / 5.0;
    let rets: Vec<f64> = (n - 20..n).map(|i| (c[i] / c[i - 1]).ln()).collect();
    let mu = rets.iter().fold(0.0, |a, r| a + r) / 20.0;
    let vol_ann = (rets.iter().fold(0.0, |a, r| a + (r - mu) * (r - mu)) / 19.0).sqrt() * 365f64.sqrt();
    let size = if vol_ann > 0.0 { let v = signal * 0.5 / vol_ann; let m = if v < 2.0 { v } else { 2.0 }; if m > -2.0 { m } else { -2.0 } } else { 0.0 };
    let sig_at = |i: usize| LOOKBACKS.iter().map(|&l| if c[i] > c[i - l] { 1i64 } else { -1 }).sum::<i64>();
    let sgn = if signal > 0.0 { 1 } else { -1 };
    let mut age = 0;
    for i in (90..n).rev() {
        if (if sig_at(i) > 0 { 1 } else { -1 }) != sgn {
            break;
        }
        age += 1;
    }
    let cmp = |a: f64, b: f64| (a > b) as i64 - (a < b) as i64;
    let f1v: Vec<V> = F1_LOOKBACKS.iter().map(|&l| json!({"L": l, "s": cmp(c[n - 1], c[n - 1 - l]), "ref_now": c[n - 1 - l]})).collect();
    let f1s: i64 = F1_LOOKBACKS.iter().map(|&l| cmp(c[n - 1], c[n - 1 - l])).sum();
    let f1 = json!({"signal": (f1s > 0) as i64 - (f1s < 0) as i64,
        "ups": F1_LOOKBACKS.iter().filter(|&&l| cmp(c[n - 1], c[n - 1 - l]) > 0).count(),
        "downs": F1_LOOKBACKS.iter().filter(|&&l| cmp(c[n - 1], c[n - 1 - l]) < 0).count(), "votes": f1v});
    let sma50: Vec<V> = (49.max(n - 14)..n).map(|i| json!({"t": ct[i], "v": round_nd(c[i - 49..=i].iter().fold(0.0, |a, x| a + x) / 50.0, 6), "c": c[i]})).collect();
    json!({"ok": true, "signal": round_nd(signal, 1), "ups": ups, "n_votes": 5, "votes": votes, "close": c[n - 1], "asof_ms": ct[n - 1],
        "vol_ann": round_nd(vol_ann, 4), "size": round_nd(size, 3), "age_days": age, "lookbacks": LOOKBACKS, "f1": f1, "sma50": sma50})
}

fn api_trend(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    let Some(sym) = market_symbol(&asset) else { return http_exc(400, Some("unsupported_asset"), &[]) };
    let c = cfg.clone();
    let r = cfg.swr.get(&format!("trend_{asset}_1d"), 300.0, 600.0, Arc::new(move || {
        let raw = klines(&c, &format!("symbol={sym}&interval=1d&limit=120"), "trend_upstream_error")?;
        let now_ms = (c.now)() * 1000.0;
        let done: Vec<&Value> = raw.iter().filter(|r| (r[6].as_i64().unwrap_or(i64::MAX) as f64) < now_ms).collect();
        Ok(trend_payload(&done.iter().map(|r| f64_of(&r[4])).collect::<Vec<_>>(), &done.iter().map(|r| r[6].as_i64().unwrap_or(0) + 1).collect::<Vec<_>>()))
    }));
    // 원본 except Exception → {"ok": False, "reason": f"{type}: {exc}"}
    Resp::json(&r.unwrap_or_else(|PyErr(m)| json!({"ok": false, "reason": m})))
}

// ───────────────────────── /api/cycle-path ─────────────────────────

/// orders.py:347~353 MANUAL_EXEC_SYMBOLS 값(eth = env DASHBOARD_MANUAL_EXEC_SYMBOL 또는 ETHUSDT)
fn exec_symbols() -> Vec<String> {
    let eth = std::env::var("DASHBOARD_MANUAL_EXEC_SYMBOL").ok().map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).unwrap_or("ETHUSDT".into());
    [eth.as_str(), "SOLUSDC", "XRPUSDC", "BTCUSDC", "SNDKUSDT", "SOXLUSDT"].iter().map(|s| s.to_string()).collect()
}

/// 파이썬 int(글자): 앞뒤 공백 · 부호 · 숫자 사이 밑줄 하나 -- 범위 밖(i64 넘침)은 None(부르는 쪽 범위 검사와 같은 400)
pub fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let (neg, d) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let b = d.as_bytes();
    if b.is_empty() || !b[0].is_ascii_digit() || !b[b.len() - 1].is_ascii_digit() || d.contains("__") || !b.iter().all(|c| c.is_ascii_digit() || *c == b'_') {
        return None;
    }
    let v: i64 = d.replace('_', "").parse().ok()?;
    Some(if neg { -v } else { v })
}

fn api_cycle_path(cfg: &Config, req: &Req) -> Resp {
    let symbol = req.q("symbol").unwrap_or("").to_uppercase();
    if !crate::hub::MARKET_SYMBOL.contains(&symbol.as_str()) && !exec_symbols().contains(&symbol) {
        return http_exc(400, Some("unsupported_symbol"), &[]);
    }
    let raw = req.q("start").filter(|s| !s.is_empty()).unwrap_or("0");
    let now_ms = ((cfg.now)() * 1000.0) as i64;
    let start = match py_int(raw) {
        Some(s) if now_ms - 30 * 86_400_000 <= s && s <= now_ms => s,
        // 숫자 꼴인데 i64 를 넘는 값도 범위 밖 = 같은 bad_start
        _ => return http_exc(400, Some("bad_start"), &[]),
    };
    let span_min = (now_ms - start) as f64 / 60_000.0;
    let interval = if span_min <= 1490.0 { "1m" } else if span_min <= 7450.0 { "5m" } else { "15m" };
    let c = cfg.clone();
    let sym = symbol.clone();
    // ponytail: 원본처럼 swr 키에 start(ms)가 들어가 키 수 상한·정리 없음 -- 화면은 순환 첫 체결 시각 하나를 보낸다(§9-1)
    let r = cfg.swr.get(&format!("cycle_path:{symbol}:{start}"), 20.0, 0.0, Arc::new(move || {
        let raw = klines(&c, &format!("symbol={sym}&interval={interval}&startTime={start}&limit=1500"), "cycle_path_upstream_error")?;
        Ok(json!({"ok": true, "interval": interval, "rows": raw.iter().map(|r| json!([r[0].as_i64().unwrap_or(0), f64_of(&r[4])])).collect::<Vec<_>>()}))
    }));
    match r {
        Ok(v) => Resp::json(&v),
        // 원본 except web.HTTPException → {"ok": False, "reason": exc.reason, "rows": []} · 그 밖 예외(연결 실패 등)는 500 · RestOff 는 «못 받음» 갈래로
        Err(PyErr(m)) if m.starts_with("HTTP") => Resp::json(&json!({"ok": false, "reason": m.split_once(": ").map_or(m.as_str(), |x| x.1), "rows": []})),
        Err(PyErr(m)) if m.starts_with("RestOff") => Resp::json(&json!({"ok": false, "reason": m, "rows": []})),
        Err(_) => server_error(),
    }
}

// ───────────────────────── /api/swing-lines ─────────────────────────

const SWING_N: usize = 50;

/// swing_lines._pivots: 봉 p 가 앞뒤 50봉(양끝 포함) 최고(최저)와 같으면 참
fn pivots(x: &[f64], high: bool) -> Vec<bool> {
    let n = x.len();
    let mut out = vec![false; n];
    for p in SWING_N..n.saturating_sub(SWING_N) {
        let w = &x[p - SWING_N..=p + SWING_N];
        let e = if high { w.iter().cloned().fold(f64::NEG_INFINITY, f64::max) } else { w.iter().cloned().fold(f64::INFINITY, f64::min) };
        out[p] = x[p] == e;
    }
    out
}

/// swing_lines.swing_state(25~44)
pub fn swing_state(h: &[f64], l: &[f64], c: &[f64], t: &[i64]) -> V {
    let (ph, pl) = (pivots(h, true), pivots(l, false));
    let (mut top, mut btm, mut trend, mut last): (Option<(f64, V)>, Option<(f64, V)>, i64, V) = (None, None, 0, V::Null);
    for i in 0..c.len() {
        if i >= SWING_N {
            let p = i - SWING_N;
            if ph[p] {
                top = Some((h[p], json!({"price": h[p], "pivot_t": t[p], "known_t": t[i]})));
            }
            if pl[p] {
                btm = Some((l[p], json!({"price": l[p], "pivot_t": t[p], "known_t": t[i]})));
            }
        }
        if let Some((px, _)) = top.as_ref().filter(|(px, _)| c[i] > *px) {
            last = json!({"dir": 1, "kind": if trend == -1 { "CHoCH" } else { "BOS" }, "level": *px, "t": t[i]});
            (top, trend) = (None, 1);
        }
        if let Some((px, _)) = btm.as_ref().filter(|(px, _)| c[i] < *px) {
            last = json!({"dir": -1, "kind": if trend == 1 { "CHoCH" } else { "BOS" }, "level": *px, "t": t[i]});
            (btm, trend) = (None, -1);
        }
    }
    let v = |x: Option<(f64, V)>| x.map_or(V::Null, |x| x.1);
    json!({"hi": v(top), "lo": v(btm), "last_break": last, "trend": trend, "asof_t": t.last()})
}

fn api_swing_lines(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    let Some(sym) = market_symbol(&asset) else { return http_exc(400, Some("unsupported_asset"), &[]) };
    let c = cfg.clone();
    let r = cfg.swr.get(&format!("swing_{asset}"), 60.0, 600.0, Arc::new(move || {
        let mut out = serde_json::Map::new();
        out.insert("ok".into(), json!(true));
        let now_ms = (c.now)() * 1000.0;
        let mut errs = Vec::new();
        for tf in ["1h", "30m"] {
            // 원본 except Exception → out[tf] = None · f"{tf}: {type(exc).__name__}" (한쪽 실패가 다른 쪽 선을 지우지 않게)
            let raw = match klines(&c, &format!("symbol={sym}&interval={tf}&limit=1000"), "swing_upstream_error") {
                Ok(r) => r,
                Err(PyErr(m)) => {
                    out.insert(tf.into(), V::Null);
                    errs.push(format!("{tf}: {}", m.split(':').next().unwrap_or("")));
                    continue;
                }
            };
            let done: Vec<&Value> = raw.iter().filter(|r| (r[6].as_i64().unwrap_or(i64::MAX) as f64) < now_ms).collect();
            let col = |k: usize| done.iter().map(|r| f64_of(&r[k])).collect::<Vec<_>>();
            out.insert(tf.into(), swing_state(&col(2), &col(3), &col(4), &done.iter().map(|r| r[6].as_i64().unwrap_or(0) + 1).collect::<Vec<_>>()));
        }
        if errs.len() == 2 {
            return Err(PyErr(format!("RuntimeError: {}", errs.join("; "))));
        }
        if !errs.is_empty() {
            out.insert("partial".into(), json!(errs));
        }
        Ok(V::Object(out))
    }));
    Resp::json(&r.unwrap_or_else(|PyErr(m)| json!({"ok": false, "reason": m})))
}

#[cfg(test)]
mod tests {
    #[test]
    fn py_int_like_python() {
        use super::py_int;
        for (s, w) in [("12", Some(12)), (" 12 ", Some(12)), ("+5", Some(5)), ("-5", Some(-5)), ("1_000", Some(1000)), ("1__0", None), ("_1", None), ("1_", None), ("", None), ("1.0", None), ("abc", None), ("99999999999999999999", None)] {
            assert_eq!(py_int(s), w, "{s:?}");
        }
    }
}
