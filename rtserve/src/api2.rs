//! 둘째 조각(2026-11-21): 나머지 파일·SQLite 엔드포인트 9 + 쓰기(메모·오답·알림 구독) + client-timing + order_request_log.
//! 원본 위치는 함수마다(dashboard/server.py 2026-11-21 판 · 별도 모듈은 그 파일). swr 캐시는 첫 조각처럼 옮기지 않았다.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rusqlite::types::Value as Sql;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::api::{guarded, hot, int_v, iso_us, now_us, or500, read_text, rows, tail_lines};
use crate::py::{self, err, f, item, PyErr, R, V};
use crate::{http_exc, server_error, Config, Handler, Req, Resp, MARKET_ASSETS};

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    match p {
        "/api/state" => vec![("GET", state)],
        "/api/breakout-detector" => vec![("GET", breakout_detector)],
        "/api/gex" => vec![("GET", gex)],
        "/api/chip-score" => vec![("GET", chip_score)],
        "/api/notify-center" => vec![("GET", notify_center)],
        "/api/news" => vec![("GET", news)],
        "/api/macro-spikes" => vec![("GET", macro_spikes)],
        "/api/bubble-tape" => vec![("GET", bubble_tape)],
        "/api/pred-record" => vec![("GET", pred_record)],
        "/api/acct-notes" => vec![("POST", acct_notes_save)],
        "/api/acct-wrong" => vec![("POST", acct_wrong_save)],
        "/api/push/config" => vec![("GET", push_config)],
        "/api/push/subscribe" => vec![("POST", push_subscribe)],
        "/api/push/unsubscribe" => vec![("POST", push_unsubscribe)],
        "/api/push/devices" => vec![("GET", push_devices)],
        "/api/client-timing" => vec![("POST", client_timing)],
        _ => Vec::new(),
    }
}

fn obj(pairs: Vec<(&str, V)>) -> V {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// `{**a, **b}`(사전 합치기 · 같은 열쇠는 자리 그대로 값만)
fn merge(mut a: Map<String, V>, b: &[(&str, V)]) -> Map<String, V> {
    for (k, v) in b {
        a.insert(k.to_string(), v.clone());
    }
    a
}

/// aiohttp `HTTPBadRequest(text=…)` 등 -- 본문 = 글자 그대로
fn exc_text(status: u16, text: &str) -> Resp {
    let mut r = http_exc(status, None, &[]);
    r.body = Some(text.as_bytes().to_vec());
    r
}

/// json_response(…, status=…) -- NOCACHE 없이(원본 client_timing 오류 응답)
fn json_plain(status: u16, v: &V) -> Resp {
    Resp {
        status,
        headers: vec![("Content-Type".into(), "application/json; charset=utf-8".into())],
        body: Some(rtcollect::py_dumps_default(v).into_bytes()),
        ..Default::default()
    }
}

// ───────────────────────── 워커 상태 (dashboard/worker_state.py) ─────────────────────────

#[derive(Default)]
pub struct Wp {
    pub require_ok: bool,
    pub stamp_available: bool,
    pub bare_missing: bool,
    pub extra: Vec<(&'static str, V)>,
}

/// worker_state.load_json: 없으면 None · 깨졌으면 예외
fn load_json(p: &Path) -> R<V> {
    if !p.exists() {
        return Ok(Value::Null);
    }
    py::loads(&read_text(p).ok_or_else(|| PyErr("UnicodeDecodeError: utf-8".into()))?)
}

/// 원본 worker_payload(worker_state.py:31)
pub fn worker_payload(path: &Path, max_age_min: f64, ts_field: &str, o: Wp, now: f64) -> R<V> {
    let mut base: Vec<(&str, V)> = vec![("available", json!(false)), ("error", json!("worker_state_missing"))];
    if !o.bare_missing {
        base.push(("tone", json!("neutral")));
    }
    base.push(("subText", json!("데이터 없음")));
    let st = load_json(path)?;
    let missing = || {
        let mut m = merge(Map::new(), &base);
        m = merge(m, &o.extra);
        Value::Object(m)
    };
    if !py::truthy(&st) {
        return Ok(missing());
    }
    let Value::Object(stm) = &st else { return err("TypeError", format!("'{}' object is not a mapping", py::type_name(&st))) };
    if o.require_ok && !py::truthy(stm.get("ok").unwrap_or(&Value::Null)) {
        if o.bare_missing {
            return Ok(missing());
        }
        let m = merge(merge(merge(stm.clone(), &base), &o.extra), &[("error", json!("worker_fetch_failed")), ("subText", json!("오류"))]);
        return Ok(Value::Object(m));
    }
    let ts = stm.get(ts_field).cloned().unwrap_or(Value::Null);
    let age = iso_us(&py::to_str(&ts)).map(|t| py::round_nd((now_us(now) - t) as f64 / 1e6 / 60.0, 1));
    let stale = age.map_or(Value::Null, |a| f(py::round_nd(a, 1)));
    if age.is_some_and(|a| a > max_age_min) {
        return Ok(Value::Object(merge(merge(stm.clone(), &base), &[("error", json!("worker_stale")), ("stale_min", stale)])));
    }
    let mut m = stm.clone();
    if o.stamp_available {
        m.insert("available".into(), json!(true));
    }
    m.insert("stale_min".into(), stale);
    Ok(Value::Object(m))
}

/// GET /api/breakout-detector (server.py:4456 · 1156) -- 워커는 꺼졌다(질문지 20) · 원본은 지금도 이 규칙대로 «없음/낡음»을 낸다
fn breakout_detector(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if !["eth", "sol", "xrp"].contains(&asset.as_str()) {
        return http_exc(404, Some("breakout_off"), &[]);
    }
    let p = cfg.live_dir().join(format!("{asset}_breakout_detector_state.json"));
    let c = cfg.clone();
    let wp = cfg.swr.get(&format!("breakout_detector_{asset}"), 60.0, 600.0, Arc::new(move || {
        let o = Wp { require_ok: true, stamp_available: true, extra: vec![("history", json!([])), ("times", json!([]))], ..Default::default() };
        worker_payload(&p, 15.0, "updated_utc", o, (c.now)())
    }));
    or500(wp.and_then(|v| match v {
        Value::Object(m) => Ok(Value::Object(merge(m, &[("asset", json!(asset))]))),
        _ => unreachable!(),
    }))
}

/// GET /api/gex (server.py:4715 → gex_payload 1208)
fn gex_payload(cfg: &Config) -> R<V> {
    let now = (cfg.now)();
    let lv = cfg.live_dir();
    let out = worker_payload(&lv.join("deribit_gex_state.json"), 35.0, "generated_at",
        Wp { stamp_available: true, bare_missing: true, extra: vec![("currencies", json!({}))], ..Default::default() }, now)?;
    let Value::Object(mut out) = out else { unreachable!() };
    // out.setdefault("currencies", {}).update(json.loads(cboe)["currencies"]) -- (OSError, ValueError, KeyError) 만 삼킨다
    let cboe = read_text(&lv.join("cboe_option_state.json")).ok_or(()).and_then(|s| py::loads(&s).map_err(|_| ()));
    if let Ok(c) = cboe {
        match py::item(&c, "currencies") {
            Ok(add) => {
                let cur = out.entry("currencies").or_insert_with(|| json!({}));
                let (Value::Object(cur), Value::Object(add)) = (cur, add) else { return err("AttributeError", "'update'") };
                for (k, v) in add {
                    cur.insert(k.clone(), v.clone());
                }
            }
            Err(PyErr(m)) if m.starts_with("KeyError") => {}
            Err(e) => return Err(e),
        }
    }
    let blk = worker_payload(&lv.join("deribit_block_trades_state.json"), 3.0, "generated_at",
        Wp { stamp_available: true, bare_missing: true, extra: vec![("blocks", json!([]))], ..Default::default() }, now)?;
    let g = |k: &str| blk.get(k).cloned().unwrap_or(Value::Null);
    let mut by_coin: Map<String, V> = ["ETH", "BTC", "SOL", "XRP"].iter().map(|c| (c.to_string(), json!([]))).collect();
    let upd = if py::truthy(&g("blocks_by_coin")) { g("blocks_by_coin") } else { json!({"ETH": if py::truthy(&g("blocks")) { g("blocks") } else { json!([]) }}) };
    let Value::Object(upd) = upd else { return err("TypeError", "'list' object is not a mapping") };
    for (k, v) in upd {
        by_coin.insert(k, v);
    }
    let list = |v: &V| -> R<Vec<V>> {
        match v {
            Value::Array(a) => Ok(a.clone()),
            v if !py::truthy(v) => Ok(Vec::new()),
            _ => err("TypeError", "not a list"),
        }
    };
    let (mut first8, mut sums, mut maxes, mut ns) = (Map::new(), Map::new(), Map::new(), Map::new());
    for (c, v) in &by_coin {
        let a = list(v)?;
        first8.insert(c.clone(), Value::Array(a.iter().take(8).cloned().collect()));
        let usd = |b: &V| -> R<V> { Ok(py::get(b, "notional_usd")?.filter(|x| py::truthy(x)).cloned().unwrap_or(json!(0))) };
        let mut s = json!(0);
        for b in &a {
            s = py::add(&s, &usd(b)?)?;
        }
        sums.insert(c.clone(), s);
        let mut best: Option<(&V, V)> = None;
        for b in &a {
            let k = usd(b)?;
            if best.as_ref().map_or(Ok(true), |(_, bk)| py::gt(&k, bk))? {
                best = Some((b, k));
            }
        }
        maxes.insert(c.clone(), best.map_or(Value::Null, |(b, _)| b.clone()));
        ns.insert(c.clone(), json!(a.len()));
    }
    let n_by = if py::truthy(&g("n_blocks_by_coin")) { g("n_blocks_by_coin") } else { Value::Object(ns) };
    let flow = if py::truthy(&g("flow_by_coin")) { g("flow_by_coin") } else { json!({}) };
    out.insert("block_trades".into(), obj(vec![("available", json!(py::truthy(&g("available")))), ("error", g("error")), ("by_coin", Value::Object(first8)),
        ("sum_by_coin", Value::Object(sums)), ("max_by_coin", Value::Object(maxes)), ("n_by_coin", n_by), ("flow_by_coin", flow)]));
    Ok(Value::Object(out))
}

fn gex(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    or500(cfg.swr.get("gex", 60.0, 600.0, Arc::new(move || gex_payload(&c))))
}

// ───────────────────────── /api/state (server.py:3647 · 1544) ─────────────────────────

/// 명시 ETag `W/"state-<sha256(repr((서명,)))>"` · 본문은 json.dumps(ensure_ascii=False, separators=(",", ":")) · application/json(문자셋 없음)
fn state(cfg: &Config, req: &Req) -> Resp {
    let p = cfg.live_dir().join("dashboard_state.json");
    let sig = fs::metadata(&p).ok().map(|m| {
        use std::os::unix::fs::MetadataExt;
        (m.mtime() as i128 * 1_000_000_000 + m.mtime_nsec() as i128, m.size())
    });
    let rep = match sig {
        Some((a, b)) => format!("(({a}, {b}),)"),
        None => "(None,)".into(),
    };
    let etag = format!("W/\"state-{}\"", &crate::hex(&Sha256::digest(rep.as_bytes()))[..16]);
    let headers = vec![("ETag".to_string(), etag.clone()), ("Cache-Control".to_string(), "no-cache".to_string())];
    if req.header("if-none-match").unwrap_or("").split(',').any(|c| {
        let c = c.trim();
        c == "*" || c == etag
    }) {
        return Resp { status: 304, headers, ..Default::default() };
    }
    let Ok(st) = load_json(&p) else { return server_error() };
    let body = py::dumps(&json!({"state": st}), (",", ":"), false, None);
    let mut h = headers;
    h.push(("Content-Type".into(), "application/json".into()));
    Resp { status: 200, headers: h, body: Some(body.into_bytes()), compress: true, ..Default::default() }
}

// ───────────────────────── 칩 성적 (server.py:2088 · 299~328) ─────────────────────────

fn ym(ts: f64) -> String {
    let days = (ts / 86400.0).floor() as i64;
    let (y, m, _) = civil(days);
    format!("{y:04}{m:02}")
}

/// 1970-01-01 부터 날 수 → (연, 월, 일)
pub fn civil(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// 파이썬 텍스트 모드 줄(newline=None: \r\n·\r 도 \n 으로) -- 줄 끝 \n 을 남긴 채
fn text_lines(s: &str) -> Vec<String> {
    let t = s.replace("\r\n", "\n").replace('\r', "\n");
    t.split_inclusive('\n').map(str::to_string).collect()
}

fn chip_score_payload(lv: &Path, now: f64, asset: &str) -> R<V> {
    let since = now - 14.0 * 86400.0;
    let mut paths = vec![lv.join(format!("chip_score_{}.jsonl", ym(since))), lv.join(format!("chip_score_{}.jsonl", ym(now)))];
    paths.sort();
    paths.dedup();
    let mut burst: Vec<V> = Vec::new();
    for p in paths {
        let Ok(b) = fs::read(&p) else { continue };
        let s = String::from_utf8(b).map_err(|_| PyErr("UnicodeDecodeError: utf-8".into()))?;
        for line in text_lines(&s) {
            let Ok(r) = py::loads(&line) else { continue };
            if !py::get(&r, "a")?.is_some_and(|a| a == asset) {
                continue;
            }
            let t = py::get(&r, "t")?.filter(|x| py::truthy(x)).cloned().unwrap_or(json!(0));
            if py::to_float(&t)? >= since && py::get(&r, "k")? == Some(&json!("burst")) {
                burst.push(r);
            }
        }
    }
    let mut out = Map::new();
    out.insert("days".into(), json!(14));
    if !burst.is_empty() {
        let n = burst.len() as f64;
        let (mut s, mut pos) = (json!(0), 0i64);
        for r in &burst {
            let v = item(r, "rev_bp")?;
            s = py::add(&s, v)?;
            pos += i64::from(py::gt(v, &json!(0))?);
        }
        out.insert("burst".into(), obj(vec![("n", json!(burst.len())), ("rev_mean_bp", f(py::div(&s, &f(n))?)), ("rev_share", f(pos as f64 / n))]));
    }
    Ok(Value::Object(out))
}

/// GET /api/chip-score -- 흐름 엔진이 없는 코인 = 404 flow_off
fn chip_score(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    if !cfg.flows().contains(&asset.as_str()) {
        return http_exc(404, Some("flow_off"), &[]);
    }
    let c = cfg.clone();
    or500(cfg.swr.get(&format!("chip_score_{asset}"), 60.0, 600.0, Arc::new(move || chip_score_payload(&c.live_dir(), (c.now)(), &asset))))
}

// ───────────────────────── 알림 센터 (server.py:336) ─────────────────────────

/// scripts/live_push_notifier_20260904.py:131 PUSH_KINDS(그 모듈은 import 때 .env 를 읽어 옮겨 적었다 · LIQ_WARN_PCT 5.0 · EXPO_WARN_X 6.0 의 :g)
fn push_kinds() -> Vec<V> {
    vec![
        json!({"key": "verdict", "name": "판정 예정일", "when": "사전등록 판정일 아침 9시(KST)에 한 번"}),
        json!({"key": "risk", "name": "포지션 위험", "when": "청산까지 5% 미만 · 명목이 순자산의 6배 초과로 들어설 때 한 번"}),
        json!({"key": "burst_hold", "name": "보유 중 청산 급증", "when": "ETH 보유 중 직전 60초 한쪽 청산이 학습 7일 상위 0.5%를 넘을 때 — 그 순간 시장가로 닫으면 평균 −17bp였다"}),
    ]
}

fn notify_center_payload(cfg: &Config) -> R<V> {
    let cal = read_text(&cfg.root.join("dashboard/verdict_calendar.json")).and_then(|s| py::loads(&s).ok()).unwrap_or_else(|| json!({}));
    let mut recent = Vec::new();
    if let Ok(b) = fs::read(cfg.live_dir().join("push_sent_log.jsonl")) {
        let s = String::from_utf8(b).map_err(|_| PyErr("UnicodeDecodeError: utf-8".into()))?;
        let ls = text_lines(&s);
        for l in &ls[ls.len().saturating_sub(30)..] {
            if let Ok(v) = py::loads(l) {
                recent.push(v);
            }
        }
    }
    let items = py::get(&cal, "items")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!([]));
    let Value::Array(mut items) = items else { return err("TypeError", "items") };
    if let Some(st) = read_text(&cfg.live_dir().join("liqmap_score_status.json")).and_then(|s| py::loads(&s).ok()) {
        for it in items.iter_mut() {
            if py::get(it, "id")? == Some(&json!("liqmap_fwd")) && !py::get(it, "done")?.is_some_and(py::truthy) {
                let what = py::get(it, "what")?.map(py::to_str).unwrap_or_default();
                let n = py::get(&st, "n_fwd")?.map_or("0".into(), py::to_str);
                let d = py::get(&st, "days_fwd")?.map_or("0".into(), py::to_str);
                if let Value::Object(o) = it {
                    o.insert("what".into(), json!(format!("{what} · 지금 누적 {n}시간({d}일)")));
                }
            }
        }
    }
    let mut keyed = Vec::new();
    for it in items {
        let k = py::get(&it, "date")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!(""));
        keyed.push((k, it));
    }
    let mut bad = None;
    keyed.sort_by(|a, b| match py::lt(&a.0, &b.0) {
        Ok(true) => std::cmp::Ordering::Less,
        Ok(false) => if py::lt(&b.0, &a.0).unwrap_or(false) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal },
        Err(e) => {
            bad = Some(e);
            std::cmp::Ordering::Equal
        }
    });
    if let Some(e) = bad {
        return Err(e);
    }
    recent.reverse();
    Ok(obj(vec![("verdicts", Value::Array(keyed.into_iter().map(|x| x.1).collect())), ("kinds", Value::Array(push_kinds())), ("recent", Value::Array(recent))]))
}

fn notify_center(cfg: &Config, _: &Req) -> Resp {
    or500(notify_center_payload(cfg))
}

// ───────────────────────── 뉴스 (server.py:729) ─────────────────────────

const NEWS_KEYS: [&str; 16] = ["source", "link", "title", "ts_ms", "seen_ms", "backfill", "sentiment", "p_bull", "p_bear", "p_neu", "asset", "impact", "incident", "relevant", "judged_ms", "model"];

fn news_payload(db: &Path, now: f64) -> R<V> {
    let since = ((now - 12.0 * 3600.0) * 1000.0).trunc() as i64;
    let q1 = "SELECT i.source, i.link, i.title, coalesce(i.pub_ms, i.first_seen_ms), i.first_seen_ms, i.backfill,
            j.sentiment, j.p_bull, j.p_bear, j.p_neu, j.asset, j.impact, j.incident, j.relevant, j.judged_ms, j.model
            FROM items i LEFT JOIN judgments j ON j.source = i.source AND j.link = i.link
            WHERE coalesce(i.pub_ms, i.first_seen_ms) >= ? ORDER BY coalesce(i.pub_ms, i.first_seen_ms) DESC LIMIT 1500";
    let q2 = "SELECT source, link, title, coalesce(pub_ms, first_seen_ms), first_seen_ms, backfill FROM items
                WHERE coalesce(pub_ms, first_seen_ms) >= ? ORDER BY coalesce(pub_ms, first_seen_ms) DESC LIMIT 1500";
    let rs = match rows(db, q1, &[Sql::Integer(since)]) {
        Ok(r) => r,
        Err(_) => match rows(db, q2, &[Sql::Integer(since)]) {
            Ok(r) => r.into_iter().map(|mut x| {
                x.resize(16, Value::Null);
                x
            }).collect(),
            Err(_) => return Ok(json!({"ok": false, "items": [], "window_h": 12, "now_ms": (now * 1000.0).trunc() as i64})),
        },
    };
    let mut items: Vec<Map<String, V>> = rs
        .into_iter()
        .map(|r| {
            let mut m: Map<String, V> = NEWS_KEYS.iter().zip(r).map(|(k, v)| (k.to_string(), v)).collect();
            let t = m["title"].as_str().filter(|s| !s.is_empty()).map(crate::text::unescape).unwrap_or_default();
            m.insert("title".into(), json!(t));
            m
        })
        .collect();
    // news_dedupe: 오래된 것부터 · 같은 글은 처음 본 것이 대표 · 판정은 판정 있는 것 중 먼저
    items.sort_by(|a, b| py::lt(&a["ts_ms"], &b["ts_ms"]).ok().map_or(std::cmp::Ordering::Equal, |l| if l { std::cmp::Ordering::Less } else if py::lt(&b["ts_ms"], &a["ts_ms"]).unwrap_or(false) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }));
    let mut out: Vec<(String, Map<String, V>)> = Vec::new();
    for it in items {
        let mut k = crate::text::news_key(it["title"].as_str().unwrap_or(""));
        if k.is_empty() {
            k = format!("{}|{}", py::to_str(&it["source"]), py::to_str(&it["link"]));
        }
        match out.iter_mut().find(|(x, _)| *x == k) {
            None => {
                let mut h = it.clone();
                h.insert("sources".into(), json!([it["source"]]));
                out.push((k, h));
            }
            Some((_, hit)) => {
                if let Value::Array(s) = hit.get_mut("sources").unwrap() {
                    if !s.iter().any(|x| py::eq(x, &it["source"])) {
                        s.push(it["source"].clone());
                    }
                }
                if !py::truthy(&hit["sentiment"]) && py::truthy(&it["sentiment"]) {
                    for fk in ["sentiment", "p_bull", "p_bear", "p_neu", "asset", "impact", "incident", "relevant", "judged_ms", "model"] {
                        hit.insert(fk.into(), it[fk].clone());
                    }
                }
            }
        }
    }
    let mut res: Vec<Map<String, V>> = out.into_iter().map(|x| x.1).collect();
    res.sort_by(|a, b| if py::lt(&b["ts_ms"], &a["ts_ms"]).unwrap_or(false) { std::cmp::Ordering::Less } else if py::lt(&a["ts_ms"], &b["ts_ms"]).unwrap_or(false) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
    let mut last = json!(0);
    for it in &res {
        if py::truthy(&it["sentiment"]) {
            let v = if py::truthy(&it["judged_ms"]) { it["judged_ms"].clone() } else { json!(0) };
            if py::gt(&v, &last)? {
                last = v;
            }
        }
    }
    Ok(obj(vec![("ok", json!(true)), ("items", Value::Array(res.into_iter().map(Value::Object).collect())), ("window_h", json!(12)),
        ("now_ms", json!((now * 1000.0).trunc() as i64)), ("model", json!("jevk5:4b")), ("last_judged_ms", if py::truthy(&last) { last } else { Value::Null })]))
}

fn news(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    or500(cfg.swr.get("news", 15.0, 300.0, Arc::new(move || news_payload(&hot(&c, "news_rss.sqlite"), (c.now)()))))
}

// ───────────────────────── 급변 칩 (server.py:675~726) ─────────────────────────

/// 원본 spikes_5m(678): 1분봉 [(초, 종가)] → 5분봉 로그 수익(bp) · 마지막 1시간 안에 끝난 |z|≥4 봉
fn spikes_5m(rs: &[(i64, f64)]) -> R<V> {
    if rs.is_empty() {
        return Ok(json!({"events": [], "n": 0}));
    }
    let mut last: Vec<(i64, f64)> = Vec::new(); // dict: 같은 봉은 자리 그대로 마지막 값
    for &(t, c) in rs {
        let b = t.div_euclid(300) * 300;
        match last.iter_mut().find(|x| x.0 == b) {
            Some(x) => x.1 = c,
            None => last.push((b, c)),
        }
    }
    let max_t = rs[rs.len() - 1].0;
    let mut bs: Vec<(i64, f64)> = last.into_iter().filter(|x| x.0 + 300 <= max_t).collect();
    bs.sort_by_key(|x| x.0);
    let rets: Vec<(i64, f64)> = bs.windows(2).filter(|w| w[1].0 - w[0].0 == 300 && w[0].1 > 0.0 && w[1].1 > 0.0).map(|w| (w[1].0, (w[1].1 / w[0].1).ln() * 1e4)).collect();
    let mut events = Vec::new();
    for k in (0..rets.len()).rev() {
        let (b, r) = rets[k];
        if b + 300 < max_t - 3600 {
            break;
        }
        let base: Vec<f64> = rets[k.saturating_sub(5400)..k].iter().map(|x| x.1).collect();
        let sd = if base.len() >= 288 { py::stdev(&base)? } else { 0.0 };
        if sd > 0.0 && (r / sd).abs() >= 4.0 {
            events.push(json!([b, f(py::round_nd(r / sd, 2)), f(py::round_nd((r / 1e4).exp_m1() * 100.0, 3))]));
        }
    }
    events.reverse();
    Ok(json!({"events": events, "n": rets.len()}))
}

/// 뉴욕 시각(미국 서머타임 2007~: 3월 둘째 일요일 02:00 ~ 11월 첫 일요일 02:00) → (요일 월=0, 시)
fn ny(ts: i64) -> (i64, i64) {
    let year = civil(ts.div_euclid(86400)).0;
    let sunday = |m: i64, nth: i64| {
        let d1 = crate::api::days_from_civil(year, m, 1);
        let wd = (d1 + 3).rem_euclid(7); // 1970-01-01 = 목(3)
        d1 + (6 - wd).rem_euclid(7) + 7 * (nth - 1)
    };
    let start = sunday(3, 2) * 86400 + 7 * 3600; // 02:00 EST = 07:00 UTC
    let end = sunday(11, 1) * 86400 + 6 * 3600; // 02:00 EDT = 06:00 UTC
    let off = if ts >= start && ts < end { -4 * 3600 } else { -5 * 3600 };
    let l = ts + off;
    ((l.div_euclid(86400) + 3).rem_euclid(7), l.rem_euclid(86400) / 3600)
}

fn cme_open(ts: i64) -> bool {
    let (wd, h) = ny(ts);
    !(h == 17 || wd == 5 || (wd == 6 && h < 18) || (wd == 4 && h >= 17))
}

fn macro_spikes_payload(cfg: &Config) -> R<V> {
    let mut out = Map::new();
    for (nm, db, t, k, key) in [("ZF", "treasury_futures.sqlite", "treasury_fut_1m", "symbol", "ZF"), ("ZN", "treasury_futures.sqlite", "treasury_fut_1m", "symbol", "ZN"),
                                ("NQ", "hl_xyz.sqlite", "hl_xyz_1m", "coin", "xyz:XYZ100")] {
        let q = format!("SELECT ts_sec, close FROM {t} WHERE {k} = ? AND close IS NOT NULL
           AND ts_sec >= (SELECT max(ts_sec) FROM {t} WHERE {k} = ?) - 8 * 86400 ORDER BY ts_sec");
        let Ok(rs) = rows(&hot(cfg, db), &q, &[Sql::Text(key.into()), Sql::Text(key.into())]) else { continue };
        let mut v = Vec::with_capacity(rs.len());
        for r in rs {
            let ts = int_v(&r[0])?;
            if nm != "NQ" || cme_open(ts) {
                v.push((ts, py::to_float(&r[1])?));
            }
        }
        out.insert(nm.into(), spikes_5m(&v)?);
    }
    Ok(obj(vec![("ok", json!(!out.is_empty())), ("z", f(4.0)), ("symbols", Value::Object(out)), ("now_sec", json!((cfg.now)().trunc() as i64))]))
}

fn macro_spikes(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    or500(cfg.swr.get("macro_spikes", 30.0, 120.0, Arc::new(move || macro_spikes_payload(&c))))
}

// ───────────────────────── 체결 버블 (server.py:1071 · 1107) ─────────────────────────

/// scripts/live_trade_tape_collector_20260916.py BUCKETS · SIZE_BANDS_USD(고래 하한 · 없으면 ETH 10만)
pub const TAPE_BUCKETS: [(&str, f64); 10] = [("ethusdt", 0.1), ("btcusdt", 1.0), ("solusdt", 0.01), ("xrpusdt", 0.0001), ("hypeusdt", 0.001),
    ("zecusdt", 0.05), ("sndkusdt", 0.05), ("soxlusdt", 0.01), ("skhynixusdt", 0.1), ("muusdt", 0.05)];
pub const WHALE_USD: [(&str, f64); 8] = [("solusdt", 55_000.0), ("xrpusdt", 23_000.0), ("btcusdt", 160_000.0), ("zecusdt", 6_000.0), ("sndkusdt", 6_700.0),
    ("soxlusdt", 36_000.0), ("skhynixusdt", 4_300.0), ("muusdt", 5_400.0)];

/// sqlite3 예외 이름(원본 503 본문 error 칸)
fn sql_err_name(e: &rusqlite::Error) -> &'static str {
    match e {
        rusqlite::Error::SqliteFailure(f, _) if f.code == rusqlite::ErrorCode::NotADatabase => "DatabaseError",
        _ => "OperationalError",
    }
}

fn bubble_payload(cfg: &Config, symbol: &str, sec: i64, since: Option<i64>) -> Result<V, &'static str> {
    let end = (cfg.now)().trunc() as i64;
    let mut lo = end - sec - 5;
    if let Some(s) = since {
        lo = lo.max(s + 1);
    }
    let q = "SELECT ts_sec, price_bin, buy_qty, sell_qty, coalesce(whale_buy_qty, 0) + coalesce(whale_sell_qty, 0)
           FROM trade_tape_1s WHERE symbol = ? AND ts_sec >= ?";
    let base = symbol[..symbol.len() - 4].to_uppercase();
    let srcs = [("binance", hot(cfg, "binance_tape.sqlite"), symbol.to_string()), ("okx", hot(cfg, "okx_tape.sqlite"), format!("{base}-USDT-SWAP")),
                ("bybit", hot(cfg, "bybit_tape.sqlite"), format!("{base}USDT"))];
    let mut acc: std::collections::BTreeMap<(i64, i64), [f64; 3]> = Default::default();
    let mut venues = Vec::new();
    for (name, path, key) in srcs {
        if name != "binance" && !path.exists() {
            continue;
        }
        let got = match rows(&path, q, &[Sql::Text(key), Sql::Integer(lo)]) {
            Ok(g) => g,
            Err(e) if name == "binance" => return Err(sql_err_name(&e)),
            Err(_) => continue,
        };
        venues.push(name);
        for r in got {
            let k = (int_v(&r[0]).map_err(|_| "TypeError")?, int_v(&r[1]).map_err(|_| "TypeError")?);
            let a = acc.entry(k).or_insert([0.0; 3]);
            for j in 0..3 {
                let v = &r[2 + j];
                a[j] += if py::truthy(v) { py::to_float(v).map_err(|_| "ValueError")? } else { 0.0 };
            }
        }
    }
    let bucket = TAPE_BUCKETS.iter().find(|x| x.0 == symbol).unwrap().1;
    let whale = WHALE_USD.iter().find(|x| x.0 == symbol).map_or(100_000.0, |x| x.1);
    let rows: Vec<V> = acc.into_iter().map(|((t, b), v)| json!([t, b, f(py::round_nd(v[0], 6)), f(py::round_nd(v[1], 6)), f(py::round_nd(v[2], 6))])).collect();
    Ok(obj(vec![("symbol", json!(symbol)), ("bucket", f(bucket)), ("lo", json!(lo)), ("end", json!(end)), ("venues", json!(venues)), ("whale_usd", f(whale)), ("rows", Value::Array(rows))]))
}

/// GET /api/bubble-tape -- ?asset(소문자로)·?sec(10~1800 · 기본 1800)·?since · 바이낸스 테이프를 못 읽으면 503
fn bubble_tape(cfg: &Config, req: &Req) -> Resp {
    let asset = req.q("asset").filter(|a| !a.is_empty()).unwrap_or("eth").to_lowercase();
    let symbol = format!("{asset}usdt");
    let int = |s: &str| crate::api::py_int_str(s).filter(|v| v.unsigned_abs() < 1 << 62).map(|v| v as i64);
    let Some(sec) = int(req.q("sec").unwrap_or("1800")) else { return http_exc(400, Some("bad_sec_or_since"), &[]) };
    let since = match req.q("since").filter(|s| !s.is_empty()) {
        Some(s) => match int(s) {
            Some(v) => Some(v),
            None => return http_exc(400, Some("bad_sec_or_since"), &[]),
        },
        None => None,
    };
    if !TAPE_BUCKETS.iter().any(|x| x.0 == symbol) || !(10..=1800).contains(&sec) {
        return http_exc(400, Some("unsupported_asset_or_window"), &[]);
    }
    // 원본: since 가 있으면 캐시 없이 · 없으면 1초 캐시(_bubble_cache · 읽기 실패는 안 남김)
    let got = match since {
        Some(_) => bubble_payload(cfg, &symbol, sec, since),
        None => {
            let (c, s2) = (cfg.clone(), symbol.clone());
            cfg.swr.get(&format!("bubble:{symbol}|{sec}"), 1.0, 0.0, Arc::new(move || bubble_payload(&c, &s2, sec, None).map_err(|e| PyErr(e.into()))))
                .map_err(|PyErr(m)| if m == "DatabaseError" { "DatabaseError" } else if m == "TypeError" { "TypeError" } else if m == "ValueError" { "ValueError" } else { "OperationalError" })
        }
    };
    match got {
        Ok(v) => Resp::json(&v),
        Err(name) => {
            let mut r = Resp::json(&json!({"error": name, "rows": []}));
            r.status = 503;
            r
        }
    }
}

// ───────────────────────── 적중 기록 (dashboard/pred_record.py) ─────────────────────────

const BAR: i64 = 300_000;
const GAP_MS: i64 = 30 * 60_000;

fn pr_tail(p: &Path) -> Vec<V> {
    if !p.exists() {
        return Vec::new();
    }
    tail_lines(p, 8_000_000).unwrap_or_default().iter().filter_map(|l| py::loads(l).ok()).collect()
}

fn kt(r: &V) -> R<String> {
    Ok(format!("{}|{}", py::hkey(py::get(r, "k")?.unwrap_or(&Value::Null)), py::hkey(py::get(r, "t")?.unwrap_or(&Value::Null))))
}

fn pr_rows(name: &str, d: &Path, bf: &Path) -> R<Vec<V>> {
    let mut rows = pr_tail(&d.join(format!("{name}.jsonl")));
    let bfp = bf.join(format!("{name}.jsonl"));
    if bfp.exists() {
        let have: std::collections::HashSet<String> = rows.iter().map(kt).collect::<R<_>>()?;
        for r in pr_tail(&bfp) {
            if !have.contains(&kt(&r)?) {
                rows.push(r);
            }
        }
    }
    Ok(rows)
}

fn s80(p: Option<&V>) -> R<V> {
    match p {
        None | Some(Value::Null) => Ok(Value::Null),
        Some(p) => Ok(json!(if py::ge(p, &json!(80))? { 1 } else if !py::gt(p, &json!(20))? { -1 } else { 0 })),
    }
}

fn mmdd(ms: &V) -> R<String> {
    let x = py::num(ms).ok_or_else(|| PyErr("TypeError: ms".into()))?;
    let (_, m, d) = civil((x / 86_400_000.0).floor() as i64);
    Ok(format!("{m:02}-{d:02}"))
}

fn pr_score(rows: &[V], now_ms: i64) -> R<V> {
    let mut px: std::collections::HashMap<String, V> = Default::default();
    for r in rows {
        px.insert(py::hkey(item(r, "t")?), item(r, "px")?.clone());
    }
    let mut sorted: Vec<&V> = rows.iter().collect();
    sorted.sort_by(|a, b| if py::lt(&a["t"], &b["t"]).unwrap_or(false) { std::cmp::Ordering::Less } else if py::lt(&b["t"], &a["t"]).unwrap_or(false) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
    let mut out = Map::new();
    for (name, ks, h) in [("15m", "s", 3i64), ("1h", "s", 12), ("4h", "4", 48)] {
        let (mut n, mut hit, mut pend) = (0i64, 0i64, 0i64);
        let mut day: std::collections::BTreeMap<String, (i64, i64)> = Default::default();
        let mut seq: Vec<V> = Vec::new();
        for r in &sorted {
            if py::get(r, "k")? != Some(&json!(ks)) {
                continue;
            }
            let s = match name {
                "15m" => s80(py::get(r, "a")?)?,
                "1h" => s80(py::get(r, "b")?)?,
                _ => py::get(r, "s")?.cloned().unwrap_or(Value::Null),
            };
            if !py::truthy(&s) {
                continue;
            }
            let tend = py::add(item(r, "t")?, &json!(h * BAR))?;
            let Some(end) = px.get(&py::hkey(&tend)) else {
                if py::lt(&py::sub(&json!(now_ms), &tend)?, &json!(GAP_MS))? {
                    pend += 1;
                }
                continue;
            };
            let ret = py::div(end, item(r, "px")?)? - 1.0;
            if ret == 0.0 {
                continue;
            }
            let good = i64::from(py::gt(&py::mul(&f(ret), &s)?, &json!(0))?);
            let c = day.entry(mmdd(&r["t"])?).or_insert((0, 0));
            c.0 += 1;
            c.1 += good;
            seq.push(json!([r["t"], good]));
            n += 1;
            hit += good;
        }
        if n != 0 || pend != 0 {
            let daily: Vec<V> = day.into_iter().map(|(d, (a, b))| json!([d, a, b])).collect();
            let dk = daily.len().saturating_sub(14);
            let sk = seq.len().saturating_sub(24);
            out.insert(name.into(), obj(vec![("n", json!(n)), ("hit", json!(hit)), ("pending", json!(pend)), ("daily", Value::Array(daily[dk..].to_vec())), ("recent", Value::Array(seq[sk..].to_vec()))]));
        }
    }
    Ok(Value::Object(out))
}

fn py_int(v: &V) -> R<i64> {
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        _ => int_v(v),
    }
}

fn pr_band(coin: &str, scored: &Path) -> R<Option<V>> {
    let mut rs = Vec::new();
    for r in pr_tail(scored) {
        if py::to_str(py::get(&r, "coin")?.unwrap_or(&Value::Null)).to_uppercase() == coin.to_uppercase() {
            rs.push(r);
        }
    }
    if rs.is_empty() {
        return Ok(None);
    }
    let n = rs.len() as i64;
    let (mut ok, mut ys) = (0i64, 0i64);
    let mut day: std::collections::BTreeMap<String, (i64, i64)> = Default::default();
    for r in &rs {
        ok += py_int(item(r, "correct")?)?;
        ys += py_int(item(r, "y")?)?;
    }
    for r in &rs {
        let ts: Vec<char> = py::to_str(item(r, "ts")?).chars().collect();
        let c = day.entry(ts[5.min(ts.len())..10.min(ts.len())].iter().collect()).or_insert((0, 0));
        c.0 += 1;
        c.1 += py_int(item(r, "correct")?)?;
    }
    let ty = ys as f64 / n as f64;
    let mut rec: Vec<(String, i64)> = rs.iter().map(|r| Ok((py::to_str(item(r, "ts")?), py_int(item(r, "correct")?)?))).collect::<R<_>>()?;
    rec.sort_by(|a, b| a.0.cmp(&b.0));
    let daily: Vec<V> = day.into_iter().map(|(d, (a, b))| json!([d, a, b])).collect();
    let dk = daily.len().saturating_sub(14);
    let rk = rec.len().saturating_sub(24);
    Ok(Some(obj(vec![("n", json!(n)), ("hit", json!(ok)), ("base", f(if 1.0 - ty > ty { 1.0 - ty } else { ty })), ("daily", Value::Array(daily[dk..].to_vec())),
        ("recent", Value::Array(rec[rk..].iter().map(|(t, c)| json!([t, c])).collect()))])))
}

const X_COINS: [&str; 9] = ["ETH", "BTC", "SOL", "XRP", "ADA", "AVAX", "BNB", "DOGE", "LINK"];

/// pandas rank(average) 의 순위
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*a].partial_cmp(&v[*b]).unwrap());
    let mut out = vec![0.0; v.len()];
    let mut k = 0;
    while k < idx.len() {
        let mut e = k;
        while e + 1 < idx.len() && v[idx[e + 1]] == v[idx[k]] {
            e += 1;
        }
        let r = (k + e) as f64 / 2.0 + 1.0;
        for &j in &idx[k..=e] {
            out[j] = r;
        }
        k = e + 1;
    }
    out
}


/// np.corrcoef(a, b)[0, 1] -- cov 는 평균을 빼고 내적 · 1/(n−1) 곱 · 대각 제곱근으로 나눔 · [-1, 1] 자름.
/// ponytail: 내적은 차례 합(numpy 는 BLAS dgemm) -- 9개짜리라 BLAS 묶음 순서가 끝자리를 바꿀 수 있다(서버 OpenBLAS 미대조).
fn corrcoef(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (rtcollect::ob5m::np_sum(a) / n, rtcollect::ob5m::np_sum(b) / n); // numpy add.reduce 규칙(공유 함수)
    let xa: Vec<f64> = a.iter().map(|x| x - ma).collect();
    let xb: Vec<f64> = b.iter().map(|x| x - mb).collect();
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).fold(0.0, |s, (x, y)| s + x * y);
    let fact = 1.0 / (n - 1.0);
    let (caa, cbb, cab) = (dot(&xa, &xa) * fact, dot(&xb, &xb) * fact, dot(&xa, &xb) * fact);
    let c = cab / caa.sqrt() / cbb.sqrt();
    if c.is_nan() { c } else { c.clamp(-1.0, 1.0) }
}

fn pr_xsec(d: &Path, bf: &Path) -> R<V> {
    let mut rows: Vec<(V, V)> = Vec::new(); // {t: px} -- 같은 t 는 자리 그대로 값만
    for r in pr_rows("xsec", d, bf)? {
        if py::get(&r, "k")? == Some(&json!("x")) {
            let (t, p) = (item(&r, "t")?.clone(), item(&r, "px")?.clone());
            match rows.iter_mut().find(|x| py::eq(&x.0, &t)) {
                Some(x) => x.1 = p,
                None => rows.push((t, p)),
            }
        }
    }
    let find = |t: &V| rows.iter().find(|x| py::eq(&x.0, t)).map(|x| x.1.clone());
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| if py::lt(&a.0, &b.0).unwrap_or(false) { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater });
    let step = json!(4 * 3600 * 1000);
    let mut ics = Vec::new();
    for (t, now) in &sorted {
        let (a, b) = (find(&py::sub(t, &step)?), find(&py::add(t, &step)?));
        let (Some(a), Some(b)) = (a.filter(py::truthy), b.filter(py::truthy)) else { continue };
        let has = |m: &V, c: &str| matches!(m, Value::Object(o) if o.contains_key(c));
        let cs: Vec<&str> = X_COINS.iter().copied().filter(|c| has(now, c) && has(&a, c) && has(&b, c)).collect();
        let mut past = Vec::new();
        let mut nxt = Vec::new();
        for c in &cs {
            past.push(py::div(&now[*c], &a[*c])? - 1.0);
            nxt.push(py::div(&b[*c], &now[*c])? - 1.0);
        }
        let ic = if cs.len() < 2 { f64::NAN } else { corrcoef(&ranks(&past), &ranks(&nxt)) };
        ics.push(ic);
    }
    let ics: Vec<f64> = ics.into_iter().filter(|x| !x.is_nan()).collect();
    let ic = if ics.is_empty() { Value::Null } else { f(ics.iter().fold(0.0, |s, x| s + x) / ics.len() as f64) };
    // latest = 서버 프로세스 메모리(_xs_latest · pred_record_loop 가 채움) -- rtserve 에는 없다: 늘 None(문서 §의도한 차이)
    Ok(obj(vec![("n", json!(ics.len())), ("ic", ic), ("latest", Value::Null)]))
}

fn pred_record_payload(cfg: &Config) -> R<V> {
    let now_ms = ((cfg.now)() * 1000.0).trunc() as i64;
    let (d, bf, band) = (cfg.live_dir().join("pred_record"), cfg.live_dir().join("pred_record_backfill"), cfg.live_dir().join("regime_band_forward_scored.jsonl"));
    let mut names = std::collections::BTreeSet::new();
    for g in [&d, &bf] {
        if let Ok(rd) = fs::read_dir(g) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
                    if let Some(s) = p.file_stem().and_then(|x| x.to_str()) {
                        if s != "xsec" {
                            names.insert(s.to_string());
                        }
                    }
                }
            }
        }
    }
    let mut coins = Map::new();
    let cut = json!(now_ms - 15 * 86_400_000);
    for name in &names {
        let mut rs = Vec::new();
        for r in pr_rows(name, &d, &bf)? {
            if py::ge(item(&r, "t")?, &cut)? {
                rs.push(r);
            }
        }
        coins.insert(name.to_lowercase(), pr_score(&rs, now_ms)?);
    }
    // 원본은 set(coins) | {eth,btc,sol,xrp} 를 집합 순서(해시 무작위)로 돈다 -- 기록 없이 띠만 있는 코인이 둘 이상이면 원본 열쇠 순서가 실행마다 다르다.
    //   여기서는 기록 있는 코인(이름 순) 다음 eth·btc·sol·xrp 순.
    let mut extra: Vec<String> = coins.keys().cloned().collect();
    for c in ["eth", "btc", "sol", "xrp"] {
        if !extra.iter().any(|x| x == c) {
            extra.push(c.into());
        }
    }
    for c in extra {
        if let Some(b) = pr_band(&c, &band)? {
            let e = coins.entry(c).or_insert_with(|| json!({}));
            if let Value::Object(o) = e {
                o.insert("band".into(), b);
            }
        }
    }
    let xs = pr_xsec(&d, &bf)?;
    Ok(obj(vec![("ok", json!(!coins.is_empty())), ("asof_ms", json!(now_ms)), ("coins", Value::Object(coins)), ("xsec", xs)]))
}

/// GET /api/pred-record -- 예외 = {"ok": false, "reason": 200자}
fn pred_record(cfg: &Config, _: &Req) -> Resp {
    let c = cfg.clone();
    match cfg.swr.get("pred_record", 30.0, 600.0, Arc::new(move || pred_record_payload(&c))) {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) => guarded(Err(PyErr(m.chars().take(200).collect()))),
    }
}

// ───────────────────────── 쓰기: 메모·오답 (dashboard/acct_notes.py) ─────────────────────────

/// aiohttp request.json(): 본문 utf-8 → json.loads
fn body_json(req: &Req) -> R<V> {
    let s = String::from_utf8(req.body.clone()).map_err(|_| PyErr("UnicodeDecodeError: utf-8".into()))?;
    py::loads(&s)
}

/// 원자적 쓰기(원본: tmp 에 쓰고 os.replace)
fn write_atomic(path: &Path, tmp_suffix: &str, text: &str) -> std::io::Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    let tmp = path.with_extension(tmp_suffix);
    fs::write(&tmp, text.as_bytes())?;
    fs::rename(&tmp, path)
}

fn is_date(v: Option<&V>) -> bool {
    // _DATE = r"\d{4}-\d{2}-\d{2}" fullmatch (ASCII 숫자만 -- 유니코드 숫자 날짜는 화면이 보내지 않는다)
    let Some(Value::String(s)) = v else { return false };
    let b = s.as_bytes();
    b.len() == 10 && b[4] == b'-' && b[7] == b'-' && b.iter().enumerate().all(|(k, c)| k == 4 || k == 7 || c.is_ascii_digit())
}

fn py_strip_empty(s: &str) -> bool {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)).is_empty()
}

/// POST /api/acct-notes (server.py:2869 → acct_notes.save) -- 잘못된 입력 = 400 + ValueError 글자
fn acct_notes_save(cfg: &Config, req: &Req) -> Resp {
    let _g = cfg.write_lock.lock().unwrap();
    let path = cfg.live_dir().join("acct_notes.json");
    let res = (|| -> R<V> {
        let body = body_json(req)?;
        let Value::Object(b) = &body else { return err("AttributeError", format!("'{}' object has no attribute 'get'", py::type_name(&body))) };
        if !is_date(b.get("date")) {
            return err("ValueError", "date must be YYYY-MM-DD");
        }
        let Some(Value::String(text)) = b.get("text") else { return err("ValueError", "text must be a string up to 4000 chars") };
        if text.chars().count() > 4000 {
            return err("ValueError", "text must be a string up to 4000 chars");
        }
        let date = b["date"].as_str().unwrap().to_string();
        let mut notes = match read_text(&path).and_then(|s| py::loads(&s).ok()) {
            Some(Value::Object(o)) => o,
            _ => Map::new(),
        };
        let at = ((cfg.now)() * 1000.0).trunc() as i64;
        let keep = !py_strip_empty(text);
        if keep {
            notes.insert(date, json!({"text": text, "at": at}));
        } else {
            notes.shift_remove(&date);
        }
        write_atomic(&path, "tmp", &py::dumps(&Value::Object(notes), (", ", ": "), false, None)).map_err(|e| PyErr(format!("OSError: {e}")))?;
        Ok(json!({"ok": true, "at": if keep { json!(at) } else { Value::Null }}))
    })();
    write_result(res)
}

/// 원본 핸들러: (ValueError, AttributeError[, TypeError]) → 400 text=str(exc) · 그 밖 = 500
fn write_result(res: R<V>) -> Resp {
    match res {
        Ok(v) => Resp::json(&v),
        Err(PyErr(m)) => match m.split_once(": ") {
            Some(("ValueError" | "AttributeError" | "TypeError" | "JSONDecodeError" | "UnicodeDecodeError", msg)) => exc_text(400, msg),
            _ => server_error(),
        },
    }
}

/// POST /api/acct-wrong (server.py:2881 → wrong_add · wrong_del)
fn acct_wrong_save(cfg: &Config, req: &Req) -> Resp {
    let _g = cfg.write_lock.lock().unwrap();
    let path = cfg.live_dir().join("acct_wrong.json");
    let load = || match read_text(&path).and_then(|s| py::loads(&s).ok()) {
        Some(Value::Array(a)) => a,
        _ => Vec::new(),
    };
    let write = |rows: &[V]| write_atomic(&path, "tmp", &py::dumps(&Value::Array(rows.to_vec()), (", ", ": "), false, None)).map_err(|e| PyErr(format!("OSError: {e}")));
    let res = (|| -> R<V> {
        let body = body_json(req)?;
        let Value::Object(b) = &body else { return err("AttributeError", format!("'{}' object has no attribute 'get'", py::type_name(&body))) };
        if let Some(rid) = b.get("del") {
            if !py::is_int(rid) {
                return err("ValueError", "id must be int");
            }
            let rows = load();
            let mut keep = Vec::new();
            for r in &rows {
                if !py::get(r, "id")?.is_some_and(|x| py::eq(x, rid)) {
                    keep.push(r.clone());
                }
            }
            if keep.len() != rows.len() {
                write(&keep)?;
            }
            return Ok(json!({"ok": true}));
        }
        if !is_date(b.get("date")) {
            return err("ValueError", "date must be YYYY-MM-DD");
        }
        let text = match b.get("text") {
            Some(Value::String(t)) if !py_strip_empty(t) && t.chars().count() <= 300 => t.clone(),
            _ => return err("ValueError", "text must be 1..300 chars"),
        };
        let tag = b.get("tag").cloned().unwrap_or(json!(""));
        if !["", "진입 성급", "원칙 위반", "가짜 돌파", "사이징", "기타"].iter().any(|t| tag == *t) {
            return err("ValueError", "unknown tag");
        }
        let mut rows = load();
        let at = ((cfg.now)() * 1000.0).trunc() as i64;
        let mut ids = Vec::new();
        for r in &rows {
            ids.push(py::get(r, "id")?.cloned().unwrap_or(json!(0)));
        }
        ids.push(json!(at - 1));
        let mx = ids.iter().try_fold(ids[0].clone(), |m, x| if py::gt(x, &m)? { Ok::<_, PyErr>(x.clone()) } else { Ok(m) })?;
        let id = py::add(&mx, &json!(1))?;
        let row = obj(vec![("id", id), ("date", b["date"].clone()), ("text", json!(py_split_join(&text))), ("tag", tag), ("at", json!(at))]);
        rows.push(row.clone());
        let k = rows.len().saturating_sub(2000);
        write(&rows[k..])?;
        Ok(row)
    })();
    write_result(res)
}

/// 파이썬 `" ".join(s.split())`
fn py_split_join(s: &str) -> String {
    s.split(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)).filter(|x| !x.is_empty()).collect::<Vec<_>>().join(" ")
}

// ───────────────────────── 알림 구독 (scripts/push_webpush_20260904.py 226~273) ─────────────────────────

fn subs_path(cfg: &Config) -> PathBuf {
    cfg.live_dir().join("push_subscriptions.json")
}

/// load_subscriptions: 없거나 깨졌으면 {} · 사전 아니면 {}
fn load_subs(cfg: &Config) -> R<Map<String, V>> {
    let Ok(b) = fs::read(subs_path(cfg)) else { return Ok(Map::new()) };
    let s = String::from_utf8(b).map_err(|_| PyErr("UnicodeDecodeError: utf-8".into()))?;
    Ok(match py::loads(&s) {
        Ok(Value::Object(o)) => o,
        _ => Map::new(),
    })
}

/// save_subscriptions: json.dump(indent=2, ensure_ascii=False) → .json.tmp → replace
fn save_subs(cfg: &Config, subs: Map<String, V>) -> R<()> {
    write_atomic(&subs_path(cfg), "json.tmp", &py::dumps(&Value::Object(subs), (", ", ": "), false, Some(2))).map_err(|e| PyErr(format!("OSError: {e}")))
}

fn sub_id(endpoint: &V) -> R<String> {
    let Value::String(e) = endpoint else { return err("AttributeError", format!("'{}' object has no attribute 'encode'", py::type_name(endpoint))) };
    Ok(crate::hex(&Sha256::digest(e.as_bytes()))[..16].to_string())
}

/// GET /api/push/config (server.py:4856) -- env 의 공개키 · 비밀키는 «있는지»만
fn push_config(cfg: &Config, _: &Req) -> Resp {
    or500(load_subs(cfg).map(|s| json!({"enabled": !cfg.vapid_public.is_empty() && cfg.vapid_private_set, "vapid_public_key": cfg.vapid_public, "subscriber_count": s.len()})))
}

/// POST /api/push/subscribe (server.py:4868)
fn push_subscribe(cfg: &Config, req: &Req) -> Resp {
    let _g = cfg.write_lock.lock().unwrap();
    let Ok(body) = body_json(req) else { return exc_text(400, "invalid json") };
    let res = (|| -> R<Result<V, &'static str>> {
        let sub = py::get(&body, "subscription")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!({}));
        let keys = py::get(&sub, "keys")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!({}));
        if !py::get(&sub, "endpoint")?.is_some_and(py::truthy) || !py::get(&keys, "p256dh")?.is_some_and(py::truthy) {
            return Ok(Err("subscription must carry endpoint and keys.p256dh"));
        }
        let label: String = py::to_str(py::get(&body, "label")?.unwrap_or(&json!(""))).chars().take(80).collect();
        let mut subs = load_subs(cfg)?;
        let sid = sub_id(&sub["endpoint"])?;
        let old = subs.get(&sid).cloned().unwrap_or(json!({}));
        let old_label = py::get(&old, "label")?.cloned().unwrap_or(json!(""));
        let old_at = py::get(&old, "subscribed_utc")?.filter(|v| py::truthy(v)).cloned();
        let now = (cfg.now)().trunc() as i64;
        let (y, m, d) = civil(now.div_euclid(86400));
        let s = now.rem_euclid(86400);
        let stamp = format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", s / 3600, s / 60 % 60, s % 60);
        subs.insert(sid.clone(), obj(vec![("endpoint", sub["endpoint"].clone()), ("keys", keys), ("label", if label.is_empty() { old_label } else { json!(label) }),
            ("subscribed_utc", old_at.unwrap_or(json!(stamp)))]));
        save_subs(cfg, subs)?;
        Ok(Ok(json!({"ok": true, "id": sid})))
    })();
    match res {
        Ok(Ok(v)) => Resp::json(&v),
        Ok(Err(msg)) => exc_text(400, msg),
        Err(_) => server_error(),
    }
}

/// POST /api/push/unsubscribe (server.py:4879)
fn push_unsubscribe(cfg: &Config, req: &Req) -> Resp {
    let _g = cfg.write_lock.lock().unwrap();
    let Ok(body) = body_json(req) else { return exc_text(400, "invalid json") };
    let res = (|| -> R<Result<V, &'static str>> {
        let sub = py::get(&body, "subscription")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!({}));
        let sid = match py::get(&body, "id")?.filter(|v| py::truthy(v)) {
            Some(v) => v.clone(),
            None if py::get(&sub, "endpoint")?.is_some_and(py::truthy) => json!(sub_id(&sub["endpoint"])?),
            None => Value::Null,
        };
        if !py::truthy(&sid) {
            return Ok(Err("id or subscription.endpoint required"));
        }
        if matches!(sid, Value::Array(_) | Value::Object(_)) {
            return err("TypeError", "unhashable type");
        }
        let mut subs = load_subs(cfg)?;
        let ok = match &sid {
            Value::String(s) if subs.contains_key(s) => {
                subs.shift_remove(s);
                save_subs(cfg, subs)?;
                true
            }
            _ => false,
        };
        Ok(Ok(json!({"ok": ok})))
    })();
    match res {
        Ok(Ok(v)) => Resp::json(&v),
        Ok(Err(msg)) => exc_text(400, msg),
        Err(_) => server_error(),
    }
}

/// GET /api/push/devices (server.py:4891) -- endpoint 원문은 안 내고 꼬리 12자만
fn push_devices(cfg: &Config, _: &Req) -> Resp {
    or500((|| {
        let mut rows = Vec::new();
        for (sid, sub) in load_subs(cfg)? {
            let ep = py::to_str(py::get(&sub, "endpoint")?.unwrap_or(&json!("")));
            let tail: String = ep.chars().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
            rows.push(obj(vec![("id", json!(sid)), ("label", py::get(&sub, "label")?.filter(|v| py::truthy(v)).cloned().unwrap_or(json!(""))),
                ("subscribed_utc", py::get(&sub, "subscribed_utc")?.cloned().unwrap_or(Value::Null)), ("endpoint_tail", json!(tail))]));
        }
        let key = |r: &V| r["subscribed_utc"].clone();
        rows.sort_by(|a, b| {
            let (x, y) = (if py::truthy(&key(a)) { key(a) } else { json!("") }, if py::truthy(&key(b)) { key(b) } else { json!("") });
            if py::lt(&x, &y).unwrap_or(false) { std::cmp::Ordering::Less } else if py::lt(&y, &x).unwrap_or(false) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal }
        });
        Ok(json!({"devices": rows}))
    })())
}

// ───────────────────────── M0 client_timing (server.py:5097~5120) ─────────────────────────

/// 원본 CLIENT_TIMING_NUM(server.py:5097) 10칸 + rt_* 9칸(main 4ff90f88 · 화면 그림자 구독) + 🔴머지 대기 2칸(rt_draw_s · rt_back_n -- 원본 PR 전).
///   정답지(gen_http2_fixture.py)는 10칸 + 9칸을 옮겨 적었다 · 머지 대기 2칸은 대조 밖(Rust 만 시험: rt_timing_keys).
const CT_NUM: [&str; 21] = ["order_rtt_ms", "order_srv_ms", "order_net_ms", "fp_n", "fp_med_ms", "fp_max_ms", "longtask_n", "px_n", "px_med_ms", "px_max_ms",
    "rt_lat_n", "rt_lat_med_ms", "rt_lat_p99_ms", "rt_lead_n", "rt_lead_med_ms", "rt_cmp_n", "rt_bad_n", "rt_fail_n", "rt_reconn_n",
    "rt_draw_s", "rt_back_n"];

/// POST /api/client-timing -- 본문 1KB·아는 키·숫자만 · 0.2초에 한 줄 · 저장 없음(로그 한 줄)
fn client_timing(cfg: &Config, req: &Req) -> Resp {
    let raw = &req.body[..req.body.len().min(1025)];
    if raw.len() > 1024 {
        return json_plain(413, &json!({"ok": false, "error": "too_large"}));
    }
    let raw = raw.strip_prefix(b"\xef\xbb\xbf".as_slice()).unwrap_or(raw);
    let Ok(body) = std::str::from_utf8(raw).map_err(|_| ()).and_then(|s| py::loads(s).map_err(|_| ())) else {
        return json_plain(400, &json!({"ok": false, "error": "bad_json"}));
    };
    let bad = || json_plain(400, &json!({"ok": false, "error": "bad_body"}));
    let Value::Object(b) = &body else { return bad() };
    let nums: Vec<(&String, &V)> = b.iter().filter(|(k, _)| CT_NUM.contains(&k.as_str())).collect();
    let sym = py::to_str(b.get("sym").unwrap_or(&Value::Null));
    let kind_ok = b.get("kind").map_or(true, |k| k == "entry" || k == "exit");
    let extra = b.keys().any(|k| !CT_NUM.contains(&k.as_str()) && k != "sym" && k != "kind");
    let num_bad = nums.iter().any(|(_, v)| match v {
        Value::Number(n) => n.as_f64().is_none_or(|x| !x.is_finite() || x.abs() > 1e7),
        _ => true,
    });
    if nums.is_empty() || extra || !MARKET_ASSETS.contains(&sym.as_str()) || !kind_ok || num_bad {
        return bad();
    }
    {
        let mut at = cfg.timing_at.lock().unwrap();
        if at.is_some_and(|t| t.elapsed().as_secs_f64() < 0.2) {
            return json_plain(429, &json!({"ok": false, "error": "too_often"}));
        }
        *at = Some(std::time::Instant::now());
    }
    let mut ns = nums.clone();
    ns.sort_by(|a, b| a.0.cmp(b.0));
    let mut line = format!("client_timing sym={sym}");
    if let Some(k) = b.get("kind") {
        line.push_str(&format!(" kind={}", py::to_str(k)));
    }
    for (k, v) in ns {
        let r = py::round0_v(v).map(|x| x.to_string()).unwrap_or_default();
        line.push_str(&format!(" {k}={r}"));
    }
    (cfg.log)(&line);
    Resp::json(&json!({"ok": true}))
}

// ───────────────────────── 주문 요청 기록 (server.py:1350 order_request_log) ─────────────────────────

/// POST /api/manual-* 마다 data/live/order_requests.jsonl 에 한 줄(시각 KST·경로·쿼리·상태·error·ms·기기·IP). 못 쓰면 로그만.
pub fn order_request_log(cfg: &Config, req: &Req, r: &Resp, t0: f64) {
    let mut err_v = Value::Null;
    if !r.exc {
        if let Some(b) = r.body.as_ref().filter(|b| b.len() < 20000) {
            if let Ok(j) = std::str::from_utf8(b).map_err(|_| ()).and_then(|s| py::loads(s).map_err(|_| ())) {
                if let Value::Object(o) = &j {
                    if !py::truthy(o.get("ok").unwrap_or(&json!(true))) {
                        err_v = o.get("error").filter(|v| py::truthy(v)).cloned().unwrap_or(json!(""));
                    }
                }
            }
        }
    }
    let mut q = Map::new();
    for (k, v) in &req.query {
        // dict(request.query): MultiDict 의 열쇠마다 첫 값(md[k])
        q.entry(k.clone()).or_insert_with(|| json!(v));
    }
    let hdr = |n: &str| req.header(n).filter(|v| !v.is_empty());
    let ua: String = req.header("user-agent").unwrap_or("").chars().take(200).collect();
    let ip = hdr("cf-connecting-ip").or(hdr("x-forwarded-for")).unwrap_or(&req.peer).to_string();
    let ms = py::round0(1000.0 * ((cfg.now)() - t0)).unwrap_or(0);
    let row = obj(vec![("at", json!(kst_iso_ms(t0))), ("path", json!(req.path)), ("query", Value::Object(q)), ("status", json!(r.status)), ("error", err_v),
        ("ms", json!(ms)), ("ua", json!(ua)), ("ip", json!(ip))]);
    let line = py::dumps(&row, (", ", ": "), false, None) + "\n";
    let p = cfg.live_dir().join("order_requests.jsonl");
    let res = fs::OpenOptions::new().create(true).append(true).open(&p).and_then(|mut fh| fh.write_all(line.as_bytes()));
    if let Err(e) = res {
        (cfg.log)(&format!("order_request_log: 못 씀 {e}"));
    }
}

/// `datetime.fromtimestamp(t, ZoneInfo("Asia/Seoul")).isoformat(timespec="milliseconds")`
fn kst_iso_ms(t: f64) -> String {
    let us = now_us(t) + 9 * 3600 * 1_000_000;
    let secs = us.div_euclid(1_000_000);
    let ms = us.rem_euclid(1_000_000) / 1000;
    let (y, m, d) = civil(secs.div_euclid(86400));
    let s = secs.rem_euclid(86400);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{ms:03}+09:00", s / 3600, s / 60 % 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_and_ny() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(20_773), (2026, 11, 16));
        // 2026-11-01 은 일요일 · 06:00 UTC 에 EDT → EST
        assert_eq!(ny(1_793_512_800 - 1), (6, 1)); // 05:59:59 UTC = 01:59 EDT
        assert_eq!(ny(1_793_512_800), (6, 1)); // 06:00 UTC = 01:00 EST
        assert!(!cme_open(1_793_491_200 + 12 * 3600)); // 일 12:00 UTC = 07:00 EST 닫힘
        assert_eq!(kst_iso_ms(1_795_000_123.456789), "2026-11-18T20:08:43.456+09:00");
    }

    #[test]
    fn stdev_like_python() {
        assert_eq!(py::stdev(&[1.5, 2.5, 2.5, 2.75, 3.25, 4.75]).unwrap(), 1.0810874155219827);
    }
}
