// 주문·계좌 주소 14개 HTTP 정답지(tools/gen_route_fixture.py -- 원본 핸들러를 진짜 aiohttp 앱에 원본 등록·미들웨어로 붙여 보낸 요청의 바이트):
// 상태 · Content-Type · Cache-Control · Allow · **본문 글자**(server_ms 숫자만 0 으로) · 접수 기록 줄(path·query·status·error) · 상태 칸.
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::*;
use rtorder::route::{Front, ROUTES};
use rtorder::runner::DeskCell;
use rtorder::userstream::Hub;
use serde_json::Value;
use std::sync::atomic::Ordering;

fn ms0(s: &str) -> String {
    match s.find("\"server_ms\": ") {
        Some(i) => {
            let j = i + 13;
            let end = s[j..].find(|c: char| !c.is_ascii_digit()).map_or(s.len(), |k| j + k);
            format!("{}0{}", &s[..j], &s[end..])
        }
        None => s.to_string(),
    }
}

#[test]
fn every_route_matches_python_bytes_status_headers_and_log() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/route_fixture.json")).unwrap()).unwrap();
    let cases = fx["cases"].as_array().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    let mut bad: Vec<String> = vec![];
    for c in cases {
        let (name, inp) = (c["name"].as_str().unwrap(), &c["in"]);
        let (desk, _d) = desk_from(inp);
        let mut io = FakeDeskIo::new(inp, desk.w.dir.clone());
        let mut q = FakeQuotes::new(inp);
        let hub = Hub::default();
        if let Some(u) = inp.get("user_stream") {
            let mut us = hub.us.lock().unwrap();
            us.connected = u["connected"].as_bool().unwrap();
            us.error = u["error"].as_str().map(String::from);
            us.last_event_at = u["last_event_at"].as_f64();
            hub.seq.store(u["seq"].as_i64().unwrap(), Ordering::SeqCst);
        }
        let cell = DeskCell::new(desk);
        let front = Front::default();
        let _held = inp.get("locked").is_some_and(|v| v == true).then(|| front.order_lock.try_lock().unwrap());
        let pairs: Vec<(String, String)> = inp["pairs"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
        let (m, path) = (inp["method"].as_str().unwrap(), inp["path"].as_str().unwrap());
        let r = block_on(front.call(&cell, &hub, &mut q, &mut io, m, path, &pairs));
        seen.insert(path.to_string());
        let h = |k: &str| r.headers.iter().find(|x| x.0 == k).map(|x| x.1.clone());
        assert_eq!(r.status as u64, c["status"].as_u64().unwrap(), "{name}: 상태 {}", r.body);
        assert_eq!(Some(r.content_type), c["content_type"].as_str(), "{name}");
        assert_eq!(h("Cache-Control").as_deref(), c["cache_control"].as_str(), "{name}: Cache-Control");
        assert_eq!(h("Allow").as_deref(), c["allow"].as_str(), "{name}: Allow");
        if ms0(&r.body) != ms0(c["body"].as_str().unwrap()) {
            bad.push(format!("{name}: {}", glyph_diff(&r.body, c["body"].as_str().unwrap())));
            continue;
        }
        let want: Vec<Value> = c["log"].as_array().unwrap().clone();
        assert_eq!(r.log.clone().into_iter().collect::<Vec<_>>(), want, "{name}: 접수 기록");
        let d = block_on(cell.lock());
        assert_same(&serde_json::json!({"entry": d.w.entry, "side": d.w.side_exits, "waits": d.w.waits}), &c["slots"], &format!("{name}: 상태 칸"));
    }
    assert!(bad.is_empty(), "본문 글자 다름 {}건:\n{}", bad.len(), bad.join("\n"));
    assert_eq!(cases.len(), 159);
    for (_, p, _, _) in ROUTES {
        assert!(seen.contains(p), "정답지에 없는 주소 {p}");
    }
}

/// 글자가 다를 때 어디가 다른지(정수/실수 표기까지) -- 첫 곳만.
fn glyph_diff(a: &str, b: &str) -> String {
    fn walk(x: &Value, y: &Value, p: &str) -> Option<String> {
        match (x, y) {
            (Value::Object(m), Value::Object(n)) => {
                let (km, kn): (Vec<_>, Vec<_>) = (m.keys().collect(), n.keys().collect());
                if km != kn {
                    return Some(format!("{p}: 키 rust {km:?} / py {kn:?}"));
                }
                m.iter().find_map(|(k, v)| walk(v, &n[k], &format!("{p}.{k}")))
            }
            (Value::Array(m), Value::Array(n)) if m.len() == n.len() => m.iter().zip(n).enumerate().find_map(|(i, (v, w))| walk(v, w, &format!("{p}[{i}]"))),
            _ => (rtorder::pyfmt::json_dumps(x) != rtorder::pyfmt::json_dumps(y)).then(|| format!("{p}: rust {} / py {}", rtorder::pyfmt::json_dumps(x), rtorder::pyfmt::json_dumps(y))),
        }
    }
    match (serde_json::from_str::<Value>(a), serde_json::from_str::<Value>(b)) {
        (Ok(x), Ok(y)) => walk(&x, &y, "").unwrap_or_else(|| "값 같음(글자만 다름)".into()),
        _ => format!("rust {a:?} / py {b:?}"),
    }
}

#[test]
fn order_log_error_rule_matches_middleware() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/route_fixture.json")).unwrap()).unwrap();
    for r in fx["log_error"].as_array().unwrap() {
        let b = r[0].as_str().unwrap();
        assert_eq!(rtorder::route::log_error(b), r[1], "{}", &b[..b.len().min(40)]);
    }
}
