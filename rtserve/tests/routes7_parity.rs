//! 일곱째 묶음(2026-11-28): /api/chart-markers · /api/position-profile -- 정답지(tools/gen_routes7_fixture.py · 원본 모듈을 그대로 불러 씀)와
//! 본문 바이트·상태·보낸 요청 열이 같다 -- 127.0.0.1 가짜 REST 만(바깥 접속 없음). 시나리오마다 서버 하나.
//! posprof 는 사례마다 시계(cfg.now · swr 단조 시계 둘 다)를 그 사례의 now 로 옮기고, 허브 상태(풋프린트 봉·초 행)는 정답지의 흐름 상태로 채운다.
//!     cargo test --release -j 2 --offline --test routes7_parity
//! 양성 대조: MARKERS_FIXTURE / POSPROF_FIXTURE=<변이 사본으로 만든 정답지> 로 돌리면 실패해야 한다.

mod common;

use common::{request, start_with};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Table = Arc<Mutex<(HashMap<String, (u16, String)>, Vec<String>)>>;

fn fake(st: Table) -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut buf = vec![0u8; 65536];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut buf[n..]) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => n += k,
                }
            }
            let target = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let mut g = st.lock().unwrap();
            g.1.push(target.clone());
            let (code, body) = g.0.get(&target).cloned().unwrap_or((404, "[]".into()));
            drop(g);
            let _ = s.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn load(env: &str, rel: &str) -> Value {
    let p = std::env::var(env).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen").join(rel));
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e} -- rtserve/tools/gen_routes7_fixture.py 먼저", p.display()))).unwrap()
}

/// 정답지 hub → 허브 상태(코인마다 풋프린트 봉 · Supply1s 초 행 [초, 0…, 가격])
fn fill_hub(live: &rtserve::hub::Live, hub: &Value) {
    let mut g = live.syms.lock().unwrap();
    for (sym, h) in hub.as_object().unwrap() {
        let s = g.entry(sym.clone()).or_default();
        for b in h["bars"].as_array().unwrap() {
            let lv = b[1].as_array().unwrap().iter().map(|c| {
                let p = c[0].as_f64().unwrap();
                (p.to_bits(), (p, c[1].as_f64().unwrap(), c[2].as_f64().unwrap(), 0.0, 0.0, 0.0, 0.0))
            }).collect();
            s.fp_bars.insert(b[0].as_i64().unwrap(), (lv, None));
        }
        let secs: Vec<Value> = h["secs"].as_array().unwrap().iter().map(|r| json!([r[0], 0, 0, 0, 0, 0, 0, r[1]])).collect();
        let meta = json!({"connected": false, "tradeAge": null, "errors": 0});
        s.supply = Some(serde_json::from_value(json!({"now": 0, "retailMaxUsd": 0, "whaleMinUsd": 0, "seconds": secs, "partial": {"bn": null, "okx": null, "spot": null},
            "liq": [], "spot": [], "spotNow": 0, "spotMeta": meta, "okx": [], "okxNow": 0,
            "okxMeta": {"connected": false, "tradeAge": null, "oiAge": null, "inst": "", "errors": 0}, "okxLiq": [], "okxOi": [], "bybitLiq": []})).unwrap());
    }
}

/// 시나리오 하나: {now, rest, files?, hub?, cases:[{now?, path, status, body, sent}]} → 다른 사례 글
fn run(tag: &str, sc: &Value) -> (usize, Vec<String>) {
    let st: Table = Default::default();
    for (k, v) in sc["rest"].as_object().unwrap() {
        st.lock().unwrap().0.insert(k.clone(), (v[0].as_u64().unwrap() as u16, v[1].as_str().unwrap().to_string()));
    }
    let rest = fake(st.clone());
    let empty = serde_json::Map::new();
    let files = sc.get("files").and_then(Value::as_object).unwrap_or(&empty);
    let clock = Arc::new(AtomicU64::new(sc["now"].as_f64().unwrap().to_bits()));
    let (ck, hub) = (clock.clone(), sc.get("hub").cloned());
    let (addr, _) = start_with(files, 0, sc["now"].as_f64().unwrap(), "eth,btc,sol,xrp", tag, move |c| {
        c.rest = rest;
        let now: rtserve::Clock = Arc::new(move || f64::from_bits(ck.load(Ordering::SeqCst)));
        c.now = now.clone();
        Arc::get_mut(&mut c.swr).unwrap().mono = now;
        if let Some(h) = hub {
            fill_hub(&c.live, &h);
        }
    });
    let mut bad = Vec::new();
    let cases = sc["cases"].as_array().unwrap();
    for c in cases {
        if let Some(t) = c.get("now").and_then(Value::as_f64) {
            clock.store(t.to_bits(), Ordering::SeqCst);
        }
        st.lock().unwrap().1.clear();
        let path = c["path"].as_str().unwrap();
        let g = request(addr, "GET", path, &[]);
        let sent: Vec<Value> = st.lock().unwrap().1.iter().map(|s| Value::from(s.as_str())).collect();
        if g.status as u64 != c["status"].as_u64().unwrap() || g.body != c["body"].as_str().unwrap().as_bytes() || sent != *c["sent"].as_array().unwrap() {
            let (a, b) = (String::from_utf8_lossy(&g.body).to_string(), c["body"].as_str().unwrap().to_string());
            let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let ctx = |s: &str| s.get(at.saturating_sub(80)..(at + 80).min(s.len())).unwrap_or("").to_string();
            bad.push(format!("{tag} {path}: 상태 {}/{} · 보냄 {}/{} · 본문 {} 째 글자부터 다름\n  rust {}\n  정답 {}", g.status, c["status"], sent.len(), c["sent"].as_array().unwrap().len(), at, ctx(&a), ctx(&b)));
        }
    }
    (cases.len(), bad)
}

fn run_all(name: &str, fx: &Value) {
    let (mut n, mut bad) = (0, Vec::new());
    for (i, sc) in fx["scenarios"].as_array().unwrap().iter().enumerate() {
        let (k, b) = run(&format!("{name}{i}"), sc);
        n += k;
        bad.extend(b);
    }
    assert!(bad.is_empty(), "다름 {}/{n}:\n{}", bad.len(), bad.join("\n"));
    eprintln!("{name} 정답지 {n} 요청 · 다름 0");
}

#[test]
fn chart_markers_match_original() {
    run_all("markers", &load("MARKERS_FIXTURE", "routes7/markers.json"));
}

#[test]
fn position_profile_matches_original() {
    run_all("posprof", &load("POSPROF_FIXTURE", "routes7/posprof.json"));
}
