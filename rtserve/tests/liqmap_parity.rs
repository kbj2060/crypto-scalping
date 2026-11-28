//! 청산맵(src/liqmap.rs): /api/liquidation-map · /api/liquidation-map/tiers 가 정답지(tools/gen_liqmap_fixture.py · 원본 적재 함수 + 원본 모듈)와
//! 본문 바이트·상태·보낸 요청 열이 같다 -- 127.0.0.1 가짜 REST 만(바깥 접속 없음). OI 는 정답지가 만든 hot SQLite 세 파일.
//!     cargo test --release -j 2 --offline --test liqmap_parity

mod common;

use common::{request, start_with};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
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

#[test]
fn liqmap_endpoints_match_original() {
    // LIQMAP_FIXTURE = 양성 대조(변이 사본으로 만든 정답지) 자리
    let p = std::env::var("LIQMAP_FIXTURE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/liqmap/liqmap.json"));
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e} -- rtserve/tools/gen_liqmap_fixture.py 먼저", p.display()))).unwrap();
    let st: Table = Default::default();
    for (k, v) in fx["rest"].as_object().unwrap() {
        st.lock().unwrap().0.insert(k.clone(), (v[0].as_u64().unwrap() as u16, v[1].as_str().unwrap().to_string()));
    }
    let rest = fake(st.clone());
    let (addr, _) = start_with(fx["files"].as_object().unwrap(), 0, fx["now"].as_f64().unwrap(), "eth,btc,sol,xrp", "liqmap_parity", move |c| c.rest = rest);
    let mut bad = Vec::new();
    let cases = fx["cases"].as_array().unwrap();
    for c in cases {
        st.lock().unwrap().1.clear();
        let path = c["path"].as_str().unwrap();
        let g = request(addr, "GET", path, &[]);
        let sent: Vec<Value> = st.lock().unwrap().1.iter().map(|s| Value::from(s.as_str())).collect();
        if g.status as u64 != c["status"].as_u64().unwrap() || g.body != c["body"].as_str().unwrap().as_bytes() || sent != *c["sent"].as_array().unwrap() {
            let (a, b) = (String::from_utf8_lossy(&g.body).to_string(), c["body"].as_str().unwrap().to_string());
            let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
            let ctx = |s: &str| s.get(at.saturating_sub(80)..(at + 80).min(s.len())).unwrap_or("").to_string();
            bad.push(format!("{path}: 상태 {}/{} · 보냄 {}/{} · 본문 {} 째 글자부터 다름\n  rust {}\n  정답 {}", g.status, c["status"], sent.len(), c["sent"].as_array().unwrap().len(), at, ctx(&a), ctx(&b)));
        }
    }
    assert!(bad.is_empty(), "다름 {}/{}:\n{}", bad.len(), cases.len(), bad.join("\n"));
    eprintln!("liqmap 정답지 {} 요청 · 다름 0", cases.len());
}

/// 2026-11-28 단계 2: 미시 틱(hub::situation_once)의 sr 레벨 = /api/liquidation-map 의 레벨(같은 swr 칸 liquidation_map:eth -- 원본 collect_micro_ref 1996~2000) ·
/// 미시가 먼저 받았으면 화면 요청은 REST 를 더 안 보낸다. 레벨 값 자체는 위 시험이 원본과 글자 대조한다.
#[test]
fn micro_tick_reads_same_liquidation_levels() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/liqmap/liqmap.json");
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(&p).expect("rtserve/tools/gen_liqmap_fixture.py 먼저")).unwrap();
    let st: Table = Default::default();
    for (k, v) in fx["rest"].as_object().unwrap() {
        st.lock().unwrap().0.insert(k.clone(), (v[0].as_u64().unwrap() as u16, v[1].as_str().unwrap().to_string()));
    }
    let rest = fake(st.clone());
    let held: Arc<Mutex<Option<rtserve::Config>>> = Default::default();
    let h2 = held.clone();
    let (addr, _) = start_with(fx["files"].as_object().unwrap(), 0, fx["now"].as_f64().unwrap(), "eth,btc,sol,xrp", "liqmap_micro", move |c| {
        c.rest = rest;
        *h2.lock().unwrap() = Some(c.clone());
    });
    let cfg = held.lock().unwrap().clone().unwrap();
    cfg.live.apply(r#"{"s":"ethusdt","seq":1,"ex_ms":0,"rx_us":0,"tx_us":0,"type":"book","u":1,"bid":3000.0,"bid_qty":5.0,"ask":3000.1,"ask_qty":4.0}"#);
    let mut run = rtserve::hub::SitRun { rest_at: Some(0.0), ..Default::default() };
    rtserve::hub::situation_once(&cfg, &mut run);
    let sent_micro: Vec<String> = st.lock().unwrap().1.iter().filter(|s| s.contains("ETHUSDT&interval=1h") || s.contains("ETHUSDT&interval=5m&limit=460")).cloned().collect();
    st.lock().unwrap().1.clear();
    let g = request(addr, "GET", "/api/liquidation-map?asset=eth", &[]);
    assert_eq!(g.status, 200);
    let map: Value = serde_json::from_slice(&g.body).unwrap();
    assert_eq!(map["warmed_up"], true, "정답지 ETH 사례는 레벨이 있어야(아니면 이 시험이 빈 것끼리 같다)");
    assert_eq!(sent_micro.len(), 2, "미시 틱이 1h·5m 을 한 번씩: {sent_micro:?}");
    assert!(st.lock().unwrap().1.is_empty(), "같은 swr 칸 -- 화면 요청은 REST 0: {:?}", st.lock().unwrap().1);
    let sr = &run.micro.payload["sr"];
    for (k, src) in [("sup_levels", "support_levels"), ("res_levels", "resistance_levels")] {
        let want: Vec<Value> = map[src].as_array().unwrap().iter().map(|l| serde_json::json!({"price": l["price"], "weight_pct": l["weight_pct"]})).collect();
        assert!(!want.is_empty(), "{src} 비면 대조가 빈다");
        assert_eq!(sr[k], Value::Array(want), "{k}");
    }
    assert!(!sr["sup_bp"].is_null() || !sr["res_bp"].is_null(), "sr 이 레벨로 채워져야: {sr}");
    eprintln!("미시 sr = /api/liquidation-map 레벨 · 지지 {} · 저항 {}", sr["sup_levels"].as_array().unwrap().len(), sr["res_levels"].as_array().unwrap().len());
}
