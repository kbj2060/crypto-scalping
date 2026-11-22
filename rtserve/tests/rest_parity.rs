//! 아홉째 조각: 요청 때 바이낸스 REST 를 부르는 주소(src/rest.rs)와 한 문(model::binance_get) -- 127.0.0.1 가짜 REST 만(바깥 접속 없음).
//! ① 정답지(tools/gen_rest_fixture.py · 원본 핸들러 + trend_rule)와 본문 바이트·상태·보낸 요청 열이 같다 ② 가짜 REST 가 `x-mbx-used-weight-1m: 1300` 을 알리면
//! 그 분 나머지 바이낸스 요청 0(주소 넷 · 모델 주소 open-rev 포함) -- 1100 이면 그대로 보냄(양성 대조) ③ RTSERVE_REST 없음 = 보내지 않음.
//!     cargo test --release -j 2 --offline --test rest_parity

mod common;

use common::{request, start_with};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Fake {
    table: HashMap<String, (u16, String)>,
    sent: Vec<String>,
    weight: Option<String>,
}

fn fake(st: Arc<Mutex<Fake>>) -> String {
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
            g.sent.push(target.clone());
            let (code, body) = g.table.get(&target).cloned().unwrap_or((404, "[]".into()));
            let w = g.weight.as_ref().map_or(String::new(), |w| format!("x-mbx-used-weight-1m: {w}\r\n"));
            drop(g);
            let _ = s.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: application/json\r\n{w}Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn fixture() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gen/rest/rest.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e} -- rtserve/tools/gen_rest_fixture.py 먼저", p.display()))).unwrap()
}

fn serve(tag: &str, now: f64, rest: Option<String>) -> std::net::SocketAddr {
    start_with(&Map::new(), 0, now, "eth,btc,sol,xrp", tag, move |c| c.rest = rest.unwrap_or_default()).0
}

#[test]
fn rest_endpoints_match_original() {
    let fx = fixture();
    let st = Arc::new(Mutex::new(Fake::default()));
    for (k, v) in fx["rest"].as_object().unwrap() {
        st.lock().unwrap().table.insert(k.clone(), (v[0].as_u64().unwrap() as u16, v[1].as_str().unwrap().to_string()));
    }
    let addr = serve("rest_parity", fx["now"].as_f64().unwrap(), Some(fake(st.clone())));
    let mut bad = Vec::new();
    let cases = fx["cases"].as_array().unwrap();
    for c in cases {
        st.lock().unwrap().sent.clear();
        let path = c["path"].as_str().unwrap();
        let g = request(addr, "GET", path, &[]);
        let sent: Vec<Value> = st.lock().unwrap().sent.iter().map(|s| Value::from(s.as_str())).collect();
        if g.status as u64 != c["status"].as_u64().unwrap() || g.body != c["body"].as_str().unwrap().as_bytes() || sent != *c["sent"].as_array().unwrap() {
            bad.push(format!("{path}: 상태 {} · 본문 {} · 보냄 {sent:?}\n  정답 {} · {} · {}", g.status, String::from_utf8_lossy(&g.body), c["status"], c["body"], c["sent"]));
        }
    }
    assert!(bad.is_empty(), "다름 {}/{}:\n{}", bad.len(), cases.len(), bad.join("\n"));
    eprintln!("rest 정답지 {} 요청 · 다름 0", cases.len());
}

/// 🔴가중치 문턱: 1300 을 알린 뒤 그 분 나머지 요청 0 · 1100 은 문턱 아래(그대로 보냄)
#[test]
fn weight_over_limit_holds_rest_of_minute() {
    let st = Arc::new(Mutex::new(Fake { weight: Some("1100".into()), ..Default::default() }));
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    // 분 끝 1.5초 안이면 다음 분으로 넘어가 «그 분 나머지»가 너무 짧다 -- 기다렸다 시작
    if now % 60.0 > 58.5 {
        std::thread::sleep(std::time::Duration::from_secs_f64(61.0 - now % 60.0));
    }
    let addr = serve("rest_weight", now, Some(fake(st.clone())));
    let paths = ["/api/trend?asset=eth", "/api/trend?asset=sol", "/api/cycle-path?symbol=XRPUSDT&start=", "/api/open-rev"];
    request(addr, "GET", paths[0], &[]);
    request(addr, "GET", paths[1], &[]);
    assert_eq!(st.lock().unwrap().sent.len(), 2, "1100 은 문턱 아래 -- 둘 다 보내야 한다");
    st.lock().unwrap().weight = Some("1300".into());
    request(addr, "GET", "/api/trend?asset=xrp", &[]);
    assert_eq!(st.lock().unwrap().sent.len(), 3);
    let start = (now * 1000.0) as i64 - 60_000;
    let mut bodies = Vec::new();
    for p in ["/api/trend?asset=btc", &format!("/api/cycle-path?symbol=XRPUSDT&start={start}"), "/api/trend?asset=hype", paths[3]] {
        bodies.push(String::from_utf8_lossy(&request(addr, "GET", p, &[]).body).into_owned());
    }
    assert_eq!(st.lock().unwrap().sent.len(), 3, "1300 뒤 그 분 나머지 요청 0 이어야 한다: {:?}", st.lock().unwrap().sent);
    assert_eq!(bodies[0], r#"{"ok": false, "reason": "HTTPServiceUnavailable: binance_weight_hold"}"#);
    assert_eq!(bodies[1], r#"{"ok": false, "reason": "binance_weight_hold", "rows": []}"#);
    assert!(bodies[3].contains("binance_weight_hold"), "{}", bodies[3]);
}

#[test]
fn rest_off_sends_nothing() {
    let addr = serve("rest_off", 1_795_000_123.0, None);
    let b = request(addr, "GET", "/api/trend?asset=eth", &[]).body;
    assert_eq!(String::from_utf8_lossy(&b), r#"{"ok": false, "reason": "RestOff: RTSERVE_REST unset -- request not sent"}"#);
    let b = request(addr, "GET", "/api/cycle-path?symbol=ETHUSDT&start=1794999523456", &[]).body;
    assert_eq!(String::from_utf8_lossy(&b), r#"{"ok": false, "reason": "RestOff: RTSERVE_REST unset -- request not sent", "rows": []}"#);
}
