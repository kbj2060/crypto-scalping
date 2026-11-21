//! rtmodel::gate -- 가짜 HTTP 서버(127.0.0.1)와 가짜 시계로: 가중치 머리 1,300(> 1,200)이면 그 분의 나머지 요청 0 · 418 이면 밴 파일 → 다음 요청 안 보냄 · 요청 사이 최소 간격.
//!     cargo test -j 2 --release --test gate -- --nocapture
use rtmodel::gate::{Fail, Gate};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

/// 응답마다 (상태, 가중치) 를 차례로 준다 · 받은 순간의 가짜 시각을 적는다.
fn server(plan: Vec<(u16, u32)>, clock: Arc<Mutex<f64>>, hits: Arc<Mutex<Vec<f64>>>) -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for (k, s) in l.incoming().enumerate() {
            let mut s = s.unwrap();
            let mut buf = vec![0u8; 4096];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                let m = s.read(&mut buf[n..]).unwrap();
                if m == 0 { break; }
                n += m;
            }
            hits.lock().unwrap().push(*clock.lock().unwrap());
            let (st, w) = plan.get(k).copied().unwrap_or((200, 10));
            let body = if st == 418 { "{\"code\":-1003,\"msg\":\"Way too many requests; IP banned until 1795300000000.\"}".to_string() } else { "[]".into() };
            let _ = s.write_all(format!("HTTP/1.1 {st} X\r\nContent-Type: application/json\r\nx-mbx-used-weight-1m: {w}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    port
}

fn gate(clock: &Arc<Mutex<f64>>, ban: std::path::PathBuf) -> Gate {
    let (c1, c2) = (clock.clone(), clock.clone());
    Gate::new(std::time::Duration::from_secs(5), ban).with_clock(Box::new(move || *c1.lock().unwrap()), Box::new(move |s| *c2.lock().unwrap() += s))
}

#[test]
fn weight_limit_holds_rest_of_minute() {
    let clock = Arc::new(Mutex::new(130.0)); // 분 120~180 의 10초째
    let hits = Arc::new(Mutex::new(vec![]));
    let port = server(vec![(200, 1300), (200, 40), (200, 40)], clock.clone(), hits.clone());
    let dir = std::env::temp_dir().join(format!("rt_gate_{}", std::process::id()));
    let mut g = gate(&clock, dir.join("ban"));
    let url = format!("http://127.0.0.1:{port}/fapi/v1/klines?symbol=ETHUSDT");
    for _ in 0..3 {
        assert!(g.send(&url, true).is_ok());
    }
    let h = hits.lock().unwrap().clone();
    println!("요청 시각 {h:?} · 쉰 초 {:.1}", g.held_s);
    assert_eq!(h[0], 130.0);
    assert!(h[1..].iter().all(|&t| t >= 180.0), "가중치 1,300 을 본 분(120~180)에는 더 보내지 않는다: {h:?}");
    assert!((g.held_s - 50.5).abs() < 1e-9 && g.weight == Some(40.0));
    assert!(h[2] - h[1] >= 0.2 - 1e-9, "요청 사이 최소 간격 200ms");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn http_418_writes_ban_and_stops() {
    let clock = Arc::new(Mutex::new(1_795_299_000.0));
    let hits = Arc::new(Mutex::new(vec![]));
    let port = server(vec![(418, 2400), (200, 1)], clock.clone(), hits.clone());
    let dir = std::env::temp_dir().join(format!("rt_gate418_{}", std::process::id()));
    let mut g = gate(&clock, dir.join("ban"));
    let url = format!("http://127.0.0.1:{port}/fapi/v1/klines");
    assert_eq!(g.send(&url, true).ok().map(|x| x.0), Some(418));
    assert!(std::fs::read_to_string(dir.join("ban")).unwrap().starts_with("1795300000.000"), "본문의 해제 시각을 밴 파일에");
    *clock.lock().unwrap() += 100.0; // 가중치 문턱 쉼이 끝난 뒤에도
    assert!(matches!(g.send(&url, true), Err(Fail::Banned(_))), "밴 중이면 보내지 않는다");
    assert_eq!(hits.lock().unwrap().len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}
