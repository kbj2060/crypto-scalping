//! market_data 실행 파일 접속 루프: 127.0.0.1 가짜 @forceOrder 웹소켓 → 청산 메시지 → liq_burst_state.json · 10초 dashboard_state.json
//! · 서버가 오류로 닫으면(1011) 3초 쉬고 다시 · 정상 종료(1000)면 곧바로 다시 · SIGTERM 0 · 미시구조 켜면 거부(종료코드 2).
mod common;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn fo(side: &str, z: f64) -> String {
    let t = (rtcollect::rest::wall_s() * 1000.0) as i64;
    format!(r#"{{"e":"forceOrder","E":{t},"o":{{"s":"ETHUSDT","S":"{side}","z":"{z}","ap":"2500.00","T":{t}}}}}"#)
}

#[test]
fn market_binary_loop_against_fake_force_order() {
    let dir = std::env::temp_dir().join(format!("rtduck_market_loop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen: Arc<Mutex<Vec<(f64, String)>>> = Arc::new(Mutex::new(vec![]));
    let (s2, t0) = (seen.clone(), Instant::now());
    std::thread::spawn(move || {
        for (k, close) in [Some(1011u16), Some(1000), None].into_iter().enumerate() {
            let (mut c, _) = l.accept().unwrap();
            let line = common::handshake(&mut c).unwrap();
            s2.lock().unwrap().push((t0.elapsed().as_secs_f64(), line));
            for m in [fo("SELL", 40.0 + k as f64), fo("BUY", 8.0)] {
                common::send(&mut c, 1, m.as_bytes());
            }
            std::thread::sleep(Duration::from_millis(300));
            match close {
                Some(code) => {
                    common::send(&mut c, 8, &code.to_be_bytes());
                    std::thread::sleep(Duration::from_millis(200));
                }
                None => {
                    std::thread::spawn(move || while let Some((op, p)) = common::read_frame(&mut c) {
                        if op == 9 {
                            common::send(&mut c, 10, &p);
                        }
                    });
                }
            }
        }
    });
    let run = |extra: &[(&str, &str)]| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtduck"));
        cmd.env("RTDUCK_KIND", "market_data").env("COLLECT_MICROSTRUCTURE", "false").env("RTDUCK_LIVE_DIR", &dir)
            .env("RTDUCK_FORCE_WS_URL", format!("ws://127.0.0.1:{port}/market/ws/{{symbol}}@forceOrder")).stdout(Stdio::piped()).stderr(Stdio::piped());
        for (k, v) in extra {
            cmd.env(k, v);
        }
        cmd
    };
    let bad = run(&[("COLLECT_MICROSTRUCTURE", "true")]).output().unwrap();
    assert_eq!(bad.status.code(), Some(2), "미시구조를 켜면 거부");
    let child = run(&[]).spawn().unwrap();
    while seen.lock().unwrap().len() < 3 && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(800));
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "SIGTERM 은 0:\n{}", String::from_utf8_lossy(&out.stderr));
    let s = seen.lock().unwrap().clone();
    assert_eq!(s.len(), 3, "연결 셋");
    assert!(s.iter().all(|x| x.1.starts_with("GET /market/ws/ethusdt@forceOrder ")), "{s:?}");
    let (r1, r2) = (s[1].0 - s[0].0, s[2].0 - s[1].0);
    assert!((3.0..4.8).contains(&r1), "오류 닫힘(1011) 뒤 3초: {r1:.2}");
    assert!(r2 < 1.2, "정상 종료(1000) 뒤 곧바로: {r2:.2}");
    let b: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("liq_burst_state.json")).unwrap()).unwrap();
    assert_eq!(b["ws_connected"], true);
    assert!(b["long_usd_1m"].as_f64().unwrap() >= 2500.0 * 40.0 && b["liq_event_count_1m"].as_i64().unwrap() >= 2, "{b}");
    let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("dashboard_state.json")).unwrap()).unwrap();
    assert_eq!(d["source"], "duckdb_persist_worker");
    assert!(d["session"]["session_us"].is_f64() && d["updated_at"].as_str().unwrap().ends_with("+09:00"), "{d}");
    eprintln!("market 접속 루프: 연결 3 · 1011 뒤 {r1:.2}초 · 1000 뒤 {r2:.2}초 · liq_burst long ${:.0}", b["long_usd_1m"].as_f64().unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}
