//! deribit 실행 파일 접속 루프: 127.0.0.1 가짜 WS + 가짜 REST. 구독·하트비트 글자(원본 json.dumps 모양) · test_request → public/test 답 ·
//! 채널 체결(코인 밖 종목 거름) → 버퍼 → write · 서버가 닫음(1000) → 5초 뒤 다시 · 다음 접속의 write 에 공백(ConnectionClosedOK, 백필 건수) ·
//! 접속마다 REST 백필 셋(ETH·BTC·USDC) · SIGTERM 이면 버퍼를 쓰고 0.
mod common;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[test]
fn deribit_binary_loop() {
    let dir = std::env::temp_dir().join(format!("rtduck_deribit_loop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (wl, rl) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
    let (wp, rp) = (wl.local_addr().unwrap().port(), rl.local_addr().unwrap().port());
    let seen: Arc<Mutex<Vec<(usize, f64, String)>>> = Arc::new(Mutex::new(vec![]));
    let t0 = Instant::now();
    let now_ms = (rtcollect::rest::wall_s() * 1000.0) as i64;
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for (k, c) in wl.incoming().enumerate() {
            let mut c = c.unwrap();
            let s3 = s2.clone();
            std::thread::spawn(move || {
                if common::handshake(&mut c).is_none() {
                    return;
                }
                let mut texts = 0;
                while texts < 2 {
                    let Some((op, p)) = common::read_frame(&mut c) else { return };
                    if op == 1 {
                        s3.lock().unwrap().push((k, t0.elapsed().as_secs_f64(), String::from_utf8(p).unwrap()));
                        texts += 1;
                    }
                }
                if k > 0 {
                    while let Some((op, p)) = common::read_frame(&mut c) {
                        if op == 9 {
                            common::send(&mut c, 10, &p);
                        }
                        if op == 8 {
                            return;
                        }
                    }
                    return;
                }
                common::send(&mut c, 1, br#"{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}"#);
                if let Some((1, p)) = common::read_frame(&mut c) {
                    s3.lock().unwrap().push((k, t0.elapsed().as_secs_f64(), String::from_utf8(p).unwrap()));
                }
                let tr = |id: u32, inst: &str| format!(r#"{{"timestamp":{},"trade_id":"ETH-{id}","trade_seq":{id},"instrument_name":"{inst}","direction":"buy","amount":2.5,"price":0.05,"mark_price":0.051,"index_price":2600.0,"iv":55.0,"tick_direction":1}}"#, now_ms - 1000);
                let m = format!(r#"{{"jsonrpc":"2.0","method":"subscription","params":{{"channel":"trades.option.ETH.100ms","data":[{},{},{}]}}}}"#,
                                tr(1, "ETH-26DEC26-3000-C"), tr(2, "AVAX_USDC-26DEC26-30-C"), tr(3, "ETH-26DEC26-2600-P"));
                common::send(&mut c, 1, m.as_bytes());
                std::thread::sleep(Duration::from_millis(1200)); // 버퍼 비우기(0.5초) 한 번 지나게
                common::send(&mut c, 1, br#"{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"heartbeat"}}"#);
                std::thread::sleep(Duration::from_millis(200));
                common::send(&mut c, 8, &1000u16.to_be_bytes());
                std::thread::sleep(Duration::from_millis(200));
            });
        }
    });
    let rest: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let r2 = rest.clone();
    std::thread::spawn(move || {
        for c in rl.incoming() {
            let mut c = c.unwrap();
            let mut buf = [0u8; 4096];
            let n = c.read(&mut buf).unwrap_or(0);
            r2.lock().unwrap().push(String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string());
            let body = r#"{"jsonrpc":"2.0","result":{"trades":[],"has_more":false}}"#;
            let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_rtduck"))
        .env("RTDUCK_KIND", "deribit").env("DERIBIT_BT_DIR", &dir).env("RTDUCK_REST", "1").env("RTDUCK_DERIBIT_CHAIN", "0").env("RTCOLLECT_TEST", "1").env("RTDUCK_DERIBIT_FLUSH_S", "0.5")
        .env("RTDUCK_DERIBIT_WS", format!("ws://127.0.0.1:{wp}/ws/api/v2")).env("RTDUCK_DERIBIT_REST", format!("http://127.0.0.1:{rp}/api/v2/public/get_last_trades_by_currency_and_time"))
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    while seen.lock().unwrap().iter().filter(|x| x.0 == 1).count() < 2 && t0.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(50));
    }
    while rest.lock().unwrap().len() < 6 && t0.elapsed() < Duration::from_secs(35) {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(500));
    assert!(std::process::Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "SIGTERM 은 0:\n{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let s = seen.lock().unwrap().clone();
    let sub = r#"{"jsonrpc": "2.0", "id": 1, "method": "public/subscribe", "params": {"channels": ["trades.option.ETH.100ms", "trades.option.BTC.100ms", "trades.option.USDC.100ms"]}}"#;
    let hb = r#"{"jsonrpc": "2.0", "id": 2, "method": "public/set_heartbeat", "params": {"interval": 30}}"#;
    let texts: Vec<&str> = s.iter().map(|x| x.2.as_str()).collect();
    assert_eq!(texts, [sub, hb, r#"{"jsonrpc": "2.0", "id": 3, "method": "public/test", "params": {}}"#, sub, hb], "보낸 글자");
    let redial = s[3].1 - s[1].1;
    assert!((5.0..12.0).contains(&redial), "닫힘 뒤 5초(+보낸 시간): {redial:.2}");
    let r = rest.lock().unwrap().clone();
    assert_eq!(r.len(), 6, "접속마다 백필 셋: {r:?}");
    for (i, api) in ["ETH", "BTC", "USDC", "ETH", "BTC", "USDC"].iter().enumerate() {
        assert!(r[i].starts_with(&format!("GET /api/v2/public/get_last_trades_by_currency_and_time?currency={api}&kind=option&start_timestamp=")) && r[i].contains("&count=1000&sorting=asc&include_old=true "), "{}", r[i]);
    }
    let c = duckdb::Connection::open(dir.join("deribit_options.duckdb")).unwrap();
    let ids: Vec<String> = c.prepare("SELECT trade_id FROM option_trades ORDER BY 1").unwrap().query_map([], |r| r.get(0)).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(ids, ["ETH-1", "ETH-3"], "코인 밖 종목은 거른다");
    let g: Vec<(String, Option<i64>)> = c.prepare("SELECT reason, backfilled FROM gaps").unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(|x| x.unwrap()).collect();
    assert_eq!(g, [("ConnectionClosedOK".to_string(), Some(0))], "다시 붙은 write 에 공백 한 줄");
    let st: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("deribit_block_trades_state.json")).unwrap()).unwrap();
    assert_eq!(st["flow_by_coin"]["ETH"].as_array().unwrap().len(), 25);
    eprintln!("deribit 접속 루프: 연결 2 · 다시 붙기 {redial:.2}초 · REST 백필 {} · 체결 {} · 공백 {g:?}", r.len(), ids.len());
    let _ = std::fs::remove_dir_all(&dir);
}
