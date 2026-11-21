//! market_data 실행 파일 접속 루프(요격기 + 미시구조 + dashboard_state 한 프로세스): 127.0.0.1 가짜 웹소켓(@forceOrder · @depth20 · @aggTrade)
//! + 가짜 바이낸스 REST(openInterest · premiumIndex). 요격기: 1011 이면 3초 쉬고 다시 · 1000 이면 곧바로 다시 · liq_burst_state.json.
//! 미시구조: 종목마다 호가·체결 줄기 · 폴링 요청 열(OI → premiumIndex, 3초마다) · 밴 파일이면 요청 0 · 표 생성. SIGTERM 0.
mod common;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn now_ms() -> i64 {
    (rtcollect::rest::wall_s() * 1000.0) as i64
}

fn fo(side: &str, z: f64) -> String {
    let t = now_ms();
    format!(r#"{{"e":"forceOrder","E":{t},"o":{{"s":"ETHUSDT","S":"{side}","z":"{z}","ap":"2500.00","T":{t}}}}}"#)
}

type Seen = Arc<Mutex<Vec<(f64, String)>>>;

/// 웹소켓 서버: 경로로 나눈다. forceOrder 는 1011 · 1000 · 그대로 둠 순서, 호가·체결은 메시지를 계속 보낸다.
fn ws_server(l: TcpListener, seen: Seen, t0: Instant, fo_n: Arc<Mutex<usize>>) {
    for c in l.incoming() {
        let (seen, fo_n) = (seen.clone(), fo_n.clone());
        std::thread::spawn(move || {
            let mut c = c.unwrap();
            let Some(line) = common::handshake(&mut c) else { return };
            seen.lock().unwrap().push((t0.elapsed().as_secs_f64(), line.clone()));
            if line.contains("@forceOrder") {
                let k = { let mut n = fo_n.lock().unwrap(); *n += 1; *n - 1 };
                for m in [fo("SELL", 40.0 + k as f64), fo("BUY", 8.0)] {
                    common::send(&mut c, 1, m.as_bytes());
                }
                std::thread::sleep(Duration::from_millis(300));
                match k {
                    0 => common::send(&mut c, 8, &1011u16.to_be_bytes()),
                    1 => common::send(&mut c, 8, &1000u16.to_be_bytes()),
                    _ => while let Some((op, p)) = common::read_frame(&mut c) {
                        if op == 9 {
                            common::send(&mut c, 10, &p);
                        }
                    },
                }
                std::thread::sleep(Duration::from_millis(200));
                return;
            }
            let depth = line.contains("@depth20");
            let _ = c.set_nonblocking(false);
            for i in 0..200 {
                let m = if depth {
                    r#"{"e":"depthUpdate","b":[["2499.9","10.0"],["2499.8","5.0"]],"a":[["2500.1","3.0"],["2500.2","1.0"]]}"#.to_string()
                } else {
                    format!(r#"{{"e":"aggTrade","a":{i},"p":"2500.00","q":"{}","T":{},"m":{}}}"#, if i % 7 == 0 { 50.0 } else { 0.5 }, now_ms(), i % 3 == 0)
                };
                common::send(&mut c, 1, m.as_bytes());
                std::thread::sleep(Duration::from_millis(100));
            }
        });
    }
}

/// REST 서버: 요청 줄 기록 · OI·premiumIndex JSON.
fn rest_server(l: TcpListener, seen: Seen, t0: Instant) {
    for c in l.incoming() {
        let mut c = c.unwrap();
        let mut buf = [0u8; 2048];
        let n = c.read(&mut buf).unwrap_or(0);
        let line = String::from_utf8_lossy(&buf[..n]).lines().next().unwrap_or("").to_string();
        seen.lock().unwrap().push((t0.elapsed().as_secs_f64(), line.clone()));
        let body = if line.contains("openInterest") { r#"{"symbol":"ETHUSDT","openInterest":"1500000.5","time":1}"# } else { r#"{"symbol":"ETHUSDT","markPrice":"2500.10","lastFundingRate":"0.00010000"}"# };
        let _ = write!(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
    }
}

#[test]
fn market_binary_loop_tail_micro_dashboard() {
    let dir = std::env::temp_dir().join(format!("rtduck_market_loop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let t0 = Instant::now();
    // 실행마다 가짜 서버 둘을 새 포트로(🔴2026-11-21: 하나를 같이 쓰면 밴 중 실행이 SIGTERM 직전에 낸 요격기 재접속이 다음 실행의 기록·각본 번호에 섞일 수 있다)
    let servers = || {
        let (wl, rl) = (TcpListener::bind("127.0.0.1:0").unwrap(), TcpListener::bind("127.0.0.1:0").unwrap());
        let (wp, rp) = (wl.local_addr().unwrap().port(), rl.local_addr().unwrap().port());
        let (ws_seen, rest_seen): (Seen, Seen) = (Arc::new(Mutex::new(vec![])), Arc::new(Mutex::new(vec![])));
        let (a, b) = (ws_seen.clone(), rest_seen.clone());
        std::thread::spawn(move || ws_server(wl, a, t0, Arc::new(Mutex::new(0usize))));
        std::thread::spawn(move || rest_server(rl, b, t0));
        (wp, rp, ws_seen, rest_seen)
    };
    let run = |ban: &str, wp: u16, rp: u16| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_rtduck"));
        cmd.env("RTDUCK_KIND", "market_data").env("BOT_SYMBOLS", "ETHUSDT").env("TAIL_SYMBOLS", "ETHUSDT").env("RTDUCK_LIVE_DIR", &dir)
            .env("RTDUCK_REST", "1").env("RTDUCK_BINANCE_REST", format!("http://127.0.0.1:{rp}")).env("RTCOLLECT_BAN_FILE", dir.join(ban))
            .env("MS_POLL_INTERVAL_SEC", "3")
            .env("RTDUCK_FORCE_WS_URL", format!("ws://127.0.0.1:{wp}/market/ws/{{symbol}}@forceOrder"))
            .env("RTDUCK_DEPTH_WS_URL", format!("ws://127.0.0.1:{wp}/ws/{{symbol}}@depth20@100ms"))
            .env("RTDUCK_TRADE_WS_URL", format!("ws://127.0.0.1:{wp}/market/ws/{{symbol}}@aggTrade"))
            .stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd
    };
    // 밴 중: REST 요청 0(웹소켓은 붙는다) -- 웹소켓이 붙은 것을 본 뒤 끈다(고정 잠 대신 조건 대기)
    std::fs::write(dir.join("ban_on"), format!("{:.3}\n", rtcollect::rest::wall_s() + 600.0)).unwrap();
    let (wp0, rp0, ws0, rest0) = servers();
    let c = run("ban_on", wp0, rp0).spawn().unwrap();
    while ws0.lock().unwrap().len() < 3 && t0.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(3500)); // 폴링 한 바퀴(3초)가 지날 만큼 -- 그동안 REST 0 이어야 한다
    let _ = Command::new("kill").args(["-TERM", &c.id().to_string()]).status();
    assert!(c.wait_with_output().unwrap().status.success());
    assert!(ws0.lock().unwrap().len() >= 3, "밴 중에도 웹소켓 셋은 붙는다");
    assert!(rest0.lock().unwrap().is_empty(), "밴 중이면 REST 를 안 보낸다");
    let (wp, rp, ws_seen, rest_seen) = servers();
    let t1 = t0.elapsed().as_secs_f64();
    let child = run("ban_none", wp, rp).spawn().unwrap();
    while ws_seen.lock().unwrap().iter().filter(|x| x.1.contains("@forceOrder")).count() < 3 && t0.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(3500));
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "SIGTERM 은 0:\n{}", String::from_utf8_lossy(&out.stderr));
    let s = ws_seen.lock().unwrap().clone();
    let fo: Vec<f64> = s.iter().filter(|x| x.1.contains("@forceOrder")).map(|x| x.0).collect();
    assert_eq!(fo.len(), 3, "요격기 연결 셋: {s:?}");
    let (r1, r2) = (fo[1] - fo[0], fo[2] - fo[1]);
    assert!((3.0..6.0).contains(&r1), "1011 뒤 3초(부하 여유): {r1:.2}");
    assert!(r2 < 2.0, "1000 뒤 곧바로(3초 쉼 없음): {r2:.2}");
    assert!(s.iter().any(|x| x.1.starts_with("GET /ws/ethusdt@depth20@100ms ")) && s.iter().any(|x| x.1.starts_with("GET /market/ws/ethusdt@aggTrade ")), "{s:?}");
    let r: Vec<(f64, String)> = rest_seen.lock().unwrap().iter().filter(|x| x.0 > t1).cloned().collect();
    let paths: Vec<&str> = r.iter().map(|x| x.1.split(' ').nth(1).unwrap()).collect();
    // 첫 바퀴: 체결을 아직 못 받았으니(마지막 수신 0 → 30초 넘음) aggTrades 대체가 붙는다(원본 그대로) · 그 뒤로 OI → premiumIndex
    let (oi, pi) = ("/fapi/v1/openInterest?symbol=ETHUSDT", "/fapi/v1/premiumIndex?symbol=ETHUSDT");
    assert!(paths.len() >= 5 && paths[..3] == [oi, pi, "/fapi/v1/aggTrades?symbol=ETHUSDT&limit=500"] && paths[3..].chunks(2).all(|p| p[0] == oi && p.get(1).is_none_or(|q| *q == pi)), "{paths:?}");
    let gap = r[3].0 - r[0].0;
    assert!((2.8..6.0).contains(&gap), "폴링 간격 3초(부하 여유): {gap:.2}");
    let b: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("liq_burst_state.json")).unwrap()).unwrap();
    assert_eq!(b["ws_connected"], true);
    let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("dashboard_state.json")).unwrap()).unwrap();
    assert_eq!(d["source"], "duckdb_persist_worker");
    let m = rtduck::duckdump::dump(&dir.join("microstructure.duckdb")).unwrap();
    assert_eq!(m["microstructure_1m"]["cols"].as_array().unwrap().len(), 34, "미시구조 표 34칸");
    eprintln!("market 접속 루프: 요격기 연결 3(1011 뒤 {r1:.2}초 · 1000 뒤 {r2:.2}초) · REST {} 요청(간격 {gap:.2}초) · 미시구조 웹소켓 둘", paths.len());
    let _ = std::fs::remove_dir_all(&dir);
}
