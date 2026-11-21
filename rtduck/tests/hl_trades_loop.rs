//! hl_trades 실행 파일 접속 루프: 127.0.0.1 가짜 웹소켓(표준 라이브러리만 -- 시험 의존성을 더하면 DuckDB 를 다시 빌드한다)으로
//! 접속 → 구독 글자 → 체결 → 서버가 닫음 → 5초 뒤 다시 붙음 → 자정 넘기(RTDUCK_FAKE_START) → 재구독 스냅숏 중복 거르기 → SIGTERM 때 남은 버퍼 쓰기.
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn sha1(msg: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut m = msg.to_vec();
    let bits = (msg.len() as u64) * 8;
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for c in m.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(c[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut cc, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & cc) | (!b & d), 0x5A827999),
                20..=39 => (b ^ cc ^ d, 0x6ED9EBA1),
                40..=59 => ((b & cc) | (b & d) | (cc & d), 0x8F1BBCDC),
                _ => (b ^ cc ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            (e, d, cc, b, a) = (d, cc, b.rotate_left(30), a, t);
        }
        for (x, y) in h.iter_mut().zip([a, b, cc, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

fn b64(b: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            s.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

/// 클라이언트 프레임 하나(가려진 것) → (opcode, 본문). None = 끊김.
fn read_frame(s: &mut TcpStream) -> Option<(u8, Vec<u8>)> {
    let mut h = [0u8; 2];
    s.read_exact(&mut h).ok()?;
    let mut n = (h[1] & 127) as u64;
    if n == 126 {
        let mut b = [0u8; 2];
        s.read_exact(&mut b).ok()?;
        n = u16::from_be_bytes(b) as u64;
    } else if n == 127 {
        let mut b = [0u8; 8];
        s.read_exact(&mut b).ok()?;
        n = u64::from_be_bytes(b);
    }
    let mut mask = [0u8; 4];
    if h[1] & 128 != 0 {
        s.read_exact(&mut mask).ok()?;
    }
    let mut p = vec![0u8; n as usize];
    s.read_exact(&mut p).ok()?;
    for (i, x) in p.iter_mut().enumerate() {
        *x ^= mask[i % 4];
    }
    Some((h[0] & 15, p))
}

fn send(s: &mut TcpStream, op: u8, p: &[u8]) {
    let mut f = vec![0x80 | op];
    if p.len() < 126 {
        f.push(p.len() as u8);
    } else {
        f.push(126);
        f.extend_from_slice(&(p.len() as u16).to_be_bytes());
    }
    f.extend_from_slice(p);
    let _ = s.write_all(&f);
}

/// 접속 하나: 손 맞추기 → 구독(코인 수만큼) 받기 → msgs 보내기 → close = 닫음(아니면 끊길 때까지 받기만).
fn session(mut s: TcpStream, coins: usize, msgs: Vec<String>, close: bool, seen: Arc<Mutex<Vec<(usize, f64, String)>>>, k: usize, t0: Instant) {
    let mut req = vec![];
    let mut b = [0u8; 1];
    while !req.ends_with(b"\r\n\r\n") {
        if s.read_exact(&mut b).is_err() {
            return;
        }
        req.push(b[0]);
    }
    let req = String::from_utf8_lossy(&req).to_string();
    let key = req.lines().find_map(|l| l.to_lowercase().starts_with("sec-websocket-key:").then(|| l[18..].trim().to_string())).unwrap();
    let acc = b64(&sha1(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes()));
    write!(s, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {acc}\r\n\r\n").unwrap();
    let mut got = 0;
    while got < coins {
        let Some((op, p)) = read_frame(&mut s) else { return };
        if op == 1 {
            seen.lock().unwrap().push((k, t0.elapsed().as_secs_f64(), String::from_utf8(p).unwrap()));
            got += 1;
        }
    }
    for m in msgs {
        send(&mut s, 1, m.as_bytes());
        std::thread::sleep(Duration::from_millis(20));
    }
    if close {
        std::thread::sleep(Duration::from_millis(300));
        send(&mut s, 8, &[0x03, 0xE8]);
        std::thread::sleep(Duration::from_millis(200));
        return;
    }
    while let Some((op, p)) = read_frame(&mut s) {
        if op == 9 {
            send(&mut s, 10, &p);
        } else if op == 8 {
            return;
        }
    }
}

fn trade(coin: &str, t: i64, tid: i64) -> Value {
    json!({"coin": coin, "side": "B", "px": "2747.5", "sz": "1.25", "time": t, "tid": tid, "hash": format!("0x{tid:x}"), "users": [format!("0xAA{tid}"), "0xBb"]})
}

#[test]
fn hl_trades_binary_loop_reconnect_midnight_sigterm() {
    let dir = std::env::temp_dir().join(format!("rtduck_hl_loop_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let day0 = 20_413i64; // 2025-11-21 의 일 번호
    let midnight = (day0 + 1) as f64 * 86_400.0; // 2025-11-22 00:00 UTC
    let sub = dir.join("ETH_BTC");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("2025-11-19.duckdb"), b"3-days-old\n").unwrap(); // 자정 뒤 첫 쓰기가 압축한다(22일 - 3일)
    let ms = (midnight * 1000.0) as i64;
    let m1 = json!({"channel": "trades", "data": [trade("ETH", ms - 2500, 1), trade("BTC", ms - 2400, 2), json!({"coin": "ETH", "time": 1, "tid": 9, "users": ["0x1"]})]}).to_string();
    let m2 = json!({"channel": "trades", "data": [trade("ETH", ms - 2300, 3)]}).to_string();
    let snap = json!({"channel": "trades", "data": [trade("ETH", ms - 2300, 3), trade("ETH", ms + 4000, 4)]}).to_string(); // 3 = 이미 적음(cut 이하)
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(vec![]));
    let t0 = Instant::now();
    let s2 = seen.clone();
    std::thread::spawn(move || {
        let script = vec![(vec![m1, m2], true), (vec![snap], false)];
        for (k, (msgs, close)) in script.into_iter().enumerate() {
            let (c, _) = l.accept().unwrap();
            let s3 = s2.clone();
            std::thread::spawn(move || session(c, 2, msgs, close, s3, k, t0));
        }
    });
    let start = midnight - 3.0;
    let child = Command::new(env!("CARGO_BIN_EXE_rtduck"))
        .env("RTDUCK_KIND", "hl_trades").env("HL_COINS", "ETH,BTC").env("HL_ROOT", &dir).env("RTDUCK_FAKE_START", format!("{start}"))
        .env("RTDUCK_WS_URL", format!("ws://127.0.0.1:{port}/ws")).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    while seen.lock().unwrap().iter().filter(|x| x.0 == 1).count() < 2 && t0.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(50));
    }
    std::thread::sleep(Duration::from_millis(1500));
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "SIGTERM 은 0:\n{}", String::from_utf8_lossy(&out.stderr));
    let s = seen.lock().unwrap().clone();
    let subs: Vec<&str> = s.iter().map(|x| x.2.as_str()).collect();
    let want = [r#"{"method": "subscribe", "subscription": {"type": "trades", "coin": "ETH"}}"#, r#"{"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}}"#];
    assert_eq!(subs, [want[0], want[1], want[0], want[1]], "연결마다 코인 순서대로 구독");
    let redial = s[2].1 - s[1].1;
    assert!((4.8..7.5).contains(&redial), "서버가 닫은 뒤 5초(+보낸 시간): {redial:.2}");
    let d1 = rtduck::duckdump::dump(&sub.join("2025-11-21.duckdb")).unwrap();
    let d2 = rtduck::duckdump::dump(&sub.join("2025-11-22.duckdb")).unwrap();
    let tids = |d: &Value| d["hl_trades"]["rows"].as_array().unwrap().iter().map(|r| r[2].as_i64().unwrap()).collect::<Vec<_>>();
    assert_eq!(tids(&d1), [1, 2, 3], "자정 전 연결: 끊길 때 버퍼를 그날 파일에(users 하나인 체결은 버림)");
    assert_eq!(tids(&d2), [4], "자정 뒤: 스냅숏의 이미 적은 3 은 거르고 4 는 SIGTERM 때 썼다");
    assert_eq!(d1["hl_trades"]["rows"][0][7], "0xaa1", "주소 소문자");
    let g = d2["gaps"]["rows"].as_array().unwrap();
    assert_eq!(g.len(), 1, "공백 한 줄(다시 붙은 날 파일)");
    assert_eq!((g[0][0].as_str(), g[0][3].as_str()), (Some("ETH,BTC"), Some("ConnectionClosedError")));
    let span = (g[0][2].as_i64().unwrap() - g[0][1].as_i64().unwrap()) as f64 / 1000.0;
    assert!((4.8..7.5).contains(&span) && g[0][1].as_i64().unwrap() < ms && g[0][2].as_i64().unwrap() >= ms, "공백 = 끊긴 시각(자정 전) → 다시 붙은 시각(자정 뒤): {g:?}");
    assert!(sub.join("2025-11-19.duckdb.gz").exists() && !sub.join("2025-11-19.duckdb").exists(), "3일 전 파일 압축");
    eprintln!("hl_trades 접속 루프: 연결 2 · 다시 붙기 {redial:.2}초 · 공백 {span:.2}초 · 자정 전 3행 · 뒤 1행");
    let _ = std::fs::remove_dir_all(&dir);
}
