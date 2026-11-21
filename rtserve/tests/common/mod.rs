//! 시험 공통: 날 HTTP 요청 · 시나리오 파일 트리 펴기 · 서버 띄우기 · 헤더 정규화.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use serde_json::Value;

pub fn b64(s: &str) -> Vec<u8> {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|c| *c != b'=') {
        acc = acc << 6 | T.iter().position(|t| *t == c).unwrap() as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

pub struct Got {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub chunked: bool,
    pub body: Vec<u8>,
}

/// 날 HTTP/1.1 요청 하나(Connection: close) → 응답을 끝까지 읽어 나눈다
pub fn request(addr: SocketAddr, method: &str, path: &str, headers: &[(String, String)]) -> Got {
    request_body(addr, method, path, headers, b"")
}

/// 본문을 실은 요청(POST) -- Content-Length 를 붙인다
pub fn request_body(addr: SocketAddr, method: &str, path: &str, headers: &[(String, String)], body: &[u8]) -> Got {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !body.is_empty() || method == "POST" {
        req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes()).unwrap();
    s.write_all(body).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut body = raw[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let first: Vec<&str> = lines.next().unwrap().splitn(3, ' ').collect();
    let headers: Vec<(String, String)> = lines.map(|l| {
        let (k, v) = l.split_once(": ").unwrap();
        (k.to_lowercase(), v.to_string())
    }).collect();
    let chunked = headers.iter().any(|(k, v)| k == "transfer-encoding" && v == "chunked");
    if chunked {
        let (mut out, mut rest) = (Vec::new(), &body[..]);
        loop {
            let nl = rest.windows(2).position(|w| w == b"\r\n").unwrap();
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..nl]).unwrap(), 16).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&rest[nl + 2..nl + 2 + n]);
            rest = &rest[nl + 4 + n..];
        }
        body = out;
    }
    Got { status: first[1].parse().unwrap(), reason: first.get(2).unwrap_or(&"").to_string(), headers, chunked, body }
}

/// 시나리오 파일 트리를 임시 폴더에 펴고(수정 시각 고정) 서버를 띄운다
pub fn start(files: &serde_json::Map<String, Value>, mtime_ns: u64, now: f64, assets: &str, tag: &str) -> (SocketAddr, PathBuf) {
    start_with(files, mtime_ns, now, assets, tag, |_| {})
}

/// 설정을 손본 뒤 띄운다(알림 키·로그 모으기)
pub fn start_with(files: &serde_json::Map<String, Value>, mtime_ns: u64, now: f64, assets: &str, tag: &str, tweak: impl FnOnce(&mut rtserve::Config)) -> (SocketAddr, PathBuf) {
    let root = std::env::temp_dir().join(format!("rtserve_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("dashboard/live")).unwrap();
    for (rel, data) in files {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b64(data.as_str().unwrap())).unwrap();
        set_mtime(&p, mtime_ns);
    }
    let mut cfg = rtserve::Config::new(root.clone(), rtserve::parse_assets(Some(assets)), Arc::new(move || now));
    tweak(&mut cfg);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap()).unwrap();
            let c = Arc::new(cfg.clone());
            tokio::spawn(c.events.clone().run(c.clone(), c.live.clone())); // /api/events 발행기(원본 시작 훅)
            rtserve::serve(l, cfg).await.unwrap();
        });
    });
    (rx.recv().unwrap(), root)
}

pub fn set_mtime(p: &Path, ns: u64) {
    let t = UNIX_EPOCH + Duration::from_nanos(ns);
    std::fs::File::options().write(true).open(p).unwrap().set_times(std::fs::FileTimes::new().set_modified(t).set_accessed(t)).unwrap();
}

pub const SKIP: [&str; 3] = ["date", "server", "connection"];

pub fn norm(h: &[(String, String)]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = h.iter().map(|(k, v)| (k.to_lowercase(), v.clone())).filter(|(k, _)| !SKIP.contains(&k.as_str())).collect();
    v.sort();
    v
}

