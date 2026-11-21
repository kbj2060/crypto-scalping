//! 프로세스 안 가짜 HTTP/1.1 서버(127.0.0.1 임의 포트 · 연결마다 스레드 · keep-alive) -- 시험과 그림자 재생(`shadow`)이 같이 쓴다(2026-11-21 시험 쪽에서 옮김).
//! 받은 요청(메서드·경로+쿼리·X-MBX-APIKEY)을 적고, 응답은 넘겨받은 함수가 정한다(각본 재생 / 가짜 거래소). 🔴127.0.0.1 에만 묶는다.
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::{Arc, Mutex};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub enum Act {
    Send { status: u16, ctype: String, body: String, headers: Vec<(String, String)> },
    Close,                  // 요청을 받고 답 없이 끊는다
    Hang(f64, Box<Act>),    // 늦게 답한다(시간초과)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Got {
    pub method: String,
    pub target: String,
    pub key: Option<String>,
    /// Content-Type 머리 · 본문 바이트(텔레그램 시험이 원본과 바이트 대조).
    pub ctype: Option<String>,
    pub body: Vec<u8>,
}

pub type Handler = Arc<dyn Fn(&Got) -> Act + Send + Sync>;

pub struct Srv {
    pub base: String,
    pub log: Arc<Mutex<Vec<Got>>>,
    pub conns: Arc<AtomicUsize>,
}

/// 🔴시험은 127.0.0.1 밖으로 나가지 않는다 -- 주소가 그렇지 않으면 시작 전에 멈춘다.
pub fn local_only(base: &str) {
    assert!(base.starts_with("http://127.0.0.1:"), "시험 주소가 127.0.0.1 이 아니다: {base}");
}

pub fn serve(h: Handler) -> Srv {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    local_only(&base);
    let (log, conns) = (Arc::new(Mutex::new(Vec::new())), Arc::new(AtomicUsize::new(0)));
    let (lg, cn) = (log.clone(), conns.clone());
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            cn.fetch_add(1, Ordering::SeqCst);
            let (h, lg) = (h.clone(), lg.clone());
            std::thread::spawn(move || conn(s, h, lg));
        }
    });
    Srv { base, log, conns }
}

fn conn(s: TcpStream, h: Handler, log: Arc<Mutex<Vec<Got>>>) {
    let mut w = s.try_clone().unwrap();
    let mut r = BufReader::new(s);
    for _ in 0..10_000 {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut it = line.split_whitespace();
        let (method, target) = (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string());
        let (mut key, mut ctype, mut len) = (None, None, 0usize);
        loop {
            let mut hl = String::new();
            if r.read_line(&mut hl).unwrap_or(0) == 0 {
                return;
            }
            let hl = hl.trim_end();
            if hl.is_empty() {
                break;
            }
            if let Some((k, v)) = hl.split_once(':') {
                match k.trim().to_ascii_lowercase().as_str() {
                    "x-mbx-apikey" => key = Some(v.trim().to_string()),
                    "content-type" => ctype = Some(v.trim().to_string()),
                    "content-length" => len = v.trim().parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
        let mut body = vec![0u8; len];
        let _ = r.read_exact(&mut body);
        let got = Got { method, target, key, ctype, body };
        log.lock().unwrap().push(got.clone());
        let mut act = h(&got);
        loop {
            match act {
                Act::Close => {
                    let _ = w.shutdown(std::net::Shutdown::Both);
                    return;
                }
                Act::Hang(sec, next) => {
                    std::thread::sleep(std::time::Duration::from_secs_f64(sec));
                    act = *next;
                }
                Act::Send { status, ctype, body, headers } => {
                    let mut head = format!("HTTP/1.1 {status} X\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n", body.len());
                    for (k, v) in headers {
                        head.push_str(&format!("{k}: {v}\r\n"));
                    }
                    head.push_str("\r\n");
                    if w.write_all(head.as_bytes()).and_then(|_| w.write_all(body.as_bytes())).is_err() {
                        return; // 손님이 먼저 끊었다(시간초과)
                    }
                    break;
                }
            }
        }
    }
}
