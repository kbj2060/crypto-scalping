//! 넷째 조각(2026-11-22): 주문·계좌 주소 14개를 rtorder 문(따로 뜨는 프로세스 · 127.0.0.1:8791 · rtorder/src/door.rs)으로 넘긴다.
//! 넘기는 규칙 = rtorder/SHADOW.md §7 «응답 서버가 프록시할 때 넘겨야 하는 것»: 메서드·경로·쿼리 원문 · User-Agent · CF-Connecting-IP ·
//! X-Forwarded-For(화면이 안 보냈으면 접속 주소를 넣어) → 받은 상태·Content-Type·Cache-Control·Allow·본문을 그대로. ETag·압축은 rtserve 미들웨어 몫
//! (원본도 바깥 미들웨어). 접수 기록(order_requests.jsonl)은 rtorder 가 직접 적는다 -- rtserve 는 넘긴 요청을 따로 적지 않는다.
//! 문이 닫혀 있으면 502 `{"ok": false, "error": "rtorder_unavailable"}`(원본 주문 핸들러의 오류 꼴 · 다른 주소는 영향 없음).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use serde_json::json;

use crate::{Config, Handler, Req, Resp};

/// 원본 등록부(server.py 5004~5017) 그대로: (주소, 메서드)
pub const ROUTES: [(&str, &str); 14] = [
    ("/api/binance-account", "GET"),
    ("/api/manual-entry/preview", "GET"),
    ("/api/manual-entry/submit", "POST"),
    ("/api/manual-entry/status", "GET"),
    ("/api/manual-entry/wait-action", "POST"),
    ("/api/manual-entry/cancel", "POST"),
    ("/api/manual-entry/limit-amend", "POST"),
    ("/api/manual-bracket/disarm", "POST"),
    ("/api/manual-bracket/rearm", "POST"),
    ("/api/manual-bracket/tp", "POST"),
    ("/api/manual-bracket/tp-legs", "POST"),
    ("/api/manual-bracket/add", "POST"),
    ("/api/manual-bracket/modify", "POST"),
    ("/api/manual-exit/submit", "POST"),
];

/// 시간 제한: rtorder 는 칸을 5초까지 기다린다(route::DESK_WAIT_SEC) -- 그보다 길게
const TIMEOUT: Duration = Duration::from_secs(15);

pub fn routes(p: &str) -> Vec<(&'static str, Handler)> {
    ROUTES.iter().filter(|(r, _)| *r == p).map(|(_, m)| (*m, proxy as Handler)).collect()
}

pub fn is_proxied(p: &str) -> bool {
    ROUTES.iter().any(|(r, _)| *r == p)
}

fn unavailable() -> Resp {
    let mut r = Resp::json(&json!({"ok": false, "error": "rtorder_unavailable"}));
    r.status = 502;
    r
}

/// 날 HTTP/1.1 요청 하나(Connection: close) → 상태 줄·머리·본문
fn send(addr: SocketAddr, raw: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(2))?;
    s.set_read_timeout(Some(TIMEOUT))?;
    s.set_write_timeout(Some(TIMEOUT))?;
    s.write_all(raw)?;
    let mut out = Vec::new();
    s.read_to_end(&mut out)?;
    Ok(out)
}

/// 응답 바이트 → (상태, 상태 글자, 머리, 본문) · 청크 전송·Content-Length 둘 다
fn parse(raw: &[u8]) -> Option<(u16, String, Vec<(String, String)>, Vec<u8>)> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    let mut lines = head.split("\r\n");
    let mut first = lines.next()?.splitn(3, ' ');
    let _ = first.next()?;
    let status: u16 = first.next()?.parse().ok()?;
    let reason = first.next().unwrap_or("").to_string();
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(": ").map(|(k, v)| (k.to_ascii_lowercase(), v.to_string()))).collect();
    let mut body = raw[split + 4..].to_vec();
    if headers.iter().any(|(k, v)| k == "transfer-encoding" && v.eq_ignore_ascii_case("chunked")) {
        let (mut out, mut rest) = (Vec::new(), &body[..]);
        loop {
            let nl = rest.windows(2).position(|w| w == b"\r\n")?;
            let n = usize::from_str_radix(std::str::from_utf8(&rest[..nl]).ok()?.trim(), 16).ok()?;
            if n == 0 {
                break;
            }
            out.extend_from_slice(rest.get(nl + 2..nl + 2 + n)?);
            rest = rest.get(nl + 4 + n..)?;
        }
        body = out;
    } else if let Some(n) = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse::<usize>().ok()) {
        body.truncate(n);
    }
    Some((status, reason, headers, body))
}

/// 14개 주소의 핸들러: 그대로 넘기고 그대로 돌려준다
fn proxy(cfg: &Config, req: &Req) -> Resp {
    let Some(addr) = cfg.rtorder else { return unavailable() };
    let target = if req.raw_query.is_empty() { req.path.clone() } else { format!("{}?{}", req.path, req.raw_query) };
    let mut head = format!("{} {target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n", req.method);
    for name in ["user-agent", "cf-connecting-ip", "content-type"] {
        if let Some(v) = req.header(name) {
            head.push_str(&format!("{name}: {v}\r\n"));
        }
    }
    let xff = req.header("x-forwarded-for").filter(|v| !v.is_empty()).map_or_else(|| req.peer.clone(), str::to_string);
    head.push_str(&format!("x-forwarded-for: {xff}\r\n"));
    head.push_str(&format!("content-length: {}\r\n\r\n", req.body.len()));
    let mut raw = head.into_bytes();
    raw.extend_from_slice(&req.body);
    let Some((status, reason, headers, body)) = send(addr, &raw).ok().and_then(|b| parse(&b)) else { return unavailable() };
    let mut r = Resp { status, body: Some(body), upstream: true, ..Default::default() };
    if !reason.is_empty() {
        r.reason = Some(reason);
    }
    for name in ["Content-Type", "Cache-Control", "Allow"] {
        if let Some((_, v)) = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
            r.headers.push((name.into(), v.clone()));
        }
    }
    r
}

/// rtorder `GET /internal/acct` 의 seq(원본 order_api.acct_seq()) -- 못 읽으면 None(부르는 쪽이 마지막 값을 쓴다)
pub fn acct_seq(cfg: &Config) -> Option<i64> {
    let addr = cfg.rtorder?;
    let raw = format!("GET /internal/acct HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_millis(500)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(1))).ok()?;
    s.write_all(raw.as_bytes()).ok()?;
    let mut out = Vec::new();
    s.read_to_end(&mut out).ok()?;
    let (status, _, _, body) = parse(&out)?;
    if status != 200 {
        return None;
    }
    serde_json::from_slice::<serde_json::Value>(&body).ok()?["seq"].as_i64()
}
