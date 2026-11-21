//! 시험 공용: 127.0.0.1 가짜 웹소켓(표준 라이브러리만 -- 시험 의존성을 더하면 DuckDB 를 다시 빌드한다).
#![allow(dead_code)]
use std::io::{Read, Write};
use std::net::TcpStream;

pub fn sha1(msg: &[u8]) -> [u8; 20] {
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

pub fn b64(b: &[u8]) -> String {
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
pub fn read_frame(s: &mut TcpStream) -> Option<(u8, Vec<u8>)> {
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

pub fn send(s: &mut TcpStream, op: u8, p: &[u8]) {
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

/// 손 맞추기(요청 줄을 돌려준다). None = 끊김.
pub fn handshake(s: &mut TcpStream) -> Option<String> {
    let mut req = vec![];
    let mut b = [0u8; 1];
    while !req.ends_with(b"\r\n\r\n") {
        s.read_exact(&mut b).ok()?;
        req.push(b[0]);
    }
    let req = String::from_utf8_lossy(&req).to_string();
    let key = req.lines().find_map(|l| l.to_lowercase().starts_with("sec-websocket-key:").then(|| l[18..].trim().to_string()))?;
    let acc = b64(&sha1(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes()));
    write!(s, "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {acc}\r\n\r\n").ok()?;
    Some(req.lines().next().unwrap_or("").to_string())
}
