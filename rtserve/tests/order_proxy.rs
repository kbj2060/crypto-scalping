//! 주문·계좌 14개 넘기기(2026-11-22): 진짜 rtorder 실행 파일(기본 빌드 = 그림자·키 없음 · ../rtorder/target/release/rtorder)을 127.0.0.1 가짜 거래소와 띄우고
//! rtserve 를 거쳐 route 정답지(rtorder/tests/fixtures/route_fixture.json = 원본 aiohttp 앱 바이트) 중 바깥을 안 부르는 사례를 묻는다 --
//! 상태·Content-Type·Cache-Control·Allow·본문이 같아야 한다(고르는 판정 = rtorder/tests/door_bin.rs pure_cases 그대로).
//! 더: 접수 기록 ip 머리(화면이 X-Forwarded-For 를 안 보내면 rtserve 가 접속 주소를 넣는다) · /internal/acct → acct_seq · 문이 닫히면 502 · 다른 주소는 영향 없음.

mod common;

#[path = "../../rtorder/tests/common/desk_io.rs"]
#[allow(dead_code, unused_imports)]
mod desk_io;

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

use common::{request_body, start_with};
use desk_io::*;
use rtorder::fakesrv::{local_only, serve, Act, Got};
use rtorder::route::Front;
use rtorder::runner::DeskCell;
use rtorder::userstream::Hub;
use serde_json::{json, Value};

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../rtorder/tests/fixtures/route_fixture.json")).unwrap()).unwrap()
}

/// door_bin.rs pure_cases 와 같은 판정: 칸·무장 파일·작업·잠금·스트림 입력이 없고, 진짜로 돌려 바깥(계좌·시세)을 한 번도 안 부르는 사례
fn pure_cases(fx: &Value) -> Vec<Value> {
    fx["cases"].as_array().unwrap().iter().filter(|c| {
        let inp = &c["in"];
        if ["slots0", "file0", "tasks", "locked", "user_stream", "t"].iter().any(|k| inp.get(*k).is_some()) || inp["path"] == "/api/binance-account" {
            return false;
        }
        let (desk, _d) = desk_from(inp);
        let mut io = FakeDeskIo::new(inp, desk.w.dir.clone());
        let mut q = FakeQuotes::new(inp);
        let pairs: Vec<(String, String)> = inp["pairs"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
        let cell = DeskCell::new(desk);
        let _ = block_on(Front::default().call(&cell, &Hub::default(), &mut q, &mut io, inp["method"].as_str().unwrap(), inp["path"].as_str().unwrap(), &pairs));
        io.calls.is_empty() && q.calls.is_empty()
    }).cloned().collect()
}

fn bin() -> std::path::PathBuf {
    let p = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../rtorder/target/release/rtorder")).to_path_buf();
    assert!(p.exists(), "{} 가 없다 -- rtorder 를 먼저 `cargo build --release` 한다", p.display());
    p
}

struct Rt {
    child: std::process::Child,
    addr: std::net::SocketAddr,
}
impl Drop for Rt {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_rtorder(dir: &Path, base: &str, exec: bool) -> Rt {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut c = std::process::Command::new(bin());
    c.env_clear().env("PATH", std::env::var("PATH").unwrap_or_default()).env("RTORDER_DIR", dir).env("RTORDER_BASE", base)
        .env("RTORDER_PERIOD", "0.05").env("RTORDER_LISTEN", "127.0.0.1:0").env("RTORDER_STREAM", "1");
    if exec {
        c.env("DASHBOARD_MANUAL_EXEC_ENABLED", "1");
    }
    c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    let mut child = c.spawn().unwrap();
    let so = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for l in BufReader::new(so).lines().map_while(Result::ok) {
            if let Some(a) = l.strip_prefix("rtorder 문 http://") {
                let _ = tx.send(a.to_string());
            }
        }
    });
    let addr = rx.recv_timeout(std::time::Duration::from_secs(20)).expect("문 주소 줄").parse().unwrap();
    Rt { child, addr }
}

fn enc(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

fn ms0(s: &str) -> String {
    match s.find("\"server_ms\": ") {
        Some(i) => {
            let j = i + 13;
            let end = s[j..].find(|c: char| !c.is_ascii_digit()).map_or(s.len(), |k| j + k);
            format!("{}0{}", &s[..j], &s[end..])
        }
        None => s.to_string(),
    }
}

fn hdr(h: &[(String, String)], k: &str) -> Option<String> {
    h.iter().find(|x| x.0 == k).map(|x| x.1.clone())
}

#[test]
fn proxied_orders_match_route_fixture() {
    let fx = fixture();
    let cases = pure_cases(&fx);
    let xlog = Arc::new(Mutex::new(Vec::<String>::new()));
    let xl = xlog.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        xl.lock().unwrap().push(format!("{} {}", g.method, g.target.split('?').next().unwrap()));
        let body = if g.target.starts_with("/fapi/v1/time") { json!({"serverTime": 1_800_000_000_000i64}) } else { json!([]) };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    let root = std::env::temp_dir().join(format!("rtserve_orders_{}", std::process::id()));
    let (mut n, mut bad) = (0, Vec::new());
    for exec in [true, false] {
        let dir = root.join(if exec { "on" } else { "off" });
        let rt = start_rtorder(&dir, &srv.base, exec);
        let a = rt.addr;
        let (addr, _r) = start_with(&Default::default(), 1_700_000_000_000_000_000, 1.7e9, "eth", &format!("orders_{exec}"), move |c| c.rtorder = Some(a));
        std::fs::create_dir_all(_r.join("data/live")).unwrap();
        let before = std::fs::read_to_string(_r.join("data/live/order_requests.jsonl")).unwrap_or_default().lines().count();
        let mine: Vec<&Value> = cases.iter().filter(|c| c["in"].get("exec").and_then(Value::as_bool).unwrap_or(true) == exec)
            .filter(|c| !(exec && c["body"].as_str().unwrap().contains("\"wait\": {"))).collect();
        let (mut later, mut first): (Vec<&Value>, Vec<&Value>) = mine.into_iter().partition(|c| c["in"]["path"] == "/api/manual-entry/status");
        first.append(&mut later);
        for (i, c) in first.iter().enumerate() {
            let inp = &c["in"];
            let q: Vec<String> = inp["pairs"].as_array().unwrap().iter().map(|p| format!("{}={}", enc(p[0].as_str().unwrap()), enc(p[1].as_str().unwrap()))).collect();
            let path = if q.is_empty() { inp["path"].as_str().unwrap().to_string() } else { format!("{}?{}", inp["path"].as_str().unwrap(), q.join("&")) };
            let extra: Vec<(String, String)> = match i % 3 {
                0 => vec![("User-Agent".into(), "rtserve-test/1".into()), ("CF-Connecting-IP".into(), "198.51.100.7".into()), ("X-Forwarded-For".into(), "203.0.113.9".into())],
                1 => vec![("User-Agent".into(), "rtserve-test/1".into()), ("X-Forwarded-For".into(), "203.0.113.9".into())],
                _ => vec![],
            };
            let g = request_body(addr, inp["method"].as_str().unwrap(), &path, &extra, b"");
            let body = String::from_utf8_lossy(&g.body).into_owned();
            // 원본은 바깥 미들웨어가 GET·200·JSON 에 Cache-Control 기본 no-cache 를 붙인다 -- 정답지(핸들러만의 바이트)에 없으면 그 값이 맞다
            let want_cc = c["cache_control"].as_str().map(str::to_string).or_else(|| {
                (inp["method"] == "GET" && c["status"] == 200 && c["content_type"].as_str().is_some_and(|t| t.starts_with("application/json"))).then(|| "no-cache".to_string())
            });
            let same = g.status as u64 == c["status"].as_u64().unwrap() && hdr(&g.headers, "content-type").as_deref() == c["content_type"].as_str()
                && hdr(&g.headers, "cache-control") == want_cc && hdr(&g.headers, "allow").as_deref() == c["allow"].as_str()
                && ms0(&body) == ms0(c["body"].as_str().unwrap());
            if !same {
                bad.push(format!("{}: {} {:?} {:.200} / py {} {:?} {:.200}", c["name"], g.status, g.headers, body, c["status"], c["cache_control"], c["body"].as_str().unwrap()));
            }
            n += 1;
        }
        // 접수 기록: 화면이 아무 머리도 안 보낸 POST → rtserve 가 X-Forwarded-For = 접속 주소(127.0.0.1)를 넣었다 · CF 가 있으면 그것
        let _ = request_body(addr, "POST", "/api/manual-entry/cancel?confirm=0", &[("User-Agent".into(), "ua-a".into())], b"");
        let _ = request_body(addr, "POST", "/api/manual-entry/cancel?confirm=0", &[("CF-Connecting-IP".into(), "198.51.100.8".into())], b"");
        // 접속 주소 127.0.0.2(루프백 /8) 에서 -- rtorder 는 rtserve 의 127.0.0.1 에서 받으므로, 넣은 X-Forwarded-For 만이 127.0.0.2 를 실어 나른다
        {
            use std::io::{Read, Write};
            let sock = tokio::net::TcpSocket::new_v4().unwrap();
            sock.bind("127.0.0.2:0".parse().unwrap()).unwrap();
            let std_s: std::net::TcpStream = {
                let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
                rt.block_on(async { sock.connect(addr).await.unwrap().into_std().unwrap() })
            };
            let mut s = std_s;
            s.set_nonblocking(false).unwrap();
            s.write_all(b"POST /api/manual-entry/cancel?confirm=0 HTTP/1.1\r\nHost: x\r\nUser-Agent: ua-b\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            let mut out = Vec::new();
            let _ = s.read_to_end(&mut out);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let log = std::fs::read_to_string(dir.join("order_requests.jsonl")).unwrap_or_default();
        let rows: Vec<Value> = log.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let last3: Vec<(Value, Value)> = rows[rows.len() - 3..].iter().map(|r| (r["ua"].clone(), r["ip"].clone())).collect();
        assert_eq!(last3, vec![(json!("ua-a"), json!("127.0.0.1")), (json!(""), json!("198.51.100.8")), (json!("ua-b"), json!("127.0.0.2"))], "접수 기록 ua·ip");
        // rtserve 는 넘긴 요청을 따로 적지 않는다(rtserve 쪽 기록 파일 없음)
        let mid = std::fs::read_to_string(_r.join("data/live/order_requests.jsonl")).unwrap_or_default();
        assert!(mid.lines().all(|l| l.contains("\"status\": 405") || l.contains("\"status\": 404")), "넘긴 요청을 rtserve 가 두 번 적었다(rtserve 는 자기가 답한 404·405 만): {mid}");
        // /internal/acct → acct_seq
        let mut cfg = rtserve::Config::new(_r.clone(), vec!["eth".into()], Arc::new(|| 1.7e9));
        cfg.rtorder = Some(a);
        assert_eq!(rtserve::order::acct_seq(&cfg), Some(0));
        // 405 는 rtserve 가 답하고 원본처럼 적는다(POST 만)
        let g = request_body(addr, "POST", "/api/manual-entry/status", &[], b"");
        assert_eq!((g.status, String::from_utf8_lossy(&g.body).into_owned()), (405, "405: Method Not Allowed".to_string()));
        // 문이 닫히면 502 · 다른 주소는 그대로
        drop(rt);
        let g = request_body(addr, "GET", "/api/manual-entry/status", &[], b"");
        assert_eq!((g.status, String::from_utf8_lossy(&g.body).into_owned()), (502, r#"{"ok": false, "error": "rtorder_unavailable"}"#.to_string()));
        assert_eq!(request_body(addr, "GET", "/api/acct-notes", &[], b"").status, 200);
        assert_eq!(rtserve::order::acct_seq(&cfg), None);
        let mine_log = std::fs::read_to_string(_r.join("data/live/order_requests.jsonl")).unwrap_or_default();
        let rows: Vec<&str> = mine_log.lines().collect();
        assert!(rows[before..].iter().all(|l| l.contains("\"status\": 405") || l.contains("\"status\": 404")), "rtserve 는 자기가 답한 404·405 만 적는다: {mine_log}");
        assert!(rows.last().unwrap().contains("/api/manual-entry/status"));
    }
    let writes: Vec<String> = xlog.lock().unwrap().iter().filter(|l| !l.starts_with("GET ")).cloned().collect();
    assert!(writes.is_empty(), "거래소 쓰기: {writes:?}");
    eprintln!("order_proxy: 사례 {n} · 불일치 {}", bad.len());
    assert!(n > 20, "고른 사례가 너무 적다: {n}");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    let _ = std::fs::remove_dir_all(root);
}
