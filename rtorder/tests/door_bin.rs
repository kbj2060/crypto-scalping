//! 루프백 HTTP 문(2026-11-21 · rtorder 따로 둔 프로세스):
//! ① 진짜 실행 파일(기본 빌드 = 그림자·키 없음)을 127.0.0.1 가짜 거래소와 띄워 route 정답지(원본 aiohttp 앱 바이트) 중 바깥을 안 부르는 사례를 루프백 HTTP 로 → 같은 바이트 ·
//!    접수 기록 줄(at·ms·ua·ip 머리 우선순위) · /internal/acct · 404 · 거래소 쓰기 0 · SIGTERM 정상 종료 · 루프백 아닌 주소 = 시작 거절
//! ② 클라이언트가 죽고(반쯤 보낸 요청·끊김·쓰레기) 다시 붙어도 감시 틱은 그대로 돈다
//! ③ ?fresh=1 = 1초 안 받은 값 재사용(거래소 한 번) · 1초 뒤 다시 · 캐시가 감시와 한 곳
include!("common/server.rs");
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::*;
use rtorder::route::Front;
use rtorder::runner::DeskCell;
use rtorder::userstream::Hub;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/route_fixture.json")).unwrap()).unwrap()
}

/// GET 만 받는 가짜 거래소(시계 · 나머지는 빈 목록) -- 쓰기가 오면 적어 둔다.
fn exchange() -> (Srv, Arc<Mutex<Vec<String>>>) {
    let log = Arc::new(Mutex::new(vec![]));
    let l2 = log.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        l2.lock().unwrap().push(format!("{} {}", g.method, g.target.split('?').next().unwrap()));
        let body = if g.target.starts_with("/fapi/v1/time") { json!({"serverTime": 1_800_000_000_000i64}) } else { json!([]) };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    (srv, log)
}

struct Bin {
    guard: Option<Guard>,
    child: std::process::Child,
    addr: String,
    out: Arc<Mutex<Vec<String>>>,
}
/// 시험이 중간에 실패해도 실행 파일을 남기지 않는다.
struct Guard(u32);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = std::process::Command::new("kill").args(["-KILL", &self.0.to_string()]).status();
    }
}
fn start(dir: &Path, base: &str, exec: bool, listen: &str) -> Bin {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    start_in(dir, base, exec, listen)
}
fn start_in(dir: &Path, base: &str, exec: bool, listen: &str) -> Bin {
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_rtorder"));
    c.env_clear().env("PATH", std::env::var("PATH").unwrap_or_default()).env("RTORDER_DIR", dir).env("RTORDER_BASE", base)
        .env("RTORDER_PERIOD", "0.05").env("RTORDER_LISTEN", listen).env("RTORDER_STREAM", "1");
    if exec {
        c.env("DASHBOARD_MANUAL_EXEC_ENABLED", "1");
    }
    c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null());
    let mut child = c.spawn().unwrap();
    let guard = Some(Guard(child.id()));
    let so = child.stdout.take().unwrap();
    let out = Arc::new(Mutex::new(vec![]));
    let (o2, (tx, rx)) = (out.clone(), std::sync::mpsc::channel());
    std::thread::spawn(move || {
        for l in BufReader::new(so).lines().map_while(Result::ok) {
            if let Some(a) = l.strip_prefix("rtorder 문 http://") {
                let _ = tx.send(a.to_string());
            }
            o2.lock().unwrap().push(l);
        }
    });
    let addr = rx.recv_timeout(std::time::Duration::from_secs(20)).expect("문 주소 줄");
    assert!(addr.starts_with("127.0.0.1:"), "{addr}");
    Bin { guard, child, addr, out }
}
fn stop(mut b: Bin) -> (Option<i32>, Vec<String>) {
    std::mem::forget(b.guard.take()); // 정상 멈춤 -- 끝난 PID 를 다시 죽이지 않는다
    std::process::Command::new("kill").args(["-TERM", &b.child.id().to_string()]).status().unwrap();
    let mut child = b.child;
    let t0 = std::time::Instant::now();
    let code = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s.code();
        }
        if t0.elapsed().as_secs() > 10 {
            let _ = child.kill();
            panic!("SIGTERM 뒤 10초 안에 안 멈췄다");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    std::thread::sleep(std::time::Duration::from_millis(100));
    (code, b.out.lock().unwrap().clone())
}

fn enc(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}
/// HTTP/1.1 요청 하나(Connection: close) → (상태, 머리, 본문).
fn http(addr: &str, method: &str, path: &str, pairs: &[(String, String)], extra: &[(&str, &str)]) -> (u16, Vec<(String, String)>, String) {
    let q: Vec<String> = pairs.iter().map(|(k, v)| format!("{}={}", enc(k), enc(v))).collect();
    let target = if q.is_empty() { path.to_string() } else { format!("{path}?{}", q.join("&")) };
    let mut s = TcpStream::connect(addr).unwrap();
    let mut req = format!("{method} {target} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
    for (k, v) in extra {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = vec![];
    s.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8(raw).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    let headers = lines.filter_map(|l| l.split_once(": ")).map(|(k, v)| (k.to_ascii_lowercase(), v.to_string())).collect();
    (status, headers, body.to_string())
}
fn hdr(h: &[(String, String)], k: &str) -> Option<String> {
    h.iter().find(|x| x.0 == k).map(|x| x.1.clone())
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

/// 정답지 사례 중 바깥(계좌·시세·거래소)을 안 부르고 칸·무장 파일·잠금·스트림 입력이 없는 것 -- 빈 상태의 진짜 실행 파일과 같은 조건.
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

#[test]
fn binary_door_serves_route_fixture_bytes_logs_and_stops() {
    let fx = fixture();
    let cases = pure_cases(&fx);
    let (srv, xlog) = exchange();
    let root = std::env::temp_dir().join(format!("rtorder_door_{}", std::process::id()));
    let (mut n, mut bad) = (0, vec![]);
    for exec in [true, false] {
        let dir: PathBuf = root.join(if exec { "on" } else { "off" });
        let b = start(&dir, &srv.base, exec, "127.0.0.1:0");
        // 스위치를 켠 기동은 시작 정리(wait_cleanup)가 eth 대기 칸에 «재시작으로 취소»를 남긴다(그림자 = 정리 «못 함» 오류 -- 원본도 오류면 같은 칸을 쓴다) --
        // 대기 칸을 본문에 싣는 사례(빈 칸 {"phase": "idle"} 를 기대)는 켠 쪽에서 뺀다.
        let mine: Vec<&Value> = cases.iter().filter(|c| c["in"].get("exec").and_then(Value::as_bool).unwrap_or(true) == exec)
            .filter(|c| !(exec && c["body"].as_str().unwrap().contains("\"wait\": {"))).collect();
        // 상태 조회는 대기 칸을 만든다(원본 setdefault) -- 칸을 보는 다른 핸들러보다 뒤에
        let (mut later, mut first): (Vec<&Value>, Vec<&Value>) = mine.into_iter().partition(|c| c["in"]["path"] == "/api/manual-entry/status");
        first.append(&mut later);
        let mut want_log: Vec<Value> = vec![];
        for (i, c) in first.iter().enumerate() {
            let inp = &c["in"];
            let pairs: Vec<(String, String)> = inp["pairs"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
            // 접수 기록 ip 머리 우선순위: CF-Connecting-IP → X-Forwarded-For → 접속 주소
            let extra: Vec<(&str, &str)> = match i % 3 {
                0 => vec![("User-Agent", "rtorder-test/1"), ("CF-Connecting-IP", "198.51.100.7"), ("X-Forwarded-For", "203.0.113.9")],
                1 => vec![("User-Agent", "rtorder-test/1"), ("X-Forwarded-For", "203.0.113.9")],
                _ => vec![],
            };
            let (st, h, body) = http(&b.addr, inp["method"].as_str().unwrap(), inp["path"].as_str().unwrap(), &pairs, &extra);
            let name = c["name"].as_str().unwrap();
            let same = st as u64 == c["status"].as_u64().unwrap() && hdr(&h, "content-type").as_deref() == c["content_type"].as_str()
                && hdr(&h, "cache-control").as_deref() == c["cache_control"].as_str() && hdr(&h, "allow").as_deref() == c["allow"].as_str()
                && ms0(&body) == ms0(c["body"].as_str().unwrap());
            if !same {
                bad.push(format!("{name}: {st} {body:.200} / py {} {:.200}", c["status"], c["body"].as_str().unwrap()));
            }
            for row in c["log"].as_array().unwrap() {
                let ip = match i % 3 { 0 => "198.51.100.7", 1 => "203.0.113.9", _ => "127.0.0.1" };
                let ua = if i % 3 == 2 { "" } else { "rtorder-test/1" };
                want_log.push(json!({"row": row, "ua": ua, "ip": ip}));
            }
            n += 1;
        }
        // 한글 쿼리 = 기록 줄에 그대로(ensure_ascii=False) -- 원본 dict(request.query)
        let (st, _, _) = http(&b.addr, "POST", "/api/manual-entry/cancel", &[("memo".into(), "한글 메모".into()), ("confirm".into(), "0".into())], &[]);
        assert_eq!(st, 400);
        want_log.push(json!({"korean": true}));
        // 내부 주소 · 모르는 주소
        let (st, _, body) = http(&b.addr, "GET", "/internal/acct", &[], &[]);
        assert_eq!(st, 200);
        assert!(body.starts_with(r#"{"seq": 0, "user_stream": {"connected": false, "last_event_at": null, "error": null}, "feed": "#), "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["feed"], json!({"connected": false, "connects": 0, "last_rx_age_s": null, "stale": true, "symbols": {}}), "허브 없음 = 한 번도 못 받음(stale)");
        let (st, h, body) = http(&b.addr, "GET", "/api/nope", &[], &[]);
        assert_eq!((st, body.as_str(), hdr(&h, "content-type").as_deref()), (404, "404: Not Found", Some("text/plain; charset=utf-8")));
        let (st, _, body) = http(&b.addr, "GET", "/api/binance-account", &[], &[]);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(st, 200);
        assert_eq!(v["user_stream"], json!({"connected": false, "error": null, "seq": 0, "last_event_age_s": null}), "{body}");
        let (code, out) = stop(b);
        assert_eq!(code, Some(0), "{out:?}");
        // 접수 기록 = 원본 칸 순서 · path·query·status·error 는 정답지 · at = KST 밀리초 · ua·ip = 넘겨받은 머리
        let rows: Vec<String> = std::fs::read_to_string(dir.join("order_requests.jsonl")).unwrap_or_default().lines().map(String::from).collect();
        assert_eq!(rows.len(), want_log.len(), "기록 줄 수");
        for (line, w) in rows.iter().zip(&want_log) {
            let r: Value = serde_json::from_str(line).unwrap();
            if w.get("korean").is_some() {
                assert!(line.contains("\"memo\": \"한글 메모\""), "{line}");
                continue;
            }
            let keys: Vec<&String> = r.as_object().unwrap().keys().collect();
            assert_eq!(keys, ["at", "path", "query", "status", "error", "ms", "ua", "ip"], "{line}");
            for k in ["path", "query", "status", "error"] {
                assert_eq!(r[k], w["row"][k], "{line}");
            }
            let at = r["at"].as_str().unwrap();
            let kst_hour = ((std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() + 9 * 3600) % 86400) / 3600;
            let h: u64 = at[11..13].parse().unwrap();
            assert!(h == kst_hour || (h + 1) % 24 == kst_hour, "KST 시각이어야 한다 {at}");
            assert!(at.len() == 29 && at.ends_with("+09:00") && at.as_bytes()[19] == b'.', "{at}");
            assert!(r["ms"].as_i64().is_some());
            assert_eq!((&r["ua"], &r["ip"]), (&w["ua"], &w["ip"]), "{line}");
            assert!(!line.contains("\\u"), "한글은 그대로(ensure_ascii=False) {line}");
        }
        // 자기 접수 기록이 있는 폴더로 재시작 = 받는다(표식 .rtorder) -- 파이썬 폴더 거절은 표식이 없을 때만
        if exec {
            let (code, _) = stop(start_in(&dir, &srv.base, exec, "127.0.0.1:0"));
            assert_eq!(code, Some(0));
        }
    }
    assert!(bad.is_empty(), "다름 {}건:\n{}", bad.len(), bad.join("\n"));
    assert!(n >= 40, "대조 사례 수 {n}");
    println!("실행 파일 대조 사례 {n}");
    let xl = xlog.lock().unwrap().clone();
    assert!(xl.iter().all(|l| l.starts_with("GET")), "거래소로 쓰기가 나갔다 {xl:?}");
    // 루프백이 아니면 시작 거절
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_rtorder")).env_clear().env("RTORDER_DIR", root.join("x")).env("RTORDER_LISTEN", "0.0.0.0:0")
        .stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).spawn().unwrap();
    let t0 = std::time::Instant::now();
    let code = loop {
        if let Some(s) = c.try_wait().unwrap() {
            break s.code();
        }
        if t0.elapsed().as_secs() > 10 {
            let _ = c.kill();
            let _ = c.wait();
            break None; // 떠 버렸다 = 거절하지 않았다
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    assert_eq!(code, Some(2), "루프백 아닌 주소는 시작 거절");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn watch_ticks_keep_running_while_clients_die() {
    let (srv, _x) = exchange();
    let dir = std::env::temp_dir().join(format!("rtorder_door_ticks_{}", std::process::id()));
    let b = start(&dir, &srv.base, true, "127.0.0.1:0");
    let t0 = std::time::Instant::now();
    // 켠 기동이 시작 정리 «재시작으로 취소»를 eth 칸에 남긴 것도 함께 본다(원본 wait_cleanup 오류 갈래)
    let (_, _, body) = http(&b.addr, "GET", "/api/manual-entry/status", &[], &[]);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["wait"]["phase"], "restart_cancelled", "{body}");
    assert_eq!(v["wait"]["errors"], json!(["그림자 모드: 주문을 내지 않았습니다"]));
    let mut held = vec![];
    while t0.elapsed().as_secs_f64() < 1.5 {
        // 반쯤 보낸 요청을 쥔 채(응답 서버가 멈춘 꼴) · 붙자마자 끊기 · 쓰레기
        let mut s = TcpStream::connect(&b.addr).unwrap();
        s.write_all(b"POST /api/manual-entry/cancel?confirm=1 HTTP/1.1\r\nHost: x\r\n").unwrap();
        held.push(s);
        drop(TcpStream::connect(&b.addr).unwrap());
        let mut g = TcpStream::connect(&b.addr).unwrap();
        let _ = g.write_all(b"\x00\x01garbage\r\n\r\n");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // 죽었다 살아난 클라이언트 -- 그대로 답한다
    let (st, _, _) = http(&b.addr, "GET", "/internal/acct", &[], &[]);
    assert_eq!(st, 200);
    drop(held);
    let (code, out) = stop(b);
    assert_eq!(code, Some(0));
    let last = out.iter().rev().find(|l| l.starts_with("rtorder 종료")).unwrap().clone();
    let ticks: u64 = last.split("틱 ").nth(1).unwrap().split(' ').next().unwrap().parse().unwrap();
    assert!(ticks >= 10, "1.5초 · 주기 0.05초인데 틱 {ticks} ({last})");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fresh_reuses_within_one_second_and_shares_watch_cache() {
    use rtorder::account::SharedAccount;
    use rtorder::assemble::DeskIo;
    let fx: Value = serde_json::from_str(include_str!("fixtures/desk_fixture.json")).unwrap();
    let routes = fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"]["routes"].clone();
    let n = Arc::new(Mutex::new(0usize));
    let n2 = n.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        let path = g.target.split('?').next().unwrap().to_string();
        if path == "/fapi/v2/account" {
            *n2.lock().unwrap() += 1;
        }
        let body = if path == "/fapi/v1/time" { json!({"serverTime": (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()) as i64}) }
                   else { routes.get(&path).cloned().unwrap_or(json!([])) };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    let bx = || rtorder::http::Binance::new(&srv.base, "test_key", "test_secret", std::time::Duration::from_secs(3), std::env::temp_dir().join("rtorder_door_ban"));
    let acct = Arc::new(SharedAccount::new("ETHUSDC"));
    let mut sh = bx();
    sh.shadow_on();
    let ops = rtorder::app::Ops { bx: sh, feed: Default::default(), hub: Default::default(), stop: Default::default(), stopped: Default::default(),
                                  would: Arc::new(Mutex::new(vec![])), alerts: vec![], telegram: None, front: Default::default(), rest_feed: None };
    let mut io = rtorder::door::DoorIo { ops, rest: bx(), acct: acct.clone(), stream: None };
    let a = io.account_screen(true).await.unwrap();
    assert_eq!(a["ok"], json!(true));
    io.account_screen(true).await.unwrap();
    assert_eq!(*n.lock().unwrap(), 1, "1초 안 둘째 ?fresh=1 = 받은 값 재사용(거래소 한 번)");
    io.account(false).await.unwrap();
    assert_eq!(*n.lock().unwrap(), 1, "감시·화면 swr 캐시가 같은 값(10초)");
    tokio::time::sleep(std::time::Duration::from_millis(1100)).await;
    io.account_screen(true).await.unwrap();
    assert_eq!(*n.lock().unwrap(), 2, "1초 지나면 다시 읽는다");
    io.account(true).await.unwrap();
    assert_eq!(*n.lock().unwrap(), 3, "접수 조립의 새 계좌(produce_account)는 재사용 안 함(원본)");
    assert!(acct.cache.lock().unwrap().is_some());
}

/// 2026-11-23 계좌 카드 빔: 화면 응답에만 원장·0시 순자산·income 이 붙는다(감시·주문이 쓰는 캐시 계좌엔 안 붙음) · 화면 몫이 없으면(None) 그대로.
#[tokio::test]
async fn screen_fields_only_on_screen_response() {
    use rtorder::account::SharedAccount;
    use rtorder::assemble::DeskIo;
    let fx: Value = serde_json::from_str(include_str!("fixtures/desk_fixture.json")).unwrap();
    let routes = fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"]["routes"].clone();
    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    let srv = serve(Arc::new(move |g: &Got| {
        let path = g.target.split('?').next().unwrap().to_string();
        let body = match path.as_str() {
            "/fapi/v1/time" => json!({"serverTime": now_ms}),
            "/fapi/v1/income" => json!([{"tranId": 9, "incomeType": "REALIZED_PNL", "asset": "USDC", "income": "4.5", "time": now_ms - 1000}]),
            _ => routes.get(&path).cloned().unwrap_or(json!([])),
        };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    let dir = std::env::temp_dir().join(format!("rtorder_screen_door_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("account_round_trips.jsonl"),
                   "{\"symbol\": \"ETHUSDC\", \"side\": \"LONG\", \"entry_time\": 1, \"exit_time\": 2, \"net_pnl\": 3.0, \"peak_notional\": null}\n").unwrap();
    // 첫 기동은 88일을 7일 창 8개씩 메운다(원본) -- 따라잡은 상태(1시간 전 커서)에서 시작해야 이번 응답에 income 이 실린다
    std::fs::write(dir.join("account_income_daily.json"), json!({"from": "2026-09-01", "cursor": now_ms - 3_600_000, "days": {}, "recent": {}}).to_string()).unwrap();
    let bx = || rtorder::http::Binance::new(&srv.base, "test_key", "test_secret", std::time::Duration::from_secs(3), std::env::temp_dir().join("rtorder_door_ban"));
    let mk_io = |acct: Arc<SharedAccount>| {
        let mut sh = bx();
        sh.shadow_on();
        let ops = rtorder::app::Ops { bx: sh, feed: Default::default(), hub: Default::default(), stop: Default::default(), stopped: Default::default(),
                                      would: Arc::new(Mutex::new(vec![])), alerts: vec![], telegram: None, front: Default::default(), rest_feed: None };
        rtorder::door::DoorIo { ops, rest: bx(), acct, stream: None }
    };
    let mut plain = mk_io(Arc::new(SharedAccount::new("ETHUSDC")));
    let a = plain.account_screen(true).await.unwrap();
    assert!(a.get("ledger").is_none() && a.get("income_days").is_none(), "화면 몫 없음 = 그대로");
    let mut acct = SharedAccount::new("ETHUSDC");
    acct.screen = Some(tokio::sync::Mutex::new(rtorder::screen::Screen::new(dir.clone())));
    let acct = Arc::new(acct);
    let mut io = mk_io(acct.clone());
    let s = io.account_screen(true).await.unwrap();
    assert_eq!(s["ledger"][0]["net_pnl"], json!(3.0));
    assert_eq!(s["ledger"][0]["trend"], Value::Null, "일봉 없음(가짜 서버 klines = []) → 추세 모름");
    assert!(s["day_start"]["date"].is_string() && dir.join("account_day_start.json").exists());
    let today = s["day_start"]["date"].as_str().unwrap().to_string();
    assert_eq!(s["income_days"][&today]["realized"], json!(4.5), "{s}");
    assert!(dir.join("account_income_daily.json").exists());
    let cached = io.account(false).await.unwrap();
    assert!(cached.get("ledger").is_none() && cached.get("income_days").is_none(), "감시·주문 계좌엔 안 붙는다");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kst_milliseconds_match_python_isoformat() {
    // python: datetime.fromtimestamp(t, ZoneInfo("Asia/Seoul")).isoformat(timespec="milliseconds")
    for (t, want) in [(1_800_000_000.123, "2027-01-15T17:00:00.123+09:00"), (1_800_000_000.9995, "2027-01-15T17:00:00.999+09:00"), (1_799_999_999.0, "2027-01-15T16:59:59.000+09:00")] {
        assert_eq!(rtorder::door::iso_kst_ms(t), want);
    }
}
