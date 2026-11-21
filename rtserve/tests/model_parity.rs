//! 넷째 조각: 요청 때 계산하는 모델 핸들러 5개(src/model.rs)를 rtserve HTTP 로 -- REST 는 127.0.0.1 가짜 서버가 rtmodel 정답지의 녹화본을 돌려준다(바깥 접속 없음).
//! 정답 = rtmodel 정답지의 원본(파이썬) 값: dir4h·dir-short payload 비트(dirrest_fixture 첫 회차) · vol-forecast HGB·open-rev·dir-signals 본문 글자(signals_fixture).
//! QX4 는 원본 payload 정답지가 이 녹화본(qx4_rest_fixture)에 없어 rtmodel::handlers::vol_forecast_qx4 를 같은 녹화본으로 돌린 값과 바이트 대조(배선 대조 · 계산 비트는 rtmodel/tests/qx4.rs).
//! 요청 열(경로·쿼리 순서)도 원본 녹화본과 같은지 · 밴 파일이 있으면 안 보냄 · «모델 낡음»·모델 없음·지원 밖 코인.
//!     cargo test --release -j 2 --test model_parity -- --nocapture   (rtmodel 정답지가 있어야 한다 -- rtmodel/tests 머리말)

mod common;

use common::{request, start_with};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

type Book = Arc<Mutex<(HashMap<String, VecDeque<Value>>, Vec<String>)>>;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../rtmodel/tests/fixtures")
}

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap_or_else(|e| panic!("{name}: {e} -- rtmodel 정답지"))).unwrap()
}

/// 가짜 REST: 경로?쿼리 → 응답 열(남은 게 하나면 그대로 둔다) · 받은 요청 순서를 적는다 · 모르는 주소 = 404
fn fake(book: Book) -> String {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let Ok(mut s) = s else { continue };
            let mut buf = vec![0u8; 65536];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut buf[n..]) {
                    Ok(0) | Err(_) => break,
                    Ok(k) => n += k,
                }
            }
            let target = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let mut g = book.lock().unwrap();
            g.1.push(target.clone());
            let hit = g.0.get_mut(&target).map(|q| if q.len() > 1 { q.pop_front().unwrap() } else { q[0].clone() });
            drop(g);
            let (code, body) = hit.map_or((404, "[]".to_string()), |v| (200, v.to_string()));
            let _ = s.write_all(format!("HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    format!("http://127.0.0.1:{port}")
}

fn add(book: &Book, target: String, v: Value) {
    book.lock().unwrap().0.entry(target).or_default().push_back(v);
}

fn qs(params: &Value) -> String {
    params.as_array().unwrap().iter().map(|p| format!("{}={}", p[0].as_str().unwrap(), p[1].as_str().unwrap())).collect::<Vec<_>>().join("&")
}

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

/// rtmodel/tests/dirrest.rs 의 same + 객체 키 순서(원본 dict 순서 = 응답 글자 순서)
fn same(g: &Value, w: &Value, path: &str, bad: &mut Vec<String>) {
    match w {
        Value::Object(m) if m.len() == 1 && m.contains_key("b") => {
            if g.as_f64().map(f64::to_bits) != Some(fb(&m["b"]).to_bits()) {
                bad.push(format!("{path}: rtserve {g} 대 파이썬 {:?}", fb(&m["b"])));
            }
        }
        Value::Object(m) => {
            let gk: Vec<&String> = g.as_object().map(|o| o.keys().collect()).unwrap_or_default();
            if gk != m.keys().collect::<Vec<_>>() {
                bad.push(format!("{path}: 키 순서 rtserve {gk:?} 대 파이썬 {:?}", m.keys().collect::<Vec<_>>()));
            }
            for (k, v) in m {
                same(&g[k], v, &format!("{path}.{k}"), bad);
            }
        }
        Value::Array(a) => {
            if g.as_array().map(Vec::len) != Some(a.len()) {
                bad.push(format!("{path}: 길이"));
            }
            for (i, v) in a.iter().enumerate() {
                same(&g[i], v, &format!("{path}[{i}]"), bad);
            }
        }
        _ if g != w => bad.push(format!("{path}: rtserve {g} 대 파이썬 {w}")),
        _ => {}
    }
}

fn link(dir: &Path, name: &str, from: &str) {
    let _ = std::fs::remove_file(dir.join(name));
    std::os::unix::fs::symlink(fixtures().join(from), dir.join(name)).unwrap();
}

fn get(addr: std::net::SocketAddr, clock: &AtomicU64, now_s: f64, path: &str) -> (u16, Vec<u8>) {
    clock.store(now_s.to_bits(), Ordering::SeqCst);
    let g = request(addr, "GET", path, &[]);
    (g.status, g.body)
}

/// 서버 하나(가짜 REST · 모델 자리 · 시계는 요청마다 바꾼다)
fn rig(tag: &str, fred: Option<String>) -> (std::net::SocketAddr, PathBuf, PathBuf, Book, Arc<AtomicU64>) {
    let book: Book = Default::default();
    let base = fake(book.clone());
    let clock = Arc::new(AtomicU64::new(0));
    let (c2, b2, models) = (clock.clone(), base.clone(), std::env::temp_dir().join(format!("rtserve_models_{tag}_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&models);
    std::fs::create_dir_all(&models).unwrap();
    let m2 = models.clone();
    let (addr, root) = start_with(&serde_json::Map::new(), 0, 0.0, "eth,sol,xrp", &format!("model_{tag}"), move |cfg| {
        cfg.now = Arc::new(move || f64::from_bits(c2.load(Ordering::SeqCst)));
        cfg.rest = b2.clone();
        cfg.ext = Some(b2);
        cfg.models = Some(m2);
        cfg.fred_key = fred;
    });
    std::fs::create_dir_all(root.join("data/live")).unwrap();
    (addr, root, models, book, clock)
}

/// 원본 joblib 자리(data/live) -- «낡음» 판정의 원천. 수정 시각은 모델 JSON 보다 앞(2001년)
fn src(root: &Path, name: &str, ns: u64) {
    let p = root.join("data/live").join(name);
    std::fs::write(&p, b"joblib").unwrap();
    common::set_mtime(&p, ns);
}

#[test]
fn model_handlers_over_fake_rest() {
    let rest = load("dirrest_fixture.json");
    let sig = load("signals_fixture.json");
    let key = sig["sig"][0]["key"].as_str().map(str::to_string);
    let (addr, root, models, book, clock) = rig("a", key);
    for c in ["sol", "xrp"] {
        link(&models, &format!("dir4h_{c}.json"), &format!("dir4h_{c}.json"));
        src(&root, &format!("{c}_dir4h_model.joblib"), 1_000_000_000_000_000_000);
    }
    let oos = json!(rest["oos"]);
    for c in ["btc", "sol", "xrp"] {
        link(&models, &format!("ds_{c}.json"), &format!("ds_{c}.json"));
        std::fs::write(models.join(format!("oos_{c}.json")), oos.to_string()).unwrap();
    }
    for a in ["eth", "xrp"] {
        link(&models, &format!("svm_{a}.json"), &format!("svm_{a}.json"));
        src(&root, &format!("{a}_sizing_vol_model.joblib"), 1_000_000_000_000_000_000);
    }
    let mut bad = Vec::new();
    // ① dir4h·dir-short: 정답지 첫 회차(빈 캐시에서 31일치) -- 같은 순서로 부르면 요청 열도 같아야 한다
    let steps: Vec<&Value> = rest["steps"].as_array().unwrap().iter().collect();
    let t0 = steps[0]["now_ms"].as_i64().unwrap();
    let first: Vec<&Value> = steps.iter().copied().filter(|s| s["now_ms"].as_i64() == Some(t0)).collect();
    let mut want_req = Vec::new();
    for st in &first {
        for r in st["requests"].as_array().unwrap() {
            let t = format!("{}?{}", r[0].as_str().unwrap(), qs(&r[1]));
            add(&book, t.clone(), r[2].clone());
            want_req.push(t);
        }
    }
    for st in &first {
        let (kind, coin) = (st["kind"].as_str().unwrap(), st["coin"].as_str().unwrap());
        let (code, body) = get(addr, &clock, t0 as f64 / 1000.0, &format!("/api/{}?asset={}", kind.replace('_', "-"), coin.to_lowercase()));
        assert_eq!(code, 200);
        let g: Value = serde_json::from_slice(&body).unwrap();
        same(&g, &st["payload"], &format!("{kind}/{coin}"), &mut bad);
    }
    let got_req = std::mem::take(&mut book.lock().unwrap().1);
    assert_eq!(got_req, want_req, "dir4h·dir-short 요청 열");
    let n_dir = first.len();
    book.lock().unwrap().0.clear(); // 앞 단계 녹화본이 같은 주소로 남아 있지 않게
    // ② vol-forecast HGB(qx4.json 없음 → 대체 길): 1000봉 × 10쪽 · endTime 이어 받기 · 본문 = 원본 글자 + "asset"
    let mut n_txt = 0;
    for v in sig["vol"].as_array().unwrap().iter().filter(|v| v["now"].as_f64() == sig["vol"][0]["now"].as_f64()) {
        let a = v["asset"].as_str().unwrap();
        let sym = format!("{}USDT", a.to_uppercase());
        let (rows, mut end) = (v["rows"].as_array().unwrap(), None::<i64>);
        for c in rows.rchunks(1000) {
            add(&book, format!("/fapi/v1/klines?symbol={sym}&interval=5m&limit=1000{}", end.map_or(String::new(), |e| format!("&endTime={e}"))), Value::Array(c.to_vec()));
            end = Some(c[0][0].as_i64().unwrap() - 1);
        }
        let (code, body) = get(addr, &clock, v["now"].as_f64().unwrap() / 1000.0, &format!("/api/vol-forecast?asset={a}"));
        let w = v["out"].as_str().unwrap();
        let want = format!("{}, \"asset\": \"{a}\"}}", &w[..w.len() - 1]);
        n_txt += 1;
        if code != 200 || body != want.as_bytes() {
            bad.push(format!("vol-forecast {a}: {code} {} · 요청 {:?}", String::from_utf8_lossy(&body[..body.len().min(300)]), book.lock().unwrap().1.iter().rev().take(2).collect::<Vec<_>>()));
        }
    }
    assert_eq!(std::mem::take(&mut book.lock().unwrap().1).len(), 20, "HGB 2코인 × 10쪽");
    book.lock().unwrap().0.clear();
    // ③ open-rev(서버 실수 시계 벌) · ④ dir-signals(FRED 키 있음)
    let o = &sig["open_rev"][4];
    add(&book, "/fapi/v1/klines?symbol=ETHUSDT&interval=1d&limit=93".into(), o["daily"].clone());
    add(&book, "/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=600".into(), o["m5"].clone());
    let (code, body) = get(addr, &clock, o["now"].as_f64().unwrap() / 1000.0, "/api/open-rev");
    n_txt += 1;
    if code != 200 || body != o["out"].as_str().unwrap().as_bytes() {
        bad.push(format!("open-rev: {}", String::from_utf8_lossy(&body)));
    }
    let s = &sig["sig"][0];
    for u in s["urls"].as_array().unwrap() {
        let url = u[0].as_str().unwrap();
        let rest_of = url.splitn(4, '/').nth(3).unwrap();
        add(&book, format!("/{rest_of}"), u[1].clone());
    }
    let (code, body) = get(addr, &clock, s["now"].as_f64().unwrap(), "/api/dir-signals");
    n_txt += 1;
    if code != 200 || body != s["out"].as_str().unwrap().as_bytes() {
        bad.push(format!("dir-signals: {}", String::from_utf8_lossy(&body[..body.len().min(400)])));
    }
    assert_eq!(std::mem::take(&mut book.lock().unwrap().1).len(), 2 + s["urls"].as_array().unwrap().len(), "open-rev 2 · dir-signals 요청 수");
    // ⑤ 지원 밖 · 모델 없음(원본 artifact_missing) -- 요청 없음
    for (p, want) in [("/api/dir4h?asset=btc", "unsupported_asset"), ("/api/dir-short?asset=doge", "unsupported_asset"), ("/api/vol-forecast?asset=hype", "unsupported_asset"),
                      ("/api/dir4h?asset=eth", "artifact_missing"), ("/api/vol-forecast?asset=btc", "artifact_missing")] {
        let (code, body) = get(addr, &clock, 1.795e9, p);
        if code != 200 || body != format!("{{\"ok\": false, \"reason\": \"{want}\"}}").as_bytes() {
            bad.push(format!("{p}: {code} {}", String::from_utf8_lossy(&body)));
        }
    }
    assert!(book.lock().unwrap().1.is_empty(), "지원 밖·모델 없음은 REST 를 안 부른다");
    println!("model_parity: dir4h·dir-short {n_dir}(요청 {} 같은 열) · 본문 글자 {n_txt} · 지원 밖·모델 없음 5 · 다른 값 {}", want_req.len(), bad.len());
    for b in bad.iter().take(10) {
        println!("  {b}");
    }
    assert!(bad.is_empty());
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(models);
}

/// QX4 배선(이어 받기 요청 열 · "asset" · 직렬화) · 밴 파일 = 안 보냄 · «모델 낡음» = 오류(옛 JSON 값을 안 냄)
#[test]
fn qx4_ban_stale() {
    let (addr, root, models, book, clock) = rig("b", None);
    link(&models, "qx4.json", "qx4.json");
    src(&root, "qx4_2026-10-23.joblib", 1_000_000_000_000_000_000);
    // 원본 정답지(qx4_fixture) 첫 벌의 4코인 5분봉을 거래소처럼 쪽으로(최근 1000 · 그 앞은 endTime 으로 1000씩)
    let set = &load("qx4_fixture.json")["sets"][0];
    let mut pages: HashMap<String, Value> = HashMap::new();
    for (coin, sym) in rtmodel::qx4::SYMS {
        let (rows, mut end) = (set["rows"][coin].as_array().unwrap(), None::<i64>);
        for c in rows.rchunks(1000) {
            pages.insert(format!("symbol={sym}&interval=5m&limit=1000{}", end.map_or(String::new(), |e| format!("&endTime={e}"))), Value::Array(c.to_vec()));
            end = Some(c[0][0].as_i64().unwrap() - 1);
        }
    }
    for (k, v) in &pages {
        add(&book, format!("/fapi/v1/klines?{k}"), v.clone());
    }
    let now_ms = set["now_ms"].as_f64().unwrap();
    let (code, body) = get(addr, &clock, now_ms / 1000.0, "/api/vol-forecast?asset=btc");
    assert_eq!(code, 200);
    // 같은 쪽들을 라이브러리로(같은 요청 열이어야 같은 행)
    let art = rtmodel::qx4::Art::from_json(&load("qx4.json"));
    let mut lib_req = Vec::new();
    let mut f = |q: &[(&'static str, String)]| {
        let k = q.iter().map(|(a, b)| format!("{a}={b}")).collect::<Vec<_>>().join("&");
        lib_req.push(format!("/fapi/v1/klines?{k}"));
        pages.get(&k).and_then(|v| v.as_array().cloned()).unwrap_or_default()
    };
    let want = rtmodel::handlers::vol_forecast_qx4(&mut f, &mut HashMap::new(), &art, now_ms, "btc").expect("QX4 ok");
    let want_req = lib_req;
    assert_eq!(std::mem::take(&mut book.lock().unwrap().1), want_req, "QX4 이어 받기 요청 열");
    assert_eq!(String::from_utf8(body).unwrap(), rtcollect::py_dumps_default(&want), "QX4 본문");
    // 밴: 공용 파일에 해제 시각(지금 + 600초) -- 보내지 않고 원본 문구
    std::fs::write(root.join("data/live/binance_ban_until"), format!("{:.3}\n", 1.796e9 + 600.0)).unwrap();
    for c in ["sol", "xrp"] {
        link(&models, &format!("dir4h_{c}.json"), &format!("dir4h_{c}.json"));
        src(&root, &format!("{c}_dir4h_model.joblib"), if c == "xrp" { 4_000_000_000_000_000_000 } else { 1_000_000_000_000_000_000 });
    }
    link(&models, "svm_eth.json", "svm_eth.json");
    src(&root, "eth_sizing_vol_model.joblib", 1_000_000_000_000_000_000);
    let (_, b) = get(addr, &clock, 1.796e9, "/api/dir4h?asset=sol");
    assert_eq!(String::from_utf8_lossy(&b), r#"{"ok": false, "reason": "BinanceBanned: binance IP ban: 600s left -- request not sent"}"#);
    let (_, b) = get(addr, &clock, 1.796e9, "/api/vol-forecast?asset=eth"); // QX4 이어 받기 = 밴이라 빈 응답 → HGB → 밴
    assert_eq!(String::from_utf8_lossy(&b), r#"{"ok": false, "reason": "HTTPServiceUnavailable: binance_ip_ban"}"#);
    assert!(book.lock().unwrap().1.is_empty(), "밴 중엔 하나도 안 보낸다");
    // 낡음: 원본 joblib(2096년)이 JSON 보다 새것
    let (_, b) = get(addr, &clock, 1.796e9, "/api/dir4h?asset=xrp");
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert!(v["ok"] == false && v["reason"].as_str().unwrap().starts_with("ModelStale: 모델 낡음: xrp_dir4h_model.joblib"), "{v}");
    std::fs::remove_file(root.join("data/live/qx4_2026-10-23.joblib")).unwrap(); // 원본 없음도 낡음과 같게(값을 안 낸다)
    let (_, b) = get(addr, &clock, 1.796e9, "/api/vol-forecast?asset=sol");
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert!(v["reason"].as_str().unwrap().starts_with("ModelStale: 모델 원본 없음"), "{v}");
    println!("qx4_ban_stale: QX4 요청 {} 같은 열 · 본문 같음 · 밴 2 · 낡음 2", want_req.len());
    let _ = std::fs::remove_dir_all(root);
    let _ = std::fs::remove_dir_all(models);
}

/// 병행 가동 기본: RTSERVE_REST 없음 = 모델 주소가 REST 를 안 보냄 · RTSERVE_READONLY = 쓰기 POST 4개 503 · 접수 기록 안 적음
#[test]
fn shadow_rest_off_readonly() {
    let (addr, root) = start_with(&serde_json::Map::new(), 0, 1.795e9, "eth", "model_ro", |cfg| cfg.readonly = true);
    std::fs::create_dir_all(root.join("data/live")).unwrap(); // 있어야 «안 적음»을 볼 수 있다(없으면 적기가 조용히 실패)
    let g = request(addr, "GET", "/api/open-rev", &[]);
    assert_eq!(String::from_utf8_lossy(&g.body), r#"{"ok": false, "reason": "RestOff: RTSERVE_REST unset -- request not sent"}"#);
    for p in rtserve::READONLY_BLOCK {
        let g = common::request_body(addr, "POST", p, &[("Content-Type".into(), "application/json".into())], b"{}");
        assert_eq!((g.status, String::from_utf8_lossy(&g.body).into_owned()), (503, r#"{"ok": false, "error": "readonly_shadow"}"#.to_string()), "{p}");
    }
    let g = common::request_body(addr, "POST", "/api/manual-entry/submit", &[], b"");
    assert_eq!(g.status, 502, "주문 = rtorder 없음 그대로");
    let g = common::request_body(addr, "POST", "/api/client-timing", &[], b"{}");
    assert_ne!(g.status, 503, "client-timing 은 로그만 -- 안 막음");
    assert!(!root.join("data/live/order_requests.jsonl").exists() && !root.join("data/live/acct_notes.json").exists(), "병행 중 파일을 안 쓴다");
    let _ = std::fs::remove_dir_all(root);
}
