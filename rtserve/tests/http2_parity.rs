//! 둘째 조각 대조: tools/gen_http2_fixture.py 가 진짜 aiohttp 객체로 뜬 바이트 == rtserve(127.0.0.1 · 실제 HTTP).
//! 쓰기 시나리오는 요청마다 파일 바이트(메모·오답·구독·주문 기록)·남은 임시 파일·로그 줄까지 대조한다.

mod common;

use std::sync::{Arc, Mutex};

use common::{b64, norm, request_body, start_with};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/http2.json")).unwrap()).unwrap()
}

#[test]
fn http2_parity() {
    let fx = fixture();
    let (now, mtime, assets) = (fx["now"].as_f64().unwrap(), fx["mtime_ns"].as_u64().unwrap(), fx["dashboard_assets"].as_str().unwrap());
    let (mut n, mut bad, mut files_n, mut bytes) = (0, Vec::new(), 0, 0);
    for sc in fx["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let logs: Arc<Mutex<Vec<String>>> = Default::default();
        let lg = logs.clone();
        let vapid = fx["vapid_public"].as_str().unwrap().to_string();
        let (addr, root) = start_with(sc["files"].as_object().unwrap(), mtime, now, assets, &format!("h2_{name}"), move |c| {
            c.vapid_public = vapid;
            c.vapid_private_set = true;
            c.log = Arc::new(move |l| lg.lock().unwrap().push(l.to_string()));
        });
        for c in sc["cases"].as_array().unwrap() {
            n += 1;
            let rh: Vec<(String, String)> = c["req_headers"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().into())).collect();
            let (m, p) = (c["method"].as_str().unwrap(), c["path"].as_str().unwrap());
            logs.lock().unwrap().clear();
            let got = request_body(addr, m, p, &rh, &b64(c["body_in"].as_str().unwrap()));
            let want_h: Vec<(String, String)> = c["headers"].as_array().unwrap().iter().map(|h| (h[0].as_str().unwrap().into(), h[1].as_str().unwrap().into())).collect();
            let want_body = b64(c["body_b64"].as_str().unwrap());
            let mut diff = Vec::new();
            if got.status != c["status"].as_u64().unwrap() as u16 || got.reason != c["reason"].as_str().unwrap() {
                diff.push(format!("상태 {} {} != {} {}", got.status, got.reason, c["status"], c["reason"]));
            }
            if norm(&got.headers) != norm(&want_h) {
                diff.push(format!("헤더\n    rust {:?}\n    py   {:?}", norm(&got.headers), norm(&want_h)));
            }
            if got.body != want_body {
                diff.push(format!("본문 {}B != {}B\n    rust {}\n    py   {}", got.body.len(), want_body.len(),
                                  String::from_utf8_lossy(&got.body[..got.body.len().min(1500)]), String::from_utf8_lossy(&want_body[..want_body.len().min(1500)])));
            }
            let want_log: Vec<String> = c["log"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
            let got_log = logs.lock().unwrap().clone();
            if got_log != want_log {
                diff.push(format!("로그 {got_log:?} != {want_log:?}"));
            }
            if let Some(fa) = c.get("files_after").and_then(Value::as_object) {
                for (rel, want) in fa {
                    files_n += 1;
                    let have = std::fs::read(root.join(rel)).ok();
                    let want = want.as_str().map(b64);
                    if have != want {
                        diff.push(format!("파일 {rel}\n    rust {:?}\n    py   {:?}", have.map(|b| String::from_utf8_lossy(&b).into_owned()), want.map(|b| String::from_utf8_lossy(&b).into_owned())));
                    }
                }
                let tmp: Vec<String> = walk(&root).into_iter().filter(|p| p.ends_with(".tmp")).collect();
                let want_tmp: Vec<String> = c["tmp_left"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
                if tmp != want_tmp {
                    diff.push(format!("임시 파일 {tmp:?} != {want_tmp:?}"));
                }
            }
            bytes += want_body.len();
            if !diff.is_empty() {
                bad.push(format!("[{name}] {m} {p}\n  {}", diff.join("\n  ")));
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    eprintln!("http2_parity: 요청 {n} · 본문 {bytes}B · 쓴 파일 대조 {files_n} · 불일치 {}", bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn walk(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut st = vec![root.to_path_buf()];
    while let Some(d) = st.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                st.push(p);
            } else {
                out.push(p.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    out.sort();
    out
}

/// 원본 상수와 같은지(정답지가 원본 모듈에서 읽어 실었다)
#[test]
fn consts_match_original() {
    let fx = fixture();
    let c = &fx["consts"];
    let fs: Vec<&str> = c["flow_specs"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert_eq!(fs, rtserve::FLOW_COINS);
    let ms: Vec<&str> = c["market_symbols"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert_eq!(ms, rtserve::MARKET_ASSETS);
    let tb: Vec<(String, f64)> = c["tape_buckets"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_f64().unwrap())).collect();
    assert_eq!(tb, rtserve::api2::TAPE_BUCKETS.iter().map(|(k, v)| (k.to_string(), *v)).collect::<Vec<_>>());
    for (s, w) in c["whale_usd"].as_object().unwrap() {
        let mine = rtserve::api2::WHALE_USD.iter().find(|x| x.0 == s).map_or(100_000.0, |x| x.1);
        assert_eq!(mine, w.as_f64().unwrap(), "{s}");
    }
}

/// statistics.stdev 와 비트까지 같음
#[test]
fn stdev_exact() {
    let fx = fixture();
    for r in fx["stdevs"].as_array().unwrap() {
        let xs: Vec<f64> = r[0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(rtserve::py::stdev(&xs).unwrap().to_bits(), r[1].as_f64().unwrap().to_bits(), "n={}", xs.len());
    }
}
