//! 원본 aiohttp 가 낸 바이트(tools/gen_http_fixture.py → fixtures/http.json) == rtserve 를 127.0.0.1 에 띄워 실제 HTTP 로 물은 답.
//! 대조: 상태 코드·상태 줄 글자·헤더(이름 대소문자 무시 · Date·Server·Connection 제외)·본문 바이트(청크는 풀어서 · 압축 바이트 그대로).

mod common;

use common::{b64, norm, request, start};
use serde_json::Value;

#[test]
fn http_parity() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/http.json")).unwrap()).unwrap();
    let (now, mtime, assets) = (fx["now"].as_f64().unwrap(), fx["mtime_ns"].as_u64().unwrap(), fx["dashboard_assets"].as_str().unwrap());
    let (mut n, mut bad, mut hdrs, mut bytes, mut intended) = (0, Vec::new(), 0, 0, 0);
    for sc in fx["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let (addr, root) = start(sc["files"].as_object().unwrap(), mtime, now, assets, name);
        for c in sc["cases"].as_array().unwrap() {
            n += 1;
            let rh: Vec<(String, String)> = c["req_headers"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().into())).collect();
            let (m, p) = (c["method"].as_str().unwrap(), c["path"].as_str().unwrap());
            let got = request(addr, m, p, &rh);
            let want_h: Vec<(String, String)> = c["headers"].as_array().unwrap().iter().map(|h| (h[0].as_str().unwrap().into(), h[1].as_str().unwrap().into())).collect();
            let want_body = b64(c["body_b64"].as_str().unwrap());
            let mut diff = Vec::new();
            if got.status != c["status"].as_u64().unwrap() as u16 || got.reason != c["reason"].as_str().unwrap() {
                diff.push(format!("상태 {} {} != {} {}", got.status, got.reason, c["status"], c["reason"]));
            }
            if norm(&got.headers) != norm(&want_h) {
                diff.push(format!("헤더\n    rust {:?}\n    py   {:?}", norm(&got.headers), norm(&want_h)));
            }
            if got.chunked != c["chunked"].as_bool().unwrap() {
                diff.push(format!("청크 {} != {}", got.chunked, c["chunked"]));
            }
            // 의도한 차이(하나뿐): 원본은 압축을 켠 StreamResponse 의 304 뒤에 빈 zlib 흐름(8·20바이트)을 «본문 없이» 덧붙여 보낸다 --
            //   304 는 본문이 없어야 해서 같은 연결의 다음 응답 앞에 쓰레기 바이트가 끼는 원본 결함이다. Rust(hyper)는 보내지 않는다.
            let stray_304 = c["status"] == 304 && got.body.is_empty() && want_h.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-encoding"));
            if stray_304 {
                intended += 1;
            } else if got.body != want_body {
                diff.push(format!("본문 {}B != {}B\n    rust {:?}\n    py   {:?}", got.body.len(), want_body.len(),
                                  String::from_utf8_lossy(&got.body[..got.body.len().min(400)]), String::from_utf8_lossy(&want_body[..want_body.len().min(400)])));
            }
            hdrs += norm(&want_h).len();
            bytes += want_body.len();
            if !diff.is_empty() {
                bad.push(format!("[{name}] {m} {p} {:?}\n  {}", rh, diff.join("\n  ")));
            }
        }
        let _ = std::fs::remove_dir_all(root);
    }
    eprintln!("http_parity: 요청 {n} · 헤더 {hdrs} · 본문 {bytes}B · 불일치 {} · 의도한 차이(304 뒤 빈 zlib) {intended}", bad.len());
    assert_eq!(intended, 3, "의도한 차이 건수가 바뀌었다 -- 정답지를 다시 읽을 것");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// 신뢰 경계(원본 정답지 없이 Rust 만): 정적 폴더 밖 파일은 어떤 경로 글자로도 200 이 나오면 안 된다
#[test]
fn static_traversal_blocked() {
    let mut files = serde_json::Map::new();
    files.insert("data/live/secret.json".into(), Value::String("eyJzZWNyZXQiOiB0cnVlfQ==".into()));
    files.insert("dashboard/live/ok.js".into(), Value::String("dmFyIGE9MTs=".into()));
    files.insert("rtui/dist/index.html".into(), Value::String("dmFyIGE9MTs=".into()));
    let (addr, root) = start(&files, 1_700_000_000_000_000_000, 1.7e9, "eth", "trav");
    std::os::unix::fs::symlink(root.join("data/live/secret.json"), root.join("dashboard/live/link.json")).unwrap();
    std::os::unix::fs::symlink(root.join("data/live"), root.join("dashboard/live/linkdir")).unwrap();
    let secret = b"{\"secret\": true}";
    for p in [
        "/dashboard/live/../../data/live/secret.json",
        "/dashboard/live/%2e%2e/%2e%2e/data/live/secret.json",
        "/dashboard/live/..%2f..%2fdata%2flive%2fsecret.json",
        "/dashboard/live/%2E%2E%2F%2E%2E%2Fdata/live/secret.json",
        "/dashboard/live//etc/passwd",
        "/dashboard/live/%2fetc%2fpasswd",
        "/dashboard/live/link.json",
        "/dashboard/live/linkdir/secret.json",
        "/dashboard/live/ok.js%00.png",
        "/dashboard/live/./../live/../../data/live/secret.json",
        "/rtweb/../../data/live/secret.json",
        "/rtweb/%2e%2e/%2e%2e/%2e%2e/data/live/secret.json",
    ] {
        let g = request(addr, "GET", p, &[]);
        assert!(g.status != 200 && !g.body.windows(secret.len()).any(|w| w == secret), "{p} → {} {:?}", g.status, String::from_utf8_lossy(&g.body));
    }
    assert_eq!(request(addr, "GET", "/dashboard/live/ok.js", &[]).status, 200); // 대조: 안쪽 파일은 나온다
    assert_eq!(request(addr, "GET", "/rtweb/", &[]).status, 200); // Rust 화면 묶음(rtui/dist/index.html)
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn loopback_only() {
    assert!(rtserve::check_loopback(&"127.0.0.1:8790".parse().unwrap()).is_ok());
    assert!(rtserve::check_loopback(&"0.0.0.0:8790".parse().unwrap()).is_err());
    assert!(rtserve::check_loopback(&"192.168.1.5:8790".parse().unwrap()).is_err());
    assert_eq!(rtserve::parse_assets(Some(" ETH,doge,btc ")), vec!["eth", "btc"]);
    assert_eq!(rtserve::parse_assets(Some("doge")), vec!["eth"]);
}

/// 파이썬 round(x, nd)·round(x) 와 비트까지 같음(정답지 rounds)
#[test]
fn round_like_python() {
    let fx: Value = serde_json::from_str(&std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/http.json")).unwrap()).unwrap();
    let rows = fx["rounds"].as_array().unwrap();
    for r in rows {
        let (x, nd) = (r[0].as_f64().unwrap(), r[1].as_i64().unwrap());
        if nd < 0 {
            assert_eq!(rtserve::py::round0(x).unwrap(), r[2].as_i64().unwrap(), "round({x})");
        } else {
            assert_eq!(rtserve::py::round_nd(x, nd as usize).to_bits(), r[2].as_f64().unwrap().to_bits(), "round({x}, {nd})");
        }
    }
    eprintln!("round_like_python: {}", rows.len());
}
