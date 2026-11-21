//! 둘째 조각 대조: tools/gen_http2_fixture.py 가 진짜 aiohttp 객체로 뜬 바이트 == rtserve(127.0.0.1 · 실제 HTTP).
//! 쓰기 시나리오는 요청마다 파일 바이트(메모·오답·구독·주문 기록)·남은 임시 파일·로그 줄까지 대조한다.
//! 넷째 조각(2026-11-22)부터 주문·계좌 14개는 rtorder 로 넘긴다 -- 이 시험엔 rtorder 가 없으니 정답(원본 정답지는 «주소 없음» 404)을
//! 의도한 다름으로 바꿔 본다: 502 `{"ok": false, "error": "rtorder_unavailable"}` · 접수 기록 줄의 status·error 도 그 값(문서 §5).

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
            let mut want_h: Vec<(String, String)> = c["headers"].as_array().unwrap().iter().map(|h| (h[0].as_str().unwrap().into(), h[1].as_str().unwrap().into())).collect();
            let mut want_body = b64(c["body_b64"].as_str().unwrap());
            let (mut want_st, mut want_rs) = (c["status"].as_u64().unwrap() as u16, c["reason"].as_str().unwrap().to_string());
            if rtserve::order::is_proxied(p.split('?').next().unwrap()) {
                want_body = br#"{"ok": false, "error": "rtorder_unavailable"}"#.to_vec();
                want_h = vec![("Content-Type".into(), "application/json; charset=utf-8".into()), ("Cache-Control".into(), "no-cache".into()), ("Content-Length".into(), want_body.len().to_string())];
                (want_st, want_rs) = (502, "Bad Gateway".into());
            }
            let mut diff = Vec::new();
            if got.status != want_st || got.reason != want_rs {
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
                    let want = want.as_str().map(b64).map(|b| if rel.ends_with("order_requests.jsonl") {
                        String::from_utf8(b).unwrap().lines().map(|l| {
                            let v: Value = serde_json::from_str(l).unwrap();
                            if rtserve::order::is_proxied(v["path"].as_str().unwrap()) { l.replace(r#""status": 404, "error": null"#, r#""status": 502, "error": "rtorder_unavailable""#) } else { l.to_string() }
                        }).map(|l| l + "\n").collect::<String>().into_bytes()
                    } else { b });
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


/// rt_* 칸(9칸 main · rt_draw_s·rt_back_n 은 머지 대기 -- 정답지에 없다): 받아서 이름 순으로 한 줄에 싣는다
#[test]
fn rt_timing_keys() {
    let logs: Arc<Mutex<Vec<String>>> = Default::default();
    let lg = logs.clone();
    let (addr, root) = start_with(&Default::default(), 1_700_000_000_000_000_000, 1.7e9, "eth", "rtkeys", move |c| {
        c.log = Arc::new(move |l| lg.lock().unwrap().push(l.to_string()));
    });
    let body = br#"{"sym": "eth", "rt_draw_s": 3.5, "rt_back_n": 1, "rt_lat_n": 40, "rt_lat_med_ms": 12.5, "rt_lat_p99_ms": 31.7, "rt_lead_n": 3, "rt_lead_med_ms": -2.5, "rt_cmp_n": 9, "rt_bad_n": 0, "rt_fail_n": 1, "rt_reconn_n": 2}"#;
    let g = request_body(addr, "POST", "/api/client-timing", &[("Content-Type".into(), "application/json".into())], body);
    assert_eq!(g.status, 200, "{}", String::from_utf8_lossy(&g.body));
    assert_eq!(logs.lock().unwrap().as_slice(), ["client_timing sym=eth rt_back_n=1 rt_bad_n=0 rt_cmp_n=9 rt_draw_s=4 rt_fail_n=1 rt_lat_med_ms=12 rt_lat_n=40 rt_lat_p99_ms=32 rt_lead_med_ms=-2 rt_lead_n=3 rt_reconn_n=2"]);
    let _ = std::fs::remove_dir_all(root);
}
