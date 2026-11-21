//! rt_radar(위험 레이더 cron 한 회차)를 127.0.0.1 가짜 HTTP(바이낸스 1분봉 · Deribit DVOL)로 5회차: 요청 열 == 원본 요청 열(순서·쿼리) ·
//! latest.json·forward.jsonl == 원본(문자열·참거짓·None 같음 · 숫자 1e-12 상대 -- X@β 는 BLAS 합 순서). 정답지: tools/gen_radar_loop_fixture.py
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn close(a: &Value, b: &Value, path: &str, bad: &mut Vec<String>, worst: &mut f64) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: 필드 수");
            for (k, v) in y {
                close(&x[k], v, &format!("{path}.{k}"), bad, worst);
            }
        }
        (Value::Number(x), Value::Number(y)) if x != y => {
            let (p, q) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            *worst = worst.max((p - q).abs() / q.abs().max(1e-300));
        }
        _ if a != b => bad.push(format!("{path}: Rust {a} 대 원본 {b}")),
        _ => {}
    }
}

#[test]
fn radar_cron_rounds_over_fake_http() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fx: Value = rtmodel::read_json(&d.join("tests/fixtures/radar_loop_fixture.json")).expect("정답지 없음 -- tools/gen_radar_loop_fixture.py");
    let out = std::env::temp_dir().join(format!("rt_radar_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let q: Arc<Mutex<VecDeque<(String, Value)>>> = Arc::new(Mutex::new(VecDeque::new()));
    let bad: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let (qq, bb) = (q.clone(), bad.clone());
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut buf = vec![0u8; 16384];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                let k = s.read(&mut buf[n..]).unwrap();
                if k == 0 { break; }
                n += k;
            }
            let path = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let (status, body) = match qq.lock().unwrap().pop_front() {
                Some((want, resp)) if want == path => if resp.get("__status").is_some() { (500, "{}".to_string()) } else { (200, resp.to_string()) },
                other => { bb.lock().unwrap().push(format!("요청 {path} 대 원본 {:?}", other.map(|o| o.0))); (200, "[]".into()) }
            };
            let _ = s.write_all(format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    let (mut worst, mut nlog, mut nreq) = (0.0f64, 0usize, 0usize);
    let mut want_log: Vec<String> = vec![];
    for run in fx["runs"].as_array().unwrap() {
        for r in run["requests"].as_array().unwrap() {
            let qs: Vec<String> = r[2].as_array().unwrap().iter().map(|p| format!("{}={}", p[0].as_str().unwrap(), p[1].as_str().unwrap())).collect();
            q.lock().unwrap().push_back((format!("{}?{}", r[1].as_str().unwrap(), qs.join("&")), r[3].clone()));
            nreq += 1;
        }
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_radar")).env("RT_RADAR_OUT", &out)
            .env("RT_RADAR_PARAMS", d.join("../dashboard/risk_radar_params_20261116.json")).env("RT_RADAR_PARAMS_SESS", d.join("../dashboard/risk_radar_params_soxl_20261117.json"))
            .env("RT_RADAR_FAPI", format!("http://127.0.0.1:{port}")).env("RT_RADAR_DERIBIT", format!("http://127.0.0.1:{port}"))
            .env("RT_SR_BAN_FILE", out.join("ban")).env("RT_RADAR_NOW_MS", run["now_ms"].to_string()).output().unwrap();
        let log = String::from_utf8_lossy(&o.stderr).to_string() + &String::from_utf8_lossy(&o.stdout);
        assert_eq!(o.status.code(), Some(0), "{log}");
        assert!(bad.lock().unwrap().is_empty(), "{:?}", bad.lock().unwrap());
        assert!(q.lock().unwrap().is_empty(), "원본 요청이 남았다\n{log}");
        let got: Value = serde_json::from_str(&std::fs::read_to_string(out.join("risk_radar_latest.json")).unwrap()).unwrap();
        let mut b = vec![];
        close(&got, &serde_json::from_str(run["latest"].as_str().unwrap()).unwrap(), "latest", &mut b, &mut worst);
        assert!(b.is_empty(), "{b:?}");
        want_log.extend(run["log"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()));
    }
    let gl: Vec<String> = std::fs::read_to_string(out.join("risk_radar_forward.jsonl")).unwrap().lines().map(str::to_string).collect();
    assert_eq!(gl.len(), want_log.len());
    for (g, w) in gl.iter().zip(&want_log) {
        let mut b = vec![];
        close(&serde_json::from_str(g).unwrap(), &serde_json::from_str(w).unwrap(), "forward", &mut b, &mut worst);
        assert!(b.is_empty(), "{b:?}");
        nlog += (g == w) as usize;
    }
    println!("cron 5회차: 요청 {nreq} 개 원본 순서 · 숫자 최대 상대 {worst:.2e} · forward 줄 {}/{} 글자 같음", nlog, gl.len());
    assert!(worst <= 1e-12);
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_radar")).env("RT_RADAR_OUT", out.join("data/live")).output().unwrap();
    assert_eq!(o.status.code(), Some(2), "파이썬 자리 거부");
}
