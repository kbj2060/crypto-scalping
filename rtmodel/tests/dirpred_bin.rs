//! rt_dirpred 실행 파일 한 회차(RT_DIR_NOW_MS 고정)를 127.0.0.1 가짜 HTTP 로: 요청 열 == 원본 요청 열(tests/dirrest.rs 와 같은 정답지의 첫 회차) ·
//! 쓴 pred_record/{코인}.jsonl == 원본 함수(라이브러리)로 같은 회차를 돈 기록 · 마지막 payload 파일. 파이썬 기록 자리(data/live)는 종료코드 2.
//!     (정답지: tests/dirrest.rs 머리말)  cargo test -j 2 --release --test dirpred_bin -- --nocapture
use rtmodel::dirrest::{closes, daily_closes, xsec_px, Record, Rest};
use rtmodel::{dir4h, dirshort};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn fb(v: &Value) -> f64 {
    f64::from_bits(v.as_u64().unwrap())
}

#[test]
fn one_round_over_fake_http() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fx: Value = rtmodel::read_json(&d.join("dirrest_fixture.json")).expect("정답지 없음");
    let steps: Vec<&Value> = fx["steps"].as_array().unwrap().iter().collect();
    let t0 = steps[0]["now_ms"].as_i64().unwrap();
    let first: Vec<&Value> = steps.iter().copied().filter(|s| s["now_ms"].as_i64() == Some(t0)).collect();
    // 모델 디렉터리(4h SOL·XRP · 15분·1시간 BTC·SOL·XRP · 분위 씨앗 = 정답지와 같은 대역)
    let dir = std::env::temp_dir().join(format!("rt_dirpred_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for c in ["sol", "xrp"] {
        std::fs::copy(d.join(format!("dir4h_{c}.json")), dir.join(format!("dir4h_{c}.json"))).unwrap();
    }
    for c in ["btc", "sol", "xrp"] {
        std::fs::copy(d.join(format!("ds_{c}.json")), dir.join(format!("ds_{c}.json"))).unwrap();
        std::fs::write(dir.join(format!("oos_{c}.json")), fx["oos"].to_string()).unwrap();
    }
    // 가짜 HTTP: 원본 요청 열을 차례로 -- 다르면 그 자리에서 실패
    let q: Arc<Mutex<VecDeque<(String, Value)>>> = Arc::new(Mutex::new(first.iter().flat_map(|s| s["requests"].as_array().unwrap().iter().map(|r| {
        let qs: Vec<String> = r[1].as_array().unwrap().iter().map(|p| format!("{}={}", p[0].as_str().unwrap(), p[1].as_str().unwrap())).collect();
        (format!("{}?{}", r[0].as_str().unwrap(), qs.join("&")), r[2].clone())
    })).collect()));
    // 회차 끝 상대 강도: 9코인 _klines(c, 60) -- 응답은 tests/dirrest.rs xsec_parity 정답지 첫 회차(쿼리에 시각이 없어 그대로 맞는다)
    let xs: Value = rtmodel::read_json(&d.join("xsec_fixture.json")).expect("python rtmodel/tools/gen_xsec_fixture.py 먼저");
    let xreq: Vec<Value> = xs["runs"][0]["requests"].as_array().unwrap().clone();
    for r in &xreq {
        let qs: Vec<String> = r[1].as_array().unwrap().iter().map(|p| format!("{}={}", p[0].as_str().unwrap(), p[1].as_str().unwrap())).collect();
        q.lock().unwrap().push_back((format!("{}?{}", r[0].as_str().unwrap(), qs.join("&")), r[2].clone()));
    }
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let (qq, bad) = (q.clone(), Arc::new(Mutex::new(vec![])));
    let bad2 = bad.clone();
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
            let body = match qq.lock().unwrap().pop_front() {
                Some((want, resp)) if want == path => resp.to_string(),
                other => { bad2.lock().unwrap().push(format!("요청 {path} 대 원본 {:?}", other.map(|o| o.0))); "[]".into() }
            };
            let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    let out = dir.join("out");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_dirpred")).env("RT_DIR_OUT", &out).env("RT_DIR_MODELS", &dir)
        .env("RT_DIR_REST", format!("http://127.0.0.1:{port}")).env("RT_SR_BAN_FILE", dir.join("ban")).env("RT_DIR_NOW_MS", t0.to_string()).output().unwrap();
    let log = String::from_utf8_lossy(&o.stderr).to_string() + &String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{log}");
    assert!(bad.lock().unwrap().is_empty(), "{:?}", bad.lock().unwrap());
    assert!(q.lock().unwrap().is_empty(), "원본 요청 {} 개를 안 보냈다\n{log}", q.lock().unwrap().len());
    // 기대 기록: 같은 회차를 라이브러리로(같은 응답)
    let exp = dir.join("exp");
    let (mut r4, mut rs, mut rec) = (Rest::default(), Rest::default(), Record::default());
    let oos: Vec<(&str, Vec<f64>)> = dirshort::NAMES.iter().map(|(_, nm)| (*nm, fx["oos"][*nm].as_array().unwrap().iter().map(fb).collect())).collect();
    let mut last = std::collections::HashMap::new();
    for st in &first {
        let (kind, coin) = (st["kind"].as_str().unwrap(), st["coin"].as_str().unwrap());
        let mut rq: VecDeque<&Value> = st["requests"].as_array().unwrap().iter().collect();
        let mut get = |_: &str, _: &[(&'static str, String)]| -> Result<Value, String> { Ok(rq.pop_front().unwrap()[2].clone()) };
        let r = if kind == "dir4h" { &mut r4 } else { &mut rs };
        let fr = r.frames(&mut get, t0, t0 as f64, coin).unwrap();
        let f = dir4h::build_features(&dir4h::frames_for(&fr, coin), &daily_closes(&mut get, t0, coin).unwrap()).unwrap();
        let pay = if kind == "dir4h" {
            dir4h::payload(&dir4h::Art::load(&dir.join(format!("dir4h_{}.json", coin.to_lowercase()))).unwrap(), &f)
        } else {
            dirshort::payload(&dirshort::Art::from_json(&rtmodel::read_json::<Value>(&dir.join(format!("ds_{}.json", coin.to_lowercase()))).unwrap()).unwrap(), &f, 13, &oos)
        };
        let cl = closes(r, coin).unwrap();
        if kind == "dir4h" { rec.h4(&exp, coin, &pay, &cl).unwrap() } else { rec.short(&exp, coin, &pay, &cl).unwrap() };
        last.insert(format!("{kind}_{}.json", coin.to_lowercase()), pay);
    }
    let mut xq: VecDeque<&Value> = xreq.iter().collect();
    let px = xsec_px(&mut |_: &str, _: &[(&'static str, String)]| Ok(xq.pop_front().unwrap()[2].clone()), t0).unwrap();
    let (xn, latest) = rec.xsec(&exp, &px).unwrap();
    let rd = |p: PathBuf| std::fs::read_to_string(p).ok();
    assert_eq!(rd(out.join("pred_record/xsec.jsonl")), rd(exp.join("xsec.jsonl")), "xsec.jsonl");
    assert_eq!(rd(out.join("xsec_latest.json")), latest, "xsec_latest.json");
    println!("상대 강도: 새 줄 {xn} · 스냅숏 {}", latest.is_some());
    let mut n = 0;
    for c in ["BTC", "SOL", "XRP"] {
        let (g, w) = (std::fs::read_to_string(out.join("pred_record").join(format!("{c}.jsonl"))).unwrap(), std::fs::read_to_string(exp.join(format!("{c}.jsonl"))).unwrap());
        assert_eq!(g, w, "{c} 적중 기록");
        n += g.lines().count();
    }
    for (f, p) in &last {
        let g: Value = serde_json::from_str(&std::fs::read_to_string(out.join(f)).unwrap()).unwrap();
        assert_eq!(&g, p, "{f}");
    }
    println!("실행 파일 한 회차: 요청 {} 개 원본 순서 그대로 · 적중 기록 {n}줄 · payload 파일 {}개 같음", first.iter().map(|s| s["requests"].as_array().unwrap().len()).sum::<usize>(), last.len());
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_dirpred")).env("RT_DIR_OUT", dir.join("data/live")).output().unwrap();
    assert_eq!(o.status.code(), Some(2), "파이썬 기록 자리 거부");
}
