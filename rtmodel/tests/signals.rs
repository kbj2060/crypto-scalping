//! 방향 전진 기록기 · 개장 되돌림 · 4h 변동성 HGB · 7일/4~13주 신호를 원본(live_eth_direction_forward_recorder_20261010 · dashboard/dir_signals)과 글자 대조.
//!     (tools/server_env.sh 의 환경) python rtmodel/tools/gen_signals_fixture.py
//!     cargo test -j 2 --release --test signals -- --nocapture
use rtmodel::signals::*;
use serde_json::Value;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

fn fx() -> (PathBuf, Value) {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let v = serde_json::from_str(&std::fs::read_to_string(d.join("signals_fixture.json")).expect("python rtmodel/tools/gen_signals_fixture.py 먼저")).unwrap();
    (d, v)
}

#[test]
fn recorder_parity() {
    let (_, fx) = fx();
    let tmp = std::env::temp_dir().join(format!("rt_dirfwd_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let out = tmp.join("eth_direction_forward.jsonl");
    let (mut lines, mut nreq) = (0, 0);
    for r in fx["rec"].as_array().unwrap() {
        let now = r["now"].as_i64().unwrap();
        let mut calls: VecDeque<&Value> = r["calls"].as_array().unwrap().iter().collect();
        let mut f = |url: &str| -> Result<Value, String> {
            let c = calls.pop_front().expect("요청이 더 많다");
            assert_eq!(url, format!("http://fake/fapi/v1/klines?symbol={}&interval={}&limit={}", c[0].as_str().unwrap(), c[1].as_str().unwrap(), c[2]), "{now}");
            nreq += 1;
            Ok(c[3].clone())
        };
        let got = record(&mut f, "http://fake", now, &out);
        match r["err"].as_str() {
            None => assert!(got.is_ok(), "{now}: {got:?}"),
            Some("exit") => assert!(got.is_err() && r["calls"].as_array().unwrap().is_empty()),
            Some(e) => assert_eq!(got.unwrap_err(), e, "{now}"),
        }
        assert!(calls.is_empty(), "{now}: 요청이 모자란다");
        let txt = std::fs::read_to_string(&out).ok();
        assert_eq!(txt.as_deref(), r["file"].as_str(), "{now} 기록 글자");
        lines = txt.map_or(0, |t| t.lines().count());
    }
    let _ = std::fs::remove_dir_all(&tmp);
    println!("기록기: 회차 {} · 요청 {nreq} · 줄 {lines} 글자 같음", fx["rec"].as_array().unwrap().len());
}

#[test]
fn open_rev_vol4h_parity() {
    let (d, fx) = fx();
    for o in fx["open_rev"].as_array().unwrap() {
        let now = &o["now"];
        let g = open_rev(o["daily"].as_array().unwrap(), o["m5"].as_array().unwrap(), now.as_f64().unwrap(), now.is_f64()).unwrap();
        assert_eq!(g, o["out"].as_str().unwrap(), "open_rev {now}");
    }
    let art = |a: &str| rtmodel::read_json::<Value>(&d.join(format!("svm_{a}.json"))).expect("svm_*.json");
    for v in fx["vol"].as_array().unwrap() {
        let el = v["expand_logit"].as_array().map(|a| (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()));
        let a = Svm::from_json(&art(v["asset"].as_str().unwrap())).unwrap().with_expand(el);
        assert_eq!(vol4h(v["rows"].as_array().unwrap(), &a, v["now"].as_f64().unwrap()), v["out"].as_str().unwrap(), "vol4h {}", v["now"]);
    }
    let a = Svm::from_json(&art("eth")).unwrap();
    let short: Vec<Value> = fx["vol"][0]["rows"].as_array().unwrap()[..599].to_vec();
    assert_eq!(vol4h(&short, &a, 9e18), fx["vol_short"].as_str().unwrap(), "봉 부족 문구");
    println!("개장 되돌림 {} · 4h 변동성 {} 글자 같음", fx["open_rev"].as_array().unwrap().len(), fx["vol"].as_array().unwrap().len());
}

#[test]
fn dir_signals_parity() {
    let (_, fx) = fx();
    let mut n = 0;
    for s in fx["sig"].as_array().unwrap() {
        let mut q: VecDeque<&Value> = s["urls"].as_array().unwrap().iter().collect();
        let mut f = |url: &str| -> Result<Value, String> {
            let w = q.pop_front().expect("요청이 더 많다");
            assert_eq!(url, w[0].as_str().unwrap());
            n += 1;
            Ok(w[1].clone())
        };
        let g = dir_signals(&mut f, s["key"].as_str(), s["now"].as_f64().unwrap());
        assert!(q.is_empty());
        assert_eq!(g, s["out"].as_str().unwrap(), "dir_signals {}", s["now"]);
    }
    println!("7일·4~13주 신호 {} 회차 · 요청 {n} 글자 같음", fx["sig"].as_array().unwrap().len());
}

/// rt_dirfwd 실행 파일 첫 회차를 127.0.0.1 가짜 HTTP 로: 기록 글자 == 정답지 · 원본 자리(data/live) 거부(종료코드 2) · 시각 밖 = 종료코드 1.
#[test]
fn dirfwd_bin_over_fake_http() {
    use std::io::{Read, Write};
    let (_, fx) = fx();
    let r0 = fx["rec"][0].clone();
    let q: std::sync::Arc<std::sync::Mutex<VecDeque<(String, Value)>>> = std::sync::Arc::new(std::sync::Mutex::new(r0["calls"].as_array().unwrap().iter().map(|c| {
        (format!("/fapi/v1/klines?symbol={}&interval={}&limit={}", c[0].as_str().unwrap(), c[1].as_str().unwrap(), c[2]), c[3].clone())
    }).collect()));
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let qq = q.clone();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut buf = vec![0u8; 8192];
            let mut n = 0;
            while !buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                let k = s.read(&mut buf[n..]).unwrap();
                if k == 0 { break; }
                n += k;
            }
            let path = String::from_utf8_lossy(&buf[..n]).split(' ').nth(1).unwrap_or("").to_string();
            let body = match qq.lock().unwrap().pop_front() { Some((w, r)) if w == path => r.to_string(), o => format!("\"요청 {path} 대 원본 {:?}\"", o.map(|x| x.0)) };
            let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    let dir = std::env::temp_dir().join(format!("rt_dirfwd_bin_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let run = |now: i64, out: &Path| std::process::Command::new(env!("CARGO_BIN_EXE_rt_dirfwd")).env("RT_DIRFWD_OUT", out).env("RT_DIRFWD_REST", format!("http://127.0.0.1:{port}"))
        .env("RT_SR_BAN_FILE", dir.join("ban")).env("RT_DIRFWD_NOW_MS", now.to_string()).output().unwrap();
    let o = run(r0["now"].as_i64().unwrap(), &dir.join("out"));
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(q.lock().unwrap().is_empty());
    assert_eq!(std::fs::read_to_string(dir.join("out/eth_direction_forward.jsonl")).unwrap(), r0["file"].as_str().unwrap());
    assert_eq!(run(fx["rec"][3]["now"].as_i64().unwrap(), &dir.join("out")).status.code(), Some(1), "시각 밖");
    assert_eq!(run(r0["now"].as_i64().unwrap(), &dir.join("data/live")).status.code(), Some(2), "원본 자리 거부");
    let _ = std::fs::remove_dir_all(&dir);
    println!("rt_dirfwd 실행 파일: 요청 6 · 기록 글자 같음 · 시각 밖 1 · 거부 2");
}

/// handlers: 서버 핸들러 몫(쪽 나눠 받기 · URL · "asset" 덧붙이기)이 라이브러리 함수와 같은 본문을 내는지.
#[test]
fn handlers_bodies() {
    let (d, fx) = fx();
    let v = &fx["vol"][0];
    let rows = v["rows"].as_array().unwrap();
    let mut pages: VecDeque<Value> = rows.rchunks(1000).map(|c| Value::Array(c.to_vec())).collect();
    pages.push_back(Value::Array(vec![]));
    let (mut urls, mut end): (Vec<String>, Option<i64>) = (vec![], None);
    let mut f = |u: &str| -> Result<Value, String> {
        urls.push(u.to_string());
        let p = pages.pop_front().unwrap();
        let want = format!("http://fake/fapi/v1/klines?symbol=ETHUSDT&interval=5m&limit=1000{}", end.map_or(String::new(), |e| format!("&endTime={e}")));
        assert_eq!(u, want);
        end = p.as_array().and_then(|a| a.first()).map(|r| r[0].as_i64().unwrap() - 1);
        Ok(p)
    };
    let art = Svm::from_json(&rtmodel::read_json::<Value>(&d.join("svm_eth.json")).unwrap()).unwrap();
    let body = rtmodel::handlers::vol_forecast_hgb(&mut f, "http://fake", "ETHUSDT", "eth", &art, v["now"].as_f64().unwrap()).unwrap();
    let want = v["out"].as_str().unwrap();
    assert_eq!(body, format!("{}, \"asset\": \"eth\"}}", &want[..want.len() - 1]));
    assert_eq!(urls.len(), 10, "1000봉 × 10쪽(서버 원문 range(10))");
    let o = &fx["open_rev"][4]; // 서버 시계(실수 ms) 벌
    let mut q: VecDeque<Value> = VecDeque::from([o["daily"].clone(), o["m5"].clone()]);
    let mut g = |u: &str| -> Result<Value, String> {
        assert!(u.starts_with("http://fake/fapi/v1/klines?symbol=ETHUSDT&interval="));
        Ok(q.pop_front().unwrap())
    };
    assert_eq!(rtmodel::handlers::open_rev_body(&mut g, "http://fake", o["now"].as_f64().unwrap()).unwrap(), o["out"].as_str().unwrap());
    println!("핸들러: vol-forecast HGB 요청 {} · open-rev 본문 같음", urls.len());
}
