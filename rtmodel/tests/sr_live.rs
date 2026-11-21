//! 레벨 5분 «메시지 처리 루프»(sr_live::Runner)와 실행 파일(rt_sr_lean)을 정답지(tests/sr_lean.rs 와 같은 sr_fixture.json)로 대조.
//! 정답지의 연산 열(체결·1분봉·다시 확인·구멍)을 aggTrade 메시지와 REST 응답으로 되돌려 넣으면 파이썬이 쓴 줄과 글자까지 같아야 한다.
//! ① Runner 직접(라이브러리): 모든 벌 + 번호 빠짐을 REST 로 메움 + 겹친 메시지
//! ② 실행 파일: 127.0.0.1 가짜 웹소켓·HTTP 서버로 «접속 → 메시지 → 끊김 → 재접속(번호 빠짐 → REST 메움) → 구멍(REST 실패) → 1분봉 다시»
//!     cargo test -j 2 --release --test sr_live -- --nocapture       (정답지: tests/sr_lean.rs 머리말)
use futures_util::SinkExt;
use rtmodel::sr_live::Runner;
use rtmodel::*;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// 정답지 한 벌 → (aggTrade 메시지 열(묶음 번호 a 붙임 · 구멍 자리는 번호 5개를 건너뜀), 1분봉 응답 열(받는 차례대로), 구멍 뒤 첫 메시지 번호들).
struct Script {
    msgs: Vec<Value>,
    klines: VecDeque<(bool, Value)>, // (다시 확인(limit=10)인가, 줄들)
    /// 구멍 자리에서 건너뛴 묶음 번호(lo, hi) -- REST 가 이 번호를 메우려 하면 실패를 준다(원본: 번호 메움 실패 → 구멍)
    holes: Vec<(i64, i64)>,
}

fn fails(holes: &[(i64, i64)], pq: &str) -> bool {
    let lo: i64 = pq.split("fromId=").nth(1).unwrap().split('&').next().unwrap().parse().unwrap();
    holes.iter().any(|&(a, b)| a <= lo && lo <= b)
}

fn script(set: &Value) -> Script {
    let (mut msgs, mut klines, mut holes) = (vec![], VecDeque::new(), vec![]);
    let mut a = 7_000_000_000i64;
    for op in set["ops"].as_array().unwrap() {
        let o = op.as_array().unwrap();
        match o[0].as_str().unwrap() {
            "trade" => {
                a += 1;
                let t = o[1].as_i64().unwrap();
                msgs.push(json!({"e": "aggTrade", "E": t + 3, "s": "ETHUSDT", "a": a, "p": format!("{}", o[2].as_f64().unwrap()), "q": format!("{}", o[3].as_f64().unwrap()),
                                 "f": o[4], "l": o[5], "T": t, "m": o[6]}));
            }
            "klines" => klines.push_back((false, o[1].clone())),
            "recheck" => klines.push_back((true, o[1].clone())),
            "hole" => {
                holes.push((a + 1, a + 5)); // 번호 메움 실패 → 다음 체결 초 기준 구멍(정답지는 다음 체결 앞에서 hole)
                a += 5;
            }
            _ => {} // init·close 는 루프가 스스로 한다
        }
    }
    Script { msgs, klines, holes }
}

fn fixture() -> (SrLean, Value) {
    let art: SrLean = read_json(&dir().join("sr_lean.json")).expect("모델 없음 -- tests/sr_lean.rs 머리말");
    let fx: Value = read_json(&dir().join("sr_fixture.json")).expect("정답지 없음 -- tests/sr_lean.rs 머리말");
    (art, fx)
}

fn want(set: &Value, k: &str) -> Vec<String> {
    set[k].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

/// Runner 를 직접: 메시지 열(빠뜨릴 번호·겹쳐 보낼 자리 지정) + 가짜 REST(1분봉 = 정답지 차례 · aggTrades = 빠뜨린 체결 · 구멍 자리는 실패).
fn drive(art: &SrLean, sc: &Script, drop: std::ops::Range<usize>, dup_at: Option<usize>) -> (Vec<String>, Vec<String>, usize) {
    let mut run = Runner::new(art);
    let mut kq = sc.klines.clone();
    let dropped: Vec<Value> = sc.msgs[drop.clone()].to_vec();
    let (mut log, mut feat, mut calls) = (vec![], vec![], 0usize);
    let now = std::cell::Cell::new(0.0);
    let mut clock = || now.get();
    let mut nap = |_s: f64| {};
    for (i, m) in sc.msgs.iter().enumerate() {
        if drop.contains(&i) {
            continue;
        }
        now.set(m["T"].as_f64().unwrap() / 1000.0); // 벽시계 = 거래소 시각(1분봉 간격 60초가 정답지와 같은 자리에서 풀린다)
        let mut rest = |pq: &str| -> Result<Value, String> {
            calls += 1;
            if pq.starts_with("aggTrades") {
                if fails(&sc.holes, pq) {
                    return Err("ClientResponseError: 500".into());
                }
                let lo: i64 = pq.split("fromId=").nth(1).unwrap().split('&').next().unwrap().parse().unwrap();
                return Ok(Value::Array(dropped.iter().filter(|d| d["a"].as_i64().unwrap() >= lo).cloned().collect()));
            }
            let (re, rows) = kq.pop_front().expect("정답지보다 1분봉 요청이 많다");
            assert_eq!(re, pq.contains("limit=10&") || pq.ends_with("limit=10"), "1분봉 요청 차례가 정답지와 다르다: {pq}");
            Ok(rows)
        };
        run.on_msg(Some(&m.to_string()), &mut rest, &mut clock, &mut nap).unwrap();
        if dup_at == Some(i) {
            run.on_msg(Some(&sc.msgs[i - 3].to_string()), &mut rest, &mut clock, &mut nap).unwrap(); // 옛 번호 = 버림
        }
        log.append(&mut run.live.log);
        feat.append(&mut run.live.feat);
    }
    assert!(kq.is_empty(), "정답지의 1분봉 응답이 남았다 {}", kq.len());
    (log, feat, calls)
}

#[test]
fn runner_matches_python_lines() {
    let (art, fx) = fixture();
    let mut n = 0;
    for set in fx["sets"].as_array().unwrap() {
        let name = set["name"].as_str().unwrap();
        let sc = script(set);
        let (log, feat, _) = drive(&art, &sc, 0..0, None);
        assert_eq!(log, want(set, "main"), "{name}: 기록 줄");
        assert_eq!(feat, want(set, "feat"), "{name}: 피쳐 줄");
        n += log.len();
        if name == "normal0" || name == "range0" {
            // 번호 5개를 웹소켓이 빠뜨림 → REST 가 메움 · 옛 번호 메시지 = 버림 → 같은 줄
            let k = sc.msgs.len() * 2 / 3;
            let (log2, feat2, calls) = drive(&art, &sc, k..k + 5, Some(k + 20));
            assert_eq!((log2, feat2), (want(set, "main"), want(set, "feat")), "{name}: 번호 메움·겹침");
            assert!(calls > sc.klines.len(), "aggTrades 를 불러야 한다");
        }
    }
    println!("Runner 직접: 벌 {} · 기록 줄 {n} 글자 같음 (+ 번호 메움·겹친 메시지 2벌)", fx["sets"].as_array().unwrap().len());
}

/// 가짜 HTTP(GET 한 줄 → JSON). klines 는 정답지 차례 · aggTrades 는 빠뜨린 체결(구멍 뒤엔 500).
async fn http(l: TcpListener, klines: Arc<Mutex<VecDeque<(bool, Value)>>>, dropped: Arc<Vec<Value>>, holes: Arc<Vec<(i64, i64)>>, log: Arc<Mutex<Vec<String>>>) {
    loop {
        let (mut s, _) = l.accept().await.unwrap();
        let (klines, dropped, holes, log) = (klines.clone(), dropped.clone(), holes.clone(), log.clone());
        tokio::spawn(async move {
            let mut buf = vec![0u8; 8192];
            let mut got = 0;
            while !buf[..got].windows(4).any(|w| w == b"\r\n\r\n") {
                let n = s.read(&mut buf[got..]).await.unwrap();
                if n == 0 {
                    return;
                }
                got += n;
            }
            let line = String::from_utf8_lossy(&buf[..got]).lines().next().unwrap().to_string();
            let path = line.split(' ').nth(1).unwrap().to_string();
            log.lock().unwrap().push(path.clone());
            let (status, body) = if path.contains("/aggTrades") {
                if fails(&holes, &path) {
                    (500, "{}".to_string())
                } else {
                    let lo: i64 = path.split("fromId=").nth(1).unwrap().split('&').next().unwrap().parse().unwrap();
                    (200, Value::Array(dropped.iter().filter(|d| d["a"].as_i64().unwrap() >= lo).cloned().collect()).to_string())
                }
            } else {
                let (re, rows) = klines.lock().unwrap().pop_front().expect("1분봉 요청이 많다");
                assert_eq!(re, path.contains("limit=10&") || path.ends_with("limit=10"), "{path}");
                (200, rows.to_string())
            };
            let r = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            let _ = s.write_all(r.as_bytes()).await;
        });
    }
}

/// 실행 파일 하나를 정답지 한 벌로: 메시지 열을 `cut`(웹소켓이 끊기는 자리 · 다음 연결은 5개를 건너뛰고 시작) 대로 보낸다.
async fn run_binary(set: &Value, cut: usize, tag: &str) -> (Vec<String>, Vec<String>, Vec<String>, String) {
    let sc = script(set);
    let out = std::env::temp_dir().join(format!("rt_sr_lean_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    std::fs::copy(dir().join("sr_lean.json"), out.join("sr_lean.json")).unwrap();
    let wl = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hl = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (wp, hp) = (wl.local_addr().unwrap().port(), hl.local_addr().unwrap().port());
    let dropped = Arc::new(sc.msgs[cut..cut + 5].to_vec());
    let rlog = Arc::new(Mutex::new(vec![]));
    tokio::spawn(http(hl, Arc::new(Mutex::new(sc.klines.clone())), dropped, Arc::new(sc.holes.clone()), rlog.clone()));
    let msgs = sc.msgs.clone();
    let ws_done = Arc::new(Mutex::new(false));
    let wd = ws_done.clone();
    tokio::spawn(async move {
        let parts = [(0, cut), (cut + 5, msgs.len())];
        for (k, (a, b)) in parts.into_iter().enumerate() {
            let (s, _) = wl.accept().await.unwrap();
            let mut ws = accept_async(s).await.unwrap();
            for m in &msgs[a..b] {
                ws.send(Message::text(m.to_string())).await.unwrap();
            }
            ws.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(1500)).await;
            if k == 0 {
                let _ = ws.close(None).await; // 끊김 → 실행 파일은 5초 뒤 재접속
            } else {
                *wd.lock().unwrap() = true;
                tokio::time::sleep(Duration::from_secs(3600)).await;
            }
        }
    });
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rt_sr_lean"))
        .env("RT_SR_OUT", &out).env("RT_SR_WS", format!("ws://127.0.0.1:{wp}/market/ws/ethusdt@aggTrade")).env("RT_SR_REST", format!("http://127.0.0.1:{hp}/fapi/v1/"))
        .env("RT_SR_BAN_FILE", out.join("ban_until")).env("RT_SR_KL_GAP_S", "0")
        .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    // 다 보낸 뒤: 기록 줄 수가 정답지에 닿거나 120초까지(실행 파일이 쌓인 메시지를 처리할 시간)
    let n_want = set["main"].as_array().unwrap().len();
    for _ in 0..600 {
        let n = std::fs::read_to_string(out.join("ethusdt.jsonl")).map_or(0, |s| s.lines().count());
        if *ws_done.lock().unwrap() && n >= n_want {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    tokio::time::sleep(Duration::from_secs(1)).await;
    let _ = child.kill();
    let o = child.wait_with_output().unwrap();
    let rd = |f: &str| std::fs::read_to_string(out.join(f)).unwrap_or_default().lines().map(str::to_string).collect::<Vec<_>>();
    let st = std::fs::read_to_string(out.join("ethusdt_state.json")).unwrap_or_default();
    let stdout = String::from_utf8_lossy(&o.stdout).to_string();
    assert!(stdout.contains("웹소켓 연결"), "{stdout}\n{}", String::from_utf8_lossy(&o.stderr));
    let calls = rlog.lock().unwrap().clone();
    (rd("ethusdt.jsonl"), rd("ethusdt_feat.jsonl"), calls, st + "\n" + &stdout)
}

#[tokio::test]
async fn binary_over_fake_exchange() {
    let (_, fx) = fixture();
    for name in ["normal0", "hole"] {
        let set = fx["sets"].as_array().unwrap().iter().find(|s| s["name"] == name).unwrap();
        let n = script(set).msgs.len();
        let (log, feat, calls, info) = run_binary(set, n / 2, name).await;
        let reconnects = info.matches("웹소켓 연결").count();
        assert_eq!(reconnects, 2, "{name}: 재접속 한 번\n{info}");
        assert!(calls.iter().any(|c| c.contains("/aggTrades")), "{name}: 번호 메움 요청");
        assert_eq!(log, want(set, "main"), "{name}: 기록 줄");
        assert_eq!(feat, want(set, "feat"), "{name}: 피쳐 줄");
        let state: Value = serde_json::from_str(info.lines().next().unwrap()).expect("생존 표시 파일");
        assert!(state["done"].is_i64() && state["rate"]["top"][0] == 989);
        println!("실행 파일 {name}: 메시지 {n} · 접속 {reconnects} · REST {} · 기록 줄 {} · 피쳐 줄 {} 글자 같음 · 구멍 {}", calls.len(), log.len(), feat.len(),
                 info.contains("구멍 처리"));
    }
}

#[test]
fn refuses_python_dir() {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rt_sr_lean")).env("RT_SR_OUT", "/tmp/x/data/live/sr_lean").output().unwrap();
    assert_eq!(o.status.code(), Some(2), "{}", String::from_utf8_lossy(&o.stderr));
}
