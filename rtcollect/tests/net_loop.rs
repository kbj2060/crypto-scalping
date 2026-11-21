//! 실행 파일(rtcollect) 접속 루프 시험 -- 거래소 대신 127.0.0.1 가짜 웹소켓 서버(네트워크 밖으로 안 나간다).
//! ① «접속 → 메시지 → 끊김 → 재접속 → 메시지 → 침묵 → 재접속 → 메시지 → SIGTERM» 의 DB 가, 같은 사건 열을 `binance::replay` 에
//!    넣은 DB 와 표 전체가 같다(체결 시각이 차례대로라 플러시 시점은 행을 안 바꾼다 -- 늦은 체결이 없으면 닫힌 초는 같은 값).
//! ② 쓰기가 멈추면(체결 아닌 메시지만 온다) 정지 감시가 종료코드 70 으로 끝낸다(문턱을 2초로 줄여서).
use futures_util::SinkExt;
use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
use std::sync::Arc;
use rusqlite::{types::ValueRef, Connection, OpenFlags};
use serde_json::json;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rtcollect_net_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn dump(db: &Path) -> Vec<String> {
    let con = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut out = vec![];
    for (t, o) in [("trade_tape_1s", "symbol, ts_sec, price_bin"), ("gaps", "rowid"), ("meta", "rowid"), ("verify_1m", "rowid")] {
        let mut st = con.prepare(&format!("SELECT * FROM {t} ORDER BY {o}")).unwrap();
        let n = st.column_count();
        let mut rows = st.query([]).unwrap();
        while let Some(r) = rows.next().unwrap() {
            let cells: Vec<String> = (0..n)
                .map(|j| match r.get_ref(j).unwrap() {
                    ValueRef::Real(f) => format!("{:016x}R", f.to_bits()),
                    ValueRef::Integer(i) => format!("{i}I"),
                    ValueRef::Text(s) => format!("{:?}", String::from_utf8_lossy(s)),
                    ValueRef::Null => "NULL".into(),
                    ValueRef::Blob(_) => "BLOB".into(),
                })
                .collect();
            out.push(format!("{t}|{}", cells.join("|")));
        }
    }
    out
}

/// 연결 k 의 체결들: 초 3개, 초마다 테이커 주문 몇 개(같은 가격·연속 ID) -- 시각은 늘 앞으로.
fn trades(k: u64) -> Vec<String> {
    let t0: u64 = 1_763_683_200_000 + 3_437 + k * 7_000; // 분 시작 3.4초 뒤 -- startup 공백(1.5초 문턱 위)
    let mut id = 50_000_000 + k * 1_000;
    let mut out = vec![];
    for s in 0..3u64 {
        for o in 0..4u64 {
            let (p, m) = (format!("{:.2}", 2500.0 + o as f64 * 0.37 + k as f64), o % 2 == 1);
            for f in 0..(1 + o % 3) {
                id += 1;
                let ts = t0 + s * 1000 + o * 150 + f * 3;
                out.push(json!({"e": "trade", "E": ts + 2, "T": ts, "t": id, "p": p, "q": format!("{:.3}", 0.05 + (o * 7 + f) as f64 * 0.9), "m": m}).to_string());
            }
        }
    }
    out
}

#[tokio::test]
async fn loop_matches_replay_through_close_silence_and_sigterm() {
    let dir = tmpdir("loop");
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let sent3 = Arc::new(AtomicBool::new(false));
    let sent3_tx = sent3.clone();
    tokio::spawn(async move {
        for k in 0..3u64 {
            let (s, _) = l.accept().await.unwrap();
            let done = sent3_tx.clone();
            tokio::spawn(async move {
                let mut ws = accept_async(s).await.unwrap();
                for m in trades(k) {
                    ws.send(Message::text(m)).await.unwrap();
                    tokio::time::sleep(Duration::from_millis(3)).await;
                }
                match k {
                    0 => ws.close(None).await.unwrap(),                   // 끊김
                    _ => {
                        if k == 2 {
                            done.store(true, SeqCst);
                        }
                        tokio::time::sleep(Duration::from_secs(60)).await; // 침묵(안 읽으니 pong 도 없다)
                    }
                }
            });
        }
    });
    let db = dir.join("rs").join("binance_tape.sqlite");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rtcollect"))
        .env("TAPE_SYMBOL", "ethusdt")
        .env("TAPE_MARKET", "futures")
        .env("TAPE_DB_PATH", &db)
        .env("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/{{symbol}}@trade"))
        .env("RTCOLLECT_SILENCE_S", "2")
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let t0 = std::time::Instant::now();
    while !sent3.load(SeqCst) {
        assert!(t0.elapsed() < Duration::from_secs(40), "셋째 접속이 안 왔다");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let st = tokio::task::spawn_blocking(move || child.wait().unwrap()).await.unwrap();
    assert_eq!(st.code(), Some(0), "SIGTERM 은 닫힌 초를 쓰고 0 으로 끝난다");

    // 같은 사건 열을 replay 에: 연결마다 글자 메시지 → 'c'(끊김·침묵·SIGTERM 모두 «진행 중인 초를 빼고 쓴다»)
    let mut ev = vec![];
    let mut t = 1.0;
    for k in 0..3 {
        for m in trades(k) {
            t += 0.003;
            ev.push(json!([t, "t", m]));
        }
        ev.push(json!([t, "c", ""]));
        t += 5.0;
    }
    let rd = dir.join("replay");
    rtcollect::binance::replay(&[json!({"symbol": "ethusdt", "market": "futures", "t0": 0.5, "events": ev})], &rd, |_, _| {}).unwrap();
    let (a, b) = (dump(&db), dump(&rd.join("binance_tape.sqlite")));
    assert_eq!(a, b, "실행 파일 DB != replay DB");
    let gaps = a.iter().filter(|r| r.starts_with("gaps|")).count();
    let rows = a.iter().filter(|r| r.starts_with("trade_tape_1s|")).count();
    assert_eq!(gaps, 3, "startup 1 + 재연결 2: {a:#?}");
    // 3연결 × 3초 × 4칸(가격 넷) -- 끊길 때 진행 중이던 초는 버퍼에 남아 다음 끊김에 나간다, 마지막 연결의 마지막 초만 안 쓴다
    assert_eq!(rows, 3 * 3 * 4 - 4, "{a:#?}");
    eprintln!("접속 루프 = replay: 행 {} (테이프 {rows} · gaps {gaps})", a.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn watchdog_exits_70_when_writes_stall() {
    let dir = tmpdir("wd");
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut ws = accept_async(s).await.unwrap();
                loop {
                    // 체결 아닌 메시지만 -- 수신은 살아 있고(침묵 아님) 쓰기만 없다
                    if ws.send(Message::text(r#"{"e":"24hrTicker","c":"2500.0"}"#)).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
            });
        }
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_rtcollect"))
        .env("TAPE_SYMBOL", "ethusdt")
        .env("TAPE_DB_PATH", dir.join("t.sqlite"))
        .env("RTCOLLECT_WS_URL", format!("ws://127.0.0.1:{port}/ws/{{symbol}}@trade"))
        .env("RTCOLLECT_WD_WRITE_S", "2")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let t0 = std::time::Instant::now();
    let out = tokio::time::timeout(Duration::from_secs(20), tokio::task::spawn_blocking(move || {
        let st = child.wait().unwrap();
        let mut e = String::new();
        std::io::Read::read_to_string(child.stderr.as_mut().unwrap(), &mut e).unwrap();
        (st, e)
    }))
    .await
    .expect("20초 안에 안 끝났다")
    .unwrap();
    assert_eq!(out.0.code(), Some(70), "{}", out.1);
    assert!(out.1.contains("쓰기 성공이"), "{}", out.1);
    eprintln!("정지 감시: {:.1}초 만에 종료코드 70 · {}", t0.elapsed().as_secs_f64(), out.1.trim());
    let _ = std::fs::remove_dir_all(&dir);
}
