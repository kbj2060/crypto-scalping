//! SIGTERM 때 스냅숏(2026-11-21 · main.rs): 실제 실행 파일을 127.0.0.1 가짜 거래소(`@trade` 만 내보냄)에 붙여 띄우고 SIGTERM 을 보낸다 →
//! 종료코드 0 · 스냅숏 파일 1개 · 로그 «SIGTERM -- 스냅숏 1개». 체결 전에 보내면 «0개»(메우기 전엔 안 쓴다는 조건 그대로).
//! 거래소 주소는 전부 가짜(빈 값 = 그 원천 끔) -- 바깥 접속 없음.
use futures_util::SinkExt;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;

async fn fake_exchange(trades: bool) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (tcp, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut path = String::new();
                let cb = |req: &Request, resp: Response| {
                    path = req.uri().path().to_string();
                    Ok(resp)
                };
                let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(tcp, cb).await else { return };
                let mut id = 1u64;
                loop {
                    if trades && path.ends_with("@trade") {
                        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
                        let m = format!(r#"{{"e":"trade","E":{t},"T":{t},"s":"ETHUSDT","t":{id},"p":"3000.10","q":"0.500","X":"MARKET","m":false}}"#);
                        id += 1;
                        if ws.send(Message::text(m)).await.is_err() {
                            return;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
        }
    });
    port
}

fn run(port: u16, dir: &std::path::Path, wait: Duration) -> (i32, String) {
    let base = format!("ws://127.0.0.1:{port}/ws/");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rtcore"))
        .env("RTCORE_SYMBOLS", "ethusdt")
        .env("RTCORE_LISTEN", "127.0.0.1:0")
        .env("RTCORE_WS_BASE", &base)
        .env("RTCORE_MARKET_WS_BASE", &base)
        .env("RTCORE_FORCE_WS_URL", "")
        .env("RTCORE_SPOT_WS_BASE", "")
        .env("RTCORE_OKX_WS_URL", "")
        .env("RTCORE_BYBIT_WS_URL", "")
        .env("RTCORE_SNAP_DIR", dir)
        .env("RTCORE_TAPE_DB", dir.join("없는_테이프.sqlite"))
        .env("RTCORE_OKX_TAPE_DB", dir.join("없음1.sqlite"))
        .env("RTCORE_OKX_CTX_DB", dir.join("없음2.sqlite"))
        .env("RTCORE_BN_CTX_DB", dir.join("없음3.sqlite"))
        .env("RTCORE_BYBIT_LIQ_DB", dir.join("없음4.sqlite"))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(wait);
    assert!(Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap().success());
    let t = Instant::now();
    let code = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s.code().unwrap_or(-1);
        }
        assert!(t.elapsed() < Duration::from_secs(10), "SIGTERM 뒤 10초 안에 안 끝났다");
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut out = String::new();
    child.stdout.take().unwrap().read_to_string(&mut out).unwrap();
    (code, out)
}

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_writes_snapshot_once() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("sigterm_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // 체결이 오는 판: 첫 체결 → 메우기(테이프 없음 → 구멍만) → SIGTERM 에 스냅숏 1개
    let port = fake_exchange(true).await;
    let (code, out) = tokio::task::spawn_blocking({
        let d = dir.join("a");
        move || run(port, &d, Duration::from_millis(2500))
    })
    .await
    .unwrap();
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("rtcore SIGTERM -- 스냅숏 1개"), "{out}");
    let snap = rtcore::restore::load(&dir.join("a/ethusdt.json"), "ethusdt", rtcore::now_us() / 1000).expect("SIGTERM 이 쓴 스냅숏");
    assert!(snap.last_ms > 0 && !snap.bars.is_empty());
    // 체결이 없는 판: 메우기 전이라 안 쓴다
    let port = fake_exchange(false).await;
    let (code, out) = tokio::task::spawn_blocking({
        let d = dir.join("b");
        move || run(port, &d, Duration::from_millis(1500))
    })
    .await
    .unwrap();
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("rtcore SIGTERM -- 스냅숏 0개"), "{out}");
    assert!(!dir.join("b/ethusdt.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
