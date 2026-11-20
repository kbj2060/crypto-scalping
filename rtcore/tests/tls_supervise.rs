//! 2026-11-20 실서버 사고 재발 막기(네트워크 없음): ① wss 접속 경로가 rustls 암호 제공자 패닉 없이 돈다
//! ② 수신 태스크가 패닉하면 first_exit 가 알아챈다. 가짜 거래소는 ws:// 라 ①을 못 잡았다.
use rtcore::state::State;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::sleep;

#[tokio::test]
async fn wss_connect_path_does_not_panic() {
    // TLS 를 모르는 TCP 서버: 받자마자 끊는다 -> 핸드셰이크는 실패하지만 그 전에 rustls 설정을 만든다(패닉 자리).
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    let accepts = Arc::new(AtomicUsize::new(0));
    let a2 = accepts.clone();
    tokio::spawn(async move {
        loop {
            let (tcp, _) = l.accept().await.unwrap();
            a2.fetch_add(1, Relaxed);
            drop(tcp);
        }
    });
    let state = State::new(&["ethusdt".to_string()], 8, 8);
    let h = tokio::spawn(rtcore::feed::run(format!("wss://127.0.0.1:{port}/ws/ethusdt@trade"), state, Duration::from_secs(5)));
    sleep(Duration::from_millis(1500)).await;
    assert!(accepts.load(Relaxed) >= 2, "재시도까지 가야 한다(접속 {}회)", accepts.load(Relaxed));
    assert!(!h.is_finished(), "수신 태스크가 죽었다(rustls 제공자 패닉?)");
    h.abort();
}

#[tokio::test]
async fn first_exit_reports_panicked_feed() {
    let ok = tokio::spawn(async { sleep(Duration::from_secs(60)).await });
    let bad = tokio::spawn(async { panic!("일부러") });
    let why = rtcore::first_exit(vec![("a".into(), ok), ("wss://x/ethusdt@trade".into(), bad)]).await;
    assert_eq!(why, "수신 태스크 패닉 wss://x/ethusdt@trade");
}
