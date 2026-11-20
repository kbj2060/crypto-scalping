//! 사용자 스트림 전송(여섯째 조각 D): 127.0.0.1 가짜 HTTP(listenKey) + 가짜 웹소켓 서버로 «연결 → 사건 → 끊김 → 재연결 → listenKey 만료 → 재연결»과
//! 끊긴 사이 알림 누락 뒤 REST 대조(어긋남 → 스트림 계좌를 안 쓴다). 실제 거래소·네트워크 없음.
include!("common/server.rs");

use futures_util::SinkExt;
use rtorder::account::{user_stream_run, UserStream};
use rtorder::http::Binance;
use rtorder::userstream::{Hub, LiveStream};
use serde_json::json;
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn connect_events_reconnect_expired_and_rest_reconcile() {
    // listenKey HTTP -- 만들 때마다 lk1, lk2, …
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let http = serve(Arc::new(move |g: &Got| {
        assert!(g.target.starts_with("/fapi/v1/listenKey") && g.key.as_deref() == Some("test_key"), "{g:?}");
        let i = n2.fetch_add(1, Ordering::SeqCst) + 1;
        Act::Send { status: 200, ctype: "application/json".into(), body: json!({"listenKey": format!("lk{i}")}).to_string(), headers: vec![] }
    }));
    local_only(&http.base);
    // 웹소켓 -- 연결마다 각본 하나
    let scripts: Vec<Vec<String>> = vec![
        vec![json!({"e": "ORDER_TRADE_UPDATE", "o": {"s": "ETHUSDC", "x": "NEW", "X": "NEW", "o": "LIMIT", "c": "dbtpL1", "S": "SELL", "q": "0.4", "z": "0", "p": "2600"}}).to_string(),
             json!({"e": "ACCOUNT_UPDATE", "a": {"P": [{"s": "ETHUSDC", "ps": "LONG", "pa": "0.4", "ep": "2500", "up": "1"}]}}).to_string()],
        vec![json!({"e": "listenKeyExpired"}).to_string()],
        vec![json!({"e": "ORDER_TRADE_UPDATE", "o": {"s": "ETHUSDC", "x": "TRADE", "X": "FILLED", "o": "LIMIT", "c": "dbtpL9", "ps": "LONG", "S": "SELL", "q": "0.4", "z": "0.4", "p": "2600"}}).to_string()],
    ];
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ws_base = format!("ws://{}/private/ws/", lis.local_addr().unwrap());
    assert!(ws_base.starts_with("ws://127.0.0.1:"));
    let paths = Arc::new(Mutex::new(vec![]));
    let p2 = paths.clone();
    tokio::spawn(async move {
        for msgs in scripts {
            let (s, _) = lis.accept().await.unwrap();
            let p3 = p2.clone();
            let mut ws = tokio_tungstenite::accept_hdr_async(s, move |req: &tokio_tungstenite::tungstenite::handshake::server::Request, r| {
                p3.lock().unwrap().push(req.uri().path().to_string());
                Ok(r)
            }).await.unwrap();
            for m in msgs {
                ws.send(Message::Text(m.into())).await.unwrap();
            }
            let _ = ws.close(None).await;
        }
    });
    let ban = std::env::temp_dir().join(format!("rtorder_us_ws_{}", std::process::id()));
    let _ = std::fs::remove_file(&ban);
    let bx = Binance::new(&http.base, "test_key", "test_secret", std::time::Duration::from_secs(5), ban.clone());
    let hub = Arc::new(Hub::default());
    // 스트림 계좌 기준(REST 스냅숏: 포지션 없음) -- 첫 연결(epoch 1)에서 맞춰 둔 것으로
    hub.stream_account.lock().unwrap().reconcile(&json!({"ok": true, "positions": [], "open_orders": []}), 0.0, 1);
    let mut io = LiveStream::new(bx, &ws_base, hub.clone());
    io.sleep_scale = 0.001;
    io.max_sleeps = Some(3);
    let mut us = UserStream::default();
    tokio::time::timeout(std::time::Duration::from_secs(20), user_stream_run(&mut io, &mut us)).await.expect("루프가 끝나야");
    assert_eq!(*paths.lock().unwrap(), ["/private/ws/lk1", "/private/ws/lk2", "/private/ws/lk3"], "/private 경로(A4)");
    assert_eq!(n.load(Ordering::SeqCst), 4, "연결마다 listenKey POST + 넷째 시도");
    assert!(http.log.lock().unwrap().iter().all(|g| g.method == "POST"), "닫기(DELETE)는 안 부른다(A5)");
    assert_eq!(io.sleeps[..3], [5.0, 10.0, 20.0], "재연결 대기 5 → ×2(A7)");
    assert_eq!(us.epoch, 3);
    assert!(!us.connected);
    assert_eq!(*hub.tp_fills.lock().unwrap(), [("ETHUSDC".to_string(), "LONG".to_string(), "dbtpL9".to_string())]);
    let logs = hub.logs.lock().unwrap().clone();
    assert!(logs.iter().filter(|l| *l == "user stream: 체결 알림 연결").count() == 3 && logs.iter().any(|l| l == "user stream: 첫 알림 수신(fill)"), "{logs:?}"); // 우리 NEW 는 무시(A9) -- 첫 알림 = 계좌 변화
    // 스트림 계좌: 첫 연결 사건(포지션 0.4 · 익절 미체결)은 반영됐지만 그 뒤 재연결 -- 끊긴 사이 사건은 다시 안 온다 → 대조 전엔 안 쓴다
    let sa = hub.stream_account.lock().unwrap().clone();
    assert_eq!(sa.base.as_ref().unwrap()["positions"][0]["qty"], json!(0.4));
    assert_eq!(sa.usable(&UserStream { connected: true, ..us.clone() }, 1.0).unwrap_err(), "reconnected");
    // 알림 누락: REST 는 이미 닫힘(포지션 0) -- 첫 연결(epoch 1) 기준이 남은 채로 대조하면 어긋남
    let mut sa2 = sa.clone();
    assert!(!sa2.reconcile(&json!({"ok": true, "positions": [], "open_orders": []}), 2.0, 1), "스트림 0.4 대 REST 0 = 어긋남");
    assert_eq!(sa2.usable(&UserStream { connected: true, epoch: 1, ..Default::default() }, 3.0).unwrap_err(), "mismatch");
    let _ = std::fs::remove_file(&ban);
}
