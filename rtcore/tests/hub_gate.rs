//! 화면 문 단속(hub.rs 머리 주석) -- 가짜 클라이언트로. 수신(거래소) 없이 hub::serve 만 띄운다.
use futures_util::{SinkExt, StreamExt};
use rtcore::hub::{HubCfg, MAX_IN};
use rtcore::state::State;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::{Error, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
const OK: &str = "https://ok.example";

fn cfg() -> HubCfg {
    HubCfg { path: "/rt".into(), origins: vec![OK.into()], max_conns: 4, max_subs: 2, ping: Duration::from_secs(30) }
}

async fn hub(cfg: HubCfg) -> String {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let a = l.local_addr().unwrap().to_string();
    tokio::spawn(rtcore::hub::serve(l, State::new(&["ethusdt".to_string(), "btcusdt".to_string()], 8, 64), cfg));
    a
}

async fn open(addr: &str, path: &str, origin: Option<&str>) -> Result<Ws, Error> {
    let mut req = format!("ws://{addr}{path}").into_client_request().unwrap();
    if let Some(o) = origin {
        req.headers_mut().insert("origin", o.parse().unwrap());
    }
    tokio_tungstenite::connect_async(req).await.map(|(ws, _)| ws)
}

fn status(r: Result<Ws, Error>) -> u16 {
    match r {
        Err(Error::Http(resp)) => resp.status().as_u16(),
        Err(e) => panic!("HTTP 거절이 아니다: {e}"),
        Ok(_) => 101,
    }
}

async fn sub(ws: &mut Ws, syms: &[&str]) {
    ws.send(Message::text(serde_json::json!({ "subscribe": syms }).to_string())).await.unwrap();
}

/// 첫 글자 메시지의 type, 또는 끊김 사유.
async fn first(ws: &mut Ws) -> String {
    loop {
        match timeout(Duration::from_secs(3), ws.next()).await.expect("3초 안에 무엇이든") {
            Some(Ok(Message::Text(t))) => return serde_json::from_str::<serde_json::Value>(t.as_str()).unwrap()["type"].as_str().unwrap().into(),
            Some(Ok(Message::Close(f))) => return format!("close {:?}", f.map(|f| f.code)),
            Some(Ok(_)) => continue,
            Some(Err(e)) => return format!("err {e}"),
            None => return "end".into(),
        }
    }
}

// 1. /rt 만 받는다 -- 경로가 다르면 404, 기존 구독 흐름(스냅숏 먼저)은 그대로.
#[tokio::test]
async fn g1_only_rt_path() {
    let a = hub(cfg()).await;
    for p in ["/", "/x", "/rt/", "/rtx", "/rt/x", "/api/footprint"] {
        assert_eq!(status(open(&a, p, None).await), 404, "{p}");
    }
    let mut ws = open(&a, "/rt?x=1", None).await.expect("질의 문자열은 경로가 아니다");
    sub(&mut ws, &["ETHUSDT"]).await;
    assert_eq!(first(&mut ws).await, "snapshot");
}

// 2. Origin: 목록에 든 것만 · 머리가 없으면(브라우저 아님) 받는다 · 목록이 비면 브라우저는 하나도 못 붙는다.
#[tokio::test]
async fn g2_origin_allow_list() {
    let a = hub(cfg()).await;
    assert_eq!(status(open(&a, "/rt", Some("https://evil.example")).await), 403);
    assert_eq!(status(open(&a, "/rt", Some("https://ok.example.evil")).await), 403);
    assert_eq!(status(open(&a, "/rt", Some("null")).await), 403);
    assert_eq!(status(open(&a, "/rt", Some(OK)).await), 101);
    assert_eq!(status(open(&a, "/rt", None).await), 101);
    let b = hub(HubCfg { origins: vec![], ..cfg() }).await;
    assert_eq!(status(open(&b, "/rt", Some(OK)).await), 403, "빈 목록 = 브라우저 전부 거절");
    let mut ws = open(&b, "/rt", None).await.unwrap();
    sub(&mut ws, &["ethusdt"]).await;
    assert_eq!(first(&mut ws).await, "snapshot");
}

// 3. 동시 연결 상한: 넘으면 503 · 하나 닫으면 다시 붙는다.
#[tokio::test]
async fn g3_connection_cap() {
    let a = hub(HubCfg { max_conns: 2, ..cfg() }).await;
    let mut c1 = open(&a, "/rt", None).await.unwrap();
    let _c2 = open(&a, "/rt", None).await.unwrap();
    assert_eq!(status(open(&a, "/rt", None).await), 503);
    assert_eq!(status(open(&a, "/x", None).await), 404, "꽉 차도 경로 판정이 먼저");
    c1.close(None).await.unwrap();
    sleep(Duration::from_millis(200)).await;
    assert_eq!(status(open(&a, "/rt", None).await), 101, "닫은 연결의 칸이 돌아와야 한다");
}

// 4. 연결당 구독 심볼 상한: 넘으면 1008 로 닫고 스냅숏을 안 보낸다.
#[tokio::test]
async fn g4_subscribe_cap() {
    let a = hub(cfg()).await;
    let mut ws = open(&a, "/rt", None).await.unwrap();
    sub(&mut ws, &["ethusdt", "btcusdt", "solusdt"]).await;
    assert_eq!(first(&mut ws).await, format!("close {:?}", Some(CloseCode::Policy)));
    let mut ok = open(&a, "/rt", None).await.unwrap();
    sub(&mut ok, &["ethusdt", "btcusdt"]).await;
    assert_eq!(first(&mut ok).await, "snapshot", "상한까지는 받는다");
}

// 5. 큰 메시지(MAX_IN 초과)는 스냅숏 없이 끊는다.
#[tokio::test]
async fn g5_big_message_rejected() {
    let a = hub(cfg()).await;
    let mut ws = open(&a, "/rt", None).await.unwrap();
    let pad = "x".repeat(MAX_IN + 1);
    ws.send(Message::text(format!(r#"{{"subscribe":["ethusdt"],"pad":"{pad}"}}"#))).await.unwrap();
    let why = first(&mut ws).await;
    assert!(why != "snapshot", "큰 메시지를 받아 줬다");
    println!("g5 큰 메시지: {why}");
}

// 6. pong 없는 연결은 ping 두 번 안에 정리되어 칸을 돌려준다 · 읽는(= pong 하는) 연결은 남는다.
#[tokio::test]
async fn g6_ping_drops_silent_peer() {
    let ping = Duration::from_millis(200);
    let a = hub(HubCfg { max_conns: 2, ping, ..cfg() }).await;
    let mut live = open(&a, "/rt", None).await.unwrap();
    sub(&mut live, &["ethusdt"]).await;
    let reader = tokio::spawn(async move {
        let mut pings = 0;
        while let Ok(Some(Ok(m))) = timeout(Duration::from_millis(1500), live.next()).await {
            pings += matches!(m, Message::Ping(_)) as u32;
            if pings == 6 {
                return pings; // 1.2초 넘게 붙어 있었다
            }
        }
        pings
    });
    let mut deaf = open(&a, "/rt", None).await.unwrap();
    sub(&mut deaf, &["ethusdt"]).await; // 이후 안 읽는다 -- pong 도 안 나간다
    assert_eq!(status(open(&a, "/rt", None).await), 503, "둘 다 붙은 동안은 꽉 참");
    sleep(ping * 2 + Duration::from_millis(250)).await;
    assert_eq!(status(open(&a, "/rt", None).await), 101, "pong 없는 연결이 칸을 안 돌려줬다");
    assert_eq!(reader.await.unwrap(), 6, "pong 하는 연결이 끊겼다");
    let mut got = vec![];
    while let Ok(Some(Ok(m))) = timeout(Duration::from_secs(1), deaf.next()).await {
        got.push(m);
    }
    assert!(got.iter().any(|m| matches!(m, Message::Ping(_))), "ping 이 갔어야 한다");
}
