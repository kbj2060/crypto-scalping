//! rtcore 허브 구독(2026-11-21 E): 127.0.0.1 가짜 허브(rtcore hub.rs 와 같은 순서 -- 구독 메시지 → 심볼마다 스냅숏 → 밀기)로
//! «붙음 → 받아 둔 값 사용(REST 0) → 끊김 → 나이가 차 REST 로 떨어짐 → 재연결 스냅숏 → 다시 받아 둔 값» · 5분봉·1초가(감시) · 모르는 칸·모르는 type 무시.
use futures_util::{SinkExt, StreamExt};
use rtorder::assemble::{Cached, Quotes};
use rtorder::coreval::{self, Feed};
use rtorder::plan::Filters;
use serde_json::{json, Value};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Message;

#[derive(Default)]
struct Rest {
    books: u32,
    vols: u32,
}
impl Quotes for Rest {
    async fn book(&mut self, _: &str) -> Result<Value, String> {
        self.books += 1;
        Ok(json!({"rest": true}))
    }
    async fn filters(&mut self, _: &str) -> Result<Filters, String> {
        unreachable!()
    }
    async fn vol(&mut self, _: &str) -> Option<f64> {
        self.vols += 1;
        Some(-1.0)
    }
    async fn funding(&mut self, _: &str) -> Option<f64> {
        unreachable!()
    }
    async fn fees(&mut self, _: &str) -> Value {
        unreachable!()
    }
}

fn now_us() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as i64
}

/// 연결 n 번째에 보낼 메시지(rtcore proto 모양 그대로 + 모르는 칸 하나).
fn script(n: usize, rx: i64, bar_end: i64) -> Vec<Value> {
    let bid = if n == 0 { "2500.10" } else { "2501.30" };
    vec![
        json!({"s": "ethusdc", "seq": 9, "ex_ms": 1, "rx_us": rx, "tx_us": rx, "type": "snapshot", "trades": [], "footprint": null, "mark": null, "new_field": 1,
               "book": {"s": "ethusdc", "seq": 8, "ex_ms": 1, "rx_us": rx, "tx_us": 0, "type": "book", "u": 1, "bid": 2500.1, "bid_qty": 1.0, "ask": 2500.2, "ask_qty": 1.0,
                        "bid_s": bid, "ask_s": "2500.20"},
               "supply": {"now": 0, "seconds": []},
               "quote": {"closes_1m": [], "vol_bpm": 4.5, "kline_rx_us": rx, "bar5m": null, "last": null}}),
        json!({"s": "ethusdt", "seq": 3, "ex_ms": 1, "rx_us": rx, "tx_us": rx, "type": "snapshot", "trades": [], "footprint": null, "mark": null, "book": null,
               "supply": {"now": 0, "seconds": [[bar_end - 2, 0, 0, 0, 0, 0, 0, 2499.5], [bar_end - 1, 0, 0, 0, 0, 0, 0, 2499.7]]},
               "quote": {"closes_1m": [], "vol_bpm": null, "kline_rx_us": rx, "bar5m": [bar_end, 2499.7], "last": null}}),
        json!({"s": "*", "seq": 1, "ex_ms": 0, "rx_us": rx, "tx_us": rx, "type": "prices", "rows": []}),
        json!({"s": "ethusdt", "seq": 4, "ex_ms": 1, "rx_us": rx, "tx_us": rx, "type": "supply1s", "now": 0, "seconds": [[bar_end, 0, 0, 0, 0, 0, 0, 2500.9]]}),
        json!({"s": "ethusdc", "seq": 10, "ex_ms": 1, "rx_us": rx, "tx_us": rx, "type": "kline", "open_ms": 1, "close": 1.0, "close_s": "1", "vol_bpm": if n == 0 { json!(5.5) } else { Value::Null }, "bar5m": null}),
        json!({"s": "ethusdc", "seq": 11, "ex_ms": 1, "rx_us": rx, "tx_us": rx, "type": "someday_new_type", "x": 1}),
    ]
}

#[tokio::test]
async fn subscribe_uses_cached_then_falls_back_on_disconnect_then_recovers() {
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/rt", lis.local_addr().unwrap());
    assert!(url.starts_with("ws://127.0.0.1:"));
    let bar_end = (now_us() / 1_000_000 / 300) * 300; // 방금 마감된 5분봉 끝
    let subs: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(vec![]));
    let (go2, s2) = (Arc::new(tokio::sync::Notify::new()), subs.clone());
    let go = go2.clone();
    tokio::spawn(async move {
        for n in 0..2 {
            let (s, _) = lis.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(s).await.unwrap();
            if let Some(Ok(Message::Text(t))) = ws.next().await {
                s2.lock().unwrap().push(serde_json::from_str(t.as_str()).unwrap());
            }
            for m in script(n, now_us(), bar_end) {
                ws.send(Message::Text(m.to_string().into())).await.unwrap();
            }
            if n == 0 {
                go2.notified().await; // 시험이 «끊어라»고 할 때까지 붙어 있는다
                let _ = ws.close(None).await;
            } else {
                go2.notified().await;
            }
        }
    });
    let feed = Arc::new(Mutex::new(Feed::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let sub = tokio::spawn(coreval::subscribe_owned(url, feed.clone(), stop.clone()));
    let wait = |want: u64| {
        let f = feed.clone();
        async move {
            for _ in 0..400 {
                {
                    let g = f.lock().unwrap();
                    if g.connects >= want && g.vols.contains_key("ETHUSDC") && g.secs.get("ETHUSDT").is_some_and(|v| v.len() == 3) {
                        return;
                    }
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            panic!("허브 값이 안 왔다");
        }
    };
    wait(1).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert_eq!(subs.lock().unwrap()[0]["subscribe"][0], "ethusdc", "첫 메시지 = 구독");
    // ① 붙음: 받아 둔 값(글자 그대로) · REST 0
    let mut c = Cached::new(Rest::default());
    let t_now = now_us() as f64 / 1e6;
    c.now = t_now;
    coreval::sync_cached(&feed.lock().unwrap(), &mut c);
    assert_eq!(c.book("ETHUSDC").await.unwrap(), json!({"symbol": "ETHUSDC", "bidPrice": "2500.10", "askPrice": "2500.20"}));
    assert_eq!(c.vol("ETHUSDC").await, Some(5.5), "스냅숏 뒤 kline 밀기가 덮는다");
    assert_eq!((c.inner.books, c.inner.vols), (0, 0));
    {
        let f = feed.lock().unwrap();
        assert_eq!(coreval::bars(f.bar5m.get("ETHUSDT").copied(), t_now), [json!({"time": bar_end - 300, "close": 2499.7})], "감시 5분봉 = 화면 심볼의 마감 봉");
        assert_eq!(coreval::sec_prices(f.secs.get("ETHUSDT").map(Vec::as_slice)).unwrap().last().copied(), Some((bar_end as f64, 2500.9)), "1초가 밀기가 이어 붙는다");
        assert!(f.connected);
    }
    // ② 끊김: 값은 남고 나이만 -- 3초 뒤로 보면 호가는 REST 로(변동성 60초는 아직 받아 둔 값)
    go.notify_one();
    for _ in 0..200 {
        if !feed.lock().unwrap().connected {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(!feed.lock().unwrap().connected, "끊김을 안다");
    c.now = t_now + 3.0;
    coreval::sync_cached(&feed.lock().unwrap(), &mut c);
    assert_eq!(c.book("ETHUSDC").await.unwrap(), json!({"rest": true}), "끊기고 2초 넘음 = REST");
    assert_eq!(c.inner.books, 1);
    // ③ 다시 붙음(1초 뒤) → 새 스냅숏 = 다시 받아 둔 값 · 새 연결의 변동성 None(31개 전) = REST 로
    wait(2).await;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    c.now = now_us() as f64 / 1e6;
    coreval::sync_cached(&feed.lock().unwrap(), &mut c);
    assert_eq!(c.book("ETHUSDC").await.unwrap()["bidPrice"], "2501.30", "재연결 스냅숏 값");
    assert_eq!(c.inner.books, 1, "다시 받아 둔 값(REST 안 탐)");
    assert_eq!(c.vol("ETHUSDC").await, Some(-1.0), "변동성 None 이면 REST");
    assert_eq!(subs.lock().unwrap().len(), 2, "다시 구독을 보냈다");
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    go.notify_one();
    sub.abort();
}

/// 가짜 허브: 접속마다 구독을 읽고 -- `greet` 면 메시지 하나 보낸 뒤 -- Close(Policy, 사유)로 닫는다.
async fn refusing_hub(greet: bool) -> String {
    use tokio_tungstenite::tungstenite::protocol::{frame::coding::CloseCode, CloseFrame};
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/rt", lis.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let (s, _) = lis.accept().await.unwrap();
            tokio::spawn(async move {
                let mut ws = tokio_tungstenite::accept_async(s).await.unwrap();
                let _ = ws.next().await;
                if greet {
                    let m = json!({"s": "ethusdt", "rx_us": now_us(), "type": "supply1s", "seconds": [[now_us() / 1_000_000, 0, 0, 0, 0, 0, 0, 2500.0]]});
                    let _ = ws.send(Message::Text(m.to_string().into())).await;
                }
                let _ = ws.close(Some(CloseFrame { code: CloseCode::Policy, reason: "구독 심볼 10개 > 상한 8".into() })).await;
                while let Some(Ok(_)) = ws.next().await {}
            });
        }
    });
    url
}

#[tokio::test]
async fn refused_subscribe_backs_off_and_first_message_resets_to_one_second() {
    let run = |greet: bool| async move {
        let url = refusing_hub(greet).await;
        let feed = Arc::new(Mutex::new(Feed::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let h = tokio::spawn(coreval::subscribe_owned(url, feed.clone(), stop.clone()));
        tokio::time::sleep(std::time::Duration::from_millis(4500)).await;
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        h.abort();
        let f = feed.lock().unwrap();
        (f.connects, f.logs.clone(), f.health(coreval::wall()))
    };
    let ((n_ref, logs, h_ref), (n_ok, _, h_ok)) = tokio::join!(run(false), run(true));
    // 거절: 0 · 1 · 3(· 7) 초 = 4.5초에 3번 · 옛 판(보내기 성공이면 1초로) = 5번
    assert_eq!(n_ref, 3, "거절이면 1·2·4초로 늘어난다");
    // 받은 뒤엔 1초: 0 · 1 · 2 · 3 · 4 초 = 5번
    assert!(n_ok >= 4, "첫 메시지를 받으면 1초로 돌아온다 ({n_ok})");
    assert_eq!(logs.len(), 1, "같은 닫힘 사유는 1분에 한 줄: {logs:?}");
    assert!(logs[0].contains("닫힘 1008 구독 심볼 10개 > 상한 8"), "{}", logs[0]);
    assert_eq!((h_ref["stale"].clone(), h_ref["last_rx_age_s"].clone(), h_ref["connected"].clone()), (json!(true), Value::Null, json!(false)), "한 번도 못 받음 = stale");
    assert_eq!(h_ok["stale"], false);
    assert!(h_ok["symbols"]["ETHUSDT"].as_f64().unwrap() < 3.0, "심볼별 마지막 가격 뒤 초: {h_ok}");
}
