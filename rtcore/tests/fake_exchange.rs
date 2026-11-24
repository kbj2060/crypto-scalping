//! 네트워크 없는 시험: 프로세스 안 가짜 거래소 웹소켓 + 직접 만든 표본 메시지(실서버 대조 전 모양 -- feed.rs 머리 주석).
use futures_util::{SinkExt, StreamExt};
use rtcore::proto::{Event, Stamped, Subscribe};
use rtcore::state::State;
use rtcore::{Cfg, BAD_JSON, SKIPPED};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

enum Act {
    Send(String),
    Sleep(u64),
    Close,
}

/// (경로, 그 경로의 몇 번째 연결) → 할 일. 할 일이 끝나면 연결을 열어 둔 채 침묵한다.
type Script = Arc<dyn Fn(&str, usize) -> Vec<Act> + Send + Sync>;

/// 가짜 거래소를 띄우고 (주소, 경로별 접속 횟수) 를 돌려준다.
async fn fake(script: Script) -> (String, Arc<Mutex<HashMap<String, usize>>>) {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap().to_string();
    let conns: Arc<Mutex<HashMap<String, usize>>> = Arc::default();
    let c2 = conns.clone();
    tokio::spawn(async move {
        loop {
            let (tcp, _) = l.accept().await.unwrap();
            let (script, conns) = (script.clone(), c2.clone());
            tokio::spawn(async move {
                let path = Arc::new(Mutex::new(String::new()));
                let p2 = path.clone();
                let cb = move |req: &Request, resp: Response| {
                    *p2.lock().unwrap() = req.uri().path().to_string();
                    Ok(resp)
                };
                let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(tcp, cb).await else { return };
                let path = path.lock().unwrap().clone();
                let n = {
                    let mut g = conns.lock().unwrap();
                    let e = g.entry(path.clone()).or_insert(0);
                    *e += 1;
                    *e - 1
                };
                for a in script(&path, n) {
                    match a {
                        Act::Send(t) => {
                            if ws.send(Message::text(t)).await.is_err() {
                                return;
                            }
                        }
                        Act::Sleep(ms) => sleep(Duration::from_millis(ms)).await,
                        Act::Close => {
                            let _ = ws.close(None).await;
                            return;
                        }
                    }
                }
                while let Some(Ok(_)) = ws.next().await {}
            });
        }
    });
    (addr, conns)
}

fn trade(id: u64) -> String {
    json!({"e":"trade","E":1_700_000_000_000u64 + id,"T":1_700_000_000_000u64 + id,"s":"ETHUSDT","t":id,
           "p":format!("{}.5", 3000 + id % 7),"q":"0.250","X":"MARKET","m":id % 2 == 0})
    .to_string()
}
fn book(u: u64) -> String {
    json!({"e":"bookTicker","u":u,"s":"ETHUSDT","b":"3000.10","B":"12.5","a":"3000.20","A":"8.1",
           "T":1_700_000_000_500u64,"E":1_700_000_000_501u64})
    .to_string()
}
fn mark() -> String {
    json!({"e":"markPriceUpdate","E":1_700_000_001_000u64,"s":"ETHUSDT","p":"3000.15","i":"2999.90",
           "P":"3001.00","r":"0.00010000","T":1_700_028_800_000u64})
    .to_string()
}

fn is(p: &str, tail: &str) -> bool {
    p.ends_with(tail)
}

async fn core(addr: &str, silence_ms: u64) -> String {
    let cfg = Cfg {
        symbols: vec!["ethusdt".into()],
        quote_symbols: vec![],
        ws_base: format!("ws://{addr}/ws/"),
        market_ws_base: format!("ws://{addr}/market/ws/"),
        force_url: String::new(),
        spot_ws_base: String::new(),
        okx_url: String::new(),
        bybit_url: String::new(),
        silence: Duration::from_millis(silence_ms),
        hub: rtcore::hub::HubCfg::from_env(),
        snap_dir: String::new(),
        tape_db: String::new(),
        okx_tape_db: String::new(),
        okx_ctx_db: String::new(),
        bn_ctx_db: String::new(),
        bybit_liq_db: String::new(),
    };
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hub = l.local_addr().unwrap().to_string();
    rtcore::start(&cfg, l);
    hub
}

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

async fn subscribe(hub: &str) -> Ws {
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{hub}/rt")).await.unwrap();
    let req = Subscribe { subscribe: vec!["ETHUSDT".into()] };
    ws.send(Message::text(serde_json::to_string(&req).unwrap())).await.unwrap();
    ws
}

async fn next(ws: &mut Ws) -> Stamped {
    loop {
        match timeout(Duration::from_secs(5), ws.next()).await.expect("5초 안에 메시지") {
            Some(Ok(Message::Text(t))) => return serde_json::from_str(t.as_str()).unwrap(),
            Some(Ok(_)) => continue,
            other => panic!("끊김: {other:?}"),
        }
    }
}

/// 스냅숏을 건너뛰고 체결 id 만 n 개 모은다.
async fn trade_ids(ws: &mut Ws, n: usize) -> Vec<u64> {
    let mut ids = vec![];
    while ids.len() < n {
        if let Event::Trade { id, .. } = next(ws).await.ev {
            ids.push(id);
        }
    }
    ids
}

// 1. 체결·호가·마크 표본이 순서대로, 시각 셋이 실린 채 도착한다.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t1_events_in_order_with_three_stamps() {
    let (addr, _) = fake(Arc::new(|p: &str, _| {
        let mut v = vec![Act::Sleep(500)]; // 구독자가 먼저 붙도록
        if is(p, "@trade") {
            v.extend((1..=3).map(|i| Act::Send(trade(i))));
        } else if is(p, "@bookTicker") {
            v.push(Act::Send(book(42)));
        } else if is(p, "@markPrice@1s") && p.starts_with("/market/ws/") {
            v.push(Act::Send(mark()));
        }
        v
    }))
    .await;
    let hub = core(&addr, 30_000).await;
    let mut ws = subscribe(&hub).await;
    let snap = next(&mut ws).await;
    assert!(matches!(snap.ev, Event::Snapshot { .. }), "첫 메시지는 스냅숏");
    let mut got = vec![];
    while got.len() < 5 {
        let m = next(&mut ws).await;
        if !matches!(m.ev, Event::Supply1s(_) | Event::Footprint(_) | Event::Prices { .. } | Event::Situation(_) | Event::Micro(_)) {
            got.push(m); // 1초 수급·풋프린트는 supply1s_parity.rs·footprint_parity.rs 가 본다 · 모든 종목 가격(Prices, s="*")은 quote_parity.rs
                         // 1초 박자로 세는 상황·미시 값(ex_ms 0)은 situation_parity.rs·micro_parity.rs -- 안 거르면 박자가 걸릴 때만 실패한다(2026-11-21 15회 중 5회)
        }
    }
    let mut last_seq = snap.seq;
    let mut ids = vec![];
    for m in &got {
        assert_eq!(m.s, "ethusdt");
        assert!(m.seq > last_seq, "seq 증가");
        last_seq = m.seq;
        assert!(m.ex_ms > 1_600_000_000_000 && m.rx_us > 0 && m.tx_us >= m.rx_us, "시각 셋: {m:?}");
        match &m.ev {
            Event::Trade { id, px, qty, sell } => {
                assert_eq!(m.ex_ms, 1_700_000_000_000 + *id as i64);
                assert!(*px > 0.0 && *qty == 0.25 && *sell == (*id % 2 == 0));
                ids.push(*id);
            }
            Event::Book { u, bid, ask, .. } => assert!(*u == 42 && *bid == 3000.10 && *ask == 3000.20 && m.ex_ms == 1_700_000_000_500),
            Event::Mark { mark, funding, next_funding_ms, .. } => {
                assert!(*mark == 3000.15 && *funding == 0.0001 && *next_funding_ms == 1_700_028_800_000)
            }
            Event::Snapshot { .. } => panic!("스냅숏은 한 번"),
            _ => unreachable!("1초 수급·다른 원천은 이 시험에서 끔"),
        }
    }
    assert_eq!(ids, vec![1, 2, 3]);
    let lat: Vec<i64> = got.iter().map(|m| m.tx_us - m.rx_us).collect();
    println!("t1 수신->송신 µs: {lat:?}");
}

// 2. 가짜 서버가 끊으면 재연결해 이어 받는다.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t2_reconnects_after_close() {
    let (addr, conns) = fake(Arc::new(|p: &str, n| {
        if !is(p, "@trade") {
            return vec![];
        }
        if n == 0 {
            vec![Act::Sleep(300), Act::Send(trade(1)), Act::Close]
        } else {
            vec![Act::Send(trade(2))]
        }
    }))
    .await;
    let hub = core(&addr, 30_000).await;
    let mut ws = subscribe(&hub).await;
    assert_eq!(trade_ids(&mut ws, 2).await, vec![1, 2]);
    assert_eq!(conns.lock().unwrap()["/ws/ethusdt@trade"], 2);
}

// 3. 침묵하면 정해진 시간 뒤 재연결한다.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t3_reconnects_after_silence() {
    let (addr, conns) = fake(Arc::new(|p: &str, n| match (is(p, "@trade"), n) {
        (true, 0) => vec![Act::Sleep(300), Act::Send(trade(1))], // 그 뒤 연결은 열린 채 침묵
        (true, _) => vec![Act::Send(trade(2))],
        _ => vec![Act::Send(book(1)), Act::Sleep(60_000)],
    }))
    .await;
    let hub = core(&addr, 400).await;
    let mut ws = subscribe(&hub).await;
    let t0 = Instant::now();
    assert_eq!(trade_ids(&mut ws, 1).await, vec![1]);
    assert_eq!(trade_ids(&mut ws, 1).await, vec![2]);
    let dt = t0.elapsed();
    assert!(dt >= Duration::from_millis(350), "침묵 시간 전에 재연결했다: {dt:?}");
    assert!(conns.lock().unwrap()["/ws/ethusdt@trade"] >= 2);
}

// 4. 읽지 않는 구독자가 있어도 빠른 구독자는 전부·순서대로 받고, 느린 쪽만 끊긴다.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t4_slow_subscriber_does_not_block_fast() {
    const N: u64 = 100_000;
    let (addr, _) = fake(Arc::new(|p: &str, _| {
        if !is(p, "@trade") {
            return vec![];
        }
        let mut v = vec![Act::Sleep(500)];
        for i in 1..=N {
            v.push(Act::Send(trade(i)));
            if i % 250 == 0 {
                v.push(Act::Sleep(5)); // 약 5만 건/초 -- ETH 실제 급증(수천/초)보다 한 자릿수 위
            }
        }
        v
    }))
    .await;
    let hub = core(&addr, 30_000).await;
    let mut slow = subscribe(&hub).await; // 이후 한 번도 읽지 않는다
    let mut fast = subscribe(&hub).await;
    let t0 = Instant::now();
    let mut lat = vec![];
    let mut want = 1;
    while want <= N {
        let m = next(&mut fast).await;
        if let Event::Trade { id, .. } = m.ev {
            assert_eq!(id, want, "빠진·뒤바뀐 체결");
            lat.push((m.tx_us - m.rx_us) as u64);
            want += 1;
        }
    }
    let took = t0.elapsed();
    // 느린 쪽: 버퍼에 쌓인 것을 다 읽어도 N 건에 못 미치고 끊겨 있어야 한다.
    let mut slow_trades = 0u64;
    while let Ok(Some(Ok(msg))) = timeout(Duration::from_secs(2), slow.next()).await {
        if let Message::Text(t) = msg {
            if t.as_str().contains("\"type\":\"trade\"") {
                slow_trades += 1;
            }
        }
    }
    lat.sort_unstable();
    let n = lat.len();
    println!(
        "t4 빠른 구독자 {N}건 {took:?} · 수신->송신 µs 중앙 {} 상위1% {} 최대 {} · 느린 구독자 받은 체결 {slow_trades}",
        lat[n / 2], lat[n * 99 / 100], lat[n - 1]
    );
    assert!(slow_trades < N, "느린 구독자가 끊기지 않았다");
}

// 5. 깨진 JSON·모르는 사건·가격 0 체결은 건너뛰고 연결도 유지한다.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn t5_bad_messages_are_skipped() {
    let (b0, s0) = (BAD_JSON.load(Relaxed), SKIPPED.load(Relaxed));
    let (addr, conns) = fake(Arc::new(|p: &str, _| {
        if !is(p, "@trade") {
            return vec![];
        }
        vec![
            Act::Sleep(300),
            Act::Send("{not json".into()),
            Act::Send(r#"{"e":"aggTrade","s":"ETHUSDT"}"#.into()),
            Act::Send(r#"{"e":"trade","E":1,"T":1,"s":"ETHUSDT","t":9,"p":"0","q":"0","X":"NA","st":1,"m":false}"#.into()),
            Act::Send(r#"{"e":"trade","s":"ETHUSDT"}"#.into()),
            Act::Send(trade(7)),
        ]
    }))
    .await;
    let hub = core(&addr, 30_000).await;
    let mut ws = subscribe(&hub).await;
    assert_eq!(trade_ids(&mut ws, 1).await, vec![7]);
    assert!(BAD_JSON.load(Relaxed) >= b0 + 1);
    assert!(SKIPPED.load(Relaxed) >= s0 + 3);
    assert_eq!(conns.lock().unwrap()["/ws/ethusdt@trade"], 1, "나쁜 메시지로 끊으면 안 된다");
}

// 6. 체결 링은 상한을 넘지 않고 가장 오래된 것부터 버린다.
#[test]
fn t6_trade_ring_is_bounded() {
    let st = State::new(&["ethusdt".to_string()], 16, 4);
    for i in 1..=100 {
        st.apply(rtcore::feed::parse(&trade(i), 1).unwrap());
    }
    st.apply(rtcore::feed::parse(&book(5), 1).unwrap());
    st.flush();
    let snap = st.snapshot("ethusdt", 0).unwrap();
    assert_eq!(snap.seq, 105); // 체결 100 + 호가 1 + 묶음 1번(1초 수급 1 · 풋프린트 1 · 틱 봉 1 · 1분봉 1 -- 2026-11-24) -- 사건마다 밀지 않는다(state.rs FLUSH_MS)
    let Event::Snapshot { trades, book, .. } = snap.ev else { panic!() };
    assert_eq!(trades.len(), 16);
    assert!(matches!(trades.first().map(|t| &t.ev), Some(Event::Trade { id: 85, .. })));
    assert!(book.is_some());
    assert!(st.snapshot("btcusdt", 0).is_none());
}

// 7. 속도 실측(기본 제외): 체결 1건 수신 → 그 초의 1초 수급 칸이 구독자에게 송신·도착. 가짜 거래소 20건마다 1ms 쉼(개발기에서 약 수천 건/초).
//    `cargo test --release -j 2 --test fake_exchange -- --ignored --nocapture t7`
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn t7_supply1s_latency_bench() {
    const N: u64 = 50_000;
    let (addr, _) = fake(Arc::new(|p: &str, _| {
        if !is(p, "@trade") {
            return vec![];
        }
        let t0 = rtcore::now_us() / 1000 + 500;
        let mut v = vec![Act::Sleep(500)];
        for i in 1..=N {
            let q = [0.01, 0.2, 1.5, 4.0, 40.0][(i % 5) as usize]; // 리테일·중형·고래가 다 나오게
            let px = 3000.0 + (i / 7 % 5) as f64 * 0.01; // 같은 가격 연속 → 되묶기
            v.push(Act::Send(
                json!({"e":"trade","E":t0 + (i / 20) as i64,"T":t0 + (i / 20) as i64,"s":"ETHUSDT","t":i,
                       "p":format!("{px:.2}"),"q":format!("{q:.3}"),"X":"MARKET","m":i % 11 < 5})
                .to_string(),
            ));
            if i % 20 == 0 {
                v.push(Act::Sleep(1));
            }
        }
        v
    }))
    .await;
    let hub = core(&addr, 30_000).await;
    let mut ws = subscribe(&hub).await;
    let (mut core_us, mut e2e_us, mut trades) = (vec![], vec![], 0u64);
    let t0 = Instant::now();
    while trades < N {
        let m = next(&mut ws).await;
        let arrived = rtcore::now_us();
        match m.ev {
            Event::Supply1s(_) => {
                core_us.push(m.tx_us - m.rx_us);
                e2e_us.push(arrived - m.rx_us);
            }
            Event::Trade { .. } => trades += 1,
            _ => {}
        }
    }
    let took = t0.elapsed().as_secs_f64();
    for (name, v) in [("수신->송신(코어 안)", &mut core_us), ("수신->구독자 도착", &mut e2e_us)] {
        v.sort_unstable();
        let n = v.len();
        println!("t7 1초 수급 {name}: n={n} 중앙 {}µs 상위1% {}µs 최대 {}µs", v[n / 2], v[n * 99 / 100], v[n - 1]);
    }
    println!("t7 체결 {trades}건 / {took:.1}초 = {:.0}건/초", trades as f64 / took);
}
