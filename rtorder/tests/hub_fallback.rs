//! 2026-11-22 사용자 결정 «허브 가격이 끊기면 거래소에서 직접 읽어 감시를 잇고 텔레그램으로 알린다»: 가짜 허브(127.0.0.1 ws · 12초 넘게 침묵 뒤 돌아옴) + 가짜 거래소(루프백)로
//! ① 실전송 시험 모드: 끊긴 동안 거래소 직접 조회(ticker/price · klines 5m)의 값으로 손절 닿음 → 발동 청산(GTX 매도) · 돌아오면 대체 요청 0 · 알림 «끊김»·«돌아옴»
//! ② 그림자: 같은 갈래 → 발동 첫걸음(익절 지우기) «그림자(안 보냄)» · 실제 대체 요청은 코인마다 60초에 한 번 · «했을 것» 줄 · 10분(시험은 3초) 다시 알림 · 텔레그램은 안 보냄(«TELEGRAM …» 목록)
//! ③ 무장이 없으면 허브가 끊겨도 대체 요청 0.
include!("common/server.rs");

use futures_util::{SinkExt, StreamExt};
use rtorder::app::{self, Stop};
use rtorder::http::Binance;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Message;

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

#[derive(Default)]
struct Ex {
    log: Vec<(f64, String)>,
    price: String,
    closed: bool,
    next: i64,
}

fn exchange(price: &str) -> (Srv, Arc<Mutex<Ex>>) {
    let fx: Value = serde_json::from_str(include_str!("fixtures/desk_fixture.json")).unwrap();
    let routes = fx["account"].as_array().unwrap().iter().find(|c| c["name"] == "normal_usdc").unwrap()["in"]["routes"].clone();
    let ex = Arc::new(Mutex::new(Ex { price: price.into(), next: 500, ..Default::default() }));
    let e2 = ex.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        let mut e = e2.lock().unwrap();
        let (path, q) = g.target.split_once('?').unwrap_or((&g.target, ""));
        let qm: std::collections::HashMap<&str, &str> = q.split('&').filter_map(|kv| kv.split_once('=')).collect();
        e.log.push((now(), format!("{} {path} {}", g.method, qm.get("interval").copied().unwrap_or(""))));
        let t = now();
        let body: Value = match (g.method.as_str(), path) {
            ("GET", "/fapi/v1/time") => json!({"serverTime": (t * 1000.0) as i64}),
            ("GET", "/fapi/v2/account") => routes["/fapi/v2/account"].clone(),
            ("GET", "/fapi/v2/positionRisk") if e.closed => json!([]),
            ("GET", "/fapi/v2/positionRisk") => json!([{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": "0.500", "entryPrice": "2450.0", "markPrice": "2450.25",
                                                       "liquidationPrice": "1900.1", "leverage": "10", "notional": "1225.125", "unRealizedProfit": "0", "updateTime": 1799990000123i64}]),
            ("GET", "/fapi/v1/userTrades") | ("GET", "/fapi/v1/openOrders") | ("GET", "/fapi/v1/openAlgoOrders") => json!([]),
            ("GET", "/fapi/v1/ticker/price") => json!({"symbol": qm["symbol"], "price": e.price, "time": (t * 1000.0) as i64}),
            ("GET", "/fapi/v1/klines") if qm.get("interval") == Some(&"5m") => {
                let open = ((t / 300.0).floor() * 300.0 - 300.0) as i64 * 1000; // 마감 봉 하나 + 형성 중 봉 하나 · 종가는 손절선 위
                json!([[open, "2450", "2451", "2449", "2450.5", "1", open + 299_999], [open + 300_000, "2450", "2451", "2449", "2450.7", "1", open + 599_999]])
            }
            ("GET", "/fapi/v1/klines") => json!([]),
            ("GET", "/fapi/v1/ticker/bookTicker") => json!({"symbol": "ETHUSDC", "bidPrice": "2390.10", "askPrice": "2390.20"}),
            ("GET", "/fapi/v1/exchangeInfo") => json!({"symbols": [{"symbol": "ETHUSDC", "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                                                     {"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "MIN_NOTIONAL", "notional": "20"}]}]}),
            ("GET", "/fapi/v1/premiumIndex") => json!({"symbol": "ETHUSDC", "lastFundingRate": "0.0001", "markPrice": "2390.15"}),
            ("GET", "/fapi/v1/commissionRate") => json!({"symbol": "ETHUSDC", "makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"}),
            ("POST", "/fapi/v1/listenKey") => json!({"listenKey": "lk1"}),
            ("POST", "/fapi/v1/order") => {
                e.next += 1;
                e.closed = true; // 청산 매도가 바로 체결된 것으로(다음 계좌 = 포지션 없음)
                json!({"orderId": e.next, "clientOrderId": qm.get("newClientOrderId").copied().unwrap_or(""), "symbol": qm["symbol"], "side": qm["side"],
                       "positionSide": qm["positionSide"], "type": qm["type"], "price": qm.get("price").copied().unwrap_or("0"), "origQty": qm["quantity"],
                       "executedQty": "0", "status": "NEW", "reduceOnly": false})
            }
            ("GET", "/fapi/v1/order") => json!({"orderId": e.next, "symbol": "ETHUSDC", "side": "SELL", "positionSide": "LONG", "type": "LIMIT", "price": "2390.2",
                                                 "origQty": "0.5", "executedQty": "0.5", "avgPrice": "2390.2", "status": "FILLED"}),
            ("DELETE", "/fapi/v1/order") => json!({"status": "CANCELED"}),
            _ => return Act::Send { status: 599, ctype: "application/json".into(), body: format!(r#"{{"msg":"규칙 없음 {} {path}"}}"#, g.method), headers: vec![] },
        };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.to_string(), headers: vec![] }
    }));
    local_only(&srv.base);
    (srv, ex)
}

/// 가짜 허브: 붙으면 ETHUSDT 스냅숏(1초가 2450 · 마감 5분봉) → `on` 인 동안 1초마다 supply1s · 꺼지면 침묵(연결은 유지).
async fn hub(on: Arc<AtomicBool>) -> String {
    let lis = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/rt", lis.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let Ok((s, _)) = lis.accept().await else { return };
            let on = on.clone();
            tokio::spawn(async move {
                let Ok(mut ws) = tokio_tungstenite::accept_async(s).await else { return };
                let _ = ws.next().await;
                let t = now() as i64;
                let snap = json!({"s": "ethusdt", "rx_us": (now() * 1e6) as i64, "type": "snapshot", "book": null,
                                  "supply": {"seconds": [[t, 0, 0, 0, 0, 0, 0, 2450.0]]}, "quote": {"vol_bpm": null, "kline_rx_us": 0, "bar5m": [t / 300 * 300, 2450.5]}});
                if ws.send(Message::Text(snap.to_string().into())).await.is_err() {
                    return;
                }
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if on.load(Ordering::SeqCst) {
                        let m = json!({"s": "ethusdt", "rx_us": (now() * 1e6) as i64, "type": "supply1s", "seconds": [[now() as i64, 0, 0, 0, 0, 0, 0, 2450.0]]});
                        if ws.send(Message::Text(m.to_string().into())).await.is_err() {
                            return;
                        }
                    }
                }
            });
        }
    });
    url
}

fn arm(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let b = json!({"symbol": "ETHUSDC", "side": "LONG", "sl_price": 2400.0, "sl_level": 2400.0, "tp_price": 2600.0, "tp_level": 2600.0, "backstop_price": null, "rule": false,
                   "sl_name": "선 손절 2400.00", "lines": true, "sl_up": null, "tp_resting": false, "armed_at": now() - 600.0, "placed": {"placed": true}});
    std::fs::write(dir.join("manual_bracket_state.json"), json!({"ETHUSDC:LONG": b}).to_string()).unwrap();
}

/// 한 판: 허브 켬(2초) → 끔(`off`초) → 켬(4초) 뒤 멈춤. 돌려주기 = (요약, 거래소 기록, 허브가 돌아온 시각, 끈 시각).
async fn scenario(name: &str, armed: bool, shadow: bool, off: f64) -> (app::Summary, Vec<(f64, String)>, f64, f64) {
    let (srv, ex) = exchange("2450.00");
    let on = Arc::new(AtomicBool::new(true));
    let url = hub(on.clone()).await;
    let dir = std::env::temp_dir().join(format!("rtorder_fallback_{}_{name}", std::process::id()));
    if armed {
        arm(&dir);
    } else {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
    }
    let (d, base) = (dir.to_string_lossy().to_string(), srv.base.clone());
    let get = move |k: &str| match k {
        "RTORDER_DIR" => Some(d.clone()),
        "RTORDER_BASE" => Some(base.clone()),
        "RTORDER_HUB" => Some(url.clone()),
        "RTORDER_PERIOD" => Some("0.5".into()),
        "RTORDER_FALLBACK_ALERT_SEC" => Some("3".into()),
        _ => None,
    };
    let cfg = app::config_from(&get).unwrap();
    let (ban, b2) = (dir.join("ban"), srv.base.clone());
    let mk = move || {
        let mut b = Binance::new(&b2, "test_key", "test_secret", std::time::Duration::from_secs(3), ban.clone());
        if shadow {
            b.shadow_on();
        }
        b
    };
    let stop = Stop::default();
    let (s2, on2, ex2) = (stop.clone(), on.clone(), ex.clone());
    let times = Arc::new(Mutex::new((0.0, 0.0)));
    let t2 = times.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        on2.store(false, Ordering::SeqCst); // 허브 침묵
        ex2.lock().unwrap().price = "2390.00".into(); // 끊긴 동안 거래소 마지막 체결가 = 손절선 아래
        t2.lock().unwrap().0 = now();
        tokio::time::sleep(std::time::Duration::from_secs_f64(off)).await;
        on2.store(true, Ordering::SeqCst);
        t2.lock().unwrap().1 = now();
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        s2.stop();
    });
    let s = app::run(&cfg, &mk, stop, Default::default(), None).await;
    let log = ex.lock().unwrap().log.clone();
    let (t_off, t_back) = *times.lock().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (s, log, t_back, t_off)
}

fn fallback_reqs(log: &[(f64, String)], from: f64) -> usize {
    log.iter().filter(|(t, l)| *t >= from && (l.starts_with("GET /fapi/v1/ticker/price") || l == "GET /fapi/v1/klines 5m")).count()
}

#[tokio::test]
async fn hub_silence_falls_back_to_exchange_reads_fires_stop_and_returns() {
    std::env::set_var("DASHBOARD_MANUAL_EXEC_ENABLED", "1");
    let ((live, llog, lback, loff), (sh, slog, sback, soff), (idle, ilog, _, ioff)) =
        tokio::join!(scenario("live", true, false, 18.0), scenario("shadow", true, true, 18.0), scenario("idle", false, false, 18.0));
    let alerts = |s: &app::Summary, k: &str| s.logs.iter().filter(|l| l.starts_with(&format!("rtorder {k}"))).count();
    // ① 실전송: 끊김(10초 넘음) 뒤 대체 요청 → 손절 닿음 → 발동 청산 GTX 매도 · 돌아오면 대체 0
    println!("실전송 알림 {:?}", live.logs.iter().filter(|l| l.contains("허브 가격")).collect::<Vec<_>>());
    assert!(fallback_reqs(&llog, loff + 8.0) >= 2 && fallback_reqs(&llog, loff) == fallback_reqs(&llog, loff + 8.0), "끊긴 뒤 10초(마지막 수신 기준) 넘어서야 거래소 직접 조회 {llog:?}");
    let sell = llog.iter().find(|(_, l)| l.starts_with("POST /fapi/v1/order")).expect("발동 청산 주문");
    assert!(sell.0 > loff + 8.0 && sell.0 < lback, "청산은 끊긴 동안(허브 값 2450 으로는 안 닿는다)");
    assert_eq!(fallback_reqs(&llog, lback + 1.5), 0, "허브가 돌아오면 대체 요청 0");
    assert!(alerts(&live, "허브 가격 끊김") >= 1 && alerts(&live, "허브 가격 돌아옴") >= 1, "알림 «끊김»·«돌아옴» {:?}", live.logs);
    assert!(live.would.is_empty(), "실전송 시험 모드");
    // ② 그림자: 같은 갈래 · 쓰기 0 · «그림자(안 보냄)» GTX · 실제 대체 요청은 60초에 한 번(이 판에서는 코인마다 ticker 1 + klines 1)
    println!("그림자 목록 {:?}", sh.would);
    assert!(slog.iter().all(|(_, l)| l.starts_with("GET") || l.starts_with("POST /fapi/v1/listenKey")), "그림자 쓰기 0 {slog:?}");
    // 선 무장의 발동 첫걸음 = 걸린 익절 지정가부터 지움(CLEAR) -- 그림자는 그 취소를 안 보내므로(오류) 10초 뒤 다시 · 청산 GTX 까지는 안 간다(실전송 판이 GTX 까지 본다)
    let fire = sh.would.iter().position(|w| w.starts_with("CLEAR ETHUSDC LONG"));
    let enter = sh.would.iter().position(|w| w.starts_with("TELEGRAM 허브 가격 끊김"));
    assert!(fire.is_some() && enter < fire, "그림자: 대체 값으로 손절 닿음 → 발동(익절 지우기 안 보냄) {:?}", sh.would);
    let during = slog.iter().filter(|(t, l)| *t >= soff && *t < sback && (l.starts_with("GET /fapi/v1/ticker/price") || l == "GET /fapi/v1/klines 5m")).count();
    assert!((1..=2).contains(&during), "그림자 대체 요청 60초에 한 번 ({during}) {slog:?}");
    assert!(sh.logs.iter().any(|l| l.contains("그림자: 허브 끊김 → 거래소 직접 조회했을 것")), "«했을 것» 줄 {:?}", sh.logs);
    assert!(alerts(&sh, "허브 가격 끊김") >= 2, "10분(시험 3초)마다 다시 알림 {:?}", sh.logs);
    assert!(sh.would.iter().any(|w| w.starts_with("TELEGRAM 허브 가격 끊김")) && sh.would.iter().any(|w| w.starts_with("TELEGRAM 허브 가격 돌아옴")), "그림자 = 텔레그램 안 보냄(목록만)");
    // ③ 무장 없음 = 끊겨도 대체 0 · 알림 0
    assert_eq!(fallback_reqs(&ilog, ioff), 0, "무장 없으면 대체 요청 0 {ilog:?}");
    assert_eq!(alerts(&idle, "허브 가격"), 0);
}
