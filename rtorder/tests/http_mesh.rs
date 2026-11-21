//! 상태 기계 + 진짜 전송이 맞물리는지(2026-11-20 둘째 조각): 첫 조각의 가짜 거래소(common/fake.rs)를 **HTTP 뒤에** 두고
#![allow(dead_code)]
//! `Binance`(서명·ureq·오류 문자열 규칙) 로 집행 함수를 돌린다. 불변식 셋(총 체결 ≤ 계획 · 살아남은 지정가 ≤ 1 · 항상 종료)을 그대로 검사한다.
//! 가짜 서버는 127.0.0.1 만 · 키는 가짜 · 서버가 받은 요청마다 서명(HMAC)을 다시 계산해 맞는지 본다.
//! 가짜의 «네트워크 오류»(숫자로 시작 안 하는 문자열)는 서버가 답 없이 끊는 것으로 -- 그러면 Rust 도 aiohttp 처럼 GET·DELETE 를 한 번 다시 보낸다.

use rtorder::http::Binance;
use serde_json::json;

include!("common/fake.rs");
include!("common/server.rs");

fn qmap(q: &str) -> HashMap<String, String> {
    q.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn to_act(r: Result<serde_json::Value, String>) -> Act {
    let send = |status: u16, body: serde_json::Value| Act::Send { status, ctype: "application/json".into(), body: body.to_string(), headers: vec![] };
    match r {
        Ok(v) => send(200, v),
        Err(e) if e.starts_with(|c: char| c.is_ascii_digit()) => {
            let (st, rest) = e.split_once(' ').unwrap();
            match rest.strip_prefix('[').and_then(|r| r.split_once("] ")) {
                Some((code, msg)) => send(st.parse().unwrap(), json!({"code": code.parse::<i64>().unwrap(), "msg": msg})),
                None => send(st.parse().unwrap(), json!({"msg": rest})),
            }
        }
        Err(_) => Act::Close, // 네트워크·시간초과 = 답 없이 끊김
    }
}

fn order_body(r: Reply) -> Result<serde_json::Value, String> {
    r.map(|o| json!({"orderId": o.order_id, "status": o.status, "executedQty": rtorder::pyfmt::repr(o.executed_qty)}))
}

fn handle(fx: &Mutex<Fake>, g: &Got) -> Act {
    let (path, q) = g.target.split_once('?').unwrap_or((&g.target, ""));
    let m = qmap(q);
    if let Some(sig) = m.get("signature") {
        let body = &q[..q.rfind("&signature=").unwrap()];
        let tag = ring::hmac::sign(&ring::hmac::Key::new(ring::hmac::HMAC_SHA256, b"test_secret"), body.as_bytes());
        assert_eq!(sig, &hex::encode(tag.as_ref()), "서명이 틀렸다: {}", g.target);
        assert_eq!(g.key.as_deref(), Some("test_key"));
        assert!(m["recvWindow"] == "5000" && m.contains_key("timestamp"));
    }
    let mut f = fx.lock().unwrap();
    let (side, ps) = (m.get("side").cloned().unwrap_or_default(), m.get("positionSide").cloned().unwrap_or_default());
    let c = Common { symbol: "ETHUSDC", side: &side, position_side: &ps };
    let r = || match m.get("orderId") {
        Some(id) => OrderRef::Id(id.parse().ok()),
        None => OrderRef::Client(m["origClientOrderId"].clone()),
    };
    let res = match (g.method.as_str(), path) {
        ("POST", "/fapi/v1/order") if m["type"] == "MARKET" => {
            assert!(!m.contains_key("timeInForce") && !m.contains_key("price") && !m.contains_key("reduceOnly"), "시장가 다리 모양 {}", g.target);
            order_body(block_on(f.place_market(&c, m["quantity"].parse().unwrap())))
        }
        ("POST", "/fapi/v1/order") => {
            assert_eq!((m["type"].as_str(), m["timeInForce"].as_str()), ("LIMIT", "GTX"), "지정가 GTX 만 나간다");
            let o = LimitOrder { symbol: m["symbol"].clone(), side: side.clone(), position_side: ps.clone(), price: m["price"].parse().unwrap(),
                                 quantity: m["quantity"].parse().unwrap(), client_order_id: m["newClientOrderId"].clone() };
            order_body(block_on(f.place_limit_gtx(&o)))
        }
        ("GET", "/fapi/v1/order") => order_body(block_on(f.get_order(&c, &r()))),
        ("DELETE", "/fapi/v1/order") => order_body(block_on(f.cancel_order(&c, &r()))),
        ("GET", "/fapi/v1/ticker/bookTicker") => block_on(f.book(&m["symbol"])).map(|(b, a)| json!({"bidPrice": b.to_string(), "askPrice": a.to_string()})),
        x => panic!("모르는 요청 {x:?}"),
    };
    to_act(res)
}

/// 집행 루프의 시계 = 가짜 거래소 시계, 주문·조회·취소·호가 = HTTP.
struct Mesh {
    bx: Binance,
    fx: Arc<Mutex<Fake>>,
}
impl Exchange for Mesh {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        self.bx.place_limit_gtx(o).await
    }
    async fn place_market(&mut self, c: &Common<'_>, q: f64) -> Reply {
        self.bx.place_market(c, q).await
    }
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.bx.get_order(c, r).await
    }
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.bx.cancel_order(c, r).await
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.bx.book(s).await
    }
    fn now(&self) -> f64 {
        self.fx.lock().unwrap().t
    }
    fn wall_ms(&self) -> i64 {
        self.fx.lock().unwrap().wall_ms()
    }
    async fn nap(&mut self, sec: f64) {
        block_on(self.fx.lock().unwrap().nap(sec))
    }
    async fn sleep(&mut self, sec: f64) {
        block_on(self.fx.lock().unwrap().sleep(sec))
    }
}

async fn go_http(fx: Fake, p: &ExecPlan, entry: bool) -> (ExecState, Fake, usize) {
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    let fx = Arc::new(Mutex::new(Fake { shared: Some(s.clone()), ..fx }));
    let f2 = fx.clone();
    let srv = serve(Arc::new(move |g: &Got| handle(&f2, g)));
    local_only(&srv.base);
    let bx = Binance::new(&srv.base, "test_key", "test_secret", std::time::Duration::from_secs(2), std::env::temp_dir().join("rtorder_mesh_ban_unused"));
    let mut ex = Mesh { bx, fx: fx.clone() };
    if entry { run_entry(&mut ex, p, &s).await } else { run_exit(&mut ex, p, &s).await }
    let st = s.lock().unwrap().clone();
    let n = srv.log.lock().unwrap().len();
    drop(ex);
    let f = std::mem::take(&mut *fx.lock().unwrap());
    (st, f, n)
}

#[tokio::test]
async fn mesh_normal_fill() {
    let (st, fx, n) = go_http(Fake { fill_after: 1, ..Default::default() }, &exit_plan(), false).await;
    check("HTTP 정상", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && st.filled == 2.0 && fx.placed.len() == 1 && n >= 2, "{st:?} 요청 {n}");
}

#[tokio::test]
async fn mesh_drift_repegs_then_market_remainder_m9() {
    let (st, fx, n) = go_http(Fake { drift: true, ..Default::default() }, &exit_plan(), false).await;
    check("HTTP 리페그", &st, &fx, 2.0);
    assert!(st.repegs >= 30 && st.phase == Phase::FilledTaker && st.filled == 2.0 && fx.markets.len() == 1, "{st:?}"); // M9 = 마감 잔량 시장가(HTTP 모양은 서버가 단언)
    assert!(fx.placed.windows(2).all(|w| w[1].price < w[0].price) && n > 4 * st.repegs as usize);
}

#[tokio::test]
async fn mesh_cancel_fail_during_repeg_stops_with_error() {
    // 취소가 500 으로 실패 · 재호가 1회 뒤 -- 확인 못 하면 또 걸지 않고 error(2026-09-13 지정가 41개)
    let (st, fx, _) = go_http(Fake { drift: true, fail: vec!["cancel"], ok_first: 1, ..Default::default() }, &exit_plan(), false).await;
    check("HTTP 취소 실패", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && st.repegs == 1 && fx.placed.len() == 2 && fx.live().len() == 1, "{st:?}");
}

#[tokio::test]
async fn mesh_unknown_post_a_503_accepted() {
    let (st, fx, _) = go_http(Fake { fill_after: 1, place_err: Some(("503 Service Unavailable".into(), true)), ..Default::default() }, &entry_plan(), true).await;
    check("HTTP 불명 a", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.lookups == 1 && fx.placed.len() == 1, "{st:?}");
}

#[tokio::test]
async fn mesh_unknown_post_c_disconnect_seen_on_second_lookup() {
    // 접수 POST 가 끊김(답 없음) -- POST 는 다시 보내지 않는다 -- 표식 조회 둘째에 보임
    let (st, fx, _) = go_http(Fake { fill_after: 1, place_err: Some(("TimeoutError: read".into(), true)), lookup_miss: 1, ..Default::default() }, &exit_plan(), false).await;
    check("HTTP 불명 c", &st, &fx, 2.0);
    assert!(st.phase == Phase::FilledMaker && fx.lookups == 2 && fx.placed.len() == 1, "{st:?} lookups {}", fx.lookups);
}

#[tokio::test]
async fn mesh_unknown_post_e_lookup_and_delete_all_fail() {
    // 조회·표식 DELETE 가 전부 끊김 -- GET·DELETE 는 aiohttp 처럼 한 번씩 다시 보낸다(조회 3번 × 2 · DELETE 1번 × 2) → error, 살아 있는 주문 1(사람에게)
    let (st, fx, _) = go_http(Fake { fill_after: 1, place_err: Some(("TimeoutError: read".into(), true)), fail: vec!["lookup", "delete_cid"], ..Default::default() }, &entry_plan(), true).await;
    check("HTTP 불명 e", &st, &fx, 2.0);
    assert!(st.phase == Phase::Error && st.error.as_deref().unwrap().contains("접수 여부를 확인하지 못했습니다"), "{st:?}");
    assert_eq!((fx.lookups, fx.deletes.len(), fx.live().len()), (6, 2, 1));
}

#[tokio::test]
async fn nap_wakes_early_on_fill_alert() {
    // A10·R7: 알림은 앞당기기만 -- 깨우면 일찍 · 안 깨우면 끝까지
    let mut bx = Binance::new("http://127.0.0.1:9", "test_key", "test_secret", std::time::Duration::from_secs(1), std::env::temp_dir().join("x"));
    let w = bx.wake.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        w.notify_waiters();
    });
    let t = std::time::Instant::now();
    bx.nap(5.0).await;
    assert!(t.elapsed().as_secs_f64() < 1.0, "알림에 일찍 깨야 한다 {:?}", t.elapsed());
    let t = std::time::Instant::now();
    bx.nap(0.2).await;
    assert!(t.elapsed().as_secs_f64() >= 0.19);
}

#[tokio::test]
async fn refused_connection_is_unknown_outcome_not_rejection() {
    // 연결 자체가 거부되면(127.0.0.1 의 닫힌 포트) 숫자로 시작하지 않는 오류 = «나갔는지 모름» 쪽
    let base = { let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap(); format!("http://{}", l.local_addr().unwrap()) };
    local_only(&base);
    let mut bx = Binance::new(&base, "test_key", "test_secret", std::time::Duration::from_secs(1), std::env::temp_dir().join("x"));
    let r = bx.signed("GET", "/fapi/v1/order", &[]).await;
    let e = r["__error__"].as_str().unwrap();
    assert!(e.starts_with("ClientConnectorError: Cannot connect to host 127.0.0.1:"), "{e}");
}

#[test]
#[should_panic(expected = "127.0.0.1 이 아니다")]
fn test_address_must_be_local() {
    local_only(rtorder::http::FAPI);
}
