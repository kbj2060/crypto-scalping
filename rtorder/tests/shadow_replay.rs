//! 그림자 실행 시험(2026-11-21 일곱째 뒤 C):
//! ① 재생 도구: 원본 «전체 흐름» 정답지를 «실제 기록처럼» 바꾼 합성 기록(서명 없음 · 시각·표식 숫자를 실시계처럼 흔듦)으로 재생 → 차이 0 ·
//!    양성 대조(파이썬 요청 수량 바꿈 · 요청 하나 빠짐 · 끝 칸 다름 · 청산 계획 입력 다름)는 전부 차이로 잡힌다.
//! ② 전송 계층 그림자 막이: 그림자 모드의 `http::Binance` 는 주문·취소·레버리지·algo 취소를 보내기 **전에** panic · GET·listenKey 는 보낸다 ·
//!    실제 주소로 만든 값은 늘 그림자로 시작한다(네트워크 없이 확인). 대조군 = 같은 호출을 그림자 아닌 루프백에서 → 보낸다.

#[path = "common/flowlog.rs"]
mod flowlog;
include!("common/server.rs");

use rtorder::bracket;
use rtorder::exec::{Common, Exchange, LimitOrder, OrderRef};
use rtorder::http::Binance;
use rtorder::shadow;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

fn flow_log() -> Vec<Value> {
    let doc: Value = serde_json::from_str(include_str!("fixtures/flow_fixture.json")).unwrap();
    flowlog::flow_to_log(&doc)
}

/// 실제 기록처럼: 서명을 빼고(비밀값은 기록에 없다) timestamp·표식 숫자를 몇 ms 씩 밀고 입력 시각을 지운다.
fn real_like(rows: &[Value]) -> Vec<Value> {
    let mut k = 0i64;
    rows.iter().map(|r| {
        let mut r = r.clone();
        if r["rec"] == "wire" {
            k += 7;
            let t = r["target"].as_str().unwrap().split("&signature=").next().unwrap().to_string();
            let t: Vec<String> = t.split('&').map(|kv| match kv.split_once('=') {
                Some(("timestamp", v)) => format!("timestamp={}", v.parse::<i64>().unwrap() + k),
                Some((key @ ("newClientOrderId" | "origClientOrderId"), v)) if v.starts_with("db") => format!("{key}={}{}", &v[..5], v[5..].parse::<i64>().unwrap_or(0) + k),
                _ => kv.to_string(),
            }).collect();
            r["target"] = json!(t.join("&"));
        }
        if r["rec"] == "input" {
            r.as_object_mut().unwrap().remove("t");
        }
        r
    }).collect()
}

async fn run(rows: &[Value]) -> Vec<shadow::Report> {
    let chs = shadow::chains(&flowlog::to_jsonl(rows)).unwrap();
    let mut out = vec![];
    let mut i = 0;
    // 시나리오(s<번호>)마다 새 폴더
    let mut groups: Vec<Vec<shadow::Chain>> = vec![];
    for c in chs {
        let sc = c.id.split('.').next().unwrap().to_string();
        if groups.last().is_none_or(|g| g[0].id.split('.').next().unwrap() != sc) {
            groups.push(vec![]);
        }
        groups.last_mut().unwrap().push(c);
    }
    for g in &groups {
        i += 1;
        let dir = std::env::temp_dir().join(format!("rtorder_shadow_{}_{i}_{}", std::process::id(), g[0].id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        out.extend(shadow::replay(g, "ETHUSDC", &dir, false, None).await);
        let _ = std::fs::remove_dir_all(&dir);
    }
    out
}

fn diffs(r: &[shadow::Report]) -> Vec<String> {
    r.iter().filter(|x| !x.diffs.is_empty()).map(|x| format!("{}: {}", x.chain, x.diffs.join(" | "))).collect()
}

#[tokio::test]
async fn replay_of_real_like_log_matches_and_catches_changes() {
    let rows = real_like(&flow_log());
    let base = run(&rows).await;
    assert!(diffs(&base).is_empty(), "합성 실기록 재생 차이: {:?}", diffs(&base));
    let n: usize = base.iter().map(|r| r.requests).sum();
    println!("합성 실기록: 작업 {} · 요청 {n} · 차이 0", base.len());
    // 양성 대조 넷 -- 어느 것이든 «차이 있음»이어야 한다
    let first_post = rows.iter().position(|r| r["rec"] == "wire" && r["method"] == "POST" && r["target"].as_str().unwrap().contains("quantity=0.4")).unwrap();
    let mut a = rows.clone();
    a[first_post]["target"] = json!(a[first_post]["target"].as_str().unwrap().replace("quantity=0.4", "quantity=0.5"));
    let mut b = rows.clone();
    let last_wire = rows.iter().rposition(|r| r["rec"] == "wire").unwrap();
    b.remove(last_wire);
    let mut c = rows.clone();
    let end0 = rows.iter().position(|r| r["rec"] == "end").unwrap();
    c[end0]["slot"]["phase"] = json!("cancelled");
    let mut d = rows.clone();
    let xp = rows.iter().position(|r| r["rec"] == "input" && r["what"] == "exit_plan").unwrap();
    d[xp]["value"]["plan"]["quantity"] = json!(0.3);
    for (name, m) in [("요청 수량 바뀜", a), ("원본 요청 하나 빠짐", b), ("끝 칸 다름", c), ("청산 계획 입력 다름", d)] {
        let r = run(&m).await;
        assert!(!diffs(&r).is_empty(), "양성 대조 «{name}» 를 못 잡았다");
        println!("양성 대조 «{name}» 잡음: {}", diffs(&r)[0].chars().take(140).collect::<String>());
    }
}

fn panics(f: impl FnOnce() + Send + 'static) -> bool {
    std::thread::spawn(f).join().is_err()
}

fn bx(base: &str, shadow: bool) -> Binance {
    let mut b = Binance::new(base, "test_key", "test_secret", std::time::Duration::from_secs(2), std::env::temp_dir().join("rtorder_shadow_ban_unused"));
    if shadow {
        b.shadow_on();
    }
    b
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
}

#[test]
fn shadow_mode_panics_before_any_write_and_allows_reads() {
    // 실제 주소로 만든 값 = 그림자로 시작(요청은 안 보낸다 -- 값만 본다)
    assert!(Binance::new("https://fapi.binance.com", "", "", std::time::Duration::from_secs(1), std::env::temp_dir().join("x")).is_shadow());
    assert!(!bx("http://127.0.0.1:9", false).is_shadow(), "루프백 시험 서버는 기본 그림자 아님(대조군)");
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let l2 = log.clone();
    let srv = serve(Arc::new(move |g: &Got| {
        l2.lock().unwrap().push(format!("{} {}", g.method, g.target.split('?').next().unwrap()));
        let body = if g.target.starts_with("/fapi/v1/openAlgoOrders") {
            r#"[{"algoId": 11, "clientAlgoId": "dbslL1", "symbol": "ETHUSDC"}]"#
        } else if g.target.starts_with("/fapi/v2/positionRisk") {
            r#"[{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": "0", "leverage": "5"}]"#
        } else if g.target.starts_with("/fapi/v1/listenKey") {
            r#"{"listenKey": "lk"}"#
        } else {
            r#"{"orderId": 1, "status": "NEW", "executedQty": "0"}"#
        };
        Act::Send { status: 200, ctype: "application/json".into(), body: body.into(), headers: vec![] }
    }));
    local_only(&srv.base);
    let c = || Common { symbol: "ETHUSDC", side: "BUY", position_side: "LONG" };
    let order = LimitOrder { symbol: "ETHUSDC".into(), side: "BUY".into(), position_side: "LONG".into(), price: 2500.0, quantity: 0.1, client_order_id: "dbpgL1".into() };
    type Job = Box<dyn FnOnce(Binance) + Send>;
    let writes: Vec<(&str, Job)> = vec![
        ("지정가 GTX", { let o = order.clone(); Box::new(move |mut b: Binance| { let _ = rt().block_on(b.place_limit_gtx(&o)); }) }),
        ("시장가", Box::new(move |mut b: Binance| { let _ = rt().block_on(b.place_market(&c(), 0.1)); })),
        ("취소", Box::new(move |mut b: Binance| { let _ = rt().block_on(b.cancel_order(&c(), &OrderRef::Id(Some(1)))); })),
        ("레버리지", Box::new(|mut b: Binance| { let _ = rt().block_on(bracket::ensure_leverage(&mut b, "ETHUSDC", 10)); })),
        ("dbsl algo 취소(손절 해제)", Box::new(|mut b: Binance| { let _ = rt().block_on(bracket::cancel_bracket(&mut b, "ETHUSDC", "LONG", true, false)); })),
    ];
    for (name, job) in writes {
        // 같은 작업을 두 번: 그림자 = panic · 대조군(그림자 아님) = 보낸다
        let n0 = log.lock().unwrap().len();
        let (b1, b2) = (bx(&srv.base, true), bx(&srv.base, false));
        let job = Arc::new(Mutex::new(Some(job)));
        let j1 = job.clone();
        assert!(panics(move || { let f = j1.lock().unwrap().take().unwrap(); f(b1) }), "그림자인데 «{name}» 이 panic 하지 않았다");
        let sent_in_shadow: Vec<String> = log.lock().unwrap()[n0..].iter().filter(|x| !x.starts_with("GET")).cloned().collect();
        assert!(sent_in_shadow.is_empty(), "그림자에서 쓰기 요청이 나갔다: {sent_in_shadow:?}");
        let _ = b2;
        println!("그림자 막음 «{name}»");
    }
    // 대조군: 그림자가 아니면 같은 쓰기가 실제로 나간다(막이가 «늘 막는» 장치가 아님을 보인다)
    let n0 = log.lock().unwrap().len();
    let mut b = bx(&srv.base, false);
    let _ = rt().block_on(b.place_limit_gtx(&order));
    assert_eq!(log.lock().unwrap()[n0..], ["POST /fapi/v1/order".to_string()]);
    // 그림자에서도 읽기·listenKey 는 보낸다(계좌·사용자 스트림은 읽는다 -- 서버에서 켤지는 사용자 결정)
    let mut b = bx(&srv.base, true);
    let n0 = log.lock().unwrap().len();
    let _ = rt().block_on(b.get_order(&c(), &OrderRef::Id(Some(1))));
    let _ = rt().block_on(b.listen_key("POST"));
    let _ = rt().block_on(b.listen_key("PUT"));
    assert_eq!(log.lock().unwrap()[n0..], ["GET /fapi/v1/order".to_string(), "POST /fapi/v1/listenKey".into(), "PUT /fapi/v1/listenKey".into()]);
}
