//! 서명·전송·브래킷 정답지 재생(2026-11-20 둘째 조각). 정답지 = rtorder/tools/gen_http_fixture.py 가 **원본 파이썬 함수 + 진짜 aiohttp** 를
//! 로컬 가짜 서버에 붙여 뽑은 «각본(응답 열) → 나간 요청 열(메서드·경로+쿼리 글자까지·키 머리) → 돌려준 값 → 밴 기록».
//! 여기서는 같은 각본을 프로세스 안 가짜 서버(127.0.0.1)에 걸고 Rust 함수를 돌려 셋을 대조한다. 키는 가짜(test_key/test_secret) · 시계는 가짜.

use rtorder::bracket;
use rtorder::exec::{self, Common, Exchange, ExecState, LimitOrder, OrderRef, Phase, Reply, Shared};
use rtorder::http::Binance;
use rtorder::sign::{self, P};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

include!("common/server.rs");

const SYM: &str = "ETHUSDC";

fn act_of(r: &Value, hang: f64) -> Act {
    match r.get("act").and_then(Value::as_str) {
        Some("close") => return Act::Close,
        Some("hang") => {
            let rest: Value = r.as_object().unwrap().iter().filter(|(k, _)| *k != "act").map(|(k, v)| (k.clone(), v.clone())).collect::<serde_json::Map<_, _>>().into();
            return Act::Hang(hang, Box::new(act_of(&rest, hang)));
        }
        _ => {}
    }
    let body = match r.get("raw") {
        Some(Value::String(s)) => s.clone(),
        _ => serde_json::to_string(r.get("body").unwrap_or(&Value::Null)).unwrap(),
    };
    Act::Send {
        status: r.get("status").and_then(Value::as_u64).unwrap_or(200) as u16,
        ctype: r.get("ctype").and_then(Value::as_str).unwrap_or("application/json").into(),
        body,
        headers: r.get("headers").and_then(Value::as_object).map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().into())).collect()).unwrap_or_default(),
    }
}

/// Binance + 가짜 시계: 네트워크는 진짜(가짜 서버로), 시간(now·nap·sleep)은 가짜 -- 원본 생성기가 time·nap·asyncio.sleep 을 갈아끼운 것과 같다.
pub struct Virt {
    pub bx: Binance,
    pub t: Arc<Mutex<f64>>,
    /// M4 각본: n 번째 주문 번호 조회 때 집행 상태에 action 을 넣는다(원본 생성기의 S.hook 과 같은 자리 -- 화면이 대기 중에 누른 것).
    pub hook: Option<(Shared, u32, exec::Action)>,
    pub gets: u32,
}
impl Virt {
    fn new(base: &str, keys: bool, timeout: f64, t0: f64, ban: std::path::PathBuf) -> Virt {
        local_only(base);
        let (k, s) = if keys { ("test_key", "test_secret") } else { ("", "") };
        let mut bx = Binance::new(base, k, s, std::time::Duration::from_secs_f64(timeout), ban);
        let t = Arc::new(Mutex::new(0.0f64));
        let tc = t.clone();
        bx.clock = Arc::new(move || t0 + *tc.lock().unwrap());
        let ts = t.clone();
        bx.virtual_sleep = Some(Arc::new(move |sec| *ts.lock().unwrap() += sec)); // 불명 응답 조회 사이 1초(resolve_resting) -- 원본 가짜 잠과 같이 시계만
        Virt { bx, t, hook: None, gets: 0 }
    }
    fn adv(&self, s: f64) {
        *self.t.lock().unwrap() += s;
    }
}
impl Exchange for Virt {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        self.bx.place_limit_gtx(o).await
    }
    async fn place_market(&mut self, c: &Common<'_>, q: f64) -> Reply {
        self.bx.place_market(c, q).await
    }
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        if let (OrderRef::Id(_), Some((s, n, a))) = (r, &self.hook) {
            self.gets += 1;
            if self.gets == *n {
                s.lock().unwrap().action = Some(*a);
            }
        }
        self.bx.get_order(c, r).await
    }
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.bx.cancel_order(c, r).await
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.bx.book(s).await
    }
    fn now(&self) -> f64 {
        *self.t.lock().unwrap()
    }
    fn wall_ms(&self) -> i64 {
        self.bx.wall_ms()
    }
    async fn nap(&mut self, sec: f64) {
        self.adv(sec)
    }
    async fn sleep(&mut self, sec: f64) {
        self.adv(sec)
    }
    async fn prepare(&mut self, symbol: &str, leverage: Option<i64>) -> Result<Option<Value>, String> {
        self.bx.prepare(symbol, leverage).await
    }
}

fn params(a: &Value) -> Vec<(String, P)> {
    a.as_array().unwrap().iter().map(|kv| (kv[0].as_str().unwrap().to_string(), P::of(Some(&kv[1])))).collect()
}

fn method(m: &str) -> &'static str {
    match m {
        "GET" => "GET",
        "POST" => "POST",
        "DELETE" => "DELETE",
        "PUT" => "PUT",
        _ => panic!("{m}"),
    }
}

fn phase_s(p: Phase) -> &'static str {
    match p {
        Phase::Rejected => "rejected",
        Phase::Error => "error",
        _ => "?",
    }
}

fn order_json(r: &Reply) -> Value {
    match r {
        Ok(o) => json!({"orderId": o.order_id, "status": o.status, "executedQty": o.executed_qty}),
        Err(e) => json!({"__error__": e}),
    }
}

async fn run(v: &mut Virt, fn_: &str, a: &Value) -> Value {
    let s = |k: &str| a[k].as_str().unwrap().to_string();
    let f = |k: &str| a[k].as_f64().unwrap();
    let b = |k: &str| a.get(k).and_then(Value::as_bool).unwrap_or(false);
    if let Some(o) = a.get("offset").and_then(Value::as_i64) {
        v.bx.offset = o;
    }
    let com = a.get("common").cloned().unwrap_or(json!({"symbol": SYM, "side": "BUY", "positionSide": "LONG"}));
    let (cs, cd, cp) = (com["symbol"].as_str().unwrap().to_string(), com["side"].as_str().unwrap().to_string(), com["positionSide"].as_str().unwrap().to_string());
    let c = Common { symbol: &cs, side: &cd, position_side: &cp };
    match fn_ {
        "sign" => {
            let p = params(&a["params"]);
            let p: Vec<(&str, P)> = p.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            json!(sign::sign_query(&p, &s("secret"), sign::ms(v.bx.time()) + a["offset"].as_i64().unwrap()))
        }
        "signed" => {
            let p = params(&a["params"]);
            let p: Vec<(&str, P)> = p.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            v.bx.signed(method(&s("method")), &s("path"), &p).await
        }
        "clock_offset" => json!(v.bx.clock_offset().await),
        "maker_price" => match v.bx.book_ticker(&s("symbol")).await {
            Ok((bid, ask)) => json!([if s("side") == "BUY" { bid } else { ask }, bid, ask]),
            Err(e) => json!({"__raise__": e}),
        },
        "load_filters" => match v.bx.load_filters(&s("symbol")).await {
            Ok(x) => json!({"step": x.step, "tick": x.tick, "min_qty": x.min_qty, "min_notional": x.min_notional}),
            Err(e) => json!({"__raise__": e}),
        },
        "ensure_closed" => json!(exec::ensure_closed(v, &c, a["oid"].as_i64(), 3).await),
        "resolve_unknown_post" => {
            let order: Reply = Err(a["order"]["__error__"].as_str().unwrap().into());
            match exec::resolve_unknown_post(v, &c, &s("cid"), order).await {
                Ok(r) => order_json(&r),
                Err(_) => Value::Null,
            }
        }
        "fill_maker" => {
            let st: Shared = Arc::new(Mutex::new(ExecState::default()));
            let dl = v.now() + 120.0;
            let (done, px, rp, err) = exec::fill_maker(v, &c, f("price"), f("total"), dl, &st).await;
            let g = st.lock().unwrap().clone();
            json!({"out": [done, px, rp, err.map(|(p, e)| json!({"phase": phase_s(p), "error": e}))],
                   "state": {"filled": g.filled, "repegs": g.repegs, "limit_price": g.limit_price, "order_id": g.order_id}})
        }
        "current_leverage" => json!(bracket::current_leverage(&mut v.bx, SYM).await),
        "ensure_leverage" => bracket::ensure_leverage(&mut v.bx, SYM, a["target"].as_i64().unwrap()).await,
        "cancel_bracket" => bracket::cancel_bracket(&mut v.bx, a.get("symbol").and_then(Value::as_str).unwrap_or(SYM), &s("pside"), b("only_sl"), b("only_tp")).await,
        "clear_bracket" => bracket::clear_bracket(&mut v.bx, SYM, &s("pside"), b("only_sl"), b("only_tp")).await,
        "commission_bp" => bracket::commission_bp(&mut v.bx, SYM).await,
        "cancel_exit_orders" => bracket::cancel_exit_orders(&mut v.bx, SYM, &s("pside")).await,
        "run_entry_limit" => {
            let plan = rtorder::watch::exec_plan_of(&a["plan"]).unwrap();
            let st: Shared = Arc::new(Mutex::new(ExecState::default()));
            v.hook = Some((st.clone(), a["act_at"].as_u64().unwrap() as u32, exec::Action::Market));
            v.gets = 0;
            exec::run_entry_limit(v, &plan, &st, a.get("max_sec").and_then(Value::as_f64).unwrap_or(600.0), false, false).await;
            v.hook = None;
            let g = st.lock().unwrap().clone();
            json!({"phase": rtorder::watch::phase_name(g.phase), "filled": g.filled, "quantity": g.quantity, "taker_qty": g.taker_qty, "taker_order_id": g.taker_order_id,
                   "taker_reason": g.taker_reason, "error": g.error, "repegs": g.repegs, "cancel_reason": g.cancel_reason})
        }
        "run_entry" | "run_exit" | "run_entry_stop" => {
            // 원본 run_* 는 시작에 _clock_offset(시계 요청 1)을 한다 -- Rust 는 exec 머리(Exchange::prepare)
            let plan = rtorder::watch::exec_plan_of(&a["plan"]).unwrap();
            let st: Shared = Arc::new(Mutex::new(ExecState::default()));
            if a.get("state0").and_then(|s| s.get("action")) == Some(&json!("market")) {
                st.lock().unwrap().action = Some(exec::Action::Market);
            }
            match fn_ {
                "run_entry" => exec::run_entry(v, &plan, &st).await,
                "run_exit" => exec::run_exit(v, &plan, &st).await,
                _ => exec::run_entry_stop(v, &plan, &st, |_| false, a.get("max_sec").and_then(Value::as_f64).unwrap_or(60.0)).await,
            }
            let g = st.lock().unwrap().clone();
            json!({"phase": rtorder::watch::phase_name(g.phase), "filled": g.filled, "quantity": g.quantity, "taker_qty": g.taker_qty, "taker_order_id": g.taker_order_id,
                   "taker_reason": g.taker_reason, "error": g.error, "repegs": g.repegs, "cancel_reason": g.cancel_reason})
        }
        "place_bracket" => bracket::place_bracket(&mut v.bx, &a["bracket"], SYM, &s("pside"), b("strict")).await,
        "place_tp_ladder" => bracket::place_tp_ladder(&mut v.bx, a["legs"].as_array().unwrap(), SYM, &s("pside")).await.unwrap_or_else(|e| json!({"__raise__": e})),
        "cancel_wait_orphans" => {
            let syms: Vec<String> = a["symbols"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
            bracket::cancel_wait_orphans(&mut v.bx, &syms.iter().map(String::as_str).collect::<Vec<_>>()).await
        }
        x => panic!("모르는 함수 {x}"),
    }
}

/// 값 비교. 파이썬 예외 문구를 그대로 못 만드는 자리(JSONDecodeError)는 앞머리(종류)만 -- «숫자로 시작하지 않는다»가 지키는 뜻이다.
fn same(py: &Value, rs: &Value, path: &str, out: &mut Vec<String>) {
    match (py, rs) {
        (Value::String(a), Value::String(b)) if a.starts_with("JSONDecodeError: ") || a.contains("(JSONDecodeError: ") => {
            if !b.starts_with("JSONDecodeError: ") && !b.contains("(JSONDecodeError: ") {
                out.push(format!("{path}: py {a} rs {b}"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let (mut ka, mut kb): (Vec<_>, Vec<_>) = (a.keys().collect(), b.keys().collect());
            (ka.sort(), kb.sort());
            if ka != kb {
                out.push(format!("{path}: 키 py {ka:?} rs {kb:?}"));
                return;
            }
            for k in a.keys() {
                same(&a[k], &b[k], &format!("{path}.{k}"), out);
            }
        }
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                same(x, y, &format!("{path}[{i}]"), out);
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            if a.as_f64().unwrap().to_bits() != b.as_f64().unwrap().to_bits() {
                out.push(format!("{path}: py {a} rs {b}"));
            }
        }
        _ => {
            if py != rs {
                out.push(format!("{path}: py {py} rs {rs}"));
            }
        }
    }
}

#[tokio::test]
async fn http_fixture_replay_matches_python() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/http_fixture.json")).unwrap();
    let (timeout, hang, t0) = (doc["timeout"].as_f64().unwrap(), doc["hang"].as_f64().unwrap(), doc["t0"].as_f64().unwrap());
    let queue: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
    let q = queue.clone();
    let srv = serve(Arc::new(move |_g: &Got| match q.lock().unwrap().pop_front() {
        Some(r) => act_of(&r, hang),
        None => Act::Send { status: 599, ctype: "application/json".into(), body: r#"{"msg":"각본 소진"}"#.into(), headers: vec![] },
    }));
    let ban_dir = std::env::temp_dir().join(format!("rtorder_http_replay_{}", std::process::id()));
    let (mut bad, mut nreq, mut ncalls, mut nurl, mut nconn_eq) = (Vec::new(), 0, 0, 0, 0);
    let scns = doc["scenarios"].as_array().unwrap();
    let mut nexec = 0;
    // 2026-11-21 원본 PR 61 도 비상 스탑을 안 건다(배포 전 dbsl 정리 경로는 양쪽에 있음) -- 스탑 요청을 빼던 대조는 없앴다: 요청 열이 통째로 같다.
    for sc in scns {
        let name = sc["name"].as_str().unwrap();
        *queue.lock().unwrap() = sc["responses"].as_array().unwrap().iter().cloned().collect();
        srv.log.lock().unwrap().clear();
        let c0 = srv.conns.load(Ordering::SeqCst);
        let ban = ban_dir.join(format!("ban_{nreq}"));
        let _ = std::fs::remove_file(&ban);
        let mut v = Virt::new(&srv.base, sc["keys"].as_bool().unwrap(), timeout, t0, ban.clone());
        let mut d = Vec::new();
        for (i, cl) in sc["calls"].as_array().unwrap().iter().enumerate() {
            v.adv(cl["advance"].as_f64().unwrap());
            let rs = run(&mut v, cl["fn"].as_str().unwrap(), &cl["args"]).await;
            let rs: Value = serde_json::from_str(&serde_json::to_string(&rs).unwrap().replace(&srv.base, "{BASE}")).unwrap();
            let mut py = sc["results"][i].clone();
            let mut rs = rs;
            match cl["fn"].as_str().unwrap() {
                // 원본은 찾은 주문 dict 를 그대로 돌려준다 -- Rust `Order` 가 읽는 세 칸만(executedQty 는 float(...) 한 값)
                "resolve_unknown_post" if py.get("orderId").is_some() => {
                    py = json!({"orderId": py["orderId"], "status": py["status"], "executedQty": sign::executed_qty(&py).unwrap()});
                }
                // 원본 state 에 한 번도 안 쓴 칸(None)은 Rust 기본값(0) -- 비교에서 뺀다
                "fill_maker" => {
                    let nulls: Vec<String> = py["state"].as_object().unwrap().iter().filter(|(_, x)| x.is_null()).map(|(k, _)| k.clone()).collect();
                    for k in nulls {
                        py["state"].as_object_mut().unwrap().remove(&k);
                        rs["state"].as_object_mut().unwrap().remove(&k);
                    }
                }
                f if f.starts_with("run_") => {
                    let nulls: Vec<String> = py.as_object().unwrap().iter().filter(|(_, x)| x.is_null()).map(|(k, _)| k.clone()).collect();
                    for k in nulls {
                        if rs[&k].is_null() || rs[&k] == json!(0.0) || rs[&k] == json!(0) {
                            py.as_object_mut().unwrap().remove(&k);
                            rs.as_object_mut().unwrap().remove(&k);
                        }
                    }
                    nexec += 1;
                }
                _ => {}
            }
            same(&py, &rs, &format!("결과[{i}]"), &mut d);
            if cl["fn"] == "sign" {
                nurl += 1;
            }
            ncalls += 1;
        }
        let got: Vec<Value> = srv.log.lock().unwrap().iter().map(|g| json!({"method": g.method, "target": g.target, "key": g.key})).collect();
        let want = sc["requests"].as_array().unwrap();
        if got != *want {
            let at = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            d.push(format!("요청 열 다름(#{at}): py {} rs {}", want.get(at).map_or("없음".into(), |x| x.to_string()), got.get(at).map_or("없음".into(), |x| x.to_string())));
        }
        // 연결 재사용(keep-alive): 시나리오 하나에서 연 TCP 연결 수가 aiohttp 와 같다(끊김·시간초과 뒤에만 새로 연다)
        let opened = srv.conns.load(Ordering::SeqCst) - c0;
        if opened as u64 != sc["conns"].as_u64().unwrap() {
            d.push(format!("연결 수 py {} rs {opened}", sc["conns"]));
        }
        nconn_eq += 1;
        nreq += got.len();
        nurl += got.iter().filter(|g| g["target"].as_str().unwrap().contains("signature=")).count();
        if !queue.lock().unwrap().is_empty() {
            d.push("각본이 남았다(요청을 덜 보냈다)".into());
        }
        // 밴 기록: 418 은 본문의 해제 시각이라 글자까지 · 429 는 «실제 지금 + Retry-After(없으면 30초)»라 실행 시각에 따라 다르다 -- 있음·폭만 본다
        let rb = std::fs::read_to_string(&ban).ok();
        match (sc["ban"].as_str(), rb.as_deref()) {
            (None, None) => {}
            (Some(p), Some(r)) if name.contains("418") => if p != r { d.push(format!("밴 기록 py {p:?} rs {r:?}")) },
            (Some(_), Some(r)) => {
                let left = r.trim().parse::<f64>().unwrap() - std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
                let want = if name.contains("retry-after") { 5.0 } else { 30.0 };
                if !(want - 3.0..=want + 0.5).contains(&left) {
                    d.push(format!("429 밴 남은 초 {left} (기대 {want})"));
                }
            }
            (p, r) => d.push(format!("밴 기록 py {p:?} rs {r:?}")),
        }
        // is_post_only_reject 를 오류 문자열에 -- 파이썬과 같은 판정
        for (i, r) in sc["results"].as_array().unwrap().iter().enumerate() {
            let e = r.get("__error__").and_then(Value::as_str).unwrap_or("");
            if exec::is_post_only_reject(e) != sc["post_only"][i].as_bool().unwrap() {
                d.push(format!("is_post_only_reject({e:?})"));
            }
        }
        if !d.is_empty() {
            bad.push(format!("{name}: {}", d.join(" | ")));
        }
    }
    let _ = std::fs::remove_dir_all(&ban_dir);
    println!("집행 시장가 다리 시나리오 {nexec}");
    assert_eq!(nexec, 13, "집행 시나리오 수(M4 2 포함 · M3 2 는 PR 61 에서 지움)");
    println!("시나리오 {} · 호출 {ncalls} · 가짜 서버가 받은 요청 {nreq} · 글자까지 비교한 서명 URL {nurl} · 연결 {} · 연결 수 대조 시나리오 {nconn_eq}", scns.len(), srv.conns.load(Ordering::SeqCst));
    for b in &bad {
        println!("불일치 {b}");
    }
    assert!(bad.is_empty(), "불일치 {}건 / {}", bad.len(), scns.len());
}
