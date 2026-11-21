//! 원본 «전체 흐름» 정답지 재생(2026-11-21 일곱째 조각). 정답지 = tools/gen_flow_fixture.py 가 진짜 `orders.entry_then_bracket`·`bracket_tick` +
//! 진짜 집행·브래킷 함수 + aiohttp 를 127.0.0.1 모의 거래소에 붙여 뽑은 «걸음마다 나간 요청 열 · 받은 응답 열 · 상태 칸 · 무장 파일 · 복구 파일 · 가짜 의존성 호출».
//! 여기서는 같은 응답 열을 프로세스 안 가짜 서버(127.0.0.1)에 걸고 Rust 실행기(`runner::entry_then_bracket`)와 감시(`Watcher::bracket_tick`)를
//! 진짜 `http::Binance`·`bracket::*` 로 돌려 대조한다. 2026-11-21 PR 61 판부터 요청 열이 통째로 같다(배포 전 dbsl 정리 목록 openAlgoOrders 도 양쪽).

use rtorder::assemble::{Desk, Quotes};
use rtorder::bracket;
use rtorder::exec::{Common, Exchange, LimitOrder, OrderRef, Reply};
use rtorder::http::Binance;
use rtorder::plan::Filters;
use rtorder::runner::entry_then_bracket;
use rtorder::watch::{iso_utc, run_exit_into, Obj, WatchIo, Watcher};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

include!("common/server.rs");

const SYM: &str = "ETHUSDC";
type Clock = Arc<Mutex<f64>>;
type Calls = Arc<Mutex<Vec<Value>>>;

fn bx(base: &str, t0: f64, t: &Clock, ban: &std::path::Path) -> Binance {
    let mut b = Binance::new(base, "test_key", "test_secret", std::time::Duration::from_secs(2), ban.to_path_buf());
    let tc = t.clone();
    b.clock = Arc::new(move || t0 + *tc.lock().unwrap());
    let ts = t.clone();
    b.virtual_sleep = Some(Arc::new(move |s| *ts.lock().unwrap() += s));
    b
}

/// 진짜 전송 + 가짜 시계(원본 생성기가 time·nap·asyncio.sleep 을 갈아끼운 것과 같다).
struct Vx {
    bx: Binance,
    t: Clock,
}
impl Exchange for Vx {
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
        self.bx.book_ticker(s).await
    }
    fn now(&self) -> f64 {
        *self.t.lock().unwrap()
    }
    fn wall_ms(&self) -> i64 {
        self.bx.wall_ms()
    }
    async fn nap(&mut self, s: f64) {
        *self.t.lock().unwrap() += s
    }
    async fn sleep(&mut self, s: f64) {
        *self.t.lock().unwrap() += s
    }
    async fn prepare(&mut self, symbol: &str, leverage: Option<i64>) -> Result<Option<Value>, String> {
        self.bx.prepare(symbol, leverage).await
    }
}

/// 감시·무장의 바깥: 거래소 함수는 진짜(`bracket::*` · 청산 = run_exit_into + 진짜 전송), 계좌·5분봉·1초 체결가·청산 계획 조립은 정답지의 가짜 값.
struct Io {
    bx: Binance,
    ex: Vx,
    t: Clock,
    t0: f64,
    calls: Calls,
    pos: Value,
    sec: Vec<(f64, f64)>,
    exit_plans: VecDeque<Value>,
}
impl Io {
    fn note(&self, v: Value) {
        let t = *self.t.lock().unwrap();
        let mut a = vec![json!(t)];
        a.extend(v.as_array().unwrap().iter().cloned());
        self.calls.lock().unwrap().push(Value::Array(a));
    }
}
impl WatchIo for Io {
    fn now(&self) -> f64 {
        self.t0 + *self.t.lock().unwrap()
    }
    fn exec_enabled(&self) -> bool {
        true
    }
    fn order_locked(&self) -> bool {
        false
    }
    async fn nap(&mut self, _s: f64) -> bool {
        unreachable!()
    }
    async fn account(&mut self) -> Result<Value, String> {
        self.note(json!(["account"]));
        let p: Vec<Value> = ["LONG", "SHORT"].iter().filter(|s| self.pos[**s].as_f64().unwrap() != 0.0).map(|s| json!({"symbol": SYM, "side": s, "qty": self.pos[*s]})).collect();
        Ok(json!({"ok": true, "generated_at": iso_utc(self.now()), "positions": p}))
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        unreachable!()
    }
    async fn bars(&mut self, a: &str) -> Result<Vec<Value>, String> {
        self.note(json!(["bars", a]));
        Ok(vec![])
    }
    fn sec_prices(&mut self, a: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        self.note(json!(["sec", a]));
        Ok(Some(self.sec.clone()))
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        Ok(bracket::clear_bracket(&mut self.bx, s, p, only_sl, only_tp).await)
    }
    async fn exit_plan(&mut self, p: &str, f: f64, a: &str, s: &str) -> Result<Value, Value> {
        self.note(json!(["exit_plan", p, f, true, a, s]));
        Ok(self.exit_plans.pop_front().expect("원본보다 청산 계획을 더 묻는다"))
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let (t, t0) = (self.t.clone(), self.t0);
        run_exit_into(&mut self.ex, plan, slot, || iso_utc(t0 + *t.lock().unwrap())).await
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        Ok(bracket::place_bracket(&mut self.bx, spec, s, p, strict).await)
    }
    async fn place_tp_ladder(&mut self, legs: &Value, s: &str, p: &str) -> Result<Value, String> {
        bracket::place_tp_ladder(&mut self.bx, legs.as_array().unwrap(), s, p).await
    }
    async fn cancel_wait_orphans(&mut self, syms: &[String]) -> Result<Value, String> {
        Ok(bracket::cancel_wait_orphans(&mut self.bx, &syms.iter().map(String::as_str).collect::<Vec<_>>()).await)
    }
    async fn cancel_exit_orders(&mut self, s: &str, p: &str) -> Result<Value, String> {
        Ok(bracket::cancel_exit_orders(&mut self.bx, s, p).await)
    }
}

/// 지정가 재조정이 읽는 공개 호가(원본 fetch_binance_json)·필터(load_filters) -- 정답지의 시각표.
struct Q {
    books: Vec<(f64, Value)>,
    t: Clock,
    calls: Calls,
}
impl Q {
    fn note(&self, v: Value) {
        let mut a = vec![json!(*self.t.lock().unwrap())];
        a.extend(v.as_array().unwrap().iter().cloned());
        self.calls.lock().unwrap().push(Value::Array(a));
    }
}
impl Quotes for Q {
    async fn book(&mut self, s: &str) -> Result<Value, String> {
        self.note(json!(["book", s]));
        let t = *self.t.lock().unwrap();
        Ok(self.books.iter().filter(|(t0, _)| t >= *t0).last().unwrap().1[s].clone())
    }
    async fn filters(&mut self, s: &str) -> Result<Filters, String> {
        self.note(json!(["filters", s]));
        Ok(Filters { step: 0.001, tick: 0.01, min_qty: 0.001, min_notional: 5.0 })
    }
    async fn vol(&mut self, _: &str) -> Option<f64> {
        unreachable!()
    }
    async fn funding(&mut self, _: &str) -> Option<f64> {
        unreachable!()
    }
    async fn fees(&mut self, _: &str) -> Value {
        unreachable!()
    }
}

fn act_of(r: &Value) -> Act {
    Act::Send { status: r["status"].as_u64().unwrap() as u16, ctype: "application/json".into(), body: serde_json::to_string(&r["body"]).unwrap(), headers: vec![] }
}

/// 원본 칸에 없고 Rust 가 비어 있는 값(None·0)으로 쓴 칸은 같은 뜻이다(파이썬 state.get 기본값).
fn slot_diff(py: &Value, rs: &Obj, out: &mut Vec<String>) {
    let py = py.as_object().unwrap();
    for (k, v) in py {
        match rs.get(k) {
            None => out.push(format!("칸 {k}: Rust 에 없음 (원본 {v})")),
            Some(r) if !same(v, r) => out.push(format!("칸 {k}: 원본 {v} · Rust {r}")),
            _ => {}
        }
    }
    for (k, r) in rs {
        if !py.contains_key(k) && !(r.is_null() || r == &json!(0.0) || r == &json!(0) || r == &json!(false)) {
            out.push(format!("칸 {k}: 원본에 없음 (Rust {r})"));
        }
    }
}
fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w)),
        _ => a == b,
    }
}

#[tokio::test]
async fn flow_fixture_matches_python_full_flow() {
    let doc: Value = serde_json::from_str(include_str!("fixtures/flow_fixture.json")).unwrap();
    let t0 = doc["t0"].as_f64().unwrap();
    let queue: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
    let q2 = queue.clone();
    let srv = serve(Arc::new(move |_g: &Got| match q2.lock().unwrap().pop_front() {
        Some(r) => act_of(&r),
        None => Act::Send { status: 599, ctype: "application/json".into(), body: r#"{"msg":"각본 소진"}"#.into(), headers: vec![] },
    }));
    local_only(&srv.base);
    let (mut bad, mut nreq, mut nstep, mut nsigned) = (vec![], 0, 0, 0);
    for sc in doc["scenarios"].as_array().unwrap() {
        let name = sc["name"].as_str().unwrap();
        let dir = std::env::temp_dir().join(format!("rtorder_flow_{}_{}", std::process::id(), nstep));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ban = dir.join("ban");
        let t: Clock = Arc::new(Mutex::new(0.0));
        let calls: Calls = Arc::new(Mutex::new(vec![]));
        let mut desk = Desk::new(Watcher::new(&dir, SYM));
        desk.w.entry = serde_json::from_value(json!({"phase": "submitting"})).unwrap();
        if sc["file0"].is_object() {
            std::fs::write(dir.join("manual_bracket_state.json"), serde_json::to_string(&sc["file0"]).unwrap()).unwrap();
        }
        let mut ex = Vx { bx: bx(&srv.base, t0, &t, &ban), t: t.clone() };
        let mut io = Io { bx: bx(&srv.base, t0, &t, &ban), ex: Vx { bx: bx(&srv.base, t0, &t, &ban), t: t.clone() }, t: t.clone(), t0, calls: calls.clone(),
                          pos: json!({"LONG": 0.0, "SHORT": 0.0}), sec: vec![], exit_plans: VecDeque::new() };
        let mut quotes = Q { books: sc["books"].as_array().unwrap().iter().map(|b| (b[0].as_f64().unwrap(), b[1].clone())).collect(), t: t.clone(), calls: calls.clone() };
        let mut prev_end = 0.0;
        for (i, st) in sc["steps"].as_array().unwrap().iter().enumerate() {
            let mut d = vec![];
            *t.lock().unwrap() = st["t"].as_f64().unwrap();
            if st["t"].as_f64().unwrap() < prev_end {
                d.push("걸음 시작이 앞 걸음 끝보다 앞선다".into());
            }
            let reqs = st["requests"].as_array().unwrap();
            let resp = st["responses"].as_array().unwrap();
            assert_eq!(reqs.len(), resp.len());
            *queue.lock().unwrap() = resp.iter().cloned().collect();
            srv.log.lock().unwrap().clear();
            calls.lock().unwrap().clear();
            io.sec = st["sec"].as_array().unwrap().iter().map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap())).collect();
            io.exit_plans = st["exit_plans"].as_array().unwrap().iter().cloned().collect();
            if st["op"] == "entry" {
                let plan = st["plan"].as_object().unwrap().clone();
                let mut slot = std::mem::take(&mut desk.w.entry);
                entry_then_bracket(&mut desk, &mut ex, &mut io, &mut quotes, &plan, &mut slot, st["limit"].as_bool().unwrap()).await;
                desk.w.entry = slot;
            } else if let Err(e) = desk.w.bracket_tick(&mut io).await {
                d.push(format!("감시 틱 예외 {e} (원본 {})", st["raised"]));
            }
            let got: Vec<Value> = srv.log.lock().unwrap().iter().map(|g| json!({"method": g.method, "target": g.target, "key": g.key})).collect();
            let want: Vec<Value> = reqs.clone();
            if got != want {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                d.push(format!("요청 열 다름 #{at}/{}: 원본 {} · Rust {}", want.len(), want.get(at).map_or("없음".into(), |x| x.to_string()), got.get(at).map_or("없음".into(), |x| x.to_string())));
            }
            if !queue.lock().unwrap().is_empty() {
                d.push(format!("응답이 {}개 남았다", queue.lock().unwrap().len()));
            }
            slot_diff(&st["entry"], &desk.w.entry, &mut d);
            let file = std::fs::read_to_string(dir.join("manual_bracket_state.json")).ok().map(|s| serde_json::from_str::<Value>(&s).unwrap());
            if !same(&st["file"], file.as_ref().unwrap_or(&Value::Null)) {
                d.push(format!("무장 파일: 원본 {} · Rust {}", st["file"], file.unwrap_or(Value::Null)));
            }
            let mut limits: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok()).filter(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json")).collect();
            limits.sort();
            if json!(limits) != st["limits"] {
                d.push(format!("복구 파일: 원본 {} · Rust {limits:?}", st["limits"]));
            }
            let c = calls.lock().unwrap().clone();
            if !same(&json!(c), &st["calls"]) {
                d.push(format!("가짜 의존성 호출: 원본 {} · Rust {}", st["calls"], json!(c)));
            }
            let logs = std::mem::take(&mut desk.w.logs);
            if json!(logs) != st["logs"] {
                d.push(format!("로그: 원본 {} · Rust {logs:?}", st["logs"]));
            }
            let t_end = *t.lock().unwrap();
            if t_end != st["t_end"].as_f64().unwrap() {
                d.push(format!("끝 시각: 원본 {} · Rust {t_end}", st["t_end"]));
            }
            io.pos = st["pos"].clone();
            prev_end = t_end;
            nreq += got.len();
            nsigned += got.iter().filter(|g| g["target"].as_str().unwrap().contains("signature=")).count();
            nstep += 1;
            if !d.is_empty() {
                bad.push(format!("{name} 걸음 {i}({}): {}", st["op"], d.join(" | ")));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("시나리오 {} · 걸음 {nstep} · 요청 {nreq}(서명 URL 글자까지 {nsigned})", doc["scenarios"].as_array().unwrap().len());
    for b in &bad {
        println!("불일치 {b}");
    }
    assert!(bad.is_empty(), "불일치 {}건", bad.len());
}
