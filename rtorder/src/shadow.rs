//! 그림자 실행(2026-11-21 일곱째 뒤 C) -- 주문은 **실제로 내지 않고** 파이썬과 대조한다.
//!
//! ① 모드: 기본 = 그림자(`Mode::Shadow`). 실전송(`Mode::Live`)은 cargo feature `live` 빌드 + `RTORDER_MODE=live` + `RTORDER_LIVE_CONFIRM=<LIVE_CONFIRM>`
//!    셋이 다 있어야만 만들어진다(환경변수 하나로는 안 켜진다). 그림자 모드의 `http::Binance` 는 GET·listenKey 가 아닌 요청을 **보내기 전에 panic**.
//! ② 재생: 파이썬이 남긴 «작업 하나(chain)»의 기록 -- 시작(계획·칸·무장 파일) · 거래소 왕복(wire: 메서드·경로·서명 전 쿼리·응답) · 읽은 입력(계좌·1초가·5분봉·청산 계획·호가·필터) ·
//!    끝(칸·무장 파일·로그) -- 을 받아 «같은 접수가 왔다면 Rust 가 낼 요청 열»을 127.0.0.1 가짜 서버(파이썬이 받은 응답을 차례로 돌려준다) 위에서 돌려 비교한다.
//!    기록 형식(파이썬이 지금은 안 남긴다 -- 필요한 칸 목록은 RULES.md «그림자 기록») = JSONL 한 줄에 `{"rec": "start"|"wire"|"input"|"end", "chain": ..}`.
//!    재생은 진짜 `http::Binance`(루프백)·`runner::entry_then_bracket`·`Watcher::bracket_tick`·`bracket::*` 를 돈다 -- 시험용 대역이 아니다.

use crate::assemble::{Desk, Quotes};
use crate::bracket;
use crate::exec::{Common, Exchange, LimitOrder, OrderRef, Reply};
use crate::fakesrv::{serve, Act, Got};
use crate::http::Binance;
use crate::plan::Filters;
use crate::intake::SlotRef;
use crate::runner::entry_then_bracket;
use crate::watch::{iso_utc, run_exit_into, Obj, WatchIo, Watcher};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ── ① 모드 ───────────────────────────────────────────────────────────────────
pub const LIVE_CONFIRM: &str = "실주문을-보낸다";

/// 실전송 허가증 -- 이 모듈 밖에서는 만들 수 없다(`mode_from` 만 · feature `live` 빌드에서만).
pub struct LiveProof(());

pub enum Mode {
    Shadow,
    Live(LiveProof),
}

/// 환경에서 모드를 읽는다(`get` = 환경변수 읽기 -- 시험이 바꿔 넣는다). 모르는 값·확인 문구 없음 = Err(시작하지 않는다).
pub fn mode_from(get: &dyn Fn(&str) -> Option<String>) -> Result<Mode, String> {
    match get("RTORDER_MODE").unwrap_or_default().as_str() {
        "" | "shadow" => Ok(Mode::Shadow),
        "live" => live(get),
        x => Err(format!("RTORDER_MODE 는 shadow 또는 live 입니다(받은 값 {x:?})")),
    }
}
#[cfg(feature = "live")]
fn live(get: &dyn Fn(&str) -> Option<String>) -> Result<Mode, String> {
    if get("RTORDER_LIVE_CONFIRM").as_deref() == Some(LIVE_CONFIRM) {
        Ok(Mode::Live(LiveProof(())))
    } else {
        Err(format!("실전송은 RTORDER_LIVE_CONFIRM={LIVE_CONFIRM} 도 있어야 켜집니다"))
    }
}
#[cfg(not(feature = "live"))]
fn live(_get: &dyn Fn(&str) -> Option<String>) -> Result<Mode, String> {
    Err("이 빌드에는 실전송이 없습니다(cargo feature live 로 빌드해야 함) -- 그림자로만 돕니다".into())
}

// ── ② 재생 ───────────────────────────────────────────────────────────────────
/// 작업 하나의 기록. `op` = "entry"(entry_then_bracket) | "tick"(bracket_tick) | "exit"(청산 접수 -- rearm = 부분 청산 뒤 익절 다시).
#[derive(Debug, Clone, Default)]
pub struct Chain {
    pub id: String,
    pub start: Value,
    pub wire: Vec<Value>,
    pub inputs: Vec<Value>,
    pub end: Value,
}

/// JSONL → 작업들(시작 순서). 모르는 rec·모르는 칸은 무시(기록 쪽이 칸을 더해도 깨지지 않게).
pub fn chains(jsonl: &str) -> Result<Vec<Chain>, String> {
    let (mut order, mut by): (Vec<String>, HashMap<String, Chain>) = (vec![], HashMap::new());
    for (n, line) in jsonl.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{}번째 줄: {e}", n + 1))?;
        let id = v.get("chain").map(crate::sign::py_str).ok_or_else(|| format!("{}번째 줄: chain 없음", n + 1))?;
        if id == "*" {
            continue; // 파이썬 체결 알림(wake_fills · 모든 작업) -- 재생은 아직 깨움을 넣지 않는다(시각만 갈린다)
        }
        let c = by.entry(id.clone()).or_insert_with(|| {
            order.push(id.clone());
            Chain { id: id.clone(), ..Default::default() }
        });
        match v.get("rec").and_then(Value::as_str) {
            Some("start") => c.start = v,
            Some("wire") => c.wire.push(v),
            Some("input") => c.inputs.push(v),
            Some("end") => c.end = v,
            _ => {}
        }
    }
    // 시작 줄이 없는 chain = 작업이 아니다(접수 요청 자체의 조회 -- 파이썬 order_request_log 의 chain) -- 재생하지 않는다
    Ok(order.into_iter().map(|id| by.remove(&id).unwrap()).filter(|c| !c.start.is_null()).collect())
}

/// 비교에서 뺄 벽시계 칸(실기록 = 진짜 시계 · 재생 = 가짜 시계): 이름이 `_at` 으로 끝나거나 `at`·`wait_until`·`retry_after`(30초 뒤 다시 -- 시각).
pub fn drop_times(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(k, _)| !(k.ends_with("_at") || *k == "at" || *k == "wait_until" || *k == "retry_after")).map(|(k, x)| (k.clone(), drop_times(x))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(drop_times).collect()),
        x => x.clone(),
    }
}

/// 요청 하나를 비교용 글자로. exact = 글자 그대로(가짜 시계 기록) · 아니면 timestamp·signature 를 빼고 표식 뒤 숫자(벽시계 ms)를 `#` 로.
pub fn normalize(method: &str, target: &str, exact: bool) -> String {
    if exact {
        return format!("{method} {target}");
    }
    let target = target.split("&signature=").next().unwrap_or(target);
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    let kept: Vec<String> = q.split('&').filter(|kv| !kv.is_empty() && !kv.starts_with("timestamp=") && !kv.starts_with("signature="))
        .map(|kv| match kv.split_once('=') {
            Some((k @ ("newClientOrderId" | "origClientOrderId"), v)) if v.starts_with("db") && v.len() > 5 => format!("{k}={}#", &v[..5]),
            _ => kv.to_string(),
        }).collect();
    format!("{method} {path}?{}", kept.join("&"))
}

type Clock = Arc<Mutex<f64>>;
type Calls = Arc<Mutex<Vec<Value>>>;
type Inputs = Arc<Mutex<HashMap<String, VecDeque<Value>>>>;

/// 기록된 입력을 차례로 내준다. 다 쓰면 마지막 값을 다시 쓰고 «추가 입력 요청»으로 적는다(비교에서 드러난다).
fn take(inp: &Inputs, last: &Mutex<HashMap<String, Value>>, kind: &str) -> Value {
    let mut m = inp.lock().unwrap();
    match m.get_mut(kind).and_then(VecDeque::pop_front) {
        Some(v) => {
            last.lock().unwrap().insert(kind.into(), v.clone());
            v
        }
        None => last.lock().unwrap().get(kind).cloned().unwrap_or(Value::Null),
    }
}

/// 작업 도중 화면 입력(원본 기록 what="handler": 접수 핸들러 이름·질의·그때 응답) -- 집행이 기다리다 깨는 자리(nap·sleep)에서 그 시각이 됐으면 부른다.
/// 원본도 집행이 잠든 사이에만 핸들러가 돈다(한 이벤트 루프) · 핸들러는 진짜 `Desk::api_*` 이고 W12 다리(push_live)로 도는 집행에 닿는다.
struct Inject {
    desk: Arc<crate::runner::DeskCell>,
    handlers: Arc<Mutex<VecDeque<Value>>>,
    t: Clock,
    calls: Calls,
    inputs: Inputs,
    last: Arc<Mutex<HashMap<String, Value>>>,
    diffs: Arc<Mutex<Vec<String>>>,
}
impl Inject {
    async fn due(&self) {
        loop {
            let now = *self.t.lock().unwrap();
            let h = {
                let mut q = self.handlers.lock().unwrap();
                match q.front() {
                    Some(h) if h.get("t").and_then(Value::as_f64).is_none_or(|t| t <= now) => q.pop_front().unwrap(),
                    _ => return,
                }
            };
            let args = h.get("args").and_then(Value::as_array).cloned().unwrap_or_default();
            let name = args.first().and_then(Value::as_str).unwrap_or("").to_string();
            let query: crate::intake::Query = args.get(1).and_then(Value::as_object).map(|m| m.iter().map(|(k, v)| (k.clone(), crate::sign::py_str(v))).collect()).unwrap_or_default();
            note(&self.t, &self.calls, json!(["handler", name, args.get(1).cloned().unwrap_or(Value::Null)]));
            let mut q = Q { t: self.t.clone(), calls: self.calls.clone(), inputs: self.inputs.clone(), last: self.last.clone() };
            let mut io = HIo { t: self.t.clone(), calls: self.calls.clone(), inputs: self.inputs.clone(), last: self.last.clone() };
            let mut d = self.desk.lock().await;
            let r = match name.as_str() {
                "api_manual_wait_action" => d.api_wait_action(&mut q, &mut io, &query).await,
                "api_manual_limit_amend" => d.api_limit_amend(&mut q, &mut io, &query).await,
                "api_manual_entry_cancel" => d.api_entry_cancel(&io, &query),
                x => {
                    self.diffs.lock().unwrap().push(format!("모르는 핸들러 {x}"));
                    continue;
                }
            };
            drop(d);
            let want = h.get("value").cloned().unwrap_or(Value::Null);
            let mut dd = vec![];
            if want.get("status").and_then(Value::as_u64) != Some(r.status as u64) {
                dd.push(format!("상태 원본 {} · Rust {}", want["status"], r.status));
            }
            // 응답 본문: 칸(wait·state)은 칸 비교 규칙(원본에 없는 빈 칸 허용) · 나머지는 값 그대로
            let (wb, rb) = (want.get("body").and_then(Value::as_object).cloned().unwrap_or_default(), r.body.as_object().cloned().unwrap_or_default());
            for k in wb.keys().chain(rb.keys().filter(|k| !wb.contains_key(*k))) {
                match (wb.get(k), rb.get(k)) {
                    (Some(a), Some(Value::Object(b))) if ["wait", "state"].contains(&k.as_str()) => slot_diff(a, b, &mut dd),
                    (a, b) if !same(a.unwrap_or(&Value::Null), b.unwrap_or(&Value::Null)) => dd.push(format!("본문 {k}: 원본 {a:?} · Rust {b:?}")),
                    _ => {}
                }
            }
            if !dd.is_empty() {
                self.diffs.lock().unwrap().push(format!("핸들러 {name}: {}", dd.join(" · ")));
            }
        }
    }
}

/// 핸들러가 닿는 계좌·1초가(기록된 입력) -- 원본 assemble_entry_plan(fresh) 의 produce_account · sec_prices.
struct HIo {
    t: Clock,
    calls: Calls,
    inputs: Inputs,
    last: Arc<Mutex<HashMap<String, Value>>>,
}
impl crate::assemble::DeskIo for HIo {
    fn now(&self) -> f64 {
        *self.t.lock().unwrap()
    }
    fn exec_enabled(&self) -> bool {
        true
    }
    async fn account(&mut self, fresh: bool) -> Result<Value, String> {
        let what = if fresh { "produce_account" } else { "account" };
        note(&self.t, &self.calls, json!([what]));
        Ok(take(&self.inputs, &self.last, what))
    }
    fn sec_prices(&mut self, a: &str) -> Option<Vec<(f64, f64)>> {
        note(&self.t, &self.calls, json!(["sec", a]));
        take(&self.inputs, &self.last, "sec").as_array().map(|r| r.iter().map(|x| (x[0].as_f64().unwrap_or(0.0), x[1].as_f64().unwrap_or(0.0))).collect())
    }
    async fn clear_bracket(&mut self, _: &str, _: &str, _: bool, _: bool) -> Result<Value, String> {
        Err("재생: 이 핸들러는 손절 함수를 안 부른다".into())
    }
    async fn place_bracket(&mut self, _: &Value, _: &str, _: &str, _: bool) -> Result<Value, String> {
        Err("재생: 이 핸들러는 손절 함수를 안 부른다".into())
    }
    async fn place_tp_ladder(&mut self, _: &Value, _: &str, _: &str) -> Result<Value, String> {
        Err("재생: 이 핸들러는 손절 함수를 안 부른다".into())
    }
}

struct Vx {
    bx: Binance,
    t: Clock,
    inj: Option<Arc<Inject>>,
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
        *self.t.lock().unwrap() += s;
        if let Some(i) = &self.inj {
            i.due().await;
        }
    }
    async fn sleep(&mut self, s: f64) {
        *self.t.lock().unwrap() += s;
        if let Some(i) = &self.inj {
            i.due().await;
        }
    }
    async fn prepare(&mut self, symbol: &str, leverage: Option<i64>) -> Result<Option<Value>, String> {
        self.bx.prepare(symbol, leverage).await
    }
}

/// 감시·무장의 바깥 = 진짜 `bracket::*`(가짜 서버로) + 기록된 입력.
struct Io {
    bx: Binance,
    ex: Vx,
    t: Clock,
    calls: Calls,
    inputs: Inputs,
    last: Arc<Mutex<HashMap<String, Value>>>,
}
fn note(t: &Clock, calls: &Calls, v: Value) {
    let mut a = vec![json!(*t.lock().unwrap())];
    a.extend(v.as_array().unwrap().iter().cloned());
    calls.lock().unwrap().push(Value::Array(a));
}
impl WatchIo for Io {
    fn now(&self) -> f64 {
        *self.t.lock().unwrap()
    }
    fn exec_enabled(&self) -> bool {
        true
    }
    fn order_locked(&self) -> bool {
        false
    }
    async fn nap(&mut self, _s: f64) -> bool {
        false
    }
    async fn account(&mut self) -> Result<Value, String> {
        note(&self.t, &self.calls, json!(["account"]));
        Ok(take(&self.inputs, &self.last, "account"))
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        note(&self.t, &self.calls, json!(["produce_account"]));
        Ok(take(&self.inputs, &self.last, "produce_account"))
    }
    async fn bars(&mut self, a: &str) -> Result<Vec<Value>, String> {
        note(&self.t, &self.calls, json!(["bars", a]));
        Ok(take(&self.inputs, &self.last, "bars").as_array().cloned().unwrap_or_default())
    }
    fn sec_prices(&mut self, a: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        note(&self.t, &self.calls, json!(["sec", a]));
        let v = take(&self.inputs, &self.last, "sec");
        Ok(v.as_array().map(|r| r.iter().map(|x| (x[0].as_f64().unwrap_or(0.0), x[1].as_f64().unwrap_or(0.0))).collect()))
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        Ok(bracket::clear_bracket(&mut self.bx, s, p, only_sl, only_tp).await)
    }
    async fn exit_plan(&mut self, p: &str, f: f64, a: &str, s: &str) -> Result<Value, Value> {
        note(&self.t, &self.calls, json!(["exit_plan", p, f, true, a, s]));
        let v = take(&self.inputs, &self.last, "exit_plan");
        match v.get("err") {
            // 파이썬 기록 = [본문, 상태](assemble_exit_plan 의 err 튜플) · 감시는 본문 dict 를 본다
            Some(Value::Array(a)) if !a.is_empty() => Err(a[0].clone()),
            Some(e) if !e.is_null() => Err(e.clone()),
            _ => Ok(v.get("plan").cloned().unwrap_or(Value::Null)),
        }
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let t = self.t.clone();
        run_exit_into(&mut self.ex, plan, slot, || iso_utc(*t.lock().unwrap())).await
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        Ok(bracket::place_bracket(&mut self.bx, spec, s, p, strict).await)
    }
    async fn place_tp_ladder(&mut self, legs: &Value, s: &str, p: &str) -> Result<Value, String> {
        bracket::place_tp_ladder(&mut self.bx, legs.as_array().map(Vec::as_slice).unwrap_or(&[]), s, p).await
    }
    async fn cancel_wait_orphans(&mut self, syms: &[String]) -> Result<Value, String> {
        Ok(bracket::cancel_wait_orphans(&mut self.bx, &syms.iter().map(String::as_str).collect::<Vec<_>>()).await)
    }
    async fn cancel_exit_orders(&mut self, s: &str, p: &str) -> Result<Value, String> {
        Ok(bracket::cancel_exit_orders(&mut self.bx, s, p).await)
    }
}

/// 지정가 재조정이 읽는 공개 호가(원본 fetch_binance_json)·필터(load_filters) = 기록된 입력.
struct Q {
    t: Clock,
    calls: Calls,
    inputs: Inputs,
    last: Arc<Mutex<HashMap<String, Value>>>,
}
impl Quotes for Q {
    async fn book(&mut self, s: &str) -> Result<Value, String> {
        note(&self.t, &self.calls, json!(["book", s]));
        Ok(take(&self.inputs, &self.last, "book"))
    }
    async fn filters(&mut self, s: &str) -> Result<Filters, String> {
        note(&self.t, &self.calls, json!(["filters", s]));
        serde_json::from_value(take(&self.inputs, &self.last, "filters")).map_err(|e| format!("filters 기록: {e}"))
    }
    async fn vol(&mut self, s: &str) -> Option<f64> {
        note(&self.t, &self.calls, json!(["vol", s]));
        take(&self.inputs, &self.last, "vol").as_f64()
    }
    async fn funding(&mut self, s: &str) -> Option<f64> {
        note(&self.t, &self.calls, json!(["funding", s]));
        take(&self.inputs, &self.last, "funding").as_f64()
    }
    async fn fees(&mut self, s: &str) -> Value {
        note(&self.t, &self.calls, json!(["fees", s]));
        take(&self.inputs, &self.last, "fees")
    }
}

/// 원본 칸에 없고 Rust 가 비어 있는 값(None·0·false)으로 쓴 칸은 같은 뜻(파이썬 state.get 기본값).
pub fn slot_diff(py: &Value, rs: &Obj, out: &mut Vec<String>) {
    let empty = Obj::new();
    let py = py.as_object().unwrap_or(&empty);
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
/// 값 비교(수는 값으로 · dict 는 키 순서 무관).
pub fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same(v, w))),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(v, w)| same(v, w)),
        _ => a == b,
    }
}

fn act_of(w: &Value) -> Act {
    match w.get("error").and_then(Value::as_str) {
        Some(e) if e.contains("Timeout") => Act::Hang(2.5, Box::new(Act::Close)), // 시간초과로 끝난 요청 = 답을 늦게(재생 클라이언트 시간초과 2초)
        Some(_) => Act::Close,                                                    // 그 밖의 전송 오류 = 끊김
        None => Act::Send { status: w.get("status").and_then(Value::as_u64).unwrap_or(200) as u16, ctype: "application/json".into(),
                            body: match w.get("body") { Some(Value::String(s)) => s.clone(), Some(v) => serde_json::to_string(v).unwrap(), None => "null".into() },
                            headers: vec![] },
    }
}

pub struct Report {
    pub chain: String,
    pub requests: usize,
    /// 차이(값 포함 -- `--show-values` 일 때만 찍는다)
    pub diffs: Vec<String>,
    /// 2026-11-21 값 없는 어긋난 위치(«요청 #3 칸 quantity» · «칸 phase» · «무장 파일 ETHUSDT:LONG.sl_level») -- 실기록 출력은 이것만
    pub where_: Vec<String>,
    pub op: String,
    /// 감시 틱 안 청산 계획 조립 중 파이썬이 보낸 읽기 왕복(/time·commissionRate) -- Rust 재생은 조립을 기록값으로 대신해 안 보낸다(따로 센다)
    pub assembly: usize,
    /// 시작은 있고 끝이 없다(프로세스 재시작으로 잘림) -- 재생하지 않는다
    pub truncated: bool,
}

/// 감시 틱 안 조립 구간의 읽기 왕복 경로(파이썬 produce_account 의 _clock_offset · commission_bp).
pub const ASSEMBLY_PATHS: [&str; 2] = ["/fapi/v1/time", "/fapi/v1/commissionRate"];

/// 감시 틱: 청산 계획 입력(what=exit_plan) 바로 앞에 붙은 조립 왕복을 떼어 낸다(시각 차례로 거꾸로 이어진 것만) · 조립 안 입력(filters)도.
fn split_assembly(c: &Chain) -> (Chain, usize) {
    if c.start.get("op").and_then(Value::as_str) != Some("tick") {
        return (c.clone(), 0);
    }
    let tt = |v: &Value| v.get("t").and_then(Value::as_f64).unwrap_or(0.0);
    let path = |w: &Value| w.get("target").and_then(Value::as_str).unwrap_or("").split('?').next().unwrap_or("").to_string();
    let mut drop = std::collections::BTreeSet::new();
    for ep in c.inputs.iter().filter(|i| i["what"] == "exit_plan") {
        let te = tt(ep);
        // exit_plan 앞의 파이썬 기록을 시각 거꾸로 -- 조립 왕복이 아닌 것(다른 요청·입력)이 나오면 멈춘다
        let mut rows: Vec<(f64, Option<usize>)> = c.wire.iter().enumerate().filter(|(_, w)| w.get("t").is_some() && tt(w) < te).map(|(i, w)| (tt(w), Some(i))).collect();
        rows.extend(c.inputs.iter().filter(|i| tt(i) < te && i["what"] != "filters").map(|i| (tt(i), None)));
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (_, w) in rows {
            match w {
                Some(i) if ASSEMBLY_PATHS.contains(&path(&c.wire[i]).as_str()) => {
                    drop.insert(i);
                }
                _ => break,
            }
        }
    }
    let mut out = c.clone();
    out.wire = c.wire.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, w)| w.clone()).collect();
    out.inputs = c.inputs.iter().filter(|i| i["what"] != "filters").cloned().collect(); // 조립 안 격자 입력(Rust 감시 틱은 조립을 기록값으로)
    (out, drop.len())
}

/// 값 없이 어디가 다른지(dict 키 경로만 · 첫 몇 곳).
pub fn diff_paths(a: &Value, b: &Value, path: &str, out: &mut Vec<String>) {
    if out.len() >= 5 {
        return;
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys().filter(|k| !x.contains_key(*k))) {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                match (x.get(k), y.get(k)) {
                    (Some(u), Some(v)) => diff_paths(u, v, &p, out),
                    _ => out.push(p),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                diff_paths(u, v, &format!("{path}[{i}]"), out);
            }
        }
        _ if !same(a, b) => out.push(if path.is_empty() { "(값)".into() } else { path.to_string() }),
        _ => {}
    }
}

/// 요청 두 개(정규화 글자)의 다른 곳: 메서드·경로가 다르면 «경로» · 같으면 다른 쿼리 키 이름만.
fn request_where(at: usize, want: Option<&String>, got: Option<&String>) -> String {
    let (Some(w), Some(g)) = (want, got) else {
        let p = want.or(got).map(|x| x.split('?').next().unwrap_or("").to_string()).unwrap_or_default();
        return format!("요청 #{at}: {} ({p})", if want.is_none() { "Rust 가 더 보냄" } else { "Rust 가 덜 보냄" });
    };
    let (wh, wq) = w.split_once('?').unwrap_or((w, ""));
    let (gh, gq) = g.split_once('?').unwrap_or((g, ""));
    if wh != gh {
        return format!("요청 #{at}: 경로 다름(원본 {wh} · Rust {gh})");
    }
    let kv = |q: &str| -> std::collections::BTreeMap<String, String> { q.split('&').filter_map(|x| x.split_once('=')).map(|(k, v)| (k.into(), v.into())).collect() };
    let (a, b) = (kv(wq), kv(gq));
    let keys: Vec<&String> = a.keys().chain(b.keys()).filter(|k| a.get(*k) != b.get(*k)).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    format!("요청 #{at} {wh}: 칸 {}", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join("·"))
}

/// 작업들을 차례로 재생한다(같은 무장 폴더·같은 칸을 이어 쓴다 -- 시작 기록이 무장 파일·칸을 주면 그 값으로 맞춘 뒤).
/// `symbol` = eth 주문 심볼(원본 MANUAL_EXEC_SYMBOL) · `dir` = 재생용 빈 폴더(원본 data/live 가 아니다).
/// `secret` = 기록을 만든 쪽의 서명 비밀값을 알 때만(가짜 키로 만든 정답지) -- 그때 exact 는 서명 글자까지. 실제 기록은 None(서명은 빼고 비교 · 진짜 비밀값을 재생에 넣지 않는다).
pub async fn replay(chs: &[Chain], symbol: &str, dir: &Path, exact: bool, secret: Option<&str>) -> Vec<Report> {
    let queue: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
    let t: Clock = Arc::new(Mutex::new(0.0));
    let (q2, t2) = (queue.clone(), t.clone());
    let srv = serve(Arc::new(move |_g: &Got| match q2.lock().unwrap().pop_front() {
        Some(w) => {
            if let Some(at) = w.get("t").and_then(Value::as_f64) {
                let mut c = t2.lock().unwrap(); // 파이썬이 이 요청을 보낸 시각까지 시계를 당긴다(실제 왕복 시간이 쌓여 조회 횟수가 갈리지 않게)
                *c = c.max(at);
            }
            act_of(&w)
        }
        None => Act::Send { status: 599, ctype: "application/json".into(), body: r#"{"msg":"기록 소진 -- 원본보다 요청을 더 낸다"}"#.into(), headers: vec![] },
    }));
    let ban = dir.join("ban_replay");
    let mk = |t: &Clock| {
        let mut b = Binance::new(&srv.base, if secret.is_some() { "test_key" } else { "replay_key" }, secret.unwrap_or("replay_secret"), std::time::Duration::from_secs(2), ban.clone());
        let tc = t.clone();
        b.clock = Arc::new(move || *tc.lock().unwrap());
        let ts = t.clone();
        b.virtual_sleep = Some(Arc::new(move |s| *ts.lock().unwrap() += s));
        b
    };
    let calls: Calls = Arc::new(Mutex::new(vec![]));
    let inputs: Inputs = Arc::new(Mutex::new(HashMap::new()));
    let last = Arc::new(Mutex::new(HashMap::new()));
    let desk = Arc::new(crate::runner::DeskCell::new(Desk::new(Watcher::new(dir, symbol))));
    let handlers: Arc<Mutex<VecDeque<Value>>> = Arc::new(Mutex::new(VecDeque::new()));
    let hdiffs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let inj = Arc::new(Inject { desk: desk.clone(), handlers: handlers.clone(), t: t.clone(), calls: calls.clone(), inputs: inputs.clone(), last: last.clone(), diffs: hdiffs.clone() });
    let mut ex = Vx { bx: mk(&t), t: t.clone(), inj: Some(inj) };
    let mut io = Io { bx: mk(&t), ex: Vx { bx: mk(&t), t: t.clone(), inj: None }, t: t.clone(), calls: calls.clone(), inputs: inputs.clone(), last: last.clone() };
    let mut quotes = Q { t: t.clone(), calls: calls.clone(), inputs: inputs.clone(), last: last.clone() };
    let fpath: PathBuf = crate::watch::bracket_path(dir);
    let mut out = vec![];
    for c0 in chs {
        let op = c0.start.get("op").and_then(Value::as_str).unwrap_or("?").to_string();
        if c0.end.is_null() {
            out.push(Report { chain: c0.id.clone(), requests: 0, diffs: vec![], where_: vec![], op, assembly: 0, truncated: true });
            continue;
        }
        let (c, assembly) = split_assembly(c0);
        let c = &c;
        let mut d = vec![];
        let mut wh: Vec<String> = vec![];
        let st = &c.start;
        *t.lock().unwrap() = st.get("t").and_then(Value::as_f64).unwrap_or(0.0);
        match st.get("file") {
            Some(Value::Null) => {
                let _ = std::fs::remove_file(&fpath);
            }
            Some(f) => std::fs::write(&fpath, serde_json::to_string(f).unwrap()).unwrap(),
            None => {}
        }
        // 작업 칸: "entry"(위 칸) | "wait:<코인>"(대기 칸) -- 화면 입력(핸들러)이 그 칸을 본다
        let sref = match st.get("slot_ref").and_then(Value::as_str) {
            Some(k) if k.starts_with("wait:") => SlotRef::Wait(k[5..].to_string()),
            _ => SlotRef::Entry,
        };
        if let Some(s) = st.get("slot").and_then(Value::as_object) {
            *desk.lock().await.slot_obj(&sref) = s.clone();
        }
        if let Some(ws) = st.get("waits").and_then(Value::as_object) {
            desk.lock().await.w.waits = ws.clone(); // 다른 대기 칸(원본 manual_wait_states 사본)
            if let Some(s) = st.get("slot").and_then(Value::as_object) {
                *desk.lock().await.slot_obj(&sref) = s.clone();
            }
        }
        *queue.lock().unwrap() = c.wire.iter().cloned().collect();
        srv.log.lock().unwrap().clear();
        calls.lock().unwrap().clear();
        {
            let mut m = inputs.lock().unwrap();
            m.clear();
            for i in &c.inputs {
                m.entry(i["what"].as_str().unwrap_or("").to_string()).or_default().push_back(i.get("value").cloned().unwrap_or(Value::Null));
            }
        }
        *handlers.lock().unwrap() = c.inputs.iter().filter(|i| i["what"] == "handler").cloned().collect();
        hdiffs.lock().unwrap().clear();
        last.lock().unwrap().clear();
        desk.lock().await.w.logs.clear();
        match st.get("op").and_then(Value::as_str) {
            Some("entry") => {
                let plan = st.get("plan").and_then(Value::as_object).cloned().unwrap_or_default();
                entry_then_bracket(&desk, &sref, &mut ex, &mut io, &mut quotes, &plan, st.get("limit").and_then(Value::as_bool).unwrap_or(false)).await;
            }
            Some("exit") => {
                // 청산 접수(원본 settle(run_exit | exit_then_rearm)) -- `ff` «팝업 청산 50% → 익절 다시 → 100%»
                let plan = st.get("plan").cloned().unwrap_or(Value::Null);
                let mut slot = std::mem::take(desk.lock().await.slot_obj(&sref));
                let tt = t.clone();
                let r = run_exit_into(&mut ex, &plan, &mut slot, || iso_utc(*tt.lock().unwrap())).await;
                if let Err(e) = r {
                    slot.insert("phase".into(), json!("error"));
                    slot.insert("error".into(), json!(format!("집행 오류 {e} — 거래소에서 직접 확인하세요")));
                } else if st.get("rearm").and_then(Value::as_bool).unwrap_or(false) {
                    let _ = desk.lock().await.w.partial_exit_rearm_tp(&mut io, plan.as_object().cloned().unwrap_or_default(), &mut slot).await;
                }
                *desk.lock().await.slot_obj(&sref) = slot;
            }
            Some("tick") => {
                if let Err(e) = desk.lock().await.w.bracket_tick(&mut io).await {
                    d.push(format!("감시 틱 예외 {e} (원본 {})", c.end.get("raised").unwrap_or(&Value::Null)));
                }
            }
            x => d.push(format!("모르는 op {x:?}")),
        }
        d.extend(hdiffs.lock().unwrap().drain(..));
        if !handlers.lock().unwrap().is_empty() {
            d.push(format!("화면 입력 {}건을 못 넣었다(그 시각까지 집행이 안 잤다)", handlers.lock().unwrap().len()));
        }
        let mut dk = desk.lock().await;
        // 요청 열
        let strip = |x: &str| if secret.is_some() { x.to_string() } else { x.split("&signature=").next().unwrap_or(x).to_string() }; // 비밀값을 모르면 서명은 비교 못 한다
        let got: Vec<String> = srv.log.lock().unwrap().iter().map(|g| normalize(&g.method, &strip(&g.target), exact)).collect();
        let want: Vec<String> = c.wire.iter().map(|w| normalize(w["method"].as_str().unwrap_or(""), &strip(w["target"].as_str().unwrap_or("")), exact)).collect();
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            d.push(format!("요청 열 다름 #{at}/{}: 원본 {} · Rust {}", want.len(), want.get(at).map_or("없음", String::as_str), got.get(at).map_or("없음", String::as_str)));
            wh.push(request_where(at, want.get(at), got.get(at)));
        }
        // 읽은 입력(종류·인자 · exact 면 시각까지)
        let rs: Vec<Value> = calls.lock().unwrap().iter().map(|x| if exact { x.clone() } else { Value::Array(x.as_array().unwrap()[1..].to_vec()) }).collect();
        let py: Vec<Value> = c.inputs.iter().map(|i| {
            let mut a: Vec<Value> = if exact { vec![i.get("t").cloned().unwrap_or(Value::Null)] } else { vec![] };
            a.push(i["what"].clone());
            a.extend(i.get("args").and_then(Value::as_array).cloned().unwrap_or_default());
            Value::Array(a)
        }).collect();
        if !same(&json!(rs), &json!(py)) {
            d.push(format!("읽은 입력: 원본 {} · Rust {}", json!(py), json!(rs)));
            let at = rs.iter().zip(&py).position(|(a, b)| !same(a, b)).unwrap_or(rs.len().min(py.len()));
            let kind = |v: Option<&Value>| v.and_then(|x| x.as_array()).and_then(|a| a.iter().find(|x| x.is_string())).map(crate::sign::py_str).unwrap_or_else(|| "없음".into());
            wh.push(format!("읽은 입력 #{at}: 원본 {} · Rust {}", kind(py.get(at)), kind(rs.get(at))));
        }
        // 끝 상태
        let tm = |v: &Value| if exact { v.clone() } else { drop_times(v) }; // 실기록(진짜 시계)은 시각 칸을 빼고 본다
        if let Some(es) = c.end.get("slot") {
            let rs_slot = dk.slot_obj(&sref).clone();
            let n0 = d.len();
            slot_diff(&tm(es), tm(&Value::Object(rs_slot)).as_object().unwrap(), &mut d);
            for x in &d[n0..] {
                wh.push(x.split(':').next().unwrap_or("칸").to_string()); // «칸 <이름>» -- 값은 ':' 뒤
            }
        }
        if let Some(ef) = c.end.get("file") {
            let file = std::fs::read_to_string(&fpath).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).unwrap_or(Value::Null);
            if !same(&tm(ef), &tm(&file)) {
                d.push(format!("무장 파일: 원본 {ef} · Rust {file}"));
                let mut ps = vec![];
                diff_paths(&tm(ef), &tm(&file), "", &mut ps);
                wh.push(format!("무장 파일 {}", ps.join("·")));
            }
        }
        if let Some(el) = c.end.get("limits") {
            let mut names: Vec<String> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json")).collect()).unwrap_or_default();
            names.sort();
            if json!(names) != *el {
                d.push(format!("복구 파일: 원본 {el} · Rust {names:?}"));
                wh.push("복구 파일".into());
            }
        }
        if let Some(lg) = c.end.get("logs") {
            if json!(dk.w.logs) != *lg {
                d.push(format!("로그: 원본 {lg} · Rust {:?}", dk.w.logs));
                wh.push("로그".into());
            }
        }
        if exact {
            if let Some(te) = c.end.get("t").and_then(Value::as_f64) {
                let now = *t.lock().unwrap();
                if now != te {
                    d.push(format!("끝 시각: 원본 {te} · Rust {now}"));
                }
            }
        }
        drop(dk);
        // 위에서 위치를 못 단 차이(감시 틱 예외·화면 입력·모르는 op·끝 시각 등) = 앞머리 이름만
        for x in &d {
            if !["요청 열", "읽은 입력", "칸 ", "무장 파일", "복구 파일", "로그"].iter().any(|p| x.starts_with(p)) {
                // 감시 틱 예외 · 화면 입력 · 모르는 op · 끝 시각 등 = 앞머리 이름만(숫자는 지운다)
                let head: String = x.split([':', '(']).next().unwrap_or("").chars().filter(|c| !c.is_ascii_digit()).collect();
                wh.push(head.trim().to_string());
            }
        }
        out.push(Report { chain: c.id.clone(), requests: got.len(), diffs: d, where_: wh, op, assembly, truncated: false });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_needs_feature_and_both_switches() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(a, _)| *a == k).map(|(_, v)| v.to_string());
        assert!(matches!(mode_from(&env(&[])), Ok(Mode::Shadow)), "기본 = 그림자");
        assert!(matches!(mode_from(&env(&[("RTORDER_MODE", "shadow")])), Ok(Mode::Shadow)));
        assert!(mode_from(&env(&[("RTORDER_MODE", "Live")])).is_err(), "모르는 값은 시작 안 함");
        assert!(mode_from(&env(&[("RTORDER_MODE", "live")])).is_err(), "확인 문구 없이는 안 켜진다");
        let both = mode_from(&env(&[("RTORDER_MODE", "live"), ("RTORDER_LIVE_CONFIRM", LIVE_CONFIRM)]));
        if cfg!(feature = "live") {
            assert!(matches!(both, Ok(Mode::Live(_))));
        } else {
            assert!(both.is_err(), "기본 빌드에는 실전송 자체가 없다");
        }
    }

    #[test]
    fn normalize_drops_clock_parts_only() {
        let a = "/fapi/v1/order?symbol=ETHUSDC&newClientOrderId=dbpgL1800000000123&recvWindow=5000&timestamp=1800000000123&signature=ab";
        let b = "/fapi/v1/order?symbol=ETHUSDC&newClientOrderId=dbpgL1800000000999&recvWindow=5000&timestamp=1800000000999&signature=cd";
        assert_eq!(normalize("POST", a, false), normalize("POST", b, false));
        assert_ne!(normalize("POST", a, true), normalize("POST", b, true));
        assert_ne!(normalize("POST", a, false), normalize("POST", &a.replace("dbpgL", "dbwtL"), false), "표식 앞머리는 남긴다");
        assert_ne!(normalize("POST", a, false), normalize("DELETE", a, false));
    }
}

/// 재생 보고서(사람이 읽는 글자): 작업마다 한 줄(요청 수 · 같음/차이) + 맨 끝 요약
/// «견준 작업 N · 요청 M · 어긋난 작업 K» + 첫 어긋남(작업 · 첫 차이 -- 요청 열이면 몇 번째·원본·Rust).
pub fn report_text(reports: &[Report]) -> String {
    report_with(reports, true)
}

/// `values=false`(실기록 출력): 작업마다 «chain · op · 요청 N · 같음/다름 + 어긋난 위치(값 없음)» · 금액·수량·키는 안 찍는다.
pub fn report_with(reports: &[Report], values: bool) -> String {
    let mut out = String::new();
    for r in reports {
        if r.truncated {
            out.push_str(&format!("{} | {} | 잘림(끝 줄 없음 -- 재시작) · 건너뜀\n", r.chain, r.op));
            continue;
        }
        let what = if r.diffs.is_empty() { "같음".to_string() } else if values { r.diffs.join(" | ") } else { format!("다름 -- {}", r.where_.join(" | ")) };
        let asm = if r.assembly > 0 { format!(" · 조립 구간 왕복 {}", r.assembly) } else { String::new() };
        out.push_str(&format!("{} | {} | 요청 {}{asm} | {what}\n", r.chain, r.op, r.requests));
    }
    let done: Vec<&Report> = reports.iter().filter(|r| !r.truncated).collect();
    let bad: Vec<&&Report> = done.iter().filter(|r| !r.diffs.is_empty()).collect();
    out.push_str(&format!("견준 작업 {} · 요청 {} · 어긋난 작업 {}", done.len(), done.iter().map(|r| r.requests).sum::<usize>(), bad.len()));
    let (cut, asm): (usize, usize) = (reports.iter().filter(|r| r.truncated).count(), done.iter().map(|r| r.assembly).sum());
    if cut > 0 {
        out.push_str(&format!(" · 잘림 {cut}"));
    }
    if asm > 0 {
        out.push_str(&format!(" · 조립 구간 왕복 {asm}(따로 셈)"));
    }
    out.push('\n');
    if let Some(b) = bad.first() {
        let first = if values {
            b.diffs.iter().find(|d| d.starts_with("요청 열")).unwrap_or(&b.diffs[0]).clone()
        } else {
            b.where_.iter().find(|d| d.starts_with("요청 #")).or(b.where_.first()).cloned().unwrap_or_default()
        };
        out.push_str(&format!("첫 어긋남: {} -- {first}\n", b.chain));
    }
    out
}

/// 실기록 파일 읽기(2026-11-21 서버 replay): `<파일>.1`(돌린 앞부분)이 있으면 먼저 · 깨진 줄(쓰는 중이던 반쪽 줄 등)은 세고 건너뛴다 ·
/// 시작 줄 없는 chain(접수 요청 자체)·chain "*"(깨움)은 세기만. 돌려주기 = (작업들, 깨진 줄, 접수만 chain, 깨움 줄).
pub fn chains_lenient(texts: &[String]) -> (Vec<Chain>, usize, usize, usize) {
    let (mut good, mut bad, mut wake) = (String::new(), 0, 0);
    for t in texts {
        for line in t.lines().filter(|l| !l.trim().is_empty()) {
            match serde_json::from_str::<Value>(line) {
                Ok(v) if v.get("chain").is_some() => {
                    if v["chain"] == "*" {
                        wake += 1;
                    }
                    good.push_str(line);
                    good.push('\n');
                }
                _ => bad += 1,
            }
        }
    }
    let ids: std::collections::BTreeSet<String> = good.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter_map(|v| v.get("chain").map(crate::sign::py_str)).filter(|c| c != "*").collect();
    let chs = chains(&good).unwrap_or_default();
    let api_only = ids.len() - chs.len();
    (chs, bad, api_only, wake)
}
