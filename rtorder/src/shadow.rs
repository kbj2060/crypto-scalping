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
    Ok(order.into_iter().map(|id| by.remove(&id).unwrap()).collect())
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
    async fn vol(&mut self, _: &str) -> Option<f64> {
        None
    }
    async fn funding(&mut self, _: &str) -> Option<f64> {
        None
    }
    async fn fees(&mut self, _: &str) -> Value {
        Value::Null
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
    pub diffs: Vec<String>,
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
    let mut desk = Desk::new(Watcher::new(dir, symbol));
    let mut ex = Vx { bx: mk(&t), t: t.clone() };
    let mut io = Io { bx: mk(&t), ex: Vx { bx: mk(&t), t: t.clone() }, t: t.clone(), calls: calls.clone(), inputs: inputs.clone(), last: last.clone() };
    let mut quotes = Q { t: t.clone(), calls: calls.clone(), inputs: inputs.clone(), last: last.clone() };
    let fpath: PathBuf = crate::watch::bracket_path(dir);
    let mut out = vec![];
    for c in chs {
        let mut d = vec![];
        let st = &c.start;
        *t.lock().unwrap() = st.get("t").and_then(Value::as_f64).unwrap_or(0.0);
        match st.get("file") {
            Some(Value::Null) => {
                let _ = std::fs::remove_file(&fpath);
            }
            Some(f) => std::fs::write(&fpath, serde_json::to_string(f).unwrap()).unwrap(),
            None => {}
        }
        if let Some(s) = st.get("slot").and_then(Value::as_object) {
            desk.w.entry = s.clone();
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
        last.lock().unwrap().clear();
        desk.w.logs.clear();
        match st.get("op").and_then(Value::as_str) {
            Some("entry") => {
                let plan = st.get("plan").and_then(Value::as_object).cloned().unwrap_or_default();
                let mut slot = std::mem::take(&mut desk.w.entry);
                entry_then_bracket(&mut desk, &mut ex, &mut io, &mut quotes, &plan, &mut slot, st.get("limit").and_then(Value::as_bool).unwrap_or(false)).await;
                desk.w.entry = slot;
            }
            Some("exit") => {
                // 청산 접수(원본 settle(run_exit | exit_then_rearm)) -- `ff` «팝업 청산 50% → 익절 다시 → 100%»
                let plan = st.get("plan").cloned().unwrap_or(Value::Null);
                let mut slot = std::mem::take(&mut desk.w.entry);
                let tt = t.clone();
                let r = run_exit_into(&mut ex, &plan, &mut slot, || iso_utc(*tt.lock().unwrap())).await;
                if let Err(e) = r {
                    slot.insert("phase".into(), json!("error"));
                    slot.insert("error".into(), json!(format!("집행 오류 {e} — 거래소에서 직접 확인하세요")));
                } else if st.get("rearm").and_then(Value::as_bool).unwrap_or(false) {
                    let _ = desk.w.partial_exit_rearm_tp(&mut io, plan.as_object().cloned().unwrap_or_default(), &mut slot).await;
                }
                desk.w.entry = slot;
            }
            Some("tick") => {
                if let Err(e) = desk.w.bracket_tick(&mut io).await {
                    d.push(format!("감시 틱 예외 {e} (원본 {})", c.end.get("raised").unwrap_or(&Value::Null)));
                }
            }
            x => d.push(format!("모르는 op {x:?}")),
        }
        // 요청 열
        let strip = |x: &str| if secret.is_some() { x.to_string() } else { x.split("&signature=").next().unwrap_or(x).to_string() }; // 비밀값을 모르면 서명은 비교 못 한다
        let got: Vec<String> = srv.log.lock().unwrap().iter().map(|g| normalize(&g.method, &strip(&g.target), exact)).collect();
        let want: Vec<String> = c.wire.iter().map(|w| normalize(w["method"].as_str().unwrap_or(""), &strip(w["target"].as_str().unwrap_or("")), exact)).collect();
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
            d.push(format!("요청 열 다름 #{at}/{}: 원본 {} · Rust {}", want.len(), want.get(at).map_or("없음", String::as_str), got.get(at).map_or("없음", String::as_str)));
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
        }
        // 끝 상태
        if let Some(es) = c.end.get("slot") {
            slot_diff(es, &desk.w.entry, &mut d);
        }
        if let Some(ef) = c.end.get("file") {
            let file = std::fs::read_to_string(&fpath).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).unwrap_or(Value::Null);
            if !same(ef, &file) {
                d.push(format!("무장 파일: 원본 {ef} · Rust {file}"));
            }
        }
        if let Some(el) = c.end.get("limits") {
            let mut names: Vec<String> = std::fs::read_dir(dir).map(|r| r.filter_map(|e| e.ok()?.file_name().into_string().ok())
                .filter(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json")).collect()).unwrap_or_default();
            names.sort();
            if json!(names) != *el {
                d.push(format!("복구 파일: 원본 {el} · Rust {names:?}"));
            }
        }
        if let Some(lg) = c.end.get("logs") {
            if json!(desk.w.logs) != *lg {
                d.push(format!("로그: 원본 {lg} · Rust {:?}", desk.w.logs));
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
        out.push(Report { chain: c.id.clone(), requests: got.len(), diffs: d });
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
