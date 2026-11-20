// 정답지 생성기(tools/gen_desk_fixture.py)의 가짜와 **같은 규칙**으로 답하는 바깥: 계좌 REST · 사용자 스트림 · 시세 · 계좌·화면 상태.
// 각본이 비면 원본 가짜처럼 예외(문자열) -- 실제 거래소·네트워크는 없다.
#![allow(dead_code)]
use rtorder::account::{AccountRest, Recv, RefreshIo, StreamIo};
use rtorder::assemble::{Desk, DeskIo, EntryArgs, Quotes};
use rtorder::intake::{Query, Task};
use rtorder::plan::Filters;
use rtorder::pyfmt::round_n;
use rtorder::sign::{err_value, http_error, P};
use rtorder::watch::Watcher;
use serde_json::{json, Map, Value};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::task::{Context, Poll, Waker};

pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("가짜 바깥은 기다리지 않는다"),
    }
}

pub const T0: f64 = 1_800_000_000.0;

/// 정답지 대조: 수는 값으로(정수/실수 표기 무시) · 키 집합은 정확히.
pub fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (x.as_f64() != y.as_f64()).then(|| format!("{path}: {x} != {y}")),
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => {
                        if let Some(d) = diff(p, q, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    (p, q) => return Some(format!("{path}.{k}: rust {p:?} / python {q:?}")),
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: 길이 {} != {} -- rust {a} / python {b}", x.len(), y.len()));
            }
            x.iter().zip(y).enumerate().find_map(|(i, (p, q))| diff(p, q, &format!("{path}[{i}]")))
        }
        _ => (a != b).then(|| format!("{path}: rust {a} / python {b}")),
    }
}
pub fn assert_same(rust: &Value, py: &Value, what: &str) {
    if let Some(d) = diff(rust, py, "") {
        panic!("{what}: {d}");
    }
}
pub fn sorted(mut v: Vec<Value>) -> Value {
    v.sort_by_key(|x| x.to_string());
    Value::Array(v)
}
fn raise_or(v: &Value) -> Result<Value, String> {
    match v.get("__raise__") {
        Some(r) => Err(format!("{}: {}", r[0].as_str().unwrap(), r[1].as_str().unwrap())),
        None => Ok(v.clone()),
    }
}
fn exc(r: &Value) -> String {
    format!("{}: {}", r[0].as_str().unwrap(), r[1].as_str().unwrap())
}

// ── 계좌 REST ──
pub struct FakeRest {
    pub routes: Value,
    pub ban: f64,
    pub keys: bool,
    pub t: f64,
    pub calls: Vec<Value>,
}
impl FakeRest {
    pub fn new(c: &Value) -> Self {
        FakeRest { routes: c["routes"].clone(), ban: c.get("ban").and_then(Value::as_f64).unwrap_or(0.0), keys: c.get("keys").and_then(Value::as_bool).unwrap_or(true),
                   t: c.get("t").and_then(Value::as_f64).unwrap_or(T0), calls: vec![] }
    }
}
impl AccountRest for FakeRest {
    fn has_keys(&self) -> bool {
        self.keys
    }
    fn now(&self) -> f64 {
        self.t
    }
    async fn clock_offset(&mut self) -> i64 {
        self.calls.push(json!(["/fapi/v1/time", {}]));
        0
    }
    async fn get(&mut self, path: &str, params: &[(&str, P)], _off: i64) -> Value {
        if self.ban > 0.0 {
            return err_value(format!("binance IP ban: {:.0}s left -- request not sent", self.ban));
        }
        let pm: Map<String, Value> = params.iter().map(|(k, v)| (k.to_string(), json!(v.py_str()))).collect();
        let sym = pm.get("symbol").and_then(Value::as_str).map(String::from);
        self.calls.push(json!([path, pm]));
        let key = match (&sym, path) {
            (Some(s), "/fapi/v1/userTrades" | "/fapi/v1/openOrders") => format!("{path}?{s}"),
            _ => path.into(),
        };
        let r = self.routes.get(&key).or(self.routes.get(path)).cloned().expect("각본 없음");
        if let Some(st) = r.get("__status__") {
            return err_value(http_error(st.as_u64().unwrap() as u16, &r["__body__"]));
        }
        r
    }
}

// ── 사용자 스트림 ──
pub struct FakeStream {
    pub t: f64,
    pub ev: Vec<Value>,
    pub script: Map<String, Value>,
    pub ban: f64,
    pub sleeps: u32,
    pub max_sleeps: u32,
    pub dirty_at: Vec<f64>,
    pub flag: bool,
    pub logs: Vec<String>,
}
impl FakeStream {
    pub fn new(c: &Value) -> Self {
        FakeStream { t: T0, ev: vec![], script: c.get("script").and_then(Value::as_object).cloned().unwrap_or_default(), ban: c.get("ban").and_then(Value::as_f64).unwrap_or(0.0),
                     sleeps: 0, max_sleeps: c.get("max_sleeps").and_then(Value::as_u64).unwrap_or(3) as u32,
                     dirty_at: c.get("dirty_at").and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_f64().unwrap()).collect()).unwrap_or_default(), flag: false, logs: vec![] }
    }
    fn sev(&mut self, mut v: Vec<Value>) {
        v.insert(0, json!(round_n(self.t, 6)));
        self.ev.push(Value::Array(v));
    }
    fn take(&mut self, name: &str) -> Option<Value> {
        let q = self.script.get_mut(name)?.as_array_mut()?;
        if q.is_empty() {
            return None;
        }
        let r = q.remove(0);
        self.t += r.get("dt").and_then(Value::as_f64).unwrap_or(0.0);
        if let Some(b) = r.get("set_ban").and_then(Value::as_f64) {
            self.ban = b;
        }
        Some(r)
    }
    async fn nap(&mut self, sec: f64) -> bool {
        self.sev(vec![json!("sleep"), json!(sec)]);
        self.sleeps += 1;
        if self.sleeps > self.max_sleeps {
            return false;
        }
        self.t += sec;
        self.take("sleep");
        true
    }
}
impl StreamIo for FakeStream {
    fn ban_remaining(&self) -> f64 {
        self.ban
    }
    fn mono(&self) -> f64 {
        self.t
    }
    fn wall(&self) -> f64 {
        self.t
    }
    async fn sleep(&mut self, sec: f64) -> bool {
        self.nap(sec).await
    }
    async fn listen_key(&mut self, method: &str) -> Result<String, String> {
        self.sev(vec![json!("listen_key"), json!(method)]);
        let r = self.take("listen_key").unwrap_or(json!({"raise": ["RuntimeError", "각본 없음"]}));
        match r.get("raise") {
            Some(x) => Err(exc(x)),
            None => Ok(r["ret"].as_str().unwrap().into()),
        }
    }
    async fn connect(&mut self, url: &str, heartbeat: f64) -> Result<(), String> {
        self.sev(vec![json!("connect"), json!(url), json!(heartbeat)]);
        let r = self.take("connect").unwrap_or(json!({}));
        match r.get("raise") {
            Some(x) => Err(exc(x)),
            None => Ok(()),
        }
    }
    async fn recv(&mut self, timeout: f64) -> Result<Recv, String> {
        let Some(r) = self.take("recv") else {
            self.sev(vec![json!("recv"), json!(timeout), json!("close")]);
            return Ok(Recv::Other);
        };
        let kind = r.get("kind").and_then(Value::as_str).unwrap_or("text").to_string();
        self.sev(vec![json!("recv"), json!(timeout), json!(kind)]);
        match kind.as_str() {
            "timeout" => Ok(Recv::Timeout),
            "raise" => Err(exc(&r["raise"])),
            "close" => Ok(Recv::Other),
            _ => Ok(Recv::Text(r["data"].as_str().unwrap().into())),
        }
    }
    fn wake(&mut self) {
        self.sev(vec![json!("wake")]);
    }
    fn dirty(&mut self) {
        self.sev(vec![json!("dirty")]);
    }
    fn tp_leg_filled(&mut self, s: &str, ps: &str, c: &str) -> Result<(), String> {
        self.sev(vec![json!("tp_leg_filled"), json!(s), json!(ps), json!(c)]);
        match self.take("tp_leg").and_then(|r| r.get("raise").cloned()) {
            Some(x) => Err(exc(&x)),
            None => Ok(()),
        }
    }
    fn log(&mut self, s: String) {
        self.logs.push(s);
    }
}
impl RefreshIo for FakeStream {
    async fn wait_dirty(&mut self) -> bool {
        self.sev(vec![json!("wait_dirty")]);
        let t = self.t;
        self.flag = self.flag || self.dirty_at.iter().any(|d| *d <= t);
        self.dirty_at.retain(|d| *d > t);
        if self.flag {
            return true;
        }
        if self.dirty_at.is_empty() {
            return false;
        }
        self.t = self.dirty_at.remove(0);
        self.flag = true;
        true
    }
    fn clear_dirty(&mut self) {
        self.sev(vec![json!("clear_dirty")]);
        self.flag = false;
    }
    async fn sleep(&mut self, sec: f64) -> bool {
        self.nap(sec).await
    }
    async fn refresh(&mut self) -> Result<(), String> {
        self.sev(vec![json!("produce")]);
        match self.take("produce").and_then(|r| r.get("raise").cloned()) {
            Some(x) => Err(exc(&x)),
            None => Ok(()),
        }
    }
    fn wake(&mut self) {
        self.sev(vec![json!("wake")]);
    }
    fn log(&mut self, s: String) {
        self.logs.push(s);
    }
}

// ── 시세 ──
pub struct FakeQuotes {
    pub books: Value,
    pub filters: Value,
    pub vol: Value,
    pub funding: Value,
    pub fees: Value,
    pub margin: Value,
    pub calls: Vec<Value>,
}
pub fn default_books() -> Value {
    json!({"ETHUSDC": {"bidPrice": "2500.11", "askPrice": "2500.37"}, "ETHUSDT": {"bidPrice": "2499.5", "askPrice": "2499.7"},
           "SOLUSDC": {"bidPrice": "150.123", "askPrice": "150.131"}, "SOLUSDT": {"bidPrice": "150.1", "askPrice": "150.12"},
           "XRPUSDC": {"bidPrice": "0.51234", "askPrice": "0.51241"}, "XRPUSDT": {"bidPrice": "0.5122", "askPrice": "0.5123"},
           "SOXLUSDT": {"bidPrice": "30.11", "askPrice": "30.13"}, "BTCUSDC": {"bidPrice": "90000.1", "askPrice": "90000.3"}, "BTCUSDT": {"bidPrice": "89990", "askPrice": "89990.2"}})
}
pub fn default_filters() -> Value {
    json!({"ETHUSDC": {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}, "ETHUSDT": {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0},
           "SOLUSDC": {"step": 0.01, "tick": 0.001, "min_qty": 0.01, "min_notional": 5.0}, "SOLUSDT": {"step": 0.01, "tick": 0.01, "min_qty": 0.01, "min_notional": 5.0},
           "XRPUSDC": {"step": 0.1, "tick": 0.00001, "min_qty": 0.1, "min_notional": 5.0},
           "SOXLUSDT": {"step": 1.0, "tick": 0.01, "min_qty": 1.0, "min_notional": 5.0}, "BTCUSDC": {"step": 0.001, "tick": 0.1, "min_qty": 0.001, "min_notional": 100.0}})
}
impl FakeQuotes {
    pub fn new(c: &Value) -> Self {
        let g = |k: &str| c.get(k).cloned().unwrap_or(json!({}));
        FakeQuotes { books: c.get("books").cloned().unwrap_or_else(default_books), filters: c.get("filters").cloned().unwrap_or_else(default_filters),
                     vol: g("vol"), funding: g("funding"), fees: g("fees"), margin: g("margin"), calls: vec![] }
    }
}
impl Quotes for FakeQuotes {
    async fn book(&mut self, symbol: &str) -> Result<Value, String> {
        self.calls.push(json!(["book", symbol]));
        match self.books.get(symbol) {
            None => Err(format!("RuntimeError: book_ticker_failed {symbol}")),
            Some(b) => raise_or(b),
        }
    }
    async fn filters(&mut self, symbol: &str) -> Result<Filters, String> {
        self.calls.push(json!(["filters", symbol]));
        match self.filters.get(symbol) {
            None => Err(format!("RuntimeError: symbol not in exchangeInfo: {symbol}")),
            Some(f) => Ok(serde_json::from_value(f.clone()).unwrap()),
        }
    }
    async fn vol(&mut self, symbol: &str) -> Option<f64> {
        self.calls.push(json!(["vol", symbol]));
        self.vol.get(symbol).and_then(Value::as_f64)
    }
    async fn funding(&mut self, symbol: &str) -> Option<f64> {
        self.calls.push(json!(["funding", symbol]));
        self.funding.get(symbol).and_then(Value::as_f64)
    }
    async fn fees(&mut self, symbol: &str) -> Value {
        self.calls.push(json!(["fees", symbol]));
        self.fees.get(symbol).cloned().unwrap_or(json!({"symbol": symbol, "maker_bp": 2.0, "taker_bp": 5.0, "source": "fallback"}))
    }
    async fn margin_type(&mut self, symbol: &str) -> Option<String> {
        self.calls.push(json!(["margin_type", symbol]));
        self.margin.get(symbol).and_then(Value::as_str).map(String::from)
    }
}

// ── 계좌·화면 상태 ──
pub struct FakeDeskIo {
    pub t: f64,
    pub exec_on: bool,
    pub account: Value,
    pub fresh: Option<Value>,
    pub sec: Value,
    pub situation: Value,
    pub script: Map<String, Value>,
    pub calls: Vec<Value>,
    pub dir: PathBuf,
    /// 스트림으로 유지하는 계좌(있으면 fresh 계좌 조회 대신 -- 받아 둔 값 시험)
    pub stream: Option<Value>,
}
impl FakeDeskIo {
    pub fn new(c: &Value, dir: PathBuf) -> Self {
        FakeDeskIo { t: c.get("t").and_then(Value::as_f64).unwrap_or(T0), exec_on: c.get("exec").and_then(Value::as_bool).unwrap_or(true), account: c["account"].clone(),
                     fresh: c.get("fresh_account").cloned(), sec: c.get("sec").cloned().unwrap_or(json!({})), situation: c.get("situation").cloned().unwrap_or(json!({})),
                     script: c.get("script").and_then(Value::as_object).cloned().unwrap_or_default(), calls: vec![], dir, stream: None }
    }
    fn respond(&mut self, name: &str) -> Result<Value, String> {
        let q = self.script.get_mut(name).and_then(Value::as_array_mut).filter(|q| !q.is_empty()).ok_or_else(|| format!("RuntimeError: 각본 없음: {name}"))?;
        let r = q.remove(0);
        if let Some(p) = r.get("patch_file").and_then(Value::as_object) {
            let path = rtorder::watch::bracket_path(&self.dir);
            let mut st: Map<String, Value> = std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            for (k, v) in p {
                st.insert(k.clone(), v.clone());
            }
            std::fs::write(&path, serde_json::to_string(&st).unwrap()).unwrap();
        }
        match r.get("raise") {
            Some(x) => Err(exc(x)),
            None => Ok(r.get("ret").cloned().unwrap_or(Value::Null)),
        }
    }
}
impl DeskIo for FakeDeskIo {
    fn now(&self) -> f64 {
        self.t
    }
    fn exec_enabled(&self) -> bool {
        self.exec_on
    }
    async fn account(&mut self, fresh: bool) -> Result<Value, String> {
        if fresh {
            if let Some(s) = &self.stream {
                return Ok(s.clone());
            }
        }
        self.calls.push(json!(["account", fresh]));
        let v = if fresh { self.fresh.clone().unwrap_or(self.account.clone()) } else { self.account.clone() };
        raise_or(&v)
    }
    fn sec_prices(&mut self, asset: &str) -> Option<Vec<(f64, f64)>> {
        self.sec.get(asset).and_then(Value::as_array).map(|a| a.iter().map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap())).collect())
    }
    fn vol_mult(&self, asset: &str) -> Value {
        self.situation.get(asset).and_then(|s| s.get("vol_mult")).cloned().unwrap_or(json!({}))
    }
    async fn clear_bracket(&mut self, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        self.calls.push(json!(["clear_bracket", symbol, pside, only_sl, only_tp]));
        self.respond("clear_bracket")
    }
    async fn place_stop(&mut self, symbol: &str, pside: &str, price: f64) -> Result<Value, String> {
        self.calls.push(json!(["place_stop", symbol, pside, price]));
        self.respond("place_stop")
    }
}

pub struct TmpDir(pub PathBuf);
impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
pub fn tmpdir() -> TmpDir {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let p = std::env::temp_dir().join(format!("rtorder_desk_{}_{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&p).unwrap();
    TmpDir(p)
}

pub fn desk_from(c: &Value) -> (Desk, TmpDir) {
    let d = tmpdir();
    if let Some(f) = c.get("file0") {
        std::fs::write(rtorder::watch::bracket_path(&d.0), serde_json::to_string(f).unwrap()).unwrap();
    }
    let mut desk = Desk::new(Watcher::new(&d.0, "ETHUSDC"));
    desk.override_x = c.get("override_x").and_then(Value::as_f64).unwrap_or(0.0);
    let s0 = c.get("slots0").cloned().unwrap_or(json!({}));
    if let Some(e) = s0.get("entry").and_then(Value::as_object) {
        desk.w.entry = e.clone();
    }
    for (k, v) in s0.get("side").and_then(Value::as_object).cloned().unwrap_or_default() {
        desk.w.side_exits.insert(k, v);
    }
    for (k, v) in s0.get("waits").and_then(Value::as_object).cloned().unwrap_or_default() {
        desk.w.waits.insert(k, v);
    }
    for k in s0.get("exiting").and_then(Value::as_array).cloned().unwrap_or_default() {
        desk.w.exiting.insert(k.as_str().unwrap().into());
    }
    for (k, done) in c.get("tasks").and_then(Value::as_object).cloned().unwrap_or_default() {
        if done == json!(false) {
            desk.running.insert(k);
        }
    }
    (desk, d)
}

pub fn read_file(dir: &Path) -> Value {
    std::fs::read_to_string(rtorder::watch::bracket_path(dir)).ok().map_or(Value::Null, |s| serde_json::from_str(&s).unwrap())
}
pub fn read_limits(dir: &Path) -> Value {
    let mut names: Vec<_> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().into_string().unwrap())
        .filter(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json")).collect();
    names.sort();
    Value::Object(names.into_iter().map(|n| { let v = serde_json::from_str(&std::fs::read_to_string(dir.join(&n)).unwrap()).unwrap(); (n, v) }).collect())
}

pub fn task_json(t: &Task) -> Value {
    match t {
        Task::Entry { plan, wait_sec, limit, .. } => json!({"fn": "entry_then_bracket", "plan": plan, "wait_sec": wait_sec, "limit": limit}),
        Task::Exit { plan, rearm, .. } => json!({"fn": if *rearm { "settle(exit_then_rearm)" } else { "settle(run_exit)" }, "plan": plan}),
    }
}

/// 사례 하나를 Rust 쪽에서 돌린다(정답지 키 이름 그대로 돌려준다).
pub fn dispatch<Q: Quotes>(desk: &mut Desk, q: &mut Q, io: &mut FakeDeskIo, c: &Value) -> Value {
    let op = c["op"].as_str().unwrap();
    match op {
        "entry" => {
            let a = &c["args"];
            let args = EntryArgs {
                side: a["side"].as_str().unwrap().into(), fraction: a.get("fraction").and_then(Value::as_f64).unwrap_or(1.0), want_lev: a.get("want_lev").and_then(Value::as_i64),
                asset: a.get("asset").and_then(Value::as_str).unwrap_or("eth").into(), fresh: a.get("fresh").and_then(Value::as_bool).unwrap_or(false),
                rule_l: a.get("rule_l").and_then(Value::as_str).map(String::from), lines: a.get("lines").and_then(Value::as_array).map(|l| (l[0].as_f64().unwrap(), l[1].as_f64().unwrap())),
                limit_usdt: a.get("limit_usdt").and_then(Value::as_f64), stop: a.get("stop").and_then(Value::as_bool).unwrap_or(false),
                skip_wait: a.get("skip_wait").and_then(Value::as_str).map(String::from),
            };
            match block_on(desk.assemble_entry_plan(q, io, &args)) {
                Ok((p, cap)) => json!({"plan": p, "cap": cap, "err": null}),
                Err((b, s)) => json!({"plan": null, "cap": {}, "err": [b, s]}),
            }
        }
        "exit" => {
            let a = &c["args"];
            let r = block_on(desk.assemble_exit_plan(q, io, a["side"].as_str().unwrap(), a.get("fraction").and_then(Value::as_f64).unwrap_or(1.0),
                a.get("fresh").and_then(Value::as_bool).unwrap_or(true), a.get("asset").and_then(Value::as_str).unwrap_or("eth"), a.get("market").and_then(Value::as_bool).unwrap_or(false),
                a.get("for_order").and_then(Value::as_bool).unwrap_or(false), a.get("only_symbol").and_then(Value::as_str)));
            match r {
                Ok(p) => json!({"plan": p, "err": null}),
                Err((b, s)) => json!({"plan": null, "err": [b, s]}),
            }
        }
        _ => {
            let qy: Query = c["query"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
            let r = match op {
                "api_manual_entry_submit" => block_on(desk.api_entry_submit(q, io, &qy)),
                "api_manual_entry_preview" => block_on(desk.api_entry_preview(q, io, &qy)),
                "api_manual_exit_submit" => block_on(desk.api_exit_submit(q, io, &qy)),
                "api_manual_exit_preview" => block_on(desk.api_exit_preview(q, io, &qy)),
                "api_manual_exit_all_submit" => block_on(desk.api_exit_all_submit(q, io, &qy)),
                "api_manual_limit_amend" => block_on(desk.api_limit_amend(q, io, &qy)),
                "api_manual_wait_action" => block_on(desk.api_wait_action(q, io, &qy)),
                "api_manual_entry_cancel" => desk.api_entry_cancel(io, &qy),
                "api_manual_bracket_disarm" => block_on(desk.api_bracket_disarm(io, &qy)),
                "api_manual_bracket_rearm" => block_on(desk.api_bracket_rearm(io, &qy)),
                x => panic!("모르는 op {x}"),
            };
            json!({"status": r.status, "body": r.body, "started": r.task.iter().map(task_json).collect::<Vec<_>>()})
        }
    }
}
