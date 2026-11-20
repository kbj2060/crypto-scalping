// 각본대로 답하는 감시 바깥(WatchIo) -- 정답지 생성기(tools/gen_watch_fixture.py)의 가짜와 **같은 규칙**:
// 응답 하나 = {"ret" | "raise": [형, 문구], "dt": 시계 전진, "patch": 저장본 고치기, "lock": 주문 잠금, "exec": 실행 스위치, "slot": run_exit 결과 칸}.
// 호출은 [시각, 이름, 인자…] 로 적는다(시각 = 응답의 dt 를 더하기 전). watch_fixture.rs·watch_invariants.rs 가 include! 로 같이 쓴다.
use rtorder::plan;
use rtorder::watch::*;
use serde_json::{json, Map, Value};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::task::{Context, Poll, Waker};

#[allow(dead_code)]
fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("가짜 바깥은 기다리지 않는다"),
    }
}

const IO_CALL_CAP: usize = 5_000;

pub struct ScriptIo {
    pub dir: PathBuf,
    pub t: f64,
    pub exec_on: bool,
    pub lock: bool,
    pub account: Value,
    pub bars: Value,
    pub sec: Value,
    pub script: Map<String, Value>,
    pub calls: Vec<Value>,
    pub bars_patch: Vec<Value>,
    pub wakes: Vec<f64>,
    pub naps: u32,
    pub max_naps: u32,
    pub per_nap: Map<String, Value>,
    /// run_exit 에서 영원히 멈춘다(프로세스가 청산 도중 죽은 것) -- 재시작 시험.
    pub hang_run_exit: bool,
}

impl ScriptIo {
    pub fn new(dir: &Path) -> ScriptIo {
        ScriptIo { dir: dir.into(), t: 1_800_000_000.0, exec_on: true, lock: false, account: Value::Null, bars: json!({}), sec: json!({}),
                   script: Map::new(), calls: vec![], bars_patch: vec![], wakes: vec![], naps: 0, max_naps: 0, per_nap: Map::new(), hang_run_exit: false }
    }
    fn note(&mut self, mut v: Vec<Value>) {
        v.insert(0, json!(self.t));
        self.calls.push(Value::Array(v));
        assert!(self.calls.len() <= IO_CALL_CAP, "호출 상한 {IO_CALL_CAP} 초과 -- 감시가 안 끝난다");
    }
    pub fn apply_patch(&self, patch: &[Value]) {
        let p = bracket_path(&self.dir);
        let mut st: Map<String, Value> = std::fs::read_to_string(&p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        for item in patch {
            let k = item["key"].as_str().unwrap().to_string();
            if item.get("remove").is_some_and(|v| v == &json!(true)) {
                st.remove(&k);
            } else if let Some(v) = item.get("value") {
                st.insert(k, v.clone());
            } else {
                let cur = st.entry(k).or_insert_with(|| json!({})).as_object_mut().unwrap();
                for (f, v) in item.get("set").and_then(Value::as_object).cloned().unwrap_or_default() {
                    cur.insert(f, v);
                }
                for f in item.get("del").and_then(Value::as_array).cloned().unwrap_or_default() {
                    cur.remove(f.as_str().unwrap());
                }
            }
        }
        std::fs::write(&p, serde_json::to_string(&st).unwrap()).unwrap();
    }
    fn respond(&mut self, name: &str, slot: Option<&mut Obj>) -> Result<Value, String> {
        let q = self.script.get_mut(name).and_then(Value::as_array_mut).filter(|a| !a.is_empty()).ok_or(format!("RuntimeError: 각본 없음: {name}"))?;
        let r = q.remove(0);
        self.t += r.get("dt").and_then(Value::as_f64).unwrap_or(0.0);
        if let Some(p) = r.get("patch").and_then(Value::as_array) {
            self.apply_patch(p);
        }
        if let Some(l) = r.get("lock").and_then(Value::as_bool) {
            self.lock = l;
        }
        if let Some(e) = r.get("exec").and_then(Value::as_bool) {
            self.exec_on = e;
        }
        if let (Some(s), Some(u)) = (slot, r.get("slot").and_then(Value::as_object)) {
            for (k, v) in u {
                s.insert(k.clone(), v.clone());
            }
        }
        if let Some(e) = r.get("raise") {
            return Err(format!("{}: {}", e[0].as_str().unwrap(), e[1].as_str().unwrap()));
        }
        Ok(r)
    }
    fn raised(v: &Value) -> Result<(), String> {
        match v.get("raise") {
            Some(e) if v.is_object() => Err(format!("{}: {}", e[0].as_str().unwrap(), e[1].as_str().unwrap())),
            _ => Ok(()),
        }
    }
}

impl WatchIo for ScriptIo {
    fn now(&self) -> f64 {
        self.t
    }
    fn exec_enabled(&self) -> bool {
        self.exec_on
    }
    fn order_locked(&self) -> bool {
        self.lock
    }
    async fn nap(&mut self, sec: f64) -> bool {
        self.note(vec![json!("nap"), json!(sec)]);
        self.naps += 1;
        if self.naps > self.max_naps {
            return false;
        }
        let nxt = self.wakes.iter().copied().filter(|w| self.t < *w && *w < self.t + sec).fold(f64::INFINITY, f64::min);
        self.t = if nxt.is_finite() { nxt } else { self.t + sec };
        if let Some(m) = self.per_nap.get(&self.naps.to_string()).and_then(Value::as_object).cloned() {
            for (k, v) in m {
                match k.as_str() {
                    "account" => self.account = v,
                    "sec" => self.sec = v,
                    "bars" => self.bars = v,
                    "lock" => self.lock = v.as_bool().unwrap(),
                    "exec_on" => self.exec_on = v.as_bool().unwrap(),
                    _ => panic!("모르는 per_nap 칸 {k}"),
                }
            }
        }
        true
    }
    async fn account(&mut self) -> Result<Value, String> {
        self.note(vec![json!("account")]);
        Self::raised(&self.account)?;
        Ok(self.account.clone())
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        self.note(vec![json!("produce_account")]);
        Ok(self.respond("produce_account", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
    async fn bars(&mut self, asset: &str) -> Result<Vec<Value>, String> {
        self.note(vec![json!("bars"), json!(asset)]);
        if !self.bars_patch.is_empty() {
            let p = std::mem::take(&mut self.bars_patch);
            self.apply_patch(&p);
        }
        let v = self.bars.get(asset).cloned().unwrap_or(Value::Null);
        Self::raised(&v)?;
        Ok(v.as_array().cloned().unwrap_or_default())
    }
    fn sec_prices(&mut self, asset: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        self.note(vec![json!("sec"), json!(asset)]);
        Ok(self.sec.get(asset).and_then(Value::as_array).map(|a| a.iter().map(|x| (x[0].as_f64().unwrap(), x[1].as_f64().unwrap())).collect()))
    }
    async fn clear_bracket(&mut self, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        self.note(vec![json!("clear_bracket"), json!(symbol), json!(pside), json!(only_sl), json!(only_tp)]);
        Ok(self.respond("clear_bracket", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
    async fn exit_plan(&mut self, pside: &str, fraction: f64, asset: &str, only_symbol: &str) -> Result<Value, Value> {
        self.note(vec![json!("exit_plan"), json!(pside), json!(fraction), json!(true), json!(asset), json!(false), json!(only_symbol)]);
        let r = self.respond("exit_plan", None).expect("청산 계획 조립은 예외를 내지 않는다(원본이 전부 잡는다)");
        match r.get("plan") {
            Some(p) => Ok(p.clone()),
            None => Err(r["err"].clone()),
        }
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let file = std::fs::read_to_string(bracket_path(&self.dir)).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).unwrap_or(Value::Null);
        self.note(vec![json!("run_exit"), plan.clone(), Value::Object(slot.clone()), file]); // 청산을 시작하는 순간의 저장본(B14·B20)
        if self.hang_run_exit {
            std::future::pending::<()>().await;
        }
        self.respond("run_exit", Some(slot)).map(|_| ())
    }
    async fn place_bracket(&mut self, spec: &Value, symbol: &str, pside: &str, strict: bool) -> Result<Value, String> {
        self.note(vec![json!("place_bracket"), spec.clone(), json!(symbol), json!(pside), json!(strict)]);
        Ok(self.respond("place_bracket", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
    async fn place_tp_ladder(&mut self, legs: &Value, symbol: &str, pside: &str) -> Result<Value, String> {
        self.note(vec![json!("place_tp_ladder"), legs.clone(), json!(symbol), json!(pside)]);
        Ok(self.respond("place_tp_ladder", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
    async fn place_add_order(&mut self, symbol: &str, pside: &str, price: f64, qty: f64) -> Result<Value, String> {
        self.note(vec![json!("place_add_order"), json!(symbol), json!(pside), json!(price), json!(qty)]);
        Ok(self.respond("place_add_order", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
    async fn cancel_wait_orphans(&mut self, symbols: &[String]) -> Result<Value, String> {
        self.note(vec![json!("cancel_wait_orphans"), json!(symbols)]);
        Ok(self.respond("cancel_wait_orphans", None)?.get("ret").cloned().unwrap_or(Value::Null))
    }
}

/// 사례 하나를 준비한다(임시 폴더 · 첫 저장본 · 대기 복구 파일 · 상태 칸).
#[allow(dead_code)]
fn setup(s: &Value, eth: &str) -> (Watcher, ScriptIo) {
    let dir = std::env::temp_dir().join(format!("rtorder_watch_{}_{}", std::process::id(), s["name"].as_str().unwrap()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    if let Some(raw) = s.get("file0_raw").and_then(Value::as_str) {
        std::fs::write(bracket_path(&dir), raw).unwrap();
    } else if let Some(v) = s.get("file0") {
        std::fs::write(bracket_path(&dir), serde_json::to_string(v).unwrap()).unwrap();
    }
    for (n, body) in s.get("limit_files").and_then(Value::as_object).cloned().unwrap_or_default() {
        std::fs::write(dir.join(n), serde_json::to_string(&body).unwrap()).unwrap();
    }
    let mut w = Watcher::new(&dir, eth);
    let s0 = s.get("slots0").cloned().unwrap_or(json!({}));
    if let Some(e) = s0.get("entry").and_then(Value::as_object) {
        w.entry = e.clone();
    }
    for (k, v) in s0.get("side").and_then(Value::as_object).cloned().unwrap_or_default() {
        w.side_exits.insert(k, v);
    }
    for (k, v) in s0.get("waits").and_then(Value::as_object).cloned().unwrap_or_default() {
        w.waits.insert(k, v);
    }
    let mut io = ScriptIo::new(&dir);
    io.script = s.get("script").and_then(Value::as_object).cloned().unwrap_or_default();
    io.bars_patch = s.get("bars_patch").and_then(Value::as_array).cloned().unwrap_or_default();
    io.t = s["ticks"][0].get("t").and_then(Value::as_f64).unwrap_or(1_800_000_000.0);
    (w, io)
}

/// 틱(또는 다른 동작) 하나를 돌리고 정답지와 같은 모양으로 결과를 돌려준다.
#[allow(dead_code)]
fn run_op(w: &mut Watcher, io: &mut ScriptIo, tk: &Value) -> Value {
    if let Some(t) = tk.get("t").and_then(Value::as_f64) {
        io.t = t;
    }
    let pick = |k: &str| tk.get(k).cloned();
    if let Some(v) = pick("exec") { io.exec_on = v.as_bool().unwrap(); }
    if let Some(v) = pick("lock") { io.lock = v.as_bool().unwrap(); }
    if let Some(v) = pick("account") { io.account = v; }
    if let Some(v) = pick("bars") { io.bars = v; }
    if let Some(v) = pick("sec") { io.sec = v; }
    io.wakes = tk.get("wakes").and_then(Value::as_array).map(|a| a.iter().map(|x| x.as_f64().unwrap()).collect()).unwrap_or_default();
    io.max_naps = tk.get("naps").and_then(Value::as_u64).unwrap_or(0) as u32;
    io.per_nap = tk.get("per_nap").and_then(Value::as_object).cloned().unwrap_or_default();
    io.calls.clear();
    w.logs.clear();
    let mut state: Option<Obj> = tk.get("state").and_then(Value::as_object).cloned();
    let plan: Obj = tk.get("plan").and_then(Value::as_object).cloned().unwrap_or_default();
    let mut ret = Value::Null;
    let op = tk.get("op").and_then(Value::as_str).unwrap_or("tick");
    let r: Result<(), String> = block_on(async {
        match op {
            "tick" => w.bracket_tick(io).await,
            "watch" => {
                w.watch_brackets(io, WATCH_PERIOD_SEC).await;
                Ok(())
            }
            "partial_exit_rearm_tp" => w.partial_exit_rearm_tp(io, plan.clone(), state.as_mut().unwrap()).await,
            "keep_rearm" => w.keep_rearm(io, &plan, state.as_mut().unwrap()).await,
            "exit_then_rearm" => w.exit_then_rearm(io, &plan, state.as_mut().unwrap()).await,
            "arm_bracket" => w.arm_bracket(io, &plan, state.as_mut().unwrap(), tk.get("add").and_then(Value::as_bool).unwrap_or(true)).await,
            "plan_from_current" => w.plan_from_current(&plan).map(|p| ret = p.map_or(Value::Null, Value::Object)),
            "limit_save" => w.limit_save(io, &plan),
            "wait_cleanup" => w.wait_cleanup(io).await,
            o => panic!("모르는 op {o}"),
        }
    });
    let raw = std::fs::read_to_string(bracket_path(&io.dir)).ok();
    let file = match &raw {
        None => Value::Null,
        Some(s) => serde_json::from_str(s).unwrap_or(json!("<unparseable>")),
    };
    let mut limits = Map::new();
    let mut names: Vec<String> = std::fs::read_dir(&io.dir).unwrap().filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json")).collect();
    names.sort();
    for n in names {
        limits.insert(n.clone(), serde_json::from_str(&std::fs::read_to_string(io.dir.join(&n)).unwrap()).unwrap());
    }
    json!({"calls": io.calls, "file": file, "limits": limits,
           "slots": {"entry": w.entry, "side": w.side_exits, "waits": w.waits},
           "exiting": w.exiting.iter().collect::<Vec<_>>(), "logs": w.logs, "raised": r.err(), "ret": ret,
           "state": state.map_or(Value::Null, Value::Object), "t_end": io.t})
}
