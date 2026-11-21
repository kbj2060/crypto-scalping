//! 작업 실행기(2026-11-21 여섯째 조각) -- 접수 문이 내놓은 `intake::Task` 를 돈다. 원본 `dashboard/orders.py` 의
//! `entry_then_bracket` · `exit_then_switch` · `limit_reprice` · `settle`(집행 머리 키·시계·레버리지는 `exec` 안 `Exchange::prepare`).
//! 상태 칸(`Obj` -- 화면·API 가 읽고 action·amend 를 쓴다) ↔ 집행 상태(`exec::ExecState`) 다리 = `slot_to_exec`·`exec_to_slot`.
//! **어디에도 연결하지 않는다** -- 거래소는 `Exchange`, 손절·익절 함수·시계는 `WatchIo`, 시세는 `Quotes` 뒤. 시험 = tests/runner_e2e.rs(가짜 거래소).

use crate::assemble::{Desk, DeskIo, EntryArgs, Quotes};
use crate::exec::{self, Action, Amend, Common, Exchange, ExecState, LimitOrder, OrderRef, Reply, Shared};
use crate::intake::{SlotRef, Task};
use crate::plan::bracket_key;
use crate::sign::{py_float, py_str, truthy};
use crate::watch::{exec_plan_of, iso_utc, phase_name, stop_touched, Obj, WatchIo};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// 상태 칸 → 집행 상태: 화면이 쓴 action(cancel·market·peg)·amend(가격·수량)만 넘긴다(W12 -- 집행 루프는 ≤3초마다 읽는다).
pub fn slot_to_exec(slot: &Obj, s: &Shared) {
    let mut g = s.lock().unwrap();
    if g.action.is_none() {
        g.action = match slot.get("action").and_then(Value::as_str) {
            Some("cancel") => Some(Action::Cancel),
            Some("market") => Some(Action::Market),
            Some("peg") => Some(Action::Peg),
            _ => None,
        };
        if g.action.is_some() {
            g.action_reason = slot.get("action_reason").and_then(Value::as_str).map(String::from);
        }
    }
    if g.amend.is_none() {
        if let Some(a) = slot.get("amend").filter(|v| truthy(Some(v))) {
            g.amend = Some(Amend { price: a.get("price").and_then(|v| py_float(v).ok()).unwrap_or(0.0), qty: a.get("qty").and_then(|v| py_float(v).ok()), auto: false });
        }
    }
}

/// 집행 상태 → 상태 칸(원본 state dict 의 칸 이름 그대로). 집행이 소비한 amend 는 칸에서도 지운다.
pub fn exec_to_slot(e: &ExecState, slot: &mut Obj) {
    let f = |x: f64| json!(x);
    for (k, v) in [("phase", json!(phase_name(e.phase))), ("filled", f(e.filled)), ("quantity", f(e.quantity)), ("limit_price", f(e.limit_price)), ("repegs", json!(e.repegs)),
                   ("deadline_sec", f(e.deadline_sec)), ("order_id", json!(e.order_id)), ("taker_qty", f(e.taker_qty))] {
        slot.insert(k.into(), v);
    }
    if !e.mode.is_empty() {
        slot.insert("mode".into(), json!(e.mode));
    }
    for (k, v) in [("client_order_id", e.client_order_id.clone().map(Value::String)), ("error", e.error.clone().map(Value::String)),
                   ("cancel_reason", e.cancel_reason.clone().map(Value::String)), ("taker_order_id", e.taker_order_id.map(|x| json!(x))),
                   ("taker_reason", e.taker_reason.map(|x| json!(x))), ("market_reason", e.market_reason.clone().map(Value::String))] {
        if let Some(v) = v {
            slot.insert(k.into(), v);
        }
    }
    if e.pegged {
        slot.insert("pegged".into(), json!(true));
    }
    if e.mode == "limit" {
        slot.insert("amends".into(), json!(e.amends)); // 원본: 지정가·감시 대기만 amends=0 으로 시작 · reprices 는 재조정이 한 번이라도 있을 때만
    }
    if e.reprices > 0 {
        slot.insert("reprices".into(), json!(e.reprices));
    }
    if let Some(w) = e.wait_until {
        slot.insert("wait_until".into(), json!(iso_utc(w)));
    }
    if let Some(l) = &e.leverage {
        slot.insert("leverage".into(), l.clone());
    }
    if e.stop {
        slot.insert("stop".into(), json!(true));
        slot.insert("triggered".into(), json!(e.triggered));
    }
    if e.amend.is_none() {
        slot.remove("amend");
    }
}

/// 집행 중 첫 체결 순간(기다리기·지정가 대기의 on_fill) 무장 · 지정가 자동 재조정(reprice) 을 하려고 바깥을 같이 쥐는 거래소 감싸개.
struct RunEx<'a, 'b, E: Exchange, I: WatchIo, Q: Quotes> {
    ex: &'a mut E,
    io: &'a RefCell<&'b mut I>,
    w: &'a mut crate::watch::Watcher,
    desk_basis: &'a Desk,
    quotes: &'a mut Q,
    plan: &'a Obj,
    slot: &'a mut Obj,
    first_fill: bool,
}

impl<E: Exchange, I: WatchIo, Q: Quotes> Exchange for RunEx<'_, '_, E, I, Q> {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        self.ex.place_limit_gtx(o).await
    }
    async fn place_market(&mut self, c: &Common<'_>, q: f64) -> Reply {
        self.ex.place_market(c, q).await
    }
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.ex.get_order(c, r).await
    }
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.ex.cancel_order(c, r).await
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.ex.book(s).await
    }
    fn now(&self) -> f64 {
        self.ex.now()
    }
    fn wall_ms(&self) -> i64 {
        self.ex.wall_ms()
    }
    async fn nap(&mut self, sec: f64) {
        self.ex.nap(sec).await
    }
    async fn sleep(&mut self, sec: f64) {
        self.ex.sleep(sec).await
    }
    async fn prepare(&mut self, symbol: &str, leverage: Option<i64>) -> Result<Option<Value>, String> {
        self.ex.prepare(symbol, leverage).await
    }
    /// 원본 first_fill: 추가 진입이면 keep_rearm, 아니면 **지금 저장된 선**(plan_from_current)으로 무장(물타기 없이) -- 예외는 결과 칸에 남기고 대기는 계속(W10).
    async fn on_fill(&mut self) {
        if !self.first_fill {
            return;
        }
        let mut io = self.io.borrow_mut();
        let r: Result<(), String> = async {
            if truthy(self.plan.get("add_keep")) {
                return self.w.keep_rearm(&mut **io, self.plan, self.slot).await;
            }
            if let Some(p2) = self.w.plan_from_current(self.plan)? {
                self.w.arm_bracket(&mut **io, &p2, self.slot, false).await?;
            }
            Ok(())
        }.await;
        if let Err(e) = r {
            self.slot.insert("bracket".into(), json!({"placed": false, "reason": e}));
        }
    }
    /// 원본 limit_reprice(2026-10-31 자동 재조정): 지금 basis 로 다시 잰 목표 주문가 · basis 를 못 읽어 1.0 이면 None(검증 S1) · 호가를 넘으면 None(P23).
    async fn reprice(&mut self) -> Result<Option<f64>, String> {
        let Some(lim) = self.plan.get("limit").and_then(|l| l.get("usdt")).filter(|v| truthy(Some(v))).and_then(|v| py_float(v).ok()) else { return Ok(None) };
        let symbol = self.plan.get("symbol").map(py_str).unwrap_or_default();
        let book = self.quotes.book(&symbol).await?;
        let tick = { let t = self.quotes.filters(&symbol).await?.tick; if t != 0.0 { t } else { 0.01 } };
        let long = self.plan.get("positionSide").and_then(Value::as_str) == Some("LONG");
        let asset = self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth");
        let b = self.desk_basis.basis_for(self.quotes, &book, asset).await;
        if b == 1.0 {
            return Ok(None);
        }
        let raw = lim * b;
        let px = crate::pyfmt::round_n(if long { (raw / tick + 1e-9).floor() } else { (raw / tick - 1e-9).ceil() } * tick, 8);
        let k = |n: &str| book.get(n).ok_or_else(|| format!("KeyError: '{n}'")).and_then(py_float);
        Ok(if (long && px >= k("askPrice")?) || (!long && px <= k("bidPrice")?) { None } else { Some(px) })
    }
}

/// 원본 settle: 집행 예외가 새면 phase 가 working 에 영영 묶인다(10-16 감사, R3).
fn settle(r: Result<(), String>, slot: &mut Obj, now: f64) {
    if let Err(e) = r {
        slot.insert("phase".into(), json!("error"));
        slot.insert("error".into(), json!(format!("집행 오류 {e} — 거래소에서 직접 확인하세요")));
        slot.insert("done_at".into(), json!(iso_utc(now)));
    }
}

/// 원본 entry_then_bracket(plan, state, wait_sec, limit): 진입 → 체결이 있으면(부분 포함, 모든 종료 경로) 익절을 걸고 감시를 무장한다(B3).
/// 키 확인·시계·레버리지는 집행 머리(`Exchange::prepare` -- 원본 run_entry·run_entry_wait·run_entry_limit 안) · 결과는 칸 leverage 에(P16).
#[allow(clippy::too_many_arguments)]
pub async fn entry_then_bracket<E: Exchange, I: WatchIo, Q: Quotes>(desk: &mut Desk, ex: &mut E, io: &mut I, quotes: &mut Q, plan: &Obj, slot: &mut Obj,
                                                                    wait_sec: Option<f64>, limit: bool) {
    let now = io.now();
    if limit {
        let _ = desk.w.limit_save_at(now, plan);
    }
    let ep = match exec_plan_of(&Value::Object(plan.clone())) {
        Ok(p) => p,
        Err(e) => return settle(Err(e), slot, now),
    };
    let stop = limit && truthy(plan.get("stop"));
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    slot_to_exec(slot, &s);
    let mut w = std::mem::replace(&mut desk.w, crate::watch::Watcher::new(std::env::temp_dir(), "ETHUSDC"));
    {
        let cell = RefCell::new(&mut *io);
        let mut rx = RunEx { ex, io: &cell, w: &mut w, desk_basis: desk, quotes, plan, slot, first_fill: wait_sec.is_some() || limit };
        if stop {
            let lvl = plan.get("limit").and_then(|l| l.get("usdt")).and_then(|v| py_float(v).ok()).unwrap_or(0.0);
            let asset = rx.w.exec_asset_of(Some(&json!(ep.symbol))).unwrap_or("eth");
            let ps = ep.position_side.clone();
            let io2 = rx.io;
            exec::run_entry_stop(&mut rx, &ep, &s, |since| stop_touched(lvl, &ps, io2.borrow_mut().sec_prices(asset).ok().flatten().as_deref(), since), exec::LIMIT_MAX_SEC).await;
        } else if limit {
            exec::run_entry_limit(&mut rx, &ep, &s, exec::LIMIT_MAX_SEC, true, true).await;
        } else if let Some(ws) = wait_sec {
            exec::run_entry_wait(&mut rx, &ep, &s, ws, true).await;
        } else {
            exec::run_entry(&mut rx, &ep, &s).await;
        }
    }
    desk.w = w;
    exec_to_slot(&s.lock().unwrap(), slot);
    slot.insert("done_at".into(), json!(iso_utc(io.now()))); // 원본 now_iso() = 집행이 끝난 시각
    let filled = slot.get("filled").and_then(Value::as_f64).unwrap_or(0.0);
    if filled > 0.0 {
        let key = bracket_key(&ep.symbol, &ep.position_side);
        let cur = desk.w.load().get(&key).cloned().unwrap_or(json!({}));
        let r: Result<(), String> = async {
            if truthy(plan.get("add_keep")) {
                desk.w.keep_rearm(io, plan, slot).await
            } else if truthy(plan.get("bracket").and_then(|b| b.get("lines"))) {
                // 선 주문 = 지금 저장된 선으로 합친 전량에 다시 -- 대기 중 바뀐 선을 지키면서 TP 크기는 맞춘다
                match desk.w.plan_from_current(plan)? {
                    Some(p2) => desk.w.arm_bracket(io, &p2, slot, true).await,
                    None => Ok(()),
                }
            } else if limit && ["mod", "sl_off", "tp_off", "fired"].iter().any(|k| truthy(cur.get(*k))) {
                Ok(()) // 검증 #2: 사용자가 옮긴 선·손절 해제·청산 진행을 옛 계획으로 덮지 않는다
            } else {
                desk.w.arm_bracket(io, plan, slot, true).await
            }
        }.await;
        settle(r, slot, io.now());
    }
    // 정상 종료에서만 복구 파일을 지운다 -- error = 거래소에 주문이 살아 있을 수 있다(검증 #1 · W4·R8)
    if limit && slot.get("phase").and_then(Value::as_str) != Some("error") {
        let wid = plan.get("wid").and_then(Value::as_i64).unwrap_or(1);
        let _ = std::fs::remove_file(crate::watch::limit_path(&desk.w.dir, &ep.symbol, wid));
    }
}


/// 원본 exit_then_switch: 전량 청산이 **다 닫힌 뒤에만**(filled_* · 체결 ≥ 수량×0.999) 반대 진입. 칸을 await 없이 진입 상태로 바꾸고,
/// 진입 버튼과 같은 assemble_entry_plan(fresh) · 막히면 주문 없이 사유 · 반대 진입은 MARKET(M2 · 2026-10-10 «즉시») → entry_then_bracket.
#[allow(clippy::too_many_arguments)]
pub async fn exit_then_switch<E: Exchange, I: WatchIo, D: DeskIo, Q: Quotes>(desk: &mut Desk, ex: &mut E, io: &mut I, dio: &mut D, quotes: &mut Q,
                                                                            plan: &Value, sw: &Value, asset: &str) {
    let mut slot = std::mem::take(&mut desk.w.entry);
    let r = crate::watch::run_exit_into(ex, plan, &mut slot, || iso_utc(io.now())).await;
    settle(r, &mut slot, io.now());
    let mut info = slot.get("switch").cloned().unwrap_or(json!({}));
    let done = matches!(slot.get("phase").and_then(Value::as_str), Some("filled_maker" | "filled_taker"))
        && slot.get("filled").and_then(Value::as_f64).unwrap_or(0.0) >= slot.get("quantity").and_then(Value::as_f64).unwrap_or(0.0) * 0.999;
    if !done {
        info["phase"] = json!("aborted");
        info["error"] = json!(format!("청산이 다 닫히지 않아({}) 반대 진입을 내지 않았습니다", slot.get("phase").map_or("None".into(), py_str)));
        slot.insert("switch".into(), info);
        desk.w.entry = slot;
        return;
    }
    info["phase"] = json!("entry");
    info["exit"] = json!({"phase": slot.get("phase"), "filled": slot.get("filled"), "quantity": slot.get("quantity"), "done_at": slot.get("done_at")});
    let to = sw["to"].as_str().unwrap_or("").to_string();
    // 🔴await 없이 바로 진입 상태로 -- 사이에 다른 주문이 already_working 검사를 빠져나가지 않게
    let mut st = Obj::new();
    for (k, v) in [("phase", json!("submitting")), ("side", json!(to)), ("asset", json!(asset)), ("switch", info.clone()), ("started_at", json!(iso_utc(io.now())))] {
        st.insert(k.into(), v);
    }
    desk.w.entry = st.clone();
    let why: Option<String> = if !dio.exec_enabled() {
        Some("DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다".into())
    } else {
        let a = EntryArgs { side: to.clone(), fraction: sw["frac"].as_f64().unwrap_or(1.0), want_lev: sw["lev"].as_i64(), asset: asset.into(), fresh: true,
                            rule_l: sw["rule"].as_str().map(String::from), ..EntryArgs::new(&to, asset) };
        match desk.assemble_entry_plan(quotes, dio, &a).await {
            Err((b, _)) => Some(b.get("detail").filter(|v| truthy(Some(v))).or(b.get("error")).map_or("None".into(), py_str)),
            Ok((p, _)) if truthy(p.get("blocked")) => Some(py_str(&p["blocked"])),
            Ok((mut p, _)) => {
                p["type"] = json!("MARKET"); // 2026-10-10 스위칭 = 즉시 체결(M2)
                info["plan"] = json!({"quantity": p.get("quantity"), "notional_usdt": p.get("notional_usdt"), "price": p.get("price"), "target_leverage": p.get("target_leverage")});
                st.insert("plan".into(), p.clone());
                st.insert("switch".into(), info.clone());
                let po = p.as_object().cloned().unwrap_or_default();
                entry_then_bracket(desk, ex, io, quotes, &po, &mut st, None, false).await;
                let ok = st.get("phase").and_then(Value::as_str).is_some_and(|ph| ph.starts_with("filled"));
                info["phase"] = json!(if ok { "done" } else { "aborted" });
                info["error"] = if ok { Value::Null } else { st.get("error").filter(|v| truthy(Some(v))).or(st.get("phase")).cloned().unwrap_or(Value::Null) };
                st.insert("switch".into(), info);
                desk.w.entry = st;
                return;
            }
        }
    };
    if let Some(why) = why {
        st.insert("phase".into(), json!("error"));
        st.insert("error".into(), json!(format!("스위칭 반대 진입 막힘 — {why}")));
        st.insert("done_at".into(), json!(iso_utc(io.now())));
        info["phase"] = json!("aborted");
        info["error"] = json!(why);
        st.insert("switch".into(), info);
        desk.w.entry = st;
    }
}

/// 접수 문의 `Task` 하나를 끝까지 돈다(청산은 감시가 쓰는 run_exit_into · 부분 청산은 exit_then_rearm). 끝나면 `Desk::task_done`.
pub async fn run_task<E: Exchange, I: WatchIo, D: DeskIo, Q: Quotes>(desk: &mut Desk, ex: &mut E, io: &mut I, dio: &mut D, quotes: &mut Q, task: Task) {
    match task {
        Task::Entry { key, slot, plan, wait_sec, limit } => {
            let po = plan.as_object().cloned().unwrap_or_default();
            let mut st = take_slot(desk, &slot);
            entry_then_bracket(desk, ex, io, quotes, &po, &mut st, wait_sec, limit).await;
            put_slot(desk, &slot, st);
            desk.task_done(&key);
        }
        Task::Exit { key, slot, plan, rearm } => {
            let mut st = take_slot(desk, &slot);
            let r = crate::watch::run_exit_into(ex, &plan, &mut st, || iso_utc(io.now())).await;
            settle(r, &mut st, io.now());
            if rearm {
                let _ = desk.w.partial_exit_rearm_tp(io, plan.as_object().cloned().unwrap_or_default(), &mut st).await; // 부분 청산 뒤 남은 수량으로 익절(B30 · C2)
            }
            put_slot(desk, &slot, st);
            desk.task_done(&key);
        }
        Task::Switch { key, plan, sw, asset } => {
            exit_then_switch(desk, ex, io, dio, quotes, &plan, &sw, &asset).await;
            desk.task_done(&key);
        }
    }
}

fn take_slot(desk: &mut Desk, s: &SlotRef) -> Obj {
    match s {
        SlotRef::Entry => std::mem::take(&mut desk.w.entry),
        SlotRef::Side(a) => desk.w.side_exits.get_mut(a).and_then(Value::as_object_mut).map(std::mem::take).unwrap_or_default(),
        SlotRef::Wait(k) => desk.w.waits.get_mut(k).and_then(Value::as_object_mut).map(std::mem::take).unwrap_or_default(),
    }
}
fn put_slot(desk: &mut Desk, s: &SlotRef, v: Obj) {
    match s {
        SlotRef::Entry => desk.w.entry = v,
        SlotRef::Side(a) => {
            desk.w.side_exits.insert(a.clone(), Value::Object(v));
        }
        SlotRef::Wait(k) => {
            desk.w.waits.insert(k.clone(), Value::Object(v));
        }
    }
}
