//! 작업 실행기(2026-11-21 여섯째 조각) -- 접수 문이 내놓은 `intake::Task` 를 돈다. 원본 `dashboard/orders.py` 의
//! `entry_then_bracket` · `limit_reprice` · `settle`(2026-11-21 스위칭 exit_then_switch·기다리기는 원본 PR 61 처럼 지웠다)(집행 머리 키·시계·레버리지는 `exec` 안 `Exchange::prepare`).
//! 상태 칸(`Obj` -- 화면·API 가 읽고 action·amend 를 쓴다) ↔ 집행 상태(`exec::ExecState`) 다리 = `slot_to_exec`·`exec_to_slot`.
//! 🔴W12 다리(2026-11-21): 원본은 같은 dict 를 집행 루프가 다시 읽는다 -- 여기는 도는 작업의 집행 상태를 `Desk::live`(칸 이름 → `Shared`)에 올리고,
//! 접수 핸들러가 칸을 읽기 전 `Desk::pull_live`(집행 → 칸) · 쓴 뒤 `Desk::push_live`(칸 → 집행). 실행기는 집행이 기다리는 동안 Desk 를 잠그지 않는다.
//! **어디에도 연결하지 않는다** -- 거래소는 `Exchange`, 손절·익절 함수·시계는 `WatchIo`, 시세는 `Quotes` 뒤. 시험 = tests/runner_e2e.rs · flow_fixture(원본 정답지).

use crate::assemble::{Desk, Quotes};
use crate::exec::{self, Action, Amend, Common, Exchange, ExecState, LimitOrder, OrderRef, Reply, Shared};
use crate::intake::{SlotRef, Task};
use crate::plan::bracket_key;
use crate::sign::{py_float, py_str, truthy};
use crate::watch::{exec_plan_of, iso_utc, phase_name, stop_touched, Obj, WatchIo};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// 상태 칸 → 집행 상태: 화면이 쓴 action(cancel·market·peg)·amend(가격·수량)와 감시 진입 선(plan.limit.usdt)을 넘긴다(W12).
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
    if let Some(l) = slot.get("plan").and_then(|p| p.get("limit")).and_then(|l| l.get("usdt")).and_then(|v| py_float(v).ok()) {
        g.stop_level = Some(l); // 원본 stop_touched 는 매번 plan["limit"]["usdt"] 를 읽는다
    }
}

/// 집행 상태 → 상태 칸(원본 state dict 의 칸 이름 그대로). 집행이 소비한 amend 는 칸에서도 지운다 · 집행이 지운 action(PEG 전환 = 원본 pop)도.
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
    if let Some(t) = e.pegged {
        slot.insert("pegged".into(), json!(iso_utc(t)));
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
        slot.insert("triggered".into(), e.triggered.map_or(Value::Null, |t| json!(iso_utc(t))));
    }
    if e.amend.is_none() {
        slot.remove("amend");
    }
    if e.action.is_none() {
        slot.remove("action"); // 원본: PEG 전환 때 state.pop("action")
    }
}

/// 접수 문과 실행기가 같이 쥐는 Desk(2026-11-21 W12 다리). 실행기는 집행이 기다리는 동안(3초 조회·1초 잠·취소 확인) 잠그지 않는다 --
/// 그 사이 접수 핸들러(취소·지금 시장가·PEG·선 옮김)가 잠그고 칸에 쓰면 `Desk::push_live` 가 도는 작업의 집행 상태에 바로 넣는다.
pub type DeskCell = tokio::sync::Mutex<Desk>;

/// 칸 이름(도는 작업 표 `Desk::live` 의 키).
pub fn slot_key(s: &SlotRef) -> String {
    match s {
        SlotRef::Entry => "entry".into(),
        SlotRef::Side(a) => format!("side:{a}"),
        SlotRef::Wait(k) => format!("wait:{k}"),
    }
}

/// 원본 entry_then_bracket 의 `plan` 은 칸의 plan 과 **같은 dict** -- 선 옮김(limit-amend)이 제자리에서 고친다. 칸에 plan 이 있으면 그것.
fn cur_plan(d: &mut Desk, sref: &SlotRef, fallback: &Obj) -> Obj {
    d.slot_obj(sref).get("plan").and_then(Value::as_object).cloned().unwrap_or_else(|| fallback.clone())
}

/// 집행 중 첫 체결 순간(지정가 대기의 on_fill) 무장 · 지정가 자동 재조정(reprice) 을 하려고 바깥을 같이 쥐는 거래소 감싸개.
struct RunEx<'a, 'b, E: Exchange, I: WatchIo, Q: Quotes> {
    ex: &'a mut E,
    io: &'a RefCell<&'b mut I>,
    desk: &'a DeskCell,
    sref: &'a SlotRef,
    quotes: &'a mut Q,
    plan: &'a Obj,
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
        let mut d = self.desk.lock().await;
        let plan = cur_plan(&mut d, self.sref, self.plan);
        let mut slot = std::mem::take(d.slot_obj(self.sref)); // 무장이 칸(bracket)에 쓴다 -- 잠금 안이라 접수 문은 빈 칸을 못 본다
        let r: Result<(), String> = async {
            if truthy(plan.get("add_keep")) {
                return d.w.keep_rearm(&mut **io, &plan, &mut slot).await;
            }
            if let Some(p2) = d.w.plan_from_current(&plan)? {
                d.w.arm_bracket(&mut **io, &p2, &mut slot).await?;
            }
            Ok(())
        }.await;
        if let Err(e) = r {
            slot.insert("bracket".into(), json!({"placed": false, "reason": e}));
        }
        *d.slot_obj(self.sref) = slot;
    }
    /// 원본 limit_reprice(2026-10-31 자동 재조정): 지금 basis 로 다시 잰 목표 주문가 · basis 를 못 읽어 1.0 이면 None(검증 S1) · 호가를 넘으면 None(P23).
    /// 선(limit.usdt)은 칸의 plan 에서 -- 선 옮김이 바꿨으면 새 선.
    async fn reprice(&mut self) -> Result<Option<f64>, String> {
        let (plan, asset) = {
            let mut d = self.desk.lock().await;
            let p = cur_plan(&mut d, self.sref, self.plan);
            let a = d.w.exec_asset_of(p.get("symbol")).unwrap_or("eth");
            (p, a)
        };
        let Some(lim) = plan.get("limit").and_then(|l| l.get("usdt")).filter(|v| truthy(Some(v))).and_then(|v| py_float(v).ok()) else { return Ok(None) };
        let symbol = plan.get("symbol").map(py_str).unwrap_or_default();
        let book = self.quotes.book(&symbol).await?;
        let tick = { let t = self.quotes.filters(&symbol).await?.tick; if t != 0.0 { t } else { 0.01 } };
        let long = plan.get("positionSide").and_then(Value::as_str) == Some("LONG");
        let b = crate::assemble::basis_of(self.quotes, &book, asset).await;
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

/// 원본 entry_then_bracket(plan, state, limit): 진입 → 체결이 있으면(부분 포함, 모든 종료 경로) 익절을 걸고 감시를 무장한다(B3).
/// `state` = `desk` 의 `sref` 칸(접수 문이 쓴 값이 도는 집행에 닿게 -- W12). 키 확인·시계·레버리지는 집행 머리(`Exchange::prepare`) · 결과는 칸 leverage 에(P16).
pub async fn entry_then_bracket<E: Exchange, I: WatchIo, Q: Quotes>(desk: &DeskCell, sref: &SlotRef, ex: &mut E, io: &mut I, quotes: &mut Q, plan0: &Obj, limit: bool) {
    let key = slot_key(sref);
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    let (ep, plan, asset) = {
        let mut d = desk.lock().await;
        let now = io.now();
        let plan = cur_plan(&mut d, sref, plan0);
        if limit {
            let _ = d.w.limit_save_at(now, &plan);
        }
        let ep = match exec_plan_of(&Value::Object(plan.clone())) {
            Ok(p) => p,
            Err(e) => return settle(Err(e), d.slot_obj(sref), now),
        };
        {
            let mut g = s.lock().unwrap();
            g.stop_level = plan.get("limit").and_then(|l| l.get("usdt")).and_then(|v| py_float(v).ok());
        }
        slot_to_exec(d.slot_obj(sref), &s);
        d.live.insert(key.clone(), s.clone());
        let asset = d.w.exec_asset_of(Some(&json!(ep.symbol))).unwrap_or("eth");
        (ep, plan, asset)
    };
    let stop = limit && truthy(plan.get("stop"));
    {
        let cell = RefCell::new(&mut *io);
        let mut rx = RunEx { ex, io: &cell, desk, sref, quotes, plan: &plan, first_fill: limit };
        if stop {
            let ps = ep.position_side.clone();
            let (io2, s2) = (rx.io, s.clone());
            // 닿음 선 = 칸 plan 의 limit.usdt(원본 stop_touched 가 매번 plan 에서 읽는다 -- 선 옮김이 push_live 로 바로 바꾼다)
            exec::run_entry_stop(&mut rx, &ep, &s, |since| {
                let lvl = s2.lock().unwrap().stop_level.unwrap_or(0.0);
                stop_touched(lvl, &ps, io2.borrow_mut().sec_prices(asset).ok().flatten().as_deref(), since)
            }, exec::LIMIT_MAX_SEC).await;
        } else if limit {
            exec::run_entry_limit(&mut rx, &ep, &s, exec::LIMIT_MAX_SEC, true, true).await;
        } else {
            exec::run_entry(&mut rx, &ep, &s).await;
        }
    }
    let mut d = desk.lock().await;
    d.live.remove(&key);
    let lines = std::mem::take(&mut s.lock().unwrap().logs);
    for l in lines {
        d.w.log(l); // ponytail: 원본은 그 순간 print -- 여기는 작업 끝에 모아서(그 사이 다른 로그와 순서가 갈릴 수 있다)
    }
    let plan = cur_plan(&mut d, sref, plan0);
    let mut slot = std::mem::take(d.slot_obj(sref));
    exec_to_slot(&s.lock().unwrap(), &mut slot);
    slot.insert("done_at".into(), json!(iso_utc(io.now()))); // 원본 now_iso() = 집행이 끝난 시각
    let filled = slot.get("filled").and_then(Value::as_f64).unwrap_or(0.0);
    if filled > 0.0 {
        let bkey = bracket_key(&ep.symbol, &ep.position_side);
        let cur = d.w.load().get(&bkey).cloned().unwrap_or(json!({}));
        let r: Result<(), String> = async {
            if truthy(plan.get("add_keep")) {
                d.w.keep_rearm(io, &plan, &mut slot).await
            } else if truthy(plan.get("bracket").and_then(|b| b.get("lines"))) {
                // 선 주문 = 지금 저장된 선으로 합친 전량에 다시 -- 대기 중 바뀐 선을 지키면서 TP 크기는 맞춘다
                match d.w.plan_from_current(&plan)? {
                    Some(p2) => d.w.arm_bracket(io, &p2, &mut slot).await,
                    None => Ok(()),
                }
            } else if limit && ["mod", "sl_off", "tp_off", "fired"].iter().any(|k| truthy(cur.get(*k))) {
                Ok(()) // 검증 #2: 사용자가 옮긴 선·손절 해제·청산 진행을 옛 계획으로 덮지 않는다
            } else {
                d.w.arm_bracket(io, &plan, &mut slot).await
            }
        }.await;
        settle(r, &mut slot, io.now());
    }
    // 정상 종료에서만 복구 파일을 지운다 -- error = 거래소에 주문이 살아 있을 수 있다(검증 #1 · W4·R8)
    if limit && slot.get("phase").and_then(Value::as_str) != Some("error") {
        let wid = plan.get("wid").and_then(Value::as_i64).unwrap_or(1);
        let _ = std::fs::remove_file(crate::watch::limit_path(&d.w.dir, &ep.symbol, wid));
    }
    *d.slot_obj(sref) = slot;
}

/// 접수 문의 `Task` 하나를 끝까지 돈다(청산은 감시가 쓰는 run_exit_into · 부분 청산은 exit_then_rearm). 끝나면 `Desk::task_done`.
/// 진입 = 칸을 Desk 에 둔 채(접수 문이 «도는 중»을 본다) · 청산 = 칸 사본으로 돌리고 끝에 되돌린다(청산은 화면 입력을 안 받는다 -- C6).
pub async fn run_task<E: Exchange, I: WatchIo, Q: Quotes>(desk: &DeskCell, ex: &mut E, io: &mut I, quotes: &mut Q, task: Task) {
    match task {
        Task::Entry { key, slot, plan, limit } => {
            let po = plan.as_object().cloned().unwrap_or_default();
            entry_then_bracket(desk, &slot, ex, io, quotes, &po, limit).await;
            desk.lock().await.task_done(&key);
        }
        Task::Exit { key, slot, plan, rearm } => {
            let mut st = desk.lock().await.slot_obj(&slot).clone();
            let r = crate::watch::run_exit_into(ex, &plan, &mut st, || iso_utc(io.now())).await;
            settle(r, &mut st, io.now());
            let mut d = desk.lock().await;
            if rearm {
                let _ = d.w.partial_exit_rearm_tp(io, plan.as_object().cloned().unwrap_or_default(), &mut st).await; // 부분 청산 뒤 남은 수량으로 익절(B30 · C2)
            }
            *d.slot_obj(&slot) = st;
            d.task_done(&key);
        }
    }
}
