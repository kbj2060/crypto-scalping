//! 접수 문(2026-11-21 넷째 조각) -- `dashboard/orders.py` 의 주문 API 핸들러를 **HTTP 와 떼어** 옮긴 것:
//! 입력 = 파싱된 질의(`Query`) · 출력 = 상태코드 + 본문 + «시작할 작업»(`Task` -- 실제로 돌리는 것은 `run_task`/다음 조각의 웹 서버).
//! 옮긴 것: entry-preview/submit · exit-preview/submit · exit-all · limit-amend · wait-action · entry-cancel · bracket-disarm/rearm · query_* 검사 · 대기 칸 배정.
//! 안 옮긴 것(다음): 익절 on/off(`api_manual_bracket_tp`) · 분할 조각(`…_tp_legs`) · 선 추가(`…_add`) · 선 수정(`…_modify`) · 상태 조회(`…_status`) · 접수 기록(G18, server.py).
//!
//! 지키는 것(값·동작 그대로): POST 전용(G1 -- 웹 서버가 `POST_ONLY` 표로) · confirm=1(G2) · 실행 스위치 매 요청(G3) · 진입 칸 하나(G5) · 코인별 청산 칸(G6) ·
//! 조립 뒤 칸 다시 보기(G7) · 주문 잠금 안/밖(G8 -- `LOCKED` 표, 잠금 자체는 `watch::busy_or`) · 대기 칸 규칙(G10–G12) · 대기 중 스위칭 409(G13) · 코인 표(G14) ·
//! 조용히 고치지 않는 400(G15) · 스위칭은 전량+반대(G16).
//!
//! 시장가 분기(M8 «즉시» 청산 · M11 스위칭 · M13 대기 «지금 시장가» · M6·M7 조립이 낸 시장가 계획)는 원본과 같이 접수한다
//! (2026-11-21 사용자 결정 «원본 동작 유지» -- 넷째 조각의 409 market_undecided 거절은 지웠다). 스위칭은 `Task::Switch`(exit_then_switch).
use crate::assemble::{Desk, DeskIo, EntryArgs, Quotes, MANUAL_RULES_ENABLED};
use crate::exec::{WAIT_CHOICES_MIN};
use crate::plan::{self, bracket_key};
use crate::sign::{py_float, py_str, truthy};
use crate::watch::{busy, iso_utc, Obj, SlotId, Watcher, WAIT_SLOTS_MAX};
use serde_json::{json, Value};
use std::collections::HashMap;

pub type Query = HashMap<String, String>;

/// 주문 잠금 안(true)/밖(false) -- 원본 server.py 라우트 등록(규칙 문서 G8). 웹 서버가 이 표대로 `busy_or` 로 감싼다.
pub const LOCKED: [(&str, bool); 10] = [
    ("entry_submit", true), ("exit_submit", true), ("exit_all_submit", true), ("limit_amend", true),
    ("wait_action", false), ("entry_cancel", false), ("bracket_disarm", false), ("bracket_rearm", false),
    ("entry_preview", false), ("exit_preview", false),
];
/// 주문을 바꾸는 요청 = POST 만(G1 -- GET 이면 링크 미리보기·프리페치·주소창 재방문이 그대로 주문). 미리보기만 GET.
pub const POST_ONLY: [&str; 8] = ["entry_submit", "exit_submit", "exit_all_submit", "limit_amend", "wait_action", "entry_cancel", "bracket_disarm", "bracket_rearm"];

#[derive(Debug, Clone, PartialEq)]
pub enum SlotRef {
    Entry,
    Side(String),
    Wait(String),
}

/// 접수가 시작하라고 내놓는 작업(원본 `asyncio.create_task(...)` 하나).
#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    /// entry_then_bracket(plan, state, wait_sec, limit)
    Entry { key: String, slot: SlotRef, plan: Value, wait_sec: Option<f64>, limit: bool },
    /// settle(run_exit | exit_then_rearm(부분 청산))
    Exit { key: String, slot: SlotRef, plan: Value, rearm: bool },
    /// exit_then_switch(plan, state, sw, asset) -- 스위칭(전량 청산이 다 닫힌 뒤 반대 진입 · 두 다리 다 시장가 M11)
    Switch { key: String, plan: Value, sw: Value, asset: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resp {
    pub status: u16,
    pub body: Value,
    pub task: Vec<Task>,
}

fn resp(status: u16, body: Value) -> Resp {
    Resp { status, body, task: vec![] }
}
fn bad(status: u16, err: &str, detail: &str) -> Resp {
    resp(status, json!({"ok": false, "error": err, "detail": detail}))
}
fn exec_off() -> Resp {
    bad(403, "exec_disabled", "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다")
}
/// 파이썬 int(float(x)) 가 OverflowError 로 새는 곳(lev=inf) -- aiohttp 는 500.
fn server_error(e: &str) -> Resp {
    resp(500, json!({"error": "500 Internal Server Error", "exception": e}))
}

fn g<'a>(q: &'a Query, k: &str) -> Option<&'a str> {
    q.get(k).map(String::as_str)
}
/// 파이썬 float(str). ponytail: «1_000» 같은 밑줄 표기는 파이썬만 받는다(화면은 보내지 않는다).
fn pf(s: &str) -> Option<f64> {
    py_float(&json!(s)).ok()
}

pub fn query_asset(d: &Desk, q: &Query) -> Option<String> {
    let a = g(q, "asset").filter(|s| !s.is_empty()).unwrap_or("eth").to_lowercase();
    d.exec_symbol(&a).map(|_| a)
}
/// 범위를 벗어나면 조용히 자르지 않고 None(모델값으로). Err = 파이썬에서 새던 OverflowError(inf).
pub fn query_leverage(q: &Query, key: &str) -> Result<Option<i64>, String> {
    let Some(raw) = g(q, key).filter(|s| !s.is_empty()) else { return Ok(None) };
    let Some(x) = pf(raw) else { return Ok(None) };
    if x.is_nan() {
        return Ok(None);
    }
    if x.is_infinite() {
        return Err("OverflowError: cannot convert float infinity to integer".into());
    }
    let v = x.trunc() as i64;
    Ok((1..=crate::assemble::EXCHANGE_MAX_LEVERAGE).contains(&v).then_some(v))
}
/// "c"(자동 2×) · "h"(보유 1.5×) · 없음 None · 그 밖 "bad"(직접 모드로 조용히 떨어지지 않는다).
pub fn query_rule(q: &Query, key: &str) -> Option<String> {
    let raw = g(q, key).filter(|s| !s.is_empty())?;
    Some(if (raw == "c" || raw == "h") && MANUAL_RULES_ENABLED { raw.into() } else { "bad".into() })
}
pub enum Lines {
    None,
    Bad,
    Some(f64, f64),
}
pub fn query_lines(q: &Query) -> Lines {
    let (tp, sl) = (g(q, "tp").filter(|s| !s.is_empty()), g(q, "sl").filter(|s| !s.is_empty()));
    if tp.is_none() && sl.is_none() {
        return Lines::None;
    }
    match (tp.and_then(pf), sl.and_then(pf)) {
        (Some(t), Some(s)) if t > 0.0 && s > 0.0 && t.is_finite() && s.is_finite() => Lines::Some(t, s),
        _ => Lines::Bad,
    }
}
/// 진입 지정가(USDT). None = 없음(peg) · Some(Err) = "bad".
pub fn query_limit(q: &Query) -> Option<Result<f64, ()>> {
    let raw = g(q, "limit").filter(|s| !s.is_empty())?;
    Some(match pf(raw) {
        Some(v) if v > 0.0 && v.is_finite() && matches!(query_lines(q), Lines::Some(..)) => Ok(v),
        _ => Err(()),
    })
}
/// 비율(%) → (0,1]. 없으면 1.0 · 이상하면 None(호출부 400 -- **조용히 1.0 으로 떨어뜨리지 않는다**).
pub fn query_fraction(q: &Query, key: &str) -> Option<f64> {
    let Some(raw) = g(q, key).filter(|s| !s.is_empty()) else { return Some(1.0) };
    let p = pf(raw)?;
    (0.0 < p && p <= 100.0).then_some(p / 100.0)
}
pub fn query_wid(q: &Query) -> i64 {
    let raw = g(q, "wid").filter(|s| !s.is_empty()).unwrap_or("1");
    match raw.trim().parse::<i64>() {
        Ok(n) if (1..=WAIT_SLOTS_MAX).contains(&n) => n,
        _ => 1,
    }
}

fn lines_of(l: &Lines) -> Option<(f64, f64)> {
    match l {
        Lines::Some(a, b) => Some((*a, *b)),
        _ => None,
    }
}
fn obj(v: &Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}
fn fl(v: Option<&Value>) -> f64 {
    if truthy(v) { py_float(v.unwrap()).unwrap_or(0.0) } else { 0.0 }
}
fn bad_rule() -> Resp {
    bad(400, "bad_rule", if !MANUAL_RULES_ENABLED { "자동 규칙이 꺼져 있습니다(수동 모드)" } else { "자동 규칙 값은 c 하나뿐입니다" })
}
fn err_resp(e: (Value, u16)) -> Resp {
    let mut b = json!({"ok": false});
    for (k, v) in obj(&e.0) {
        b[k] = v;
    }
    resp(e.1, b)
}
impl Desk {
    fn slot_obj(&mut self, s: &SlotRef) -> &mut Obj {
        match s {
            SlotRef::Entry => &mut self.w.entry,
            SlotRef::Side(a) => self.w.slot_mut(&SlotId::Side(a.clone())),
            SlotRef::Wait(k) => self.w.waits.get_mut(k).and_then(Value::as_object_mut).expect("대기 칸"),
        }
    }
    fn wait_pending(&self, asset: &str) -> bool {
        self.w.wslots(asset).iter().any(|k| self.w.waits[k].as_object().is_some_and(busy))
    }
    fn busy_waits(&self, asset: &str) -> Vec<(i64, Value)> {
        self.w.wslots(asset).into_iter().filter(|k| self.w.waits[k].as_object().is_some_and(busy))
            .map(|k| (k.split_once('#').map_or(1, |x| x.1.parse().unwrap_or(1)), self.w.waits[&k].clone())).collect()
    }

    pub async fn api_entry_preview<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        let side = g(q, "side").unwrap_or("").to_uppercase();
        if side != "LONG" && side != "SHORT" {
            return resp(400, json!({"ok": false, "error": "side must be LONG or SHORT"}));
        }
        let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };
        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "진입 비율은 0 초과 100 이하여야 합니다") };
        let lines = query_lines(q);
        if matches!(lines, Lines::Bad) {
            return bad(400, "bad_lines", "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다");
        }
        let rule = query_rule(q, "rule");
        if rule.as_deref() == Some("bad") {
            return bad_rule();
        }
        let lim = query_limit(q);
        if matches!(lim, Some(Err(()))) {
            return bad(400, "bad_limit", "지정가(limit)는 양수이고 익절·손절 선과 같이 와야 합니다");
        }
        let lev = match query_leverage(q, "lev") { Ok(v) => v, Err(e) => return server_error(&e) };
        let lim = lim.and_then(Result::ok);
        let a = EntryArgs { side, fraction: frac, want_lev: lev, asset, fresh: g(q, "fresh") == Some("1"), rule_l: rule, lines: lines_of(&lines),
                            limit_usdt: lim, stop: lim.is_some() && g(q, "stop") == Some("1"), skip_wait: None };
        match self.assemble_entry_plan(qs, io, &a).await {
            Err(e) => err_resp(e),
            Ok((plan, cap)) => resp(200, json!({"ok": true, "recommended_qty": plan.get("recommended_qty"), "recommended_source": plan.get("recommended_source"),
                                                "plan": plan, "cap": cap, "exec_enabled": io.exec_enabled()})),
        }
    }

    /// 실주문. **POST 전용 + confirm=1 필수 + 게이트가 켜져 있어야** 나간다(G1·G2·G3).
    pub async fn api_entry_submit<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        let side = g(q, "side").unwrap_or("").to_uppercase();
        if side != "LONG" && side != "SHORT" {
            return resp(400, json!({"ok": false, "error": "side must be LONG or SHORT"}));
        }
        if g(q, "confirm") != Some("1") {
            return resp(400, json!({"ok": false, "error": "confirm=1 required"}));
        }
        let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };
        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "진입 비율은 0 초과 100 이하여야 합니다") };
        // 2026-10-10 «기다리기»(mode=wait&wait_min=15|60) · 2026-10-31 지정가 대기(mode=limit, 선 진입 전용)
        let (mut wait_min, mut limit_px) = (0i64, None::<f64>);
        if g(q, "mode") == Some("limit") {
            match query_limit(q) {
                Some(Ok(v)) => limit_px = Some(v),
                _ => return bad(400, "bad_limit", "지정가(limit)는 양수이고 익절·손절 선(tp·sl)과 같이 와야 합니다"),
            }
        } else if g(q, "mode") == Some("wait") {
            wait_min = g(q, "wait_min").filter(|s| !s.is_empty()).unwrap_or("0").trim().parse().unwrap_or(0);
            if !WAIT_CHOICES_MIN.iter().any(|m| *m as i64 == wait_min) {
                return bad(400, "bad_wait", "기다리기는 15·60분만 됩니다");
            }
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        if busy(&self.w.entry) || self.w.side_exits.get(&asset).and_then(Value::as_object).is_some_and(busy) {
            return resp(409, json!({"ok": false, "error": "already_working", "state": self.w.entry}));
        }
        // 11-18 칸 여럿: 걸린 대기가 모두 같은 쪽 «추가 진입 대기(지정가·감시)»면 빈 칸 후보(G10) · 같은 종류·같은 선 대기 = 중복 거절(G11 · 응답 끊겨 재시도·두 기기)
        let stop_q = g(q, "stop") == Some("1");
        let add_ok = |w: &Value| w.get("mode").and_then(Value::as_str) == Some("limit") && truthy(w.get("plan").and_then(|p| p.get("add_keep")))
            && w.get("side").and_then(Value::as_str) == Some(side.as_str());
        let dup = |w: &Value| {
            let st = truthy(w.get("stop")) || truthy(w.get("plan").and_then(|p| p.get("stop")));
            st == stop_q && (fl(w.get("plan").and_then(|p| p.get("limit")).and_then(|l| l.get("usdt"))) - limit_px.unwrap_or(0.0)).abs() < 1e-9
        };
        let bw = self.busy_waits(&asset);
        let extra = !bw.is_empty() && limit_px.is_some() && (bw.len() as i64) < WAIT_SLOTS_MAX && bw.iter().all(|(_, w)| add_ok(w) && !dup(w));
        if self.wait_pending(&asset) && !extra {
            let w = Value::Object(self.w.wait_slot(&asset, 1).clone());
            return resp(409, json!({"ok": false, "error": "wait_pending", "wait": w,
                "detail": "이 코인에 기다리는 진입 주문이 있습니다 — 취소하거나 체결된 뒤에 넣으세요(청산·다른 코인은 됩니다 · 같은 쪽 추가 진입 지정가·감시는 여러 개 됩니다)"}));
        }
        let lines = query_lines(q);
        if matches!(lines, Lines::Bad) {
            return bad(400, "bad_lines", "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다");
        }
        let rule = query_rule(q, "rule");
        if rule.as_deref() == Some("bad") {
            return bad_rule();
        }
        let lev = match query_leverage(q, "lev") { Ok(v) => v, Err(e) => return server_error(&e) };
        // 비율은 여기서 다시 적용한다 -- 프런트가 계산한 수량을 받지 않는다(P1)
        let a = EntryArgs { side: side.clone(), fraction: frac, want_lev: lev, asset: asset.clone(), fresh: g(q, "fresh") == Some("1"), rule_l: rule,
                            lines: lines_of(&lines), limit_usdt: limit_px, stop: limit_px.is_some() && stop_q, skip_wait: None };
        let mut plan = match self.assemble_entry_plan(qs, io, &a).await {
            Err(e) => return err_resp(e),
            Ok((p, _)) => p,
        };
        if truthy(plan.get("blocked")) {
            return resp(400, json!({"ok": false, "error": "blocked", "detail": plan["blocked"]}));
        }
        let now = io.now();
        if wait_min != 0 || limit_px.is_some() {
            let bw = self.busy_waits(&asset); // 계획 조립(await) 사이에 바뀌었을 수 있어 다시 본다(G7)
            if !bw.is_empty() && (!truthy(plan.get("add_keep")) || bw.len() as i64 >= WAIT_SLOTS_MAX || !bw.iter().all(|(_, w)| add_ok(w) && !dup(w))) {
                return resp(409, json!({"ok": false, "error": "wait_pending", "wait": bw[0].1,
                    "detail": format!("이 코인에 기다리는 진입 주문이 있습니다 — 여러 개는 같은 쪽 추가 진입(지정가·감시)끼리만, 같은 선 중복 없이, {WAIT_SLOTS_MAX}개까지 됩니다")}));
            }
            // 검토 11-18 #2: 끝난 칸도 집행 작업(체결 뒤 익절 다시 걸기)이 아직 돌면 비우지 않는다(G12)
            let mut taken: Vec<i64> = bw.iter().map(|x| x.0).collect();
            for n in 1..=WAIT_SLOTS_MAX {
                if self.running.contains(&format!("manual_wait_{}", Watcher::wkey(&asset, n))) && !taken.contains(&n) {
                    taken.push(n);
                }
            }
            if taken.len() as i64 >= WAIT_SLOTS_MAX {
                return bad(409, "wait_pending", &format!("대기 칸 {WAIT_SLOTS_MAX}개가 모두 쓰이는 중입니다 — 잠시 뒤 다시"));
            }
            let wid = (1..=WAIT_SLOTS_MAX).find(|n| !taken.contains(n)).unwrap();
            plan["wid"] = json!(wid);
            let key = Watcher::wkey(&asset, wid);
            let ws = self.w.wait_slot(&asset, wid);
            ws.clear();
            for (k, v) in [("phase", json!("submitting")), ("mode", json!(if limit_px.is_some() { "limit" } else { "wait" })), ("wait_min", json!(wait_min)),
                           ("side", json!(side)), ("plan", plan.clone()), ("asset", json!(asset)), ("wid", json!(wid)), ("started_at", json!(iso_utc(now)))] {
                ws.insert(k.into(), v);
            }
            let ws = Value::Object(ws.clone());
            let tkey = format!("manual_wait_{key}");
            self.running.insert(tkey.clone());
            return Resp { status: 200, body: json!({"ok": true, "plan": plan, "state": ws, "wait": true, "wid": wid}),
                          task: vec![Task::Entry { key: tkey, slot: SlotRef::Wait(key), plan, wait_sec: (wait_min != 0).then(|| wait_min as f64 * 60.0), limit: limit_px.is_some() }] };
        }
        let e = &mut self.w.entry;
        e.clear();
        for (k, v) in [("phase", json!("submitting")), ("side", json!(side)), ("plan", plan.clone()), ("asset", json!(asset)), ("started_at", json!(iso_utc(now)))] {
            e.insert(k.into(), v);
        }
        self.running.insert("manual_entry".into());
        Resp { status: 200, body: json!({"ok": true, "plan": plan, "state": self.w.entry}),
               task: vec![Task::Entry { key: "manual_entry".into(), slot: SlotRef::Entry, plan, wait_sec: None, limit: false }] }
    }

    pub async fn api_exit_preview<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        let side = g(q, "side").unwrap_or("").to_uppercase();
        if side != "LONG" && side != "SHORT" {
            return resp(400, json!({"ok": false, "error": "side must be LONG or SHORT"}));
        }
        let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };
        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "청산 비율은 0 초과 100 이하여야 합니다") };
        // 🔴청산 미리보기도 fresh(2026-09-14 «청산 누르면 강제 조회부터» · C1)
        let ex = g(q, "exec").unwrap_or("limit");
        if ex != "limit" && ex != "market" {
            return bad(400, "bad_exec", "exec 는 limit 또는 market");
        }
        match self.assemble_exit_plan(qs, io, &side, frac, true, &asset, ex == "market", false, None).await {
            Err(e) => err_resp(e),
            Ok(plan) => resp(200, json!({"ok": true, "plan": plan, "exec_enabled": io.exec_enabled()})),
        }
    }

    /// 실제 청산. 진입과 같은 방어(POST + confirm=1 + 게이트) · 칸은 코인별(G6).
    pub async fn api_exit_submit<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        let side = g(q, "side").unwrap_or("").to_uppercase();
        if side != "LONG" && side != "SHORT" {
            return resp(400, json!({"ok": false, "error": "side must be LONG or SHORT"}));
        }
        if g(q, "confirm") != Some("1") {
            return resp(400, json!({"ok": false, "error": "confirm=1 required"}));
        }
        let Some(asset) = query_asset(self, q) else { return bad(400, "bad_asset", "주문할 수 없는 코인입니다") };
        let Some(frac) = query_fraction(q, "pct") else { return bad(400, "bad_pct", "청산 비율은 0 초과 100 이하여야 합니다") };
        let ex = g(q, "exec").unwrap_or("limit");
        if ex != "limit" && ex != "market" {
            return bad(400, "bad_exec", "exec 는 limit 또는 market");
        }
        let to = g(q, "switch_to").unwrap_or("").to_uppercase();
        let mut sw: Option<Value> = None;
        if !to.is_empty() {
            let (sw_frac, sw_rule) = (query_fraction(q, "sw_pct"), query_rule(q, "sw_rule"));
            if to != if side == "LONG" { "SHORT" } else { "LONG" } || frac != 1.0 || sw_frac.is_none() || sw_rule.as_deref() == Some("bad") {
                return bad(400, "bad_switch", "스위칭은 전량(100%) 청산 + 반대 방향 진입만 됩니다");
            }
            let lev = match query_leverage(q, "sw_lev") { Ok(v) => v, Err(e) => return server_error(&e) };
            sw = Some(json!({"to": to, "frac": sw_frac, "lev": lev, "rule": sw_rule}));
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let st = self.w.exit_slot(&asset);
        if st.is_none() || (sw.is_some() && st != Some(SlotId::Entry)) {
            return resp(409, json!({"ok": false, "error": "already_working", "state": self.w.entry})); // 스위칭은 반대 진입이 따라오므로 위 칸만(C4)
        }
        if sw.is_some() && self.wait_pending(&asset) {
            return bad(409, "wait_pending", "기다리는 진입 주문이 있어 스위칭을 막습니다 — 먼저 취소하세요(그냥 청산은 됩니다)"); // G13
        }
        // 2026-10-10 사용자 결정: 스위칭은 즉시 체결(청산·반대 진입 둘 다 시장가 M11) · exec=market = «즉시»(M8) -- 2026-11-21 «원본 동작 유지»
        let plan = match self.assemble_exit_plan(qs, io, &side, frac, true, &asset, sw.is_some() || ex == "market", true, None).await {
            Err(e) => return err_resp(e),
            Ok(p) => p,
        };
        if truthy(plan.get("blocked")) {
            return resp(400, json!({"ok": false, "error": "blocked", "detail": plan["blocked"]}));
        }
        if self.w.exit_slot(&asset) != st {
            return resp(409, json!({"ok": false, "error": "already_working", "state": self.w.entry})); // 조립 사이 손절 청산이 칸을 잡았으면 두 번 내지 않는다(G7)
        }
        let st = st.unwrap();
        let sref = match &st { SlotId::Entry => SlotRef::Entry, SlotId::Side(a) => SlotRef::Side(a.clone()) };
        let now = io.now();
        let s = self.w.slot_mut(&st);
        s.clear();
        for (k, v) in [("phase", json!("submitting")), ("kind", json!("exit")), ("side", json!(side)), ("plan", plan.clone()), ("asset", json!(asset)), ("started_at", json!(iso_utc(now)))] {
            s.insert(k.into(), v);
        }
        if let Some(sw) = &sw {
            s.insert("switch".into(), json!({"from": side, "to": sw["to"], "phase": "exit"}));
        }
        let body = json!({"ok": true, "plan": plan, "state": s});
        let key = if st == SlotId::Entry { "manual_entry".to_string() } else { format!("manual_exit_{asset}") };
        self.running.insert(key.clone());
        let task = match sw {
            Some(sw) => Task::Switch { key, plan, sw, asset },
            None => Task::Exit { key, slot: sref, plan, rearm: frac < 1.0 },
        };
        Resp { status: 200, body, task: vec![task] }
    }

    /// 2026-10-22 «코인 여러 개를 한꺼번에 청산». ⚠화면 단추는 삭제된 것으로 조사됐다(docs/rtcore_port_scope -- «불확실»). 옮기되 보고에 표시.
    pub async fn api_exit_all_submit<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return resp(400, json!({"ok": false, "error": "confirm=1 required"}));
        }
        let ex = g(q, "exec").unwrap_or("limit");
        if ex != "limit" && ex != "market" {
            return bad(400, "bad_exec", "exec 는 limit 또는 market");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let account = match io.account(true).await {
            Ok(a) => a,
            Err(e) => return server_error(&e),
        };
        if !truthy(account.get("ok")) {
            return bad(503, "account_unavailable", &format!("계좌 조회 실패: {}", account.get("error").map_or("None".into(), py_str)));
        }
        let mut targets: Vec<(Option<&'static str>, String, String)> = vec![];
        for p in account.get("positions").and_then(Value::as_array).cloned().unwrap_or_default() {
            let side = p.get("side").map(py_str).unwrap_or_else(|| "".into()).to_uppercase();
            if fl(p.get("qty")) > 0.0 && (side == "LONG" || side == "SHORT") {
                let sym = if truthy(p.get("symbol")) { py_str(&p["symbol"]) } else { String::new() };
                let t = (self.w.exec_asset_of(Some(&json!(sym))), side, sym);
                if !targets.contains(&t) {
                    targets.push(t);
                }
            }
        }
        if targets.is_empty() {
            return bad(400, "no_position", "열린 포지션이 없습니다");
        }
        let (mut results, mut seen, mut tasks) = (vec![], vec![], vec![]);
        for (asset, side, symbol) in targets {
            let mut r = json!({"asset": asset, "side": side, "symbol": symbol, "ok": false});
            let Some(asset) = asset else {
                r["error"] = json!("unknown_symbol");
                r["detail"] = json!(format!("{symbol} 은 대시보드 주문 코인이 아닙니다 — 거래소에서 직접 닫으세요"));
                results.push(r);
                continue;
            };
            if seen.contains(&(asset, side.clone())) {
                r["error"] = json!("already_working");
                r["detail"] = json!("같은 코인·방향의 다른 심볼 — 앞 청산이 끝난 뒤 다시 누르세요");
                results.push(r);
                continue;
            }
            seen.push((asset, side.clone()));
            let Some(st) = self.w.exit_slot(asset) else {
                r["error"] = json!("already_working");
                r["detail"] = json!("이 코인 주문이 진행 중입니다 — 끝난 뒤 다시 누르세요");
                results.push(r);
                continue;
            };
            let plan = match self.assemble_exit_plan(qs, io, &side, 1.0, true, asset, ex == "market", true, None).await {
                Err(e) => {
                    for (k, v) in obj(&e.0) {
                        r[k] = v;
                    }
                    results.push(r);
                    continue;
                }
                Ok(p) => p,
            };
            if truthy(plan.get("blocked")) {
                r["error"] = json!("blocked");
                r["detail"] = plan["blocked"].clone();
                results.push(r);
                continue;
            }
            if self.w.exit_slot(asset) != Some(st.clone()) {
                r["error"] = json!("already_working");
                r["detail"] = json!("이 코인 주문이 막 시작됐습니다 — 끝난 뒤 다시 누르세요");
                results.push(r);
                continue;
            }
            let now = io.now();
            let s = self.w.slot_mut(&st);
            s.clear();
            for (k, v) in [("phase", json!("submitting")), ("kind", json!("exit")), ("side", json!(side)), ("plan", plan.clone()), ("asset", json!(asset)), ("started_at", json!(iso_utc(now)))] {
                s.insert(k.into(), v);
            }
            let key = if st == SlotId::Entry { "manual_entry".to_string() } else { format!("manual_exit_{asset}") };
            self.running.insert(key.clone());
            tasks.push(Task::Exit { key, slot: match &st { SlotId::Entry => SlotRef::Entry, SlotId::Side(a) => SlotRef::Side(a.clone()) }, plan: plan.clone(), rearm: false });
            r["ok"] = json!(true);
            r["quantity"] = plan.get("quantity").cloned().unwrap_or(Value::Null);
            r["type"] = plan.get("type").cloned().unwrap_or(Value::Null);
            r["symbol"] = plan.get("symbol").cloned().unwrap_or(Value::Null);
            results.push(r);
        }
        let ok = results.iter().any(|r| r["ok"] == json!(true));
        Resp { status: if ok { 200 } else { 409 }, body: json!({"ok": ok, "results": results}), task: tasks }
    }

    /// 2026-10-31 «지정가 선을 끌면 주문 수정»(W6). 🔴일부라도 체결된 뒤엔 거절 · 앞 수정이 반영 중이면 거절 · 조립 뒤 상태를 다시 본다.
    pub async fn api_limit_amend<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let qa = query_asset(self, q).unwrap_or_else(|| "eth".into());
        let wid = query_wid(q);
        let st = Value::Object(self.w.wait_slot(&qa, wid).clone());
        let t = |k: &str| truthy(st.get(k));
        let watching = t("stop") && !t("triggered"); // 감시 진입은 닿기 전엔 거래소 주문이 없다
        if st.get("mode").and_then(Value::as_str) != Some("limit") || st.get("phase").and_then(Value::as_str) != Some("working") || t("action")
            || !(t("order_id") || watching) || (t("stop") && t("triggered")) || t("pegged") {
            return resp(409, json!({"ok": false, "error": "no_limit", "detail": "수정할 대기 지정가 주문이 없습니다", "wait": st}));
        }
        if fl(st.get("filled")) > 0.0 {
            return bad(409, "partially_filled", "일부 체결돼 진입가를 못 바꿉니다 — 그 몫의 익절·손절은 포지션 선으로 수정하세요");
        }
        if t("amend") {
            return bad(409, "amend_pending", "이전 수정이 반영되는 중입니다(≤3초)");
        }
        let old = st.get("plan").cloned().unwrap_or(Value::Null);
        let (lines, lim) = (query_lines(q), query_limit(q));
        let (Lines::Some(tp, sl), Some(Ok(lim))) = (lines, lim) else { return bad(400, "bad_limit", "limit·tp·sl 이 모두 양수여야 합니다") };
        let asset = st.get("asset").filter(|v| truthy(Some(v))).map_or("eth".into(), py_str);
        let lev = { let x = fl(old.get("target_leverage")).trunc() as i64; (x != 0).then_some(x) };
        let skip = Watcher::wkey(&asset, wid);
        let a = EntryArgs { side: st.get("side").map(py_str).unwrap_or_default(), fraction: { let f = fl(old.get("fraction")); if f == 0.0 { 1.0 } else { f } },
                            want_lev: lev, asset: asset.clone(), fresh: true, rule_l: None, lines: Some((tp, sl)), limit_usdt: Some(lim), stop: watching, skip_wait: Some(skip) };
        let plan = match self.assemble_entry_plan(qs, io, &a).await {
            Err(e) => return err_resp(e),
            Ok((p, _)) => p,
        };
        if truthy(plan.get("blocked")) {
            return resp(400, json!({"ok": false, "error": "blocked", "detail": plan["blocked"]}));
        }
        let ws = self.w.wait_slot(&qa, wid);
        let lp = if truthy(ws.get("limit_price")) { fl(ws.get("limit_price")) } else { fl(old.get("price")) };
        let moved = (fl(plan.get("price")) - lp).abs() > 1e-9; // 걸려 있는 실제 주문가 기준(자동 재조정이 바꿨을 수 있다)
        let tw = |k: &str| truthy(ws.get(k));
        if ws.get("phase").and_then(Value::as_str) != Some("working") || tw("action") || fl(ws.get("filled")) > 0.0 || tw("amend")
            || (ws.get("order_id").is_none_or(Value::is_null) && !watching) || (watching && tw("triggered")) || tw("pegged") {
            return bad(409, "state_changed", "그 사이 주문 상태가 바뀌었습니다 — 다시 확인하세요");
        }
        let now = io.now();
        let p = ws.entry("plan").or_insert(json!({}));
        p["bracket"] = plan["bracket"].clone();
        p["limit"] = plan.get("limit").cloned().unwrap_or(Value::Null);
        let saved = obj(p);
        if moved {
            for k in ["price", "quantity", "notional_usdt"] {
                p[k] = plan[k].clone(); // 명목 유지: 새 가격에 맞춰 다시 잰 수량(안 그러면 숏을 올릴수록 명목이 커진다)
            }
            ws.insert("amend".into(), json!({"price": plan["price"], "qty": plan["quantity"], "at": now}));
        }
        let wv = Value::Object(ws.clone());
        let _ = self.w.limit_save_at(now, &saved); // 재시작 복구가 «지금 선»으로 무장하도록(W4) -- 원본은 가격을 고치기 전에 저장한다
        self.w.log(format!("limit entry: 수정(사용자) 지정가 {} · 익절 {} 손절 {}", if moved { format!("→ {}", py_str(&plan["price"])) } else { "그대로".into() },
                           plan["bracket"].get("tp_level").map_or("None".into(), py_str), plan["bracket"].get("sl_level").map_or("None".into(), py_str)));
        resp(200, json!({"ok": true, "moved": moved, "price": plan["price"], "bracket": plan["bracket"], "wait": wv}))
    }

    /// 기다리는 진입에 «취소»(잔량 버림) · «PEG»(지정가 대기만, W8). «지금 시장가»는 접수에서 미정 거절(M13).
    pub async fn api_wait_action<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        let action = g(q, "action").unwrap_or("");
        if !["cancel", "market", "peg"].contains(&action) || g(q, "confirm") != Some("1") {
            return bad(400, "bad_action", "action=cancel|market|peg 과 confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let qa = query_asset(self, q).unwrap_or_else(|| "eth".into());
        let wid = query_wid(q);
        let st = Value::Object(self.w.wait_slot(&qa, wid).clone());
        if !busy(st.as_object().unwrap()) {
            return resp(409, json!({"ok": false, "error": "no_wait", "detail": "기다리는 진입 주문이 없습니다", "wait": st}));
        }
        let mode = st.get("mode").and_then(Value::as_str).unwrap_or("");
        if action == "peg" && mode != "limit" {
            return bad(400, "bad_action", "PEG 는 지정가 대기에서만 됩니다");
        }
        // 지정가 대기의 «지금 시장가»(M13 → M4·M5 · 2026-11-21 «원본 동작 유지»)·«PEG»: 주문이 걸려 있고 따라가는 중이 아닐 때만 · 체결될 쪽 호가로 선 다시 검사(W7)
        //   기다리기(mode=wait)의 market 은 검사 없이 접수 → 집행 루프가 잔량 시장가(M3)
        if (action == "market" || action == "peg") && mode == "limit" {
            let t = |k: &str| truthy(st.get(k));
            if st.get("phase").and_then(Value::as_str) != Some("working") || !(t("order_id") || t("stop")) || (t("stop") && t("triggered")) || t("pegged") || t("action") {
                self.w.log(format!("wait-action {action}: 거절 not_ready phase={} order_id={} pegged={} action={}",
                    st.get("phase").map_or("None".into(), py_str), st.get("order_id").map_or("None".into(), py_str), st.get("pegged").map_or("None".into(), py_str), st.get("action").map_or("None".into(), py_str)));
                return resp(409, json!({"ok": false, "error": "not_ready", "detail": "주문이 아직 걸리지 않았거나 이미 처리 중입니다", "wait": st}));
            }
            // peg = 걸린 지정가를 취소 확인한 뒤 남은 수량을 최우선 호가 따라가기(시장가 없음, E26) -- 선 검사는 메이커 쪽 호가로(W7)
            let r: Result<(), (bool, String)> = async {
                let plan = st.get("plan").ok_or((false, "KeyError: 'plan'".to_string()))?;
                let symbol = plan.get("symbol").map(py_str).ok_or((false, "KeyError: 'symbol'".to_string()))?;
                let br = plan.get("bracket").filter(|v| truthy(Some(v))).cloned().unwrap_or(json!({}));
                let book = qs.book(&symbol).await.map_err(|e| (false, e))?;
                let asset = self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth");
                let basis = self.basis_for(qs, &book, asset).await;
                let filters = qs.filters(&symbol).await.map_err(|e| (false, e))?;
                let pside = plan.get("positionSide").map(py_str).ok_or((false, "KeyError: 'positionSide'".to_string()))?;
                let k = if (pside == "LONG") == (action == "market") { "askPrice" } else { "bidPrice" }; // 시장가 = 반대편 · peg = 내 쪽
                let r#ref = py_float(book.get(k).ok_or((false, format!("KeyError: '{k}'")))?).map_err(|e| (false, e))? / basis;
                if !truthy(plan.get("add_keep")) {
                    let lv = |n: &str| br.get(n).ok_or((false, format!("KeyError: '{n}'"))).and_then(|v| py_float(v).map_err(|e| (e.starts_with("ValueError"), e)));
                    plan::line_bracket(&pside, r#ref, lv("tp_level")?, lv("sl_level")?, basis, &filters, false).map_err(|e| (true, e.py_msg()))?;
                }
                Ok(())
            }.await;
            match r {
                Err((true, e)) => {
                    let msg = e.strip_prefix("ValueError: ").unwrap_or(&e).to_string();
                    self.w.log(format!("wait-action {action}: 거절 lines_off {msg}"));
                    return bad(409, "lines_off", &format!("지금 호가 기준으로 익절·손절선이 맞지 않아 {}로 넣지 않습니다 — {msg}", if action == "market" { "시장가" } else { "peg" }));
                }
                Err((false, e)) => {
                    self.w.log(format!("wait-action {action}: 거절 quote_failed {e}"));
                    return bad(502, "quote_failed", &e);
                }
                Ok(()) => {}
            }
            let cur = self.w.wait_slot(&qa, wid);
            if cur.get("phase").and_then(Value::as_str) != Some("working") || truthy(cur.get("action")) {
                return bad(409, "state_changed", "그 사이 주문 상태가 바뀌었습니다 — 다시 확인하세요");
            }
        }
        let now = io.now();
        let ws = self.w.wait_slot(&qa, wid);
        ws.insert("action".into(), json!(action));
        ws.insert("action_at".into(), json!(iso_utc(now)));
        ws.insert("action_reason".into(), if action == "cancel" { json!("사용자 취소") } else { Value::Null });
        let wv = Value::Object(ws.clone());
        self.w.log(format!("wait-action {action}: 접수 {} {} order_id={}", wv.get("asset").map_or("None".into(), py_str), wv.get("side").map_or("None".into(), py_str),
                           wv.get("order_id").map_or("None".into(), py_str)));
        resp(200, json!({"ok": true, "wait": wv}))
    }

    /// 따라가기 진입의 «대기 취소»(E18). 🔴청산은 안 받는다(C6 -- 남은 포지션을 그대로 두는 것이라).
    pub fn api_entry_cancel<I: DeskIo>(&mut self, io: &I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let st = &mut self.w.entry;
        if !matches!(st.get("phase").and_then(Value::as_str), Some("working" | "submitting")) || st.get("kind").and_then(Value::as_str) == Some("exit")
            || truthy(st.get("switch")) || st.get("plan").and_then(|p| p.get("type")).and_then(Value::as_str) == Some("MARKET") {
            return resp(409, json!({"ok": false, "error": "no_entry", "detail": "기다리는 따라가기 진입이 없습니다", "state": st}));
        }
        st.insert("action".into(), json!("cancel"));
        st.insert("action_at".into(), json!(iso_utc(io.now())));
        st.insert("action_reason".into(), json!("사용자 취소"));
        resp(200, json!({"ok": true, "state": st}))
    }

    fn side_symbol(&self, q: &Query) -> Option<(String, String)> {
        let (side, symbol) = (g(q, "side")?, g(q, "symbol")?);
        ((side == "LONG" || side == "SHORT") && self.exec_symbols().iter().any(|s| s == symbol)).then(|| (side.into(), symbol.into()))
    }

    /// 2026-10-15 «손절 해제»(B21): 우리 비상 스탑(dbsl)만 지우고 sl_off 를 남긴다. 🔴거래소 취소가 실패하면 아무것도 안 바꾼다.
    pub async fn api_bracket_disarm<I: DeskIo>(&mut self, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let Some((side, symbol)) = self.side_symbol(q) else { return bad(400, "bad_side", "side=LONG|SHORT · symbol 이 필요합니다") };
        let key = bracket_key(&symbol, &side);
        let b = self.w.load().get(&key).cloned().unwrap_or(json!({}));
        if !truthy(Some(&b)) {
            return bad(409, "not_armed", "서버가 무장한 손절이 없습니다");
        }
        if truthy(b.get("sl_off")) {
            return resp(200, json!({"ok": true, "sl_off": b["sl_off"], "already": true}));
        }
        // 2026-11-21 사용자 결정 «−5% 비상 스탑(M10) Rust 에서 제거»: 지울 거래소 스탑이 없다 -- 원본의 clear_bracket(only_sl) 요청·취소 실패 502 는 없다(의도한 차이)
        let r = json!({"cancelled": 0, "errors": []});
        let mut st = self.w.load();
        if !st.get(&key).is_some_and(|c| crate::watch::same_armed(c.get("armed_at"), b.get("armed_at"))) {
            return bad(409, "rearmed", "그 사이 새로 무장됐습니다 — 다시 확인하세요");
        }
        let off = json!({"at": io.now(), "cancelled": fl(r.get("cancelled")) as i64});
        st[&key]["sl_off"] = off.clone();
        if let Err(e) = self.w.save(&st) {
            return server_error(&e);
        }
        self.w.log(format!("bracket {side}: 손절 해제(사용자) 거래소 스탑 {}개 취소 · 5분봉 감시 끔", r.get("cancelled").map_or("None".into(), py_str)));
        resp(200, json!({"ok": true, "sl_off": off}))
    }

    /// «손절 다시 켜기»(B22). 선 무장 = 감시만 다시 · 규칙 무장 = 거래소 스탑 다시(못 걸면 아무것도 안 바꾼다 · 시장가형 스탑 = M10).
    pub async fn api_bracket_rearm<I: DeskIo>(&mut self, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let Some((side, symbol)) = self.side_symbol(q) else { return bad(400, "bad_side", "side=LONG|SHORT · symbol 이 필요합니다") };
        let key = bracket_key(&symbol, &side);
        let b = self.w.load().get(&key).cloned().unwrap_or(json!({}));
        if !truthy(Some(&b)) {
            return bad(409, "not_armed", "서버가 무장한 손절이 없습니다");
        }
        if !truthy(b.get("sl_off")) {
            return resp(200, json!({"ok": true, "already": true}));
        }
        if truthy(b.get("lines")) {
            if !truthy(b.get("sl_level")) {
                return bad(409, "no_sl_line", "다시 켤 손절선이 없습니다 — 진입가 팝업의 «손절선 추가»로 거세요");
            }
            if b.get("fired") == Some(&json!("tp")) || self.w.exiting.contains(&key) {
                return bad(409, "fired", "익절 청산이 진행 중입니다 — 끝난 뒤 다시");
            }
            let mut st = self.w.load();
            if !st.get(&key).is_some_and(|c| crate::watch::same_armed(c.get("armed_at"), b.get("armed_at"))) {
                return bad(409, "rearmed", "그 사이 새로 무장됐습니다 — 다시 확인하세요");
            }
            let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
            cur.remove("sl_off");
            cur.remove("fired");
            if truthy(cur.get("sl_legs")) {
                let legs: Vec<Value> = cur["sl_legs"].as_array().cloned().unwrap_or_default().into_iter()
                    .map(|x| { let mut o = obj(&x); o.remove("fired"); Value::Object(o) }).collect();
                cur.insert("sl_legs".into(), Value::Array(legs));
            }
            cur.insert("watch_from".into(), json!(io.now())); // 해제 중에 지나간 가격으로 바로 닫지 않게
            if let Err(e) = self.w.save(&st) {
                return server_error(&e);
            }
            return resp(200, json!({"ok": true, "lines": true}));
        }
        let price = if truthy(b.get("backstop_price")) { b["backstop_price"].clone() } else { b.get("sl_price").cloned().unwrap_or(Value::Null) };
        if !truthy(Some(&price)) {
            return bad(409, "no_price", "다시 걸 손절가가 기록에 없습니다");
        }
        // 2026-11-21 사용자 결정 «−5% 비상 스탑(M10) Rust 에서 제거»: 거래소 스탑을 다시 걸지 않는다 -- 5분봉 종가 감시만 다시 켠다(의도한 차이 · 원본 place_stop 실패 502 는 없다)
        let r = Value::Null;
        let mut st = self.w.load();
        if !st.get(&key).is_some_and(|c| crate::watch::same_armed(c.get("armed_at"), b.get("armed_at"))) {
            return bad(409, "rearmed", "그 사이 새로 무장됐습니다 — 다시 확인하세요");
        }
        let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
        cur.remove("sl_off");
        if let Err(e) = self.w.save(&st) {
            return server_error(&e);
        }
        self.w.log(format!("bracket {side}: 손절 다시 켬(사용자) 거래소 스탑 없음(M10 제거) · 5분봉 감시 켬 {}", py_str(&price)));
        resp(200, json!({"ok": true, "backstop": r}))
    }

    /// 작업이 끝났다(원본 asyncio Task.done()) -- 대기 칸 재사용 규칙(G12)이 본다.
    pub fn task_done(&mut self, key: &str) {
        self.running.remove(key);
    }
    /// 칸 하나 읽기(시험·화면).
    pub fn slot_value(&mut self, s: &SlotRef) -> Value {
        Value::Object(self.slot_obj(s).clone())
    }
}

// ── 선 핸들러 4개(2026-11-21 여섯째 조각) -- orders.py 의 api_manual_bracket_tp_legs · _tp · _add · _modify ──────────────
// 방어 = 진입과 같다(POST + confirm=1 + 게이트 + 주문 잠금 -- LOCKED 표). 손절 = 서버 감시 레벨뿐(거래소 주문 없음) · 익절 = 우리 post-only 지정가.

fn tp_legs_of(b: &Obj) -> Vec<Value> {
    if truthy(b.get("tp_legs")) {
        return b["tp_legs"].as_array().cloned().unwrap_or_default();
    }
    if truthy(b.get("tp_level")) && truthy(b.get("tp_price")) {
        return vec![json!({"level": b["tp_level"], "price": b["tp_price"], "w": 1.0, "resting": truthy(b.get("tp_resting")), "cid": null})];
    }
    vec![]
}
fn sl_legs_of(b: &Obj) -> Vec<Value> {
    if truthy(b.get("sl_legs")) {
        return b["sl_legs"].as_array().cloned().unwrap_or_default();
    }
    if truthy(b.get("sl_level")) {
        return vec![json!({"level": b["sl_level"], "price": b.get("sl_price"), "up": b.get("sl_up"), "w": 1.0})];
    }
    vec![]
}
fn query_slup(q: &Query) -> Option<bool> {
    match g(q, "slup") {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}
fn rearmed() -> Resp {
    bad(409, "rearmed", "그 사이 무장이 바뀌었거나 청산이 시작됐습니다 — 다시 확인하세요")
}
fn fired_busy() -> Resp {
    bad(409, "fired", "청산이 진행 중입니다 — 끝난 뒤 다시")
}
fn fv(v: Option<&Value>) -> f64 {
    v.and_then(|x| py_float(x).ok()).unwrap_or(0.0)
}

impl Desk {
    fn same_arm(&self, key: &str, b: &Value) -> bool {
        let st = self.w.load();
        st.get(key).is_some_and(|c| crate::watch::same_armed(c.get("armed_at"), b.get("armed_at")) && !truthy(c.get("fired"))) && !self.w.exiting.contains(key)
    }

    /// 선 검사 기준 = (지금가 USDT, basis, 필터). 지금가 = 1초 마지막 체결(5초 넘게 묵으면 안 씀, 검토 S6) · 없으면 주문 심볼 중간가 ÷ basis.
    async fn lines_quote<Q: Quotes, I: DeskIo>(&self, qs: &mut Q, io: &mut I, symbol: &str, asset: &str) -> Result<(f64, f64, plan::Filters), String> {
        let book = qs.book(symbol).await?;
        let basis = self.basis_for(qs, &book, asset).await;
        let filters = qs.filters(symbol).await?;
        let pts = io.sec_prices(asset).unwrap_or_default();
        let r = match pts.last() {
            Some((t, p)) if io.now() - t <= 5.0 => *p,
            _ => {
                let k = |n: &str| book.get(n).ok_or_else(|| format!("KeyError: '{n}'")).and_then(py_float);
                (k("bidPrice")? + k("askPrice")?) / 2.0 / basis
            }
        };
        Ok((r, basis, filters))
    }

    /// line_bracket(sl_watch) 을 한쪽 선만으로도 -- 없는 쪽은 늘 통과하는 자리(지금가 ±1%). ponytail(원본): 대역 값을 넣는 꼼수.
    fn check_lines(side: &str, r: f64, basis: f64, filters: &plan::Filters, tp: Option<f64>, sl: Option<f64>, slup: Option<bool>) -> Result<Obj, String> {
        let up = side == "LONG";
        let lb = plan::line_bracket(side, r, tp.filter(|x| *x != 0.0).unwrap_or(r * if up { 1.01 } else { 0.99 }), sl.filter(|x| *x != 0.0).unwrap_or(r * if up { 0.99 } else { 1.01 }),
                                    basis, filters, true).map_err(|e| e.py_msg())?;
        let mut nb = obj(&serde_json::to_value(lb).unwrap());
        if let (Some(_), Some(u)) = (sl.filter(|x| *x != 0.0), slup) {
            nb.insert("sl_up".into(), json!(u)); // 2026-11-12 검토 «손절 방향 뒤집힘»: 화면이 놓은 순간의 방향
        }
        Ok(nb)
    }

    /// 조각을 거래소에 다시 걸고 무장에 쓴다. 실패 = 오류 응답 -- 🔴키·포지션 조회 실패나 옛 익절 정리 실패면 아무것도 안 바꾼다 · 그 사이 무장이 바뀌면 방금 건 조각을 지운다.
    async fn ladder_apply<I: DeskIo>(&mut self, io: &mut I, key: &str, b: &Value, symbol: &str, side: &str, legs: &[Value]) -> Result<Obj, Resp> {
        let r = io.place_tp_ladder(&json!(legs), symbol, side).await.map_err(|e| server_error(&e))?;
        if r.get("cleared").is_none() {
            return Err(bad(502, "tp_place_failed", &format!("익절 지정가를 못 바꿨습니다(아무것도 안 바뀜): {}", r.get("reason").filter(|v| truthy(Some(v))).map_or("?".into(), py_str))));
        }
        if truthy(r["cleared"].get("errors")) {
            let _ = self.w.tp_watch_only_at(io.now(), key, &obj(b));
            let es: Vec<String> = r["cleared"]["errors"].as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_default();
            return Err(bad(502, "tp_partial_cancel", &format!("익절 주문 정리가 일부 실패해 새로 걸지 않았습니다 — 기존 익절선은 이제 서버가 닿으면 닫습니다(거래소에 익절 주문이 일부 남았을 수 있으니 확인하세요): {}", es.join("; "))));
        }
        let rl = r.get("legs").and_then(Value::as_array).cloned().unwrap_or_default();
        if !self.same_arm(key, b) {
            if rl.iter().any(|x| truthy(x.get("resting"))) {
                io.clear_bracket(symbol, side, false, true).await.map_err(|e| server_error(&e))?;
            }
            return Err(rearmed());
        }
        let mut st = self.w.load();
        let now = io.now();
        let cur = st.get_mut(key).and_then(Value::as_object_mut).unwrap();
        crate::watch::set_tp_legs(cur, &rl).map_err(|e| server_error(&e))?;
        let n = cur.get("tp_legs").and_then(Value::as_array).map_or(1, Vec::len);
        let m = json!({"at": now, "tp": cur.get("tp_level"), "sl": cur.get("sl_level"), "legs": n});
        cur.insert("watch_from".into(), json!(now));
        cur.insert("mod".into(), m);
        let out = cur.clone();
        self.w.save(&st).map_err(|e| server_error(&e))?;
        let tup: Vec<String> = rl.iter().map(|x| format!("({}, {}, {})", x.get("level").map_or("None".into(), crate::sign::py_repr), x.get("qty").map_or("None".into(), crate::sign::py_repr),
                                                       x.get("resting").map_or("None".into(), crate::sign::py_repr))).collect();
        self.w.log(format!("bracket {side}: 분할 익절 다시 걸음(사용자) [{}]", tup.join(", ")));
        Ok(out)
    }

    /// 2026-11-11 분할 익절·손절 조각 조작: act=remove&i=N · act=weights&w=1,1,2 · kind=tp|sl.
    pub async fn api_bracket_tp_legs<I: DeskIo>(&mut self, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let (side, symbol, act, kind) = (g(q, "side").unwrap_or(""), g(q, "symbol").unwrap_or(""), g(q, "act").unwrap_or(""), g(q, "kind").unwrap_or("tp"));
        if !(side == "LONG" || side == "SHORT") || !self.exec_symbols().iter().any(|s| s == symbol) || !(act == "remove" || act == "weights") || !(kind == "tp" || kind == "sl") {
            return bad(400, "bad_args", "side · symbol · act=remove|weights · kind=tp|sl 이 필요합니다");
        }
        let (key, asset, name) = (bracket_key(symbol, side), self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth"), if kind == "tp" { "익절" } else { "손절" });
        let b = self.w.load().get(&key).cloned().unwrap_or(json!({}));
        let bo = obj(&b);
        let mut legs = if kind == "tp" { tp_legs_of(&bo) } else { sl_legs_of(&bo) };
        if !truthy(Some(&b)) || !truthy(b.get("lines")) || truthy(b.get(&format!("{kind}_off"))) || legs.len() < 2 {
            return bad(409, "no_legs", &format!("나눠 걸린 {name}선이 없습니다"));
        }
        if truthy(b.get("fired")) || self.w.exiting.contains(&key) || self.w.exit_slot(asset).is_none() || legs.iter().any(|x| truthy(x.get("fired"))) {
            return fired_busy();
        }
        let parsed: Option<Vec<Value>> = if act == "remove" {
            g(q, "i").unwrap_or("").trim().parse::<i64>().ok().filter(|i| (0..legs.len() as i64).contains(i)).map(|i| { legs.remove(i as usize); legs.clone() })
        } else {
            let w: Option<Vec<f64>> = g(q, "w").unwrap_or("").split(',').map(pf).collect();
            w.filter(|w| w.len() == legs.len() && w.iter().all(|x| *x > 0.0 && x.is_finite()))
                .map(|w| legs.iter().zip(w).map(|(x, v)| { let mut m = obj(x); m.insert("w".into(), json!(v)); Value::Object(m) }).collect())
        };
        let Some(legs) = parsed else { return bad(400, "bad_args", "i(조각 번호) 또는 w(조각 수만큼 양수)가 맞지 않습니다") };
        if kind == "sl" {
            // 2026-11-11 분할 손절 = 서버 감시 레벨뿐(거래소 주문 없음)
            if !self.w.load().get(&key).is_some_and(|c| crate::watch::same_armed(c.get("armed_at"), b.get("armed_at")) && !truthy(c.get("fired"))) {
                return rearmed();
            }
            let mut st = self.w.load();
            let now = io.now();
            let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
            crate::watch::set_sl_legs(cur, &legs);
            let n = cur.get("sl_legs").and_then(Value::as_array).map_or(1, Vec::len);
            let m = json!({"at": now, "tp": cur.get("tp_level"), "sl": cur.get("sl_level"), "sl_legs": n});
            cur.insert("watch_from".into(), json!(now));
            cur.insert("mod".into(), m);
            let body = json!({"ok": true, "act": act, "kind": "sl", "sl_legs": cur.get("sl_legs"), "sl_level": cur.get("sl_level"), "sl_price": cur.get("sl_price"), "sl_up": cur.get("sl_up")});
            if let Err(e) = self.w.save(&st) {
                return server_error(&e);
            }
            let tup: Vec<String> = legs.iter().map(|x| format!("({}, {})", crate::sign::py_repr(&x["level"]), x.get("w").map_or("None".into(), crate::sign::py_repr))).collect();
            self.w.log(format!("bracket {side}: 분할 손절 {act}(사용자) [{}]", tup.join(", ")));
            return resp(200, body);
        }
        match self.ladder_apply(io, &key, &b, symbol, side, &legs).await {
            Err(r) => r,
            Ok(cur) => resp(200, json!({"ok": true, "act": act, "tp_legs": cur.get("tp_legs"), "tp_level": cur.get("tp_level"), "tp_resting": cur.get("tp_resting")})),
        }
    }

    /// 2026-11-11 «익절선 지우기 버튼»: act=off = 우리 익절(dbtp)만 지우고 tp_off · act=on = 저장된 익절 주문가로 post-only 다시(지금가가 이미 너머면 409).
    pub async fn api_bracket_tp<I: DeskIo>(&mut self, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let (side, symbol, act) = (g(q, "side").unwrap_or(""), g(q, "symbol").unwrap_or(""), g(q, "act").unwrap_or(""));
        if !(side == "LONG" || side == "SHORT") || !self.exec_symbols().iter().any(|s| s == symbol) || !(act == "off" || act == "on") {
            return bad(400, "bad_side", "side=LONG|SHORT · symbol · act=off|on 이 필요합니다");
        }
        let (key, asset) = (bracket_key(symbol, side), self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth"));
        let b = self.w.load().get(&key).cloned().unwrap_or(json!({}));
        if !truthy(Some(&b)) || !truthy(b.get("lines")) || !truthy(b.get("tp_price")) {
            return bad(409, "not_armed_lines", "선으로 들어간 익절이 없습니다");
        }
        if truthy(b.get("fired")) || self.w.exiting.contains(&key) || self.w.exit_slot(asset).is_none() {
            return fired_busy();
        }
        if (act == "off") == truthy(b.get("tp_off")) {
            return resp(200, json!({"ok": true, "already": true, "tp_off": b.get("tp_off")}));
        }
        let now = io.now();
        let mut res = Obj::new();
        if act == "off" {
            let r = match io.clear_bracket(symbol, side, false, true).await { Ok(r) => r, Err(e) => return server_error(&e) };
            if truthy(r.get("errors")) {
                // 2026-11-12 검토: 조각 몇 개는 지워졌을 수 있다 -- «그대로»가 아니라 서버 감시로 돌린다
                let _ = self.w.tp_watch_only_at(now, &key, &obj(&b));
                let es: Vec<String> = r["errors"].as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_default();
                return bad(502, "cancel_failed", &format!("거래소 익절 지정가 취소가 일부 실패했습니다 — 익절선은 남아 이제 서버가 닿으면 닫습니다(거래소에 익절 주문이 일부 남았을 수 있음): {}", es.join("; ")));
            }
            res.insert("tp_off".into(), json!({"at": now, "cancelled": fv(r.get("cancelled")) as i64}));
            res.insert("tp_resting".into(), json!(false));
            if truthy(b.get("tp_legs")) {
                let mut st0 = self.w.load();
                if let Some(Value::Object(c)) = st0.get_mut(&key) {
                    c.remove("tp_legs");
                    if let Err(e) = self.w.save(&st0) {
                        return server_error(&e);
                    }
                }
            }
        } else {
            let pts = io.sec_prices(asset).unwrap_or_default();
            let r0 = match pts.last() { Some((t, p)) if now - t <= 5.0 => *p, _ => 0.0 };
            let tp = fv(b.get("tp_level"));
            if !(r0 > 0.0) {
                return bad(502, "quote_failed", "지금가를 못 읽었습니다 — 잠시 뒤 다시");
            }
            if if side == "LONG" { r0 >= tp } else { r0 <= tp } {
                return bad(409, "past_tp", &format!("지금가 {} 가 이미 익절선 {} 너머입니다 — 다시 걸면 바로 닫힙니다", crate::pyfmt::fixed(r0, 2, false), crate::pyfmt::fixed(tp, 2, false)));
            }
            let r = match io.place_bracket(&json!({"tp_price": b["tp_price"]}), symbol, side, true).await { Ok(r) => r, Err(e) => return server_error(&e) };
            if r.get("cleared").is_none() {
                return bad(502, "tp_place_failed", &format!("익절 지정가를 못 걸었습니다(아무것도 안 바뀜): {}", r.get("reason").filter(|v| truthy(Some(v))).map_or("?".into(), py_str)));
            }
            res.insert("tp_resting".into(), json!(truthy(r.get("tp").and_then(|t| t.get("placed")))));
            res.insert("watch_from".into(), json!(now));
            let mut placed = obj(b.get("placed").unwrap_or(&Value::Null));
            placed.insert("tp".into(), r.get("tp").cloned().unwrap_or(Value::Null));
            res.insert("placed".into(), Value::Object(placed));
        }
        if !self.same_arm(&key, &b) {
            if act == "on" && truthy(res.get("placed").and_then(|p| p.get("tp")).and_then(|t| t.get("placed"))) {
                let _ = io.clear_bracket(symbol, side, false, true).await; // 방금 건 지정가가 고아로 남지 않게
            }
            return rearmed();
        }
        let mut st = self.w.load();
        let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
        for (k, v) in &res {
            cur.insert(k.clone(), v.clone());
        }
        if act == "on" {
            cur.remove("tp_off");
        }
        let body = json!({"ok": true, "act": act, "tp_resting": cur.get("tp_resting"), "tp_off": cur.get("tp_off")});
        if let Err(e) = self.w.save(&st) {
            return server_error(&e);
        }
        self.w.log(format!("bracket {side}: 익절선 {}(사용자) {}", if act == "off" { "지움" } else { "다시 걸음" }, crate::sign::py_repr(&Value::Object(res))));
        resp(200, body)
    }

    /// 2026-11-11 «손절/익절 라인 추가»(leg=sl|tp · level USDT): 지운 쪽 다시 · 무장 없는 포지션 = 새 선 무장 · 그 선이 이미 있으면 분할 조각 하나 더(최대 4).
    pub async fn api_bracket_add<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let (side, symbol, leg) = (g(q, "side").unwrap_or(""), g(q, "symbol").unwrap_or(""), g(q, "leg").unwrap_or(""));
        let level = g(q, "level").filter(|s| !s.is_empty()).map_or(Some(0.0), pf).unwrap_or(0.0);
        if !(side == "LONG" || side == "SHORT") || !self.exec_symbols().iter().any(|s| s == symbol) || !(leg == "sl" || leg == "tp") || !(level > 0.0 && level.is_finite()) {
            return bad(400, "bad_args", "side=LONG|SHORT · symbol · leg=sl|tp · level(양수) 이 필요합니다");
        }
        let (key, asset, name) = (bracket_key(symbol, side), self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth"), if leg == "sl" { "손절" } else { "익절" });
        let b: Option<Value> = self.w.load().get(&key).cloned().filter(|v| truthy(Some(v)));
        if b.as_ref().is_some_and(|b| !truthy(b.get("lines"))) {
            return bad(409, "rule_armed", "규칙 손절(거래소 스탑)이 걸린 포지션이라 선을 더하지 않습니다");
        }
        if b.as_ref().is_some_and(|b| truthy(b.get("fired"))) || self.w.exiting.contains(&key) || self.w.exit_slot(asset).is_none() {
            return fired_busy();
        }
        let bo = b.as_ref().map(obj).unwrap_or_default();
        let ladder = b.is_some() && !truthy(bo.get(&format!("{leg}_off"))) && truthy(bo.get(&format!("{leg}_level")));
        if b.is_some() && !truthy(bo.get(&format!("{leg}_off"))) && !ladder {
            return bad(409, "has_line", &format!("{name}선이 이미 있습니다 — 끌어서 옮기세요"));
        }
        if ladder && (if leg == "tp" { tp_legs_of(&bo) } else { sl_legs_of(&bo) }).len() >= plan::TP_LEGS_MAX {
            return bad(409, "too_many", &format!("{name}선은 {}개까지입니다", plan::TP_LEGS_MAX));
        }
        if ladder && leg == "sl" && bo.get("sl_legs").and_then(Value::as_array).is_some_and(|l| l.iter().any(|x| truthy(x.get("fired")))) {
            return bad(409, "fired", "손절 조각 청산이 진행 중입니다 — 끝난 뒤 다시");
        }
        let account = match io.account(false).await { Ok(a) => a, Err(e) => return server_error(&e) };
        let pos = account.get("positions").and_then(Value::as_array).is_some_and(|ps| ps.iter().any(|p| p.get("symbol").and_then(Value::as_str) == Some(symbol)
            && p.get("side").and_then(Value::as_str) == Some(side) && fv(p.get("qty")).abs() > 0.0));
        if !(truthy(account.get("ok")) && pos) {
            return bad(409, "no_position", "그 포지션을 계좌에서 못 찾았습니다");
        }
        let (basis, nb) = match self.lines_quote(qs, io, symbol, asset).await {
            Err(e) => return bad(502, "quote_failed", &e),
            Ok((r, basis, filters)) => match Self::check_lines(side, r, basis, &filters, (leg == "tp").then_some(level), (leg == "sl").then_some(level), query_slup(q)) {
                Err(m) => return bad(400, "bad_lines", &m),
                Ok(nb) => (basis, nb),
            },
        };
        let now = io.now();
        let near = |old: &[Value], lv: f64| old.iter().any(|x| (fv(x.get("level")) / lv - 1.0).abs() < 5e-4);
        let wavg = |old: &[Value]| old.iter().map(|x| { let w = fv(x.get("w")); if w == 0.0 { 1.0 } else { w } }).sum::<f64>() / old.len() as f64;
        if ladder && leg == "sl" {
            let old = sl_legs_of(&bo);
            if near(&old, fv(nb.get("sl_level"))) {
                return bad(409, "same_level", "이미 그 가격(±0.05%)에 손절선이 있습니다");
            }
            if !self.same_arm(&key, b.as_ref().unwrap()) {
                return rearmed();
            }
            let mut st = self.w.load();
            let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
            let mut all = old.clone();
            all.push(json!({"level": nb["sl_level"], "price": nb["sl_price"], "up": nb["sl_up"], "w": wavg(&old)}));
            crate::watch::set_sl_legs(cur, &all);
            let m = json!({"at": now, "tp": cur.get("tp_level"), "sl": cur.get("sl_level"), "add": "sl"});
            cur.insert("watch_from".into(), json!(now));
            cur.insert("mod".into(), m);
            let n = cur.get("sl_legs").and_then(Value::as_array).map_or(1, Vec::len);
            let body = json!({"ok": true, "leg": "sl", "level": nb["sl_level"], "sl_legs": cur.get("sl_legs"), "sl_level": cur.get("sl_level"), "sl_price": cur.get("sl_price"),
                              "sl_up": cur.get("sl_up"), "basis": basis});
            if let Err(e) = self.w.save(&st) {
                return server_error(&e);
            }
            self.w.log(format!("bracket {side}: 손절 조각 추가(사용자) {} -- {n}개", crate::sign::py_repr(&nb["sl_level"])));
            return resp(200, body);
        }
        if ladder {
            let old = tp_legs_of(&bo);
            if near(&old, fv(nb.get("tp_level"))) {
                return bad(409, "same_level", "이미 그 가격(±0.05%)에 익절선이 있습니다");
            }
            let mut all = old.clone();
            all.push(json!({"level": nb["tp_level"], "price": nb["tp_price"], "w": wavg(&old), "resting": false, "cid": null}));
            return match self.ladder_apply(io, &key, b.as_ref().unwrap(), symbol, side, &all).await {
                Err(r) => r,
                Ok(cur) => resp(200, json!({"ok": true, "leg": "tp", "level": nb["tp_level"], "tp_legs": cur.get("tp_legs"), "tp_resting": cur.get("tp_resting"), "basis": basis,
                                            "warn": if truthy(cur.get("tp_resting")) { Value::Null } else { json!("일부 익절 지정가를 못 걸어 서버가 닿으면 닫습니다") }})),
            };
        }
        let mut tp_res: Option<Value> = None;
        if leg == "tp" {
            let r = match io.place_bracket(&json!({"tp_price": nb["tp_price"]}), symbol, side, true).await { Ok(r) => r, Err(e) => return server_error(&e) };
            if r.get("cleared").is_none() {
                return bad(502, "tp_place_failed", &format!("익절 지정가를 못 걸었습니다(아무것도 안 바뀜): {}", r.get("reason").filter(|v| truthy(Some(v))).map_or("?".into(), py_str)));
            }
            tp_res = Some(r);
        }
        let now = io.now();
        let mut st = self.w.load();
        let cur0 = st.get(&key).cloned();
        let same = crate::watch::same_armed(cur0.as_ref().and_then(|c| c.get("armed_at")), b.as_ref().and_then(|b| b.get("armed_at")));
        if !same || cur0.as_ref().is_some_and(|c| truthy(c.get("fired"))) || self.w.exiting.contains(&key) {
            if tp_res.as_ref().is_some_and(|r| truthy(r.get("tp").and_then(|t| t.get("placed")))) {
                let _ = io.clear_bracket(symbol, side, false, true).await;
            }
            return rearmed();
        }
        if cur0.is_none() {
            // 무장 없는 포지션 = 새 선 무장 · 다른 쪽은 꺼진 채(never = 걸린 적 없음 -- «다시 켜기» 대상이 아님)
            let other = if leg == "sl" { "tp" } else { "sl" };
            st.insert(key.clone(), json!({"symbol": symbol, "side": side, "lines": true, "rule": false, "armed_at": now, "backstop_price": null, "basis": basis,
                                          format!("{other}_off"): {"at": now, "never": true}}));
        }
        let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
        if leg == "sl" {
            for k in ["sl_level", "sl_price", "sl_name", "sl_up"] {
                cur.insert(k.into(), nb.get(k).cloned().unwrap_or(Value::Null));
            }
            cur.remove("sl_legs"); // 지운 뒤 새로 거는 손절 = 하나(옛 조각은 버린다)
        } else {
            let r = tp_res.as_ref().unwrap();
            cur.insert("tp_level".into(), nb["tp_level"].clone());
            cur.insert("tp_price".into(), nb["tp_price"].clone());
            cur.insert("tp_resting".into(), json!(truthy(r.get("tp").and_then(|t| t.get("placed")))));
            let mut placed = obj(cur.get("placed").unwrap_or(&Value::Null));
            placed.insert("tp".into(), r.get("tp").cloned().unwrap_or(Value::Null));
            cur.insert("placed".into(), Value::Object(placed));
        }
        cur.remove(&format!("{leg}_off"));
        let m = json!({"at": now, "tp": cur.get("tp_level"), "sl": cur.get("sl_level"), "add": leg});
        cur.insert("watch_from".into(), json!(now));
        cur.insert("mod".into(), m);
        let body = json!({"ok": true, "leg": leg, "level": nb[&format!("{leg}_level")], "sl_up": cur.get("sl_up"), "tp_resting": cur.get("tp_resting"), "basis": basis,
                          "warn": if leg == "tp" && !truthy(cur.get("tp_resting")) { json!("익절 지정가를 못 걸어 서버가 닿으면 닫습니다") } else { Value::Null }});
        if let Err(e) = self.w.save(&st) {
            return server_error(&e);
        }
        self.w.log(format!("bracket {side}: {name}선 추가(사용자) {} · {} · {}", crate::sign::py_repr(&nb[&format!("{leg}_level")]), if b.is_none() { "새 무장" } else { "지운 쪽 다시" },
                           tp_res.as_ref().map_or("None".into(), |r| r.get("tp").map_or("None".into(), crate::sign::py_repr))));
        resp(200, body)
    }

    /// 2026-10-29 «선을 끌면 그대로 주문 수정»(선 무장만). 익절 레벨(USDT)이 그대로면 지정가를 안 건드린다(검토 F3) · 옮기면 strict 로 다시(F2) ·
    /// 무장이 바뀌면 방금 건 지정가를 지운다(S2) · watch_from = 지금 · 지운 쪽 선은 검사·갱신하지 않음.
    pub async fn api_bracket_modify<Q: Quotes, I: DeskIo>(&mut self, qs: &mut Q, io: &mut I, q: &Query) -> Resp {
        if g(q, "confirm") != Some("1") {
            return bad(400, "confirm_required", "confirm=1 이 필요합니다");
        }
        if !io.exec_enabled() {
            return exec_off();
        }
        let Some((side, symbol)) = self.side_symbol(q) else { return bad(400, "bad_side", "side=LONG|SHORT · symbol 이 필요합니다") };
        let Lines::Some(ltp, lsl) = query_lines(q) else { return bad(400, "bad_lines", "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다") };
        let (key, asset) = (bracket_key(&symbol, &side), self.w.exec_asset_of(Some(&json!(symbol))).unwrap_or("eth"));
        let mut b = self.w.load().get(&key).cloned().unwrap_or(json!({}));
        if !truthy(Some(&b)) || !truthy(b.get("lines")) {
            return bad(409, "not_armed_lines", "선으로 들어간 무장이 없습니다");
        }
        if truthy(b.get("fired")) || self.w.exiting.contains(&key) || self.w.exit_slot(asset).is_none() {
            return fired_busy(); // 청산이 도는 중엔 익절을 다시 걸지 않는다(검토 B1)
        }
        let account = match io.account(false).await { Ok(a) => a, Err(e) => return server_error(&e) };
        let pos = account.get("positions").and_then(Value::as_array).and_then(|ps| ps.iter().find(|p| p.get("symbol").and_then(Value::as_str) == Some(symbol.as_str())
            && p.get("side").and_then(Value::as_str) == Some(side.as_str()) && fv(p.get("qty")).abs() > 0.0).cloned());
        if !(truthy(account.get("ok")) && pos.as_ref().is_some_and(|p| fv(p.get("entry_price")) > 0.0)) {
            return bad(409, "no_position", "그 포지션을 계좌에서 못 찾았습니다");
        }
        let (mut tp_off, mut sl_off) = (truthy(b.get("tp_off")), truthy(b.get("sl_off")));
        let (basis_unused, mut nb) = match self.lines_quote(qs, io, &symbol, asset).await {
            Err(e) => return bad(502, "quote_failed", &e),
            Ok((r, basis, filters)) => match Self::check_lines(&side, r, basis, &filters, (!tp_off).then_some(ltp), (!sl_off).then_some(lsl), query_slup(q)) {
                Err(m) => return bad(400, "bad_lines", &m),
                Ok(nb) => (basis, nb),
            },
        };
        let _ = basis_unused;
        if !sl_off && !truthy(b.get("sl_legs")) && (fv(nb.get("sl_level")) - fv(b.get("sl_level"))).abs() <= 1e-9 {
            // 2026-11-12 검토: 손절을 안 옮겼으면(익절만 끌기) 방향도 그대로
            nb.insert("sl_up".into(), json!(plan::sl_up_of(&side, &plan::Armed::from_value(&b))));
        }
        let mut sl_new: Option<Vec<Value>> = None;
        let slegs = b.get("sl_legs").and_then(Value::as_array).cloned().filter(|l| !l.is_empty());
        if let (Some(slegs), false) = (&slegs, sl_off) {
            if slegs.iter().any(|x| truthy(x.get("fired"))) {
                return bad(409, "fired", "손절 조각 청산이 진행 중입니다 — 끝난 뒤 다시");
            }
            let sli = g(q, "sli").unwrap_or("-1").trim().parse::<i64>().unwrap_or(-1);
            if (0..slegs.len() as i64).contains(&sli) && (fv(nb.get("sl_level")) - fv(slegs[sli as usize].get("level"))).abs() > 1e-9 {
                let lv = fv(nb.get("sl_level"));
                if slegs.iter().enumerate().any(|(j, x)| j as i64 != sli && (fv(x.get("level")) / lv - 1.0).abs() < 5e-4) {
                    return bad(409, "same_level", "다른 손절선과 같은 가격(±0.05%)입니다");
                }
                sl_new = Some(slegs.iter().enumerate().map(|(j, x)| if j as i64 == sli {
                    let mut m = obj(x);
                    for (k, n) in [("level", "sl_level"), ("price", "sl_price"), ("up", "sl_up")] {
                        m.insert(k.into(), nb.get(n).cloned().unwrap_or(Value::Null));
                    }
                    Value::Object(m)
                } else { x.clone() }).collect());
            }
            sl_off = true; // 아래 단일 손절 갱신은 건너뛴다(조각은 저장 때 sl_new 로)
        }
        let legs = b.get("tp_legs").and_then(Value::as_array).cloned().filter(|l| !l.is_empty());
        if let (Some(legs), false) = (&legs, tp_off) {
            let tpi = g(q, "tpi").unwrap_or("-1").trim().parse::<i64>().unwrap_or(-1);
            if (0..legs.len() as i64).contains(&tpi) && (fv(nb.get("tp_level")) - fv(legs[tpi as usize].get("level"))).abs() > 1e-9 {
                let lv = fv(nb.get("tp_level"));
                if legs.iter().enumerate().any(|(j, x)| j as i64 != tpi && (fv(x.get("level")) / lv - 1.0).abs() < 5e-4) {
                    return bad(409, "same_level", "다른 익절선과 같은 가격(±0.05%)입니다");
                }
                let nl: Vec<Value> = legs.iter().enumerate().map(|(j, x)| if j as i64 == tpi {
                    let mut m = obj(x);
                    m.insert("level".into(), nb["tp_level"].clone());
                    m.insert("price".into(), nb["tp_price"].clone());
                    Value::Object(m)
                } else { x.clone() }).collect();
                match self.ladder_apply(io, &key, &b, &symbol, &side, &nl).await {
                    Err(r) => return r,
                    Ok(cur2) => b = Value::Object(cur2), // 아래 손절 갱신·무장 확인은 방금 쓴 무장 기준
                }
            }
            tp_off = true;
        }
        let tp_moved = !tp_off && (fv(nb.get("tp_level")) - fv(b.get("tp_level"))).abs() > 1e-9;
        let (mut tp_res, mut warn) = (None::<Value>, Value::Null);
        if !tp_moved {
            let p = if truthy(b.get("tp_price")) { b["tp_price"].clone() } else { nb.get("tp_price").cloned().unwrap_or(Value::Null) };
            nb.insert("tp_price".into(), p);
        } else {
            // strict = 옛 주문을 다 못 지웠으면 새로 안 건다(같은 쪽 전량 익절이 둘 되지 않게, 검토 F2)
            let r = match io.place_bracket(&json!({"tp_price": nb["tp_price"]}), &symbol, &side, true).await { Ok(r) => r, Err(e) => return server_error(&e) };
            if r.get("cleared").is_none() {
                return bad(502, "tp_replace_failed", &format!("익절 지정가를 못 바꿨습니다(아무것도 안 바뀜): {}", r.get("reason").filter(|v| truthy(Some(v))).map_or("?".into(), py_str)));
            }
            if truthy(r["cleared"].get("errors")) {
                let es: Vec<String> = r["cleared"]["errors"].as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_default();
                warn = json!(format!("익절 지정가 정리 실패 — 새 익절선은 서버가 감시합니다. 거래소에 옛 익절 지정가가 남았을 수 있으니 확인하세요: {}", es.join("; ")));
            }
            tp_res = Some(r);
        }
        if !self.same_arm(&key, &b) {
            if tp_moved && tp_res.as_ref().is_some_and(|r| truthy(r.get("tp").and_then(|t| t.get("placed")))) {
                let _ = io.clear_bracket(&symbol, &side, false, false).await; // 방금 건 지정가가 고아로 남지 않게(검토 S2)
            }
            return rearmed();
        }
        let now = io.now();
        let mut st = self.w.load();
        let cur = st.get_mut(&key).and_then(Value::as_object_mut).unwrap();
        if !tp_off {
            cur.insert("tp_level".into(), nb["tp_level"].clone());
            cur.insert("tp_price".into(), nb["tp_price"].clone());
        }
        if !sl_off {
            for k in ["sl_level", "sl_price", "sl_name", "sl_up"] {
                cur.insert(k.into(), nb.get(k).cloned().unwrap_or(Value::Null));
            }
        }
        if let Some(sn) = &sl_new {
            crate::watch::set_sl_legs(cur, sn);
        }
        let m = json!({"at": now, "tp": cur.get("tp_level"), "sl": cur.get("sl_level")});
        cur.insert("watch_from".into(), json!(now));
        cur.insert("mod".into(), m);
        if let Some(r) = &tp_res {
            cur.insert("tp_resting".into(), json!(truthy(r.get("tp").and_then(|t| t.get("placed")))));
            let mut placed = obj(cur.get("placed").unwrap_or(&Value::Null));
            placed.insert("tp".into(), r.get("tp").cloned().unwrap_or(Value::Null));
            cur.insert("placed".into(), Value::Object(placed));
        }
        let body = json!({"ok": true, "tp_level": cur.get("tp_level"), "sl_level": cur.get("sl_level"), "sl_price": cur.get("sl_price"), "sl_up": cur.get("sl_up"),
                          "tp_resting": cur.get("tp_resting"), "tp_legs": cur.get("tp_legs"), "sl_legs": cur.get("sl_legs"),
                          "tp": tp_res.as_ref().map_or(Value::Null, |r| r.get("tp").cloned().unwrap_or(Value::Null)), "warn": warn});
        let (new_tp, new_sl) = (cur.get("tp_level").cloned().unwrap_or(Value::Null), cur.get("sl_level").cloned().unwrap_or(Value::Null));
        if let Err(e) = self.w.save(&st) {
            return server_error(&e);
        }
        let pr = |v: Option<&Value>| v.map_or("None".into(), crate::sign::py_repr);
        self.w.log(format!("bracket {side}: 선 수정(사용자) 익절 {}→{} 손절 {}→{} · 익절 지정가 {} {}", pr(b.get("tp_level")), pr(Some(&new_tp)), pr(b.get("sl_level")), pr(Some(&new_sl)),
                           if tp_moved { "다시 걸음" } else { "그대로" }, tp_res.as_ref().map_or("None".into(), |r| pr(r.get("tp")))));
        resp(200, body)
    }
}
