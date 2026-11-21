#![allow(dead_code)]
//! 여섯째 조각 통합: 접수(intake) → 실행기(runner::run_task → exec → 가짜 거래소) → 익절 걸기·무장(watch::arm_bracket) → 감시 틱(선 손절 닿음) → 청산(run_exit_into) → 무장 해제.
//! 거래소 = tests/common/fake.rs(127.0.0.1 도 안 쓰는 메모리 가짜) · 손절·익절 함수 = 이 파일의 `Io`(무엇이 불렸는지 적는다). 실제 거래소·네트워크 없음.

include!("common/fake.rs");
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::{FakeDeskIo, FakeQuotes, T0};
use rtorder::assemble::Desk;
use rtorder::intake::{Query, Task};
use rtorder::plan::{build_exit_plan, ExitInput, Filters};
use rtorder::runner::run_task;
use rtorder::watch::*;
use serde_json::{json, Map, Value};

struct Io {
    fx: Fake,
    t: f64,
    qty: f64,
    sec: Vec<(f64, f64)>,
    calls: Vec<String>,
    tp_fail: bool,
    exit_vol: Option<f64>,
}
impl WatchIo for Io {
    fn now(&self) -> f64 {
        self.t
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
        Ok(json!({"ok": true, "generated_at": iso_utc(self.t), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": self.qty}]}))
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        self.account().await
    }
    async fn bars(&mut self, _a: &str) -> Result<Vec<Value>, String> {
        Ok(vec![])
    }
    fn sec_prices(&mut self, _a: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        Ok(Some(self.sec.clone()))
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, _a: bool, only_tp: bool) -> Result<Value, String> {
        self.calls.push(format!("clear_bracket {s} {p} only_tp={only_tp}"));
        Ok(json!({"cancelled": 1, "errors": []}))
    }
    async fn exit_plan(&mut self, p: &str, f: f64, _a: &str, s: &str) -> Result<Value, Value> {
        self.calls.push(format!("exit_plan {p} {f} {s}"));
        let fl = Filters { step: 0.001, tick: 0.01, min_qty: 0.001, min_notional: 5.0 };
        let x = build_exit_plan(&ExitInput { position_side: "LONG", position_qty: self.qty, best_bid: 2399.9, best_ask: 2400.0, filters: fl, symbol: "ETHUSDC",
                                             entry_price: 2500.0, mark_price: 2400.0, vol_bpm: self.exit_vol, fraction: f }).unwrap();
        Ok(serde_json::to_value(x).unwrap())
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Map<String, Value>) -> Result<(), String> {
        self.calls.push("run_exit".into());
        let t = self.t;
        let r = run_exit_into(&mut self.fx, plan, slot, || iso_utc(t)).await;
        if matches!(slot.get("phase").and_then(Value::as_str), Some("filled_maker" | "filled_taker")) {
            self.qty = 0.0;
        }
        r
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        self.calls.push(format!("place_bracket {s} {p} tp={} strict={strict}", spec["tp_price"]));
        Ok(json!({"qty": self.qty, "cleared": {"cancelled": 0, "errors": []}, "tp": {"placed": !self.tp_fail, "price": spec["tp_price"]}, "placed": !self.tp_fail}))
    }
    async fn place_tp_ladder(&mut self, _: &Value, _: &str, _: &str) -> Result<Value, String> {
        unreachable!()
    }
    async fn place_add_order(&mut self, _: &str, _: &str, _: f64, _: f64) -> Result<Value, String> {
        unreachable!("선 진입은 물타기를 안 건다")
    }
    async fn cancel_wait_orphans(&mut self, _: &[String]) -> Result<Value, String> {
        unreachable!()
    }
    async fn cancel_exit_orders(&mut self, s: &str, p: &str) -> Result<Value, String> {
        self.calls.push(format!("cancel_exit_orders {s} {p}"));
        Ok(json!({"cancelled": 0, "errors": []}))
    }
}

fn q(kv: &[(&str, &str)]) -> Query {
    kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// 선(익절 2600 · 손절 2400) 진입 접수 → 실행 → 무장. 돌려주기 = (Desk, Io, 폴더).
fn enter(name: &str, fx: Fake, tp_fail: bool) -> (Desk, Io, desk_io::TmpDir) {
    let inp = json!({"account": {"ok": true, "positions": [], "open_orders": [], "leverage_by_symbol": {"ETHUSDC": 10.0},
                                 "balances": {"USDC": {"margin": 1000.0, "available": 800.0, "initial_margin": 0.0}}},
                     "sec": {"eth": [[T0 - 1.0, 2499.6]]}, "books": {"ETHUSDC": {"bidPrice": "2500.0", "askPrice": "2500.1"}, "ETHUSDT": {"bidPrice": "2500.0", "askPrice": "2500.1"}}});
    let (mut desk, dir) = desk_io::desk_from(&inp);
    let mut dio = FakeDeskIo::new(&inp, desk.w.dir.clone());
    let mut qs = FakeQuotes::new(&inp);
    let r = desk_io::block_on(desk.api_entry_submit(&mut qs, &mut dio, &q(&[("side", "LONG"), ("confirm", "1"), ("lev", "10"), ("pct", "10"), ("tp", "2600"), ("sl", "2400")])));
    assert_eq!(r.status, 200, "{name}: {}", r.body);
    let task = r.task.into_iter().next().unwrap();
    assert!(matches!(task, Task::Entry { .. }));
    let mut io = Io { fx: Fake::default(), t: T0, qty: 0.0, sec: vec![(T0 - 1.0, 2499.6)], calls: vec![], tp_fail, exit_vol: Some(5.0) };
    let mut ex = fx;
    block_on(run_task(&mut desk, &mut ex, &mut io, &mut dio, &mut qs, task));
    io.qty = desk.w.entry.get("filled").and_then(Value::as_f64).unwrap_or(0.0);
    io.fx = Fake { fill_after: 1, ..Default::default() };
    (desk, io, dir)
}

#[test]
fn entry_fill_arms_lines_then_stop_touch_exits_and_disarms() {
    let (mut desk, mut io, _d) = enter("ok", Fake { fill_after: 1, ..Default::default() }, false);
    assert_eq!(desk.w.entry["phase"], "filled_maker", "{:?}", desk.w.entry);
    assert!(desk.running.is_empty(), "작업이 끝나면 칸 이름을 뺀다(G12)");
    assert!(io.calls[0].starts_with("place_bracket ETHUSDC LONG tp=2600"), "{:?}", io.calls);
    let b = desk.w.load()["ETHUSDC:LONG"].clone();
    assert_eq!((b["lines"].clone(), b["tp_resting"].clone(), b["sl_level"].clone()), (json!(true), json!(true), json!(2400.0)));
    // 1초 체결가가 손절선 2400 에 닿음 → 우리 익절·물타기 지정가 먼저 지우고(원본 clear_bracket 전체) → 청산(메이커) → 다 닫힘 → 무장 내림
    io.t = T0 + 10.0;
    io.sec = vec![(T0 + 5.0, 2401.0), (T0 + 9.0, 2400.0)];
    io.calls.clear();
    block_on(desk.w.bracket_tick(&mut io)).unwrap();
    assert_eq!(io.calls, ["clear_bracket ETHUSDC LONG only_tp=false", "exit_plan LONG 1 ETHUSDC", "run_exit", "clear_bracket ETHUSDC LONG only_tp=false"], "{:?}", desk.w.logs);
    assert_eq!(desk.w.entry["phase"], "filled_maker");
    assert!(io.fx.placed.iter().all(|o| o.side == "SELL" && o.position_side == "LONG"));
    assert!(desk.w.load().is_empty(), "다 닫혔으면 무장을 내린다");
}

#[test]
fn tp_place_failure_leaves_server_watch_and_tp_touch_closes() {
    // 실패 주입: 익절 지정가를 못 걸었다 → tp_resting False(서버가 1초 닿음으로 닫는다, B7)
    let (mut desk, mut io, _d) = enter("tpfail", Fake { fill_after: 1, ..Default::default() }, true);
    assert_eq!(desk.w.load()["ETHUSDC:LONG"]["tp_resting"], json!(false));
    io.t = T0 + 10.0;
    io.sec = vec![(T0 + 9.0, 2600.5)];
    io.calls.clear();
    block_on(desk.w.bracket_tick(&mut io)).unwrap();
    assert!(io.calls.iter().any(|c| c == "run_exit"), "{:?}", io.calls);
    assert!(desk.w.load().is_empty());
}

#[test]
fn exit_cancel_unconfirmed_errors_then_refire_cleans_up_first() {
    // 실패 주입: 청산 따라가기 중 취소 확인 실패 → error · 무장 유지 30초 → 재발동 전 cancel_exit_orders(결함 수정 규칙)
    let (mut desk, mut io, _d) = enter("refire", Fake { fill_after: 1, ..Default::default() }, false);
    io.fx = Fake { drift: true, fail: vec!["cancel"], ..Default::default() };
    io.t = T0 + 10.0;
    io.sec = vec![(T0 + 9.0, 2399.0)];
    block_on(desk.w.bracket_tick(&mut io)).unwrap();
    assert_eq!(desk.w.entry["phase"], "error");
    let b = desk.w.load()["ETHUSDC:LONG"].clone();
    assert_eq!(b["retry_after"].as_f64().unwrap(), T0 + 10.0 + RETRY_SEC);
    io.fx = Fake { fill_after: 1, ..Default::default() };
    io.t = T0 + 45.0;
    io.sec.push((T0 + 44.0, 2398.0));
    io.calls.clear();
    block_on(desk.w.bracket_tick(&mut io)).unwrap();
    assert_eq!(io.calls[0], "cancel_exit_orders ETHUSDC LONG", "재발동 전에 앞선 청산 주문부터 정리 {:?}", io.calls);
    assert!(io.calls.iter().any(|c| c == "run_exit"));
    assert!(desk.w.load().is_empty());
}

#[test]
fn entry_unfilled_peg_deadline_goes_market_m1_then_arms() {
    // 따라가기 120초 안 찬 잔량 = 시장가(M1 · 원본 동작 유지) → filled_taker → 무장
    let (desk, io, _d) = enter("m1", Fake::default(), false);
    assert_eq!(desk.w.entry["phase"], "filled_taker", "{:?}", desk.w.entry);
    assert!(desk.w.entry["taker_order_id"].is_i64());
    assert!(io.calls.iter().any(|c| c.starts_with("place_bracket")), "시장가로 든 몫에도 익절·손절 무장");
}
