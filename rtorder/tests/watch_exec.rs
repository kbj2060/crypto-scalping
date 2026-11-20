#![allow(dead_code)]
//! 감시 → 실제 집행(`watch::run_exit_into` → `exec::run_exit`) → 가짜 거래소(tests/common/fake.rs). 2026-11-20 셋째 조각.
//! 🔴손절 발동 뒤 청산이 시장가 분기(규칙 문서 §9)에 닿는 자리: 파이썬은 시장가로 닫는다 -- Rust 는 멈추고(`needs_market_decision`) 무장에 `market_wait` 를 적고
//! 다시 쏘지 않는다(사용자 결정 대기). 시장가 주문은 거래소에 한 건도 안 나간다(trait 에 시장가 메서드가 없다).

include!("common/fake.rs");

use rtorder::plan::{self, build_exit_plan, ExitInput, Filters};
use rtorder::watch::*;
use serde_json::{json, Map, Value};

const T0: f64 = 1_800_000_000.0;

struct ExecIo {
    fx: Fake,
    plan: Value,
    sec: Vec<(f64, f64)>,
    calls: Vec<String>,
}

impl WatchIo for ExecIo {
    fn now(&self) -> f64 {
        T0 + self.fx.t
    }
    fn exec_enabled(&self) -> bool {
        true
    }
    fn order_locked(&self) -> bool {
        false
    }
    async fn nap(&mut self, _sec: f64) -> bool {
        false
    }
    async fn account(&mut self) -> Result<Value, String> {
        Ok(json!({"ok": true, "generated_at": iso_utc(T0), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 2.0}]}))
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        unreachable!()
    }
    async fn bars(&mut self, _a: &str) -> Result<Vec<Value>, String> {
        Ok(vec![])
    }
    fn sec_prices(&mut self, _a: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        Ok(Some(self.sec.clone()))
    }
    async fn clear_bracket(&mut self, _s: &str, _p: &str, _a: bool, _b: bool) -> Result<Value, String> {
        self.calls.push("clear_bracket".into());
        Ok(json!({"cancelled": 1}))
    }
    async fn exit_plan(&mut self, _p: &str, _f: f64, _a: &str, _s: &str) -> Result<Value, Value> {
        self.calls.push("exit_plan".into());
        Ok(self.plan.clone())
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Map<String, Value>) -> Result<(), String> {
        self.calls.push("run_exit".into());
        let t = T0 + self.fx.t;
        run_exit_into(&mut self.fx, plan, slot, || iso_utc(t)).await
    }
    async fn place_bracket(&mut self, _: &Value, _: &str, _: &str, _: bool) -> Result<Value, String> {
        unreachable!()
    }
    async fn place_tp_ladder(&mut self, _: &Value, _: &str, _: &str) -> Result<Value, String> {
        unreachable!()
    }
    async fn place_add_order(&mut self, _: &str, _: &str, _: f64, _: f64) -> Result<Value, String> {
        unreachable!()
    }
    async fn cancel_wait_orphans(&mut self, _: &[String]) -> Result<Value, String> {
        unreachable!()
    }
}

fn exit_plan_v(vol: Option<f64>, min_notional: f64) -> Value {
    let f = Filters { step: 0.001, tick: 0.01, min_qty: 0.001, min_notional };
    let p = build_exit_plan(&ExitInput { position_side: "LONG", position_qty: 2.0, best_bid: 2499.99, best_ask: 2500.0, filters: f, symbol: "ETHUSDC",
                                         entry_price: 2600.0, mark_price: 2500.0, vol_bpm: vol, fraction: 1.0 }).unwrap();
    serde_json::to_value(p).unwrap()
}

/// 선 손절 닿음 → 감시 → run_exit_into → 실제 지정가 따라가기(가짜 거래소). 돌려주기 = (감시, 거래소, 무장 폴더).
fn fire(plan: Value, fx: Fake, name: &str) -> (Watcher, ExecIo, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("rtorder_watch_exec_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut st = Map::new();
    st.insert("ETHUSDC:LONG".into(), json!({"symbol": "ETHUSDC", "side": "LONG", "lines": true, "sl_level": 2480.0, "sl_price": 2480.0, "tp_level": 2550.0,
                                            "tp_price": 2550.0, "tp_resting": false, "fired": true, "armed_at": T0 - 600.0}));
    bracket_save(&dir, &st).unwrap();
    let mut w = Watcher::new(&dir, "ETHUSDC");
    let mut io = ExecIo { fx, plan, sec: vec![(T0, 2470.0)], calls: vec![] };
    block_on(w.bracket_tick(&mut io)).unwrap();
    (w, io, dir)
}

#[test]
fn stop_exit_fills_with_maker_and_disarms() {
    let fx = Fake { fill_after: 2, ..Default::default() };
    let (w, io, dir) = fire(exit_plan_v(Some(5.0), 5.0), fx, "fill");
    assert_eq!(w.entry["phase"], "filled_maker");
    assert_eq!(w.entry["filled"], json!(2.0));
    assert_eq!(io.calls, ["exit_plan", "run_exit", "clear_bracket"]);
    assert!(io.fx.placed.iter().all(|o| o.side == "SELL" && o.position_side == "LONG" && o.quantity <= 2.0), "청산은 SELL·LONG·수량 이하");
    assert!(io.fx.live().is_empty());
    assert!(bracket_load(&dir).is_empty());
}

#[test]
fn stop_exit_deadline_remainder_is_market_branch_m9_and_waits() {
    // 체결이 안 됨 → 계획 마감(변동성 5bp → 120초 상한 쪽) 뒤 잔량: 파이썬 = 시장가(M9) · Rust = 멈춤
    let (mut w, mut io, dir) = fire(exit_plan_v(Some(5.0), 5.0), Fake::default(), "m9");
    assert_eq!(w.entry["phase"], "needs_market_decision");
    assert_eq!(w.entry["market_branch"], "M9");
    assert_eq!(w.entry["market_qty"], json!(2.0));
    assert!(io.fx.live().is_empty(), "살아 있는 지정가 없이 멈춘다");
    assert!(!io.fx.placed.is_empty() && io.fx.placed.iter().all(|o| o.side == "SELL"), "지정가 따라가기는 했다");
    let b = &bracket_load(&dir)["ETHUSDC:LONG"];
    assert_eq!(b["market_wait"]["branch"], "M9");
    assert!(b["last_error"].as_str().unwrap().starts_with("시장가 결정 대기 M9"));
    assert!(w.logs.last().unwrap().ends_with("-> needs_market_decision"), "{:?}", w.logs);
    // 다음 틱: 여전히 닿아 있어도 다시 쏘지 않는다(조용히 지정가로 바꾸지 않는다 -- 결정 대기)
    let placed = io.fx.placed.len();
    io.calls.clear();
    block_on(w.bracket_tick(&mut io)).unwrap();
    assert!(io.calls.is_empty(), "결정 대기 중 다시 쏨 {:?}", io.calls);
    assert_eq!(io.fx.placed.len(), placed);
}

#[test]
fn stop_exit_high_vol_plan_is_market_m6_no_order_sent() {
    let p = exit_plan_v(Some(plan::EXIT_TAKER_VOL_BPM), 5.0);
    assert_eq!(p["type"], "MARKET", "변동성 ≥30bp/√분 = 계획부터 시장가(M6, 원본 값 그대로)");
    let (w, io, dir) = fire(p, Fake::default(), "m6");
    assert_eq!(w.entry["phase"], "needs_market_decision");
    assert_eq!(w.entry["market_branch"], "M6/M7/M8");
    assert_eq!(w.entry["market_qty"], json!(2.0));
    assert!(io.fx.placed.is_empty(), "시장가 계획이면 거래소에 아무것도 안 나간다");
    assert_eq!(bracket_load(&dir)["ETHUSDC:LONG"]["market_wait"]["branch"], "M6/M7/M8");
}

#[test]
fn stop_exit_dust_position_is_market_m7_no_order_sent() {
    let p = exit_plan_v(Some(5.0), 1e9);
    assert_eq!(p["type"], "MARKET");
    assert!(p["market_reason"].as_str().unwrap().contains("최소"), "먼지 사유(M7)");
    let (w, io, _dir) = fire(p, Fake::default(), "m7");
    assert_eq!(w.entry["phase"], "needs_market_decision");
    assert_eq!(w.entry["market_branch"], "M6/M7/M8");
    assert!(io.fx.placed.is_empty());
}

#[test]
fn stop_exit_cancel_unconfirmed_is_error_and_retries_after_30s() {
    // 취소 확인 실패 = phase error(사람에게) · 무장 유지 · 30초 뒤 다시(원본 그대로 -- «살아 있을 수 있는 주문» 위에 다시 건다, 진행 기록 4 남은 위험)
    let fx = Fake { drift: true, fail: vec!["cancel"], ..Default::default() };
    let (w, io, dir) = fire(exit_plan_v(Some(5.0), 5.0), fx, "err");
    assert_eq!(w.entry["phase"], "error");
    assert!(w.entry["error"].as_str().unwrap().contains("취소를 확인하지 못했습니다"));
    let b = &bracket_load(&dir)["ETHUSDC:LONG"];
    assert_eq!(b["retry_after"].as_f64().unwrap(), io.now() + RETRY_SEC);
    assert_eq!(b["last_error"], w.entry["error"]);
    assert!(b.get("market_wait").is_none());
}
