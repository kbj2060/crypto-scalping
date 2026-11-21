#![allow(dead_code)]
//! 감시 → 실제 집행(`watch::run_exit_into` → `exec::run_exit`) → 가짜 거래소(tests/common/fake.rs). 2026-11-20 셋째 조각.
//! 손절 발동 뒤 청산이 시장가 분기(규칙 문서 §9 M6·M7·M9)에 닿으면 원본처럼 시장가로 닫는다(2026-11-21 사용자 결정 «원본 동작 유지»).

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
    async fn cancel_exit_orders(&mut self, _s: &str, _p: &str) -> Result<Value, String> {
        Ok(serde_json::json!({"cancelled": 0, "errors": []}))
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
fn stop_exit_deadline_remainder_goes_market_m9_and_disarms() {
    // 체결 안 됨 → 계획 마감 뒤 잔량 = 시장가(M9 · 2026-11-21 사용자 결정 «원본 동작 유지») → 다 닫힘 → 정리·무장 내림
    let (w, io, dir) = fire(exit_plan_v(Some(5.0), 5.0), Fake::default(), "m9");
    assert_eq!(w.entry["phase"], "filled_taker");
    assert_eq!(w.entry["filled"], json!(2.0), "D3 원본 그대로: 잔량 시장가 뒤 filled = 전량");
    assert_eq!(w.entry["taker_qty"], json!(2.0));
    assert!(io.fx.live().is_empty(), "시장가 전에 지정가를 다 지웠다");
    assert_eq!(io.fx.markets, vec![("SELL".to_string(), "LONG".to_string(), 2.0)]);
    assert!(!io.fx.placed.is_empty() && io.fx.placed.iter().all(|o| o.side == "SELL"), "지정가 따라가기 먼저");
    assert_eq!(io.calls, ["exit_plan", "run_exit", "clear_bracket"]);
    assert!(bracket_load(&dir).is_empty());
}

#[test]
fn stop_exit_high_vol_plan_is_market_m6_closes_at_once() {
    let p = exit_plan_v(Some(plan::EXIT_TAKER_VOL_BPM), 5.0);
    assert_eq!(p["type"], "MARKET", "변동성 ≥30bp/√분 = 계획부터 시장가(M6, 원본 값 그대로)");
    let (w, io, dir) = fire(p, Fake::default(), "m6");
    assert_eq!(w.entry["phase"], "filled_taker");
    assert!(io.fx.placed.is_empty(), "지정가 없이 바로 시장가");
    assert_eq!(io.fx.markets.len(), 1);
    assert!(bracket_load(&dir).is_empty());
}

#[test]
fn stop_exit_dust_position_is_market_m7() {
    let p = exit_plan_v(Some(5.0), 1e9);
    assert_eq!(p["type"], "MARKET");
    assert!(p["market_reason"].as_str().unwrap().contains("최소"), "먼지 사유(M7)");
    let (w, io, _dir) = fire(p, Fake::default(), "m7");
    assert_eq!(w.entry["phase"], "filled_taker");
    assert!(io.fx.placed.is_empty() && io.fx.markets.len() == 1);
}

#[test]
fn stop_exit_market_leg_error_is_taker_failed_and_retries_after_30s() {
    let fx = Fake { market_err: Some("400 [-2019] Margin is insufficient.".into()), ..Default::default() };
    let p = exit_plan_v(Some(plan::EXIT_TAKER_VOL_BPM), 5.0);
    let (w, io, dir) = fire(p, fx, "mfail");
    assert_eq!(w.entry["phase"], "taker_failed");
    let b = &bracket_load(&dir)["ETHUSDC:LONG"];
    assert_eq!(b["retry_after"].as_f64().unwrap(), io.now() + RETRY_SEC);
    assert_eq!(b["last_error"], json!("400 [-2019] Margin is insufficient."));
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
