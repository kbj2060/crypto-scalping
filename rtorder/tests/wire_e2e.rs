#![allow(dead_code)]
//! 넷째 조각 통합: 감시(watch.rs) → 실제 계좌 읽기(account.rs, 가짜 REST) → 실제 청산 조립(assemble.rs, 가짜 시세) → 실제 집행(exec.rs, 가짜 거래소).
//! 5분봉·1초 가격·손절/익절 거래소 함수만 가짜 `WatchIo`(ops). 실제 거래소·네트워크 없음.

include!("common/fake.rs");
#[path = "common/desk_io.rs"]
mod desk_io;

use desk_io::{FakeQuotes, FakeRest};
use rtorder::account::UserStream;
use rtorder::userstream::Hub;
use rtorder::watch::*;
use rtorder::wire::Wired;
use serde_json::{json, Map, Value};

const T0: f64 = 1_800_000_000.0;

struct Ops {
    bars: Vec<Value>,
    calls: Vec<String>,
}
impl WatchIo for Ops {
    fn now(&self) -> f64 {
        T0
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
        unreachable!("Wired 가 계좌를 맡는다")
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        unreachable!()
    }
    async fn bars(&mut self, _a: &str) -> Result<Vec<Value>, String> {
        self.calls.push("bars".into());
        Ok(self.bars.clone())
    }
    fn sec_prices(&mut self, _a: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        Ok(None)
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, _a: bool, _b: bool) -> Result<Value, String> {
        self.calls.push(format!("clear_bracket {s} {p}"));
        Ok(json!({"cancelled": 1, "errors": []}))
    }
    async fn exit_plan(&mut self, _p: &str, _f: f64, _a: &str, _s: &str) -> Result<Value, Value> {
        unreachable!("Wired 가 조립한다")
    }
    async fn run_exit(&mut self, _p: &Value, _s: &mut Map<String, Value>) -> Result<(), String> {
        unreachable!("Wired 가 집행한다")
    }
    async fn place_bracket(&mut self, _: &Value, _: &str, _: &str, _: bool) -> Result<Value, String> {
        unreachable!()
    }
    async fn place_tp_ladder(&mut self, _: &Value, _: &str, _: &str) -> Result<Value, String> {
        unreachable!()
    }
    async fn cancel_wait_orphans(&mut self, _: &[String]) -> Result<Value, String> {
        unreachable!()
    }
    async fn cancel_exit_orders(&mut self, _s: &str, _p: &str) -> Result<Value, String> {
        Ok(serde_json::json!({"cancelled": 0, "errors": []}))
    }
}

fn routes(risk_ok: bool, qty: &str) -> Value {
    let bal = json!({"multiAssetsMargin": false, "totalWalletBalance": "1", "totalMarginBalance": "1", "totalInitialMargin": "0", "totalMaintMargin": "0",
                     "availableBalance": "1", "totalUnrealizedProfit": "0",
                     "assets": [{"asset": "USDC", "walletBalance": "1000", "availableBalance": "900", "marginBalance": "1000", "initialMargin": "100", "maintMargin": "5", "unrealizedProfit": "0"}]});
    let risk = if risk_ok {
        json!([{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": qty, "entryPrice": "2400", "markPrice": "2450", "liquidationPrice": "1800",
                "leverage": "10", "notional": "980", "unRealizedProfit": "20", "updateTime": 1799999000000i64}])
    } else {
        json!({"__status__": 418, "__body__": {"code": -1003, "msg": "banned until 1800000999000"}})
    };
    json!({"/fapi/v2/account": bal, "/fapi/v2/positionRisk": risk, "/fapi/v1/userTrades": [], "/fapi/v1/openOrders": []})
}

/// 규칙 무장(5분봉 종가 손절) + 마감 봉 종가가 손절선 아래.
fn setup(name: &str, vol: Option<f64>, risk_ok: bool) -> (Watcher, Wired<Ops, FakeRest, FakeQuotes, Fake>, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("rtorder_wire_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut st = Map::new();
    st.insert("ETHUSDC:LONG".into(), json!({"symbol": "ETHUSDC", "side": "LONG", "rule": true, "sl_price": 2375.1, "sl_level": 2374.5, "backstop_price": 2375.1,
                                            "tp_price": null, "armed_at": T0 - 2000.0}));
    bracket_save(&dir, &st).unwrap();
    let ops = Ops { bars: vec![json!({"time": T0 - 1200.0, "close": 2380.0}), json!({"time": T0 - 900.0, "close": 2370.0})], calls: vec![] };
    let rest = FakeRest::new(&json!({"routes": routes(risk_ok, "0.4"), "t": T0}));
    let mut q = json!({});
    if let Some(v) = vol {
        q["vol"] = json!({"ETHUSDC": v});
    }
    let quotes = FakeQuotes::new(&q);
    (Watcher::new(&dir, "ETHUSDC"), Wired::new(ops, rest, quotes, Fake { fill_after: 2, ..Default::default() }, "ETHUSDC"), dir)
}

fn count(rest: &FakeRest, path: &str) -> usize {
    rest.calls.iter().filter(|c| c[0] == path).count()
}

#[test]
fn rule_stop_fires_assembles_from_fresh_rest_account_and_exits_with_maker() {
    let (mut w, mut io, dir) = setup("rest", Some(5.0), true);
    block_on(w.bracket_tick(&mut io)).unwrap();
    assert_eq!(w.entry["phase"], "filled_maker", "{:?} {:?}", w.entry, w.logs);
    assert_eq!(w.entry["filled"], json!(0.4));
    // 계좌: 틱의 캐시 읽기 1 + 청산 조립의 새로 읽기 1(C1 fresh) = REST 2번
    assert_eq!(count(&io.rest, "/fapi/v2/account"), 2);
    assert_eq!(io.fresh_source, ["rest"]);
    // 시세: 원본처럼 호가·필터·변동성 + (감시 조립은 for_order 를 안 넘겨) 펀딩·수수료까지
    let names: Vec<&str> = io.quotes.calls.iter().map(|c| c[0].as_str().unwrap()).collect();
    assert_eq!(names, ["book", "filters", "vol", "funding", "fees"]);
    assert!(io.ex.placed.iter().all(|o| o.side == "SELL" && o.position_side == "LONG" && o.quantity <= 0.4 && o.symbol == "ETHUSDC"));
    assert!(io.ex.live().is_empty());
    assert_eq!(io.ops.calls, ["bars", "clear_bracket ETHUSDC LONG"]);
    assert!(bracket_load(&dir).is_empty(), "다 닫혔으면 무장을 내린다");
}

#[test]
fn stream_account_skips_the_fresh_rest_read_and_uses_stream_quantity() {
    let (mut w, mut io, _dir) = setup("stream", Some(5.0), true);
    let hub = Arc::new(Hub::default());
    *hub.us.lock().unwrap() = UserStream { connected: true, epoch: 3, ..Default::default() };
    io.stream = Some(hub.clone());
    // 틱 시작 전에 캐시를 한 번 채워 둔다(대조 = 기준 스냅숏) · 그 뒤 스트림 사건: 부분 청산으로 0.4 → 0.25
    block_on(io.account()).unwrap();
    hub.stream_account.lock().unwrap().apply(&json!({"e": "ACCOUNT_UPDATE", "a": {"P": [{"s": "ETHUSDC", "ps": "LONG", "pa": "0.25", "ep": "2400", "up": "12"}]}}));
    block_on(w.bracket_tick(&mut io)).unwrap();
    assert_eq!(io.fresh_source, ["stream"]);
    assert_eq!(count(&io.rest, "/fapi/v2/account"), 1, "청산 조립이 REST 계좌를 안 읽는다(왕복 13 요청 · 무게 ~50 이 빠진다)");
    assert_eq!(w.entry["quantity"], json!(0.25), "스트림이 말한 수량으로 닫는다");
    assert_eq!(w.entry["phase"], "filled_maker");
}

#[test]
fn stream_down_falls_back_to_rest() {
    let (mut w, mut io, _dir) = setup("down", Some(5.0), true);
    let hub = Arc::new(Hub::default());
    *hub.us.lock().unwrap() = UserStream { connected: false, epoch: 3, ..Default::default() };
    io.stream = Some(hub);
    block_on(w.bracket_tick(&mut io)).unwrap();
    assert_eq!(io.fresh_source, ["stream_down"]);
    assert_eq!(count(&io.rest, "/fapi/v2/account"), 2);
    assert_eq!(w.entry["quantity"], json!(0.4));
}

#[test]
fn high_vol_exit_plan_is_market_and_closes_with_one_market_order() {
    // M6(2026-11-21 사용자 결정 «원본 동작 유지»): 변동성 ≥30 → 계획부터 MARKET → 지정가 없이 시장가 한 번 → 다 닫힘 → 정리·무장 내림
    let (mut w, mut io, dir) = setup("m6", Some(35.0), true);
    block_on(w.bracket_tick(&mut io)).unwrap();
    assert_eq!(w.entry["phase"], "filled_taker");
    assert!(io.ex.placed.is_empty());
    assert_eq!(io.ex.markets, vec![("SELL".to_string(), "LONG".to_string(), 0.4)]);
    assert!(bracket_load(&dir).is_empty());
}

#[test]
fn account_read_failure_in_exit_plan_keeps_stop_and_retries() {
    let (mut w, mut io, dir) = setup("503", Some(5.0), false);
    // 틱의 계좌는 캐시(정상) -- 청산 조립의 새로 읽기에서 positionRisk 418 → 503 «못 읽음 = 모름»(P2·B17)
    *io.cache.lock().unwrap() = Some((T0, json!({"ok": true, "generated_at": iso_utc(T0), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 0.4}]})));
    block_on(w.bracket_tick(&mut io)).unwrap();
    let b = &bracket_load(&dir)["ETHUSDC:LONG"];
    assert!(b["last_error"].as_str().unwrap().contains("계좌 조회 실패"), "{b}");
    assert_eq!(b["retry_after"].as_f64().unwrap(), T0 + RETRY_SEC);
    assert!(io.ex.placed.is_empty());
    assert!(io.ops.calls.iter().all(|c| !c.starts_with("clear_bracket")), "조회 실패를 «닫힘»으로 보고 정리하지 않는다");
}
