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
    *io.acct.cache.lock().unwrap() = Some((T0, json!({"ok": true, "generated_at": iso_utc(T0), "positions": [{"symbol": "ETHUSDC", "side": "LONG", "qty": 0.4}]})));
    block_on(w.bracket_tick(&mut io)).unwrap();
    let b = &bracket_load(&dir)["ETHUSDC:LONG"];
    assert!(b["last_error"].as_str().unwrap().contains("계좌 조회 실패"), "{b}");
    assert_eq!(b["retry_after"].as_f64().unwrap(), T0 + RETRY_SEC);
    assert!(io.ex.placed.is_empty());
    assert!(io.ops.calls.iter().all(|c| !c.starts_with("clear_bracket")), "조회 실패를 «닫힘»으로 보고 정리하지 않는다");
}

/// 집행 전송을 감싸 첫 기다림(nap)에서 문이 열릴 때까지 멈춘다 -- 감시 발동 청산이 «도는 중»인 순간을 만든다.
struct Gated {
    inner: Fake,
    entered: std::rc::Rc<tokio::sync::Notify>,
    open: std::rc::Rc<tokio::sync::Notify>,
    held: bool,
}
impl rtorder::exec::Exchange for Gated {
    async fn place_limit_gtx(&mut self, o: &rtorder::exec::LimitOrder) -> rtorder::exec::Reply {
        self.inner.place_limit_gtx(o).await
    }
    async fn place_market(&mut self, c: &rtorder::exec::Common<'_>, q: f64) -> rtorder::exec::Reply {
        self.inner.place_market(c, q).await
    }
    async fn get_order(&mut self, c: &rtorder::exec::Common<'_>, r: &rtorder::exec::OrderRef) -> rtorder::exec::Reply {
        self.inner.get_order(c, r).await
    }
    async fn cancel_order(&mut self, c: &rtorder::exec::Common<'_>, r: &rtorder::exec::OrderRef) -> rtorder::exec::Reply {
        self.inner.cancel_order(c, r).await
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.inner.book(s).await
    }
    fn now(&self) -> f64 {
        self.inner.now()
    }
    fn wall_ms(&self) -> i64 {
        self.inner.wall_ms()
    }
    async fn nap(&mut self, s: f64) {
        if !std::mem::replace(&mut self.held, true) {
            self.entered.notify_one();
            self.open.notified().await;
        }
        self.inner.nap(s).await
    }
    async fn sleep(&mut self, s: f64) {
        self.inner.sleep(s).await
    }
}

#[tokio::test(flavor = "current_thread")]
async fn watch_exit_runs_outside_desk_lock_status_answers_and_same_coin_exit_is_409() {
    // 원본: 감시가 칸을 «submitting» 으로 바꾼 뒤 run_exit(최대 ~2분)를 기다리는 사이 다른 접수가 돈다 -- 상태 조회는 바로, 같은 코인 청산은 칸이 바빠 409.
    // 2026-11-21 재구성 전에는 틱이 칸 잠금을 끝까지 쥐어 이 동안 접수가 5초 뒤 409(의도한 차이)였다.
    use rtorder::route::Front;
    use rtorder::runner::DeskCell;
    use std::rc::Rc;
    let (w, io0, _dir) = setup("unlocked", Some(5.0), true);
    let (entered, open) = (Rc::new(tokio::sync::Notify::new()), Rc::new(tokio::sync::Notify::new()));
    let Wired { ops, rest, quotes, .. } = io0;
    let mut io = Wired::new(ops, rest, quotes, Gated { inner: Fake { fill_after: 2, ..Default::default() }, entered: entered.clone(), open: open.clone(), held: false }, "ETHUSDC");
    let desk = std::sync::Arc::new(DeskCell::new(rtorder::assemble::Desk::new(w)));
    io.desk = Some(desk.clone());
    let local = tokio::task::LocalSet::new();
    local.run_until(async {
        let d2 = desk.clone();
        let tick = tokio::task::spawn_local(async move {
            let mut w2 = d2.lock().await.w.clone();
            let r = w2.bracket_tick(&mut io).await;
            (r, w2, io.touched.clone())
        });
        entered.notified().await; // 청산이 거래소 체결을 기다리는 중
        let front = Front::default();
        let hub = Hub::default();
        let inp = json!({"account": {"ok": true, "positions": []}});
        let mut dio = desk_io::FakeDeskIo::new(&inp, desk.lock().await.w.dir.clone());
        let mut q = FakeQuotes::new(&json!({}));
        let st = tokio::time::timeout(std::time::Duration::from_millis(500),
            front.call(&desk, &hub, &mut q, &mut dio, "GET", "/api/manual-entry/status", &[("asset".into(), "eth".into())])).await
            .expect("감시 청산이 도는 동안에도 상태 조회는 바로 답한다");
        assert_eq!(st.status, 200);
        let v: Value = serde_json::from_str(&st.body).unwrap();
        assert_eq!((v["state"]["phase"].as_str(), v["state"]["trigger"].as_str()), (Some("submitting"), Some("bracket_sl")), "{}", st.body);
        let ex2 = front.call(&desk, &hub, &mut q, &mut dio, "POST", "/api/manual-exit/submit", &[("side".into(), "LONG".into()), ("confirm".into(), "1".into())]).await;
        assert_eq!(ex2.status, 409, "같은 코인 청산 = 칸이 바빠 409 {}", ex2.body);
        assert!(ex2.body.contains("already_working"));
        open.notify_one();
        let (r, mut w2, touched) = tick.await.unwrap();
        r.unwrap();
        assert_eq!(touched, vec![SlotId::Entry]);
        // 틱 끝에 원본으로(app 의 합치기와 같은 일)
        let mut d = desk.lock().await;
        *d.w.slot_mut(&SlotId::Entry) = w2.slot_mut(&SlotId::Entry).clone();
        assert_eq!(d.w.entry["phase"], "filled_maker", "{:?}", d.w.entry);
    }).await;
}

#[test]
fn run_exit_at_does_not_fire_when_intake_took_the_slot_after_snapshot() {
    // 감시 틱은 칸 사본에서 돈다 -- 사본을 뜬 뒤 접수가 같은 칸(위 칸)을 청산으로 잡았으면 발동 청산은 쏘지 않는다(두 번 닫지 않게 · 30초 뒤 다시)
    use rtorder::runner::DeskCell;
    let (w, io0, _dir) = setup("busy_after_snapshot", Some(5.0), true);
    let Wired { ops, rest, quotes, .. } = io0;
    let mut io = Wired::new(ops, rest, quotes, Fake { fill_after: 2, ..Default::default() }, "ETHUSDC");
    let desk = std::sync::Arc::new(DeskCell::new(rtorder::assemble::Desk::new(w)));
    io.desk = Some(desk.clone());
    let mut w2 = block_on(desk.lock()).w.clone();
    let intake = json!({"phase": "working", "kind": "exit", "side": "LONG", "asset": "eth", "note": "접수가 먼저"});
    block_on(desk.lock()).w.entry = intake.as_object().unwrap().clone();
    block_on(w2.bracket_tick(&mut io)).unwrap();
    assert!(io.ex.placed.is_empty() && io.ex.markets.is_empty(), "칸을 뺏긴 발동 청산이 주문을 냈다");
    assert_eq!(Value::Object(block_on(desk.lock()).w.entry.clone()), intake, "접수의 칸을 덮지 않는다");
    assert!(w2.entry.get("error").is_some_and(|e| e.as_str().unwrap().contains("다른 청산")), "{:?}", w2.entry);
}

#[test]
fn tick_end_merge_pushes_wait_cancel_into_running_exec() {
    // 감시가 포지션이 닫혀 같은 쪽 대기 진입을 거둘 때(action=cancel): 사본에서 쓴 값이 원본 칸과 **도는 대기 집행**(live)에 바로 들어간다
    use rtorder::exec::{Action, ExecState, Phase};
    let dir = std::env::temp_dir().join(format!("rtorder_merge_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let mut d = rtorder::assemble::Desk::new(Watcher::new(&dir, "ETHUSDC"));
    d.w.waits.insert("eth".into(), json!({"phase": "working", "mode": "limit", "side": "LONG"}));
    let sh = std::sync::Arc::new(std::sync::Mutex::new(ExecState { phase: Phase::Working, ..Default::default() }));
    d.live.insert(rtorder::runner::slot_key(&rtorder::intake::SlotRef::Wait("eth".into())), sh.clone());
    let waits0 = d.w.waits.clone();
    let mut w2 = d.w.clone();
    let ws = w2.waits.get_mut("eth").unwrap().as_object_mut().unwrap();
    ws.insert("action".into(), json!("cancel"));
    ws.insert("action_reason".into(), json!("포지션이 닫힘(손절·청산) — 같은 쪽 대기 진입 취소"));
    // 그 사이 접수가 원본 칸에 다른 칸을 썼다 -- 그것은 원본이 이긴다
    d.w.waits["eth"]["amend"] = json!({"price": 2400.0, "qty": 0.1});
    rtorder::app::merge_tick(&mut d, &mut w2, &waits0, &[]);
    assert_eq!(d.w.waits["eth"]["action"], "cancel");
    assert_eq!(d.w.waits["eth"]["amend"]["price"], json!(2400.0), "접수 값은 그대로");
    assert_eq!(sh.lock().unwrap().action, Some(Action::Cancel), "도는 대기 집행에 바로");
    let _ = std::fs::remove_dir_all(&dir);
}
