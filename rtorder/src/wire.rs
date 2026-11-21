//! 감시(`watch.rs`)가 바깥에 맡겨 둔 자리 중 **계좌·청산 계획 조립·청산 집행**을 넷째 조각의 실제 구현으로 잇는다(2026-11-21).
//! - `account()`  = 계좌 캐시(swr 10초 -- `account::swr_decide`) · 비었거나 낡으면 `AccountBook::produce`(REST)
//! - `produce_account()` = REST 로 새로 읽기(재시작 복구)
//! - `exit_plan()` = `Desk::assemble_exit_plan(fresh=True, only_symbol)` -- fresh 계좌는 스트림 계좌가 쓸 수 있으면 그것(`account::StreamAccount`), 아니면 REST
//! - `run_exit()` = `watch::run_exit_into`(exec::run_exit → `Exchange`)
//! 5분봉·1초 체결가·잠금·스위치·시계·거래소 손절/익절 함수는 여전히 `ops`(다른 `WatchIo`) 뒤 -- 실시간 코어와 잇는 다음 조각이 준다.
//! **어디에도 연결하지 않는다**(시험: tests/wire_e2e.rs -- 가짜 REST·가짜 시세·가짜 거래소).

use crate::account::{AccountRest, SharedAccount};
use crate::assemble::{Desk, DeskIo, Quotes};
use crate::exec::Exchange;
use crate::watch::{iso_utc, run_exit_into, Obj, SlotId, WatchIo, Watcher};
use crate::userstream::Hub;
use serde_json::Value;
use std::sync::Arc;

pub struct Wired<O: WatchIo, R: AccountRest, Q: Quotes, E: Exchange> {
    pub ops: O,
    pub rest: R,
    pub quotes: Q,
    pub ex: E,
    /// 청산 조립 전용(상태 칸 없음 -- 주문 심볼 표만 쓴다). ponytail: 감시가 `&mut Watcher` 를 쥔 채 부르므로 같은 Desk 를 못 빌린다 -- 조립은 감시 칸을 안 읽어 따로 둔다.
    pub planner: Desk,
    /// 계좌 한 곳(감시·접수·갱신기·화면이 같이) -- `account::SharedAccount`.
    pub acct: Arc<SharedAccount>,
    /// 스트림으로 유지하는 계좌(사용자 결정 «받아 둔 값» -- Hub.stream_account · Hub.us). None = 늘 REST(원본 그대로).
    pub stream: Option<Arc<Hub>>,
    /// 청산 조립의 계좌가 어디서 왔나("stream" | 못 쓴 이유) -- 시험·로그.
    pub fresh_source: Vec<&'static str>,
    /// 2026-11-21 실행 파일: 감시 틱은 칸 사본에서 돌고, 발동 청산의 칸 상태는 이 원본(접수가 보는 칸)에 올린다(`run_exit_at`). None = 사본 없이(시험).
    pub desk: Option<Arc<crate::runner::DeskCell>>,
    /// 이 틱에 원본에 올린 칸 -- 틱 끝에 사본의 마지막 값(정리 결과 등)을 원본에 옮긴다(app).
    pub touched: Vec<SlotId>,
}

impl<O: WatchIo, R: AccountRest, Q: Quotes, E: Exchange> Wired<O, R, Q, E> {
    pub fn new(ops: O, rest: R, quotes: Q, ex: E, eth_symbol: &str) -> Self {
        Wired { ops, rest, quotes, ex, acct: Arc::new(SharedAccount::new(eth_symbol)),
                planner: Desk::new(Watcher::new(std::env::temp_dir(), eth_symbol)), stream: None, fresh_source: vec![], desk: None, touched: vec![] }
    }
}

/// 조립이 보는 계좌(청산은 fresh 만).
struct Acct<'a, R: AccountRest, O: WatchIo> {
    rest: &'a mut R,
    acct: &'a SharedAccount,
    stream: Option<&'a Hub>,
    ops: &'a mut O,
    source: &'a mut Vec<&'static str>,
}

impl<R: AccountRest, O: WatchIo> DeskIo for Acct<'_, R, O> {
    fn now(&self) -> f64 {
        self.ops.now()
    }
    fn exec_enabled(&self) -> bool {
        self.ops.exec_enabled()
    }
    async fn account(&mut self, fresh: bool) -> Result<Value, String> {
        debug_assert!(fresh, "감시 청산은 늘 새 계좌(C1)");
        match self.stream {
            // 스트림 계좌를 쓸 수 있으면 그것(REST 왕복 없음) · 아니면 REST 로 새로 읽고 맞춰 본다(설계 §4-2). 잠금은 await 밖에서만.
            Some(hub) => {
                let now = self.rest.now();
                let why = match hub.stream_account.lock().unwrap().usable(&hub.us.lock().unwrap(), now) {
                    Ok(v) => {
                        self.source.push("stream");
                        return Ok(v);
                    }
                    Err(why) => why,
                };
                let v = self.acct.produce(self.rest, Some(hub)).await?;
                self.source.push(why);
                Ok(v)
            }
            None => {
                self.source.push("rest");
                self.acct.produce(self.rest, None).await
            }
        }
    }
    fn sec_prices(&mut self, asset: &str) -> Option<Vec<(f64, f64)>> {
        self.ops.sec_prices(asset).ok().flatten()
    }
    async fn clear_bracket(&mut self, _s: &str, _p: &str, _a: bool, _b: bool) -> Result<Value, String> {
        unreachable!("청산 조립은 손절 해제를 안 부른다")
    }
    async fn place_bracket(&mut self, _: &Value, _: &str, _: &str, _: bool) -> Result<Value, String> {
        unreachable!("청산 조립은 익절을 안 건다")
    }
    async fn place_tp_ladder(&mut self, _: &Value, _: &str, _: &str) -> Result<Value, String> {
        unreachable!("청산 조립은 익절을 안 건다")
    }
}

impl<O: WatchIo, R: AccountRest, Q: Quotes, E: Exchange> WatchIo for Wired<O, R, Q, E> {
    fn now(&self) -> f64 {
        self.ops.now()
    }
    fn exec_enabled(&self) -> bool {
        self.ops.exec_enabled()
    }
    fn order_locked(&self) -> bool {
        self.ops.order_locked()
    }
    async fn nap(&mut self, sec: f64) -> bool {
        self.ops.nap(sec).await
    }
    /// swr_cached("binance_account", 10초, max_stale) -- `SharedAccount::cached`.
    async fn account(&mut self) -> Result<Value, String> {
        self.acct.cached(&mut self.rest, self.stream.as_deref()).await
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        self.acct.produce(&mut self.rest, None).await
    }
    async fn bars(&mut self, asset: &str) -> Result<Vec<Value>, String> {
        self.ops.bars(asset).await
    }
    fn sec_prices(&mut self, asset: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        self.ops.sec_prices(asset)
    }
    async fn clear_bracket(&mut self, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        self.ops.clear_bracket(symbol, pside, only_sl, only_tp).await
    }
    /// 감시 발동 청산 = assemble_exit_plan(pside, frac, fresh=True, asset, only_symbol) -- 원본은 for_order 를 넘기지 않아 화면용 펀딩·수수료까지 묻는다(그대로).
    async fn exit_plan(&mut self, pside: &str, fraction: f64, asset: &str, only_symbol: &str) -> Result<Value, Value> {
        let mut io = Acct { rest: &mut self.rest, acct: &self.acct, stream: self.stream.as_deref(), ops: &mut self.ops, source: &mut self.fresh_source };
        self.planner.assemble_exit_plan(&mut self.quotes, &mut io, pside, fraction, true, asset, false, Some(only_symbol)).await.map_err(|e| e.0)
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let ops = &self.ops;
        run_exit_into(&mut self.ex, plan, slot, || iso_utc(ops.now())).await // done_at = 끝난 시각(원본 now_iso)
    }
    /// 원본: 감시가 칸을 «submitting» 으로 바꾼 **뒤** run_exit 를 기다린다 -- 그동안 접수는 그 칸을 바쁜 것으로 보고(409) 다른 칸·다른 요청은 돈다.
    /// 칸 잠금은 올릴 때·내릴 때만 잡는다. 사본을 뜬 뒤 접수가 그 칸을 먼저 잡았으면 쏘지 않는다(칸은 «submitting» 그대로 → 감시가 30초 뒤 다시).
    async fn run_exit_at(&mut self, at: &SlotId, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let Some(desk) = self.desk.clone() else { return self.run_exit(plan, slot).await };
        {
            let mut d = desk.lock().await;
            if crate::watch::busy(d.w.slot_mut(at)) {
                slot.insert("error".into(), serde_json::json!("다른 청산이 먼저 이 칸을 잡았다 -- 다음 틱에 다시"));
                return Ok(());
            }
            *d.w.slot_mut(at) = slot.clone();
        }
        self.touched.push(at.clone());
        let r = self.run_exit(plan, slot).await;
        *desk.lock().await.w.slot_mut(at) = slot.clone();
        r
    }
    async fn place_bracket(&mut self, spec: &Value, symbol: &str, pside: &str, strict: bool) -> Result<Value, String> {
        self.ops.place_bracket(spec, symbol, pside, strict).await
    }
    async fn place_tp_ladder(&mut self, legs: &Value, symbol: &str, pside: &str) -> Result<Value, String> {
        self.ops.place_tp_ladder(legs, symbol, pside).await
    }
    async fn cancel_wait_orphans(&mut self, symbols: &[String]) -> Result<Value, String> {
        self.ops.cancel_wait_orphans(symbols).await
    }
    async fn cancel_exit_orders(&mut self, symbol: &str, pside: &str) -> Result<Value, String> {
        self.ops.cancel_exit_orders(symbol, pside).await
    }
    fn alert(&mut self, msg: &str) {
        self.ops.alert(msg)
    }
}
