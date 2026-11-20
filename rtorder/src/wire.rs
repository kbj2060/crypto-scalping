//! 감시(`watch.rs`)가 바깥에 맡겨 둔 자리 중 **계좌·청산 계획 조립·청산 집행**을 넷째 조각의 실제 구현으로 잇는다(2026-11-21).
//! - `account()`  = 계좌 캐시(swr 10초 -- `account::swr_decide`) · 비었거나 낡으면 `AccountBook::produce`(REST)
//! - `produce_account()` = REST 로 새로 읽기(재시작 복구)
//! - `exit_plan()` = `Desk::assemble_exit_plan(fresh=True, only_symbol)` -- fresh 계좌는 스트림 계좌가 쓸 수 있으면 그것(`account::StreamAccount`), 아니면 REST
//! - `run_exit()` = `watch::run_exit_into`(exec::run_exit → `Exchange`)
//! 5분봉·1초 체결가·잠금·스위치·시계·거래소 손절/익절 함수는 여전히 `ops`(다른 `WatchIo`) 뒤 -- 실시간 코어와 잇는 다음 조각이 준다.
//! **어디에도 연결하지 않는다**(시험: tests/wire_e2e.rs -- 가짜 REST·가짜 시세·가짜 거래소).

use crate::account::{self, AccountBook, AccountRest, StreamAccount, Swr, UserStream};
use crate::assemble::{Desk, DeskIo, Quotes};
use crate::exec::Exchange;
use crate::watch::{iso_utc, run_exit_into, Obj, WatchIo, Watcher};
use serde_json::{json, Value};

pub struct Wired<O: WatchIo, R: AccountRest, Q: Quotes, E: Exchange> {
    pub ops: O,
    pub rest: R,
    pub quotes: Q,
    pub ex: E,
    pub book: AccountBook,
    /// 청산 조립 전용(상태 칸 없음 -- 주문 심볼 표만 쓴다). ponytail: 감시가 `&mut Watcher` 를 쥔 채 부르므로 같은 Desk 를 못 빌린다 -- 조립은 감시 칸을 안 읽어 따로 둔다.
    pub planner: Desk,
    /// (받은 시각 mono, 계좌) -- 원본 binance_account_cache.
    pub cache: Option<(f64, Value)>,
    pub stale_grace: f64,
    /// 스트림으로 유지하는 계좌(사용자 결정 «받아 둔 값»). None = 늘 REST(원본 그대로).
    pub stream: Option<(StreamAccount, UserStream)>,
    /// 청산 조립의 계좌가 어디서 왔나("stream" | 못 쓴 이유) -- 시험·로그.
    pub fresh_source: Vec<&'static str>,
}

impl<O: WatchIo, R: AccountRest, Q: Quotes, E: Exchange> Wired<O, R, Q, E> {
    pub fn new(ops: O, rest: R, quotes: Q, ex: E, eth_symbol: &str) -> Self {
        Wired { ops, rest, quotes, ex, book: AccountBook { eth_symbol: eth_symbol.into(), last_good: None },
                planner: Desk::new(Watcher::new(std::env::temp_dir(), eth_symbol)), cache: None, stale_grace: 60.0, stream: None, fresh_source: vec![] }
    }
}

/// 조립이 보는 계좌(청산은 fresh 만).
struct Acct<'a, R: AccountRest, O: WatchIo> {
    rest: &'a mut R,
    book: &'a mut AccountBook,
    stream: Option<&'a mut (StreamAccount, UserStream)>,
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
        match self.stream.as_deref_mut() {
            Some((sa, us)) => {
                let (v, why) = account::fresh_account(sa, us, self.book, self.rest).await?;
                self.source.push(why);
                Ok(v)
            }
            None => {
                self.source.push("rest");
                self.book.produce(self.rest).await
            }
        }
    }
    fn sec_prices(&mut self, asset: &str) -> Option<Vec<(f64, f64)>> {
        self.ops.sec_prices(asset).ok().flatten()
    }
    fn vol_mult(&self, _asset: &str) -> Value {
        json!({})
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
    /// swr_cached("binance_account", 10초, max_stale). ponytail: 원본은 묵은 값을 주고 뒤에서 갱신한다 -- 여기는 기다려 새로 읽는다(더 새 값 · 감시 틱이 그만큼 늦다).
    async fn account(&mut self) -> Result<Value, String> {
        let now = self.rest.now();
        let (ts, has) = self.cache.as_ref().map_or((0.0, false), |c| (c.0, true));
        match account::swr_decide(now, ts, has, account::ACCOUNT_CACHE_SEC, self.stale_grace) {
            Swr::Serve => Ok(self.cache.as_ref().unwrap().1.clone()),
            Swr::ServeAndRefresh | Swr::Fetch => {
                let v = self.book.produce(&mut self.rest).await?;
                if let Some((sa, us)) = self.stream.as_mut() {
                    sa.reconcile(&v, now, us.epoch);
                }
                self.cache = Some((now, v.clone()));
                Ok(v)
            }
        }
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        self.book.produce(&mut self.rest).await
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
        let mut io = Acct { rest: &mut self.rest, book: &mut self.book, stream: self.stream.as_mut(), ops: &mut self.ops, source: &mut self.fresh_source };
        self.planner.assemble_exit_plan(&mut self.quotes, &mut io, pside, fraction, true, asset, false, false, Some(only_symbol)).await.map_err(|e| e.0)
    }
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String> {
        let t = self.ops.now();
        run_exit_into(&mut self.ex, plan, slot, || iso_utc(t)).await
    }
    async fn place_bracket(&mut self, spec: &Value, symbol: &str, pside: &str, strict: bool) -> Result<Value, String> {
        self.ops.place_bracket(spec, symbol, pside, strict).await
    }
    async fn place_tp_ladder(&mut self, legs: &Value, symbol: &str, pside: &str) -> Result<Value, String> {
        self.ops.place_tp_ladder(legs, symbol, pside).await
    }
    async fn place_add_order(&mut self, symbol: &str, pside: &str, price: f64, qty: f64) -> Result<Value, String> {
        self.ops.place_add_order(symbol, pside, price, qty).await
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
    async fn ensure_leverage(&mut self, symbol: &str, target: i64) -> Value {
        self.ops.ensure_leverage(symbol, target).await
    }
}
