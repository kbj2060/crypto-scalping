//! 실행 파일 뼈대(2026-11-21 D) -- `bin/rtorder` 가 부르는 몸통. 설정(환경변수) · 시작 복구(원본 기동 훅 `start_wait_cleanup` → 감시 시작과 같은 순서) ·
//! 감시 루프(틱 앞에서 `drain_tp_fills`) · 사용자 스트림(켜면) · 그림자 모드.
//! 접수 HTTP(웹 서버)는 붙이지 않는다(R6 몫).
//!
//! 🔴그림자 모드(기본): 쓰기(주문·취소·레버리지·익절·고아 정리)는 **보내지 않고** `would` 에 적은 뒤 «못 함» 결과를 돌려준다(원본의 «키 없음»과 같은 모양 --
//! 감시는 무장을 남기고 30초 뒤 다시). 그래도 어딘가 쓰기 경로가 새면 `http::Binance` 의 그림자 막이가 보내기 전에 panic 한다(마지막 방어).
//! 상태 폴더는 Rust 전용이어야 한다(파이썬 서버의 data/live 를 같이 쓰면 무장 파일을 서로 덮는다 -- `config_from` 이 막는다).

use crate::account::{user_stream_run, UserStream};
use crate::bracket;
use crate::coreval::{self, Feed};
use crate::exec::{Common, Exchange, LimitOrder, OrderRef, Reply};
use crate::http::Binance;
use crate::userstream::{Hub, LiveStream};
use crate::watch::{Obj, WatchIo, Watcher};
use crate::wire::Wired;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct Config {
    pub dir: PathBuf,
    pub base: String,
    pub ws_base: String,
    pub eth_symbol: String,
    pub period: f64,
    pub max_ticks: Option<u64>,
    pub stream: bool,
    pub hub_url: Option<String>,
    pub ban_file: PathBuf,
}

/// RTORDER_DIR(필수 · Rust 전용 폴더) · RTORDER_BASE(기본 바이낸스 선물) · RTORDER_WS(사용자 스트림 앞부분) · DASHBOARD_MANUAL_EXEC_SYMBOL(eth 주문 심볼 -- 원본과 같은 이름) ·
/// RTORDER_PERIOD(감시 주기 초, 기본 2) · RTORDER_MAX_TICKS(시험용) · RTORDER_STREAM=1(사용자 스트림) · RTORDER_HUB(rtcore 허브 ws 주소) · RTORDER_BAN_FILE(기본 폴더/binance_ban_until).
pub fn config_from(get: &dyn Fn(&str) -> Option<String>) -> Result<Config, String> {
    let dir = PathBuf::from(get("RTORDER_DIR").filter(|s| !s.is_empty()).ok_or("RTORDER_DIR 가 필요합니다(Rust 전용 상태 폴더)")?);
    if dir.join("order_requests.jsonl").exists() {
        return Err(format!("{} 는 파이썬 서버 폴더로 보입니다(order_requests.jsonl) -- 무장 파일을 서로 덮지 않게 Rust 전용 폴더를 쓰세요", dir.display()));
    }
    let num = |k: &str, d: f64| get(k).filter(|s| !s.is_empty()).map_or(Ok(d), |s| s.parse::<f64>().map_err(|e| format!("{k}: {e}")));
    let period = num("RTORDER_PERIOD", crate::watch::WATCH_PERIOD_SEC)?;
    if !(period > 0.0 && period.is_finite()) {
        return Err("RTORDER_PERIOD 는 양수".into());
    }
    Ok(Config {
        ban_file: get("RTORDER_BAN_FILE").filter(|s| !s.is_empty()).map(PathBuf::from).unwrap_or_else(|| dir.join("binance_ban_until")),
        dir,
        base: get("RTORDER_BASE").filter(|s| !s.is_empty()).unwrap_or_else(|| crate::http::FAPI.into()),
        ws_base: get("RTORDER_WS").filter(|s| !s.is_empty()).unwrap_or_else(|| crate::account::USER_STREAM_WS.into()),
        eth_symbol: get("DASHBOARD_MANUAL_EXEC_SYMBOL").filter(|s| !s.is_empty()).unwrap_or_else(|| "ETHUSDC".into()),
        period,
        max_ticks: get("RTORDER_MAX_TICKS").and_then(|s| s.parse().ok()),
        stream: get("RTORDER_STREAM").as_deref() == Some("1"),
        hub_url: get("RTORDER_HUB").filter(|s| !s.is_empty()),
    })
}

type Would = Arc<Mutex<Vec<String>>>;
const SHADOW_NOTE: &str = "그림자 모드: 주문을 내지 않았습니다";

/// 집행(청산 따라가기) 전송 -- 그림자면 쓰기를 거절 응답으로 바꾼다(«400 [shadow]» = 확정 거부라 «나갔는지 모름» 조회·표식 DELETE 도 안 탄다).
pub struct GuardEx {
    pub bx: Binance,
    pub would: Would,
}
impl GuardEx {
    fn refuse(&self, what: String) -> Reply {
        self.would.lock().unwrap().push(what);
        Err(format!("400 [shadow] {SHADOW_NOTE}"))
    }
}
impl Exchange for GuardEx {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply {
        if self.bx.is_shadow() {
            return self.refuse(format!("GTX {} {} {} {}@{}", o.symbol, o.side, o.position_side, o.quantity, o.price));
        }
        self.bx.place_limit_gtx(o).await
    }
    async fn place_market(&mut self, c: &Common<'_>, q: f64) -> Reply {
        if self.bx.is_shadow() {
            return self.refuse(format!("MARKET {} {} {} {q}", c.symbol, c.side, c.position_side));
        }
        self.bx.place_market(c, q).await
    }
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        self.bx.get_order(c, r).await
    }
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply {
        if self.bx.is_shadow() {
            return self.refuse(format!("CANCEL {} {r:?}", c.symbol));
        }
        self.bx.cancel_order(c, r).await
    }
    async fn book(&mut self, s: &str) -> Result<(f64, f64), String> {
        self.bx.book_ticker(s).await
    }
    fn now(&self) -> f64 {
        self.bx.now()
    }
    fn wall_ms(&self) -> i64 {
        self.bx.wall_ms()
    }
    async fn nap(&mut self, s: f64) {
        self.bx.nap(s).await
    }
    async fn sleep(&mut self, s: f64) {
        self.bx.sleep(s).await
    }
    async fn prepare(&mut self, symbol: &str, leverage: Option<i64>) -> Result<Option<Value>, String> {
        if self.bx.is_shadow() {
            // 레버리지 POST 는 쓰기 -- 그림자에서는 시계만 맞추고 바꿨을 값만 적는다
            if !self.bx.has_keys() {
                return Err("API 키가 없습니다".into());
            }
            self.bx.sync_clock().await;
            if let Some(t) = leverage.filter(|t| *t > 0) {
                self.would.lock().unwrap().push(format!("LEVERAGE {symbol} {t}"));
            }
            return Ok(leverage.map(|t| json!({"changed": false, "shadow": true, "to": t})));
        }
        self.bx.prepare(symbol, leverage).await
    }
}

/// 감시가 바깥에 닿는 곳(계좌·청산 조립·집행은 `Wired` 가 덮는다). 5분봉·1초가 = `Feed`(rtcore 허브 구독 -- E) · 손절·익절 함수 = `bracket::*`(그림자면 적기만).
pub struct Ops {
    pub bx: Binance,
    pub feed: Arc<Mutex<Feed>>,
    pub hub: Arc<Hub>,
    pub stop: Arc<tokio::sync::Notify>,
    pub stopped: Arc<AtomicBool>,
    pub would: Would,
    pub alerts: Vec<String>,
}
impl Ops {
    fn shadow(&self, what: String) -> bool {
        if self.bx.is_shadow() {
            self.would.lock().unwrap().push(what);
            return true;
        }
        false
    }
}
impl WatchIo for Ops {
    fn now(&self) -> f64 {
        self.bx.time()
    }
    fn exec_enabled(&self) -> bool {
        crate::plan::exec_enabled()
    }
    fn order_locked(&self) -> bool {
        false // ponytail: 접수 문이 없다(R6) -- 웹 서버를 붙이면 그 잠금을 여기로
    }
    /// 감시 주기 잠 -- 체결 알림(Hub.wake)이면 일찍 깨고, 멈춤 신호면 false(원본 CancelledError).
    async fn nap(&mut self, sec: f64) -> bool {
        if self.stopped.load(Ordering::SeqCst) {
            return false;
        }
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs_f64(sec)) => {}
            _ = self.hub.wake.notified() => {}
            _ = self.stop.notified() => {}
        }
        !self.stopped.load(Ordering::SeqCst)
    }
    async fn account(&mut self) -> Result<Value, String> {
        unreachable!("계좌는 Wired 가 읽는다")
    }
    async fn produce_account(&mut self) -> Result<Value, String> {
        unreachable!("계좌는 Wired 가 읽는다")
    }
    async fn bars(&mut self, asset: &str) -> Result<Vec<Value>, String> {
        let f = self.feed.lock().unwrap(); // 감시 5분봉 = 화면 심볼(USDT) -- 원본 load_market_history
        Ok(coreval::bars(crate::assemble::market_symbol(asset).and_then(|s| f.bar5m.get(s)).copied(), self.bx.time()))
    }
    fn sec_prices(&mut self, asset: &str) -> Result<Option<Vec<(f64, f64)>>, String> {
        let f = self.feed.lock().unwrap();
        Ok(coreval::sec_prices(crate::assemble::market_symbol(asset).and_then(|s| f.secs.get(s)).map(Vec::as_slice)))
    }
    async fn clear_bracket(&mut self, s: &str, p: &str, only_sl: bool, only_tp: bool) -> Result<Value, String> {
        if self.shadow(format!("CLEAR {s} {p} only_sl={only_sl} only_tp={only_tp}")) {
            return Ok(json!({"cancelled": 0, "errors": [SHADOW_NOTE]}));
        }
        Ok(bracket::clear_bracket(&mut self.bx, s, p, only_sl, only_tp).await)
    }
    async fn exit_plan(&mut self, _: &str, _: f64, _: &str, _: &str) -> Result<Value, Value> {
        unreachable!("청산 조립은 Wired 가 한다")
    }
    async fn run_exit(&mut self, _: &Value, _: &mut Obj) -> Result<(), String> {
        unreachable!("청산 집행은 Wired 가 한다")
    }
    async fn place_bracket(&mut self, spec: &Value, s: &str, p: &str, strict: bool) -> Result<Value, String> {
        if self.shadow(format!("BRACKET {s} {p} {spec}")) {
            return Ok(json!({"placed": false, "reason": SHADOW_NOTE}));
        }
        Ok(bracket::place_bracket(&mut self.bx, spec, s, p, strict).await)
    }
    async fn place_tp_ladder(&mut self, legs: &Value, s: &str, p: &str) -> Result<Value, String> {
        if self.shadow(format!("TP_LADDER {s} {p} {legs}")) {
            return Ok(json!({"placed": false, "reason": SHADOW_NOTE}));
        }
        bracket::place_tp_ladder(&mut self.bx, legs.as_array().map(Vec::as_slice).unwrap_or(&[]), s, p).await
    }
    async fn cancel_wait_orphans(&mut self, syms: &[String]) -> Result<Value, String> {
        if self.shadow(format!("CANCEL_WAIT_ORPHANS {syms:?}")) {
            return Ok(json!({"cancelled": [], "errors": [SHADOW_NOTE]}));
        }
        Ok(bracket::cancel_wait_orphans(&mut self.bx, &syms.iter().map(String::as_str).collect::<Vec<_>>()).await)
    }
    async fn cancel_exit_orders(&mut self, s: &str, p: &str) -> Result<Value, String> {
        if self.shadow(format!("CANCEL_EXIT_ORDERS {s} {p}")) {
            return Ok(json!({"cancelled": 0, "errors": [SHADOW_NOTE]}));
        }
        Ok(bracket::cancel_exit_orders(&mut self.bx, s, p).await)
    }
    fn alert(&mut self, msg: &str) {
        self.alerts.push(msg.into()); // ponytail: 텔레그램은 안 보낸다 -- 로그로만(서버에 붙일 때 보내기 구현)
    }
}

pub struct Summary {
    pub ticks: u64,
    pub logs: Vec<String>,
    pub would: Vec<String>,
    pub shadow: bool,
}

/// 멈춤 손잡이(SIGTERM 이 누른다).
#[derive(Clone, Default)]
pub struct Stop {
    pub notify: Arc<tokio::sync::Notify>,
    pub flag: Arc<AtomicBool>,
}
impl Stop {
    pub fn stop(&self) {
        self.flag.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }
}

/// 몸통: 시작 복구 → (사용자 스트림 · 허브 구독과 함께) 감시 루프 → 멈춤. `mk` = 같은 설정의 `Binance` 를 하나 더 만든다(계좌·시세·집행·손절 함수가 각자 하나).
pub async fn run(cfg: &Config, mk: &dyn Fn() -> Binance, stop: Stop) -> Summary {
    let would: Would = Arc::new(Mutex::new(vec![]));
    let hub = Arc::new(Hub::default());
    let feed = Arc::new(Mutex::new(Feed::default()));
    let shadow = mk().is_shadow();
    let ops = Ops { bx: mk(), feed: feed.clone(), hub: hub.clone(), stop: stop.notify.clone(), stopped: stop.flag.clone(), would: would.clone(), alerts: vec![] };
    let mut io = Wired::new(ops, mk(), crate::assemble::Cached::new(mk()), GuardEx { bx: mk(), would: would.clone() }, &cfg.eth_symbol);
    let mut w = Watcher::new(&cfg.dir, &cfg.eth_symbol);
    w.log(format!("rtorder 시작 · {} · 폴더 {} · 주기 {}초", if shadow { "그림자" } else { "실전송" }, cfg.dir.display(), cfg.period));
    // 원본 기동 훅 순서: start_wait_cleanup(남은 대기 주문 정리 → 지정가 대기 복구 파일로 무장) → 감시 시작
    // ponytail: 원본은 정리를 작업으로 띄우고 바로 감시를 시작한다 -- 여기는 정리를 끝낸 뒤 감시(첫 틱이 정리 전 상태를 보지 않게 · 기동이 그만큼 늦다)
    if let Err(e) = w.wait_cleanup(&mut io).await {
        w.log(format!("wait cleanup 실패: {e}"));
    }
    let mut ticks = 0u64;
    let watch = async {
        loop {
            if !io.nap(cfg.period).await {
                break;
            }
            io.quotes.now = io.ops.bx.time();
            coreval::sync_cached(&feed.lock().unwrap(), &mut io.quotes); // 받아 둔 호가·변동성(허브) -- 낡으면 Cached 가 REST 로
            if let Err(e) = w.drain_tp_fills(&hub.tp_fills) {
                w.log(format!("tp_leg_filled: {e}"));
            }
            if let Err(e) = w.bracket_tick(&mut io).await {
                w.log(format!("bracket watch: {e}"));
            }
            ticks += 1;
            if cfg.max_ticks.is_some_and(|m| ticks >= m) {
                stop.stop();
                break;
            }
        }
    };
    let stream = async {
        if cfg.stream {
            let mut ls = LiveStream::new(mk(), &cfg.ws_base, hub.clone());
            let mut us = UserStream::default();
            tokio::select! { _ = user_stream_run(&mut ls, &mut us) => {}, _ = stop.notify.notified() => {} }
        }
    };
    let sub = async {
        if let Some(url) = &cfg.hub_url {
            tokio::select! { _ = coreval::subscribe(url, feed.clone(), stop.flag.clone()) => {}, _ = stop.notify.notified() => {} }
        }
    };
    tokio::join!(watch, stream, sub);
    let mut logs = std::mem::take(&mut w.logs);
    logs.extend(hub.logs.lock().unwrap().drain(..));
    logs.extend(io.ops.alerts.drain(..));
    let would = would.lock().unwrap().clone();
    Summary { ticks, logs, would, shadow }
}
