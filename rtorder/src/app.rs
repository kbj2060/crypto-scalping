//! 실행 파일 뼈대(2026-11-21 D) -- `bin/rtorder` 가 부르는 몸통. 설정(환경변수) · 시작 복구(원본 기동 훅 `start_wait_cleanup` → 감시 시작과 같은 순서) ·
//! 감시 루프(틱 앞에서 `drain_tp_fills`) · 사용자 스트림(켜면) · 그림자 모드.
//! 접수 HTTP(웹 서버)는 붙이지 않는다(R6 몫).
//!
//! 🔴그림자 모드(기본): 쓰기(주문·취소·레버리지·익절·고아 정리)는 **보내지 않고** `would` 에 적은 뒤 «못 함» 결과를 돌려준다(원본의 «키 없음»과 같은 모양 --
//! 감시는 무장을 남기고 30초 뒤 다시). 그래도 어딘가 쓰기 경로가 새면 `http::Binance` 의 그림자 막이가 보내기 전에 panic 한다(마지막 방어).
//! 상태 폴더는 Rust 전용이어야 한다(파이썬 서버의 data/live 를 같이 쓰면 무장 파일을 서로 덮는다 -- `config_from` 이 막는다).

use crate::account::{account_refresher, user_stream_run, RefreshIo, SharedAccount, UserStream};
use crate::bracket;
use crate::coreval::{self, Feed};
use crate::exec::{Common, Exchange, LimitOrder, OrderRef, Reply};
use crate::http::Binance;
use crate::telegram::{html_escape, Telegram};
use crate::userstream::{Hub, LiveStream};
use crate::assemble::{Cached, Desk};
use crate::route::Front;
use crate::runner::DeskCell;
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
    /// TELEGRAM_BOT_TOKEN·TELEGRAM_CHAT_ID(둘 다 있어야) · RTORDER_TELEGRAM_BASE(시험용 주소). 그림자에서는 안 보낸다.
    pub telegram: Option<Telegram>,
    /// 루프백 HTTP 문(RTORDER_LISTEN · 기본 127.0.0.1:8791 · 루프백이 아니면 설정 오류). 실제로 여는 것은 bin(묶기 실패 = 시작 안 함 -- 같은 주소로 둘째 rtorder 가 뜨지 않게).
    pub listen: std::net::SocketAddr,
    /// 로그를 그때그때 표준출력으로(bin) · false = 끝에 Summary 로(시험).
    pub echo: bool,
}

/// RTORDER_DIR 안 표식 -- rtorder 가 쓰는 폴더라는 뜻(접수 기록 order_requests.jsonl 이 있어도 파이썬 폴더가 아니다).
pub const DIR_MARK: &str = ".rtorder";

/// RTORDER_DIR(필수 · Rust 전용 폴더) · RTORDER_BASE(기본 바이낸스 선물) · RTORDER_WS(사용자 스트림 앞부분) · DASHBOARD_MANUAL_EXEC_SYMBOL(eth 주문 심볼 -- 원본과 같은 이름) ·
/// RTORDER_PERIOD(감시 주기 초, 기본 2) · RTORDER_MAX_TICKS(시험용) · RTORDER_STREAM=1(사용자 스트림) · RTORDER_HUB(rtcore 허브 ws 주소) · RTORDER_BAN_FILE(기본 폴더/binance_ban_until).
pub fn config_from(get: &dyn Fn(&str) -> Option<String>) -> Result<Config, String> {
    let dir = PathBuf::from(get("RTORDER_DIR").filter(|s| !s.is_empty()).ok_or("RTORDER_DIR 가 필요합니다(Rust 전용 상태 폴더)")?);
    if dir.join("order_requests.jsonl").exists() && !dir.join(DIR_MARK).exists() {
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
        telegram: Telegram::from_env(get),
        listen: {
            let raw = get("RTORDER_LISTEN").filter(|s| !s.is_empty()).unwrap_or_else(|| "127.0.0.1:8791".into());
            let a: std::net::SocketAddr = raw.parse().map_err(|e| format!("RTORDER_LISTEN {raw:?}: {e}"))?;
            crate::door::loopback_only(&a)?;
            a
        },
        echo: false,
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
    pub telegram: Option<Telegram>,
    /// 주문 잠금(원본 manual_order_lock) -- 감시가 손절 발동을 미룬다(D9 · G9).
    pub front: Arc<Front>,
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
        self.front.order_lock.try_lock().is_err()
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
        self.alerts.push(msg.into());
        // 원본 bracket_tick_alert: html.escape 해서 뒤에서 보낸다(최대 8초 -- 감시를 기다리게 하지 않는다 · 결과는 안 본다). 그림자 = 보내지 않고 «안 보낸 알림»
        if self.shadow(format!("TELEGRAM {msg}")) {
            return;
        }
        if let Some(t) = self.telegram.clone() {
            let text = html_escape(msg);
            tokio::task::spawn_blocking(move || t.send(&text));
        }
    }
}

/// 원본 `account_refresher` 의 바깥: 체결 알림(Hub.dirty) → 계좌 새로 읽기 → 공유 캐시 · 스트림 계좌 대조 → 번호(Hub.seq)+1 → 감시 깨움.
struct Refresher {
    rest: Binance,
    acct: Arc<SharedAccount>,
    hub: Arc<Hub>,
    stop: Stop,
}
impl RefreshIo for Refresher {
    async fn wait_dirty(&mut self) -> bool {
        if self.stop.flag.load(Ordering::SeqCst) {
            return false;
        }
        tokio::select! { _ = self.hub.dirty.notified() => {}, _ = self.stop.notify.notified() => {} }
        !self.stop.flag.load(Ordering::SeqCst)
    }
    /// acct_dirty.clear() -- 묶는 0.3초 사이에 온 알림은 이번 조회가 덮는다.
    fn clear_dirty(&mut self) {
        use futures_util::FutureExt;
        let _ = self.hub.dirty.notified().now_or_never();
    }
    async fn sleep(&mut self, sec: f64) -> bool {
        if self.stop.flag.load(Ordering::SeqCst) {
            return false;
        }
        tokio::select! { _ = tokio::time::sleep(std::time::Duration::from_secs_f64(sec)) => {}, _ = self.stop.notify.notified() => {} }
        !self.stop.flag.load(Ordering::SeqCst)
    }
    async fn refresh(&mut self) -> Result<(), String> {
        self.acct.refresh(&mut self.rest, Some(&self.hub), false).await.map(|_| ())
    }
    fn wake(&mut self) {
        self.hub.seq.fetch_add(1, Ordering::SeqCst); // 원본: seq += 1 바로 뒤 wake_fills.set()
        self.hub.wake.notify_waiters();
    }
    fn log(&mut self, s: String) {
        self.hub.logs.lock().unwrap().push(s);
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

/// 몸통: 시작 복구 → (사용자 스트림 · 계좌 갱신기 · 허브 구독 · 루프백 HTTP 문과 함께) 감시 루프 → 멈춤. `mk` = 같은 설정의 `Binance` 를 하나 더 만든다(계좌·시세·집행·손절 함수가 각자 하나).
/// `hub` = 스트림 상태(`hub.acct_seq()` · `hub.us` -- 문의 /internal/acct 가 낸다) · `listener` = bin 이 묶어 둔 루프백 문(None = 문 없음 · 시험).
/// 감시·접수·작업이 **같은 `DeskCell`**(무장·대기 칸이 메모리에서도 하나 -- 원본 같은 dict) 과 **같은 계좌 한 곳**(`SharedAccount`)을 쓴다.
pub async fn run(cfg: &Config, mk: &dyn Fn() -> Binance, stop: Stop, hub: Arc<Hub>, listener: Option<tokio::net::TcpListener>) -> Summary {
    let _ = std::fs::write(cfg.dir.join(DIR_MARK), "rtorder 상태 폴더\n"); // 이 폴더의 접수 기록(order_requests.jsonl)은 rtorder 것 -- 재기동 때 파이썬 폴더로 보지 않게
    let would: Would = Arc::new(Mutex::new(vec![]));
    let feed = Arc::new(Mutex::new(Feed::default()));
    let shadow = mk().is_shadow();
    let front = Arc::new(Front::default());
    let acct = Arc::new(SharedAccount::new(&cfg.eth_symbol));
    let desk = Arc::new(DeskCell::new(Desk::new(Watcher::new(&cfg.dir, &cfg.eth_symbol))));
    let ops = |front: &Arc<Front>| Ops { bx: mk(), feed: feed.clone(), hub: hub.clone(), stop: stop.notify.clone(), stopped: stop.flag.clone(), would: would.clone(), alerts: vec![],
                                         telegram: cfg.telegram.clone(), front: front.clone() };
    // 원본 기동 훅 게이트: start_wait_cleanup = 실행 스위치 · start_user_stream = 실행 스위치 + 키(주문이 없는 곳은 알림도 필요 없다)
    let gate = crate::plan::exec_enabled();
    let streaming = cfg.stream && gate && mk().has_keys();
    let wired = || {
        let mut w = Wired::new(ops(&front), mk(), Cached::new(mk()), GuardEx { bx: mk(), would: would.clone() }, &cfg.eth_symbol);
        w.acct = acct.clone();
        if streaming {
            w.stream = Some(hub.clone()); // 청산 조립의 새 계좌 = 스트림 계좌(쓸 수 있을 때) · 아니면 REST
        }
        w
    };
    let mut io = wired();
    io.desk = Some(desk.clone());
    let mut logs: Vec<String> = vec![];
    let echo = |logs: &mut Vec<String>, more: Vec<String>| {
        if cfg.echo {
            for l in &more {
                println!("{l}");
            }
        } else {
            logs.extend(more);
        }
    };
    {
        let mut d = desk.lock().await;
        d.w.log(format!("rtorder 시작 · {} · 폴더 {} · 주기 {}초", if shadow { "그림자" } else { "실전송" }, cfg.dir.display(), cfg.period));
        // 원본 기동 훅 순서: start_wait_cleanup(남은 대기 주문 정리 → 지정가 대기 복구 파일로 무장) → 감시 시작
        // ponytail: 원본은 정리를 작업으로 띄우고 바로 감시를 시작한다 -- 여기는 정리를 끝낸 뒤 감시(첫 틱이 정리 전 상태를 보지 않게 · 기동이 그만큼 늦다)
        if gate {
            if let Err(e) = d.w.wait_cleanup(&mut io).await {
                d.w.log(format!("wait cleanup 실패: {e}"));
            }
        }
        let l = std::mem::take(&mut d.w.logs);
        echo(&mut logs, l);
    }
    let mut ticks = 0u64;
    let watch = async {
        loop {
            if !io.nap(cfg.period).await {
                break;
            }
            io.quotes.now = io.ops.bx.time();
            coreval::sync_cached(&feed.lock().unwrap(), &mut io.quotes); // 받아 둔 호가·변동성(허브) -- 낡으면 Cached 가 REST 로
            {
                // 틱은 칸 **사본**에서 돈다(칸 잠금을 쥐지 않는다 -- 원본처럼 기다리는 사이 접수가 돈다). 사본과 원본을 잇는 것:
                // ① exiting 표시 = 같은 Arc(B1) ② 발동 청산의 칸 = `Wired::run_exit_at` 이 원본에 «submitting»·결과를 올린다
                // ③ 틱 끝에 원본으로: 올린 칸의 마지막 값 · 대기 칸에 쓴 취소(action) -- 도는 대기 집행에 바로(push_live) · 로그·알림 표시
                let mut w2 = desk.lock().await.w.clone();
                let waits0 = w2.waits.clone();
                if let Err(e) = w2.drain_tp_fills(&hub.tp_fills) {
                    w2.log(format!("tp_leg_filled: {e}"));
                }
                io.touched.clear();
                // R5: 멈춤 신호면 도는 틱(손절 청산 따라가기 포함)도 그 자리에서 취소한다(원본 stop_bracket_watcher 가 감시 작업을 cancel)
                let r = tokio::select! { r = w2.bracket_tick(&mut io) => Some(r), _ = stop.notify.notified() => None };
                let Some(r) = r else { break };
                if let Err(e) = r {
                    w2.log(format!("bracket watch: {e}"));
                }
                let mut d = desk.lock().await;
                for at in io.touched.drain(..) {
                    *d.w.slot_mut(&at) = w2.slot_mut(&at).clone();
                }
                for (k, v) in w2.waits.iter() {
                    if waits0.get(k) == Some(v) {
                        continue;
                    }
                    if let (Some(cur), Some(nv)) = (d.w.waits.get_mut(k).and_then(Value::as_object_mut), v.as_object()) {
                        for f in ["action", "action_reason"] {
                            if let Some(x) = nv.get(f) {
                                cur.insert(f.into(), x.clone());
                            }
                        }
                        d.push_live(&crate::intake::SlotRef::Wait(k.clone()));
                    }
                }
                d.w.alerted.extend(w2.alerted.iter().cloned());
                let mut l = std::mem::take(&mut w2.logs);
                l.extend(hub.logs.lock().unwrap().drain(..));
                l.extend(io.ops.alerts.drain(..));
                echo(&mut logs, l);
            }
            ticks += 1;
            if cfg.max_ticks.is_some_and(|m| ticks >= m) {
                stop.stop();
                break;
            }
        }
    };
    let stream = async {
        if streaming {
            let mut ls = LiveStream::new(mk(), &cfg.ws_base, hub.clone());
            let mut us = UserStream::default();
            tokio::select! { _ = user_stream_run(&mut ls, &mut us) => {}, _ = stop.notify.notified() => {} }
        }
    };
    let refresher = async {
        if streaming {
            let mut r = Refresher { rest: mk(), acct: acct.clone(), hub: hub.clone(), stop: stop.clone() };
            account_refresher(&mut r, &mut UserStream::default()).await;
        }
    };
    let sub = async {
        if let Some(url) = &cfg.hub_url {
            tokio::select! { _ = coreval::subscribe(url, feed.clone(), stop.flag.clone()) => {}, _ = stop.notify.notified() => {} }
        }
    };
    let door = async {
        if let Some(l) = listener {
            let dio = crate::door::DoorIo { ops: ops(&front), rest: mk(), acct: acct.clone(), stream: streaming.then(|| hub.clone()) };
            let d = crate::door::Door { desk: desk.clone(), hub: hub.clone(), front: front.clone(), feed: feed.clone(), dir: cfg.dir.clone(),
                                        io: tokio::sync::Mutex::new((dio, Cached::new(mk()))),
                                        make_task_io: Box::new(|| (wired(), GuardEx { bx: mk(), would: would.clone() }, Cached::new(mk()))) };
            d.serve(l, &stop).await;
        }
    };
    tokio::join!(watch, stream, refresher, sub, door);
    let mut d = desk.lock().await;
    logs.extend(std::mem::take(&mut d.w.logs));
    logs.extend(hub.logs.lock().unwrap().drain(..));
    logs.extend(io.ops.alerts.drain(..));
    let would = would.lock().unwrap().clone();
    Summary { ticks, logs, would, shadow }
}
