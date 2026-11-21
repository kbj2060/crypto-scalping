//! 지정가 집행 상태 기계 -- `live/execution/live_manual_peg_execute_20260912.py` 의 `fill_maker`·`ensure_closed`·`run_entry`·`run_exit`·
//! `run_entry_limit`·`run_entry_stop` 을 그대로 옮긴 것. 거래소 호출·시계는 `Exchange` trait 뒤(구현 = 시험의 가짜 · `http::Binance`).
//!
//! 시장가 다리(M1·M2 감시 진입 TAKER 몫·M4·M5·M6·M7·M9)는 원본과 같은 순서·같은 요청·같은 결과(2026-11-21 사용자 결정 «원본 동작 유지») -- `Exchange::place_market`.
//! M3(기다리기)·M8(«즉시» 청산)·스위칭은 원본 PR 61 처럼 지웠다(2026-11-21 사용자 «sltp 만»).
//!
//! 원본 run_* 머리(API 키 없음 → error · `_clock_offset` · 진입이면 `ensure_leverage`)는 `Exchange::prepare`(2026-11-21 일곱째 조각 --
//! 전체 흐름 정답지에서 실행기가 시계 요청·레버리지를 빼먹던 것을 잡고 옮김. run_entry_stop 은 원본처럼 머리 없이 닿은 뒤 run_entry 가 한다).
//! 실제 전송 = `http::Binance`(이 trait 의 구현) · 브래킷·고아 정리 = `bracket`.

use crate::pyfmt::round_n;
use std::sync::{Arc, Mutex};

pub const FALLBACK_SEC: f64 = 120.0; // D1: 계획 쪽 plan::FALLBACK_SEC 와 두 벌(원본 그대로)
pub const POLL_SEC: f64 = 3.0; // 재호가 판단 주기 -- 설계 문서 §2 «안 바꾼다»
pub const TERMINAL: [&str; 4] = ["FILLED", "CANCELED", "EXPIRED", "REJECTED"]; // PARTIALLY_FILLED·NEW 는 종료가 아니다(E11)
pub const REPEG_MAX: u32 = 40; // 3초 × 120초 = 40회가 물리적 상한. 폭주 방지 이중 안전장치(E15)
pub const PEG_TAG: &str = "dbpg"; // 2026-10-16 따라가기 GTX 표식 -- 모호한 POST 조회 · 재시작 고아 정리
pub const WAIT_TAG: &str = "dbwt"; // 기다리기·지정가 대기 표식
pub const LIMIT_MAX_SEC: f64 = 4.0 * 3600.0; // 지정가 대기 최대(검증 D2) -- 넘으면 «만료 취소»(시장가 없음)
pub const REPRICE_SEC: f64 = 20.0;
pub const REPRICE_MIN_REL: f64 = 1e-4;
pub const REPRICE_MAX: u32 = 50;
const _: () = assert!(REPEG_MAX as f64 * POLL_SEC >= FALLBACK_SEC); // 리페그 상한이 마감보다 먼저 걸리면 남은 시간을 못 쓴다

/// 거래소 주문 응답. 오류는 파이썬 `signed()` 와 같은 모양의 문자열이다:
/// HTTP 응답이면 `"<상태> [<바이낸스 code>] msg"`(숫자로 시작 -- 🔴code 를 버리면 -5022 판정이 죽는다, 2026-09-20 실사고),
/// 네트워크·시간초과면 `"TimeoutError: …"`(숫자로 시작하지 않음 = «나갔는지 모른다»).
#[derive(Debug, Clone, Default)]
pub struct Order {
    pub order_id: Option<i64>,
    pub status: String,
    pub executed_qty: f64,
}
pub type Reply = Result<Order, String>;

#[derive(Debug, Clone, PartialEq)]
pub enum OrderRef {
    Id(Option<i64>),
    Client(String),
}

/// 보내는 지정가 하나. 늘 LIMIT · GTX(post-only).
#[derive(Debug, Clone)]
pub struct LimitOrder {
    pub symbol: String,
    pub side: String,          // BUY/SELL
    pub position_side: String, // 헤지 모드라 필수 · reduceOnly 는 보내지 않는다(-1106)
    pub price: f64,
    pub quantity: f64,
    pub client_order_id: String,
}

/// 거래소 + 시계. 시험은 가짜 하나로 3초·120초를 실제로 기다리지 않는다.
/// ponytail: `on_fill`·`reprice`(파이썬의 선택 콜백)도 여기 둔다 -- 호출부가 넘겼는지는 함수 인자 bool 로. 콜백 종류가 늘면 따로 뺀다.
#[allow(async_fn_in_trait)]
pub trait Exchange {
    async fn place_limit_gtx(&mut self, o: &LimitOrder) -> Reply;
    /// 시장가 다리 `{**common, "type": "MARKET", "quantity": q}`(2026-11-21 사용자 결정: 원본 동작 유지). reduceOnly 없음(헤지 모드).
    async fn place_market(&mut self, c: &Common<'_>, quantity: f64) -> Reply;
    /// `c` = 원본의 `common`(심볼·측면·포지션 측면) -- 주문 번호로 묻고 지울 때 원본은 셋 다 보낸다(`{**common, "orderId": oid}`), 표식이면 심볼만.
    async fn get_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply;
    async fn cancel_order(&mut self, c: &Common<'_>, r: &OrderRef) -> Reply;
    /// 최우선 (매수호가, 매도호가) -- 🔴**주문이 나가는 그 심볼**에서(E14, 2026-09-19·20 USDT 호가로 USDC 주문 → -5022 5/5).
    async fn book(&mut self, symbol: &str) -> Result<(f64, f64), String>;
    fn now(&self) -> f64; // time.monotonic()
    fn wall_ms(&self) -> i64; // int(time.time()*1000) -- 표식
    /// 파이썬 `nap`: sec 동안 기다리되 체결 알림이면 일찍 깬다(알림은 앞당기기만, A10).
    async fn nap(&mut self, sec: f64);
    async fn sleep(&mut self, sec: f64);
    async fn on_fill(&mut self) {}
    async fn reprice(&mut self) -> Result<Option<f64>, String> {
        Ok(None)
    }
    /// 원본 run_entry·run_exit·run_entry_limit 머리: 키 없음 = Err("API 키가 없습니다") · 시계 맞추기 · leverage = Some(목표) 면
    /// ensure_leverage 결과(state["leverage"] -- 실패해도 주문은 보낸다, P16). 기본(시험 가짜) = 아무것도 안 함.
    async fn prepare(&mut self, _symbol: &str, _leverage: Option<i64>) -> Result<Option<serde_json::Value>, String> {
        Ok(None)
    }
}

/// `prepare` 를 부르고 실패면 error 로 끝낸다(false = 멈춤).
async fn head<E: Exchange>(ex: &mut E, plan: &ExecPlan, s: &Shared, leverage: bool) -> bool {
    match ex.prepare(&plan.symbol, leverage.then_some(plan.target_leverage)).await {
        Err(e) => {
            st(s).error = Some(e);
            finish(s, Phase::Error);
            false
        }
        Ok(l) => {
            if l.is_some() {
                st(s).leverage = l;
            }
            true
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Idle,
    Working,
    FilledMaker,
    Rejected,
    Error,
    Cancelled,
    /// 시장가 다리가 체결됐다(2026-11-21 사용자 결정 «원본 동작 유지» -- M1·M2·M4·M5·M6·M7·M9).
    FilledTaker,
    /// 시장가 다리 접수가 오류(원본 그대로 조회·재시도 없이 멈춤 -- 불명 응답도 같다).
    TakerFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Cancel,
    Market,
    Peg,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Amend {
    pub price: f64,
    pub qty: Option<f64>,
    pub auto: bool,
}

/// 파이썬 state dict. 집행 루프와 API 가 **같은 칸**을 주고받는다(W12) -- 그래서 Arc<Mutex>. 잠금은 await 너머로 들고 있지 않는다.
#[derive(Debug, Clone, Default)]
pub struct ExecState {
    pub phase: Phase,
    pub mode: &'static str,
    pub filled: f64,
    pub quantity: f64,
    pub limit_price: f64,
    pub repegs: u32,
    pub deadline_sec: f64,
    pub order_id: Option<i64>,
    pub client_order_id: Option<String>,
    pub error: Option<String>,
    pub taker_qty: f64,
    pub cancel_reason: Option<String>,
    pub action: Option<Action>,
    pub action_reason: Option<String>,
    pub amend: Option<Amend>,
    pub amends: u32,
    pub reprices: u32,
    pub pegged: bool,
    pub vol_bpm: Option<f64>,
    pub market_reason: Option<String>,
    /// 시장가 다리의 주문 번호(원본 taker_order_id).
    pub taker_order_id: Option<i64>,
    pub taker_reason: Option<&'static str>,
    /// 감시 진입(run_entry_stop) 표시 · 닿아서 따라가기를 시작했나(원본은 시각 문자열).
    pub stop: bool,
    /// 닿은 벽시계(epoch 초) -- 원본 state["triggered"] = now_iso()(닿기 전 None).
    pub triggered: Option<f64>,
    pub done: bool,
    /// ensure_leverage 결과(원본 state["leverage"]).
    pub leverage: Option<serde_json::Value>,
    /// 지정가·감시 대기 만료 벽시계(epoch 초 · 원본 wait_until iso).
    pub wait_until: Option<f64>,
}
pub type Shared = Arc<Mutex<ExecState>>;

/// 집행 입력(계획 dict 중 집행이 읽는 칸).
#[derive(Debug, Clone)]
pub struct ExecPlan {
    pub symbol: String,
    pub side: String,
    pub position_side: String,
    pub market: bool, // plan["type"] == "MARKET"
    pub price: f64,
    pub quantity: f64,
    pub fallback_after_sec: f64, // 청산 마감(계획이 정한다 -- 변동성 연동 15~120초, P27). 0 = 없음 → 120
    pub maker_only: bool,        // 감시 진입: 마감 뒤 잔량은 버린다(시장가 없음)
    pub vol_bpm: Option<f64>,
    pub market_reason: Option<String>,
    /// int(plan.get("target_leverage") or 0) -- 진입 머리의 ensure_leverage.
    pub target_leverage: i64,
}

fn r8(x: f64) -> f64 {
    round_n(x, 8)
}
fn terminal(s: &str) -> bool {
    TERMINAL.contains(&s)
}
fn id_str(id: Option<i64>) -> String {
    id.map(|v| v.to_string()).unwrap_or_else(|| "None".into())
}
fn truthy_id(o: &Order) -> bool {
    o.order_id.is_some_and(|v| v != 0) // 파이썬 found.get("orderId") 진릿값
}
fn st(s: &Shared) -> std::sync::MutexGuard<'_, ExecState> {
    s.lock().unwrap()
}
fn cancel_msg(oid: Option<i64>) -> String {
    format!("주문 {} 취소를 확인하지 못했습니다 — 거래소에서 직접 확인하세요", id_str(oid))
}

/// 내 지정가가 시장에서 밀렸나. 키는 **주문 측면**(BUY/SELL, 2026-09-15) -- 진입 롱과 청산 숏은 둘 다 BUY 라 역학이 같다(E13).
pub fn drifted(order_side: &str, price: f64, bid: f64, ask: f64) -> bool {
    if order_side == "BUY" { bid > price } else { ask < price }
}

/// «지금 걸면 테이커가 된다»(-5022). 🔴코드와 **문구 둘 다** 본다(2026-09-20 실사고: code 가 빠진 문자열이라 리페그 0회로 포기, E3).
pub fn is_post_only_reject(err: &str) -> bool {
    err.contains("5022") || err.contains("could not be executed as maker")
}

async fn maker_price<E: Exchange>(ex: &mut E, side: &str, symbol: &str) -> Result<(f64, f64, f64), String> {
    let (bid, ask) = ex.book(symbol).await?;
    Ok((if side == "BUY" { bid } else { ask }, bid, ask))
}

/// 주문이 **더 이상 살아있지 않음을 확인**하고 확정 체결량을 돌려준다. 확인 못 하면 None.
/// 🔴건너뛰면 «취소 실패 → 그래도 재호가» 가 되어 지정가가 겹겹이 산다 -- 2026-09-13 모의 거래소에서 **41개** 동시 생존(2.0 ETH 포지션에 82 ETH 매도).
/// 취소 응답이 아니라 **재조회 상태**를 믿는다(취소 직전 체결 · 취소 실패인데 주문 멀쩡 둘 다 있다).
pub async fn ensure_closed<E: Exchange>(ex: &mut E, c: &Common<'_>, oid: Option<i64>, tries: u32) -> Option<f64> {
    for _ in 0..tries.max(1) {
        let _ = ex.cancel_order(c, &OrderRef::Id(oid)).await;
        if let Ok(f) = ex.get_order(c, &OrderRef::Id(oid)).await {
            if terminal(&f.status) {
                return Some(f.executed_qty);
            }
        }
        ex.sleep(POLL_SEC.min(1.0)).await;
    }
    None
}

/// 지정가 POST 응답이 «나갔는지 모름»이면 표식(cid)으로 찾아 본다. Ok = 찾은 주문 또는 확정 응답(접수·4xx 거부) 그대로 ·
/// Err(사유) = 끝내 모름 -- 표식으로 DELETE 를 보낸 뒤(고아 방지, 검증 D3) 호출부가 Phase::Error 로 멈춘다.
/// 2026-11-20 D2 해결: 따라가기(fill_maker = 진입·청산·손절 청산)·기다리기·지정가 대기가 **이 함수 하나**를 쓴다 -- 따로 두었더니 지정가 대기만 방어가 있었다(2026-09-13 교훈).
pub async fn resolve_unknown_post<E: Exchange>(ex: &mut E, c: &Common<'_>, cid: &str, order: Reply) -> Result<Reply, String> {
    let Err(e0) = &order else { return Ok(order) };
    // 5xx·-1007·-1006 = «체결 상태 불명»(검증 #1) -- 4xx 확정 거부와 다르다. 숫자로 시작 안 함 = 네트워크·시간초과
    if !(!e0.starts_with(|ch: char| ch.is_ascii_digit()) || e0.starts_with('5') || e0.contains("-1007") || e0.contains("-1006")) {
        return Ok(order);
    }
    for _ in 0..3 {
        // 한 번 못 찾았다고 «없다»가 아니다(처리 전일 수 있다) -- 세 번 본다(E7)
        if let Ok(f) = ex.get_order(c, &OrderRef::Client(cid.into())).await {
            if truthy_id(&f) {
                return Ok(Ok(f));
            }
        }
        ex.sleep(1.0).await;
    }
    // 응답은 안 본다 -- «없음»이든 실패든 결과는 같다(멈춤)
    let _ = ex.cancel_order(c, &OrderRef::Client(cid.into())).await;
    Err(format!("주문 접수 여부를 확인하지 못했습니다({e0}) — 거래소에서 직접 확인하세요"))
}

pub struct Common<'a> {
    pub symbol: &'a str,
    pub side: &'a str,
    pub position_side: &'a str,
}

/// GTX 로 `total` 을 채운다 -- 호가가 달아나면 취소 **확인** 뒤 새 호가에 다시 건다. 진입·청산이 이 루프 **한 벌**(2026-09-15, E17).
/// 반환 (체결량, 마지막 지정가, 리페그 수, 오류(phase, 사유)).
pub async fn fill_maker<E: Exchange>(ex: &mut E, c: &Common<'_>, mut price: f64, total: f64, deadline: f64, s: &Shared)
    -> (f64, f64, u32, Option<(Phase, String)>) {
    let (mut done, mut repegs) = (0.0, 0u32);
    // 2026-10-11 state.action = cancel(진입 «대기 취소»)이면 다음 조회(≤3초)에 취소 확인 뒤 멈춘다 -- 잔량 처리는 호출부.
    let cancel = |s: &Shared| st(s).action == Some(Action::Cancel);
    while ex.now() < deadline && r8(total - done) > 0.0 && repegs <= REPEG_MAX && !cancel(s) {
        let remaining = r8(total - done);
        let cid = format!("{PEG_TAG}{}{}", &c.position_side[..1], ex.wall_ms());
        let req = LimitOrder { symbol: c.symbol.into(), side: c.side.into(), position_side: c.position_side.into(), price: r8(price), quantity: remaining, client_order_id: cid.clone() };
        let placed = ex.place_limit_gtx(&req).await;
        // 🔴나갔는지 모르는 채로 «거부»로 치면 살아 있는 GTX 가 손절 없이 체결된다(E6·D2) -- 끝내 모르면 또 걸지 않고 멈춘다(청산·손절 청산도)
        let order = match resolve_unknown_post(ex, c, &cid, placed).await {
            Ok(r) => r,
            Err(msg) => return (done, price, repegs, Some((Phase::Error, msg))),
        };
        let order = match order {
            Ok(o) => o,
            Err(e) => {
                // -5022 = 실패가 아니라 «호가가 움직였다» → 새 호가로 다시. 그 밖 오류는 멈춤 -- 몰래 테이커로 바꾸지 않는다(E4).
                if is_post_only_reject(&e) && repegs < REPEG_MAX {
                    match maker_price(ex, c.side, c.symbol).await {
                        // 걸린 주문 없음 -- 예외로 새면 phase 가 working 에 영영 묶인다(E5)
                        Err(x) => return (done, price, repegs, Some((Phase::Rejected, format!("호가 조회 실패 {x}")))),
                        Ok((p, _, _)) => price = p,
                    }
                    repegs += 1;
                    let mut g = st(s);
                    (g.repegs, g.limit_price) = (repegs, price);
                    continue;
                }
                return (done, price, repegs, Some((Phase::Rejected, e)));
            }
        };
        let oid = order.order_id;
        {
            let mut g = st(s);
            (g.order_id, g.limit_price) = (oid, price);
        }
        let (mut this_filled, mut status, mut need_repeg, mut last_chk) = (0.0, String::new(), false, ex.now());
        while ex.now() < deadline && !cancel(s) {
            ex.nap(POLL_SEC).await;
            let Ok(cur) = ex.get_order(c, &OrderRef::Id(oid)).await else { continue }; // 조회 실패는 재시도 -- 주문은 살아 있다(E12)
            this_filled = cur.executed_qty;
            status = cur.status;
            st(s).filled = r8(done + this_filled);
            if terminal(&status) {
                break;
            }
            if ex.now() - last_chk < POLL_SEC {
                continue; // 2026-11-09 체결 알림에 일찍 깨면 상태만 -- 리페그 판단은 3초마다(E16, 알림 빈도로 REPEG_MAX 가 일찍 바닥나지 않게)
            }
            last_chk = ex.now();
            let Ok((now_px, bid, ask)) = maker_price(ex, c.side, c.symbol).await else { continue }; // 호가 실패 = 앞지름 모름, 주문은 살아 있다
            if drifted(c.side, price, bid, ask) {
                (need_repeg, price) = (true, now_px);
                break;
            }
        }
        if !terminal(&status) {
            match ensure_closed(ex, c, oid, 3).await {
                // 살아 있는지 모르는 채로 **또 걸면 안 된다** -- 멈추고 사람에게. 마지막에 본 체결량(하한)을 돌려준다(E9 -- 빼면 부분 체결에 손절이 안 걸림).
                None => return (r8(done + this_filled), price, repegs, Some((Phase::Error, cancel_msg(oid)))),
                Some(q) => this_filled = q,
            }
        }
        done = r8(done + this_filled);
        st(s).filled = done;
        if !need_repeg {
            break;
        }
        repegs += 1;
        let mut g = st(s);
        (g.repegs, g.limit_price) = (repegs, price);
    }
    (done, price, repegs, None)
}

fn finish(s: &Shared, phase: Phase) {
    let mut g = st(s);
    (g.phase, g.done) = (phase, true);
}

/// 시장가 다리(원본 signed POST MARKET). 오류면 taker_failed(taker_qty 0 · error) -- 원본은 불명 응답도 조회하지 않는다(그대로).
/// 성공이면 체결량 = 응답 executedQty, 0 이면 보낸 수량(«ACK 면 전량», E10). `filled_of` 가 상태 filled 를 정한다(경로마다 원본이 다르다 -- D3).
async fn taker<E: Exchange>(ex: &mut E, c: &Common<'_>, qty: f64, s: &Shared, filled_of: impl Fn(f64, f64) -> f64, reason: Option<&'static str>) {
    match ex.place_market(c, qty).await {
        Err(e) => {
            let mut g = st(s);
            (g.taker_qty, g.error) = (0.0, Some(e));
            drop(g);
            finish(s, Phase::TakerFailed);
        }
        Ok(o) => {
            let done = if o.executed_qty != 0.0 { o.executed_qty } else { qty };
            let mut g = st(s);
            g.filled = filled_of(g.filled, done);
            (g.taker_qty, g.taker_order_id) = (done, o.order_id);
            if reason.is_some() {
                g.taker_reason = reason;
            }
            drop(g);
            finish(s, Phase::FilledTaker);
        }
    }
}

/// peg 를 걸고 호가가 달아나면 따라가며 다시 걸다가 마감에서 멈춘다(파이썬: 마감 잔량 시장가).
pub async fn run_entry<E: Exchange>(ex: &mut E, plan: &ExecPlan, s: &Shared) {
    if !head(ex, plan, s, true).await {
        return;
    }
    let c =Common { symbol: &plan.symbol, side: &plan.side, position_side: &plan.position_side };
    let total = plan.quantity;
    {
        let mut g = st(s);
        (g.phase, g.filled, g.limit_price, g.quantity, g.repegs, g.deadline_sec) = (Phase::Working, 0.0, plan.price, total, 0, FALLBACK_SEC);
    }
    // M2(2026-10-10 스위칭 반대 진입·감시 진입 TAKER): 지정가 없이 아래 시장가 다리로 바로
    let (done, price, repegs, err) = if plan.market { (0.0, plan.price, 0, None) } else { fill_maker(ex, &c, plan.price, total, ex.now() + FALLBACK_SEC, s).await };
    {
        let mut g = st(s);
        (g.filled, g.repegs, g.limit_price) = (done, repegs, price);
    }
    if let Some((ph, e)) = err {
        st(s).error = Some(e);
        return finish(s, ph);
    }
    let remaining = r8(total - st(s).filled);
    if remaining <= 0.0 {
        st(s).taker_qty = 0.0;
        return finish(s, Phase::FilledMaker);
    }
    if st(s).action == Some(Action::Cancel) {
        // 2026-10-11 대기 취소 -- 잔량은 시장가로 안 넘긴다(든 몫은 그대로, 손절은 호출부, E18)
        let mut g = st(s);
        g.cancel_reason = Some(g.action_reason.clone().filter(|r| !r.is_empty()).unwrap_or_else(|| "사용자 취소".into()));
        drop(g);
        return finish(s, Phase::Cancelled);
    }
    if plan.maker_only {
        // 2026-11-09 감시 진입: 사용자 «지정가 체결은 절대 시장가로 바꾸지 않는다» -- 따라가기 마감 뒤 잔량은 버린다(E25)
        st(s).cancel_reason = Some("따라가기 마감 — 남은 수량 버림(시장가 없음)".into());
        return finish(s, Phase::Cancelled);
    }
    // M1: 따라가기 마감 잔량 = 시장가(사용자 선택 b · 2026-11-21 «원본 동작 유지») -- 체결량은 응답에서 읽는다(«보통»을 적지 않는다)
    taker(ex, &c, remaining, s, |f, d| r8(f + d), None).await;
}

/// 메이커로 포지션을 닫는다(진입과 같은 루프 · 리페그). 🔴헤지 모드라 reduceOnly 없음 -- 과청산 방어는 수량(`total - done`)뿐.
pub async fn run_exit<E: Exchange>(ex: &mut E, plan: &ExecPlan, s: &Shared) {
    if !head(ex, plan, s, false).await {
        return;
    }
    let c =Common { symbol: &plan.symbol, side: &plan.side, position_side: &plan.position_side };
    let total = plan.quantity;
    if plan.market {
        // M6·M7·M8: 계획이 MARKET(변동성 ≥30 · 먼지 · exec=market) -- 지정가 없이 전량 시장가 · 체결량 = 응답(스위칭이 «다 닫혔나»를 본다), ACK 면 전량
        {
            let mut g = st(s);
            (g.phase, g.mode, g.quantity, g.filled, g.repegs, g.market_reason) = (Phase::Working, "exit", total, 0.0, 0, plan.market_reason.clone());
        }
        return taker(ex, &c, total, s, |_, d| d, None).await;
    }
    // 마감은 **계획이 정한다** -- 모듈 상수를 쓰면 변동성 연동이 조용히 무력화된다(P27).
    let window = if plan.fallback_after_sec != 0.0 { plan.fallback_after_sec } else { FALLBACK_SEC };
    {
        let mut g = st(s);
        (g.phase, g.mode, g.quantity, g.filled, g.limit_price, g.repegs, g.deadline_sec, g.vol_bpm) =
            (Phase::Working, "exit", total, 0.0, plan.price, 0, window, plan.vol_bpm);
    }
    let (done, _price, repegs, err) = fill_maker(ex, &c, plan.price, total, ex.now() + window, s).await;
    if let Some((ph, e)) = err {
        let mut g = st(s);
        (g.error, g.filled, g.repegs) = (Some(e), done, repegs);
        drop(g);
        return finish(s, ph);
    }
    let remaining = r8(total - done);
    st(s).repegs = repegs;
    if remaining <= 0.0 {
        st(s).taker_qty = 0.0;
        return finish(s, Phase::FilledMaker);
    }
    // M9: 마감 잔량 = 시장가(2026-11-21 «원본 동작 유지»). D3 원본 그대로: taker_qty = 보낸 잔량 · filled = 전량(응답 체결량을 안 읽는다)
    match ex.place_market(&c, remaining).await {
        Err(e) => {
            let mut g = st(s);
            (g.taker_qty, g.error) = (0.0, Some(e));
            drop(g);
            finish(s, Phase::TakerFailed);
        }
        Ok(o) => {
            let mut g = st(s);
            (g.taker_qty, g.filled, g.taker_order_id) = (remaining, total, o.order_id);
            drop(g);
            finish(s, Phase::FilledTaker);
        }
    }
}

/// 감시 진입(2026-11-09 사용자 «롱은 지금가 위·숏은 아래 선을 감시하다가 걸기»): 닿기 전엔 거래소에 주문이 없다 · 닿으면 따라가기(maker_only) --
/// 마감 뒤 잔량은 버린다(사용자 «지정가 체결은 절대 시장가로 바꾸지 않는다», E25). 1초마다 `touched(since)`(since = epoch 초 -- 호출부가 1초 체결가로 본다,
/// 못 읽으면 false = 안 들어간다). amend = 선 옮김(닿음 판정을 그 시각부터 다시) · action cancel = 취소 · peg = 지금 닿은 것으로 · market = 시장가(미정 M5).
pub async fn run_entry_stop<E: Exchange>(ex: &mut E, plan: &ExecPlan, s: &Shared, mut touched: impl FnMut(f64) -> bool, max_sec: f64) {
    let mut plan = plan.clone();
    {
        let mut g = st(s);
        (g.phase, g.mode, g.stop, g.triggered, g.filled, g.limit_price, g.quantity, g.amends, g.deadline_sec) =
            (Phase::Working, "limit", true, None, 0.0, plan.price, plan.quantity, 0, max_sec);
        g.wait_until = Some(ex.wall_ms() as f64 / 1000.0 + max_sec);
    }
    let (mut since, deadline) = (ex.wall_ms() as f64 / 1000.0, ex.now() + max_sec);
    while st(s).action.is_none() {
        let am = st(s).amend.take();
        if let Some(am) = am {
            // 수정 API 가 plan(선·수량)을 이미 바꿨다 -- 옛 선에 닿았던 가격은 버린다
            since = ex.wall_ms() as f64 / 1000.0;
            let Some(q) = am.qty else {
                st(s).error = Some("집행 오류: KeyError: 'qty' — 거래소에서 직접 확인하세요".into()); // 원본 float(am["qty"]) 예외 → settle
                return finish(s, Phase::Error);
            };
            // ponytail: 원본은 수정 API 가 plan dict 를 제자리에서 바꾼다 -- 여기선 amend 칸의 수량을 계획 사본에 싣는다(같은 값이라는 가정, 접수 조각에서 대조)
            plan.quantity = q;
            let mut g = st(s);
            (g.limit_price, g.quantity, g.amends) = (am.price, q, g.amends + 1);
        }
        if touched(since) {
            break;
        }
        if ex.now() >= deadline {
            let mut g = st(s);
            (g.taker_qty, g.cancel_reason) = (0.0, Some("대기 시간 만료".into()));
            drop(g);
            return finish(s, Phase::Cancelled);
        }
        ex.sleep(1.0).await;
    }
    let action = {
        let mut g = st(s);
        if g.action == Some(Action::Peg) {
            g.action = None; // PEG 버튼 = 지금 닿은 것으로 치고 따라가기
        }
        g.action
    };
    if action == Some(Action::Market) {
        // M5: run_entry({**plan, "type": "MARKET"}) = 전량 시장가(TAKER 두 번 · 2026-11-21 «원본 동작 유지»)
        return run_entry(ex, &ExecPlan { market: true, ..plan }, s).await;
    }
    if action.is_some() {
        let mut g = st(s);
        g.taker_qty = 0.0;
        g.cancel_reason = Some(g.action_reason.clone().filter(|r| !r.is_empty()).unwrap_or_else(|| "사용자 취소".into()));
        drop(g);
        return finish(s, Phase::Cancelled);
    }
    st(s).triggered = Some(ex.wall_ms() as f64 / 1000.0);
    // D11: 원본은 이 호가 조회가 try 밖이라 예외가 settle 의 «집행 오류»로 간다(그 시점 걸린 주문 없음)
    let price = match maker_price(ex, &plan.side, &plan.symbol).await {
        Ok((p, _, _)) => p,
        Err(x) => {
            st(s).error = Some(format!("집행 오류: 호가 조회 실패 {x} — 거래소에서 직접 확인하세요"));
            return finish(s, Phase::Error);
        }
    };
    run_entry(ex, &ExecPlan { price, maker_only: true, ..plan }, s).await;
}

/// 지정가 대기(2026-10-31): **사용자가 정한 가격**에 GTX, 따라가지도 시장가로 넘기지도 않는다(E21).
/// action cancel → 취소 확인 뒤 멈춤 · amend → 취소 확인 뒤 새 가격(체결 0 일 때만, E22) · reprice(basis 따라가기, E23) · action peg → 따라가기(E26).
pub async fn run_entry_limit<E: Exchange>(ex: &mut E, plan: &ExecPlan, s: &Shared, max_sec: f64, mut on_fill: bool, reprice: bool) {
    if !head(ex, plan, s, true).await {
        return;
    }
    let c = Common { symbol: &plan.symbol, side: &plan.side, position_side: &plan.position_side };
    let ps = &plan.position_side;
    let (mut total, mut price) = (plan.quantity, plan.price);
    {
        let mut g = st(s);
        (g.phase, g.mode, g.filled, g.limit_price, g.quantity, g.repegs, g.amends, g.deadline_sec) = (Phase::Working, "limit", 0.0, price, total, 0, 0, max_sec);
        g.wait_until = Some(ex.wall_ms() as f64 / 1000.0 + max_sec);
    }
    let deadline = ex.now() + max_sec;
    let (mut done, mut outside, mut rej) = (0.0, false, 0u32);
    while r8(total - done) > 0.0 && ex.now() < deadline && st(s).action.is_none() {
        let remaining = r8(total - done);
        let cid = format!("{WAIT_TAG}{}{}", &ps[..1], ex.wall_ms());
        let req = LimitOrder { symbol: c.symbol.into(), side: c.side.into(), position_side: ps.clone(), price: r8(price), quantity: remaining, client_order_id: cid.clone() };
        let placed = ex.place_limit_gtx(&req).await;
        let order = match resolve_unknown_post(ex, &c, &cid, placed).await {
            Ok(r) => r,
            Err(msg) => {
                let mut g = st(s);
                (g.filled, g.error) = (done, Some(msg));
                drop(g);
                return finish(s, Phase::Error);
            }
        };
        let order = match order {
            Ok(o) => o,
            Err(err) => {
                let (n, acted) = { let g = st(s); (g.amends + g.reprices, g.action.is_some()) };
                if is_post_only_reject(&err) && n > 0 && rej < 3 && !acted {
                    rej += 1; // 수정·재조정 직후 호가가 넘어갔다(검증 D6) -- 새 목표가로 다시(연속 최대 3번)
                    if reprice {
                        if let Ok(Some(px)) = ex.reprice().await {
                            if px != 0.0 {
                                price = px;
                                st(s).limit_price = price;
                            }
                        }
                    }
                    ex.sleep(1.0).await;
                    continue;
                }
                let mut g = st(s);
                g.filled = done;
                g.error = Some(if is_post_only_reject(&err) { "지정가가 호가를 넘어 지금 걸면 체결됩니다(메이커 불가) — 가격을 호가 안쪽으로 옮기세요".into() } else { err });
                drop(g);
                return finish(s, Phase::Rejected);
            }
        };
        let oid = order.order_id;
        rej = 0; // 접수 성공 -- -5022 재시도 횟수는 «연속» 기준
        {
            let mut g = st(s);
            (g.order_id, g.client_order_id, g.limit_price) = (oid, Some(cid), price);
        }
        let (mut this_filled, mut status, mut last_chk) = (0.0, String::new(), ex.now());
        while ex.now() < deadline && { let g = st(s); g.action.is_none() && g.amend.is_none() } {
            ex.nap(POLL_SEC).await;
            let Ok(cur) = ex.get_order(&c, &OrderRef::Id(oid)).await else { continue };
            (this_filled, status) = (cur.executed_qty, cur.status);
            st(s).filled = r8(done + this_filled);
            if this_filled > 0.0 && on_fill {
                ex.on_fill().await;
                on_fill = false;
            }
            if terminal(&status) {
                break;
            }
            if reprice && done + this_filled == 0.0 && st(s).reprices < REPRICE_MAX && ex.now() - last_chk >= REPRICE_SEC {
                // 체결 시작 뒤엔 안 건드린다(큐 자리 · 부분 체결 수량 어긋남)
                last_chk = ex.now();
                let px = ex.reprice().await.ok().flatten().filter(|p| *p != 0.0);
                let mut g = st(s);
                if let Some(px) = px {
                    if (px / price - 1.0).abs() >= REPRICE_MIN_REL && g.amend.is_none() && g.action.is_none() {
                        g.amend = Some(Amend { price: px, qty: None, auto: true });
                    }
                }
            }
        }
        outside = terminal(&status) && status != "FILLED"; // 우리가 취소하기 전에 끝났다 = 거래소(앱)에서 취소됨
        if !terminal(&status) {
            match ensure_closed(ex, &c, oid, 3).await {
                None => {
                    let mut g = st(s);
                    (g.filled, g.error) = (r8(done + this_filled), Some(cancel_msg(oid)));
                    drop(g);
                    return finish(s, Phase::Error);
                }
                Some(q) => this_filled = q,
            }
        }
        done = r8(done + this_filled);
        st(s).filled = done;
        if outside {
            break;
        }
        if this_filled > 0.0 && on_fill {
            ex.on_fill().await; // 취소 확인 중에 체결된 몫도 손절을 건다(E24)
            on_fill = false;
        }
        let mut g = st(s);
        if let Some(am) = g.amend.take() {
            // 🔴한 주라도 체결됐으면 가격 수정은 버린다(한 포지션이 두 가격에 체결, 검증 D4) -- 남은 수량은 옛 가격에 다시
            if g.action.is_none() && done == 0.0 {
                price = am.price;
                total = am.qty.filter(|q| *q != 0.0).unwrap_or(total);
                g.quantity = total;
                g.limit_price = price;
                if am.auto { g.reprices += 1 } else { g.amends += 1 }
            }
        }
    }
    let remaining = r8(total - done);
    let action = st(s).action;
    if remaining <= 0.0 {
        st(s).taker_qty = 0.0;
        finish(s, Phase::FilledMaker);
    } else if action == Some(Action::Market) && !outside {
        // M4: 걸린 지정가를 취소 확인한 뒤(위 ensure_closed) 남은 수량만 시장가(2026-11-04 · 2026-11-21 «원본 동작 유지»)
        st(s).filled = r8(done);
        taker(ex, &c, remaining, s, |f, d| r8(f + d), Some("지금 시장가(사용자)")).await;
    } else if action == Some(Action::Peg) && !outside {
        // 2026-11-09 사용자 «taker 말고 peg 버튼»: 취소 확인 뒤 남은 수량을 최우선 호가 따라가기 · 마감 뒤 잔량은 버림(시장가 없음, E26).
        {
            let mut g = st(s);
            g.action = None; // fill_maker 는 action=cancel 만 멈춤으로 본다 -- 비워 둬야 따라가는 중 ✕ 취소가 먹는다
            g.pegged = true;
        }
        // D11: 원본은 이 호가 조회가 try 밖이라 예외가 settle 의 «집행 오류»로 간다(그 시점 걸린 주문 없음). 같은 결과(error)로 옮긴다.
        let px = match maker_price(ex, c.side, c.symbol).await {
            Ok((p, _, _)) => p,
            Err(x) => {
                st(s).error = Some(format!("집행 오류: 호가 조회 실패 {x} — 거래소에서 직접 확인하세요"));
                return finish(s, Phase::Error);
            }
        };
        let (got, px, rp, err) = fill_maker(ex, &c, px, remaining, ex.now() + FALLBACK_SEC, s).await;
        let mut g = st(s);
        (g.filled, g.repegs, g.limit_price, g.taker_qty) = (r8(done + got), rp, px, 0.0);
        let ph = if let Some((ph, e)) = err {
            g.error = Some(e);
            ph
        } else if r8(remaining - got) <= 0.0 {
            Phase::FilledMaker
        } else {
            g.cancel_reason = Some(g.action_reason.clone().filter(|r| !r.is_empty()).unwrap_or_else(|| "따라가기 마감 — 남은 수량 버림(시장가 없음)".into()));
            Phase::Cancelled
        };
        drop(g);
        finish(s, ph);
    } else {
        let mut g = st(s);
        g.taker_qty = 0.0;
        g.cancel_reason = Some(if action.is_some() {
            g.action_reason.clone().filter(|r| !r.is_empty()).unwrap_or_else(|| "사용자 취소".into())
        } else if outside { "거래소에서 취소됨".into() } else { "대기 시간 만료".into() });
        drop(g);
        finish(s, Phase::Cancelled);
    }
}
