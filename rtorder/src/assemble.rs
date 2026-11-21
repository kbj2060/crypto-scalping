//! 계획 조립(2026-11-21 넷째 조각) -- `dashboard/orders.py` 의 `assemble_entry_plan`·`assemble_exit_plan` 과 그 도우미
//! (`notional_sum`·`pool_balance`·`pool_positions`·`entry_projection`·`effective_cap`·`basis_for`·`remaining_hold`·`risk_sizing`)와
//! `scripts/live_eth_trade_plan_20260913.py` 의 `plan_now`·`execution_plan`·`leverage_setting` 중 **지금 스위치에서 닿는 몫**.
//! 순수 계산은 `plan.rs`(build_entry_plan 등) -- 여기는 «입력을 모아 넣고 상한·차단 사유·화면 칸을 붙이는» 부분.
//!
//! 입력은 둘로 나뉜다(사용자 결정 «받아 둔 값»): 시세(호가·필터·변동성·펀딩·수수료·마진 방식) = `Quotes` · 계좌·화면 상태 = `DeskIo`.
//! `Quotes` 구현은 둘 -- (가) 매번 묻는 것(`impl Quotes for Binance` · 시험의 가짜) · (나) `Cached`(값마다 나이를 달고, 한도보다 낡았으면 (가)로 떨어진다).
//!
//! 🔴사이징 스위치(P30)는 지금 값 그대로 상수: 위험모델 끔 · 모델 상한만 · 증거금 % 상한 0 · 규칙 켬 · 자동 물타기 켬.
//! 위험모델이 꺼져 있어 사이징은 늘 단순 규칙(`SIMPLE_SIZING`) -- 위험 표·atr 가 없으니 보유 지평 권고·처방·사다리는 «없음» 갈래만 닿는다.
//! ponytail: 위험모델이 켜졌을 때의 갈래(recommend_hold·prescribe·entry_notional·exit_fraction_required)는 옮기지 않았다 -- 스위치를 켜면 먼저 옮긴다
//! (`SIZING_RISK_MODEL_ENABLED` 를 true 로 바꾸면 컴파일 단언이 막는다).

use crate::exec::FALLBACK_SEC as EXEC_FALLBACK_SEC;
use crate::plan::{self, Filters, PlanError};
use crate::pyfmt::{self, fixed, round_n};
use crate::sign::{py_float, py_str, truthy};
use crate::watch::{busy, parse_iso, Watcher};
use serde_json::{json, Value};
use std::collections::HashMap;

pub const SIZING_RISK_MODEL_ENABLED: bool = false; // 2026-09-27 사용자 «변동성 모델이 필요없다고 판명 났으니 제거»
pub const SIZING_CAP_MODEL_ONLY: bool = true; // 2026-09-25 위험모델 상한만(모델 없으면 정책 천장 HARD_CAP_X)
pub const SIZING_MARGIN_CAP_PCT: f64 = 0.0; // 2026-10-09 증거금 상한 해제(되돌리기 = 50)
pub const SIZING_CAP_EQUITY_X: f64 = 6.0;
pub const HARD_CAP_X: f64 = 25.0; // live_eth_risk_sizing_policy 정책 상한
pub const HOLD_FIXED_MIN: i64 = 240;
pub const HOLD_CHOICES: [i64; 5] = [60, 120, 240, 480, 1440];
pub const EXCHANGE_MAX_LEVERAGE: i64 = 150;
pub const LEVERAGE_STEPS: [i64; 18] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 75, 100, 125, 150];
pub const ENFORCE_SLACK: f64 = 1.25;
const _: () = assert!(!SIZING_RISK_MODEL_ENABLED, "위험모델 갈래는 옮기지 않았다 -- 켜기 전에 recommend_hold·prescribe 를 옮긴다");

// trade_plan 표시 상수(live_eth_trade_plan_20260913)
const ROUND_TRIP_COST_BP: f64 = 5.88;
const STOP_EXTRA_COST_BP: f64 = 16.08;
const FUNDING_BP_8H_FALLBACK: f64 = 0.52;
const STOP_HIT_RATE_BY_HOLD: [(i64, f64, f64); 5] = [(60, 0.0085, 0.0064), (120, 0.0230, 0.0178), (240, 0.0574, 0.0457), (480, 0.1229, 0.1066), (1440, 0.3195, 0.3007)];
const FILL_K: f64 = 52.0;
const FILL_EXP: f64 = 1.472;
const MAKER_BP: f64 = 2.0;
const TAKER_BP: f64 = 5.0;
const REBALANCE_BAND: f64 = 0.25;

pub const MARKET_SYMBOLS: [(&str, &str); 9] = [("eth", "ETHUSDT"), ("sol", "SOLUSDT"), ("btc", "BTCUSDT"), ("xrp", "XRPUSDT"), ("hype", "HYPEUSDT"),
    ("sndk", "SNDKUSDT"), ("soxl", "SOXLUSDT"), ("skhynix", "SKHYNIXUSDT"), ("mu", "MUUSDT")];

pub fn market_symbol(asset: &str) -> Option<&'static str> {
    MARKET_SYMBOLS.iter().find(|x| x.0 == asset).map(|x| x.1)
}

/// 시세 원천(사용자 결정 «받아 둔 값» 의 trait). Err = 파이썬 예외 문자열(조립이 502 로 받는다).
#[allow(async_fn_in_trait)]
pub trait Quotes {
    /// /fapi/v1/ticker/bookTicker 의 dict({bidPrice, askPrice}).
    async fn book(&mut self, symbol: &str) -> Result<Value, String>;
    async fn filters(&mut self, symbol: &str) -> Result<Filters, String>;
    /// realized_vol_now -- 실패는 None(필수 아님).
    async fn vol(&mut self, symbol: &str) -> Option<f64>;
    /// funding_now(bp/8h) -- 실패는 None.
    async fn funding(&mut self, symbol: &str) -> Option<f64>;
    /// commission_bp dict(표시 전용 · P29).
    async fn fees(&mut self, symbol: &str) -> Value;
}

/// 조립이 닿는 계좌·화면 상태.
#[allow(async_fn_in_trait)]
pub trait DeskIo {
    fn now(&self) -> f64;
    fn exec_enabled(&self) -> bool;
    /// fresh=true = produce_account(새로 읽기 · 청산은 스트림 계좌 가능 -- account::fresh_account) · false = swr 캐시.
    async fn account(&mut self, fresh: bool) -> Result<Value, String>;
    /// 1초 마지막 체결가 [(초, 가격)] · 원천 없는 코인 None.
    fn sec_prices(&mut self, asset: &str) -> Option<Vec<(f64, f64)>>;
    /// 손절 해제(접수 문) -- bracket::clear_bracket.
    async fn clear_bracket(&mut self, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Result<Value, String>;
    /// 선 핸들러(익절 켜기·선 추가·선 수정) -- bracket::place_bracket(strict).
    async fn place_bracket(&mut self, spec: &Value, symbol: &str, pside: &str, strict: bool) -> Result<Value, String>;
    /// 분할 익절(조각 조작·선 추가) -- bracket::place_tp_ladder. Err = 원본에서 새던 예외(→ 500).
    async fn place_tp_ladder(&mut self, legs: &Value, symbol: &str, pside: &str) -> Result<Value, String>;
    /// 화면 /api/binance-account(원본 api_binance_account): fresh = 잠금 안에서 1초 안 받은 값이면 그것 · 아니면 새로 읽어 캐시에(A3) · 아니면 swr 캐시. 기본 = account(fresh).
    async fn account_screen(&mut self, fresh: bool) -> Result<Value, String> {
        self.account(fresh).await
    }
}

// ── (나) 받아 둔 값 ────────────────────────────────────────────────────────────
/// 값마다 허용 나이(초). 근거:
/// - 호가 2초: 원본은 발주마다 새로 묻는다(캐시 없음). 코어의 웹소켓 bookTicker 는 바뀔 때마다 오므로(실서버 70초에 45,883건) 2초 넘게 안 바뀌었다면
///   흐름이 멈춘 것으로 본다. 이 값이 «지금 호가»라는 원본의 뜻을 지키는 가장 느슨한 선. 환산비(basis)도 같은 호가 두 개라 같은 한도.
/// - 필터 24시간: 원본은 프로세스 수명 동안 캐시(심볼 규격은 안 변한다) -- 하루마다 새로 읽는 것은 더 보수적.
/// - 수수료 1시간: 원본 FEE_TTL_SEC(성공 3600초) 그대로 · 표시 전용(P29).
/// - 펀딩 5분: 원본은 매번 묻지만 화면 «예상 비용» 표시 전용이고 premiumIndex 의 lastFundingRate 는 정산(8시간) 사이에 천천히 움직인다.
/// - 변동성 60초: 마감 1분봉 30개로 잰다(마지막 미완결 봉은 버림) -- 1분에 한 번만 바뀐다. 청산 마감·시장가 문턱(M6)에 들어가므로 1분을 넘기지 않는다.
/// - 마진 방식: 받아 두지 않는다(늘 (가)) -- 자동 규칙 진입의 안전 검사(G17 · 10-09 격리 마진 사고)라 사용자가 거래소에서 바꾼 직후를 놓치면 안 된다.
pub const MAX_AGE_BOOK: f64 = 2.0;
pub const MAX_AGE_FILTERS: f64 = 86_400.0;
pub const MAX_AGE_FEES: f64 = 3_600.0;
pub const MAX_AGE_FUNDING: f64 = 300.0;
pub const MAX_AGE_VOL: f64 = 60.0;

/// (나) 실시간 코어가 받아 둔 최신 값을 주는 원천. 값마다 받은 시각을 달고, 한도보다 낡았거나 없으면 안쪽 (가)에 묻는다(낡은 값으로 **조용히** 내지 않는다).
/// `now` 는 조립 직전에 호출부가 맞춘다. `misses` = (가)로 떨어진 조회(이름, 심볼, 이유) -- 시험·로그가 본다.
pub struct Cached<Q: Quotes> {
    pub inner: Q,
    pub now: f64,
    pub book: HashMap<String, (Value, f64)>,
    pub filters: HashMap<String, (Filters, f64)>,
    pub vol: HashMap<String, (Option<f64>, f64)>,
    pub funding: HashMap<String, (Option<f64>, f64)>,
    pub fees: HashMap<String, (Value, f64)>,
    pub misses: Vec<(String, String, &'static str)>,
}

impl<Q: Quotes> Cached<Q> {
    pub fn new(inner: Q) -> Self {
        Cached { inner, now: 0.0, book: HashMap::new(), filters: HashMap::new(), vol: HashMap::new(), funding: HashMap::new(), fees: HashMap::new(), misses: vec![] }
    }
    pub fn put_book(&mut self, symbol: &str, bid: f64, ask: f64, at: f64) {
        self.book.insert(symbol.into(), (json!({"symbol": symbol, "bidPrice": bid, "askPrice": ask}), at));
    }
    fn fresh<'a, T>(&mut self, name: &'static str, m: &'a HashMap<String, (T, f64)>, symbol: &str, max_age: f64) -> Option<&'a T> {
        match m.get(symbol) {
            Some((v, at)) if self.now - at <= max_age && self.now >= *at => Some(v),
            Some(_) => {
                self.misses.push((name.into(), symbol.into(), "stale"));
                None
            }
            None => {
                self.misses.push((name.into(), symbol.into(), "missing"));
                None
            }
        }
    }
}

impl<Q: Quotes> Quotes for Cached<Q> {
    async fn book(&mut self, symbol: &str) -> Result<Value, String> {
        let m = std::mem::take(&mut self.book);
        let hit = self.fresh("book", &m, symbol, MAX_AGE_BOOK).cloned();
        self.book = m;
        match hit {
            Some(v) => Ok(v),
            None => self.inner.book(symbol).await,
        }
    }
    async fn filters(&mut self, symbol: &str) -> Result<Filters, String> {
        let m = std::mem::take(&mut self.filters);
        let hit = self.fresh("filters", &m, symbol, MAX_AGE_FILTERS).copied();
        self.filters = m;
        match hit {
            Some(v) => Ok(v),
            None => self.inner.filters(symbol).await,
        }
    }
    async fn vol(&mut self, symbol: &str) -> Option<f64> {
        let m = std::mem::take(&mut self.vol);
        let hit = self.fresh("vol", &m, symbol, MAX_AGE_VOL).copied();
        self.vol = m;
        match hit {
            Some(v) => v,
            None => self.inner.vol(symbol).await,
        }
    }
    async fn funding(&mut self, symbol: &str) -> Option<f64> {
        let m = std::mem::take(&mut self.funding);
        let hit = self.fresh("funding", &m, symbol, MAX_AGE_FUNDING).copied();
        self.funding = m;
        match hit {
            Some(v) => v,
            None => self.inner.funding(symbol).await,
        }
    }
    async fn fees(&mut self, symbol: &str) -> Value {
        let m = std::mem::take(&mut self.fees);
        let hit = self.fresh("fees", &m, symbol, MAX_AGE_FEES).cloned();
        self.fees = m;
        match hit {
            Some(v) => v,
            None => self.inner.fees(symbol).await,
        }
    }
}

/// (가) 실제 거래소 -- 원본처럼 매번 묻는다.
impl Quotes for crate::http::Binance {
    async fn book(&mut self, symbol: &str) -> Result<Value, String> {
        self.public("/fapi/v1/ticker/bookTicker", &[("symbol", crate::sign::P::s(symbol))]).await
    }
    async fn filters(&mut self, symbol: &str) -> Result<Filters, String> {
        self.load_filters(symbol).await
    }
    async fn vol(&mut self, symbol: &str) -> Option<f64> {
        // 마지막 봉은 미완결이라 버린다(종가 = 인덱스 4)
        let kl = self.public("/fapi/v1/klines", &[("symbol", crate::sign::P::s(symbol)), ("interval", crate::sign::P::s("1m")),
                                                   ("limit", crate::sign::P::I(plan::EXIT_VOL_WINDOW as i64 + 2))]).await.ok()?;
        let rows = kl.as_array().filter(|a| !a.is_empty())?;
        let closes: Option<Vec<f64>> = rows[..rows.len() - 1].iter().map(|r| r.get(4).and_then(|v| py_float(v).ok())).collect();
        plan::realized_vol_bpm(&closes?)
    }
    async fn funding(&mut self, symbol: &str) -> Option<f64> {
        let r = self.public("/fapi/v1/premiumIndex", &[("symbol", crate::sign::P::s(symbol))]).await.ok()?;
        r.get("lastFundingRate").filter(|v| !v.is_null()).and_then(|v| py_float(v).ok()).map(|x| x * 1e4)
    }
    async fn fees(&mut self, symbol: &str) -> Value {
        crate::bracket::commission_bp(self, symbol).await
    }
}

// ── 도우미(원본 모듈 수준 함수) ──────────────────────────────────────────────────
fn pe(e: PlanError) -> String {
    format!("{}: {}", e.py_type(), e.py_msg())
}
fn fl(v: Option<&Value>) -> Result<f64, String> {
    if truthy(v) { py_float(v.unwrap()) } else { Ok(0.0) }
}
fn key_f(o: &Value, k: &str) -> Result<f64, String> {
    py_float(o.get(k).ok_or_else(|| format!("KeyError: '{k}'"))?)
}
pub fn quote_of(symbol: &str) -> &'static str {
    if symbol.ends_with("USDC") { "USDC" } else { "USDT" }
}
/// |명목| 합 -- 헤지 모드라 상쇄를 가정하지 않는다(2026-09-19 상한이 심볼 하나만 세고 있었다).
pub fn notional_sum(positions: &[Value], symbol: Option<&str>) -> Result<f64, String> {
    let mut s = 0.0;
    for p in positions {
        if symbol.is_none_or(|x| p.get("symbol").and_then(Value::as_str) == Some(x)) {
            s += fl(p.get("notional"))?.abs();
        }
    }
    Ok(s)
}
/// 이 심볼 주문이 증거금을 먹는 풀의 잔고(2026-11-05 단일자산 모드는 USDC·USDT 풀이 따로).
pub fn pool_balance(account: &Value, symbol: &str) -> Value {
    match account.get("balances") {
        None | Some(Value::Null) => account.get("balance").filter(|v| truthy(Some(v))).cloned().unwrap_or(json!({})),
        Some(b) => b.get(quote_of(symbol)).filter(|v| truthy(Some(v))).cloned().unwrap_or(json!({})),
    }
}
pub fn pool_positions(account: &Value, symbol: &str) -> Vec<Value> {
    let ps: Vec<Value> = account.get("positions").and_then(Value::as_array).cloned().unwrap_or_default();
    if truthy(account.get("multi_assets_margin")) || account.get("balances").is_none_or(Value::is_null) {
        ps
    } else {
        ps.into_iter().filter(|p| quote_of(&p.get("symbol").map(py_str).unwrap_or_default()) == quote_of(symbol)).collect()
    }
}

/// «이 주문을 넣으면 계좌 카드가 어떻게 바뀌나»(2026-09-12) -- 계좌 카드 세 타일과 같은 정의.
pub fn entry_projection(plan: &Value, balance: &Value, positions: &[Value], existing: f64, equity: f64) -> Result<Value, String> {
    let used = match balance.get("initial_margin") {
        Some(v) if !v.is_null() => py_float(v)?,
        _ => pyfmt::max(0.0, fl(balance.get("margin"))? - fl(balance.get("available"))?),
    };
    let available = fl(balance.get("available"))?;
    let add_margin = fl(plan.get("margin_usdt"))?;
    let total = fl(plan.get("total_notional_usdt"))?;
    let mut liq_before: Option<f64> = None;
    for p in positions {
        let (mark, liq) = (fl(p.get("mark_price"))?, fl(p.get("liquidation_price"))?);
        if mark > 0.0 && liq > 0.0 {
            let d = (mark - liq).abs() / mark * 100.0;
            liq_before = Some(liq_before.map_or(d, |b| pyfmt::min(b, d)));
        }
    }
    let liq_after = match liq_before {
        Some(b) if total > 0.0 && existing > 0.0 => Some(b * existing / total),
        _ if equity > 0.0 && total > 0.0 => Some(100.0 * equity / total),
        _ => None,
    };
    let side = |m: f64, n: f64, liq: Option<f64>| {
        let eq = equity != 0.0;
        json!({"margin_usdt": round_n(m, 2), "margin_used_pct": if eq { json!(round_n(100.0 * m / equity, 1)) } else { Value::Null },
               "available_usdt": if eq { json!(round_n(equity - m, 2)) } else { Value::Null }, "notional_usdt": round_n(n, 2),
               "exposure_x": if eq { json!(round_n(n / equity, 2)) } else { Value::Null }, "liq_pct": liq.map(|l| round_n(l, 2))})
    };
    let mut out = json!({"equity_usdt": round_n(equity, 2), "available_before_usdt": round_n(available, 2),
              "before": side(used, existing, liq_before), "after": side(used + add_margin, total, liq_after)});
    if positions.is_empty() {
        out["before"]["notional_usdt"] = json!(0); // 원본 existing = 빈 sum() = 정수 0 → round(0, 2) = 0(응답 글자 «0»)
    }
    Ok(out)
}

/// `leverage_setting(...)["setting"]` -- 거래소에 걸 레버리지(정책 상한을 거래소에 새기는 값). 순자산·상한이 없으면 None.
pub fn leverage_setting(cap_notional: f64, equity: f64, current_notional: f64) -> Option<i64> {
    if equity <= 0.0 || cap_notional <= 0.0 {
        return None;
    }
    let min_feasible = cap_notional / equity;
    let position_floor = pyfmt::max(0.0, current_notional) / equity;
    // 🔴창은 열린 구간(x > floor_x) + 상대 엡실론(2026-09-14 하네스 46,000 봉 중 45,148 봉 차단)
    let floor_x = pyfmt::max(min_feasible, position_floor) * (1.0 + 1e-9);
    let hi = (cap_notional / equity) * ENFORCE_SLACK;
    let window: Vec<i64> = LEVERAGE_STEPS.iter().copied().filter(|&x| floor_x < x as f64 && x as f64 <= hi).collect();
    let s = window.iter().max().copied().unwrap_or_else(|| LEVERAGE_STEPS.iter().copied().find(|&x| x as f64 > floor_x).unwrap_or(EXCHANGE_MAX_LEVERAGE));
    Some(s.min(EXCHANGE_MAX_LEVERAGE))
}

/// **남은** 보유시간을 지평 표에 맞춰 올림한다(물타기해도 시계는 안 늘어난다).
pub fn remaining_hold(entry_at: Option<&Value>, fixed_min: i64, now: f64) -> i64 {
    let Some(s) = entry_at.filter(|v| truthy(Some(v))).and_then(Value::as_str) else { return fixed_min };
    // 시간대 없는 값 = 파이썬에서 aware − naive 의 TypeError → fixed
    let tz = s.len() > 10 && (s.ends_with('Z') || s[10..].contains('+') || s[10..].contains('-'));
    let Some(t) = parse_iso(s).filter(|_| tz) else { return fixed_min };
    let age = (now - t) / 60.0;
    let left = fixed_min as f64 - pyfmt::max(0.0, age);
    if left <= 0.0 {
        return HOLD_CHOICES[0];
    }
    HOLD_CHOICES.iter().copied().find(|&h| h as f64 >= left).unwrap_or(fixed_min)
}

fn stop_hit_rate(hold: i64, side: &str) -> f64 {
    let mut best = STOP_HIT_RATE_BY_HOLD[0];
    for r in STOP_HIT_RATE_BY_HOLD {
        if (r.0 - hold).abs() < (best.0 - hold).abs() {
            best = r;
        }
    }
    if side.to_uppercase() == "LONG" { best.1 } else { best.2 }
}
fn funding_cost_bp(hold: i64, side: &str, rate: Option<f64>) -> f64 {
    let r = rate.unwrap_or(FUNDING_BP_8H_FALLBACK);
    (if side.to_uppercase() == "LONG" { r } else { -r }) * (hold as f64 / 480.0)
}

/// 원본 `execution_plan`(표시용 예상 비용·체결 시간 -- 크기·손절은 표준 상수).
fn execution_plan(vol: Option<f64>, hold: i64, side: &str, funding: Option<f64>, fees: &Value) -> Result<Value, String> {
    let fill = vol.filter(|v| truthy(Some(&json!(v))) && *v > 0.0).map(|v| FILL_K / v.powf(FILL_EXP));
    let market = vol.is_some_and(|v| v >= plan::EXIT_TAKER_VOL_BPM);
    let fund = funding_cost_bp(hold, side, funding);
    let p = stop_hit_rate(hold, side);
    let (mk, tk) = if truthy(Some(fees)) { (key_f(fees, "maker_bp")?, key_f(fees, "taker_bp")?) } else { (MAKER_BP, TAKER_BP) };
    let (d_rt, d_stop) = (2.0 * (mk - MAKER_BP), (tk - TAKER_BP) - (mk - MAKER_BP));
    let expected = ROUND_TRIP_COST_BP + p * STOP_EXTRA_COST_BP + fund;
    let fill_r = fill.filter(|x| *x != 0.0).map(|x| round_n(x, 1));
    Ok(json!({
        "vol_bpm": vol.map(|v| round_n(v, 2)),
        "cost": {"round_trip_bp": round_n(ROUND_TRIP_COST_BP + d_rt, 2), "stop_hit_rate": p, "stop_extra_bp": round_n(p * (STOP_EXTRA_COST_BP + d_stop), 2),
                 "funding_bp": round_n(fund, 2), "funding_rate_bp_8h": round_n(funding.unwrap_or(FUNDING_BP_8H_FALLBACK), 3), "funding_is_live": funding.is_some(),
                 "total_bp": round_n(expected + d_rt + p * d_stop, 2),
                 "fee": {"symbol": fees.get("symbol").cloned().unwrap_or(Value::Null), "maker_bp": mk, "taker_bp": tk,
                         "source": fees.get("source").cloned().unwrap_or(json!("fallback")), "promo": mk < MAKER_BP - 1e-9 || tk < TAKER_BP - 1e-9}},
        "entry": {"mode": "peg_gtx", "expected_fill_sec": fill_r, "fallback_sec": plan::FALLBACK_SEC, "maker_bp": mk, "taker_bp": tk,
                  "note": if fill.is_some_and(|x| x > 10.0) { "저변동은 체결이 느리고 미체결이 몰립니다 — 진입 미체결은 무해하니 기다립니다" } else { "빠른 장일수록 메이커가 더 잘 체결됩니다" }},
        "exit": {"mode": if market { "market" } else { "peg_repeg" },
                 "deadline_sec": if market { Value::Null } else { json!(round_n(plan::exit_deadline_sec(vol).map_err(pe)?, 1)) },
                 "expected_fill_sec": fill_r},
    }))
}

/// 원본 `plan_now` 의 위험모델 없는 갈래(위험 표·atr 없음 → 권고·처방 «없음» · 크기 None · 사다리 빈 것).
fn plan_now(side: &str, hold: i64, vol: Option<f64>, funding: Option<f64>, fees: &Value) -> Result<Value, String> {
    let none = json!({"available": false, "reason": "atr_pct 없음"});
    Ok(json!({
        "side": side, "hold": none, "hold_min": hold, "prescription": none,
        "size": {"leverage": null, "safe_mae_pct": null, "total_notional": null, "room_notional": 0.0},
        "execution": execution_plan(vol, hold, side, funding, fees)?,
        "entry_split": {"tranches": 1, "spread_min": 0, "reason": "일괄 — 분할은 «같은 평균 노출의 작은 단일 진입»에 869일 6/6 칸 전패", "room_notional": 0.0,
                        "rule": "추가는 상한 여유 안에서만 하면 됩니다 — 실계좌 69왕복에서 분할 자체는 건당 수익률과 무관했고(상관 −0.04), 손실을 만든 건 분할이 데려온 크기(상관 +0.51)와 보유시간이었습니다"},
        "exit_ladder": {"budget_min": 0, "ladder": [], "rebalance_band": REBALANCE_BAND, "note": "지금 크기는 어떤 보유시간 예산도 넘습니다 — 사다리대로 줄이세요"},
    }))
}

/// 원본 `risk_sizing` -- 위험모델이 꺼져 있어 늘 «없음»(사이징은 단순 규칙이라 available=True → worker_no_risk).
fn risk_none(hold: i64) -> Value {
    json!({"available": false, "hold_min": hold, "reason": "worker_no_risk"})
}

pub type Fail = (Value, u16);
fn fail(err: &str, detail: impl Into<String>, status: u16) -> Fail {
    (json!({"error": err, "detail": detail.into()}), status)
}
fn crash(e: String) -> Fail {
    (json!({"error": e}), 502) // 여기서 터져도 주문은 아직 안 나갔다
}

/// 진입 조립 인자(원본 assemble_entry_plan 의 키워드).
#[derive(Debug, Clone)]
pub struct EntryArgs {
    pub side: String,
    pub fraction: f64,
    pub want_lev: Option<i64>,
    pub asset: String,
    pub fresh: bool,
    pub lines: Option<(f64, f64)>,
    pub limit_usdt: Option<f64>,
    pub stop: bool,
    pub skip_wait: Option<String>,
}

impl EntryArgs {
    pub fn new(side: &str, asset: &str) -> Self {
        EntryArgs { side: side.into(), fraction: 1.0, want_lev: None, asset: asset.into(), fresh: false, lines: None, limit_usdt: None, stop: false, skip_wait: None }
    }
}

/// 조립하는 쪽(감시·상태 칸의 주인과 같은 것 -- 대기 칸·청산 진행·무장 파일을 읽는다).
pub struct Desk {
    pub w: Watcher,
    /// DASHBOARD_SIZING_CAP_EQUITY_X(상한 무시 스위치 · 오타·음수 = 0 = 끔). 원본은 모듈 읽을 때 환경에서 한 번.
    pub override_x: f64,
    /// 아직 도는 작업 이름(원본 refresh_tasks 의 not done) -- 대기 칸 재사용 규칙(G12).
    pub running: std::collections::BTreeSet<String>,
    /// 2026-11-21 W12 다리: 도는 진입 작업의 집행 상태(칸 이름 `runner::slot_key` → Shared). 접수 핸들러가 `pull_live`·`push_live` 로 읽고 쓴다.
    pub live: std::collections::HashMap<String, crate::exec::Shared>,
}

pub fn override_x_from(v: Option<&str>) -> f64 {
    v.map(str::trim).filter(|s| !s.is_empty()).map_or(0.0, |s| s.parse::<f64>().map_or(0.0, |x| pyfmt::max(0.0, x))) // 오타를 «무제한»으로 읽지 않는다
}

impl Desk {
    pub fn new(w: Watcher) -> Desk {
        Desk { w, override_x: override_x_from(std::env::var("DASHBOARD_SIZING_CAP_EQUITY_X").ok().as_deref()), running: Default::default(), live: Default::default() }
    }

    /// 코인 → 주문 심볼(MANUAL_EXEC_SYMBOLS).
    pub fn exec_symbol(&self, asset: &str) -> Option<String> {
        match asset {
            "eth" => Some(self.w.eth_symbol.clone()),
            "sol" => Some("SOLUSDC".into()),
            "xrp" => Some("XRPUSDC".into()),
            "btc" => Some("BTCUSDC".into()),
            "sndk" => Some("SNDKUSDT".into()),
            "soxl" => Some("SOXLUSDT".into()),
            _ => None,
        }
    }
    pub fn exec_symbols(&self) -> Vec<String> {
        ["eth", "sol", "xrp", "btc", "sndk", "soxl"].iter().filter_map(|a| self.exec_symbol(a)).collect()
    }

    /// 주문 심볼(USDC) 중간가 ÷ 화면 심볼(USDT) 중간가. ponytail(원본): 못 읽으면 1(5분봉 감시만 수 bp 어긋남 · P21).
    pub async fn basis_for<Q: Quotes>(&self, q: &mut Q, book: &Value, asset: &str) -> f64 {
        basis_of(q, book, asset).await
    }
}

/// `Desk::basis_for` 의 몸(Desk 상태를 안 읽는다 -- 실행기가 Desk 를 잠그지 않고 부른다).
pub async fn basis_of<Q: Quotes>(q: &mut Q, book: &Value, asset: &str) -> f64 {
    {
        let Some(ms) = market_symbol(asset) else { return 1.0 };
        let Ok(r) = q.book(ms).await else { return 1.0 };
        let g = |o: &Value, k: &str| o.get(k).ok_or(()).and_then(|v| py_float(v).map_err(|_| ()));
        match (g(book, "bidPrice"), g(book, "askPrice"), g(&r, "bidPrice"), g(&r, "askPrice")) {
            (Ok(b), Ok(a), Ok(rb), Ok(ra)) if rb + ra != 0.0 => {
                let x = (b + a) / (rb + ra);
                if x == 0.0 { 1.0 } else { x }
            }
            _ => 1.0,
        }
    }
}

impl Desk {
    /// 대기 칸의 남은 명목(`(quantity − filled) × limit_price`).
    fn wait_left(w: &Value) -> Result<f64, String> {
        let pl = w.get("plan").filter(|v| truthy(Some(v))).cloned().unwrap_or(json!({}));
        let qty = if truthy(w.get("quantity")) { py_float(&w["quantity"])? } else { fl(pl.get("quantity"))? };
        let px = if truthy(w.get("limit_price")) { py_float(&w["limit_price"])? } else { fl(pl.get("price"))? };
        Ok(((qty - fl(w.get("filled"))?) * px).abs())
    }

    /// 원본 `assemble_entry_plan`. 반환 Ok((계획, cap)) · Err((본문, 상태)). 미리보기와 실주문이 이 함수 하나를 지난다(P1).
    pub async fn assemble_entry_plan<Q: Quotes, I: DeskIo>(&mut self, q: &mut Q, io: &mut I, a: &EntryArgs) -> Result<(Value, Value), Fail> {
        let symbol = self.exec_symbol(&a.asset).ok_or_else(|| crash(format!("KeyError: '{}'", a.asset)))?;
        // 2026-11-02 서로 독립인 조회를 한꺼번에(원본 gather) -- 값·검사 순서는 그대로: 계좌 못 읽음(503) → 호가·필터 실패 순서.
        let r_acct = io.account(a.fresh).await;
        let r_book = q.book(&symbol).await;
        let r_filters = q.filters(&symbol).await;
        let vol = q.vol(&symbol).await;
        let funding = q.funding(&symbol).await;
        let fees = q.fees(&symbol).await;
        let account = r_acct.map_err(crash)?;
        if !truthy(account.get("ok")) {
            // 못 읽은 계좌로 상한(합산 노출)을 계산하면 상한이 헐거워진다(P2)
            return Err(fail("account_unavailable", format!("계좌 조회 실패: {}", account.get("error").map_or("None".into(), py_str)), 503));
        }
        self.entry_body(q, io, a, &symbol, account, r_book, r_filters, vol, funding, fees).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn entry_body<Q: Quotes, I: DeskIo>(&mut self, q: &mut Q, io: &mut I, a: &EntryArgs, symbol: &str, account: Value, r_book: Result<Value, String>,
                                             r_filters: Result<Filters, String>, vol: Option<f64>, funding: Option<f64>, fees: Value) -> Result<(Value, Value), Fail> {
        let side = a.side.as_str();
        let c = |e: String| crash(e);
        let all_pos: Vec<Value> = account.get("positions").and_then(Value::as_array).cloned().unwrap_or_default();
        let positions: Vec<Value> = all_pos.iter().filter(|p| p.get("symbol").and_then(Value::as_str) == Some(symbol)).cloned().collect();
        let existing = notional_sum(&positions, None).map_err(c)?;
        // 🔴상한은 계좌 전체(같은 증거금 풀)에 걸린다(2026-09-19 · 11-05) -- existing(이 심볼)은 청산거리 투영에만
        let mut exposure = notional_sum(&pool_positions(&account, symbol), None).map_err(c)?;
        // 2026-10-20 다른 심볼에 걸린 늘리는 지정가(reduce-only 아님)는 체결되면 같은 증거금을 먹는다 -- 미리 뺀다(P4)
        let orders: Vec<Value> = account.get("open_orders").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut other: Vec<(String, f64)> = vec![];
        let add = |m: &mut Vec<(String, f64)>, k: String, v: f64| match m.iter_mut().find(|x| x.0 == k) {
            Some(x) => x.1 += v,
            None => m.push((k, v)),
        };
        for o in &orders {
            let os = o.get("symbol").map(py_str).unwrap_or_else(|| "None".into());
            if os != symbol && !truthy(o.get("reduce_only")) && quote_of(&os) == quote_of(symbol) {
                let k = o.get("symbol").map(py_str).ok_or_else(|| c("KeyError: 'symbol'".into()))?;
                add(&mut other, k, (fl(o.get("price")).map_err(c)? * fl(o.get("qty")).map_err(c)?).abs());
            }
        }
        // 2026-11-10 다른 코인 대기 진입의 남은 몫(그 심볼 걸린 주문 합과 큰 쪽만) · 11-18 같은 코인 다른 대기는 합(P5)
        let (mut same_w, mut wother) = (0.0, vec![]);
        for (k, w) in &self.w.waits {
            let ws_sym = self.exec_symbol(Watcher::wasset(k));
            let ok = w.as_object().is_some_and(busy) && ws_sym.as_deref().is_some_and(|s| quote_of(s) == quote_of(symbol));
            if !ok || Some(k) == a.skip_wait.as_ref() {
                continue;
            }
            let left = Self::wait_left(w).map_err(c)?;
            let ws = ws_sym.unwrap();
            if ws != symbol { add(&mut wother, ws, left) } else { same_w += left }
        }
        for (k, v) in wother {
            match other.iter_mut().find(|x| x.0 == k) {
                Some(x) => x.1 = pyfmt::max(x.1, v),
                None => other.push((k, pyfmt::max(0.0, v))),
            }
        }
        exposure += other.iter().map(|x| x.1).sum::<f64>() + same_w;
        // 추가 진입 맥락은 같은 방향만(헤지 모드 반대 다리는 다른 결정)
        let same: Vec<&Value> = positions.iter().filter(|p| p.get("side").and_then(Value::as_str) == Some(side)).collect();
        let bal = pool_balance(&account, symbol);
        let equity = fl(bal.get("margin")).map_err(c)?;
        let mut leverage = 0.0f64;
        for p in &positions {
            leverage = pyfmt::max(leverage, fl(p.get("leverage")).map_err(c)?);
        }
        if leverage == 0.0 {
            leverage = fl(account.get("leverage_by_symbol").and_then(|m| m.get(symbol))).map_err(c)?;
        }
        let book = r_book.map_err(c)?;
        let filters = r_filters.map_err(c)?;
        let cap_equity = if equity > 0.0 { Some(equity * SIZING_CAP_EQUITY_X) } else { None };
        // 🔴거래소 레버리지 기준 = 지평과 무관한 정책 천장(2026-09-13) -- 모델 상한만이면 HARD_CAP_X
        let policy_cap_x = if SIZING_CAP_MODEL_ONLY && equity > 0.0 { Some((equity * HARD_CAP_X) / equity) } else { cap_equity.map(|v| v / equity) };
        let plan_hold = HOLD_FIXED_MIN; // planning_hold: 위험 표가 없으면 고정 지평
        let hold_min = match same.first() { Some(p) => remaining_hold(p.get("entry_at"), plan_hold, io.now()), None => plan_hold };
        let risk = risk_none(hold_min);
        let mut binding: Vec<(f64, &str)> = vec![]; // 단순 규칙(위험모델 끔): 원장·순자산 6배·모델 상한 없음
        let margin_lev: f64 = match a.want_lev {
            Some(l) if l != 0 => l as f64,
            _ => policy_cap_x.and_then(|x| leverage_setting(equity * x, equity, exposure)).map(|s| s as f64).filter(|s| *s != 0.0).unwrap_or(leverage),
        };
        if SIZING_MARGIN_CAP_PCT > 0.0 && equity > 0.0 && margin_lev != 0.0 {
            binding.push((equity * margin_lev * SIZING_MARGIN_CAP_PCT / 100.0, "margin"));
        }
        let mut cap_notional = binding.iter().map(|x| x.0).reduce(pyfmt::min);
        let overridden = self.override_x > 0.0 && equity > 0.0;
        if overridden {
            cap_notional = Some(equity * self.override_x);
        }
        // 🔴키로 값만 비교(튜플 min 동점이 이름순으로 갈리지 않게) -- 처음 최소
        let who: Option<String> = if overridden {
            Some("override".into())
        } else {
            binding.iter().fold(None::<(f64, &str)>, |b, x| match b { Some(y) if y.0 <= x.0 => Some(y), _ => Some(*x) }).map(|x| x.1.to_string())
        };
        let mut cap = json!({});
        if let Some(cn) = cap_notional {
            cap = json!({"available": true, "cap_notional_usdt": round_n(cn, 2), "cap_equity_usdt": cap_equity.map(|v| round_n(v, 2)),
                         "cap_ledger_usdt": null, "cap_model_usdt": null, "margin_cap_pct": if SIZING_MARGIN_CAP_PCT != 0.0 { json!(SIZING_MARGIN_CAP_PCT) } else { Value::Null },
                         "risk": risk, "equity_x": SIZING_CAP_EQUITY_X, "liq_floor_pct": round_n(100.0 / SIZING_CAP_EQUITY_X, 1),
                         "override_x": if overridden { json!(self.override_x) } else { Value::Null }, "binding": who});
        }
        let (bid, ask) = (key_f(&book, "bidPrice").map_err(c)?, key_f(&book, "askPrice").map_err(c)?);
        let _price_ref = if side == "LONG" { ask } else { bid };
        let rec_qty = 0.0; // SIMPLE_SIZING vol_equivalent_qty
        // 🔴2026-11-21 사용자 «sltp(차트 선 주문 + 진입가 팝업 청산)만 쓴다»(원본 PR 61): 진입은 익절·손절 선이 있어야 받는다 -- 선 없는 요청은 주문 없이 거절
        let Some(mut ln) = a.lines else {
            return Err(fail("lines_required", "익절·손절 선(tp·sl)이 있어야 진입합니다 — 차트 선 주문으로 넣으세요", 400));
        };
        let key = plan::bracket_key(symbol, side);
        let prev: Option<Value> = if same.is_empty() { None } else { self.w.load().get(&key).cloned() };
        let pt = |k: &str| prev.as_ref().is_some_and(|p| truthy(p.get(k)));
        // 선 없는 포지션(옛 규칙 손절 · 앱 진입)에 선 도구로 추가 진입 = 거절(옛 판은 보낸 선을 버리고 −5% 규칙 손절로 넣었다)
        if !same.is_empty() && !pt("lines") {
            return Err(fail("add_on_unlined_position", "선이 걸리지 않은 포지션에는 추가 진입을 받지 않습니다 — 먼저 진입가 팝업에서 손절·익절선을 거세요", 409));
        }
        if !same.is_empty() && prev.is_some() && (pt("fired") || self.w.exiting.contains(&key)
            || prev.as_ref().unwrap().get("sl_legs").and_then(Value::as_array).is_some_and(|l| l.iter().any(|g| truthy(g.get("fired"))))) {
            return Err(fail("exiting", "이 포지션의 청산이 진행 중입니다 — 끝난 뒤 다시", 409));
        }
        let p_in = if side == "LONG" { bid } else { ask }; // 이번 주문 지정가(메이커 쪽)
        // 2026-11-18 가용 잔고에서 «아직 거래소에 안 걸린» 대기 몫의 증거금을 미리 뺀다(검토 #1·#4 · P6)
        //   ponytail(원본): 주문 id 대조 없이 심볼별 합으로 근사 · 레버리지는 이번 주문 것
        let mut wsum: Vec<(String, f64)> = vec![];
        let mut own: Option<(String, f64)> = None;
        for (k, w) in &self.w.waits {
            let ws_sym = self.exec_symbol(Watcher::wasset(k));
            if !(w.as_object().is_some_and(busy) && ws_sym.as_deref().is_some_and(|s| quote_of(s) == quote_of(symbol))) {
                continue;
            }
            let left = Self::wait_left(w).map_err(c)?;
            if Some(k) == a.skip_wait.as_ref() {
                own = if truthy(w.get("order_id")) && !(truthy(w.get("stop")) && !truthy(w.get("triggered"))) { Some((ws_sym.unwrap(), left)) } else { None };
                continue;
            }
            add(&mut wsum, ws_sym.unwrap(), left);
        }
        let mut rsum: Vec<(String, f64)> = vec![];
        for o in &orders {
            let cid = if truthy(o.get("cid")) { py_str(&o["cid"]) } else { String::new() };
            if !truthy(o.get("reduce_only")) && (cid.starts_with("dbwt") || cid.starts_with("dbpg")) {
                let k = o.get("symbol").map(py_str).ok_or_else(|| c("KeyError: 'symbol'".into()))?;
                add(&mut rsum, k, (fl(o.get("price")).map_err(c)? * fl(o.get("qty")).map_err(c)?).abs());
            }
        }
        let rget = |m: &Vec<(String, f64)>, k: &str| m.iter().find(|x| x.0 == k).map_or(0.0, |x| x.1);
        let mut avail: Option<f64> = match bal.get("available") { None | Some(Value::Null) => None, Some(v) => Some(py_float(v).map_err(c)?) };
        if let Some((s, l)) = &own {
            if rget(&rsum, s) >= l * 0.999 {
                if let Some(x) = rsum.iter_mut().find(|x| &x.0 == s) {
                    x.1 -= l;
                }
                if let (Some(av), true) = (avail, margin_lev != 0.0) {
                    avail = Some(av + l / margin_lev);
                }
            }
        }
        let unres: f64 = wsum.iter().map(|(k, v)| pyfmt::max(0.0, v - rget(&rsum, k))).sum();
        if let (Some(av), true) = (avail, unres > 0.0 && margin_lev != 0.0) {
            avail = Some(av - unres / margin_lev);
        }
        let label = match who.as_deref() {
            Some("margin") => format!("증거금 상한 {}%", pyfmt::g(SIZING_MARGIN_CAP_PCT)),
            Some("model") => "위험모델 상한".into(),
            _ => "상한".into(),
        };
        let ep = plan::build_entry_plan(&plan::EntryInput {
            side, best_bid: bid, best_ask: ask, recommended_qty: rec_qty, cap_notional, filters, symbol, existing_notional: exposure,
            equity, leverage, fraction: a.fraction, fraction_of_equity: true, order_leverage: margin_lev, cap_label: &label, available_margin: avail,
        }).map_err(|e| c(pe(e)))?;
        let mut plan = serde_json::to_value(&ep).unwrap();
        plan["recommended_source"] = json!("vol_equivalent");
        plan["recommended_qty"] = json!(round_n(rec_qty, 8));
        plan["projection"] = entry_projection(&plan, &bal, &positions, existing, equity).map_err(c)?;
        plan["trade_plan"] = plan_now(side, hold_min, vol, funding, &fees).map_err(c)?;
        let mut notes: Vec<Value> = plan["notes"].as_array().cloned().unwrap_or_default();
        {
            // 2026-10-28 차트 선(익절, 손절 · USDT): 거래소 스탑 없이 익절 = post-only 지정가 · 손절 = 1초 체결가 닿음 → peg 청산
            let mut add_lines = false;
            let pv = prev.clone().unwrap_or(json!({}));
            if !same.is_empty() {
                // 2026-11-03·11-12·11-13·11-14 같은 쪽 선 포지션이면 크기만 늘린다(보낸 선 무시, 무장된 선 그대로)
                let (tp_off, sl_off) = (pt("tp_off"), pt("sl_off")); // 선 없는 포지션은 위에서 거절했다
                ln = (if tp_off { 0.0 } else { fl(pv.get("tp_level")).map_err(c)? }, if sl_off { 0.0 } else { fl(pv.get("sl_level")).map_err(c)? });
                add_lines = true;
            }
            if io.sec_prices(&a.asset).is_none() {
                return Err(fail("lines_no_feed", "이 코인은 1초 체결가가 없어 선 손절을 감시할 수 없습니다", 400));
            }
            let b_ = self.basis_for(q, &book, &a.asset).await;
            let ref_ = a.limit_usdt.unwrap_or(p_in / b_);
            if add_lines {
                let sl_off = pt("sl_off");
                plan["bracket"] = json!({"available": true, "lines": true, "rule": false, "backstop_price": null,
                    "tp_price": pv.get("tp_price"), "sl_price": pv.get("sl_price"), "tp_level": pv.get("tp_level"), "sl_level": pv.get("sl_level"),
                    "sl_name": if sl_off { pv.get("sl_name").cloned().unwrap_or(Value::Null) } else { pv.get("sl_name").filter(|v| truthy(Some(v))).cloned().unwrap_or(Value::Null) },
                    "sl_pct": if sl_off || !pt("sl_level") { Value::Null } else { json!(round_n(100.0 * (fl(pv.get("sl_level")).map_err(c)? / ref_ - 1.0), 2)) },
                    "tp_off": pv.get("tp_off"), "sl_off": pv.get("sl_off"), "sl_up": pv.get("sl_up").cloned().unwrap_or(Value::Null)});
                plan["add_lines"] = json!(true);
                plan["add_keep"] = json!(true);
                let legs = pt("tp_legs") || pt("sl_legs");
                notes.push(json!(format!("추가 진입 — 기존 익절·손절 그대로{} · {}{}", if legs { " (나눈 조각 그대로)" } else { "" },
                    if pt("tp_off") { "익절 지운 상태 그대로(지정가 안 걸음)" } else { "합친 전량에 익절 지정가를 다시 겁니다" },
                    if sl_off { " · 🔴손절 지운 상태 그대로(손절 감시 없음)" } else { "" })));
            } else {
                let up = side == "LONG";
                let lb = plan::line_bracket(side, ref_, if ln.0 != 0.0 { ln.0 } else { ref_ * if up { 1.01 } else { 0.99 } },
                                            if ln.1 != 0.0 { ln.1 } else { ref_ * if up { 0.99 } else { 1.01 } }, b_, &filters, false)
                    .map_err(|e| (json!({"error": "bad_lines", "detail": e.py_msg()}), 400))?;
                plan["bracket"] = serde_json::to_value(&lb).unwrap();
            }
            if let Some(lu) = a.limit_usdt {
                // 2026-10-31 지정가 진입: 주문가 = 진입선 × basis · 롱 내림 · 숏 올림 · 호가를 넘으면 막는다(P22) · 명목은 지금가 크기 그대로
                let tick = if filters.tick != 0.0 { filters.tick } else { 0.01 };
                let step = filters.step;
                let lp = round_n(if side == "LONG" { (lu * b_ / tick + 1e-9).floor() } else { (lu * b_ / tick - 1e-9).ceil() } * tick, 8);
                if !a.stop && ((side == "LONG" && lp >= ask) || (side == "SHORT" && lp <= bid)) {
                    return Err(fail("limit_marketable", "지정가가 호가를 넘어 지금 체결됩니다 — 롱은 지금가 아래 · 숏은 위로(메이커)", 400));
                }
                if lp == 0.0 || step == 0.0 {
                    return Err(c("ZeroDivisionError: float division by zero".into()));
                }
                let qn = (fl(plan.get("quantity")).map_err(c)? * p_in / lp / step + 1e-9).floor() * step;
                plan["price"] = json!(lp);
                plan["quantity"] = json!(round_n(qn, 8));
                plan["notional_usdt"] = json!(round_n(qn * lp, 2));
                plan["limit"] = json!({"usdt": round_n(lu, 4), "price": lp});
                notes.push(json!(if a.stop {
                    format!("감시 진입 — 차트가 {} 에 닿으면 최우선 호가 peg 로 따라갑니다({}초, 남은 수량 버림 · 시장가 전환 없음)", fixed(lu, 2, true), fixed(EXEC_FALLBACK_SEC, 0, false))
                } else {
                    format!("지정가 {} 메이커 대기 — 안 닿으면 체결되지 않습니다(시장가 전환 없음)", fixed(lp, 2, true))
                }));
                plan["stop"] = json!(a.stop); // 2026-11-09 감시 진입 · 집행 = run_entry_stop
                if !truthy(plan.get("blocked")) && (qn < filters.min_qty || qn <= 0.0 || (filters.min_notional != 0.0 && qn * lp < filters.min_notional)) {
                    plan["blocked"] = json!("지정가 크기가 거래소 최소 주문보다 작습니다"); // P12
                }
            }
        }
        plan["notes"] = Value::Array(notes);
        // 게이지 값이 있으면 그것, «자동»이면 모델 추천 -- 처방이 없어(위험모델 끔) 모델 값은 None
        plan["target_leverage"] = a.want_lev.filter(|l| *l != 0).map_or(Value::Null, |l| json!(l));
        plan["leverage_source"] = json!(if a.want_lev.is_some_and(|l| l != 0) { "manual" } else { "model" });
        plan["leverage_model"] = Value::Null;
        plan["leverage_min_feasible"] = match cap_notional { Some(cn) if cn != 0.0 && equity > 0.0 => json!(round_n(cn / equity, 2)), _ => Value::Null };
        plan["leverage_position_floor"] = Value::Null;
        plan["hold_planned_min"] = json!(plan_hold);
        plan["hold_remaining_min"] = json!(hold_min);
        plan["leverage_steps"] = json!(LEVERAGE_STEPS);
        Ok((plan, cap))
    }

    /// 원본 `assemble_exit_plan`. 수량은 반드시 **방금 읽은 포지션**에서(헤지 모드 과청산 방어는 수량뿐), 심볼은 실제 열린 포지션에서(P14).
    /// for_order=true(실주문) = 화면용 펀딩·수수료 조회를 건너뛴다(P28) · «즉시»(M8 market)는 원본 PR 61 처럼 지웠다.
    #[allow(clippy::too_many_arguments)]
    pub async fn assemble_exit_plan<Q: Quotes, I: DeskIo>(&mut self, q: &mut Q, io: &mut I, position_side: &str, fraction: f64, fresh: bool, asset: &str,
                                                         for_order: bool, only_symbol: Option<&str>) -> Result<Value, Fail> {
        let c = |e: String| crash(e);
        let mut candidates: Vec<String> = vec![];
        let es = self.exec_symbol(asset).ok_or_else(|| c(format!("KeyError: '{asset}'")))?;
        for s in [es, market_symbol(asset).ok_or_else(|| c(format!("KeyError: '{asset}'")))?.to_string()] {
            if !candidates.contains(&s) {
                candidates.push(s);
            }
        }
        if let Some(o) = only_symbol {
            candidates = vec![o.to_string()]; // 2026-11-12 검토: 브래킷 감시는 무장한 심볼만
        }
        let account = io.account(fresh).await.map_err(c)?;
        if !truthy(account.get("ok")) {
            // 🔴«못 읽음»을 no_position 으로 돌려주면 감시가 정리(=익절·비상 스탑 삭제)로 간다(P2)
            return Err(fail("account_unavailable", format!("계좌 조회 실패: {}", account.get("error").map_or("None".into(), py_str)), 503));
        }
        let all: Vec<Value> = account.get("positions").and_then(Value::as_array).cloned().unwrap_or_default();
        let ps: Vec<plan::Position> = all.iter().map(|p| plan::Position {
            symbol: p.get("symbol").map_or("None".into(), py_str), side: p.get("side").map_or("\u{0}".into(), py_str), qty: None,
        }).collect();
        let cands: Vec<&str> = candidates.iter().map(String::as_str).collect();
        let Some((pos_ref, symbol, _)) = plan::resolve_exit_position(&ps, position_side, &cands) else {
            return Err(fail("no_position", format!("{position_side} 포지션이 없습니다"), 400));
        };
        let idx = ps.iter().position(|p| std::ptr::eq(p, pos_ref)).unwrap();
        let position = all[idx].clone();
        let mut leftover = vec![];
        let mut m: Vec<usize> = (0..ps.len()).filter(|&i| cands.contains(&ps[i].symbol.as_str()) && ps[i].side == position_side).collect();
        m.sort_by_key(|&i| cands.iter().position(|x| *x == ps[i].symbol).unwrap());
        for &i in &m[1..] {
            leftover.push(json!({"symbol": ps[i].symbol, "qty": fl(all[i].get("qty")).map_err(c)?}));
        }
        let r_book = q.book(&symbol).await;
        let r_filters = q.filters(&symbol).await;
        let vol = q.vol(&symbol).await;
        let (funding, fees) = if for_order { (None, Value::Null) } else { (q.funding(&symbol).await, q.fees(&symbol).await) };
        let book = r_book.map_err(c)?;
        let filters = r_filters.map_err(c)?;
        let xp = plan::build_exit_plan(&plan::ExitInput {
            position_side, position_qty: fl(position.get("qty")).map_err(c)?, best_bid: key_f(&book, "bidPrice").map_err(c)?, best_ask: key_f(&book, "askPrice").map_err(c)?,
            filters, symbol: &symbol, entry_price: fl(position.get("entry_price")).map_err(c)?, mark_price: fl(position.get("mark_price")).map_err(c)?, vol_bpm: vol, fraction,
        }).map_err(|e| c(pe(e)))?;
        let mut plan = serde_json::to_value(&xp).unwrap();
        plan["unrealized_pnl"] = position.get("unrealized_pnl").cloned().unwrap_or(Value::Null);
        if !leftover.is_empty() {
            plan["other_symbol_open"] = Value::Array(leftover);
        }
        let hold_min = remaining_hold(position.get("entry_at"), HOLD_FIXED_MIN, io.now());
        plan["risk"] = risk_none(hold_min);
        plan["hold_planned_min"] = json!(HOLD_FIXED_MIN);
        plan["hold_remaining_min"] = json!(hold_min);
        plan["trade_plan"] = plan_now(position_side, hold_min, vol, funding, &fees).map_err(c)?;
        Ok(plan)
    }
}
