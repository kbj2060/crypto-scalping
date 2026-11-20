//! `scripts/live_manual_peg_entry_20260912.py` 의 순수 함수를 그대로 옮긴 것(무엇을 보낼지 정한다 -- 네트워크·시계 없음).
//! 규칙: 연산 순서·비교 방향·파이썬 truthiness(0 = 없음)·`max`/`min` 의 동점 처리까지 원본과 같게 쓴다.
//! 파이썬이 예외를 내는 입력(0 나누기·inf 를 int 로)은 같은 종류의 `PlanError` 로 끝난다 -- 조용히 inf/NaN 을 흘리지 않는다.
//! 정답지: `rtorder/tools/gen_plan_fixture.py` → `tests/fixtures/plan_fixture.json` · 대조 시험 `tests/plan_fixture.rs`.

use crate::pyfmt::{self, fixed, repr, round0, round_n};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const FALLBACK_SEC: f64 = 120.0; // 진입 기본값. 섀도우 워커의 MAKER_SHADOW_TIMEOUT_S 와 같은 값 (D1: 집행 쪽 exec::FALLBACK_SEC 와 두 벌 -- 원본 그대로)

// ── 청산 마감을 변동성에 연동한다(2026-09-13, 사용자 요청) ── 섀도우 7,982legs: 체결 시간이 변동성에 반비례(순위상관 -0.429).
//   중앙체결시간 ≈ 52.0 / vol^1.472 · 마감은 그 ~29배로 K=1500. K=1500 에서 메이커 체결 중 0.03%(2건)만 잘린다.
pub const EXIT_DEADLINE_K: f64 = 1500.0;
pub const EXIT_DEADLINE_EXP: f64 = 1.472;
pub const EXIT_DEADLINE_MIN: f64 = 15.0;
pub const EXIT_DEADLINE_MAX: f64 = 120.0;
pub const EXIT_VOL_WINDOW: usize = 30; // 1분봉 개수. 수익률 30개를 쓰므로 종가는 31개 필요하다.
// 🔴이 값을 넘으면 지정가를 아예 안 걸고 시장가로 닫는다(사용자 요청) -- 비용이 아니라 «확실성»(대가 약 1.7bp). 30 = 1분봉 869일 99.3분위.
pub const EXIT_TAKER_VOL_BPM: f64 = 30.0;
// 가용 잔고에서 남기는 몫(거래소는 증거금을 불리한 쪽으로 재서 다 쓰면 -2019). ponytail(원본): 고정 2%. 더 자주 거부되면 늘린다.
pub const MARGIN_BUFFER: f64 = 0.02;
pub const STOP_LOSS_PCT: f64 = 0.03; // 2026-09-13 옛 고정 3% 손절(지금은 표시용 상수만 남음)
pub const STOP_SLIP_BP: f64 = 17.0;
pub const STOP_SLIP_P99_BP: f64 = 230.0;
// 2026-10-14 사용자 «권장 B»: 손절 첫 진입가 ∓5% · 2026-10-25 «크기는 2배로»: 상한 2.
pub const RULE_STOP_PCT: f64 = 0.05;
pub const COMMIT_UNIT_X: f64 = 4.0 * 0.303 * 10.26;
pub const COMMIT_CAP_X: f64 = 2.0;
pub const HOLD_CAP_X: f64 = 1.5; // 2026-10-16 사용자 «보유 1.5×»
pub const COMMIT_MMR: f64 = 0.02;
pub const COMMIT_AVG_OFF: f64 = 0.0066;
pub const COMMIT_LIQ_BUF: f64 = 0.005;
pub const COMMIT_STOP_SIG: f64 = 3.0;
pub const REC_FIRST_SHARE: f64 = 1.0 / 3.0; // 🔴2026-10-14 사용자 «첫 1/3 → −0.75%에 2/3»
// 🔴2026-10-13 사용자 «레버리지 20배 → 10배»: 실효 L = min(L, 0.9 × 거래소 레버리지) -- 증거금 100% 면 거래소가 막판 물타기를 거부.
pub const RULE_MARGIN_MAX: f64 = 0.9;
// 🔴2026-10-14 사용자 «−0.75% 물타기 지정가도 자동으로».
pub const REC_ADD_DIST: f64 = 0.0075;
pub const TP_LEGS_MAX: usize = 4; // 2026-11-11 분할 익절 선 최대 개수
pub const OWNED_BY_SAVED: [&str; 13] = [
    "sl_off", "tp_off", "tp_legs", "sl_legs", "tp_level", "tp_price", "sl_level", "sl_price", "sl_name", "sl_up", "watch_from", "mod", "placed",
];

/// 파이썬 예외와 1:1. 메시지는 파이썬 `str(exc)` 그대로(정답지가 사유까지 비교한다).
#[derive(Debug, Clone, PartialEq)]
pub enum PlanError {
    Value(String),
    ZeroDivision,
    Overflow(String),
    Index,
}

impl PlanError {
    pub fn py_type(&self) -> &'static str {
        match self {
            PlanError::Value(_) => "ValueError",
            PlanError::ZeroDivision => "ZeroDivisionError",
            PlanError::Overflow(_) => "OverflowError",
            PlanError::Index => "IndexError",
        }
    }
    pub fn py_msg(&self) -> String {
        match self {
            PlanError::Value(m) | PlanError::Overflow(m) => m.clone(),
            PlanError::ZeroDivision => "float division by zero".into(),
            PlanError::Index => "list index out of range".into(),
        }
    }
}

type R<T> = Result<T, PlanError>;

/// 파이썬 `/` (0 이면 ZeroDivisionError).
fn div(a: f64, b: f64) -> R<f64> {
    if b == 0.0 { Err(PlanError::ZeroDivision) } else { Ok(a / b) }
}

/// 파이썬 `int(x)`(0 쪽으로 자름)를 다시 float 로 쓸 때. inf → OverflowError · NaN → ValueError.
fn int_trunc(x: f64) -> R<f64> {
    if x.is_nan() {
        Err(PlanError::Value("cannot convert float NaN to integer".into()))
    } else if x.is_infinite() {
        Err(PlanError::Overflow("cannot convert float infinity to integer".into()))
    } else {
        Ok(x.trunc())
    }
}

fn py_floor(x: f64) -> R<f64> {
    int_trunc(x).map(|_| x.floor())
}

fn py_ceil(x: f64) -> R<f64> {
    int_trunc(x).map(|_| x.ceil())
}

/// 파이썬 `x ** y`(float). 결과가 넘치면 OverflowError(errno ERANGE), 0 으로 줄어드는 것은 그대로.
fn py_pow(x: f64, y: f64) -> R<f64> {
    let r = x.powf(y);
    if r.is_infinite() && x.is_finite() && y.is_finite() {
        return Err(PlanError::Overflow("(34, 'Numerical result out of range')".into()));
    }
    Ok(r)
}

/// 파이썬 truthiness(0.0 = 없음).
fn truthy(x: Option<f64>) -> Option<f64> {
    x.filter(|v| *v != 0.0)
}

fn str_repr(s: &str) -> String {
    // ponytail: 작은따옴표·역슬래시가 든 문자열의 파이썬 repr 은 재현 안 함 -- 측면 문자열에만 쓴다.
    format!("'{s}'")
}

/// 실주문 게이트. **코드 기본값이 아니라 서버 .env 가 정답**이라 매번 읽는다(G3).
pub fn exec_enabled() -> bool {
    exec_enabled_from(std::env::var("DASHBOARD_MANUAL_EXEC_ENABLED").ok().as_deref())
}

pub fn exec_enabled_from(v: Option<&str>) -> bool {
    matches!(v.unwrap_or("").trim().to_lowercase().as_str(), "1" | "true" | "yes")
}

/// 거래소 수량/가격 격자(`load_filters` 가 늘 네 칸을 채운다).
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Filters {
    pub step: f64,
    pub tick: f64,
    pub min_qty: f64,
    pub min_notional: f64,
}

/// 격자에 **내림**. 반올림하면 상한을 넘길 수 있고, 상한은 넘지 않는 게 요점이다.
pub fn floor_to(value: f64, grid: f64) -> R<f64> {
    if grid <= 0.0 {
        return Ok(value);
    }
    Ok(int_trunc(value / grid + 1e-9)? * grid)
}

/// 격자에 **올림**. 매도 쪽 peg 에 쓴다 -- 내리면 한 틱 공격적이 되어 GTX 가 거부한다.
pub fn ceil_to(value: f64, grid: f64) -> R<f64> {
    if grid <= 0.0 {
        return Ok(value);
    }
    Ok(py_ceil(value / grid - 1e-9)? * grid)
}

#[derive(Debug, Clone)]
pub struct EntryInput<'a> {
    pub side: &'a str,
    pub best_bid: f64,
    pub best_ask: f64,
    pub recommended_qty: f64,
    pub cap_notional: Option<f64>,
    pub filters: Filters,
    pub symbol: &'a str,
    pub existing_notional: f64,
    pub equity: f64,
    pub leverage: f64,
    pub fraction: f64,
    pub fraction_of_equity: bool,
    pub order_leverage: f64,
    pub cap_label: &'a str,
    pub available_margin: Option<f64>,
}

impl Default for EntryInput<'_> {
    fn default() -> Self {
        EntryInput {
            side: "LONG", best_bid: 0.0, best_ask: 0.0, recommended_qty: 0.0, cap_notional: None,
            filters: Filters { step: 0.001, tick: 0.01, min_qty: 0.0, min_notional: 0.0 }, symbol: "ETHUSDT",
            existing_notional: 0.0, equity: 0.0, leverage: 0.0, fraction: 1.0, fraction_of_equity: false,
            order_leverage: 0.0, cap_label: "상한", available_margin: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EntryPlan {
    pub symbol: String,
    pub side: String,
    #[serde(rename = "positionSide")]
    pub position_side: String, // 헤지 모드라 필수. 빼면 롱/숏이 서로를 상계한다.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "timeInForce")]
    pub time_in_force: String, // post-only -- 테이커가 되면 체결 대신 거부된다
    pub price: f64,
    pub quantity: f64,
    pub notional_usdt: f64,
    pub margin_usdt: f64,
    pub margin_pct_of_equity: Option<f64>,
    pub leverage: Option<f64>,
    pub existing_notional_usdt: f64,
    pub total_notional_usdt: f64,
    pub effective_leverage: Option<f64>,
    pub liq_distance_pct: Option<f64>,
    pub cap_used_pct: Option<f64>,
    pub fraction: f64,
    pub capped: bool,
    pub available_qty: f64,
    pub cap_room_after_usdt: Option<f64>,
    pub fallback_after_sec: f64,
    pub fallback: String, // 원본 표시값 "taker" 그대로. 집행은 시장가를 내지 않고 NeedsMarketDecision 으로 멈춘다(exec.rs M1)
    pub cap_notional_usdt: Option<f64>,
    pub notes: Vec<String>,
    pub blocked: Option<String>,
    pub dry_run: bool,
}

/// 보낼 주문 하나를 만든다. **순수 함수** -- 네트워크도 시계도 안 본다.
/// 🔴상한은 **합산 포지션**에 건다(2026-09-12 수정): 기존 숏 7,642 가 열린 채 6,803 을 더하면 14,444 로 상한 13,579 를 넘었다.
pub fn build_entry_plan(a: &EntryInput) -> R<EntryPlan> {
    if a.side != "LONG" && a.side != "SHORT" {
        return Err(PlanError::Value(format!("side must be LONG/SHORT, got {}", str_repr(a.side))));
    }
    if !(a.best_bid > 0.0 && a.best_ask > 0.0 && a.best_ask >= a.best_bid) {
        return Err(PlanError::Value(format!("bad book: bid={} ask={}", repr(a.best_bid), repr(a.best_ask))));
    }
    // 분할 진입 비율(2026-09-13). 범위를 벗어나면 막는다 -- 조용히 1.0 으로 자르면 3분의 1만 넣으려던 게 전량이 된다.
    if !(0.0 < a.fraction && a.fraction <= 1.0) {
        return Err(PlanError::Value(format!("fraction must be in (0, 1], got {}", repr(a.fraction))));
    }
    let f = &a.filters;
    let price = if a.side == "LONG" { floor_to(a.best_bid, f.tick)? } else { ceil_to(a.best_ask, f.tick)? };
    let mut notes: Vec<String> = vec![];
    let mut qty = pyfmt::max(0.0, a.recommended_qty);
    let existing = pyfmt::max(0.0, a.existing_notional);
    let cap = truthy(a.cap_notional);

    // 2026-09-25 사용자: fraction_of_equity 면 비율 = 순자산 대비 이번 주문 증거금.
    // 🔴2026-09-26 사용자 «넘치면 50%까지만 채우기»: 상한을 넘으면 막지 않고 남은 여유까지만. 줄였다는 사실은 notes·capped 로.
    let mut capped = false;
    if a.fraction_of_equity {
        qty = if a.equity > 0.0 && a.order_leverage > 0.0 { div(a.equity * a.fraction * a.order_leverage, price)? } else { 0.0 };
        let room = cap.map(|c| c - existing);
        if let Some(room) = room {
            if qty * price > room {
                let want_pct = 100.0 * a.fraction;
                qty = div(pyfmt::max(0.0, room), price)?;
                capped = true;
                if room > 0.0 && a.equity > 0.0 && a.order_leverage > 0.0 {
                    notes.push(format!("요청 {}% → {}까지 {}% 만 넣습니다", fixed(want_pct, 0, false), a.cap_label,
                        fixed(100.0 * room / a.order_leverage / a.equity, 1, false)));
                }
            }
        }
    } else if let Some(c) = cap {
        let room = c - existing;
        if room <= 0.0 {
            qty = 0.0;
            notes.push(format!("기존 포지션 {} USDT 가 이미 상한을 채웠습니다", fixed(existing, 0, true)));
        } else if qty * price > room {
            qty = div(room, price)?;
            notes.push(format!("상한까지 남은 여유 {} USDT 로 줄였습니다 (기존 {} + 신규 = 상한 {})",
                fixed(room, 0, true), fixed(existing, 0, true), fixed(c, 0, true)));
        }
    }
    // 2026-11-07 순자산이 아니라 **가용 잔고** 안으로 자른다(11-07 ETHUSDC 4.43개 -2019 · SL/TP 도 안 걸림).
    let mut short_of_margin = false;
    if let Some(am) = a.available_margin {
        if a.order_leverage > 0.0 {
            let most = div(pyfmt::max(0.0, am) * (1.0 - MARGIN_BUFFER) * a.order_leverage, price)?;
            if qty > most {
                qty = most;
                capped = true;
                short_of_margin = true;
                notes.push(format!("가용 잔고 {} 안으로 줄였습니다 (증거금 {})", fixed(am, 2, true),
                    fixed(div(most * price, a.order_leverage)?, 2, true)));
            }
        }
    }
    // 🔴비율은 **상한에 자른 뒤** 곱한다(2026-09-13 실측: 권고에 곱하면 «33%» 가 상한의 71%).
    let available_qty = qty;
    if a.fraction < 1.0 && !a.fraction_of_equity {
        qty = available_qty * a.fraction;
        notes.push(format!("지금 넣을 수 있는 {} ETH 중 {}% 만 넣습니다 (나머지 {} 는 남겨 둡니다)",
            fixed(available_qty, 3, false), fixed(round0(100.0 * a.fraction), 0, false), fixed(available_qty - qty, 3, false)));
    }
    qty = floor_to(qty, f.step)?;

    let blocked = if qty < f.min_qty || qty <= 0.0 {
        Some(if short_of_margin {
            "가용 잔고가 부족합니다".to_string()
        } else if capped || cap.is_some_and(|c| existing >= c) {
            format!("{} 여유가 없습니다", a.cap_label)
        } else {
            format!("수량이 최소 {} 미만", repr(f.min_qty))
        })
    } else if f.min_notional != 0.0 && qty * price < f.min_notional {
        Some(format!("명목이 최소 {} USDT 미만", fixed(f.min_notional, 0, true)))
    } else {
        None
    };

    let notional = qty * price;
    let total = notional + existing;
    let margin = if a.leverage != 0.0 { div(notional, a.leverage)? } else { 0.0 };
    let eq = truthy(Some(a.equity));
    Ok(EntryPlan {
        symbol: a.symbol.into(),
        side: if a.side == "LONG" { "BUY" } else { "SELL" }.into(),
        position_side: a.side.into(),
        kind: "LIMIT".into(),
        time_in_force: "GTX".into(),
        price: round_n(price, 8),
        quantity: round_n(qty, 8),
        notional_usdt: round_n(notional, 2),
        margin_usdt: round_n(margin, 2),
        margin_pct_of_equity: eq.map(|e| div(100.0 * margin, e).map(|v| round_n(v, 1))).transpose()?,
        leverage: truthy(Some(a.leverage)),
        existing_notional_usdt: round_n(existing, 2),
        total_notional_usdt: round_n(total, 2),
        effective_leverage: eq.map(|e| round_n(total / e, 2)),
        // 교차 마진 근사. 거래소 실제 청산가와 0.5pp 안쪽으로 맞았다(실측).
        liq_distance_pct: if eq.is_some() && total != 0.0 { Some(round_n(100.0 * a.equity / total, 1)) } else { None },
        cap_used_pct: cap.map(|c| round0(100.0 * total / c)),
        fraction: round_n(a.fraction, 4),
        capped,
        available_qty: round_n(available_qty, 8),
        cap_room_after_usdt: cap.map(|c| round_n(pyfmt::max(0.0, c - total), 2)),
        fallback_after_sec: FALLBACK_SEC,
        fallback: "taker".into(),
        cap_notional_usdt: a.cap_notional,
        notes,
        blocked,
        dry_run: !exec_enabled(),
    })
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuleL {
    pub l: f64,
    pub stop_pct: f64,
    pub binding: String,
}

/// 재생 규칙의 배수 L 과 손절 폭. L = min(COMMIT_UNIT_X × 권장배수, 청산 안전(max(3σ24, RULE_STOP_PCT)), cap).
/// cap = COMMIT_CAP_X(자동 2×, 10-25) 또는 HOLD_CAP_X(보유 1.5×) -- 그 밖은 거절.
pub fn commit_rule_l(position_side: &str, sigma24_bp: f64, vol_mult: f64, cap: f64) -> R<RuleL> {
    if !(cap == COMMIT_CAP_X || cap == HOLD_CAP_X) {
        return Err(PlanError::Value(format!("bad cap: {}", repr(cap))));
    }
    if !(sigma24_bp > 0.0 && vol_mult > 0.0) {
        return Err(PlanError::Value(format!("bad sigma/vol_mult: {} {}", repr(sigma24_bp), repr(vol_mult))));
    }
    let sd = pyfmt::max(COMMIT_STOP_SIG * sigma24_bp / 1e4, RULE_STOP_PCT); // 청산 안전용 거리(넓은 쪽)
    let den = if position_side == "LONG" {
        1.0 - (1.0 - COMMIT_MMR) * (1.0 - sd) * (1.0 - COMMIT_LIQ_BUF) / (1.0 - COMMIT_AVG_OFF)
    } else {
        (1.0 + COMMIT_MMR) * (1.0 + sd) * (1.0 + COMMIT_LIQ_BUF) / (1.0 + COMMIT_AVG_OFF) - 1.0
    };
    let liq = if den > 0.0 { 1.0 / den } else { f64::INFINITY };
    // 파이썬 min(dict, key=…) = 삽입 순서에서 **처음** 최소인 것.
    let mut who = ("노출 맞춤", COMMIT_UNIT_X * vol_mult);
    for c in [("청산 안전", liq), ("상한", cap)] {
        if c.1 < who.1 {
            who = c;
        }
    }
    Ok(RuleL { l: who.1, stop_pct: RULE_STOP_PCT, binding: who.0.into() })
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuleSize {
    pub cap_notional: f64,
    pub l_eff: f64,
    pub fraction: f64,
    pub first: bool,
    pub room: f64,
    pub target_notional: f64,
}

/// 규칙 크기. 상한 = rule_l × 순자산(계좌 전체 명목). 첫 진입 = 상한 × 1/3, 물타기 = 남은 여유 전부.
pub fn rec_rule_size(rule_l: f64, equity: f64, existing_notional: f64, order_leverage: f64, first: Option<bool>,
                     risk_used: f64, add_stop_frac: Option<f64>) -> R<RuleSize> {
    if !(0.0 < rule_l && rule_l <= COMMIT_CAP_X) {
        return Err(PlanError::Value(format!("rule_l must be in (0, {}], got {}", repr(COMMIT_CAP_X), repr(rule_l))));
    }
    if !(equity > 0.0 && order_leverage > 0.0) {
        // l_eff: 서버가 읽는다(없으면 KeyError 502). 원본 그대로 `first` 인자를 안 본다.
        return Ok(RuleSize { cap_notional: 0.0, l_eff: rule_l, fraction: 1.0, first: existing_notional <= 0.0, room: 0.0, target_notional: 0.0 });
    }
    let l_eff = pyfmt::min(rule_l, RULE_MARGIN_MAX * order_leverage);
    let cap = l_eff * equity;
    // 2026-10-10 검증: «첫 진입» = 이 방향 포지션 없음(호출부가 준다). 계좌 전체 명목으로 정하면 숏 $300 만 있어도 롱 첫 진입이 98%.
    let first = first.unwrap_or(existing_notional <= 0.0);
    let mut want = if first { cap * REC_FIRST_SHARE } else { pyfmt::max(0.0, cap - existing_notional) };
    // 2026-10-10 동일위험 지킴: 이번 주문이 손절선까지 잃을 몫 ≤ (L×손절폭×순자산 − 이미 든 몫의 손절 손실). 손절선 너머면 0.
    if let Some(asf) = add_stop_frac {
        let lim = if asf > 0.0 { pyfmt::max(0.0, l_eff * RULE_STOP_PCT * equity - risk_used) / asf } else { 0.0 };
        want = pyfmt::min(want, lim);
    }
    Ok(RuleSize {
        cap_notional: cap,
        l_eff,
        first,
        room: pyfmt::max(0.0, cap - existing_notional),
        target_notional: pyfmt::min(want, pyfmt::max(0.0, cap - existing_notional)),
        fraction: pyfmt::min(1.0, pyfmt::max(1e-9, want / (equity * order_leverage))),
    })
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AddOrder {
    pub price: f64,
    pub qty: f64,
    pub notional: f64,
    pub anchor: f64,
}

/// 첫 진입 직후 자동 물타기 지정가. 너무 작거나 여유가 없으면 None.
/// 기준가 = 손절선에서 거꾸로 푼 첫 진입가(손절과 같은 기준) · 크기 = 수동 물타기와 같은 rec_rule_size.
/// (집행 쪽 `place_add_order` 는 GTC = §9 M12 시장가 분기 -- 이번 범위 아님.)
#[allow(clippy::too_many_arguments)]
pub fn rec_rule_add_order(position_side: &str, sl_price: f64, stop_pct: f64, l_eff: f64, cap_notional: f64, room_before: f64,
                          filled_qty: f64, order_leverage: f64, filters: &Filters) -> R<Option<AddOrder>> {
    if !(sl_price > 0.0 && 0.0 < stop_pct && stop_pct < 1.0 && l_eff > 0.0 && cap_notional > 0.0 && filled_qty > 0.0) {
        return Ok(None);
    }
    let long = position_side == "LONG";
    let anchor = if long { sl_price / (1.0 - stop_pct) } else { sl_price / (1.0 + stop_pct) };
    let px = round_n(if long { floor_to(anchor * (1.0 - REC_ADD_DIST), filters.tick)? } else { ceil_to(anchor * (1.0 + REC_ADD_DIST), filters.tick)? }, 8);
    let sgn = if long { 1.0 } else { -1.0 };
    let r = rec_rule_size(l_eff, cap_notional / l_eff, cap_notional - room_before + filled_qty * anchor, order_leverage, Some(false),
                          filled_qty * sgn * (anchor - sl_price), Some(div(sgn * (px - sl_price), px)?))?;
    let qty = round_n(floor_to(div(r.target_notional, px)?, filters.step)?, 8);
    if qty <= 0.0 || qty < filters.min_qty || qty * px < filters.min_notional {
        return Ok(None);
    }
    Ok(Some(AddOrder { price: px, qty, notional: round_n(qty * px, 2), anchor: round_n(anchor, 8) }))
}

/// 이미 무장된 이 측면의 규칙 상태(손절 유지 판단에 쓰는 칸만).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PrevRule {
    pub sl_price: Option<f64>,
    pub sl_level: Option<f64>,
    pub backstop_price: Option<f64>,
    pub sl_name: Option<String>,
    #[serde(default)]
    pub rule: bool,
}

/// `{**b, …}` 의 «…» 부분. 호출부가 b 위에 덮는다(파이썬은 이 칸들이 b 의 같은 키를 항상 이긴다).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuleBracket {
    pub available: bool,
    pub tp_price: Option<f64>,
    pub sl_price: Option<f64>,
    pub sl_level: Option<f64>,
    pub backstop_price: Option<f64>,
    pub sl_pct: Option<f64>,
    pub sl_name: String,
    pub rule: bool,
}

/// 규칙 손절 = «기준가 ∓stop_pct» 거래소 스탑 · 익절 없음. prev 가 규칙 무장이면 그 손절을 **그대로**(물타기해도 안 물러남).
/// (거래소 스탑 자체 = Algo STOP_MARKET = §9 M10 시장가 분기 -- 이번 범위 아님. 여기는 가격 계산만.)
pub fn rec_rule_bracket(position_side: &str, anchor_price: f64, basis: f64, stop_pct: f64, filters: &Filters,
                        prev: Option<&PrevRule>, anchor_kind: &str) -> R<RuleBracket> {
    if let Some(p) = prev {
        if truthy(p.sl_price).is_some() && truthy(p.backstop_price).is_some() && p.rule {
            return Ok(RuleBracket {
                available: true, tp_price: None, sl_price: p.sl_price, sl_level: p.sl_level, backstop_price: p.backstop_price,
                sl_name: p.sl_name.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "규칙 손절 유지".into()), sl_pct: None, rule: true,
            });
        }
    }
    if !(anchor_price > 0.0 && basis > 0.0 && 0.0 < stop_pct && stop_pct < 1.0) {
        return Err(PlanError::Value(format!("bad anchor/basis/stop: {} {} {}", repr(anchor_price), repr(basis), repr(stop_pct))));
    }
    let tick = truthy(Some(filters.tick)).unwrap_or(0.01);
    let long = position_side == "LONG";
    let sl = if long {
        round_n(py_floor(anchor_price * (1.0 - stop_pct) / tick + 1e-9)? * tick, 8)
    } else {
        round_n(py_ceil(anchor_price * (1.0 + stop_pct) / tick - 1e-9)? * tick, 8)
    };
    Ok(RuleBracket {
        available: true, tp_price: None, sl_price: Some(sl), sl_level: Some(round_n(sl / basis, 4)), backstop_price: Some(sl),
        sl_pct: Some(round_n(if long { -100.0 * stop_pct } else { 100.0 * stop_pct }, 2)),
        sl_name: format!("{} {}{}%", anchor_kind, if long { "−" } else { "+" }, pyfmt::g(100.0 * stop_pct)), rule: true,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LineBracket {
    pub available: bool,
    pub lines: bool,
    pub rule: bool,
    pub tp_price: f64,
    pub tp_level: f64,
    pub sl_up: bool,
    pub sl_price: f64,
    pub sl_level: f64,
    pub backstop_price: Option<f64>,
    pub sl_pct: f64,
    pub sl_name: String,
}

/// 선 브래킷. 선(레벨) = USDT 차트 가격 · basis = 주문 심볼/USDT. 익절 = 주문 심볼 post-only 지정가 · 손절 = 서버 1초 닿음 → peg 청산.
/// 2026-11-10 사용자: sl_watch(들어간 포지션 선 수정)면 손절선이 지금가 어느 쪽이든 받는다.
/// 2026-11-14 사용자 «±15% 제한 자체를 제거»: 방향만 본다.
pub fn line_bracket(position_side: &str, ref_level: f64, tp_level: f64, sl_level: f64, basis: f64, filters: &Filters, sl_watch: bool) -> R<LineBracket> {
    if !(ref_level > 0.0 && basis > 0.0 && tp_level > 0.0 && sl_level > 0.0) {
        return Err(PlanError::Value("선 가격이 없습니다".into()));
    }
    let long = position_side == "LONG";
    let sl_up = if sl_watch { sl_level > ref_level } else { !long };
    let tp_ok = if long { ref_level < tp_level } else { tp_level < ref_level };
    let sl_ok = sl_watch || if long { sl_level < ref_level } else { ref_level < sl_level };
    if !tp_ok || !sl_ok {
        return Err(PlanError::Value(if !sl_watch { "롱은 익절선이 현재가 위·손절선이 아래, 숏은 반대여야 합니다" } else { "롱은 익절선이 현재가 위, 숏은 아래여야 합니다" }.into()));
    }
    let tick = truthy(Some(filters.tick)).unwrap_or(0.01);
    let tp = tp_level * basis; // 익절 지정가는 선보다 불리하지 않게(롱 올림 · 숏 내림)
    let tp = round_n(if long { py_ceil(tp / tick - 1e-9)? } else { py_floor(tp / tick + 1e-9)? } * tick, 8);
    Ok(LineBracket {
        available: true, lines: true, rule: false, tp_price: tp, tp_level: round_n(tp_level, 4), sl_up,
        sl_price: round_n(sl_level * basis, 8), sl_level: round_n(sl_level, 4), backstop_price: None,
        sl_pct: round_n(100.0 * (sl_level / ref_level - 1.0), 2), sl_name: format!("선 손절 {}", fixed(sl_level, 2, false)),
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct TpLegIn {
    pub price: f64,
    #[serde(default)]
    pub w: Option<f64>,
}

/// 2026-11-11 사용자 «청산 100% 를 나눠 SLTP 선 여러 개». 가까운 쪽부터 격자 내림 · 최소 미달 조각은 먼 조각에 몰아줌 · 합 ≤ qty.
pub fn split_tp_qty(qty: f64, legs: &[TpLegIn], filters: &Filters, position_side: &str) -> R<Vec<f64>> {
    let step = truthy(Some(filters.step)).unwrap_or(0.001);
    let (mn, mnn) = (filters.min_qty, filters.min_notional);
    let sgn = if position_side == "LONG" { 1.0 } else { -1.0 };
    let mut order: Vec<usize> = (0..legs.len()).collect();
    order.sort_by(|&a, &b| (legs[a].price * sgn).partial_cmp(&(legs[b].price * sgn)).unwrap()); // 안정 정렬(파이썬 sorted 와 같다)
    let w = |l: &TpLegIn| pyfmt::max(truthy(l.w).unwrap_or(0.0), 0.0);
    let tot = legs.iter().fold(0.0, |s, l| s + w(l)); // 파이썬 3.11 sum = 앞에서부터 단순 덧셈(3.12+ 는 보정 합 -- 서버 버전 확인 필요)
    let tot_w = if tot != 0.0 { tot } else { 1.0 };
    let fl = |v: f64| -> R<f64> { Ok(int_trunc(v / step + 1e-9)? * step) };
    let ok = |q: f64, px: f64| q > 0.0 && q >= mn - 1e-12 && q * px >= mnn - 1e-9;
    let sum = |o: &[f64]| o.iter().fold(0.0, |s, v| s + v);
    let mut out = vec![0.0; legs.len()];
    let mut carry = 0.0;
    for (j, &i) in order.iter().enumerate() {
        let raw = qty * w(&legs[i]) / tot_w + carry;
        let q = if j == order.len() - 1 { fl(qty - sum(&out))? } else { fl(raw)? };
        if ok(q, legs[i].price) {
            out[i] = round_n(q, 8);
            carry = raw - q;
        } else {
            carry = raw;
        }
    }
    let rest = fl(qty - sum(&out))?;
    if rest > 1e-12 {
        // 마지막 조각이 미달이었다 -- 마지막으로 건 조각에 더한다
        let placed: Vec<usize> = order.iter().copied().filter(|&i| out[i] > 0.0).collect();
        if let Some(&p) = placed.last() {
            out[p] = round_n(out[p] + rest, 8);
        } else {
            let v = round_n(fl(qty)?, 8);
            let &last = order.last().ok_or(PlanError::Index)?;
            out[last] = v;
        }
    }
    Ok(out)
}

/// 쏜 상태: 원본은 False/None · True · "tp" 세 값을 한 칸에 쓴다.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Fired {
    #[default]
    No,
    Yes,
    Tp,
}

#[derive(Debug, Clone, Default)]
pub struct SlLeg {
    pub level: f64,
    pub up: Option<bool>,
    pub fired: bool,
}

#[derive(Debug, Clone, Default)]
pub struct TpLeg {
    pub level: f64,
    pub resting: bool,
    pub merged: bool,
}

/// 무장 상태 한 칸(`manual_bracket_state.json` 의 값)에서 판정이 읽는 칸. 파이썬 truthiness 로 읽는다(`from_value`).
#[derive(Debug, Clone, Default)]
pub struct Armed {
    pub symbol: String,
    pub armed_at: f64,
    pub watch_from: f64,
    pub lines: bool,
    pub sl_level: Option<f64>,
    pub sl_price: Option<f64>,
    pub tp_level: Option<f64>,
    pub tp_resting: bool,
    pub tp_off: bool,
    pub sl_off: bool,
    pub fired: Fired,
    pub sl_up: Option<bool>,
    pub sl_legs: Vec<SlLeg>,
    pub tp_legs: Vec<TpLeg>,
}

/// 파이썬 `bool(v)`.
pub fn py_truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

impl Armed {
    pub fn from_value(v: &Value) -> Armed {
        let f = |k: &str| v.get(k).and_then(Value::as_f64);
        let t = |k: &str| py_truthy(v.get(k));
        let ob = |x: Option<&Value>| match x { None | Some(Value::Null) => None, Some(x) => Some(py_truthy(Some(x))) };
        let arr = |k: &str| v.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
        Armed {
            symbol: v.get("symbol").and_then(Value::as_str).unwrap_or("").into(),
            armed_at: f("armed_at").unwrap_or(0.0),
            watch_from: f("watch_from").unwrap_or(0.0),
            lines: t("lines"),
            sl_level: f("sl_level"),
            sl_price: f("sl_price"),
            tp_level: f("tp_level"),
            tp_resting: t("tp_resting"),
            tp_off: t("tp_off"),
            sl_off: t("sl_off"),
            fired: match v.get("fired") {
                Some(Value::String(s)) if s == "tp" => Fired::Tp,
                x if py_truthy(x) => Fired::Yes,
                _ => Fired::No,
            },
            sl_up: ob(v.get("sl_up")),
            sl_legs: arr("sl_legs").iter().map(|l| SlLeg {
                level: l.get("level").and_then(Value::as_f64).unwrap_or(0.0), up: ob(l.get("up")), fired: py_truthy(l.get("fired")),
            }).collect(),
            tp_legs: arr("tp_legs").iter().map(|l| TpLeg {
                level: l.get("level").and_then(Value::as_f64).unwrap_or(0.0), resting: py_truthy(l.get("resting")), merged: py_truthy(l.get("merged")),
            }).collect(),
        }
    }
}

/// 2026-11-11 분할 손절: 닿은 손절 조각 번호들. 이미 쏜 조각은 닫힐 때까지 계속 «닿음».
/// 닿음 = 1초 체결가(무장·수정 뒤) 또는 무장 뒤 마감된 5분봉 종가. up = 올라 닿으면.
pub fn sl_legs_hit(position_side: &str, armed: &Armed, touch: Option<(f64, f64)>, last_bar: Option<(f64, f64)>) -> Vec<usize> {
    let since = pyfmt::max(armed.armed_at, armed.watch_from);
    let mut out = vec![];
    for (i, leg) in armed.sl_legs.iter().enumerate() {
        let up = leg.up.unwrap_or(position_side != "LONG");
        let lv = leg.level;
        if leg.fired
            || touch.is_some_and(|t| if up { t.1 >= lv } else { t.0 <= lv })
            || last_bar.is_some_and(|b| b.0 > since && if up { b.1 >= lv } else { b.1 <= lv })
        {
            out.push(i);
        }
    }
    out
}

/// 손절이 «올라 닿으면» 쏘나. 숏 기본 · 롱 손절을 지금가 위에 옮긴 무장(sl_up True).
pub fn sl_up_of(position_side: &str, armed: &Armed) -> bool {
    armed.sl_up.unwrap_or(position_side != "LONG")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LineHit {
    Sl,
    Tp,
}

/// 선 무장 닿음 판정. 손절이 먼저(같은 틱에 둘 다면 손절 -- 보수적). 익절은 거래소 지정가가 못 걸렸을 때(tp_resting 아님)만 서버가 닫는다.
pub fn line_hit(position_side: &str, armed: &Armed, touch: Option<(f64, f64)>) -> Option<LineHit> {
    let (lo, hi) = touch?;
    let long = position_side == "LONG";
    let sl = truthy(armed.sl_level);
    let tp = if armed.tp_off { None } else { truthy(armed.tp_level) }; // tp_off = 2026-11-11 «익절선 지우기»
    let up = sl_up_of(position_side, armed);
    if let Some(sl) = sl {
        if if up { hi >= sl } else { lo <= sl } {
            return Some(LineHit::Sl);
        }
    }
    if !armed.tp_legs.is_empty() && !armed.tp_off {
        // 2026-11-11 분할 익절: 거래소에 못 건 조각이 닿으면 -- ponytail(원본): 그 조각 몫만이 아니라 전량을 닫는다(드문 대비책)
        if armed.tp_legs.iter().any(|l| !l.resting && !l.merged && if long { hi >= l.level } else { lo <= l.level }) {
            return Some(LineHit::Tp);
        }
        return None;
    }
    if let Some(tp) = tp {
        if !armed.tp_resting && if long { hi >= tp } else { lo <= tp } {
            return Some(LineHit::Tp);
        }
    }
    None
}

/// SL 이탈 = **무장 뒤 마감된 5분봉 종가**가 레벨(USDT)을 넘었나(2026-09-27 사용자 지시 -- 꼬리 한 번에 나가던 옛 판 대체).
/// 2026-10-29 선 수정·다시 켜기 뒤 마감된 봉만(검토 F1). ponytail(원본): 09-27 이전 무장은 sl_level 이 없어 sl_price 로 비교.
pub fn bracket_sl_hit(position_side: &str, last_bar: Option<(f64, f64)>, armed: &Armed) -> bool {
    let level = truthy(armed.sl_level).or(truthy(armed.sl_price));
    let since = pyfmt::max(armed.armed_at, armed.watch_from);
    let (Some(level), Some(bar)) = (level, last_bar) else { return false };
    if bar.0 <= since {
        return false;
    }
    if sl_up_of(position_side, armed) { bar.1 >= level } else { bar.1 <= level }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Position {
    pub symbol: String,
    pub side: String,
    #[serde(default)]
    pub qty: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TickAction {
    Clear,
    Fire,
    SlLeg,
    Hold,
}

/// 감시 루프 한 틱의 판단. 🔴«사라짐»은 **무장보다 5초 넘게 새** 계좌 조회에서만 -- 진입 직후 캐시엔 포지션이 안 보이고,
/// 조회 실패의 빈 목록도 «없음»이 아니다. 잘못 판단하면 살아 있는 포지션의 **비상 스탑을 지운다**.
pub fn bracket_action(armed: &Armed, position_side: &str, positions: &[Position], account_ok: bool, account_ts: f64,
                      last_bar: Option<(f64, f64)>, touch: Option<(f64, f64)>) -> TickAction {
    let held = positions.iter().any(|p| p.symbol == armed.symbol && p.side == position_side && p.qty.unwrap_or(0.0).abs() > 0.0);
    if account_ok && !held && account_ts > armed.armed_at + 5.0 {
        return TickAction::Clear;
    }
    let no_sl = Armed { sl_level: None, ..armed.clone() };
    if armed.lines {
        // 2026-10-28 선: 1초 체결가 닿음 · 체결 흐름이 끊겨도 5분봉 종가 이탈이면 닫는다 · 한 번 쏘면(fired) 닫힐 때까지
        if armed.sl_off {
            // 2026-11-11 «손절선만 제거»: 손절은 끄고 익절(서버 감시 몫)은 계속. fired "tp" = 손절을 지운 뒤 익절로 쏜 청산(닫힐 때까지)
            //   · fired True + sl_off = 손절로 쏜 청산을 «손절 해제»로 멈춘 것(10-15 동작 그대로 hold)
            return if armed.fired == Fired::Tp || line_hit(position_side, &no_sl, touch).is_some() { TickAction::Fire } else { TickAction::Hold };
        }
        if !armed.sl_legs.is_empty() && armed.fired == Fired::No {
            // 2026-11-11 분할 손절: 닿은 조각 몫만(sl_leg) · 남은 조각이 다 닿았거나 익절(서버 감시 몫)이 닿으면 전량(fire)
            let hit = sl_legs_hit(position_side, armed, touch, last_bar);
            if hit.len() >= armed.sl_legs.len() || line_hit(position_side, &no_sl, touch).is_some() {
                return TickAction::Fire;
            }
            return if hit.is_empty() { TickAction::Hold } else { TickAction::SlLeg };
        }
        return if armed.fired != Fired::No || line_hit(position_side, armed, touch).is_some() || bracket_sl_hit(position_side, last_bar, armed) {
            TickAction::Fire
        } else {
            TickAction::Hold
        };
    }
    if armed.sl_off {
        return TickAction::Hold; // 2026-10-15 «손절 해제» 버튼: 5분봉 종가 손절도 끈다
    }
    if bracket_sl_hit(position_side, last_bar, armed) { TickAction::Fire } else { TickAction::Hold }
}

/// 무장 상태의 키 = «심볼:측면»(2026-09-26 -- 측면만 쓰면 ETH 롱과 SOL 롱이 서로 덮는다).
pub fn bracket_key(symbol: &str, position_side: &str) -> String {
    format!("{symbol}:{position_side}")
}

/// 옛 키(«LONG»/«SHORT»)를 «심볼:측면»으로 옮긴다.
/// 옛 키와 새 키가 같은 칸을 가리키면 **파일 순서상 뒤의 것**(자리는 앞의 것) -- 파이썬 dict 와 같다(serde_json preserve_order · 시험 watch_fixture `legacy_and_new_key_collide`).
pub fn bracket_rekey(state: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, v) in state {
        if k == "LONG" || k == "SHORT" {
            let mut nv = v.as_object().cloned().unwrap_or_default();
            let sym = v.get("symbol").and_then(Value::as_str).unwrap_or("").to_string();
            nv.insert("side".into(), Value::String(k.clone()));
            out.insert(bracket_key(&sym, k), Value::Object(nv));
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

fn py_eq(a: Option<&Value>, b: &Value) -> bool {
    match (a, b) {
        (None, Value::Null) => true,
        (Some(Value::Number(x)), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Some(x), y) => x == y,
        _ => false,
    }
}

/// 감시 틱 결과(`mine`)를 저장본(`current`)에 합친다. 내가 읽을 때 본 무장(`seen_at` 의 armed_at)과 같은 측면에만 --
/// 그 사이 새로 무장된 측면(물타기 포함)을 지우지 않는다.
pub fn bracket_merge(current: &Map<String, Value>, mine: &Map<String, Value>, seen_at: &Map<String, Value>) -> Map<String, Value> {
    let mut out = current.clone();
    for (side, armed_at) in seen_at {
        if !py_eq(out.get(side).and_then(|v| v.get("armed_at")), armed_at) {
            continue;
        }
        if let Some(m) = mine.get(side) {
            // 2026-10-15 sl_off 는 저장본이 주인(틱은 버튼 전 사본) · 2026-10-29 «선 끌어 수정» 칸도 저장본이 주인.
            //   단 틱이 이번에 쏘았으면(fired) 그 틱의 tp_resting(익절 지정가를 지웠다 = False)을 쓴다.
            let cur = out.get(side).and_then(Value::as_object).cloned().unwrap_or_default();
            let mut m = m.as_object().cloned().unwrap_or_default();
            let mut own: Vec<&str> = OWNED_BY_SAVED.to_vec();
            if !(py_truthy(m.get("fired")) && !py_truthy(cur.get("fired"))) {
                own.push("tp_resting");
            }
            for k in own {
                m.remove(k);
                if let Some(v) = cur.get(k) {
                    m.insert(k.into(), v.clone());
                }
            }
            out.insert(side.clone(), Value::Object(m));
        } else {
            out.remove(side);
        }
    }
    out
}

/// 1분봉 종가들로 실현변동성(bp/√분). 표본이 모자라면 None -- **0 이 아니다**(0 이면 마감이 최대로 늘어나는데 실제로는 «모른다»).
pub fn realized_vol_bpm(closes: &[f64]) -> Option<f64> {
    let px: Vec<f64> = closes.iter().copied().filter(|c| *c > 0.0).collect();
    if px.len() < 10 {
        return None;
    }
    let rets: Vec<f64> = (1..px.len()).map(|i| (px[i] / px[i - 1]).ln()).collect();
    let n = rets.len() as f64;
    let mean = rets.iter().fold(0.0, |s, r| s + r) / n; // 파이썬 3.11 sum = 단순 덧셈(split_tp_qty 주석)
    let var = rets.iter().fold(0.0, |s, r| s + (r - mean).powf(2.0)) / (n - 1.0);
    Some(var.sqrt() * 1e4)
}

/// 변동성이 클수록 짧은 마감. 변동성을 모르면 보수적으로 최대값.
pub fn exit_deadline_sec(vol_bpm: Option<f64>) -> R<f64> {
    let Some(v) = vol_bpm.filter(|v| !(*v <= 0.0)) else { return Ok(EXIT_DEADLINE_MAX) }; // NaN 은 파이썬처럼 통과시킨다
    let raw = div(EXIT_DEADLINE_K, py_pow(v, EXIT_DEADLINE_EXP)?)?;
    Ok(pyfmt::min(EXIT_DEADLINE_MAX, pyfmt::max(EXIT_DEADLINE_MIN, raw)))
}

#[derive(Debug, Clone)]
pub struct ExitInput<'a> {
    pub position_side: &'a str,
    pub position_qty: f64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub filters: Filters,
    pub symbol: &'a str,
    pub entry_price: f64,
    pub mark_price: f64,
    pub vol_bpm: Option<f64>,
    pub fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExitPlan {
    pub symbol: String,
    pub side: String,
    #[serde(rename = "positionSide")]
    pub position_side_field: String, // 반전하지 않는다 -- 반전하면 청산이 아니라 신규 진입이다
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "timeInForce", skip_serializing_if = "Option::is_none")]
    pub time_in_force: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<f64>,
    pub quantity: f64,
    pub notional_usdt: f64,
    pub reference_price: f64,
    pub position_side: String,
    pub position_qty: f64,
    pub fraction: f64,
    pub remaining_qty: f64,
    pub entry_price: Option<f64>,
    pub mark_price: Option<f64>,
    pub exit_move_pct: Option<f64>,
    pub fallback_after_sec: f64,
    pub fallback: String,
    pub repeg: bool,
    pub vol_bpm: Option<f64>,
    pub market_reason: Option<String>,
    pub notes: Vec<String>,
    pub blocked: Option<String>,
    pub dry_run: bool,
}

/// 열린 포지션 하나를 **메이커로 닫는** 주문. 🔴`reduceOnly` 를 보내지 않는다(헤지 모드 -1106) -- 안전장치는 **수량**뿐.
/// 메이커 가격은 진입의 거울: 롱 청산 = SELL = 최우선 매도호가(올림), 숏 청산 = BUY = 최우선 매수호가(내림).
pub fn build_exit_plan(a: &ExitInput) -> R<ExitPlan> {
    if a.position_side != "LONG" && a.position_side != "SHORT" {
        return Err(PlanError::Value(format!("position_side must be LONG/SHORT, got {}", str_repr(a.position_side))));
    }
    if !(a.best_bid > 0.0 && a.best_ask > 0.0 && a.best_ask >= a.best_bid) {
        return Err(PlanError::Value(format!("bad book: bid={} ask={}", repr(a.best_bid), repr(a.best_ask))));
    }
    // 부분 청산 비율. 범위 밖이면 **막는다** -- 1.0 으로 자르면 «절반이 전량», 0 으로 자르면 «안 닫힘».
    if !(0.0 < a.fraction && a.fraction <= 1.0) {
        return Err(PlanError::Value(format!("fraction must be in (0, 1], got {}", repr(a.fraction))));
    }
    let f = &a.filters;
    let mut dust_reason: Option<String> = None;
    let closing_long = a.position_side == "LONG";
    let price = if closing_long { ceil_to(a.best_ask, f.tick)? } else { floor_to(a.best_bid, f.tick)? };
    let held = pyfmt::max(0.0, a.position_qty);
    // 전량이면 곱셈을 아예 안 한다 -- 부동소수로 한 스텝 모자라 «거의 전량»이 남는 걸 막는다.
    let want = if a.fraction >= 1.0 { held } else { held * a.fraction };
    let qty = pyfmt::min(floor_to(want, f.step)?, floor_to(held, f.step)?);
    let deadline = exit_deadline_sec(a.vol_bpm)?;
    // 미정: 시장가 분기 #M6 — 사용자 결정 대기. 변동성 ≥ 30bp/√분이면 파이썬은 계획을 MARKET 으로 만든다.
    //   계획은 «무엇을 할지»의 기술이라 원본 값 그대로 둔다(정답지 일치) -- 집행(exec.rs)이 MARKET 계획을 받으면 NeedsMarketDecision 으로 멈춘다.
    let mut market = a.vol_bpm.is_some_and(|v| v >= EXIT_TAKER_VOL_BPM);
    let blocked = if qty <= 0.0 || qty < f.min_qty {
        Some(if held <= 0.0 {
            "닫을 포지션이 없습니다".to_string()
        } else {
            format!("{}% 는 {} ETH 라 최소 {} 미만입니다 -- 비율을 올리세요", fixed(round0(100.0 * a.fraction), 0, false), repr(qty), repr(f.min_qty))
        })
    } else if f.min_notional != 0.0 && qty * price < f.min_notional {
        // 미정: 시장가 분기 #M7 — 사용자 결정 대기. 🔴먼지 포지션(최소명목 미만)은 막지 않고 시장가(2026-09-13 감사 «막으면 화면에서 영영 못 닫는다»).
        market = true;
        dust_reason = Some(format!("명목 {} 가 거래소 최소 {} USDT 미만이라 지정가로는 못 보냅니다 — 시장가로 닫습니다",
            fixed(qty * price, 0, true), fixed(f.min_notional, 0, true)));
        None
    } else {
        None
    };
    let notional = qty * price;
    let mv = if a.entry_price > 0.0 {
        Some(if closing_long { (price - a.entry_price) / a.entry_price } else { (a.entry_price - price) / a.entry_price })
    } else {
        None
    };
    let market_reason = match (&dust_reason, market) {
        (Some(d), _) => Some(d.clone()),
        (None, true) => Some(format!("변동성 {} ≥ {} bp/√분 — 지정가를 걸지 않고 즉시 시장가로 닫습니다 (비용은 메이커보다 약 1.7bp 비쌉니다)",
            fixed(a.vol_bpm.unwrap_or(0.0), 1, false), fixed(EXIT_TAKER_VOL_BPM, 0, false))),
        (None, false) => None,
    };
    Ok(ExitPlan {
        symbol: a.symbol.into(),
        side: if closing_long { "SELL" } else { "BUY" }.into(),
        position_side_field: a.position_side.into(),
        kind: if market { "MARKET" } else { "LIMIT" }.into(),
        time_in_force: (!market).then(|| "GTX".into()),
        price: (!market).then(|| round_n(price, 8)),
        quantity: round_n(qty, 8),
        notional_usdt: round_n(notional, 2),
        reference_price: round_n(price, 8),
        position_side: a.position_side.into(),
        position_qty: round_n(held, 8),
        fraction: round_n(a.fraction, 4),
        remaining_qty: round_n(pyfmt::max(0.0, held - qty), 8),
        entry_price: truthy(Some(a.entry_price)),
        mark_price: truthy(Some(a.mark_price)),
        exit_move_pct: mv.map(|m| round_n(100.0 * m, 3)),
        fallback_after_sec: round_n(deadline, 1),
        fallback: "taker".into(),
        repeg: !market,
        vol_bpm: a.vol_bpm.map(|v| round_n(v, 2)),
        market_reason,
        notes: vec![],
        blocked,
        dry_run: !exec_enabled(),
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Leftover {
    pub symbol: String,
    pub qty: f64,
}

/// 청산할 포지션을 «설정» 이 아니라 **실제 열려 있는 곳**에서 고른다(2026-09-19) -- 심볼이 틀리면 청산이 신규 진입이 된다.
/// 둘 다 열려 있으면 candidates 순서(진입 심볼 우선), 나머지는 leftover 로 함께. 없으면 None.
pub fn resolve_exit_position<'p>(positions: &'p [Position], position_side: &str, candidates: &[&str]) -> Option<(&'p Position, String, Vec<Leftover>)> {
    let mut m: Vec<&Position> = positions.iter().filter(|p| candidates.contains(&p.symbol.as_str()) && p.side == position_side).collect();
    if m.is_empty() {
        return None;
    }
    m.sort_by_key(|p| candidates.iter().position(|c| *c == p.symbol).unwrap());
    let left = m[1..].iter().map(|p| Leftover { symbol: p.symbol.clone(), qty: p.qty.unwrap_or(0.0) }).collect();
    Some((m[0], m[0].symbol.clone(), left))
}
