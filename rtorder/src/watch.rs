//! 손절·익절 감시와 주문 상태 칸 -- `dashboard/orders.py` 의 `register(...)` 안 클로저를 그대로 옮긴 것(2026-11-20 셋째 조각):
//! 무장 저장본(`bracket_load`·`bracket_save`) · 상태 칸(`manual_entry_state`·`side_exit_states`·`manual_wait_states`·`exit_slot`·`_busy`·`busy_or`) ·
//! 감시 한 틱(`bracket_tick`) · 감시 루프(`watch_brackets`) · 부분 청산 뒤 익절 다시 걸기(`partial_exit_rearm_tp`·`tp_watch_only`·`set_tp_legs`·`set_sl_legs`) ·
//! 무장 쓰기(`arm_bracket`·`plan_from_current`·`keep_rearm`·`exit_then_rearm`) · 재시작(`limit_save`·`limit_recover`·`wait_cleanup`) · `stop_touched`.
//! **어디에도 연결하지 않는다** -- 바깥(계좌·5분봉·1초 체결가·거래소 함수·청산 계획 조립·시계·잠금·스위치)은 `WatchIo` 뒤. 구현은 시험의 가짜뿐.
//!
//! 저장본·상태 칸은 파이썬 dict 를 그대로 `serde_json::Map`(순서 보존)으로 둔다 -- 같은 파일을 파이썬과 주고받고(그림자·전환), 정답지와 칸 단위로 대조한다.
//! 발동 청산이 시장가 분기(§9 M6·M7·M9)에 닿으면 원본처럼 시장가로 닫는다(2026-11-21 사용자 결정 «원본 동작 유지» -- 넷째 조각까지의 market_wait 멈춤은 지웠다).
//! D7·D8·D9(규칙 문서 «어긋나 보이는 곳»)는 파이썬 동작 그대로 -- 시험(tests/watch_fixture.rs)이 그 동작을 고정한다.

use crate::exec::{self, Exchange, ExecPlan, ExecState, Phase, Shared};
use crate::plan::{bracket_action, bracket_key, bracket_merge, bracket_rekey, sl_legs_hit, Armed, Position, TickAction};
use crate::pyfmt;
use crate::sign::{py_repr, py_str, truthy};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type Obj = Map<String, Value>;

pub const WATCH_PERIOD_SEC: f64 = 2.0; // watch_brackets 주기(2026-11-09 체결 알림이면 일찍 깸) -- 주입 가능(`watch_brackets(period)`), «체결마다»는 다음에
pub const BAR_SEC: f64 = 300.0; // FOOTPRINT_BAR_SECONDS -- 차트 캔들과 같은 5분봉
pub const RETRY_SEC: f64 = 30.0; // 정리 실패·조회 실패·청산 실패 뒤 다시
pub const TP_CANCEL_RETRY_SEC: f64 = 10.0; // 선 발동 전 익절 지정가 취소 실패 -- 계좌 조회(무게 ~50)를 2초마다 반복하지 않게
pub const LIMIT_REC_MAX_AGE: f64 = 6.0 * 3600.0; // 대기 최대 4시간 + 여유 -- 이보다 오래된 파일은 지금 포지션의 것이 아니다(검증 #3)
pub const WAIT_SLOTS_MAX: i64 = 10; // ponytail(원본): 대기마다 집행 루프가 3초마다 주문 조회 -- 10개 ≈ 분당 200회. 더 필요하면 올린다
const MARKET_SYMBOLS: [(&str, &str); 9] = [("eth", "ETHUSDT"), ("sol", "SOLUSDT"), ("btc", "BTCUSDT"), ("xrp", "XRPUSDT"), ("hype", "HYPEUSDT"),
    ("sndk", "SNDKUSDT"), ("soxl", "SOXLUSDT"), ("skhynix", "SKHYNIXUSDT"), ("mu", "MUUSDT")];
/// 코인 → 주문 심볼(eth 는 환경 `DASHBOARD_MANUAL_EXEC_SYMBOL` -- `Watcher::eth_symbol`). 순서 = 원본 dict 순서(exec_asset_of 가 앞에서부터 찾는다).
const EXEC_REST: [(&str, &str); 5] = [("sol", "SOLUSDC"), ("xrp", "XRPUSDC"), ("btc", "BTCUSDC"), ("sndk", "SNDKUSDT"), ("soxl", "SOXLUSDT")];

/// 감시가 바깥에 닿는 곳 전부. 오류(Err)는 파이썬에서 «예외»였던 것 -- 문자열은 `"<예외 형>: <문구>"`.
/// ponytail: 계좌·청산 계획 조립(`assemble_exit_plan`)·5분봉은 다음 조각(조립·계좌)이 실제 구현을 붙인다 -- 지금 구현은 시험의 가짜뿐.
#[allow(async_fn_in_trait)]
pub trait WatchIo {
    fn now(&self) -> f64; // time.time()
    fn exec_enabled(&self) -> bool; // DASHBOARD_MANUAL_EXEC_ENABLED(매번 다시 읽는다, G3)
    fn order_locked(&self) -> bool; // manual_order_lock.locked()
    /// 감시 주기 잠 -- 체결 알림이면 일찍 깬다(A10). false = 멈춤(원본의 CancelledError).
    async fn nap(&mut self, sec: f64) -> bool;
    /// 캐시된 계좌(swr_cached 10초 + 묵은 값 허용). 실패 응답(ok=false)은 Ok 로 그대로 온다 -- «못 읽음 = 모름».
    async fn account(&mut self) -> Result<Value, String>;
    /// 새로 읽은 계좌(produce_account, 재시작 복구).
    async fn produce_account(&mut self) -> Result<Value, String>;
    /// load_market_history(코인) -- USDT 5분봉 [{time, close, …}].
    async fn bars(&mut self, asset: &str) -> Result<Vec<Value>, String>;
    /// sec_prices(코인) -- 1초 마지막 체결가 [(초, 가격)] · 원천 없는 코인 None.
    fn sec_prices(&mut self, asset: &str) -> Result<Option<Vec<(f64, f64)>>, String>;
    async fn clear_bracket(&mut self, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Result<Value, String>;
    /// assemble_exit_plan(pside, fraction, fresh=True, asset, only_symbol) -- Ok(계획) · Err(오류 dict) (원본은 (None, (err, status))).
    async fn exit_plan(&mut self, pside: &str, fraction: f64, asset: &str, only_symbol: &str) -> Result<Value, Value>;
    /// run_exit(session, plan, state) -- 상태 칸을 고친다. 실제 구현 = `run_exit_into`(exec::run_exit + 칸 옮기기).
    async fn run_exit(&mut self, plan: &Value, slot: &mut Obj) -> Result<(), String>;
    async fn place_bracket(&mut self, spec: &Value, symbol: &str, pside: &str, strict: bool) -> Result<Value, String>;
    async fn place_tp_ladder(&mut self, legs: &Value, symbol: &str, pside: &str) -> Result<Value, String>;
    async fn cancel_wait_orphans(&mut self, symbols: &[String]) -> Result<Value, String>;
    /// 2026-11-21 cancel_exit_orders(symbol, pside) -- 앞선 청산 따라가기(dbpg+측면 · 닫는 쪽) 지우고 목록 다시 읽어 확인. {"cancelled","errors"}.
    async fn cancel_exit_orders(&mut self, symbol: &str, pside: &str) -> Result<Value, String>;
    /// 망가진 칸 알림(텔레그램 -- 뒤에서 보내 감시를 막지 않는다). 기본 = 안 보냄.
    fn alert(&mut self, _msg: &str) {}
}

// ── 파이썬 값 다루기 ──────────────────────────────────────────────────────────
fn t(o: &Obj, k: &str) -> bool {
    truthy(o.get(k))
}
fn val(o: &Obj, k: &str) -> Value {
    o.get(k).cloned().unwrap_or(Value::Null)
}
/// `x or {}` 의 dict.
fn obj_or_empty(v: Option<&Value>) -> Obj {
    match v {
        Some(Value::Object(m)) if !m.is_empty() => m.clone(),
        _ => Obj::new(),
    }
}
/// 파이썬 `a == b`(없음 = None · 수는 값으로).
fn py_eq(a: Option<&Value>, b: Option<&Value>) -> bool {
    let n = |v: Option<&Value>| match v {
        None | Some(Value::Null) => Value::Null,
        Some(x) => x.clone(),
    };
    match (n(a), n(b)) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Bool(x), Value::Number(y)) | (Value::Number(y), Value::Bool(x)) => y.as_f64() == Some(if x { 1.0 } else { 0.0 }),
        (x, y) => x == y,
    }
}
/// 무장이 그대로인가(`a.get("armed_at") == b.get("armed_at")` -- 파이썬 == 의미).
pub fn same_armed(a: Option<&Value>, b: Option<&Value>) -> bool {
    py_eq(a, b)
}
fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
/// 파이썬 `float(v)`.
fn py_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(0.0)),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.trim().parse::<f64>().map_err(|_| format!("ValueError: could not convert string to float: {}", py_repr(v))),
        x => Err(format!("TypeError: float() argument must be a string or a real number, not '{}'", type_name(x))),
    }
}
/// `float(o[k])`.
fn req_float(o: &Obj, k: &str) -> Result<f64, String> {
    py_float(o.get(k).ok_or_else(|| format!("KeyError: '{k}'"))?)
}
/// `float(o.get(k) or 0)`.
fn float_or0(o: &Obj, k: &str) -> f64 {
    if t(o, k) { py_float(&o[k]).unwrap_or(0.0) } else { 0.0 }
}
/// `int(v or 1)`.
fn int_or1(v: Option<&Value>) -> i64 {
    if !truthy(v) {
        return 1;
    }
    match v.unwrap() {
        Value::Number(n) => n.as_i64().unwrap_or_else(|| n.as_f64().unwrap_or(1.0).trunc() as i64),
        Value::String(s) => s.trim().parse().unwrap_or(1),
        Value::Bool(b) => *b as i64,
        _ => 1,
    }
}
fn req_str(o: &Obj, k: &str) -> Result<String, String> {
    match o.get(k) {
        None => Err(format!("KeyError: '{k}'")),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(v) => Ok(py_str(v)),
    }
}
/// 파이썬 `a or b or c` -- 첫 참값, 다 거짓이면 마지막 값.
fn py_or(vals: Vec<Value>) -> Value {
    let last = vals.last().cloned().unwrap_or(Value::Null);
    vals.into_iter().find(|v| truthy(Some(v))).unwrap_or(last)
}
fn f(x: f64) -> Value {
    json!(x)
}

/// `datetime.fromtimestamp(t, timezone.utc).isoformat()`.
pub fn iso_utc(t: f64) -> String {
    let (mut s, mut us) = (t.trunc() as i64, ((t - t.trunc()) * 1e6).round_ties_even() as i64);
    if us >= 1_000_000 {
        (s, us) = (s + 1, us - 1_000_000);
    }
    if us < 0 {
        (s, us) = (s - 1, us + 1_000_000);
    }
    let (days, secs) = (s.div_euclid(86400), s.rem_euclid(86400));
    // civil_from_days(Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    let frac = if us != 0 { format!(".{us:06}") } else { String::new() };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}{frac}+00:00", secs / 3600, secs % 3600 / 60, secs % 60)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `datetime.fromisoformat(s).timestamp()` 의 흔한 모양(YYYY-MM-DD[THH:MM[:SS[.ffffff]]][Z|±HH:MM]) -- 못 읽으면 None(원본: ValueError → 0.0).
/// ponytail: 시간대 없는 값은 UTC 로 본다(원본은 서버 지역 시각 -- 계좌 generated_at 은 늘 +00:00 을 단다). 파이썬 3.11 이 받는 다른 모양(기본형 20270115T…)은 None.
pub fn parse_iso(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    let num = |a: usize, n: usize| -> Option<i64> {
        let x = b.get(a..a + n)?;
        x.iter().all(u8::is_ascii_digit).then(|| std::str::from_utf8(x).ok()?.parse().ok())?
    };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, mo, d) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let (mut secs, mut i, mut frac) = (0i64, 10usize, 0.0f64);
    if b.len() > 10 {
        if b[10] != b'T' && b[10] != b' ' {
            return None;
        }
        let (h, mi) = (num(11, 2)?, num(14, 2)?);
        if b.get(13) != Some(&b':') || h > 23 || mi > 59 {
            return None;
        }
        secs = h * 3600 + mi * 60;
        i = 16;
        if b.get(16) == Some(&b':') {
            let sec = num(17, 2)?;
            if sec > 59 {
                return None;
            }
            secs += sec;
            i = 19;
            if b.get(19) == Some(&b'.') {
                let n = b[20..].iter().take_while(|c| c.is_ascii_digit()).count();
                if !(n == 3 || n == 6) {
                    return None;
                }
                frac = std::str::from_utf8(&b[19..20 + n]).ok()?.parse().ok()?;
                i = 20 + n;
            }
        }
    }
    let off = match &b[i..] {
        [] => 0,
        [b'Z'] => 0,
        [sg @ (b'+' | b'-'), ..] if b.len() == i + 6 && b[i + 3] == b':' => {
            let o = num(i + 1, 2)? * 3600 + num(i + 4, 2)? * 60;
            if *sg == b'+' { o } else { -o }
        }
        _ => return None,
    };
    Some((days_from_civil(y, mo, d) * 86400 + secs - off) as f64 + frac)
}

// ── 저장본(manual_bracket_state.json) ────────────────────────────────────────
/// 무장 상태는 **파일**에 둔다 -- 옛 3% 손절의 실패가 메모리 state 에만 있어 재시작하면 사라졌다(B1).
pub fn bracket_path(dir: &Path) -> PathBuf {
    dir.join("manual_bracket_state.json")
}
/// 못 읽으면 {}(원본: OSError·ValueError). 옛 키(LONG/SHORT)는 «심볼:측면»으로(B2).
/// 🔴원본은 NaN·Infinity 가 든 파일도 읽는다(json 기본) -- serde_json 은 못 읽어 {} 가 된다(= 감시 멈춤). 지금 쓰는 쪽(arm_bracket 등)은 유한값만 쓴다.
pub fn bracket_load(dir: &Path) -> Obj {
    let Ok(raw) = std::fs::read_to_string(bracket_path(dir)) else { return Obj::new() };
    match serde_json::from_str::<Value>(&raw) {
        Ok(Value::Object(m)) => bracket_rekey(&m),
        _ => Obj::new(), // ponytail: 원본은 dict 아닌 JSON(목록 등)이면 AttributeError 로 틱이 죽는다 -- 여기는 {}
    }
}
/// 임시 파일 → 교체(원자적, B1). 원본 `json.dumps(ensure_ascii=False)` 와 글자는 다르고(구분자·실수 표기) 뜻은 같다(양방향 왕복 시험).
pub fn bracket_save(dir: &Path, st: &Obj) -> Result<(), String> {
    let tmp = bracket_path(dir).with_extension("tmp");
    let body = serde_json::to_string(st).map_err(|e| format!("ValueError: {e}"))?;
    std::fs::write(&tmp, body).and_then(|_| std::fs::rename(&tmp, bracket_path(dir))).map_err(|e| format!("OSError: {e}"))
}

/// 지정가 대기 복구 파일(W4) -- 심볼마다 하나 · 칸 2부터 `_n`.
pub fn limit_path(dir: &Path, symbol: &str, wid: i64) -> PathBuf {
    dir.join(if wid == 1 { format!("manual_limit_wait_{symbol}.json") } else { format!("manual_limit_wait_{symbol}_{wid}.json") })
}

/// 감시 진입 닿음 = since 뒤 1초 체결가(USDT) 범위가 선에 닿았나(선 손절 감시와 같은 원천 · 1초 사이 꼬리도 빠짐없이).
pub fn stop_touched(limit_usdt: f64, pside: &str, prices: Option<&[(f64, f64)]>, since: f64) -> bool {
    let pts: Vec<f64> = prices.unwrap_or(&[]).iter().filter(|(t, _)| *t > since).map(|x| x.1).collect();
    !pts.is_empty() && if pside == "LONG" { pts.iter().copied().fold(f64::NEG_INFINITY, f64::max) >= limit_usdt } else { pts.iter().copied().fold(f64::INFINITY, f64::min) <= limit_usdt }
}

/// `_busy`: 진행 중(working·submitting).
pub fn busy(st: &Obj) -> bool {
    !st.is_empty() && matches!(st.get("phase").and_then(Value::as_str), Some("working" | "submitting"))
}

/// 주문 잠금(G8). 잠겨 있으면 None(= 409, 기다리지 않는다) · 아니면 잠근 채 돌린다.
/// 2026-10-16 접수 검사 ~ 상태 점유 사이 await 동안 두 번째 접수(폰+데스크탑)가 검사를 빠져나가 두 번 나갔다.
pub async fn busy_or<T>(lock: &tokio::sync::Mutex<()>, fut: impl std::future::Future<Output = T>) -> Option<T> {
    let _g = lock.try_lock().ok()?;
    Some(fut.await)
}

#[derive(Debug, Clone, PartialEq)]
pub enum SlotId {
    Entry,
    Side(String),
}

/// exec 의 상태 → 파이썬 상태 칸 이름.
pub fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::Idle => "idle",
        Phase::Working => "working",
        Phase::FilledMaker => "filled_maker",
        Phase::Rejected => "rejected",
        Phase::Error => "error",
        Phase::Cancelled => "cancelled",
        Phase::FilledTaker => "filled_taker",
        Phase::TakerFailed => "taker_failed",
    }
}

/// 계획 dict → 집행 입력. 원본 run_exit 가 읽는 칸(plan["symbol"] 등은 없으면 KeyError).
pub fn exec_plan_of(plan: &Value) -> Result<ExecPlan, String> {
    let p = plan.as_object().ok_or_else(|| format!("TypeError: '{}' object is not subscriptable", type_name(plan)))?;
    Ok(ExecPlan {
        symbol: req_str(p, "symbol")?,
        side: req_str(p, "side")?,
        position_side: req_str(p, "positionSide")?,
        market: p.get("type").and_then(Value::as_str) == Some("MARKET"),
        price: if p.get("type").and_then(Value::as_str) == Some("MARKET") { 0.0 } else { req_float(p, "price")? },
        quantity: req_float(p, "quantity")?,
        fallback_after_sec: float_or0(p, "fallback_after_sec"),
        maker_only: false,
        vol_bpm: p.get("vol_bpm").and_then(Value::as_f64),
        market_reason: p.get("market_reason").and_then(Value::as_str).map(String::from),
        target_leverage: p.get("target_leverage").filter(|v| truthy(Some(v))).map_or(Ok(0.0), crate::sign::py_float)?.trunc() as i64,
    })
}

/// 실제 청산 = `exec::run_exit` 를 돌리고 결과를 상태 칸에 옮긴다(원본 run_exit 가 state 에 쓰는 칸 이름 그대로).
/// 시장가 분기(M6·M7·M8·M9)는 원본처럼 시장가로 닫는다(2026-11-21 사용자 결정) -- filled_taker / taker_failed.
/// ponytail: 칸은 끝난 뒤 한 번 옮긴다(도는 동안 화면 폴링은 "submitting" 을 본다) -- 화면이 진행 중 체결량을 읽게 되면 Shared 를 칸 대신 둔다.
pub async fn run_exit_into<E: Exchange>(ex: &mut E, plan: &Value, slot: &mut Obj, done_at: impl Fn() -> String) -> Result<(), String> {
    let p = exec_plan_of(plan)?;
    let s: Shared = Arc::new(Mutex::new(ExecState::default()));
    exec::run_exit(ex, &p, &s).await;
    let e = s.lock().unwrap().clone();
    slot.insert("phase".into(), json!(phase_name(e.phase)));
    slot.insert("kind".into(), json!("exit"));
    for (k, v) in [("quantity", f(e.quantity)), ("filled", f(e.filled)), ("limit_price", f(e.limit_price)), ("repegs", json!(e.repegs)),
                   ("deadline_sec", f(e.deadline_sec)), ("vol_bpm", json!(e.vol_bpm)), ("order_id", json!(e.order_id)), ("taker_qty", f(e.taker_qty))] {
        slot.insert(k.into(), v);
    }
    if let Some(err) = e.error {
        slot.insert("error".into(), json!(err));
    }
    if let Some(id) = e.taker_order_id {
        slot.insert("taker_order_id".into(), json!(id));
    }
    if e.market_reason.is_some() {
        slot.insert("market_reason".into(), json!(e.market_reason));
    }
    slot.insert("done_at".into(), json!(done_at()));
    Ok(())
}

/// 감시·상태 칸의 주인. 원본 register() 의 지역 변수들(진행 중 상태는 **메모리** -- 재시작하면 사라진다, R9).
#[derive(Debug, Clone)]
pub struct Watcher {
    pub dir: PathBuf,
    pub eth_symbol: String,
    /// 진행 중인 수동 주문 하나(진입·스위칭은 계좌 전체에 이 칸 하나, G5).
    pub entry: Obj,
    /// 2026-10-20 사용자 «청산만 코인별 허용»: 위 칸을 다른 코인이 쓰는 동안 이 코인 청산은 자기 칸으로(G6).
    pub side_exits: Obj,
    /// 대기 칸(코인별·칸별 `<코인>` / `<코인>#n`, W1).
    pub waits: Obj,
    /// 2026-10-29 선 무장을 감시 틱이 쏘기 시작한 키 -- «선 끌어 수정»이 이걸 보고 거절한다(검토 B1).
    pub exiting: BTreeSet<String>,
    /// 원본 print 줄(서버 로그).
    pub logs: Vec<String>,
    /// 알린 (칸 키, 예외 종류) -- 프로세스 수명 동안 한 번(2026-11-21).
    pub alerted: BTreeSet<(String, String)>,
}

impl Watcher {
    pub fn new(dir: impl Into<PathBuf>, eth_symbol: &str) -> Watcher {
        let mut entry = Obj::new();
        entry.insert("phase".into(), json!("idle"));
        Watcher { dir: dir.into(), eth_symbol: eth_symbol.into(), entry, side_exits: Obj::new(), waits: Obj::new(), exiting: BTreeSet::new(), logs: vec![], alerted: BTreeSet::new() }
    }
    pub(crate) fn log(&mut self, s: String) {
        self.logs.push(s);
    }
    pub fn load(&self) -> Obj {
        bracket_load(&self.dir)
    }
    pub fn save(&self, st: &Obj) -> Result<(), String> {
        bracket_save(&self.dir, st)
    }
    /// 원본 tp_leg_filled(orders.py): 사용자 스트림 «우리 익절(dbtp) 체결» → 그 조각을 무장에서 뺀다. 모르는 조각·조각 없는 무장은 그대로
    /// (스트림이 놓치면 다음 다시 걸기가 «목록에 없는 조각 = 체결»로 뺀다).
    pub fn tp_leg_filled(&mut self, symbol: &str, side: &str, cid: &str) -> Result<(), String> {
        let mut st = self.load();
        let key = bracket_key(symbol, side);
        let Some(cur) = st.get_mut(&key).and_then(Value::as_object_mut) else { return Ok(()) };
        let legs = match cur.get("tp_legs").and_then(Value::as_array) {
            Some(l) if !l.is_empty() && l.iter().any(|x| x.get("cid").and_then(Value::as_str) == Some(cid)) => l.clone(),
            _ => return Ok(()),
        };
        let legs: Vec<Value> = legs.into_iter().map(|mut x| {
            let f = x.get("cid").and_then(Value::as_str) == Some(cid);
            x["filled"] = json!(f);
            x
        }).collect();
        set_tp_legs(cur, &legs)?;
        let left = cur.get("tp_legs").and_then(Value::as_array).map_or(1, Vec::len);
        self.save(&st)?;
        self.log(format!("bracket {side}: 분할 익절 조각 체결 {cid} -- 남은 조각 {left}"));
        Ok(())
    }
    /// 사용자 스트림 대기열(`userstream::Hub::tp_fills`)을 비우며 tp_leg_filled. 감시 루프가 틱 앞에서 부른다.
    /// ponytail: 원본은 사건 순간 부른다 -- 여기는 다음 틱(≤2초·체결 알림이면 바로 깸) 앞에서. 화면 선이 그만큼 늦게 사라진다.
    pub fn drain_tp_fills(&mut self, q: &Mutex<Vec<(String, String, String)>>) -> Result<(), String> {
        let items = std::mem::take(&mut *q.lock().unwrap());
        for (s, ps, c) in items {
            self.tp_leg_filled(&s, &ps, &c)?;
        }
        Ok(())
    }
    fn exec_symbols(&self) -> Vec<(&str, String)> {
        let mut v = vec![("eth", self.eth_symbol.clone())];
        v.extend(EXEC_REST.iter().map(|(a, s)| (*a, s.to_string())));
        v
    }
    /// 주문/시장 심볼 → 코인(exec_asset_of).
    pub fn exec_asset_of(&self, symbol: Option<&Value>) -> Option<&'static str> {
        let sym = symbol.and_then(Value::as_str)?;
        for (a, s) in self.exec_symbols() {
            let m = MARKET_SYMBOLS.iter().find(|x| x.0 == a).map(|x| x.1);
            if sym == s || Some(sym) == m {
                return MARKET_SYMBOLS.iter().find(|x| x.0 == a).map(|x| x.0);
            }
        }
        None
    }
    fn asset_of(&self, symbol: Option<&Value>) -> &'static str {
        self.exec_asset_of(symbol).unwrap_or("eth")
    }

    /// 청산이 쓸 상태 칸. 위 칸이 비었으면 위 칸 · 다른 코인이 쓰는 중이면 이 코인 청산 칸 · 같은 코인이 쓰는 중이면 None(409). 원본처럼 코인 칸을 만들어 둔다.
    pub fn exit_slot(&mut self, asset: &str) -> Option<SlotId> {
        let side = self.side_exits.entry(asset.to_string()).or_insert_with(|| json!({"phase": "idle"}));
        if side.as_object().is_some_and(busy) {
            return None;
        }
        if !busy(&self.entry) {
            return Some(SlotId::Entry);
        }
        let ea = if t(&self.entry, "asset") { self.entry.get("asset").cloned().unwrap() } else { json!("eth") };
        if ea == json!(asset) { None } else { Some(SlotId::Side(asset.into())) }
    }
    pub fn slot_mut(&mut self, id: &SlotId) -> &mut Obj {
        match id {
            SlotId::Entry => &mut self.entry,
            SlotId::Side(a) => self.side_exits.get_mut(a).and_then(Value::as_object_mut).expect("exit_slot 이 만든 칸"),
        }
    }
    pub fn wkey(asset: &str, wid: i64) -> String {
        if wid == 1 { asset.into() } else { format!("{asset}#{wid}") }
    }
    pub fn wasset(key: &str) -> &str {
        key.split('#').next().unwrap_or(key)
    }
    /// 이 코인의 대기 칸 키(칸 번호 순) -- 있는 칸만.
    pub fn wslots(&self, asset: &str) -> Vec<String> {
        let mut v: Vec<(i64, String)> = self.waits.keys().filter(|k| Self::wasset(k) == asset)
            .map(|k| (k.split_once('#').map_or(1, |(_, n)| n.parse().unwrap_or(1)), k.clone())).collect();
        v.sort_by_key(|x| x.0);
        v.into_iter().map(|x| x.1).collect()
    }
    pub fn wait_slot(&mut self, asset: &str, wid: i64) -> &mut Obj {
        self.waits.entry(Self::wkey(asset, wid)).or_insert_with(|| json!({"phase": "idle"})).as_object_mut().expect("대기 칸은 dict")
    }
    fn settle(&self, r: Result<(), String>, slot: &mut Obj, now: f64) {
        // 집행 예외가 새면 phase 가 working 에 영영 묶인다 -- 모든 접수가 409, 서버 손절 감시도 쏘지 않는다(10-16 감사, R3)
        if let Err(e) = r {
            slot.insert("phase".into(), json!("error"));
            slot.insert("error".into(), json!(format!("집행 오류 {e} — 거래소에서 직접 확인하세요")));
            slot.insert("done_at".into(), json!(iso_utc(now)));
        }
    }

    /// 2026-11-21 결함 수정(파이썬 exit_retry_clear): 청산이 실패(retry_after)한 뒤 다시 쏘기 전에 앞선 청산의 따라가기 GTX 를 지우고 없음을 확인한다.
    /// false = 이번엔 쏘지 않는다(무장·fired 유지 · 실패면 30초 뒤 다시). 첫 발동(retry_after 없음)은 조회를 늘리지 않는다(손절 속도).
    /// 같은 코인 청산 칸이 바쁘면(그 사이 수동 청산 시작) 그 따라가기 주문을 지우지 않게 쏘지 않고 넘어간다.
    async fn exit_retry_clear<I: WatchIo>(&mut self, io: &mut I, b: &mut Obj, pside: &str, b_asset: &str, symbol: &str) -> Result<bool, String> {
        if !t(b, "retry_after") {
            return Ok(true);
        }
        if io.order_locked() || self.exit_slot(b_asset).is_none() {
            return Ok(false);
        }
        let r = io.cancel_exit_orders(symbol, pside).await?;
        if truthy(r.get("errors")) {
            let es: Vec<String> = r["errors"].as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_else(|| vec![py_str(&r["errors"])]);
            b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
            b.insert("last_error".into(), json!(format!("앞선 청산 주문 정리 실패: {}", es.join("; "))));
            return Ok(false);
        }
        Ok(true)
    }

    /// 2026-11-21 결함 수정(파이썬 bracket_tick_alert): 건너뛴 칸을 **칸·예외 종류마다 한 번** 로그·알림(보내기는 `WatchIo::alert` -- 감시를 막지 않게 뒤에서).
    pub fn tick_alert<I: WatchIo>(&mut self, io: &mut I, key: &str, why: &str) {
        let kind = why.split_once(": ").map_or(why, |x| x.0).to_string();
        if !self.alerted.insert((key.to_string(), kind)) {
            return;
        }
        let msg = format!("🔴 손절 감시: 무장 칸 {key} 을 판정하지 못해 건너뜁니다({why}) — 다른 코인은 계속 봅니다 · 칸은 지우지 않았습니다(manual_bracket_state.json 확인)");
        self.log(format!("bracket watch: {msg}"));
        io.alert(&msg);
    }

    /// 감시 한 틱. 무장된 측면마다: 포지션이 사라졌으면(TP·비상 스탑·수동 청산) 남은 우리 주문을 지우고 내린다.
    /// SL 을 넘었으면 **메이커 추격 청산**(청산 버튼과 같은 run_exit)을 돌린다. Err = 원본에서 틱 밖으로 새던 예외(감시 루프가 삼킨다).
    pub async fn bracket_tick<I: WatchIo>(&mut self, io: &mut I) -> Result<(), String> {
        let mut st = self.load();
        if st.is_empty() || !io.exec_enabled() {
            return Ok(()); // G4: 스위치가 꺼지면 감시 틱도 아무것도 안 한다(무장 파일은 그대로)
        }
        let account = io.account().await?;
        let Some(acc) = account.as_object() else {
            return Err(format!("AttributeError: '{}' object has no attribute 'get'", type_name(&account)));
        };
        let acct_ts = parse_iso(&py_str(&val(acc, "generated_at"))).unwrap_or(0.0);
        let acct_ok = t(acc, "ok");
        let positions: Vec<Position> = match acc.get("positions") {
            Some(Value::Array(a)) => a.iter().map(|p| Position {
                symbol: p.get("symbol").and_then(Value::as_str).unwrap_or("\u{0}없음").into(),
                side: p.get("side").and_then(Value::as_str).unwrap_or("\u{0}없음").into(),
                qty: Some(if truthy(p.get("qty")) { py_float(&p["qty"]).unwrap_or(0.0) } else { 0.0 }),
            }).collect(),
            _ => vec![],
        };
        let mut changed = false;
        // dict 아닌 칸 = 망가짐 -- 합치기(bracket_merge)가 건드리지 않아 파일에 그대로 남는다
        let seen_at: Obj = st.iter().filter(|(_, v)| v.is_object()).map(|(k, v)| (k.clone(), v.get("armed_at").cloned().unwrap_or(Value::Null))).collect();
        for key in st.keys().cloned().collect::<Vec<_>>() {
            let mut b = match st.get(&key).cloned() {
                Some(Value::Object(b)) => b,
                Some(v) => {
                    // 2026-11-21 결함 수정: 망가진 칸 하나가 틱을 매번 죽여 뒤 코인 손절 판정이 멈췄다 -- 그 칸만 건너뛴다(지우지 않는다)
                    self.tick_alert(io, &key, &format!("AttributeError: '{}' object has no attribute 'get'", type_name(&v)));
                    continue;
                }
                None => continue,
            };
            match self.tick_one(io, &key, &mut b, acct_ok, acct_ts, &positions).await {
                Ok((keep, ch)) => {
                    changed |= ch;
                    if keep {
                        st.insert(key, Value::Object(b));
                    } else {
                        st.remove(&key);
                    }
                }
                Err(e) => {
                    st.insert(key.clone(), Value::Object(b)); // 원본: 예외 전까지 고친 사본이 그대로 남는다(합치기는 changed 일 때만)
                    self.tick_alert(io, &key, &e);
                }
            }
        }
        if changed {
            // 🔴다시 읽고 **내가 본 무장(armed_at)** 에만 적용한다 -- 위 await 동안 새 진입이 다른 측면(또는 같은 측면 물타기)을 무장했으면 통째로 쓰면 그걸 지운다(B19).
            self.save(&bracket_merge(&self.load(), &st, &seen_at))?;
        }
        Ok(())
    }

    /// 한 측면. 돌려주기 = (무장을 남기나, 바뀌었나).
    async fn tick_one<I: WatchIo>(&mut self, io: &mut I, key: &str, b: &mut Obj, acct_ok: bool, acct_ts: f64, positions: &[Position])
        -> Result<(bool, bool), String> {
        let pside = match b.get("side") {
            Some(v) if truthy(Some(v)) => py_str(v),
            _ => key.rsplit(':').next().unwrap_or(key).to_string(),
        };
        let sym_v = b.get("symbol").cloned();
        let sym_err = || "KeyError: 'symbol'".to_string();
        // 2026-09-27 SL = 마감된 USDT 5분봉 종가(사용자 지시). 차트가 쓰는 60초 캐시라 REST 가 늘지 않는다. 못 읽으면 None -> hold -- «정리»는 계속 판단한다.
        let now_s = io.now();
        let last_bar: Option<(f64, f64)> = match &sym_v {
            None => {
                self.log(format!("bracket {pside}: 5분봉 못 읽음 {}", sym_err()));
                None
            }
            Some(sv) => {
                let asset = self.asset_of(Some(sv));
                let r: Result<Option<(f64, f64)>, String> = async {
                    let rows = io.bars(asset).await?;
                    let mut closed = vec![];
                    for c in rows {
                        let tm = c.get("time").ok_or("KeyError: 'time'")?;
                        let tm = py_float(tm)?.trunc();
                        if tm + BAR_SEC <= now_s {
                            closed.push(c);
                        }
                    }
                    match closed.last() {
                        None => Ok(None),
                        Some(c) => Ok(Some((py_float(&c["time"])?.trunc() + BAR_SEC, py_float(c.get("close").ok_or("KeyError: 'close'")?)?))),
                    }
                }.await;
                r.unwrap_or_else(|e| {
                    self.log(format!("bracket {pside}: 5분봉 못 읽음 {e}"));
                    None
                })
            }
        };
        let mut touch: Option<(f64, f64)> = None;
        if t(b, "lines") {
            // 2026-10-28 무장 뒤 1초 마지막 체결가 범위(USDT) -- 감시 2초 틱 사이도 빠짐없이 · 손절 해제 → 다시 켜기 사이에 지나간 가격은 안 본다
            let since = pyfmt::max(req_float(b, "armed_at")?, float_or0(b, "watch_from"));
            let sv = sym_v.as_ref().ok_or_else(sym_err)?;
            let pts: Vec<f64> = io.sec_prices(self.asset_of(Some(sv)))?.unwrap_or_default().into_iter().filter(|x| x.0 > since).map(|x| x.1).collect();
            if !pts.is_empty() {
                touch = Some((pts.iter().copied().reduce(pyfmt::min).unwrap(), pts.iter().copied().reduce(pyfmt::max).unwrap()));
            }
        }
        // 원본 bracket_action·bracket_sl_hit 의 float(armed["armed_at"]) -- 망가진 값이면 예외(→ 그 칸만 건너뜀 · 2026-11-21)
        // ponytail: 원본은 그 식에 닿는 갈래에서만 터진다(손절 해제된 규칙 무장 + 포지션 있음이면 안 닿는다) -- 여기는 늘 본다(arm_bracket 은 늘 숫자를 쓴다)
        req_float(b, "armed_at")?;
        let armed = Armed::from_value(&Value::Object(b.clone()));
        let held = positions.iter().any(|p| Some(&json!(p.symbol)) == sym_v.as_ref() && p.side == pside && p.qty.unwrap_or(0.0).abs() > 0.0);
        if acct_ok && !held {
            req_float(b, "armed_at")?; // 원본 bracket_action 의 float(armed["armed_at"])
        }
        let act = bracket_action(&armed, &pside, positions, acct_ok, acct_ts, last_bar, touch);
        let sv = sym_v.clone().ok_or_else(sym_err)?;
        let b_asset = self.asset_of(Some(&sv));
        for wk in self.wslots(b_asset) {
            let ws = self.waits.get_mut(&wk).and_then(Value::as_object_mut).unwrap();
            if matches!(act, TickAction::Clear | TickAction::Fire) && busy(ws) && py_eq(ws.get("side"), Some(&json!(pside)))
                && py_eq(obj_or_empty(ws.get("plan")).get("symbol"), Some(&sv)) {
                // 2026-10-10 손절·청산으로 포지션이 닫히는데 같은 쪽 대기 진입이 걸려 있으면 그것도 거둔다(나중에 체결돼 손절 없이 다시 열리지 않게, W9) · 2026-10-20 다른 코인 대기는 둔다
                ws.insert("action".into(), json!("cancel"));
                ws.insert("action_reason".into(), json!("포지션이 닫힘(손절·청산) — 같은 쪽 대기 진입 취소"));
            }
        }
        let symbol = py_str(&sv);
        if act == TickAction::Clear {
            if io.now() < float_or0(b, "retry_after") {
                return Ok((true, false));
            }
            let r = io.clear_bracket(&symbol, &pside, false, false).await?;
            self.log(format!("bracket {pside}: 포지션 없음 -> 정리 {}", py_str(&r)));
            if truthy(r.get("errors")) {
                // 🔴못 지운 물타기 지정가(dbad)가 남으면 손절 없이 다시 열린다 -- 무장을 남겨 30초 뒤 다시 지운다(B10)
                b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
                return Ok((true, true));
            }
            return Ok((false, true));
        }
        if act == TickAction::SlLeg {
            return self.fire_leg(io, key, b, &pside, &symbol, b_asset, &armed, touch, last_bar).await;
        }
        if act != TickAction::Fire || self.exit_slot(b_asset).is_none() || io.now() < float_or0(b, "retry_after") {
            return Ok((true, false));
        }
        let mut changed = false;
        if t(b, "lines") && !t(b, "fired") {
            // 2026-10-28 검증: 걸린 익절 지정가를 **먼저** 지운다 -- 청산 수량을 그 뒤에 읽어야 겹쳐 넘기지 않고, 못 지우면 계좌 조회(무게 ~50)를
            //   매 2초 반복하지 않게 10초 쉰다. 지웠으면 fired 로 굳혀 닫힐 때까지 쏜다(익절이 없는데 닿음 창이 지나 조용히 멈추지 않게, B14).
            // 2026-11-11 손절선만 지움(sl_off): 그 상태에서 내린 «fire» 는 익절 감시뿐이라 쏜다 · 판단 뒤 새로 지웠으면(b 엔 없음) 손절일 수 있어 다음 틱이 다시 판단
            if io.order_locked() || (t(&obj_or_empty(self.load().get(key)), "sl_off") && !t(b, "sl_off")) {
                return Ok((true, false)); // D9: 주문 잠금 중이면 손절 발동을 미룬다(G9) -- 다음 틱에
            }
            let fresh = obj_or_empty(self.load().get(key));
            if !py_eq(fresh.get("armed_at"), b.get("armed_at"))
                || !py_eq(obj_or_empty(fresh.get("mod")).get("at"), obj_or_empty(b.get("mod")).get("at")) {
                return Ok((true, false)); // 2026-10-29 그 사이 선이 수정됐다 -- 옛 사본(옛 레벨·watch_from)으로 쏘지 않는다, 다음 틱이 새 선으로(검토 S1, B15)
            }
            self.exiting.insert(key.into()); // 🔴첫 await 전에 -- 수정 요청이 잠금 검사 뒤 이 틈에 끼어들어 익절을 다시 걸지 못하게(검토 B1)
            let r = match io.clear_bracket(&symbol, &pside, false, false).await {
                Ok(r) => r,
                Err(e) => {
                    return Err(e); // 원본도 discard 없이 예외가 샌다 -- exiting 에 남는다
                }
            };
            if truthy(r.get("errors")) {
                self.exiting.remove(key);
                b.insert("retry_after".into(), f(io.now() + TP_CANCEL_RETRY_SEC));
                let errs: Vec<String> = r["errors"].as_array().map(|a| a.iter().map(py_str).collect()).unwrap_or_else(|| vec![py_str(&r["errors"])]);
                b.insert("last_error".into(), json!(format!("익절 지정가 취소 실패: {}", errs.join("; "))));
                return Ok((true, true));
            }
            // 2026-11-11 손절을 지운 뒤의 fire = 익절 -- 표시해야 «손절 해제로 멈춘 손절 청산»(True)과 구별된다(B23)
            let fired = if t(b, "sl_off") { json!("tp") } else { json!(true) };
            b.insert("fired".into(), fired.clone());
            b.insert("tp_resting".into(), json!(false));
            let mut st2 = self.load(); // 2026-10-29 fired 를 곧바로 파일에 -- run_exit(최대 2분) 동안 수정이 «안 쏜 무장»으로 보지 않게(검토 B1)
            if let Some(Value::Object(c)) = st2.get_mut(key) {
                if py_eq(c.get("armed_at"), b.get("armed_at")) {
                    c.insert("fired".into(), fired);
                    c.insert("tp_resting".into(), json!(false));
                    self.save(&st2)?;
                }
            }
            self.exiting.remove(key);
            changed = true;
        }
        if !self.exit_retry_clear(io, b, &pside, b_asset, &symbol).await? {
            return Ok((true, true));
        }
        let pl = io.exit_plan(&pside, 1.0, b_asset, &symbol).await;
        // 🔴검사는 조립(await) **뒤**에 -- 그 사이 «손절 해제»나 사용자 청산이 들어오면 쏘지 않는다(남의 상태를 clear 하고 두 번 청산하지 않게, B16)
        if t(&obj_or_empty(self.load().get(key)), "sl_off") && !(t(b, "lines") && t(b, "sl_off")) {
            return Ok((true, changed)); // 선 무장이 이미 손절 없이 판단한 fire = 익절(위와 같음)
        }
        let slot = self.exit_slot(b_asset); // 2026-10-20 다른 코인 주문이 도는 중이면 이 코인 칸으로 청산(옛 판: 그 주문이 끝날 때까지 손절을 미뤘다)
        if io.order_locked() || slot.is_none() {
            return Ok((true, changed)); // D9
        }
        let slot = slot.unwrap();
        let plan_v = match pl {
            Err(err) if err.get("error").and_then(Value::as_str) == Some("no_position") => {
                // 이미 닫혔다 -- 정리하고 내린다(못 지웠으면 다음 틱 «clear» 가 다시)
                let r = io.clear_bracket(&symbol, &pside, false, false).await?;
                return Ok((truthy(r.get("errors")), true));
            }
            Err(err) => {
                // 🔴조회 실패는 «닫혔다»가 아니다 -- 비상 스탑을 두고 30초 뒤 다시 본다(B17).
                b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
                b.insert("last_error".into(), py_or(vec![val(err.as_object().unwrap_or(&Obj::new()), "detail"), val(err.as_object().unwrap_or(&Obj::new()), "error"), Value::Null]));
                return Ok((true, true));
            }
            Ok(p) if truthy(p.get("blocked")) => {
                b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
                b.insert("last_error".into(), p["blocked"].clone()); // 2026-10-29 조립 예외는 plan None·detail 없음(검토 P1)
                return Ok((true, true));
            }
            Ok(p) => p,
        };
        let now = io.now();
        {
            let s = self.slot_mut(&slot);
            s.clear();
            for (k, v) in [("phase", json!("submitting")), ("kind", json!("exit")), ("trigger", json!("bracket_sl")), ("side", json!(pside)),
                           ("plan", plan_v.clone()), ("asset", json!(b_asset)), ("started_at", json!(iso_utc(now)))] {
                s.insert(k.into(), v);
            }
        }
        // D8: 청산(최대 ~120초)을 감시 루프 안에서 기다린다 -- 그동안 다른 코인·측면 판정이 멈춘다(원본 그대로)
        let mut s = std::mem::take(self.slot_mut(&slot));
        let r = io.run_exit(&plan_v, &mut s).await;
        self.settle(r, &mut s, io.now());
        let phase = val(&s, "phase");
        let mut keep = true;
        // 🔴다 닫혔을 때만 비상 스탑을 지운다 -- 청산이 실패했는데 지우면 무방비가 된다(B18).
        if matches!(phase.as_str(), Some("filled_maker" | "filled_taker")) {
            let cl = io.clear_bracket(&symbol, &pside, false, false).await;
            let cl = match cl {
                Ok(v) => v,
                Err(e) => {
                    *self.slot_mut(&slot) = s;
                    return Err(e);
                }
            };
            keep = truthy(cl.get("errors")); // 못 지웠으면 남겨 둔다 -- 다음 틱 «clear» 가 다시 지운다
            s.insert("bracket_cleanup".into(), cl);
        } else {
            b.insert("retry_after".into(), f(io.now() + RETRY_SEC)); // 🔴불명(error) 뒤 30초에 다시 쏠 때 «살아 있을 수 있는 주문» 위에 새로 건다(원본 그대로 · 진행 기록 4)
            b.insert("last_error".into(), val(&s, "error"));
        }
        let msg = format!("bracket {pside}: SL {} (레벨 {}) 5분봉 종가 {} · 1초 범위 {} 이탈 -> {}", py_str(&val(b, "sl_price")), py_str(&val(b, "sl_level")),
                          bar_repr(last_bar), touch_repr(touch), py_str(&val(&s, "phase")));
        *self.slot_mut(&slot) = s;
        self.log(msg);
        Ok((keep, true))
    }

    /// 2026-11-11 분할 손절: 닿은 조각 몫만 peg 청산 -- 익절 지정가는 남기고, 끝나면 그 조각을 빼고 남은 수량으로 익절 조각을 다시 건다(B20).
    #[allow(clippy::too_many_arguments)]
    async fn fire_leg<I: WatchIo>(&mut self, io: &mut I, key: &str, b: &mut Obj, pside: &str, symbol: &str, b_asset: &'static str, armed: &Armed,
                                  touch: Option<(f64, f64)>, last_bar: Option<(f64, f64)>) -> Result<(bool, bool), String> {
        if self.exit_slot(b_asset).is_none() || io.order_locked() || io.now() < float_or0(b, "retry_after") {
            return Ok((true, false)); // D9
        }
        if !self.exit_retry_clear(io, b, pside, b_asset, symbol).await? {
            return Ok((true, true));
        }
        let legs: Vec<Value> = b.get("sl_legs").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut hit = sl_legs_hit(pside, armed, touch, last_bar);
        hit.sort();
        hit.dedup();
        let w = |x: &Value| if truthy(x.get("w")) { py_float(&x["w"]).unwrap_or(1.0) } else { 1.0 };
        let num = legs.iter().enumerate().filter(|(i, _)| hit.contains(i)).fold(0.0, |s, (_, x)| s + w(x));
        let den = legs.iter().fold(0.0, |s, x| s + w(x));
        let frac = num / if den != 0.0 { den } else { 1.0 };
        let mut st2 = self.load(); // 쏜 조각을 곧바로 파일에 -- 청산(최대 2분) 동안 수정·다음 틱이 «안 쏜 조각»으로 보지 않게
        let ok = match st2.get(key).and_then(Value::as_object) {
            Some(c) => py_eq(c.get("armed_at"), b.get("armed_at")) && !t(c, "sl_off")
                && c.get("sl_legs").and_then(Value::as_array).map_or(0, Vec::len) == legs.len(),
            None => false,
        };
        if !ok {
            return Ok((true, false));
        }
        let c = st2.get_mut(key).and_then(Value::as_object_mut).unwrap();
        let fired: Vec<Value> = c["sl_legs"].as_array().unwrap().iter().enumerate()
            .map(|(i, x)| if hit.contains(&i) { let mut m = x.as_object().cloned().unwrap_or_default(); m.insert("fired".into(), json!(true)); Value::Object(m) } else { x.clone() })
            .collect();
        c.insert("sl_legs".into(), Value::Array(fired));
        self.save(&st2)?;
        let pl = io.exit_plan(pside, pyfmt::min(1.0, frac), b_asset, symbol).await;
        let slot = self.exit_slot(b_asset);
        if t(&obj_or_empty(self.load().get(key)), "sl_off") || io.order_locked() || slot.is_none() {
            return Ok((true, false));
        }
        let slot = slot.unwrap();
        let plan_v = match pl {
            Ok(p) if !truthy(p.get("blocked")) => p,
            other => {
                // 닫혔으면(no_position) 다음 틱 «clear» 가 정리 · 조회 실패는 30초 뒤 다시(쏜 표시는 남는다)
                b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
                let le = match other {
                    Err(e) => { let e = e.as_object().cloned().unwrap_or_default(); py_or(vec![val(&e, "detail"), val(&e, "error"), Value::Null]) }
                    Ok(p) => p["blocked"].clone(),
                };
                b.insert("last_error".into(), le);
                return Ok((true, true));
            }
        };
        let now = io.now();
        {
            let s = self.slot_mut(&slot);
            s.clear();
            for (k, v) in [("phase", json!("submitting")), ("kind", json!("exit")), ("trigger", json!("bracket_sl_leg")), ("side", json!(pside)),
                           ("plan", plan_v.clone()), ("asset", json!(b_asset)), ("started_at", json!(iso_utc(now)))] {
                s.insert(k.into(), v);
            }
        }
        let mut s = std::mem::take(self.slot_mut(&slot));
        let r = io.run_exit(&plan_v, &mut s).await;
        self.settle(r, &mut s, io.now());
        let phase = val(&s, "phase");
        if matches!(phase.as_str(), Some("filled_maker" | "filled_taker")) {
            let mut st3 = self.load();
            if let Some(Value::Object(c)) = st3.get_mut(key) {
                if py_eq(c.get("armed_at"), b.get("armed_at")) {
                    let live: Vec<Value> = c.get("sl_legs").and_then(Value::as_array).cloned().unwrap_or_default().into_iter().filter(|x| !truthy(x.get("fired"))).collect();
                    set_sl_legs(c, &live);
                    self.save(&st3).inspect_err(|_| *self.slot_mut(&slot) = s.clone())?;
                }
            }
            self.partial_exit_rearm_tp(io, plan_v.as_object().cloned().unwrap_or_default(), &mut s).await?; // 익절 지정가(조각) 수량을 남은 포지션에 맞춘다
        } else {
            b.insert("retry_after".into(), f(io.now() + RETRY_SEC));
            b.insert("last_error".into(), val(&s, "error"));
        }
        let hs: Vec<String> = hit.iter().map(|i| i.to_string()).collect();
        let msg = format!("bracket {pside}: 손절 조각 [{}] 닿음 -> {}% 청산 {}", hs.join(", "), (100.0 * frac).round_ties_even() as i64, py_str(&val(&s, "phase")));
        *self.slot_mut(&slot) = s;
        self.log(msg);
        Ok((true, true))
    }

    /// 2026-11-12 검토: 우리 익절 지정가 취소가 **일부만** 됐을 수 있다 -- 어느 게 남았는지 모르니 익절선(조각 포함)을 «거래소에 안 걸림»으로 바꿔
    /// 서버가 1초 닿음으로 닫게 한다. 레벨은 그대로. 무장이 바뀌었으면 손대지 않는다(false).
    pub fn tp_watch_only<I: WatchIo>(&mut self, io: &I, key: &str, b: &Obj) -> Result<bool, String> {
        self.tp_watch_only_at(io.now(), key, b)
    }
    /// 접수 문(익절 끄기·분할 조각)이 시계만 넘겨 부른다.
    pub fn tp_watch_only_at(&mut self, now: f64, key: &str, b: &Obj) -> Result<bool, String> {
        let mut st = self.load();
        let Some(Value::Object(cur)) = st.get_mut(key) else { return Ok(false) };
        if cur.is_empty() || !py_eq(cur.get("armed_at"), b.get("armed_at")) {
            return Ok(false);
        }
        cur.insert("tp_resting".into(), json!(false));
        if t(cur, "tp_legs") {
            let legs: Vec<Value> = cur["tp_legs"].as_array().cloned().unwrap_or_default().into_iter()
                .map(|x| { let mut m = x.as_object().cloned().unwrap_or_default(); m.insert("resting".into(), json!(false)); Value::Object(m) }).collect();
            cur.insert("tp_legs".into(), Value::Array(legs));
        }
        cur.insert("watch_from".into(), f(now));
        let m = json!({"at": now, "tp": val(cur, "tp_level"), "sl": val(cur, "sl_level"), "watch_only": true});
        cur.insert("mod".into(), m);
        let side = val(cur, "side");
        self.save(&st)?;
        self.log(format!("bracket {}: 익절 취소 일부 실패 → 익절선은 서버 감시로", py_str(&side)));
        Ok(true)
    }

    /// 2026-11-04 사용자 «SLTP 주문 걸린 채 중간 청산 50% 하면 익절 지정가가 사라진다» -- 부분 청산이 끝나면 **남은 포지션 수량**으로 익절을 다시 건다(B30).
    /// 지정가가 원래 안 걸려 있던(tp_resting False · 서버 감시) 무장·청산 진행(fired)·손절 해제 상태는 손대지 않는다.
    /// D7: 원본 그대로 strict 없이 건다(같은 쪽 전량 익절 둘이 될 수 있는 자리). 예외는 삼켜 state["tp_rearm"] 에.
    pub async fn partial_exit_rearm_tp<I: WatchIo>(&mut self, io: &mut I, plan: Obj, state: &mut Obj) -> Result<(), String> {
        if let Err(e) = self.partial_inner(io, &plan, state).await {
            // 청산 결과는 이미 났다 · 못 걸면 서버가 1초 닿음으로 감시
            state.insert("tp_rearm".into(), json!({"placed": false, "reason": e}));
        }
        Ok(())
    }

    async fn partial_inner<I: WatchIo>(&mut self, io: &mut I, plan: &Obj, state: &mut Obj) -> Result<(), String> {
        let (symbol, pside) = (req_str(plan, "symbol")?, req_str(plan, "positionSide")?);
        let key = bracket_key(&symbol, &pside);
        let b = obj_or_empty(self.load().get(&key));
        let pick = |r: &Value, ks: &[&str]| -> Value {
            Value::Object(ks.iter().map(|k| (k.to_string(), r.get(*k).cloned().unwrap_or(Value::Null))).collect())
        };
        let exiting = self.exiting.contains(&key);
        if t(&b, "tp_legs") && !t(&b, "fired") && !t(&b, "tp_off") && !exiting {
            // 2026-11-11 분할 익절 -- 조각 전부를 남은 수량 무게대로
            let r = io.place_tp_ladder(&b["tp_legs"], &symbol, &pside).await?;
            let r = as_dict(r)?;
            state.insert("tp_rearm".into(), pick(&Value::Object(r.clone()), &["qty", "placed", "reason"]));
            if truthy(obj_or_empty(r.get("cleared")).get("errors")) {
                // 2026-11-12 검토: 여기서 r["legs"] 가 없어 KeyError 로 조용히 끝났다(일부 조각만 지워진 채)
                self.tp_watch_only(io, &key, &b)?;
                return Ok(());
            }
            let mut st = self.load();
            if r.contains_key("cleared") {
                if let Some(Value::Object(c)) = st.get_mut(&key) {
                    if py_eq(c.get("armed_at"), b.get("armed_at")) {
                        let legs = r.get("legs").ok_or("KeyError: 'legs'")?.as_array().cloned().unwrap_or_default();
                        set_tp_legs(c, &legs)?;
                        self.save(&st)?;
                    }
                }
            }
            self.log(format!("bracket {pside}: 부분 청산 뒤 분할 익절 다시 걸음 {} (남은 수량 {})", py_str(&val(&r, "placed")), py_str(&val(&r, "qty"))));
            return Ok(());
        }
        if !(t(&b, "tp_resting") && t(&b, "tp_price")) || t(&b, "fired") || t(&b, "tp_off") || exiting {
            return Ok(()); // tp_off = 사용자가 익절선을 지움(11-11)
        }
        let mut spec = Obj::new();
        spec.insert("tp_price".into(), b["tp_price"].clone());
        if t(&b, "backstop_price") && !t(&b, "sl_off") {
            spec.insert("backstop_price".into(), b["backstop_price"].clone()); // 규칙 무장은 스탑도 같이(§9 M10 -- 요청 모양만 옮김)
        }
        let r = as_dict(io.place_bracket(&Value::Object(spec), &symbol, &pside, false).await?)?;
        state.insert("tp_rearm".into(), pick(&Value::Object(r.clone()), &["qty", "tp", "backstop", "placed", "reason"]));
        let mut st = self.load();
        let Some(Value::Object(c)) = st.get_mut(&key) else { return Ok(()) };
        if !py_eq(c.get("armed_at"), b.get("armed_at")) {
            return Ok(()); // 그 사이 새로 무장됐다 -- 남의 상태를 덮지 않는다
        }
        if r.contains_key("cleared") {
            // 키·포지션 조회 실패(아무것도 안 건드림)는 상태를 두고, 지운 뒤 못 걸었으면 서버 감시로 넘긴다
            c.insert("tp_resting".into(), json!(truthy(obj_or_empty(r.get("tp")).get("placed"))));
            let mut placed = obj_or_empty(c.get("placed"));
            placed.insert("tp".into(), val(&r, "tp"));
            c.insert("placed".into(), Value::Object(placed));
            self.save(&st)?;
        }
        self.log(format!("bracket {pside}: 부분 청산 뒤 익절 지정가 다시 걸음 {} (남은 수량 {})", py_str(&val(&r, "tp")), py_str(&val(&r, "qty"))));
        Ok(())
    }

    /// 무장 쓰기: 익절을 걸고(place_bracket) 저장본에 이 측면을 새로 적는다(B3·B4). 원본 칸 순서 그대로.
    /// D7: strict 없이 건다(원본 그대로). 2026-11-21 자동 물타기(add·add_order·auto)는 원본 PR 61 처럼 지웠다.
    pub async fn arm_bracket<I: WatchIo>(&mut self, io: &mut I, plan: &Obj, state: &mut Obj) -> Result<(), String> {
        let b = obj_or_empty(plan.get("bracket"));
        let pside = req_str(plan, "positionSide")?;
        if !t(&b, "available") {
            state.insert("bracket".into(), json!({"placed": false, "reason": py_or(vec![val(&b, "reason"), json!("청산맵 레벨 없음")])}));
            return Ok(());
        }
        let spec = if t(&b, "tp_off") { let mut m = b.clone(); m.insert("tp_price".into(), Value::Null); m } else { b.clone() }; // 익절 지운 선에 추가 진입 = 익절 지정가 안 건다
        let placed = match req_str(plan, "symbol") {
            Err(e) => Err(e),
            Ok(s) => io.place_bracket(&Value::Object(spec), &s, &pside, false).await,
        };
        state.insert("bracket".into(), placed.unwrap_or_else(|e| json!({"placed": false, "reason": e}))); // 결과 칸은 반드시 채운다(화면 폴링이 이걸 기다린다)
        let symbol = req_str(plan, "symbol")?;
        let mut st = self.load();
        let sb = state.get("bracket").cloned().unwrap_or(Value::Null);
        let tp_resting = truthy(obj_or_empty(obj_or_empty(Some(&sb)).get("tp")).get("placed"));
        let mut rec = Obj::new();
        for (k, v) in [("symbol", json!(symbol)), ("side", json!(pside)), ("sl_price", val(&b, "sl_price")),
                       ("sl_level", val(&b, "sl_level")), // 5분봉 종가 판정은 USDT 레벨로(bracket_sl_hit)
                       ("tp_price", val(&b, "tp_price")), ("backstop_price", val(&b, "backstop_price")),
                       ("rule", json!(t(&b, "rule"))), ("sl_name", val(&b, "sl_name")), // 2026-10-09 권고 규칙 손절 = 물타기해도 유지
                       ("lines", json!(t(&b, "lines"))), ("tp_level", val(&b, "tp_level")), ("sl_up", val(&b, "sl_up")),
                       ("tp_resting", json!(tp_resting)), // 익절 지정가가 못 걸렸으면 닿을 때 서버가 peg 로
                       ("armed_at", f(io.now())), ("placed", sb)] {
            rec.insert(k.into(), v);
        }
        for k in ["tp_off", "sl_off"] {
            if t(&b, k) {
                rec.insert(k.into(), b[k].clone()); // 2026-11-12·13 지운 익절·손절은 추가 진입 뒤에도 꺼진 채
            }
        }
        st.insert(bracket_key(&symbol, &pside), Value::Object(rec));
        self.save(&st)
    }

    /// 2026-11-13 검증: 선 주문 진입·추가가 체결된 뒤 TP 를 다시 걸 때 **지금 저장된 선**을 쓴다(W11). None = 다시 걸지 않는다(청산 진행 중 · 분할 조각).
    pub fn plan_from_current(&self, plan: &Obj) -> Result<Option<Obj>, String> {
        let b = obj_or_empty(plan.get("bracket"));
        let key = bracket_key(&req_str(plan, "symbol")?, &req_str(plan, "positionSide")?);
        let cur = self.load().get(&key).cloned();
        let cur = match cur {
            Some(Value::Object(c)) if t(&b, "lines") && !c.is_empty() && t(&c, "lines") => c,
            _ => return Ok(Some(plan.clone())),
        };
        if t(&cur, "fired") || t(&cur, "tp_legs") || t(&cur, "sl_legs") || self.exiting.contains(&key) {
            return Ok(None);
        }
        let mut nb = b.clone();
        for k in ["tp_price", "tp_level", "sl_price", "sl_level", "sl_name", "sl_up", "tp_off", "sl_off"] {
            nb.insert(k.into(), val(&cur, k));
        }
        let mut out = plan.clone();
        out.insert("bracket".into(), Value::Object(nb));
        Ok(Some(out))
    }

    /// 2026-11-14 사용자 «추가 진입 = 기존 포지션 SLTP 조건 그대로»: 무장은 그대로 두고 익절 지정가 수량만 합친 전량으로 다시 건다(B31).
    /// ponytail: 원본의 포지션별 asyncio.Lock(검토 11-18 #5) 대신 `&mut self` 가 한 줄로 세운다 -- 여러 작업이 나눠 쥐게 되면 키별 잠금을 다시 둔다.
    pub async fn keep_rearm<I: WatchIo>(&mut self, io: &mut I, plan: &Obj, state: &mut Obj) -> Result<(), String> {
        let key = bracket_key(&req_str(plan, "symbol")?, &req_str(plan, "positionSide")?);
        let cur = obj_or_empty(self.load().get(&key));
        if cur.is_empty() {
            // 검증 11-14(high): 대기 중 포지션이 닫혀 무장이 지워진 뒤 체결 = 새 포지션 -- 계획에 복사한 선으로 다시 무장
            return self.arm_bracket(io, plan, state).await;
        }
        if cur.get("sl_legs").and_then(Value::as_array).is_some_and(|a| a.iter().any(|g| truthy(g.get("fired")))) {
            // 검증 11-14: 손절 조각 청산 중 -- 그 청산 뒤 다시 걸기와 겹쳐 익절이 두 벌 걸리지 않게 이쪽은 건너뛴다
            state.insert("bracket".into(), json!({"kept": true, "note": "기존 익절·손절 그대로 · 손절 조각 청산 중이라 익절 수량은 그 뒤에 맞춥니다"}));
            return Ok(());
        }
        self.partial_exit_rearm_tp(io, plan.clone(), state).await?;
        let r = obj_or_empty(state.get("tp_rearm"));
        let note = if t(&r, "placed") {
            "기존 익절·손절 그대로 · 익절 지정가를 합친 수량으로 다시 걸음".to_string()
        } else if !r.is_empty() {
            format!("기존 익절·손절 그대로 · 🔴익절 지정가 다시 걸기 실패({}) — 서버가 닿으면 닫습니다", py_str(&py_or(vec![val(&r, "reason"), json!("?")])))
        } else {
            "기존 익절·손절 그대로".to_string()
        };
        state.insert("bracket".into(), json!({"kept": true, "tp_rearm": Value::Object(r), "note": note}));
        Ok(())
    }

    /// 부분 청산(<100%) = 청산 뒤 익절 다시 걸기(C2). 원본도 settle 밖 -- 예외는 부르는 쪽 settle 이 받는다.
    pub async fn exit_then_rearm<I: WatchIo>(&mut self, io: &mut I, plan: &Obj, state: &mut Obj) -> Result<(), String> {
        io.run_exit(&Value::Object(plan.clone()), state).await?;
        self.partial_exit_rearm_tp(io, plan.clone(), state).await
    }

    /// 지정가 대기 접수·수정 때 계획을 파일에(W4) -- 못 남기면 복구만 못 한다(접수는 막지 않는다).
    pub fn limit_save<I: WatchIo>(&mut self, io: &I, plan: &Obj) -> Result<(), String> {
        self.limit_save_at(io.now(), plan)
    }
    /// 접수 문(수정 API)이 시계만 넘겨 부른다.
    pub fn limit_save_at(&mut self, now: f64, plan: &Obj) -> Result<(), String> {
        let path = limit_path(&self.dir, &req_str(plan, "symbol")?, int_or1(plan.get("wid")));
        let body = json!({"saved_at": now, "plan": Value::Object(plan.clone())});
        let tmp = path.with_extension("tmp");
        if let Err(e) = std::fs::write(&tmp, serde_json::to_string(&body).unwrap_or_default()).and_then(|_| std::fs::rename(&tmp, &path)) {
            self.log(format!("limit wait save: OSError: {e}"));
        }
        Ok(())
    }

    /// 재시작 직후(주문 정리 뒤): 저장된 지정가 대기가 있으면 그 사이 체결된 포지션에 익절·손절을 건다(W5).
    /// 계좌를 못 읽거나 남은 주문을 못 지웠으면 파일을 남겨 다음 기동에 다시.
    pub async fn limit_recover<I: WatchIo>(&mut self, io: &mut I, path: &Path, orphan_errors: &Value) -> Result<Option<Obj>, String> {
        let Ok(raw) = std::fs::read_to_string(path) else { return Ok(None) };
        let Ok(rec) = serde_json::from_str::<Value>(&raw) else { return Ok(None) };
        let rec = rec.as_object().cloned().ok_or_else(|| format!("AttributeError: '{}' object has no attribute 'get'", type_name(&rec)))?;
        let plan = obj_or_empty(rec.get("plan"));
        let (sym, pside) = (val(&plan, "symbol"), val(&plan, "positionSide"));
        let mut out = Obj::new();
        for (k, v) in [("symbol", sym.clone()), ("side", pside.clone()), ("armed", json!(false)), ("wid", json!(int_or1(plan.get("wid"))))] {
            out.insert(k.into(), v);
        }
        let with = |o: &Obj, reason: Value| { let mut m = o.clone(); m.insert("reason".into(), reason); Some(m) };
        if io.now() - float_or0(&rec, "saved_at") > LIMIT_REC_MAX_AGE {
            let _ = std::fs::remove_file(path);
            return Ok(with(&out, json!("저장된 지정가 대기가 너무 오래돼 복구하지 않았습니다 — 포지션의 익절·손절을 직접 확인하세요")));
        }
        let step: Result<Option<Obj>, String> = async {
            let acct = io.produce_account().await?;
            let acct = acct.as_object().cloned().ok_or("AttributeError: 'NoneType' object has no attribute 'get'")?;
            if !t(&acct, "ok") {
                return Ok(with(&out, json!("계좌 조회 실패 — 포지션의 익절·손절을 직접 확인하세요(다음 기동에 다시 시도)")));
            }
            let has = acct.get("positions").and_then(Value::as_array).is_some_and(|a| a.iter().any(|p| py_eq(p.get("symbol"), Some(&sym))
                && py_eq(p.get("side"), Some(&pside)) && (if truthy(p.get("qty")) { py_float(&p["qty"]).unwrap_or(0.0) } else { 0.0 }).abs() > 0.0));
            if has && !self.load().contains_key(&bracket_key(&py_str(&sym), &py_str(&pside))) {
                let mut st = Obj::new();
                self.arm_bracket(io, &plan, &mut st).await?;
                let br = obj_or_empty(st.get("bracket"));
                let ok = t(&br, "placed"); // arm_bracket 은 예외를 삼킨다 -- 실제로 걸렸는지 본다(검증 #3)
                out.insert("armed".into(), json!(ok));
                out.insert("bracket".into(), val(&st, "bracket"));
                if !ok {
                    out.insert("reason".into(), json!(format!("익절·손절을 다 걸지 못했습니다({}) — 거래소에서 직접 확인하세요", py_str(&py_or(vec![val(&br, "reason"), json!("일부 실패")])))));
                }
            }
            Ok(None)
        }.await;
        match step {
            Err(e) => return Ok(with(&out, json!(format!("복구 실패 {e} — 포지션의 익절·손절을 직접 확인하세요")))), // 기동은 막지 않는다 · 파일은 남긴다
            Ok(Some(early)) => return Ok(Some(early)),
            Ok(None) => {}
        }
        if truthy(Some(orphan_errors)) {
            // 남은 주문을 못 지웠다 = 살아 있는 GTX 가 나중에 체결될 수 있다 -- 파일을 남겨 다음 기동에 다시(검증 #1)
            let reason = py_or(vec![val(&out, "reason"), json!("대기 주문 취소에 실패했습니다 — 거래소에서 직접 확인하세요(다음 기동에 다시 시도)")]);
            return Ok(with(&out, reason));
        }
        if !t(&out, "reason") {
            let _ = std::fs::remove_file(path);
        }
        Ok(Some(out))
    }

    /// 재시작 직후 한 번(R1·R2): 죽은 대기 작업이 남긴 GTX(dbwt·dbpg)를 지우고, 저장된 지정가 대기 파일로 복구한다. 기록은 대기 칸에 -- 화면이 «재시작으로 취소»를 말한다.
    /// ponytail(원본): 복구(이어서 기다리기) 대신 정리 -- 배포·재시작 때 대기 주문은 사라진다(이미 든 몫·손절은 그대로).
    pub async fn wait_cleanup<I: WatchIo>(&mut self, io: &mut I) -> Result<(), String> {
        let mut syms: Vec<String> = self.exec_symbols().into_iter().map(|x| x.1).collect();
        syms.sort();
        syms.dedup();
        let r = match io.cancel_wait_orphans(&syms).await {
            Ok(v) => v,
            Err(e) => json!({"cancelled": [], "errors": [e]}), // 기동은 막지 않는다, 대신 남긴다
        };
        let cancelled: Vec<Value> = r.get("cancelled").and_then(Value::as_array).cloned().ok_or("KeyError: 'cancelled'")?;
        let errors = r.get("errors").cloned().ok_or("KeyError: 'errors'")?;
        // 2026-11-10 코인별: 지운 주문은 그 코인 칸에 · 어느 코인인지 모르는 정리 실패는 모든 기록 칸에(없으면 eth)
        let mut hit: BTreeSet<&'static str> = cancelled.iter().map(|c| self.asset_of(c.get("symbol"))).collect();
        if hit.is_empty() && truthy(Some(&errors)) {
            hit.insert("eth");
        }
        let now = io.now();
        for a in hit {
            let mine: Vec<Value> = cancelled.iter().filter(|c| self.asset_of(c.get("symbol")) == a).cloned().collect();
            let ws = self.wait_slot(a, 1);
            ws.clear();
            for (k, v) in [("phase", json!("restart_cancelled")), ("mode", json!("wait")), ("cancelled", Value::Array(mine)), ("errors", errors.clone()),
                           ("done_at", json!(iso_utc(now)))] {
                ws.insert(k.into(), v);
            }
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir).map(|d| d.filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("manual_limit_wait") && n.ends_with(".json"))).collect()).unwrap_or_default();
        files.sort();
        let mut recs = vec![];
        for path in files {
            // 2026-10-31 지정가 대기 중 재시작: 그 사이 체결된 몫에 익절·손절을 건다(검증 D2)
            let Some(rec) = self.limit_recover(io, &path, &val(&r.as_object().cloned().unwrap_or_default(), "errors")).await? else { continue };
            recs.push(Value::Object(rec.clone()));
            let asset = self.asset_of(Some(&json!(py_or(vec![val(&rec, "symbol"), json!("")]))));
            let now = io.now();
            let ws = self.wait_slot(asset, int_or1(rec.get("wid")));
            if ws.get("phase").and_then(Value::as_str) != Some("restart_cancelled") {
                ws.clear();
                for (k, v) in [("phase", json!("restart_cancelled")), ("mode", json!("limit")), ("cancelled", json!([])), ("errors", json!([])), ("done_at", json!(iso_utc(now)))] {
                    ws.insert(k.into(), v);
                }
            }
            ws.insert("recovered".into(), Value::Object(rec));
        }
        self.log(format!("wait cleanup: {} limit_recover: {}", py_str(&r), py_str(&Value::Array(recs))));
        Ok(())
    }

    /// 감시 루프(B8): 주기(기본 2초 · 체결 알림이면 일찍 깸)마다 한 틱. 한 틱의 실패는 삼키고 다음 틱에(감시는 죽지 않는다). nap 이 false = 멈춤.
    pub async fn watch_brackets<I: WatchIo>(&mut self, io: &mut I, period: f64) {
        while io.nap(period).await {
            if let Err(e) = self.bracket_tick(io).await {
                self.log(format!("bracket watch: {e}"));
            }
        }
    }
}

fn as_dict(v: Value) -> Result<Obj, String> {
    match v {
        Value::Object(m) => Ok(m),
        x => Err(format!("AttributeError: '{}' object has no attribute 'get'", type_name(&x))),
    }
}

fn bar_repr(b: Option<(f64, f64)>) -> String {
    b.map_or("None".into(), |(t, c)| format!("({}, {})", t as i64, pyfmt::repr(c)))
}
/// ponytail: 1초 가격이 정수(JSON int)로 오면 파이썬은 «2480» 으로 찍는다 -- 여기는 늘 실수 표기(수집기는 실수만 준다).
fn touch_repr(t: Option<(f64, f64)>) -> String {
    t.map_or("None".into(), |(a, b)| format!("({}, {})", pyfmt::repr(a), pyfmt::repr(b)))
}

/// 조각 목록을 무장에 쓴다 + 옛 단일 칸(tp_level·tp_price = 가장 가까운 살아 있는 조각 · tp_resting = 전부 걸렸나)도 맞춘다.
/// 체결된 조각은 버리고, 살아 있는 조각이 하나면 tp_legs 를 지워 옛 단일 경로로 돌아간다(B29).
pub fn set_tp_legs(cur: &mut Obj, legs: &[Value]) -> Result<(), String> {
    let live: Vec<Obj> = legs.iter().filter(|x| !truthy(x.get("filled")))
        .map(|x| ["level", "price", "w", "resting", "merged", "cid", "order_id", "qty"].iter().map(|k| (k.to_string(), x.get(*k).cloned().unwrap_or(Value::Null))).collect())
        .collect();
    if live.is_empty() {
        // 2026-11-12 검토: 조각이 다 빠졌는데 tp_resting 이 True 로 남아 익절 주문도 서버 감시도 없었다 -- 포지션이 남았으면 마지막 익절선을 서버가 감시
        cur.remove("tp_legs");
        cur.insert("tp_resting".into(), json!(false));
        return Ok(());
    }
    let sgn = if cur.get("side").and_then(Value::as_str) != Some("SHORT") { 1.0 } else { -1.0 };
    let mut near = &live[0];
    for x in &live[1..] {
        if py_float(&x["level"])? * sgn < py_float(&near["level"])? * sgn {
            near = x;
        }
    }
    let all = live.iter().all(|x| truthy(x.get("resting")) || truthy(x.get("merged")));
    let (lv, px) = (near["level"].clone(), near["price"].clone());
    cur.insert("tp_level".into(), lv);
    cur.insert("tp_price".into(), px);
    cur.insert("tp_resting".into(), json!(all));
    if live.len() > 1 {
        cur.insert("tp_legs".into(), Value::Array(live.into_iter().map(Value::Object).collect()));
    } else {
        cur.remove("tp_legs");
    }
    Ok(())
}

/// 손절 조각을 무장에 쓴다 + 옛 단일 칸(대표 조각: 롱 = 가장 높은 · 숏 = 가장 낮은). 하나뿐이면 sl_legs 를 지운다. 비면 그대로(감시 clear 가 정리).
pub fn set_sl_legs(cur: &mut Obj, legs: &[Value]) {
    let live: Vec<Obj> = legs.iter().map(|x| ["level", "price", "up", "w", "fired"].iter().map(|k| (k.to_string(), x.get(*k).cloned().unwrap_or(Value::Null))).collect()).collect();
    if live.is_empty() {
        cur.remove("sl_legs");
        return;
    }
    let sgn = if cur.get("side").and_then(Value::as_str) != Some("SHORT") { 1.0 } else { -1.0 };
    let lv = |x: &Obj| x.get("level").and_then(Value::as_f64).unwrap_or(0.0) * sgn;
    let mut rep = &live[0];
    for x in &live[1..] {
        if lv(x) > lv(rep) {
            rep = x;
        }
    }
    let level = rep.get("level").and_then(Value::as_f64).unwrap_or(0.0);
    let name = format!("선 손절 {}", pyfmt::fixed(level, 2, false)) + &if live.len() > 1 { format!(" 외 {}", live.len() - 1) } else { String::new() };
    cur.insert("sl_level".into(), val(rep, "level"));
    cur.insert("sl_price".into(), val(rep, "price"));
    cur.insert("sl_up".into(), val(rep, "up"));
    cur.insert("sl_name".into(), json!(name));
    if live.len() > 1 {
        cur.insert("sl_legs".into(), Value::Array(live.into_iter().map(Value::Object).collect()));
    } else {
        cur.remove("sl_legs");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn iso_round_trip_matches_python_shape() {
        assert_eq!(iso_utc(1_800_000_000.0), "2027-01-15T08:00:00+00:00");
        assert_eq!(iso_utc(1_800_000_000.5), "2027-01-15T08:00:00.500000+00:00");
        assert_eq!(parse_iso("2027-01-15T08:00:01+00:00"), Some(1_800_000_001.0));
        assert_eq!(parse_iso("2027-01-15T17:00:01+09:00"), Some(1_800_000_001.0));
        assert_eq!(parse_iso("2027-01-15T08:00:00.250000Z"), Some(1_800_000_000.25));
        assert_eq!(parse_iso("None"), None);
        assert_eq!(parse_iso("nope"), None);
        assert!(stop_touched(2500.0, "LONG", Some(&[(1.0, 2499.0), (3.0, 2500.0)]), 2.0));
        assert!(!stop_touched(2500.0, "LONG", Some(&[(1.0, 2600.0), (3.0, 2499.0)]), 2.0));
        assert!(stop_touched(2500.0, "SHORT", Some(&[(3.0, 2500.0)]), 2.0) && !stop_touched(2500.0, "SHORT", None, 0.0));
    }
}
