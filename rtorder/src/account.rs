//! 계좌 읽기와 사용자 스트림(2026-11-21 넷째 조각) -- 원본:
//! `scripts/live_binance_account_20260910.py` 의 `fetch_account`·`pick_balance`·`round_trips`(왕복 접기 -- 포지션 `entry_at` 이 여기서 온다) ·
//! `dashboard/orders.py` 의 `produce_account`(exec_symbol·last_good) · `swr_cached` 규칙(server.py) · `user_event_kind` · `_user_stream_run` · `account_refresher`.
//! **어디에도 연결하지 않는다** -- 거래소는 `AccountRest`(서명 GET) · 스트림은 `StreamIo` 뒤. 시험은 가짜뿐(tests/desk_fixture.rs).
//! 화면 몫(왕복 원장 파일 기록·income·day_start·추세 칸)은 옮기지 않았다 -- 응답 서버(R6) 몫. 주문 경로는 이 칸들을 안 읽는다.
//!
//! «받아 둔 값» 자리(사용자 결정): `StreamAccount` = 마지막 REST 스냅숏 + 사용자 스트림 사건으로 고친 포지션·미체결.
//! 스트림이 살아 있고 · 그 연결에서 · `STREAM_TRUST_SEC` 안에 REST 와 맞춰 봤고 · 마지막 대조가 어긋나지 않았을 때만 쓴다. 아니면 REST(`wire::Wired` 의 청산 조립 계좌).

use crate::sign::{err_value, py_float, py_str, truthy, P};
use crate::watch::iso_utc;
use crate::pyfmt::round_n;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub type Obj = Map<String, Value>;

pub const QTY_EPS: f64 = 1e-9;
pub const ADD_EVENT_GAP_MS: i64 = 180_000; // 이 안의 증가 체결은 같은 «추가» 이벤트
pub const DEFAULT_TRADE_LIMIT: i64 = 1000; // Binance max; 잘리면 trades_truncated
/// 2026-10-02 30 -> 10(사용자 지시). 조회 1회 ≈ 무게 50 → 분당 ~300 = IP 한도 2,400 의 ~12% (D10: 주석의 «30초»는 옛 값)
pub const ACCOUNT_CACHE_SEC: f64 = 10.0;
/// `?fresh=1` 연타: 1초 안에 새로 받은 값이 있으면 그것(A3 · 2026-10-06)
pub const FRESH_REUSE_SEC: f64 = 1.0;
/// 화면용 마지막 정상 계좌(A2 · 2026-11-02) -- 감시·주문 경로는 못 쓴다
pub const LAST_GOOD_SEC: f64 = 1800.0;
pub const USER_STREAM_WS: &str = "wss://fstream.binance.com/private/ws/"; // 2026-11-09 공식 문서: /private 를 빼면 계좌 알림이 안 온다(A4)
pub const USER_STREAM_KEEPALIVE_S: f64 = 1800.0; // listenKey 60분 유효 -- 30분마다 연장(A5)
pub const ACCOUNT_PUSH_GAP_S: f64 = 3.0; // 체결 알림 뒤 계좌 재조회 사이 최소 간격(IP 한도 · A11)
pub const PUSH_COALESCE_S: f64 = 0.3; // 한 체결의 알림 두세 개를 조회 한 번으로(A11)
pub const WS_RECV_TIMEOUT_S: f64 = 60.0; // 수신 60초 시간초과는 계속(A6)
pub const WS_HEARTBEAT_S: f64 = 120.0;
pub const RECONNECT_MIN_S: f64 = 5.0; // 5초 → ×2 → 300초 · 5분 넘게 붙어 있었으면 5초(A7)
pub const RECONNECT_MAX_S: f64 = 300.0;
pub const RECONNECT_RESET_AFTER_S: f64 = 300.0;
/// 스트림으로 고친 계좌를 믿는 최대 시간(마지막 REST 대조 뒤). 근거: 체결 알림마다 계좌를 REST 로 다시 읽으므로(account_refresher, 0.3+3초)
/// 체결이 있었다면 대조는 몇 초 안에 새로 된다 -- 60초 넘게 대조가 없다는 건 «체결 없음» 이거나 재조회가 막힌 것. 감시 틱의 캐시(10초)보다 넉넉하고
/// listenKey 수명(60분)보다 훨씬 짧다. ponytail: 고정 60초 -- 그림자 대조(M2)로 어긋남 0 이 확인되면 늘린다.
pub const STREAM_TRUST_SEC: f64 = 60.0;

fn f(x: f64) -> Value {
    json!(x)
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
/// `float(o[k])` (없으면 KeyError).
fn req_f(o: &Value, k: &str) -> Result<f64, String> {
    py_float(o.get(k).ok_or_else(|| format!("KeyError: '{k}'"))?)
}
/// `float(o.get(k) or 0.0)`.
fn or0(o: &Value, k: &str) -> Result<f64, String> {
    if truthy(o.get(k)) { py_float(&o[k]) } else { Ok(0.0) }
}
fn py_int(v: &Value) -> Result<i64, String> {
    match v {
        Value::Number(n) => n.as_i64().map_or_else(|| Ok(n.as_f64().unwrap_or(0.0).trunc() as i64), Ok),
        Value::String(s) => s.trim().parse().map_err(|_| format!("ValueError: invalid literal for int() with base 10: '{s}'")),
        Value::Bool(b) => Ok(*b as i64),
        x => Err(format!("TypeError: int() argument must be a string, a bytes-like object or a real number, not '{}'", type_name(x))),
    }
}
fn req_i(o: &Value, k: &str) -> Result<i64, String> {
    py_int(o.get(k).ok_or_else(|| format!("KeyError: '{k}'"))?)
}
fn req_s(o: &Value, k: &str) -> Result<String, String> {
    o.get(k).map(py_str).ok_or_else(|| format!("KeyError: '{k}'"))
}
fn iso_ms(ms: &Value) -> Result<Value, String> {
    if !truthy(Some(ms)) {
        return Ok(Value::Null);
    }
    Ok(json!(iso_utc(py_int(ms)? as f64 / 1000.0)))
}

// ── 잔고 ─────────────────────────────────────────────────────────────────────
/// (쓸 잔고, 잔액이 있는 자산 목록). 🔴최상위 total* 합계는 **단일자산 담보 모드에서 USDT 전용**이다(2026-09-19 USDC 1,472 인데 잔고 0).
pub fn pick_balance(account: &Value, quote: &str) -> Result<(Value, Value), String> {
    let none = vec![];
    let assets = account.get("assets").and_then(Value::as_array).unwrap_or(&none);
    let mut listed = vec![];
    for a in assets {
        if or0(a, "walletBalance")? != 0.0 || or0(a, "availableBalance")? != 0.0 {
            listed.push(json!({"asset": a.get("asset").cloned().unwrap_or(Value::Null), "wallet": or0(a, "walletBalance")?, "available": or0(a, "availableBalance")?,
                               "margin": or0(a, "marginBalance")?, "unrealized": or0(a, "unrealizedProfit")?}));
        }
    }
    let top = || -> Result<Value, String> {
        Ok(json!({"wallet": req_f(account, "totalWalletBalance")?, "margin": req_f(account, "totalMarginBalance")?,
                  "initial_margin": req_f(account, "totalInitialMargin")?, "maint_margin": req_f(account, "totalMaintMargin")?,
                  "available": req_f(account, "availableBalance")?, "unrealized": req_f(account, "totalUnrealizedProfit")?}))
    };
    if truthy(account.get("multiAssetsMargin")) {
        return Ok((top()?, Value::Array(listed)));
    }
    match assets.iter().find(|a| a.get("asset").and_then(Value::as_str) == Some(quote)) {
        None => Ok((top()?, Value::Array(listed))), // 그 자산 항목 자체가 없다 -- 최상위(=USDT)로 떨어뜨리되 목록은 그대로
        Some(p) => Ok((json!({"wallet": or0(p, "walletBalance")?, "margin": or0(p, "marginBalance")?, "initial_margin": or0(p, "initialMargin")?,
                              "maint_margin": or0(p, "maintMargin")?, "available": or0(p, "availableBalance")?, "unrealized": or0(p, "unrealizedProfit")?}),
                       Value::Array(listed))),
    }
}

// ── 왕복 접기(포지션 entry_at 의 출처) ──────────────────────────────────────────
#[derive(Debug, Clone)]
struct Fill {
    symbol: String,
    side: String,
    qty: f64,
    price: f64,
    realized: f64,
    commission: f64,
    time: i64,
    maker: bool,
}

fn fill_of(v: &Value) -> Result<Fill, String> {
    Ok(Fill { symbol: req_s(v, "symbol")?, side: req_s(v, "side")?, qty: req_f(v, "qty")?, price: req_f(v, "price")?, realized: req_f(v, "realizedPnl")?,
              commission: req_f(v, "commission")?, time: req_i(v, "time")?, maker: truthy(v.get("maker")) })
}

/// 닫으면서 반대로 여는 체결을 둘로 가른다(단방향 모드 뒤집기). realizedPnl 은 닫는 쪽 몫 전부 · 수수료는 비례.
fn split_flips(fills: &[Fill]) -> Vec<Fill> {
    let (mut net, mut out) = (0.0f64, vec![]);
    for fl in fills {
        let dir = if fl.side == "BUY" { 1.0 } else { -1.0 };
        let closing = if net * dir < 0.0 { fl.qty.min(net.abs()) } else { 0.0 };
        // 양 끝을 QTY_EPS로 막는다: 0.3-0.1=0.19999999999999998 잔여분이 유령 왕복을 만들었다(2026-09-10 실계좌 5건)
        if QTY_EPS < closing && closing < fl.qty - QTY_EPS {
            for (part, realized) in [(closing, fl.realized), (fl.qty - closing, 0.0)] {
                out.push(Fill { qty: part, realized, commission: fl.commission * part / fl.qty, ..fl.clone() });
            }
        } else {
            out.push(fl.clone());
        }
        net += fl.qty * dir;
        if net.abs() < QTY_EPS {
            net = 0.0;
        }
    }
    out
}

fn resync_index(fills: &[Fill], position_side: &str, end_qty: Option<f64>) -> Option<usize> {
    let end = end_qty?;
    if position_side != "LONG" && position_side != "SHORT" {
        return None;
    }
    let sgn = if position_side == "LONG" { 1.0 } else { -1.0 };
    let mut cum = vec![0.0f64];
    for fl in fills {
        cum.push(cum[cum.len() - 1] + fl.qty * (if fl.side == "BUY" { 1.0 } else { -1.0 }) * sgn);
    }
    let q0 = end - cum[cum.len() - 1];
    if q0 <= QTY_EPS {
        return None;
    }
    (1..cum.len()).find(|&i| (cum[i] + q0).abs() <= 1e-6 * 1f64.max(q0)).map(|i| i - 1)
}

struct Trip {
    symbol: String,
    side: String,
    entry_time: i64,
    max_qty: f64,
    qty_in: f64,
    qty_out: f64,
    in_: f64,
    out_: f64,
    realized: f64,
    commission: f64,
    fills: i64,
    adds: i64,
    peak: f64,
    last_inc: Option<i64>,
    legs: Vec<Value>,
    exit_time: Option<i64>,
    closed: bool,
}

fn finish(t: Trip) -> Obj {
    let entry_price = if t.qty_in != 0.0 { Some(t.in_ / t.qty_in) } else { None };
    let exit_price = if t.qty_out != 0.0 { Some(t.out_ / t.qty_out) } else { None };
    let sign = if t.side == "LONG" { 1.0 } else { -1.0 };
    let check = match (t.closed, entry_price, exit_price) {
        (true, Some(e), Some(x)) if e != 0.0 && x != 0.0 && t.qty_out != 0.0 => {
            let implied = (x - e) * t.qty_out * sign;
            let notional = e * t.qty_out;
            if notional != 0.0 { f(round_n((t.realized - implied) / notional * 1e4, 4)) } else { Value::Null }
        }
        _ => Value::Null,
    };
    let v = json!({"symbol": t.symbol, "side": t.side, "entry_time": t.entry_time, "max_qty": t.max_qty, "qty_in": t.qty_in, "qty_out": t.qty_out,
                   "realized_pnl": t.realized, "commission": t.commission, "fills": t.fills, "adds": t.adds, "peak_notional": t.peak, "legs": t.legs,
                   "exit_time": t.exit_time, "exit_price": exit_price, "closed": t.closed, "entry_price": entry_price, "net_pnl": t.realized - t.commission,
                   "price_basis": "leg_vwap", "pnl_check_bp": check});
    v.as_object().unwrap().clone()
}

fn fold(fills: &[Fill], position_side: &str, end_qty: Option<f64>) -> Vec<Obj> {
    let mut trips: Vec<Obj> = vec![];
    let mut cur: Option<Trip> = None;
    let (mut net, mut truncated) = (0.0f64, false);
    for fl in split_flips(fills) {
        let signed = fl.qty * if fl.side == "BUY" { 1.0 } else { -1.0 };
        let c = cur.get_or_insert_with(|| Trip {
            symbol: fl.symbol.clone(),
            // 🔴측면은 positionSide 가 정한다(2026-09-22 실계좌: 첫 체결 방향으로 정하면 창이 자른 헤지 LONG 조각을 SHORT 로 찍었다)
            side: if position_side == "LONG" || position_side == "SHORT" { position_side.into() } else if signed > 0.0 { "LONG".into() } else { "SHORT".into() },
            entry_time: fl.time, max_qty: 0.0, qty_in: 0.0, qty_out: 0.0, in_: 0.0, out_: 0.0, realized: 0.0, commission: 0.0, fills: 0, adds: 0,
            peak: 0.0, last_inc: None, legs: vec![], exit_time: None, closed: false,
        });
        let opening = net == 0.0 || (signed > 0.0) == (net > 0.0);
        if opening {
            c.qty_in += fl.qty;
            c.in_ += fl.price * fl.qty;
            if c.last_inc.is_some_and(|l| fl.time - l > ADD_EVENT_GAP_MS) {
                c.adds += 1;
            }
            c.last_inc = Some(fl.time);
        } else {
            c.qty_out += fl.qty;
            c.out_ += fl.price * fl.qty;
        }
        c.legs.push(json!([fl.time, fl.price, fl.qty, opening as i64, fl.maker as i64, round_n(fl.realized, 6), round_n(fl.commission, 6)]));
        net += signed;
        // 🔴헤지 스트림은 부호를 못 벗어난다 -- 벗어났다면 userTrades 7일 창이 진입 체결을 잘랐다(2026-09-22 실측 8건 중 6건 불일치)
        if (position_side == "LONG" && net < -QTY_EPS) || (position_side == "SHORT" && net > QTY_EPS) {
            truncated = true;
        }
        c.max_qty = c.max_qty.max(net.abs());
        c.peak = c.peak.max(net.abs() * fl.price);
        c.realized += fl.realized;
        c.commission += fl.commission;
        c.fills += 1;
        if net.abs() < QTY_EPS {
            c.exit_time = Some(fl.time);
            c.closed = true;
            trips.push(finish(cur.take().unwrap()));
            net = 0.0;
        }
    }
    if let Some(c) = cur {
        trips.push(finish(c));
    }
    if truncated {
        if let Some(cut) = resync_index(fills, position_side, end_qty).filter(|c| c + 1 < fills.len()) {
            let tc = fills[cut].time;
            for t in trips.iter_mut() {
                if t["entry_time"].as_i64().unwrap() <= tc {
                    t.insert("truncated_open".into(), json!(true));
                }
            }
            // 🔴2026-11-02 열린 조각은 버린다 -- 창 앞 포지션의 파편(«롱 순환 2개» · entry_at 이 파편의 이른 시각이 됐다)
            let mut kept: Vec<Obj> = trips.into_iter().filter(|t| t["entry_time"].as_i64().unwrap() <= tc && t["closed"] == json!(true)).collect();
            kept.extend(fold(&fills[cut + 1..], position_side, None));
            return kept;
        }
        for t in trips.iter_mut() {
            t.insert("truncated_open".into(), json!(true));
        }
    }
    trips
}

/// userTrades 체결 → 왕복(심볼·positionSide 마다 접는다 -- 헤지 모드).
pub fn round_trips(fills: &[Value], open_qty: &HashMap<(String, String), f64>) -> Result<Vec<Obj>, String> {
    let mut keyed: Vec<((i64, i64), Fill, String)> = vec![];
    for v in fills {
        keyed.push(((req_i(v, "time")?, req_i(v, "id")?), fill_of(v)?, v.get("positionSide").map_or("BOTH".into(), py_str)));
    }
    keyed.sort_by_key(|x| x.0);
    let mut groups: Vec<((String, String), Vec<Fill>)> = vec![];
    for (_, fl, ps) in keyed {
        let k = (fl.symbol.clone(), ps);
        match groups.iter_mut().find(|g| g.0 == k) {
            Some(g) => g.1.push(fl),
            None => groups.push((k, vec![fl])),
        }
    }
    let mut trips = vec![];
    for ((sym, ps), g) in groups {
        trips.extend(fold(&g, &ps, open_qty.get(&(sym, ps.clone())).copied()));
    }
    trips.sort_by_key(|t| t["entry_time"].as_i64().unwrap());
    Ok(trips)
}

// ── 계좌 조회 ──────────────────────────────────────────────────────────────────
/// 서명 GET 하나(`_get`). 예외 대신 `{"__error__": …}` · 밴 중이면 보내지 않고 오류(원본 그대로).
#[allow(async_fn_in_trait)]
pub trait AccountRest {
    fn has_keys(&self) -> bool;
    fn now(&self) -> f64; // time.time()
    async fn clock_offset(&mut self) -> i64;
    async fn get(&mut self, path: &str, params: &[(&str, P)], offset: i64) -> Value;
    /// 여러 GET 을 한꺼번에(원본 asyncio.gather). 기본 = 차례로(가짜·정답지용 -- 요청 열이 정해진다).
    async fn get_many(&mut self, reqs: &[(String, Vec<(String, P)>)], offset: i64) -> Vec<Value> {
        let mut out = vec![];
        for (path, params) in reqs {
            let p: Vec<(&str, P)> = params.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
            out.push(self.get(path, &p, offset).await);
        }
        out
    }
}

impl AccountRest for crate::http::Binance {
    fn has_keys(&self) -> bool {
        crate::http::Binance::has_keys(self)
    }
    fn now(&self) -> f64 {
        self.time()
    }
    async fn clock_offset(&mut self) -> i64 {
        crate::http::Binance::clock_offset(self).await
    }
    async fn get(&mut self, path: &str, params: &[(&str, P)], offset: i64) -> Value {
        let left = self.ban_left();
        if left > 0.0 {
            // aiohttp 라 requests 전역 가드 밖 -- 밴 중 10초 계좌 폴링이 밴을 늘리지 않게(10-16 감사)
            return err_value(format!("binance IP ban: {left:.0}s left -- request not sent"));
        }
        self.offset = offset;
        self.signed("GET", path, params).await
    }
    async fn get_many(&mut self, reqs: &[(String, Vec<(String, P)>)], offset: i64) -> Vec<Value> {
        let left = self.ban_left();
        if left > 0.0 {
            return reqs.iter().map(|_| err_value(format!("binance IP ban: {left:.0}s left -- request not sent"))).collect();
        }
        self.offset = offset;
        self.signed_get_many(reqs).await
    }
}

fn err_of(v: &Value) -> Option<String> {
    v.as_object().and_then(|o| o.get("__error__")).map(py_str)
}

/// 원본 `fetch_account`(income 없이 -- 화면 몫). 요청 순서: 시계 → /fapi/v2/account → /fapi/v2/positionRisk → 심볼마다 userTrades → 심볼마다 openOrders.
/// Err = 파이썬에서 예외로 새던 것(KeyError 등) -- 조립이 502 로 받는다.
pub async fn fetch_account<R: AccountRest>(rest: &mut R, symbols: &[String], quote: &str, trade_limit: i64) -> Result<Value, String> {
    if !rest.has_keys() {
        return Ok(json!({"ok": false, "error": "BINANCE_API_KEY/BINANCE_SECRET_KEY가 .env에 없습니다."}));
    }
    let off = rest.clock_offset().await;
    let balance = rest.get("/fapi/v2/account", &[], off).await;
    if let Some(e) = err_of(&balance) {
        // 힌트는 실제로 그 오류일 때만(예전엔 시계 드리프트 -1021 까지 권한 문제로 오인)
        let hint = if e.contains("-2015") || e.contains("permissions") {
            Some("API 키에 Futures 읽기 권한(Enable Futures)이 켜져 있는지, 그리고 IP 접근 제한에 이 서버 주소가 들어 있는지 확인하세요.")
        } else if e.contains("-1021") || e.contains("Timestamp") {
            Some("서버 시계가 바이낸스와 어긋났습니다(NTP 동기화 확인).")
        } else {
            None
        };
        let mut o = json!({"ok": false, "error": e});
        if let Some(h) = hint {
            o["hint"] = json!(h);
        }
        return Ok(o);
    }
    let risk = rest.get("/fapi/v2/positionRisk", &[], off).await;
    let Some(risk) = risk.as_array().cloned() else {
        // 🔴2026-09-26 사고: 포지션 조회만 실패(418)했는데 ok=True + positions=[] → 감시가 «사라짐»으로 읽고 익절·비상 스탑을 스스로 지웠다(P2)
        return Ok(json!({"ok": false, "error": format!("positionRisk: {}", risk.get("__error__").map(py_str).unwrap_or_else(|| "None".into()))}));
    };
    let mut positions: Vec<Value> = vec![];
    for p in &risk {
        let amt = req_f(p, "positionAmt")?;
        if amt == 0.0 {
            continue;
        }
        let ps = p.get("positionSide").and_then(Value::as_str);
        positions.push(json!({"symbol": p.get("symbol").cloned().ok_or("KeyError: 'symbol'")?,
            "side": match ps { Some(s @ ("LONG" | "SHORT")) => s.to_string(), _ => if amt > 0.0 { "LONG".into() } else { "SHORT".into() } },
            "qty": amt.abs(), "entry_price": req_f(p, "entryPrice")?, "mark_price": req_f(p, "markPrice")?, "liquidation_price": req_f(p, "liquidationPrice")?,
            "leverage": req_f(p, "leverage")?, "notional": req_f(p, "notional")?.abs(), "unrealized_pnl": req_f(p, "unRealizedProfit")?,
            "updated_at": iso_ms(p.get("updateTime").unwrap_or(&Value::Null))?}));
    }
    // 심볼별 설정 레버리지. 포지션이 없어도 읽어야 한다(위 목록은 수량 0 을 거른다 -- 2026-09-13 «증거금 0»)
    let mut lev_by = Obj::new();
    for p in &risk {
        if truthy(p.get("leverage")) {
            lev_by.insert(req_s(p, "symbol")?, f(py_float(&p["leverage"])?));
        }
    }
    let mut open_qty = HashMap::new();
    for p in &risk {
        if let Some(ps @ ("LONG" | "SHORT")) = p.get("positionSide").and_then(Value::as_str) {
            open_qty.insert((req_s(p, "symbol")?, ps.to_string()), req_f(p, "positionAmt")?.abs());
        }
    }
    // 2026-11-05 원본은 심볼별 조회를 한꺼번에(gather) -- 요청 수·무게는 같다. 2026-11-21 Rust 도 `get_many`(Binance = 동시 · 가짜 = 차례로)
    let tr: Vec<(String, Vec<(String, P)>)> = symbols.iter().map(|s| ("/fapi/v1/userTrades".to_string(), vec![("symbol".into(), P::s(s)), ("limit".into(), P::I(trade_limit))])).collect();
    let all_fills = rest.get_many(&tr, off).await;
    let oo: Vec<(String, Vec<(String, P)>)> = symbols.iter().map(|s| ("/fapi/v1/openOrders".to_string(), vec![("symbol".into(), P::s(s))])).collect();
    let all_rows = rest.get_many(&oo, off).await;
    let (mut trips, mut truncated, mut failed) = (vec![], vec![], vec![]);
    for (s, fills) in symbols.iter().zip(&all_fills) {
        let Some(fl) = fills.as_array() else {
            failed.push(json!(s)); // 2026-11-02 «거래 없음»이 아니라 «모름»
            continue;
        };
        if fl.is_empty() {
            continue;
        }
        let folded = round_trips(fl, &open_qty)?;
        if fl.len() as i64 >= trade_limit || folded.iter().any(|t| truthy(t.get("truncated_open"))) {
            truncated.push(json!(s));
        }
        trips.extend(folded);
    }
    for t in trips.iter_mut() {
        let (e, x) = (iso_ms(&t["entry_time"])?, iso_ms(&t["exit_time"])?);
        t.insert("entry_at".into(), e);
        t.insert("exit_at".into(), x);
    }
    trips.sort_by(|a, b| b["entry_time"].as_i64().cmp(&a["entry_time"].as_i64()));
    // 🔴2026-11-02 내림차순이라 덮어쓰면 오래된 쪽이 이겼다 -- 잘리지 않은 열린 왕복을 먼저, 없으면 잘린 것(setdefault)
    let mut open_entry: HashMap<(String, String), Value> = HashMap::new();
    for cut in [false, true] {
        for t in &trips {
            if t["closed"] != json!(true) && truthy(t.get("truncated_open")) == cut {
                open_entry.entry((py_str(&t["symbol"]), py_str(&t["side"]))).or_insert_with(|| t["entry_at"].clone());
            }
        }
    }
    for p in positions.iter_mut() {
        let k = (py_str(&p["symbol"]), py_str(&p["side"]));
        let e = open_entry.get(&k).filter(|v| truthy(Some(v))).cloned().unwrap_or_else(|| p["updated_at"].clone());
        p["entry_at"] = e;
    }
    let mut open_orders: Value = json!([]);
    for rows in &all_rows {
        let Some(rs) = rows.as_array() else {
            open_orders = Value::Null; // 못 읽으면 None -- «없음»이 아니라 «모름»
            break;
        };
        for o in rs {
            if o.get("type").and_then(Value::as_str) != Some("LIMIT") {
                continue;
            }
            open_orders.as_array_mut().unwrap().push(json!({"symbol": o.get("symbol").cloned().ok_or("KeyError: 'symbol'")?, "side": o.get("side").cloned().ok_or("KeyError: 'side'")?,
                "price": req_f(o, "price")?, "qty": req_f(o, "origQty")? - or0(o, "executedQty")?, "reduce_only": truthy(o.get("reduceOnly")),
                "cid": if truthy(o.get("clientOrderId")) { py_str(&o["clientOrderId"]) } else { String::new() }}));
        }
    }
    let (chosen, listed) = pick_balance(&balance, quote)?;
    let multi = truthy(balance.get("multiAssetsMargin"));
    let mut balances = Obj::new();
    for q in ["USDC", "USDT"] {
        let has = balance.get("assets").and_then(Value::as_array).is_some_and(|a| a.iter().any(|x| x.get("asset").and_then(Value::as_str) == Some(q)));
        if multi || has {
            balances.insert(q.into(), pick_balance(&balance, q)?.0);
        }
    }
    Ok(json!({"ok": true, "generated_at": iso_utc(rest.now()), "balance": chosen, "balances": balances,
              "balance_asset": if multi { Value::Null } else { json!(quote) }, "multi_assets_margin": multi, "assets": listed,
              "positions": positions, "open_orders": open_orders, "leverage_by_symbol": lev_by, "trades": trips,
              "trades_truncated": truncated, "trades_failed": failed}))
}

/// `produce_account` 의 주문 경로 몫: fetch_account + exec_symbol·exec_symbols · 마지막 정상값(화면용 last_good, 30분).
pub struct AccountBook {
    pub eth_symbol: String,
    pub last_good: Option<(f64, Value)>,
}

impl AccountBook {
    /// 조회 대상 심볼 = 시세 심볼(시세만 코인 빼고) + 주문 심볼(원본 dict.fromkeys 순서).
    pub fn symbols(&self) -> Vec<String> {
        let mut v: Vec<String> = vec![];
        for s in ["ETHUSDT", "SOLUSDT", "BTCUSDT", "XRPUSDT", "HYPEUSDT", "SNDKUSDT", "SOXLUSDT"].iter().map(|s| s.to_string())
            .chain([self.eth_symbol.clone(), "SOLUSDC".into(), "XRPUSDC".into(), "BTCUSDC".into(), "SNDKUSDT".into(), "SOXLUSDT".into()]) {
            if !v.contains(&s) {
                v.push(s);
            }
        }
        v
    }
    pub fn exec_symbols(&self) -> Value {
        json!({"eth": self.eth_symbol, "sol": "SOLUSDC", "xrp": "XRPUSDC", "btc": "BTCUSDC", "sndk": "SNDKUSDT", "soxl": "SOXLUSDT"})
    }
    pub async fn produce<R: AccountRest>(&mut self, rest: &mut R) -> Result<Value, String> {
        let quote = if self.eth_symbol.ends_with("USDC") { "USDC" } else { "USDT" };
        let mut p = fetch_account(rest, &self.symbols(), quote, DEFAULT_TRADE_LIMIT).await?;
        p["exec_symbol"] = json!(self.eth_symbol);
        p["exec_symbols"] = self.exec_symbols();
        // 🔴2026-11-02 ok=false 는 그대로 둔다(감시·주문 경로는 «못 읽음 = 모름», 09-26) -- 화면용으로만 마지막 정상값을 같이 싣는다(30분까지)
        let now = rest.now();
        if truthy(p.get("ok")) {
            self.last_good = Some((now, p.clone()));
        } else if let Some((ts, g)) = &self.last_good {
            if now - ts < LAST_GOOD_SEC {
                let mut g = g.clone();
                g["age_s"] = json!((now - ts) as i64);
                p["last_good"] = g;
            }
        }
        Ok(p)
    }
}

/// 계좌 한 곳(2026-11-21 · 원본 `binance_account_cache` + `swr_locks["binance_account"]`): 감시 틱(swr 10초)·접수(새로 읽기)·체결 뒤 갱신기·화면(/api/binance-account)이 같이 쓴다.
/// `book` 잠금 = 원본 swr 잠금(새로 읽기가 겹치지 않는다 · 마지막 정상값 last_good 도 하나).
pub struct SharedAccount {
    /// (받은 시각, 계좌)
    pub cache: std::sync::Mutex<Option<(f64, Value)>>,
    pub book: tokio::sync::Mutex<AccountBook>,
    pub stale_grace: f64,
}

impl SharedAccount {
    pub fn new(eth_symbol: &str) -> Self {
        SharedAccount { cache: Default::default(), book: tokio::sync::Mutex::new(AccountBook { eth_symbol: eth_symbol.into(), last_good: None }), stale_grace: 60.0 }
    }
    /// 원본 produce_account(새로 읽기 · 캐시에 안 넣음) + 스트림 계좌 대조.
    pub async fn produce<R: AccountRest>(&self, rest: &mut R, hub: Option<&crate::userstream::Hub>) -> Result<Value, String> {
        let now = rest.now();
        let v = self.book.lock().await.produce(rest).await?;
        if let Some(h) = hub {
            h.reconcile(&v, now);
        }
        Ok(v)
    }
    /// swr_cached("binance_account", 10초, max_stale). ponytail: 원본은 묵은 값을 주고 뒤에서 갱신한다 -- 여기는 기다려 새로 읽는다(더 새 값 · 그만큼 늦다).
    pub async fn cached<R: AccountRest>(&self, rest: &mut R, hub: Option<&crate::userstream::Hub>) -> Result<Value, String> {
        let now = rest.now();
        let c = self.cache.lock().unwrap().clone();
        let (ts, has) = c.as_ref().map_or((0.0, false), |c| (c.0, true));
        match swr_decide(now, ts, has, ACCOUNT_CACHE_SEC, self.stale_grace) {
            Swr::Serve => Ok(c.unwrap().1),
            Swr::ServeAndRefresh | Swr::Fetch => self.store(rest, hub, now).await,
        }
    }
    /// 새로 읽어 캐시에 넣는다. `reuse_1s` = 화면 ?fresh=1(원본: 잠금 안에서 1초 안 받은 값이면 그것 -- 연타에도 거래소는 한 번 · A3) · 갱신기 = false.
    pub async fn refresh<R: AccountRest>(&self, rest: &mut R, hub: Option<&crate::userstream::Hub>, reuse_1s: bool) -> Result<Value, String> {
        let mut book = self.book.lock().await;
        let now = rest.now();
        if reuse_1s {
            if let Some((ts, v)) = self.cache.lock().unwrap().clone() {
                if !fresh_needs_fetch(now, ts) {
                    return Ok(v);
                }
            }
        }
        let v = book.produce(rest).await?;
        drop(book);
        if let Some(h) = hub {
            h.reconcile(&v, now);
        }
        *self.cache.lock().unwrap() = Some((now, v.clone()));
        Ok(v)
    }
    async fn store<R: AccountRest>(&self, rest: &mut R, hub: Option<&crate::userstream::Hub>, now: f64) -> Result<Value, String> {
        let v = self.produce(rest, hub).await?;
        *self.cache.lock().unwrap() = Some((now, v.clone()));
        Ok(v)
    }
}

/// `swr_cached` 의 결정(server.py): 새것이면 그대로 · ttl+max_stale 안이면 묵은 값 주고 뒤에서 갱신 · 그 밖은 기다려 새로 읽기.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swr {
    Serve,
    ServeAndRefresh,
    Fetch,
}
pub fn swr_decide(now_mono: f64, ts: f64, has_payload: bool, ttl: f64, max_stale: f64) -> Swr {
    let age = now_mono - ts;
    if has_payload && age < ttl {
        Swr::Serve
    } else if has_payload && age < ttl + max_stale {
        Swr::ServeAndRefresh
    } else {
        Swr::Fetch
    }
}
/// `?fresh=1`: 1초 안에 이미 새로 받았으면 그 값(연타에도 거래소는 한 번 · A3).
pub fn fresh_needs_fetch(now_mono: f64, ts: f64) -> bool {
    now_mono - ts > FRESH_REUSE_SEC
}

// ── 사용자 스트림 ──────────────────────────────────────────────────────────────
/// 메시지 → "fill"(체결·잔고 변화 -- 계좌를 다시 읽는다) · "order"(취소·만료 -- 기다리던 루프만 깨운다) · "expired"(listenKey 만료 -- 다시 연결) · ""(무시: 우리가 낸 NEW 등 -- 리페그마다 계좌를 읽지 않게).
pub fn user_event_kind(msg: &Value) -> Result<&'static str, String> {
    let Some(m) = msg.as_object() else { return Ok("") };
    match m.get("e").and_then(Value::as_str) {
        Some("ORDER_TRADE_UPDATE") => {
            let o = m.get("o").filter(|v| truthy(Some(v)));
            let x = match o {
                None => None,
                Some(Value::Object(o)) => o.get("x").and_then(Value::as_str),
                Some(v) => return Err(format!("AttributeError: '{}' object has no attribute 'get'", type_name(v))),
            };
            Ok(match x { Some("TRADE" | "CALCULATED") => "fill", Some("CANCELED" | "EXPIRED" | "AMENDMENT") => "order", _ => "" })
        }
        Some("ACCOUNT_UPDATE") => Ok("fill"),
        Some("listenKeyExpired") => Ok("expired"),
        _ => Ok(""),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserStream {
    pub connected: bool,
    pub seq: i64,
    pub last_event_at: Option<f64>,
    pub error: Option<String>,
    /// 연결마다 1씩 -- 스트림으로 고친 계좌가 «이 연결에서» 맞춰 본 것인지(끊긴 사이 사건은 다시 안 온다).
    pub epoch: u64,
}

pub enum Recv {
    Text(String),
    Timeout,
    /// CLOSE·ERROR 등 TEXT 아닌 것 -- 다시 연결.
    Other,
}

/// 스트림 루프가 닿는 바깥. Err = 파이썬 예외 문자열(`"<형>: <문구>"`).
#[allow(async_fn_in_trait)]
pub trait StreamIo {
    fn ban_remaining(&self) -> f64;
    fn mono(&self) -> f64;
    fn wall(&self) -> f64;
    /// asyncio.sleep -- false = 취소(루프 끝).
    async fn sleep(&mut self, sec: f64) -> bool;
    async fn listen_key(&mut self, method: &str) -> Result<String, String>;
    async fn connect(&mut self, url: &str, heartbeat: f64) -> Result<(), String>;
    async fn recv(&mut self, timeout: f64) -> Result<Recv, String>;
    fn wake(&mut self);
    fn dirty(&mut self);
    /// 계좌 사건(스트림으로 계좌 유지 -- `StreamAccount::apply`). 원본엔 없다(받아 둔 값 자리).
    fn event(&mut self, _msg: &Value) {}
    /// `us` 가 바뀐 직후(연결·끊김·알림 시각·오류) -- 바깥의 사본을 맞춘다. 원본은 같은 dict 를 같이 본다.
    fn state(&mut self, _us: &UserStream) {}
    fn tp_leg_filled(&mut self, symbol: &str, ps: &str, cid: &str) -> Result<(), String>;
    fn log(&mut self, s: String);
}

/// 원본 `_user_stream_run`. 🔴알림은 앞당기기만 -- 끊겨도 기존 3초·10초 조회가 그대로 돈다(A10). listenKey 는 지우지 않는다(같은 계좌의 다른 연결까지 끊긴다).
pub async fn user_stream_run<S: StreamIo>(io: &mut S, us: &mut UserStream) {
    let mut wait = RECONNECT_MIN_S;
    loop {
        let mut t_conn: Option<f64> = None;
        let r: Result<(), String> = async {
            let ban = io.ban_remaining();
            if ban > 0.0 {
                return Err("ban".into());
            }
            let lk = io.listen_key("POST").await?;
            if lk.is_empty() {
                return Err("RuntimeError: listenKey 빈 응답".into());
            }
            io.connect(&format!("{USER_STREAM_WS}{lk}"), WS_HEARTBEAT_S).await?;
            us.connected = true;
            us.error = None;
            us.epoch += 1;
            io.state(us);
            io.log("user stream: 체결 알림 연결".into());
            io.dirty(); // 끊긴 동안의 체결은 다시 오지 않는다 -- 붙자마자 계좌를 한 번 읽는다(A8)
            io.wake();
            let mut kept = io.mono();
            t_conn = Some(kept);
            let mut first = true;
            loop {
                if io.mono() - kept > USER_STREAM_KEEPALIVE_S {
                    kept = io.mono();
                    if let Err(e) = io.listen_key("PUT").await {
                        io.log(format!("user stream: 연장 실패 {}", e.split_once(": ").map_or(e.as_str(), |x| x.1)));
                    }
                }
                let msg = match io.recv(WS_RECV_TIMEOUT_S).await? {
                    Recv::Timeout => continue,
                    Recv::Other => break, // 거래소가 24시간마다 끊는다 → 다시 연결
                    Recv::Text(s) => s,
                };
                let Ok(data) = serde_json::from_str::<Value>(&msg) else { continue };
                let kind = user_event_kind(&data)?;
                io.event(&data); // 스트림 계좌(받아 둔 값) -- NEW 접수 등 원본이 무시하는 사건도 미체결 목록엔 필요하다
                let o = if data.get("e").and_then(Value::as_str) == Some("ORDER_TRADE_UPDATE") { data.get("o").and_then(Value::as_object).cloned().unwrap_or_default() } else { Obj::new() };
                if o.get("X").and_then(Value::as_str) == Some("FILLED") && truthy(o.get("c")) && py_str(&o["c"]).starts_with("dbtp") {
                    // 2026-11-11 분할 익절 조각 체결
                    let g = |k: &str| o.get(k).map(py_str).unwrap_or_else(|| "None".into());
                    if let Err(e) = io.tp_leg_filled(&g("s"), &g("ps"), &g("c")) {
                        io.log(format!("user stream: 익절 조각 체결 반영 실패 {e}"));
                    }
                }
                if kind == "expired" {
                    break;
                }
                if !kind.is_empty() {
                    if first {
                        io.log(format!("user stream: 첫 알림 수신({kind})"));
                        first = false;
                    }
                    us.last_event_at = Some(io.wall());
                    io.state(us);
                    io.wake();
                    if kind == "fill" {
                        io.dirty();
                    }
                }
            }
            Ok(())
        }
        .await;
        match r {
            Err(e) if e == "ban" => {
                // 원본: 차단 중이면 남은 시간 + 1초 자고 바로 다음 바퀴(아래 connected=False·대기는 건너뛴다)
                let b = io.ban_remaining();
                if !io.sleep(b + 1.0).await {
                    return;
                }
                continue;
            }
            Err(e) => {
                us.error = Some(e.chars().take(200).collect());
                io.log(format!("user stream: {}", us.error.as_deref().unwrap()));
            }
            Ok(()) => {}
        }
        us.connected = false;
        io.state(us); // 오류 문구도 함께
        if t_conn.is_some_and(|t| io.mono() - t > RECONNECT_RESET_AFTER_S) {
            wait = RECONNECT_MIN_S; // 5분 넘게 붙어 있었으면 정상 끊김(24시간) -- 바로 다시
        }
        if !io.sleep(wait).await {
            return;
        }
        wait = (wait * 2.0).min(RECONNECT_MAX_S);
    }
}

/// 원본 `account_refresher` 가 닿는 바깥.
#[allow(async_fn_in_trait)]
pub trait RefreshIo {
    /// acct_dirty.wait() -- false = 멈춤.
    async fn wait_dirty(&mut self) -> bool;
    fn clear_dirty(&mut self);
    async fn sleep(&mut self, sec: f64) -> bool;
    /// produce_account → 캐시에 넣기(swr 잠금 안). Err = 예외 문자열.
    async fn refresh(&mut self) -> Result<(), String>;
    fn wake(&mut self);
    fn log(&mut self, s: String);
}

/// 체결 알림 → 0.3초 묶고 → 계좌 다시 읽기 → seq+1 → 루프 깨움 → 최소 3초 간격(A11).
pub async fn account_refresher<R: RefreshIo>(io: &mut R, us: &mut UserStream) {
    loop {
        if !io.wait_dirty().await || !io.sleep(PUSH_COALESCE_S).await {
            return;
        }
        io.clear_dirty();
        match io.refresh().await {
            Ok(()) => {
                us.seq += 1;
                io.wake(); // 감시 틱이 새 계좌로 다시 본다
            }
            Err(e) => io.log(format!("user stream: 계좌 재조회 실패 {e}")),
        }
        if !io.sleep(ACCOUNT_PUSH_GAP_S).await {
            return;
        }
    }
}

// ── 스트림으로 유지하는 계좌(받아 둔 값 · 사용자 결정) ────────────────────────────
/// 마지막 REST 스냅숏 + 그 뒤 사용자 스트림 사건. 어긋남을 보면 다음 깨끗한 대조까지 REST 로 떨어진다.
#[derive(Debug, Clone, Default)]
pub struct StreamAccount {
    pub base: Option<Value>,
    pub base_at: f64,
    pub base_epoch: u64,
    /// 마지막 REST 대조가 스트림 예측과 어긋났다(다음 대조가 맞을 때까지 스트림 값을 안 쓴다).
    pub suspect: bool,
    pub mismatches: u64,
    pub applied: u64,
}

fn pos_key(p: &Value) -> (String, String) {
    (p.get("symbol").map(py_str).unwrap_or_default(), p.get("side").map(py_str).unwrap_or_default())
}
fn qty_of(p: &Value) -> f64 {
    p.get("qty").and_then(Value::as_f64).unwrap_or(0.0)
}

impl StreamAccount {
    /// 스트림 사건 하나를 반영한다(ACCOUNT_UPDATE 포지션 · ORDER_TRADE_UPDATE 지정가 미체결). 기준 스냅숏이 없으면 무시.
    pub fn apply(&mut self, msg: &Value) {
        let Some(base) = self.base.as_mut() else { return };
        match msg.get("e").and_then(Value::as_str) {
            Some("ACCOUNT_UPDATE") => {
                let none = vec![];
                for p in msg.get("a").and_then(|a| a.get("P")).and_then(Value::as_array).unwrap_or(&none) {
                    let g = |k: &str| p.get(k).and_then(|v| py_float(v).ok());
                    let (Some(amt), Some(sym)) = (g("pa"), p.get("s").map(py_str)) else { continue };
                    let ps = p.get("ps").map(py_str).unwrap_or_default();
                    let side = if ps == "LONG" || ps == "SHORT" { ps } else if amt > 0.0 { "LONG".into() } else { "SHORT".into() };
                    let list = base["positions"].as_array_mut().expect("ok 스냅숏엔 positions 가 있다");
                    let at = list.iter().position(|x| pos_key(x) == (sym.clone(), side.clone()));
                    if amt == 0.0 {
                        if let Some(i) = at {
                            list.remove(i);
                        }
                        continue;
                    }
                    let mark = at.and_then(|i| list[i].get("mark_price").and_then(Value::as_f64)).unwrap_or(g("ep").unwrap_or(0.0));
                    let mut np = at.map(|i| list[i].clone()).unwrap_or_else(|| json!({"symbol": sym, "side": side, "liquidation_price": 0.0, "leverage": 0.0,
                        "mark_price": mark, "updated_at": Value::Null, "entry_at": Value::Null}));
                    np["qty"] = json!(amt.abs());
                    if let Some(ep) = g("ep") {
                        np["entry_price"] = json!(ep);
                    }
                    if let Some(up) = g("up") {
                        np["unrealized_pnl"] = json!(up);
                    }
                    // ponytail: 명목 = 수량 × 마지막 REST 마크가(스트림엔 마크가가 없다) -- 화면·위험 표시 몫만 어긋날 수 있다(수량·진입가는 정확)
                    np["notional"] = json!(amt.abs() * mark);
                    match at {
                        Some(i) => list[i] = np,
                        None => list.push(np),
                    }
                }
            }
            Some("ORDER_TRADE_UPDATE") => {
                let Some(o) = msg.get("o").and_then(Value::as_object) else { return };
                if o.get("o").and_then(Value::as_str) != Some("LIMIT") || !base["open_orders"].is_array() {
                    return;
                }
                let cid = o.get("c").map(py_str).unwrap_or_default();
                let list = base["open_orders"].as_array_mut().unwrap();
                let at = list.iter().position(|x| x.get("cid").map(py_str).unwrap_or_default() == cid);
                let num = |k: &str| o.get(k).and_then(|v| py_float(v).ok()).unwrap_or(0.0);
                match o.get("X").and_then(Value::as_str) {
                    Some("NEW" | "PARTIALLY_FILLED") => {
                        let row = json!({"symbol": o.get("s").map(py_str).unwrap_or_default(), "side": o.get("S").map(py_str).unwrap_or_default(),
                                         "price": num("p"), "qty": num("q") - num("z"), "reduce_only": truthy(o.get("R")), "cid": cid});
                        match at {
                            Some(i) => list[i] = row,
                            None => list.push(row),
                        }
                    }
                    _ => {
                        if let Some(i) = at {
                            list.remove(i);
                        }
                    }
                }
            }
            _ => return,
        }
        self.applied += 1;
    }

    /// REST 로 새로 읽은 정상 계좌와 맞춰 본다. 같은 연결 안에서 스트림이 예측한 포지션(심볼·측면·수량)·미체결(표식·수량)이 다르면 어긋남.
    /// 어느 쪽이든 기준은 REST 로 바뀐다(REST 가 정답).
    /// 주의: REST 응답이 도착하기 전에 그 뒤의 스트림 사건이 먼저 오면 «가짜 어긋남»이 난다 -- 안전한 쪽(REST 로 떨어짐)이라 그대로 둔다.
    pub fn reconcile(&mut self, rest: &Value, now: f64, epoch: u64) -> bool {
        if !truthy(rest.get("ok")) {
            return true;
        }
        let same = match &self.base {
            Some(b) if self.base_epoch == epoch => {
                let pos = |v: &Value| {
                    let mut x: Vec<(String, String, String)> = v["positions"].as_array().map(|a| a.iter().map(|p| { let (s, d) = pos_key(p); (s, d, format!("{:.8}", qty_of(p))) }).collect()).unwrap_or_default();
                    x.sort();
                    x
                };
                let ords = |v: &Value| {
                    let mut x: Vec<String> = v["open_orders"].as_array().map(|a| a.iter().map(|o| format!("{}|{:.8}", o.get("cid").map(py_str).unwrap_or_default(), o.get("qty").and_then(Value::as_f64).unwrap_or(0.0))).collect()).unwrap_or_default();
                    x.sort();
                    x
                };
                pos(b) == pos(rest) && ords(b) == ords(rest)
            }
            _ => true, // 처음이거나 다른 연결에서 만든 기준 -- 비교할 게 없다
        };
        if !same {
            self.mismatches += 1;
        }
        self.suspect = !same;
        self.base = Some(rest.clone());
        self.base_at = now;
        self.base_epoch = epoch;
        same
    }

    /// 쓸 수 있으면 스트림으로 고친 계좌 · 아니면 그 이유.
    pub fn usable(&self, us: &UserStream, now: f64) -> Result<Value, &'static str> {
        let Some(b) = &self.base else { return Err("no_base") };
        if !us.connected {
            return Err("stream_down");
        }
        if self.base_epoch != us.epoch {
            return Err("reconnected"); // 끊긴 사이 사건은 다시 안 온다(A8)
        }
        if self.suspect {
            return Err("mismatch");
        }
        if now - self.base_at > STREAM_TRUST_SEC {
            return Err("stale");
        }
        let mut v = b.clone();
        v["generated_at"] = json!(iso_utc(now));
        v["source"] = json!("stream");
        Ok(v)
    }
}
