//! 손절·익절 걸기와 그 주변 -- `live/execution/live_manual_peg_execute_20260912.py` 의 `current_leverage`·`ensure_leverage`·`margin_type`·
//! `commission_bp`·`bracket_tag`·`cancel_bracket`·`clear_bracket`·`place_add_order`·`_post_stop`·`place_stop`·`place_bracket`·`place_tp_ladder`·
//! `cancel_wait_orphans` 를 그대로(2026-11-20 둘째 조각). 돌려주는 값은 원본 dict 와 같은 JSON(호출부 orders.py 가 dict 로 읽는다).
//! 정답지: tests/fixtures/http_fixture.json 의 «같은 순서로 같은 요청(글자까지) + 같은 결과».
//! 원본의 «키 없음» 검사는 `Binance::has_keys`, `_clock_offset` 은 `Binance::sync_clock`, 키·시계보정을 인자로 받는 함수(cancel_bracket 등)는 `bx.offset` 을 그대로 쓴다.
//!
//! 🔴원본 그대로 옮긴 «남은 위험»(고치지 않음 -- 보고만): 여기의 POST(익절 GTX · 분할 익절 · 자동 물타기 GTC · 비상 스탑)는
//!   `resolve_unknown_post` 를 거치지 않는다. 끊김·시간초과·5xx·-1007 이면 그대로 «못 걸었다»(placed=false)로 돌려준다 -- 실제로는 걸렸을 수 있다.
//! 🔴비상 스탑(STOP_MARKET closePosition, `post_stop`)은 **시장가형 주문**이다(발동 시 거래소가 전량 시장가로 닫는다, 시장가 분기 M10) --
//!   요청 모양만 옮겨 정답지와 맞췄다. 이 크레이트에서 쓸지는 사용자 결정 목록에 올림.

use crate::http::Binance;
use crate::plan::{split_tp_qty, TpLegIn};
use crate::sign::{error_of, executed_qty, py_float, py_str, str_or_empty, truthy, P};
use serde_json::{json, Map, Value};

pub const TP_TAG: &str = "dbtp";
pub const SL_TAG: &str = "dbsl";
/// 2026-10-14 자동 물타기 지정가 -- cancel_bracket 이 TP 와 함께 지운다(포지션 닫힘·손으로 물타기·새 진입)
pub const ADD_TAG: &str = "dbad";
/// 표준 USDT-M 메이커·테이커(bp) -- 조회도 못 하고 지난 값도 없을 때
pub const STD_FEE_BP: (f64, f64) = (2.0, 5.0);
/// 수수료율 캐시: 성공 1시간 · 실패 5분(가중치 20 이라 봇과 공유하는 IP 예산을 아낀다 -- 실패해도 매 미리보기마다 재호출하지 않는다).
pub const FEE_TTL_SEC: f64 = 3600.0;
pub const FEE_RETRY_SEC: f64 = 300.0;
const NO_KEYS: &str = "API 키가 없습니다";

/// 우리가 건 주문만 고른다 -- 사용자는 앱에서도 지정가 청산을 건다(09-24 aos_… 실측). 측면 글자까지 붙인다(롱 정리가 숏 주문을 지우지 않게).
pub fn bracket_tag(tag: &str, pside: &str) -> String {
    format!("{tag}{}", pside.chars().next().map(String::from).unwrap_or_default())
}

fn close_side(pside: &str) -> &'static str {
    if pside == "LONG" { "SELL" } else { "BUY" }
}

/// `x.get('__error__') if isinstance(x, dict) else x`
fn why(r: &Value) -> String {
    if r.is_object() { r.get("__error__").map_or("None".into(), py_str) } else { py_str(r) }
}

/// `"__error__" not in r`(dict 일 때만 키 검사 -- 목록이면 원소 비교라 거의 늘 참)
fn no_err(r: &Value) -> bool {
    error_of(r).is_none()
}

/// `r.get(k)`(dict 아니면 None -- ponytail: 원본은 dict 가 아니면 AttributeError 로 죽는다. 거래소 주문 응답은 늘 dict 라 None 으로 둔다)
fn get(r: &Value, k: &str) -> Value {
    r.get(k).cloned().unwrap_or(Value::Null)
}

/// `round(sum(abs(float(p.get("positionAmt") or 0.0)) for p in pos if p.get("positionSide") == pside), 8)` -- 파이썬 3.11 sum(앞에서부터 덧셈).
fn side_qty(pos: &[Value], pside: &str) -> Result<f64, String> {
    let mut q = 0.0;
    for p in pos.iter().filter(|p| p.get("positionSide").and_then(Value::as_str) == Some(pside)) {
        let v = p.get("positionAmt").filter(|v| truthy(Some(v)));
        q += v.map_or(Ok(0.0), py_float)?.abs();
    }
    Ok(crate::pyfmt::round_n(q, 8))
}

/// 지금 걸린 심볼 레버리지. 못 읽으면 None -- 그때는 «모른다»이지 «맞다»가 아니다.
pub async fn current_leverage(bx: &mut Binance, symbol: &str) -> Option<i64> {
    let r = bx.signed("GET", "/fapi/v2/positionRisk", &[("symbol", P::s(symbol))]).await;
    let first = r.as_array().and_then(|a| a.first())?;
    let v = first.get("leverage").filter(|v| truthy(Some(v)));
    let lev = v.map_or(Ok(0.0), py_float).ok()? as i64; // int(float(... or 0)) or None
    (lev != 0).then_some(lev)
}

/// 심볼 레버리지를 target 으로(**계정 설정**이라 주문과 별도 호출). 이미 같으면 안 보낸다(실패 지점 하나 줄임).
/// 🔴내리는 쪽은 거부될 수 있다(-4028 류) -- 실패를 삼키면 «걸었다고 생각했는데 안 걸린» 상태라 결과를 그대로 돌려준다. 실패해도 주문은 보낸다(호출부 몫, P16).
pub async fn ensure_leverage(bx: &mut Binance, symbol: &str, target: i64) -> Value {
    if target <= 0 {
        return json!({"changed": false, "reason": "목표값 없음"});
    }
    let now = current_leverage(bx, symbol).await;
    if now == Some(target) {
        return json!({"changed": false, "from": now, "to": target, "reason": "이미 같음"});
    }
    let r = bx.signed("POST", "/fapi/v1/leverage", &[("symbol", P::s(symbol)), ("leverage", P::I(target))]).await;
    if let Some(e) = error_of(&r) {
        return json!({"changed": false, "from": now, "to": target, "error": e});
    }
    let to = r.get("leverage").filter(|v| truthy(Some(v))).map_or(Ok(target as f64), py_float).map_or(target, |f| f as i64);
    json!({"changed": true, "from": now, "to": to})
}

/// 심볼의 마진 방식 "cross"·"isolated"(헤지 두 다리가 섞이면 isolated). 못 읽으면 None -- 읽기만(주문 없음).
/// 2026-10-09 자동 규칙은 교차 전제다(격리 20배는 청산 −4.6% 가 3σ 손절보다 안쪽 -- 실사고: 롱 0.893 청산 2,384 > 손절 2,376).
pub async fn margin_type(bx: &mut Binance, symbol: &str) -> Option<&'static str> {
    if !bx.has_keys() {
        return None;
    }
    bx.sync_clock().await;
    let r = bx.signed("GET", "/fapi/v2/positionRisk", &[("symbol", P::s(symbol))]).await;
    let a = r.as_array().filter(|a| !a.is_empty())?;
    let kinds: std::collections::BTreeSet<String> = a.iter().map(|p| str_or_empty(p, "marginType").to_lowercase()).collect();
    if kinds.len() == 1 && kinds.contains("cross") { Some("cross") } else if kinds.contains("isolated") { Some("isolated") } else { None }
}

/// 2026-10-10 주문 심볼의 계정 실수수료율. 🔴화면 «예상 비용» 표시 전용 -- 크기·손절·판정은 표준 상수(경제성은 표준 수수료 기준, USDC 0% 는 기간 한정).
/// source: live(1시간 안 조회) · stale(조회 실패, 마지막 성공값) · fallback(성공 이력 없음, 표준 2/5bp). 캐시는 `Binance` 하나에 하나(원본 모듈 전역).
pub async fn commission_bp(bx: &mut Binance, symbol: &str) -> Value {
    let now = bx.time();
    let hit = bx.fees.get(symbol).copied();
    if let Some((t, mk, tk, src)) = hit {
        if now - t < if src == "live" { FEE_TTL_SEC } else { FEE_RETRY_SEC } {
            return json!({"symbol": symbol, "maker_bp": mk, "taker_bp": tk, "source": src});
        }
    }
    let r = if bx.has_keys() {
        bx.sync_clock().await;
        Some(bx.signed("GET", "/fapi/v1/commissionRate", &[("symbol", P::s(symbol))]).await)
    } else {
        None
    };
    let rate = |k: &str| -> Option<f64> { r.as_ref()?.get(k).and_then(|v| py_float(v).ok()).map(|f| crate::pyfmt::round_n(f * 1e4, 4)) };
    let (mk, tk, src) = match (rate("makerCommissionRate"), rate("takerCommissionRate")) {
        (Some(m), Some(t)) => (m, t, "live"),
        _ => match hit.filter(|h| h.3 != "fallback") {
            Some(h) => (h.1, h.2, "stale"),
            None => (STD_FEE_BP.0, STD_FEE_BP.1, "fallback"),
        },
    };
    bx.fees.insert(symbol.into(), (now, mk, tk, src));
    json!({"symbol": symbol, "maker_bp": mk, "taker_bp": tk, "source": src})
}

/// 이 측면에 **우리가 건** TP 지정가와 비상 스탑을 지운다. 목록을 못 읽으면 그 사실을 돌려준다(못 지웠는데 «지웠다»로 넘어가면 닫힌 포지션 위에 주문이 남는다).
/// only_sl(2026-10-15 «손절 해제») = 비상 스탑만 -- 물타기·TP 는 그대로 · only_tp(2026-11-11 «익절선 지우기») = TP 만 -- 물타기·스탑 그대로.
pub async fn cancel_bracket(bx: &mut Binance, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Value {
    let (tp_pre, sl_pre, add_pre) = (bracket_tag(TP_TAG, pside), bracket_tag(SL_TAG, pside), bracket_tag(ADD_TAG, pside));
    let (mut n, mut errors) = (0i64, Vec::<String>::new());
    let opens = if only_sl { json!([]) } else { bx.signed("GET", "/fapi/v1/openOrders", &[("symbol", P::s(symbol))]).await };
    if !opens.is_array() {
        errors.push(format!("openOrders: {}", why(&opens)));
    }
    let list = opens.as_array().cloned().unwrap_or_default();
    // 2026-11-11 분할 익절: 지우기 전에 살아 있던 우리 익절(없어진 조각 = 체결) -- dict 가 아니면 «모름»(None)
    let tp_open: Option<Vec<String>> = opens.is_array().then(|| {
        list.iter().filter(|o| str_or_empty(o, "clientOrderId").starts_with(&tp_pre)).map(|o| py_str(&get(o, "clientOrderId"))).collect()
    });
    for o in &list {
        let cid = str_or_empty(o, "clientOrderId");
        if cid.starts_with(&tp_pre) || (!only_tp && cid.starts_with(&add_pre)) {
            let r = bx.signed("DELETE", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("orderId", P::of(o.get("orderId")))]).await;
            match error_of(&r) {
                None => n += 1,
                Some(e) => errors.push(e.into()),
            }
        }
    }
    if only_tp {
        return json!({"cancelled": n, "errors": errors, "tp_open": tp_open});
    }
    let algos = bx.signed("GET", "/fapi/v1/openAlgoOrders", &[("symbol", P::s(symbol))]).await;
    if !algos.is_array() {
        errors.push(format!("openAlgoOrders: {}", why(&algos)));
    }
    for a in algos.as_array().cloned().unwrap_or_default() {
        // 2026-10-20 두 코인 보유: 다른 코인 스탑은 절대 안 지운다(심볼 칸이 없으면 이 심볼로 본다 -- 원본 a.get("symbol", symbol))
        let same = a.get("symbol").map_or(true, |s| s.as_str() == Some(symbol));
        if str_or_empty(&a, "clientAlgoId").starts_with(&sl_pre) && same {
            let r = bx.signed("DELETE", "/fapi/v1/algoOrder", &[("algoId", P::of(a.get("algoId")))]).await;
            match error_of(&r) {
                None => n += 1,
                Some(e) => errors.push(e.into()),
            }
        }
    }
    json!({"cancelled": n, "errors": errors})
}

/// cancel_bracket 을 키·시계보정까지 스스로 챙겨 부른다(서버 감시 루프 · «손절 해제» 버튼).
pub async fn clear_bracket(bx: &mut Binance, symbol: &str, pside: &str, only_sl: bool, only_tp: bool) -> Value {
    if !bx.has_keys() {
        return json!({"cancelled": 0, "errors": [NO_KEYS]});
    }
    bx.sync_clock().await;
    cancel_bracket(bx, symbol, pside, only_sl, only_tp).await
}

/// 자동 물타기 지정가(2026-10-14). 🔴**GTC**(post-only 아님 -- 이미 그 아래면 그 자리에서 테이커 체결, 시장가 분기 M12: 원본 그대로 옮김 · 사용자 결정 목록).
/// 실패해도 진입·손절은 그대로(결과를 돌려 화면이 말한다). 지우기는 cancel_bracket(ADD_TAG). 불명 응답 조회 없음(남은 위험).
pub async fn place_add_order(bx: &mut Binance, symbol: &str, pside: &str, price: f64, qty: f64) -> Value {
    if !bx.has_keys() {
        return json!({"placed": false, "reason": NO_KEYS});
    }
    let cid = format!("{}{}", bracket_tag(ADD_TAG, pside), crate::sign::ms(bx.time())); // 원본: dict(표식 시각)를 먼저 만들고 시계보정을 묻는다
    bx.sync_clock().await;
    let side = if pside == "LONG" { "BUY" } else { "SELL" };
    let r = bx.signed("POST", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("side", P::s(side)), ("positionSide", P::s(pside)), ("type", P::s("LIMIT")),
        ("timeInForce", P::s("GTC")), ("price", P::F(price)), ("quantity", P::F(qty)), ("newClientOrderId", P::S(cid))]).await;
    json!({"placed": no_err(&r), "price": price, "qty": qty, "order_id": get(&r, "orderId"), "error": get(&r, "__error__")})
}

/// 비상 스탑 = Algo STOP_MARKET closePosition(마크가 기준). place_bracket · place_stop 이 같이 쓴다.
/// 🔴조건부 주문은 `/fapi/v1/algoOrder` 로만(2026-09-25: `/fapi/v1/order` 의 STOP_MARKET 은 -4120 거부 -- 4일 이력 STOP 0건, 실패는 메모리에만 있었다).
/// 🔴시장가형 주문(발동 시 전량 시장가, M10) -- 사용자 결정 목록.
pub async fn post_stop(bx: &mut Binance, symbol: &str, pside: &str, price: &Value, stamp: i64) -> Value {
    let r = bx.signed("POST", "/fapi/v1/algoOrder", &[("algoType", P::s("CONDITIONAL")), ("symbol", P::s(symbol)), ("side", P::s(close_side(pside))),
        ("positionSide", P::s(pside)), ("type", P::s("STOP_MARKET")), ("triggerPrice", P::of(Some(price))), ("closePosition", P::s("true")),
        ("workingType", P::s("MARK_PRICE")), ("clientAlgoId", P::S(format!("{}{stamp}", bracket_tag(SL_TAG, pside))))]).await;
    json!({"placed": no_err(&r), "price": price, "algo_id": get(&r, "algoId"), "error": get(&r, "__error__")})
}

/// 2026-10-15 «손절 다시 켜기» -- 비상 스탑만 다시 건다(place_bracket 은 TP·물타기까지 지우고 다시 걸어서 못 쓴다).
/// 남은 우리 스탑은 먼저 지운다(두 개 걸리지 않게). 포지션이 없거나 못 걸면 placed=false -- 🔴호출 쪽은 그때 아무것도 안 바꾼다.
pub async fn place_stop(bx: &mut Binance, symbol: &str, pside: &str, price: f64) -> Value {
    if !bx.has_keys() {
        return json!({"placed": false, "error": NO_KEYS});
    }
    bx.sync_clock().await;
    let pos = bx.signed("GET", "/fapi/v2/positionRisk", &[("symbol", P::s(symbol))]).await;
    let Some(list) = pos.as_array() else {
        return json!({"placed": false, "error": format!("포지션 조회 실패: {}", why(&pos))});
    };
    let mut any = false;
    for p in list {
        if p.get("positionSide").and_then(Value::as_str) == Some(pside) {
            let v = p.get("positionAmt").filter(|v| truthy(Some(v)));
            if v.map_or(Ok(0.0), py_float).unwrap_or(0.0).abs() > 0.0 {
                any = true;
                break;
            }
        }
    }
    if !any {
        return json!({"placed": false, "error": "포지션 없음"});
    }
    let cl = cancel_bracket(bx, symbol, pside, true, false).await;
    let errs = cl["errors"].as_array().cloned().unwrap_or_default();
    if !errs.is_empty() {
        return json!({"placed": false, "error": format!("남은 스탑 정리 실패: {}", errs.iter().map(py_str).collect::<Vec<_>>().join("; "))});
    }
    let stamp = crate::sign::ms(bx.time());
    post_stop(bx, symbol, pside, &json!(price), stamp).await
}

/// 체결 뒤 TP 지정가(GTX, 포지션 **전량**)와 비상 스탑(Algo STOP_MARKET closePosition)을 건다. 물타기면 우리 옛 주문을 지우고 새 레벨·새 수량으로 다시.
/// 🔴실패해도 진입을 되돌리지 않는다(이미 체결됐다) -- 결과를 그대로 돌려 화면이 크게 말하게 한다.
/// strict(2026-10-29 선 수정): 옛 주문을 다 못 지웠으면 새로 걸지 않는다(같은 쪽 전량 익절 둘 방지).
/// ponytail(원본 주석 그대로): TP 수량은 거는 순간의 포지션이다. 뒤에 일부만 수동 청산하면 TP 가 포지션보다 크게 남는다 -- 감시 루프가 포지션이 사라지면 지운다.
pub async fn place_bracket(bx: &mut Binance, bracket: &Value, symbol: &str, pside: &str, strict: bool) -> Value {
    if !bx.has_keys() {
        return json!({"placed": false, "reason": NO_KEYS});
    }
    bx.sync_clock().await;
    let pos = bx.signed("GET", "/fapi/v2/positionRisk", &[("symbol", P::s(symbol))]).await;
    let Some(list) = pos.as_array() else {
        return json!({"placed": false, "reason": format!("포지션 조회 실패: {}", py_str(&get(&pos, "__error__")))});
    };
    let qty = match side_qty(list, pside) {
        Ok(q) => q,
        Err(e) => return json!({"__raise__": e}),
    };
    if qty <= 0.0 {
        return json!({"placed": false, "reason": "포지션 없음"});
    }
    let cleared = cancel_bracket(bx, symbol, pside, false, false).await;
    let mut out = Map::new();
    out.insert("qty".into(), json!(qty));
    out.insert("cleared".into(), cleared.clone());
    if strict && truthy(cleared.get("errors")) {
        out.insert("placed".into(), json!(false));
        out.insert("reason".into(), json!("옛 주문 정리 실패"));
        return Value::Object(out);
    }
    let stamp = crate::sign::ms(bx.time());
    if truthy(bracket.get("tp_price")) {
        let tp = &bracket["tp_price"];
        let r = bx.signed("POST", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("side", P::s(close_side(pside))), ("positionSide", P::s(pside)),
            ("type", P::s("LIMIT")), ("timeInForce", P::s("GTX")), ("price", P::of(Some(tp))), ("quantity", P::F(qty)),
            ("newClientOrderId", P::S(format!("{}{stamp}", bracket_tag(TP_TAG, pside))))]).await;
        // 🔴남은 위험(원본 그대로): 익절 GTX 의 불명 응답(끊김·시간초과·5xx·-1007)을 조회하지 않고 placed=false 로 돌려준다 -- 실제로 걸렸을 수 있다
        out.insert("tp".into(), json!({"placed": no_err(&r), "price": tp, "order_id": get(&r, "orderId"), "error": get(&r, "__error__")}));
    }
    if truthy(bracket.get("backstop_price")) {
        let b = post_stop(bx, symbol, pside, &bracket["backstop_price"], stamp).await;
        out.insert("backstop".into(), b);
    }
    // all(...) -- 둘 다 없으면 all([]) = True(원본 그대로)
    let placed = ["tp", "backstop"].iter().filter_map(|k| out.get(*k)).all(|v| truthy(v.get("placed")));
    out.insert("placed".into(), json!(placed));
    Value::Object(out)
}

/// 2026-11-11 분할 익절: 우리 익절(dbtp)을 지우고 legs(주문가 price · 무게 w) 대로 포지션 수량을 나눠 조각마다 GTX 로 다시 건다.
/// 🔴옛 익절을 다 못 지웠으면 아무것도 안 건다(같은 쪽 익절이 겹쳐 포지션보다 커지지 않게) · 키·포지션 조회 실패면 "cleared" 없이 돌려준다(호출부 = 아무것도 안 바꿈).
/// 걸려 있던 조각이 목록에 없다 = 체결 **또는** 거래소가 지움 -- 2026-11-12 검토: 그 주문을 직접 조회해 FILLED 일 때만 체결(못 읽음·취소 = 남은 수량으로 다시 건다).
/// Err = 원본이 예외를 내는 경로(필터 읽기 실패 등 -- 호출부의 settle 몫).
pub async fn place_tp_ladder(bx: &mut Binance, legs: &[Value], symbol: &str, pside: &str) -> Result<Value, String> {
    if !bx.has_keys() {
        return Ok(json!({"placed": false, "reason": NO_KEYS}));
    }
    bx.sync_clock().await;
    let pos = bx.signed("GET", "/fapi/v2/positionRisk", &[("symbol", P::s(symbol))]).await;
    let Some(list) = pos.as_array() else {
        return Ok(json!({"placed": false, "reason": format!("포지션 조회 실패: {}", why(&pos))}));
    };
    let qty = side_qty(list, pside)?;
    if qty <= 0.0 {
        return Ok(json!({"placed": false, "reason": "포지션 없음"}));
    }
    let cleared = cancel_bracket(bx, symbol, pside, false, true).await;
    let mut out = Map::new();
    out.insert("qty".into(), json!(qty));
    out.insert("cleared".into(), cleared.clone());
    if truthy(cleared.get("errors")) {
        out.insert("placed".into(), json!(false));
        out.insert("reason".into(), json!("옛 익절 정리 실패"));
        return Ok(Value::Object(out));
    }
    let seen: Option<Vec<String>> = cleared.get("tp_open").and_then(Value::as_array).map(|a| a.iter().map(py_str).collect());
    let mut live: Vec<Map<String, Value>> = legs.iter().map(|l| l.as_object().cloned().unwrap_or_default()).collect();
    for l in live.iter_mut() {
        let filled = truthy(l.get("filled"));
        l.insert("filled".into(), json!(filled));
        let resting_gone = !filled && truthy(l.get("resting")) && truthy(l.get("cid"))
            && seen.as_ref().is_some_and(|s| !s.contains(&py_str(&l["cid"])));
        if resting_gone {
            // 그 주문 상태를 직접 본다(드묾 · 조각당 1회)
            let o = bx.signed("GET", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("origClientOrderId", P::of(l.get("cid")))]).await;
            l.insert("filled".into(), json!(o.is_object() && o.get("status").and_then(Value::as_str) == Some("FILLED")));
        }
    }
    let todo: Vec<usize> = (0..live.len()).filter(|&i| !truthy(live[i].get("filled"))).collect();
    let filters = bx.load_filters(symbol).await?; // 원본: 조각이 없어도 읽는다
    let qs = if todo.is_empty() {
        vec![]
    } else {
        let inp = todo.iter().map(|&i| {
            let l = &live[i];
            let price = l.get("price").ok_or("KeyError: 'price'").map_err(String::from).and_then(py_float)?;
            let w = l.get("w").filter(|v| truthy(Some(v))).map(py_float).transpose()?;
            Ok(TpLegIn { price, w })
        }).collect::<Result<Vec<_>, String>>()?;
        split_tp_qty(qty, &inp, &filters, pside).map_err(|e| format!("{e:?}"))?
    };
    let stamp = crate::sign::ms(bx.time());
    for (j, (&i, &q)) in todo.iter().zip(qs.iter()).enumerate() {
        let cid = format!("{}{stamp}{j}", bracket_tag(TP_TAG, pside));
        let l = &mut live[i];
        for (k, v) in [("qty", json!(q)), ("merged", json!(q <= 0.0)), ("resting", json!(false)), ("order_id", Value::Null), ("error", Value::Null), ("cid", json!(cid))] {
            l.insert(k.into(), v);
        }
        if q <= 0.0 {
            continue;
        }
        let price = l["price"].clone();
        let r = bx.signed("POST", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("side", P::s(close_side(pside))), ("positionSide", P::s(pside)),
            ("type", P::s("LIMIT")), ("timeInForce", P::s("GTX")), ("price", P::of(Some(&price))), ("quantity", P::F(q)), ("newClientOrderId", P::S(cid))]).await;
        // 🔴남은 위험(원본 그대로): 불명 응답이면 resting=false -- 서버 감시로 넘어가지만 거래소엔 걸렸을 수 있다
        let l = &mut live[i];
        l.insert("resting".into(), json!(no_err(&r)));
        l.insert("order_id".into(), get(&r, "orderId"));
        l.insert("error".into(), get(&r, "__error__"));
    }
    let placed = live.iter().all(|l| truthy(l.get("resting")) || truthy(l.get("merged")) || truthy(l.get("filled")));
    out.insert("legs".into(), Value::Array(live.into_iter().map(Value::Object).collect()));
    out.insert("placed".into(), json!(placed));
    Ok(Value::Object(out))
}

/// 서버 재시작 때: 기다리기·따라가기 표식(dbwt·dbpg) 주문이 거래소에 남아 있으면 지운다 -- 그 주문을 지켜보던 작업은 죽었다(고아, R1).
/// dbtp·dbsl·dbad 는 안 지운다. 목록을 못 읽으면 그 사실을 돌려준다(못 지웠는데 «없다»로 넘어가면 안 된다).
pub async fn cancel_wait_orphans(bx: &mut Binance, symbols: &[&str]) -> Value {
    if !bx.has_keys() {
        return json!({"cancelled": [], "errors": [NO_KEYS]});
    }
    bx.sync_clock().await;
    let (mut done, mut errors) = (Vec::<Value>::new(), Vec::<String>::new());
    for &symbol in symbols {
        let opens = bx.signed("GET", "/fapi/v1/openOrders", &[("symbol", P::s(symbol))]).await;
        let Some(list) = opens.as_array().cloned() else {
            errors.push(format!("{symbol} openOrders: {}", why(&opens)));
            continue;
        };
        for o in list {
            let cid = str_or_empty(&o, "clientOrderId");
            if cid.starts_with(crate::exec::WAIT_TAG) || cid.starts_with(crate::exec::PEG_TAG) {
                let r = bx.signed("DELETE", "/fapi/v1/order", &[("symbol", P::s(symbol)), ("orderId", P::of(o.get("orderId")))]).await;
                match error_of(&r) {
                    Some(e) => errors.push(e.into()),
                    None => {
                        let oq = o.get("origQty").filter(|v| truthy(Some(v))).map_or(Ok(0.0), py_float).unwrap_or(f64::NAN);
                        done.push(json!({"symbol": symbol, "order_id": get(&o, "orderId"), "side": get(&o, "positionSide"),
                                         "filled": executed_qty(&o).unwrap_or(f64::NAN), "qty": oq}));
                    }
                }
            }
        }
    }
    json!({"cancelled": done, "errors": errors})
}
