"""대시보드 수동 진입 — 실주문 집행 (2026-09-12, 2단계).

`live_manual_peg_entry_20260912.py` 가 **무엇을 보낼지**(순수 함수, 네트워크 없음)를 정하고
이 파일이 **실제로 보낸다**. 둘을 가른 이유: 실돈이 오가는 코드는 격리해 두는 편이 읽기도
검사하기도 쉽고, 1단계 미리보기는 이 파일을 아예 import 하지 않아도 돌아간다.

정책(사용자 선택 b): peg post-only(GTX) 로 걸고 FALLBACK_SEC 까지 지켜본 뒤
**남은 수량만** 테이커로 넘긴다. 실측 1,340legs 에서 peg 2.76bp/leg, 폴백분 14.82bp.

**진입과 청산이 같은 리페그 루프를 쓴다**(`fill_maker`). 2026-09-15 이전에는 진입만
리페그하지 않았는데, 그 근거("어차피 테이커였으니 순수 개선")는 **배포 전 기준**이라
배포 후에는 더 이상 맞지 않았다.
  청산(2026-09-13): 미체결이 «안 닫음»이라 리페그. 섀도우 23,332legs 에서 90.4% → 99.2%.
  진입(2026-09-15): 사용자 *"좀 비싸게 사고 있는 것 같다"* 에서 출발한 재측정. 리페그 없는
    진입은 섀도우의 `static` 과 같은 모양이고, 그 다리의 **미체결이 저변동 14.4% ·
    고변동 7.6%** 다. 미체결은 120초 뒤 시장가로 끝나 평균 **8.75~11.5bp**(체결 다리 1.98).
    평균은 두 정책이 거의 같지만(2.957 vs 2.999) **꼬리가 다르다 — 최악 20.8 vs 87.3bp**.
    ⚠️리페그는 «쫓아가며 산다»는 뜻이라 의도한 가격보다 높게 들어갈 수 있다. 그래도 택한
    이유는 시장가 전락(5.0bp + 이미 달아난 가격)이 그보다 비싸기 때문이다.
🔴**오프셋(호가보다 깊게 눕히기)은 지렛대가 아니다**(같은 날 검토·기각). 1틱 = 0.0405bp 인데
  스프레드가 정확히 1틱이고, 체결 대기 4.2초 동안 가격이 1.25bp(31틱) 흔들린다. 미체결 한 건이
  +9.07bp 라 1틱당 체결률이 0.45%p 만 떨어져도 본전이다. 비용의 82%는 수수료(2.0bp)다.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any

# 서명·시계보정은 계좌 조회 모듈 것을 그대로 쓴다. 밑줄 이름을 건너 쓰는 건 보기 나쁘지만
# HMAC 서명과 /fapi/v1/time 앵커를 **두 벌로 두는 것보다 낫다** -- 어긋나면 한쪽만 조용히
# -1021 로 실패한다(2026-09-10 WSL 시계 1초 드리프트 전례).
from scripts.live_binance_account_20260910 import _clock_offset, _sign

FAPI = "https://fapi.binance.com"
FALLBACK_SEC = 120.0
POLL_SEC = 3.0
TERMINAL = ("FILLED", "CANCELED", "EXPIRED", "REJECTED")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def signed(session, method: str, path: str, params: dict, key: str, secret: str,
                 offset: int) -> Any:
    """서명 요청 하나. 실패해도 예외 대신 {"__error__": ...} 를 돌려준다 -- 주문 흐름이
    중간에 예외로 끊기면 «보냈는지 안 보냈는지 모르는» 상태가 남는다. 호출부가 매번 본다."""
    url = f"{FAPI}{path}?{_sign(params, secret, offset)}"
    try:
        async with session.request(method, url, headers={"X-MBX-APIKEY": key}) as response:
            payload = await response.json()
            if response.status != 200:
                return {"__error__": f"{response.status} {payload.get('msg', payload)}"}
            return payload
    except Exception as exc:  # noqa: BLE001 -- 네트워크/TLS/JSON 전부 같은 취급
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def executed_qty(order: dict) -> float:
    return float(order.get("executedQty") or 0.0)


async def current_leverage(session, symbol: str, key: str, secret: str, offset: int) -> int | None:
    """지금 걸린 심볼 레버리지. 못 읽으면 None -- 그때는 «모른다»이지 «맞다»가 아니다."""
    r = await signed(session, "GET", "/fapi/v2/positionRisk", {"symbol": symbol},
                     key, secret, offset)
    if "__error__" in r or not isinstance(r, list) or not r:
        return None
    try:
        return int(float(r[0].get("leverage") or 0)) or None
    except (TypeError, ValueError):
        return None


async def ensure_leverage(session, symbol: str, target: int, key: str, secret: str,
                          offset: int) -> dict:
    """심볼 레버리지를 target 으로 맞춘다. **주문 파라미터가 아니라 계정 설정**이라 별도 호출이다.

    이미 같으면 아무것도 안 보낸다 -- 같은 값을 다시 써도 거래소는 받지만, 보내지 않는 편이
    실패 지점을 하나 줄인다.

    🔴**내리는 쪽은 거부될 수 있다.** 레버리지를 낮추면 기존 포지션의 초기증거금 요건이 올라가고
    (명목/레버리지), 가용잔고를 넘으면 거래소가 -4028 류로 막는다. 실패를 삼키면 «걸었다고
    생각했는데 안 걸린» 상태가 되므로 결과를 그대로 돌려준다.
    """
    if not target or target <= 0:
        return {"changed": False, "reason": "목표값 없음"}
    now = await current_leverage(session, symbol, key, secret, offset)
    if now == target:
        return {"changed": False, "from": now, "to": target, "reason": "이미 같음"}
    r = await signed(session, "POST", "/fapi/v1/leverage",
                     {"symbol": symbol, "leverage": int(target)}, key, secret, offset)
    if "__error__" in r:
        return {"changed": False, "from": now, "to": target, "error": r["__error__"]}
    return {"changed": True, "from": now, "to": int(float(r.get("leverage") or target))}


async def ensure_stop(session, stop_plan: dict, key: str, secret: str, offset: int) -> dict:
    """손절을 **다시 건다**. 기존 같은 측면 STOP_MARKET 을 지우고 새로 넣는다.

    🔴다시 거는 이유: 물타기로 평단이 움직이면 옛 손절은 엉뚱한 자리에 남는다. 수량은
    `closePosition=true` 가 알아서 따라오지만 **가격은 안 따라온다**.
    취소를 재조회로 확인하지 않는 이유: 여기서는 남아도 과청산이 안 생긴다(closePosition 이
    막는다). 대신 지운 개수를 상태에 싣는다 -- 계속 늘면 취소가 안 되고 있다는 신호다.
    """
    sym, pside = stop_plan["symbol"], stop_plan["positionSide"]
    opens = await signed(session, "GET", "/fapi/v1/openOrders", {"symbol": sym},
                         key, secret, offset)
    stale = 0
    if isinstance(opens, list):
        for o in opens:
            if o.get("type") == "STOP_MARKET" and o.get("positionSide") == pside:
                await signed(session, "DELETE", "/fapi/v1/order",
                             {"symbol": sym, "orderId": o.get("orderId")}, key, secret, offset)
                stale += 1
    params = {k: v for k, v in stop_plan.items()
              if k in ("symbol", "side", "positionSide", "type", "stopPrice",
                       "closePosition", "timeInForce", "workingType")}
    r = await signed(session, "POST", "/fapi/v1/order", params, key, secret, offset)
    if "__error__" in r:
        return {"placed": False, "replaced": stale, "stop_price": stop_plan["stopPrice"],
                "error": r["__error__"]}
    return {"placed": True, "replaced": stale, "stop_price": stop_plan["stopPrice"],
            "order_id": r.get("orderId")}


async def _place_stop(session, plan: dict, state: dict, key, secret, offset) -> None:
    """체결 뒤 손절을 건다. **실패해도 진입을 되돌리지 않는다**(이미 체결됐다) -- 대신 상태에
    실어 화면이 «손절 없음»을 크게 말하게 한다. 무방비 포지션은 조용하면 안 된다.

    🔴**모든 종료 경로에서 불려야 한다**(2026-09-13 감사). 예전에는 전량 메이커 체결과
    테이커 성공에서만 불렀다 -- peg 가 일부만 체결된 뒤 폴백 시장가가 에러나면 실제 포지션이
    남는데 손절이 안 걸렸고, `state["stop"]` 이 아예 없어서 **화면 경고도 안 떴다**.
    체결이 0 이면 걸 포지션이 없으므로 그 사실을 기록만 한다(경고 아님)."""
    if not (float(state.get("filled") or 0.0) > 0):
        state["stop"] = {"placed": False, "no_position": True, "reason": "체결 없음"}
        return
    sp = plan.get("stop_plan")
    if not sp:
        state["stop"] = {"placed": False, "reason": "손절 계획 없음"}
        return
    state["stop"] = await ensure_stop(session, sp, key, secret, offset)


async def run_entry(session, plan: dict, state: dict) -> dict:
    """peg 를 걸고 **호가가 달아나면 따라가며 다시 걸다가**, 마감까지 남은 수량만 테이커로
    넘긴다. state 를 제자리에서 갱신한다(프런트가 /api/manual-entry/status 로 같은 dict 를
    읽는다). 리페그를 붙인 근거는 모듈 설명 참조(2026-09-15)."""
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and secret):
        state.update(phase="error", error="API 키가 없습니다", done_at=now_iso())
        return state
    offset = await _clock_offset(session)
    common = {"symbol": plan["symbol"], "side": plan["side"], "positionSide": plan["positionSide"]}

    # 거래소 레버리지를 처방값으로 맞춘다(2026-09-13). 위험이 아니라 **상한을 거래소에 새기는**
    # 값이라, 안 걸면 천장이 순자산의 30배(=정책의 7배)로 남는다.
    # 실패해도 **주문은 보낸다**. 수량이 이미 모델 상한에 잘려 있어 이번 주문의 위험은
    # 레버리지와 무관하고(교차 마진), 여기서 막으면 정작 들어가야 할 때 못 들어간다.
    # 대신 조용히 넘기지 않는다 -- 상태에 실어 화면이 «천장이 그대로입니다»를 말하게 한다.
    lev = await ensure_leverage(session, plan["symbol"], int(plan.get("target_leverage") or 0),
                                key, secret, offset)
    state["leverage"] = lev

    total = float(plan["quantity"])
    state.update(phase="working", filled=0.0, limit_price=plan["price"], quantity=total,
                 repegs=0, deadline_sec=FALLBACK_SEC)

    # 2026-09-15: 진입도 청산과 **같은 루프**를 쓴다. 걸어두기만 하면 호가가 달아났을 때
    # 미체결로 남고(섀도우 static: 저변동 14.4% · 고변동 7.6%) 그 다리는 120초 뒤 시장가라
    # 평균 8.75~11.5bp 다 -- 체결 다리 1.98bp 의 네댓 배.
    done, price, repegs, err = await fill_maker(
        session, common=common, price=float(plan["price"]), total=total,
        deadline=time.monotonic() + FALLBACK_SEC, state=state,
        key=key, secret=secret, offset=offset)
    state.update(filled=done, repegs=repegs, limit_price=price)
    if err is not None:
        # 🔴부분체결이 남아 있을 수 있다 -- 그건 **무방비 포지션**이라 손절부터 건다.
        # (GTX 거부처럼 체결이 0 인 경로는 _place_stop 이 «포지션 없음»으로 기록한다.)
        await _place_stop(session, plan, state, key, secret, offset)
        state.update(phase=err["phase"], error=err["error"], done_at=now_iso())
        return state

    remaining = round(total - state["filled"], 8)
    if remaining <= 0:
        state.update(phase="filled_maker", taker_qty=0.0)
        await _place_stop(session, plan, state, key, secret, offset)
        state.update(done_at=now_iso())
        return state

    taker = await signed(session, "POST", "/fapi/v1/order",
                         {**common, "type": "MARKET", "quantity": remaining},
                         key, secret, offset)
    if "__error__" in taker:
        # 🔴peg 로 일부 체결됐을 수 있다 -- 그건 **무방비 포지션**이다. 손절을 걸고 끝낸다.
        state.update(phase="taker_failed", taker_qty=0.0, error=taker["__error__"])
        await _place_stop(session, plan, state, key, secret, offset)
        state.update(done_at=now_iso())
        return state
    # 체결량은 **응답에서 읽는다**. 시장가는 보통 전량이지만 «보통»을 상태에 적으면 안 된다.
    done = executed_qty(taker) or remaining
    state.update(phase="filled_taker", taker_qty=done,
                 filled=round(state["filled"] + done, 8),
                 taker_order_id=taker.get("orderId"))
    await _place_stop(session, plan, state, key, secret, offset)
    state.update(done_at=now_iso())
    return state


async def ensure_closed(session, common: dict, oid, key: str, secret: str, offset: int,
                        tries: int = 3) -> float | None:
    """주문이 **더 이상 살아있지 않음을 확인**하고 확정 체결량을 돌려준다. 확인 못 하면 None.

    🔴리페그 루프에서 이걸 건너뛰면 «취소 실패 -> 그래도 재호가» 가 되어 지정가 주문이
    겹겹이 살아남는다. 2026-09-13 모의 거래소 검증에서 실제로 **41개**가 동시에 살아 있었다
    (2.0 ETH 포지션에 82 ETH 어치 매도 주문). 취소 응답이 아니라 **재조회 상태**를 믿는다 --
    취소 직전에 체결됐을 수도 있고, 취소가 실패했는데 주문은 멀쩡할 수도 있다.
    """
    for _ in range(max(1, tries)):
        await signed(session, "DELETE", "/fapi/v1/order",
                     {**common, "orderId": oid}, key, secret, offset)
        final = await signed(session, "GET", "/fapi/v1/order",
                             {**common, "orderId": oid}, key, secret, offset)
        if "__error__" not in final and str(final.get("status") or "") in TERMINAL:
            return executed_qty(final)
        await asyncio.sleep(min(POLL_SEC, 1.0))
    return None


REPEG_MAX = 40          # 3초 폴링 × 120초면 40회가 물리적 상한. 폭주 방지용 이중 안전장치.


async def maker_price(session, order_side: str, symbol: str) -> tuple[float, float, float]:
    """지금 **메이커로 남는** 가격. 매수는 최우선 매수호가, 매도는 최우선 매도호가.
    공개 엔드포인트라 서명하지 않는다.

    ⚠️키는 **주문 측면(BUY/SELL)** 이지 포지션 방향이 아니다 -- 진입 롱과 청산 숏은 둘 다
    BUY 라 역학이 같다. 포지션 방향으로 키를 잡으면 같은 것을 둘로 다루게 된다.

    🔴호가는 **주문이 나가는 그 심볼**에서 읽는다. 여기 "ETHUSDT" 가 박혀 있었고 수동 주문은
      2026-09-19 에 ETHUSDC 로 옮겨갔다. 두 심볼은 같은 이더가 아니다 -- 실측 ETHUSDT 가
      $0.54~0.79 비싸다(5회 연속). 그래서 BUY 지정가를 «USDT 최우선 매수호가»에 걸면 그
      값이 USDC 매도호가보다 위라 즉시 체결될 주문이 되고, post-only 라 거래소가 -5022 로
      거부한다(실측 5/5). 게다가 drifted() 도 같은 값을 받아 **항상 참**이 되어, 옳게 걸린
      첫 주문마저 3초 만에 취소했다. 증상은 「체결 0 · rejected」였다."""
    async with session.get(f"{FAPI}/fapi/v1/ticker/bookTicker",
                           params={"symbol": symbol}) as response:
        book = await response.json()
    bid, ask = float(book["bidPrice"]), float(book["askPrice"])
    return (bid if order_side == "BUY" else ask), bid, ask


def drifted(order_side: str, price: float, bid: float, ask: float) -> bool:
    """내 지정가가 시장에서 떨어졌나. 매수는 최우선 매수호가가 **내 위로** 올라가면
    (=누가 나를 앞질렀으면) 그 뒤에 서게 되어 체결이 안 된다. 매도는 거울상."""
    return bid > price if order_side == "BUY" else ask < price


async def fill_maker(session, *, common: dict, price: float, total: float, deadline: float,
                     state: dict, key: str, secret: str,
                     offset: int) -> tuple[float, float, int, dict | None]:
    """GTX 로 `total` 을 채운다 — 호가가 달아나면 취소하고 **새 호가에 다시 건다**.

    반환 `(체결량, 마지막 지정가, 리페그 수, 오류|None)`. 오류가 나오면 호출부가 멈춘다
    (`phase` 는 주문 거부면 "rejected", 취소를 확인 못 했으면 "error").

    진입과 청산이 이 루프 **한 벌**을 쓴다(2026-09-15). 두 벌로 두면 한쪽만 고쳐진다.
    다른 점은 호출부에 남는다 -- 레버리지·손절(진입), 극단변동성 시장가·변동성 연동
    마감(청산), 잔량을 테이커로 넘기는 처리(양쪽).
    """
    side = str(common["side"])          # BUY/SELL -- 포지션 방향이 아니다(maker_price 주석)
    done, repegs = 0.0, 0
    while time.monotonic() < deadline and round(total - done, 8) > 0 and repegs <= REPEG_MAX:
        remaining = round(total - done, 8)
        order = await signed(session, "POST", "/fapi/v1/order",
                             {**common, "type": "LIMIT", "timeInForce": "GTX",
                              "price": round(price, 8), "quantity": remaining},
                             key, secret, offset)
        if "__error__" in order:
            # -5022 = «지금 걸면 테이커가 된다». 실패가 아니라 호가가 움직였다는 뜻이라
            # 새 호가로 다시 건다. 그 외 오류는 그대로 멈춘다 -- 몰래 테이커로 바꾸지 않는다.
            if "5022" in str(order["__error__"]) and repegs < REPEG_MAX:
                price, _, _ = await maker_price(session, side, common["symbol"])
                repegs += 1
                state.update(repegs=repegs, limit_price=price)
                continue
            return done, price, repegs, {"phase": "rejected", "error": order["__error__"]}

        oid = order.get("orderId")
        state.update(order_id=oid, limit_price=price)
        this_filled, status, need_repeg = 0.0, "", False
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_SEC)
            cur = await signed(session, "GET", "/fapi/v1/order",
                               {**common, "orderId": oid}, key, secret, offset)
            if "__error__" in cur:
                continue        # 조회 실패는 재시도 -- 주문은 거래소에 살아 있다
            this_filled = executed_qty(cur)
            status = str(cur.get("status") or "")
            state["filled"] = round(done + this_filled, 8)
            if status in TERMINAL:
                break
            price_now, bid, ask = await maker_price(session, side, common["symbol"])
            if drifted(side, price, bid, ask):
                need_repeg, price = True, price_now
                break

        if status not in TERMINAL:
            confirmed = await ensure_closed(session, common, oid, key, secret, offset)
            if confirmed is None:
                # 살아 있는지 아닌지를 모르는 채로 **또 걸면 안 된다**. 멈추고 사람에게 넘긴다.
                return done, price, repegs, {
                    "phase": "error",
                    "error": f"주문 {oid} 취소를 확인하지 못했습니다 — 거래소에서 직접 확인하세요"}
            this_filled = confirmed

        done = round(done + this_filled, 8)
        state["filled"] = done
        if not need_repeg:
            break
        repegs += 1
        state.update(repegs=repegs, limit_price=price)
    return done, price, repegs, None


async def run_exit(session, plan: dict, state: dict) -> dict:
    """메이커로 포지션을 닫는다. **진입과 달리 리페그한다.**

    왜 다른가: 진입이 미체결이면 «안 들어간 것»으로 끝나지만, 청산이 미체결이면
    **포지션이 그대로 남는다**. 섀도우 23,332legs 실측에서 걸어두기만 하는 정책은 90.4%,
    리페그는 99.2% 체결이었다 -- 그 9.6% 가 청산에서는 비용이 아니라 «못 닫음»이다.

    🔴헤지 모드라 reduceOnly 를 안 보낸다(-1106). 과청산을 막는 건 **수량뿐**이라
    주문마다 `total - done` 로 다시 계산하고 그 위로는 올리지 않는다.
    """
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and secret):
        state.update(phase="error", error="API 키가 없습니다", done_at=now_iso())
        return state
    offset = await _clock_offset(session)
    common = {"symbol": plan["symbol"], "side": plan["side"], "positionSide": plan["positionSide"]}
    pside = plan["positionSide"]
    total = float(plan["quantity"])

    # 극단 변동성이면 계획 자체가 MARKET 이다(build_exit_plan). 지정가를 걸지 않고 바로 닫는다.
    if plan.get("type") == "MARKET":
        state.update(phase="working", kind="exit", quantity=total, filled=0.0, repegs=0,
                     market_reason=plan.get("market_reason"))
        taker = await signed(session, "POST", "/fapi/v1/order",
                             {**common, "type": "MARKET", "quantity": total}, key, secret, offset)
        if "__error__" in taker:
            state.update(phase="taker_failed", taker_qty=0.0, error=taker["__error__"],
                         done_at=now_iso())
            return state
        state.update(phase="filled_taker", taker_qty=total, filled=total,
                     taker_order_id=taker.get("orderId"), done_at=now_iso())
        return state

    price = float(plan["price"])
    # 마감은 **계획이 정한다** -- 변동성에 따라 15~120초로 달라진다(build_exit_plan 주석 참조).
    # 모듈 상수를 그대로 쓰면 변동성 연동이 조용히 무력화된다.
    window = float(plan.get("fallback_after_sec") or FALLBACK_SEC)
    state.update(phase="working", kind="exit", quantity=total, filled=0.0,
                 limit_price=price, repegs=0, deadline_sec=window,
                 vol_bpm=plan.get("vol_bpm"))

    done, price, repegs, err = await fill_maker(
        session, common=common, price=price, total=total,
        deadline=time.monotonic() + window, state=state, key=key, secret=secret, offset=offset)
    if err is not None:
        state.update(phase=err["phase"], error=err["error"], filled=done, repegs=repegs,
                     done_at=now_iso())
        return state

    remaining = round(total - done, 8)
    if remaining <= 0:
        state.update(phase="filled_maker", taker_qty=0.0, repegs=repegs, done_at=now_iso())
        return state

    taker = await signed(session, "POST", "/fapi/v1/order",
                         {**common, "type": "MARKET", "quantity": remaining},
                         key, secret, offset)
    if "__error__" in taker:
        state.update(phase="taker_failed", taker_qty=0.0, repegs=repegs,
                     error=taker["__error__"], done_at=now_iso())
        return state
    state.update(phase="filled_taker", taker_qty=remaining, filled=total, repegs=repegs,
                 taker_order_id=taker.get("orderId"), done_at=now_iso())
    return state


def _self_check() -> None:
    """네트워크를 안 타는 부분만 검사한다. 주문 경로 자체는 게이트가 닫힌 채 서버에서
    미리보기로 확인하고, 실제 전송은 사용자가 게이트를 켠 뒤 소액으로 확인한다."""
    assert executed_qty({}) == 0.0
    assert executed_qty({"executedQty": "1.5"}) == 1.5
    assert executed_qty({"executedQty": None}) == 0.0
    assert "T" in now_iso() and now_iso().endswith("+00:00")
    for s in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
        assert s in TERMINAL
    assert "NEW" not in TERMINAL and "PARTIALLY_FILLED" not in TERMINAL, \
        "부분체결·대기는 종료 상태가 아니다 -- 종료로 치면 잔량을 테이커로 안 넘긴다"

    # ── 손절 (2026-09-13) ────────────────────────────────────────────────────
    from scripts.live_manual_peg_entry_20260912 import build_stop_plan
    f2 = {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}
    sp = build_stop_plan(position_side="LONG", entry_price=2521.11, filters=f2, leverage=6.0)
    sent = {k: v for k, v in sp.items()
            if k in ("symbol", "side", "positionSide", "type", "stopPrice",
                     "closePosition", "timeInForce", "workingType")}
    assert set(sent) == {"symbol", "side", "positionSide", "type", "stopPrice",
                         "closePosition", "timeInForce", "workingType"}, sent
    assert "quantity" not in sent, "closePosition 주문에 수량을 실으면 거래소가 거부한다"
    assert "reduceOnly" not in sent, "헤지 모드에서 reduceOnly 는 -1106"
    # 손절 계획이 없으면 조용히 넘어가지 않고 이유를 남긴다
    import asyncio as _a2
    st = {"filled": 1.0}
    _a2.run(_place_stop(None, {}, st, "", "", 0))
    assert st["stop"]["placed"] is False and st["stop"]["reason"], st
    assert not st["stop"].get("no_position"), "체결이 있는데 «포지션 없음»으로 빠지면 안 된다"
    # 🔴체결이 0 이면 걸 포지션이 없다 -- 경고가 아니라 사실 기록이다(화면이 구분해야 한다)
    st0 = {"filled": 0.0}
    _a2.run(_place_stop(None, {"stop_plan": sp}, st0, "", "", 0))
    assert st0["stop"]["no_position"] is True and st0["stop"]["placed"] is False, st0
    # 🔴**체결이 남을 수 있는** 두 종료 경로가 손절을 걸어야 한다(2026-09-13 감사).
    # 그 경로에서 peg 가 일부 체결돼 있으면 포지션이 무방비로 남는데, state["stop"] 이
    # 아예 없어서 화면 경고("🔴손절을 못 걸었습니다")조차 안 떴다.
    # (GTX 거부·API키 없음 경로는 체결이 0 이라 여기 해당 없다.)
    import inspect as _i
    _src = _i.getsource(run_entry)
    # 마커는 «체결이 남을 수 있는 종료 경로»다. 2026-09-15 리페그 도입으로 취소 미확인 경로가
    # fill_maker 안으로 들어가면서 진입 쪽 마커가 `if err is not None:` 로 바뀌었다.
    for _mark in ('phase="taker_failed"', "if err is not None:"):
        _blk = _src[_src.index(_mark):]
        _blk = _blk[:_blk.index("return state")]
        assert "_place_stop" in _blk, f"{_mark} 경로에 손절이 없다"

    # ── 리페그 루프 공유 (2026-09-15) ────────────────────────────────────────
    # 진입과 청산이 **같은 함수**를 쓴다. 한쪽만 고쳐지는 걸 막는 검사다.
    for _fn in (run_entry, run_exit):
        assert "fill_maker(" in _i.getsource(_fn), f"{_fn.__name__} 이 공용 루프를 안 쓴다"
    # 호가 기준은 **주문 측면**이다. 진입 롱과 청산 숏은 둘 다 BUY 라 같은 값이어야 한다.
    assert drifted("BUY", 100.0, 100.01, 100.02) is True, "매수는 최우선 매수호가가 위로 가면 밀린다"
    assert drifted("BUY", 100.0, 100.00, 100.01) is False
    assert drifted("SELL", 100.0, 99.98, 99.99) is True, "매도는 최우선 매도호가가 밑으로 가면 밀린다"
    assert drifted("SELL", 100.0, 99.99, 100.00) is False

    # ── 호가를 «주문 심볼»에서 읽는가 (2026-09-20 사고) ──────────────────────
    # maker_price 가 "ETHUSDT" 를 박아 읽는데 수동 주문은 ETHUSDC 로 나갔다. 두 심볼은
    # 실측 $0.54~0.79 벌어져 있어 BUY 지정가가 매번 상대 호가를 넘었고(5/5) post-only 가
    # 전부 -5022 로 거부됐다. drifted() 도 같은 값을 받아 항상 참이 되어 옳게 걸린 첫 주문
    # 마저 3초 만에 취소했다. 심볼 리터럴이 다시 기어들어오는 것을 여기서 막는다.
    assert "symbol" in _i.signature(maker_price).parameters, "maker_price 가 심볼을 안 받는다"
    _mp = _i.getsource(maker_price)
    _code = _mp[_mp.index('async with'):]
    assert "ETHUSD" not in _code, "maker_price 가 호가 심볼을 하드코딩했다 -- 주문 심볼을 써야 한다"
    for _fn in (run_entry, run_exit):
        _fs = _i.getsource(_fn)
        assert "maker_price(session, side)" not in _fs, f"{_fn.__name__} 이 심볼 없이 호출한다"

    # ── 레버리지 설정 (2026-09-13) ───────────────────────────────────────────
    # 네트워크를 안 타는 계약만 본다: 목표가 없으면 아무것도 안 보낸다.
    import asyncio as _a
    assert _a.run(ensure_leverage(None, "ETHUSDT", 0, "", "", 0))["changed"] is False
    assert _a.run(ensure_leverage(None, "ETHUSDT", -5, "", "", 0))["changed"] is False

    # ── 리페그 판정 ──────────────────────────────────────────────────────────
    # ⚠️2026-09-15 키가 포지션 방향 → **주문 측면**으로 바뀌었다(진입·청산 공용).
    # 롱 청산 = 매도. 내 지정가 2470.01 인데 최우선 매도호가가 2470.00 이면 누가 앞질렀다.
    assert drifted("SELL", 2470.01, 2470.00, 2470.00) is True
    assert drifted("SELL", 2470.01, 2470.00, 2470.01) is False   # 내가 아직 최우선
    assert drifted("SELL", 2470.01, 2470.50, 2470.60) is False   # 시장이 위로 -- 그대로 둔다
    # 숏 청산과 **롱 진입**은 둘 다 매수다. 거울상이라 부등호가 반대다.
    assert drifted("BUY", 2470.00, 2470.01, 2470.02) is True
    assert drifted("BUY", 2470.00, 2470.00, 2470.01) is False
    assert drifted("BUY", 2470.00, 2469.50, 2469.60) is False
    assert REPEG_MAX * POLL_SEC >= FALLBACK_SEC, \
        "리페그 상한이 마감보다 먼저 걸리면 남은 시간을 못 쓴다"

    # 계획이 마감을 정한다 -- 모듈 상수로 되돌아가면 변동성 연동이 무력화된다.
    from scripts.live_manual_peg_entry_20260912 import build_exit_plan, exit_deadline_sec
    f = {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}
    fast = build_exit_plan(position_side="LONG", position_qty=2.0, best_bid=2470.00,
                           best_ask=2470.01, filters=f, vol_bpm=17.74)
    assert fast["fallback_after_sec"] < FALLBACK_SEC, fast["fallback_after_sec"]
    assert float(fast.get("fallback_after_sec") or FALLBACK_SEC) == fast["fallback_after_sec"]
    extreme = build_exit_plan(position_side="LONG", position_qty=2.0, best_bid=2470.00,
                              best_ask=2470.01, filters=f, vol_bpm=45.0)
    assert extreme["type"] == "MARKET", "극단 변동성은 지정가를 건너뛴다"
    assert "price" not in extreme, "MARKET 계획에 price 가 있으면 run_exit 이 옛 경로를 탄다"
    calm = build_exit_plan(position_side="LONG", position_qty=2.0, best_bid=2470.00,
                           best_ask=2470.01, filters=f, vol_bpm=None)
    assert calm["fallback_after_sec"] == exit_deadline_sec(None) == 120.0
    print("통과 21/21 — 집행 보조 함수 + 손절 주문 형태 + 청산 리페그 판정 + 변동성 마감 계약 유지")


if __name__ == "__main__":
    _self_check()
