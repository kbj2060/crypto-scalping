"""대시보드 수동 진입 — 실주문 집행 (2026-09-12, 2단계).

`live_manual_peg_entry_20260912.py` 가 **무엇을 보낼지**(순수 함수, 네트워크 없음)를 정하고
이 파일이 **실제로 보낸다**. 둘을 가른 이유: 실돈이 오가는 코드는 격리해 두는 편이 읽기도
검사하기도 쉽고, 1단계 미리보기는 이 파일을 아예 import 하지 않아도 돌아간다.

정책(사용자 선택 b): peg post-only(GTX) 로 걸고 FALLBACK_SEC 까지 지켜본 뒤
**남은 수량만** 테이커로 넘긴다. 실측 1,340legs 에서 peg 2.76bp/leg, 폴백분 14.82bp.

**진입은 리페그하지 않고, 청산은 리페그한다.** 같은 정책이 아닌 이유가 있다.
  진입(run_entry): 호가가 달아나면 체결이 안 되고 그대로 테이커로 간다. 사용자가 손으로
    넣던 방식이 어차피 테이커라 «되면 2.2bp 이득, 안 되면 현행과 동일»이다 -- 순수 개선.
  청산(run_exit, 2026-09-13 추가): 미체결이 «현행과 동일»이 아니다. **포지션이 남는다.**
    섀도우 23,332legs 실측에서 걸어두기만 하면 90.4%, 리페그하면 99.2% 체결이었고
    그 차이 9.6% 가 청산에서는 비용이 아니라 «못 닫음»이다. 그래서 취소·재호가 경쟁
    상태를 감수한다 -- 대신 REPEG_MAX 로 폭주를 막고, 취소 응답이 아니라 재조회 값을 믿는다.
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


async def run_entry(session, plan: dict, state: dict) -> dict:
    """peg 를 걸고 지켜보다가 남은 수량만 테이커로 넘긴다. state 를 제자리에서 갱신한다
    (프런트가 /api/manual-entry/status 로 같은 dict 를 읽는다)."""
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and secret):
        state.update(phase="error", error="API 키가 없습니다", done_at=now_iso())
        return state
    offset = await _clock_offset(session)
    common = {"symbol": plan["symbol"], "side": plan["side"], "positionSide": plan["positionSide"]}

    order = await signed(session, "POST", "/fapi/v1/order",
                         {**common, "type": "LIMIT", "timeInForce": "GTX",
                          "price": plan["price"], "quantity": plan["quantity"]},
                         key, secret, offset)
    if "__error__" in order:
        # GTX 는 «지금 걸면 테이커가 된다» 싶으면 거부한다(-5022). 실패가 아니라 정상 동작이라
        # 사용자에게 그대로 보여주고 끝낸다 -- 몰래 테이커로 바꿔 넣지 않는다.
        state.update(phase="rejected", error=order["__error__"], done_at=now_iso())
        return state
    state.update(phase="working", order_id=order.get("orderId"), filled=0.0,
                 limit_price=plan["price"], quantity=plan["quantity"])

    deadline = time.monotonic() + FALLBACK_SEC
    status = ""
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_SEC)
        cur = await signed(session, "GET", "/fapi/v1/order",
                           {**common, "orderId": state["order_id"]}, key, secret, offset)
        if "__error__" in cur:
            continue           # 조회 실패는 재시도한다 -- 주문 자체는 거래소에 살아 있다
        state["filled"] = executed_qty(cur)
        status = str(cur.get("status") or "")
        if status in TERMINAL:
            break

    if status not in TERMINAL:
        cancel = await signed(session, "DELETE", "/fapi/v1/order",
                              {**common, "orderId": state["order_id"]}, key, secret, offset)
        # 취소 직전에 체결됐을 수 있다. 취소 응답이 아니라 **다시 조회한 값**을 믿는다.
        final = await signed(session, "GET", "/fapi/v1/order",
                             {**common, "orderId": state["order_id"]}, key, secret, offset)
        if "__error__" not in final:
            state["filled"] = executed_qty(final)
        elif "__error__" in cancel:
            state.update(phase="error", error=f"취소 실패: {cancel['__error__']}", done_at=now_iso())
            return state

    remaining = round(float(plan["quantity"]) - state["filled"], 8)
    if remaining <= 0:
        state.update(phase="filled_maker", taker_qty=0.0, done_at=now_iso())
        return state

    taker = await signed(session, "POST", "/fapi/v1/order",
                         {**common, "type": "MARKET", "quantity": remaining},
                         key, secret, offset)
    if "__error__" in taker:
        state.update(phase="taker_failed", taker_qty=0.0,
                     error=taker["__error__"], done_at=now_iso())
        return state
    state.update(phase="filled_taker", taker_qty=remaining, filled=float(plan["quantity"]),
                 taker_order_id=taker.get("orderId"), done_at=now_iso())
    return state


REPEG_MAX = 40          # 3초 폴링 × 120초면 40회가 물리적 상한. 폭주 방지용 이중 안전장치.


async def maker_price(session, position_side: str) -> tuple[float, float, float]:
    """지금 **메이커로 남는** 가격. 롱을 닫으면 SELL 이라 최우선 매도호가, 숏이면 매수호가.
    공개 엔드포인트라 서명하지 않는다."""
    async with session.get(f"{FAPI}/fapi/v1/ticker/bookTicker",
                           params={"symbol": "ETHUSDT"}) as response:
        book = await response.json()
    bid, ask = float(book["bidPrice"]), float(book["askPrice"])
    return (ask if position_side == "LONG" else bid), bid, ask


def drifted(position_side: str, price: float, bid: float, ask: float) -> bool:
    """내 지정가가 시장에서 떨어졌나. 롱 청산(매도)은 최우선 매도호가가 **내 밑으로** 내려가면
    (=누가 나를 앞질렀으면) 체결이 안 된다. 숏 청산은 거울상."""
    return ask < price if position_side == "LONG" else bid > price


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
    price = float(plan["price"])
    done = 0.0
    repegs = 0
    deadline = time.monotonic() + FALLBACK_SEC
    state.update(phase="working", kind="exit", quantity=total, filled=0.0,
                 limit_price=price, repegs=0)

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
                price, _, _ = await maker_price(session, pside)
                repegs += 1
                state.update(repegs=repegs, limit_price=price)
                continue
            state.update(phase="rejected", error=order["__error__"], filled=done, done_at=now_iso())
            return state

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
            price_now, bid, ask = await maker_price(session, pside)
            if drifted(pside, price, bid, ask):
                need_repeg, price = True, price_now
                break

        if status not in TERMINAL:
            await signed(session, "DELETE", "/fapi/v1/order",
                         {**common, "orderId": oid}, key, secret, offset)
            # 취소 직전에 체결됐을 수 있다. 취소 응답이 아니라 **다시 조회한 값**을 믿는다.
            final = await signed(session, "GET", "/fapi/v1/order",
                                 {**common, "orderId": oid}, key, secret, offset)
            if "__error__" not in final:
                this_filled = executed_qty(final)

        done = round(done + this_filled, 8)
        state["filled"] = done
        if not need_repeg:
            break
        repegs += 1
        state.update(repegs=repegs, limit_price=price)

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

    # ── 리페그 판정(청산) ────────────────────────────────────────────────────
    # 롱 청산 = 매도. 내 지정가 2470.01 인데 최우선 매도호가가 2470.00 이면 누가 앞질렀다.
    assert drifted("LONG", 2470.01, 2470.00, 2470.00) is True
    assert drifted("LONG", 2470.01, 2470.00, 2470.01) is False   # 내가 아직 최우선
    assert drifted("LONG", 2470.01, 2470.50, 2470.60) is False   # 시장이 위로 -- 그대로 둔다
    # 숏 청산 = 매수. 거울상이라 부등호가 반대다.
    assert drifted("SHORT", 2470.00, 2470.01, 2470.02) is True
    assert drifted("SHORT", 2470.00, 2470.00, 2470.01) is False
    assert drifted("SHORT", 2470.00, 2469.50, 2469.60) is False
    assert REPEG_MAX * POLL_SEC >= FALLBACK_SEC, \
        "리페그 상한이 마감보다 먼저 걸리면 남은 시간을 못 쓴다"
    print("통과 11/11 — 집행 보조 함수 + 청산 리페그 판정 계약 유지")


if __name__ == "__main__":
    _self_check()
