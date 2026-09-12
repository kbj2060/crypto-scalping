"""대시보드 수동 진입 — 실주문 집행 (2026-09-12, 2단계).

`live_manual_peg_entry_20260912.py` 가 **무엇을 보낼지**(순수 함수, 네트워크 없음)를 정하고
이 파일이 **실제로 보낸다**. 둘을 가른 이유: 실돈이 오가는 코드는 격리해 두는 편이 읽기도
검사하기도 쉽고, 1단계 미리보기는 이 파일을 아예 import 하지 않아도 돌아간다.

정책(사용자 선택 b): peg post-only(GTX) 로 걸고 FALLBACK_SEC 까지 지켜본 뒤
**남은 수량만** 테이커로 넘긴다. 실측 1,340legs 에서 peg 2.76bp/leg, 폴백분 14.82bp.

리페그는 **하지 않는다**.
  ponytail: 호가가 달아나면 체결이 안 되고 그대로 테이커로 간다. 지금 사용자가 손으로 넣는
  방식이 어차피 테이커이므로 «되면 2.2bp 이득, 안 되면 현행과 동일»이라 순수 개선이다.
  리페그를 넣으면 체결률은 오르지만 취소·재호가 경쟁 상태가 생긴다 -- 실주문이 안정되고
  원장이 쌓인 뒤에 올린다.
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
    print("통과 4/4 — 집행 보조 함수 계약 유지")


if __name__ == "__main__":
    _self_check()
