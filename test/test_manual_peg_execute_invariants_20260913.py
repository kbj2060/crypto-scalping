"""peg 진입/청산 집행의 **불변식**을 모의 거래소로 검증한다 (2026-09-13, 사용자 요청).

실돈이 오가는 경로라 «돌려보니 되더라»로는 부족하다. 여기서 고정하는 것은 셋이다:
  ① 총 체결이 계획 수량을 **절대 넘지 않는다** (헤지 모드엔 reduceOnly 가 없어 수량이 유일한 방어)
  ② 동시에 살아 있는 지정가 주문이 **둘이 되지 않는다** (취소 실패 후 재호가하면 이게 깨진다)
  ③ 어떤 경로로도 **종료 상태로 끝난다** (phase 가 working 인 채 반환되지 않는다)

네트워크를 타지 않는다 -- `signed`/`maker_price` 를 갈아끼운다.
"""
from __future__ import annotations

import asyncio
import sys
import types

import scripts.live_manual_peg_execute_20260912 as ex


class FakeExchange:
    """주문 장부를 들고 있는 최소 모의체. 실패는 `fail` 로 주입한다."""

    def __init__(self, *, fill_after=1, fail=(), partial=None):
        self.orders: dict[int, dict] = {}
        self.next_id = 1000
        self.polls: dict[int, int] = {}
        self.fill_after = fill_after          # 몇 번째 조회에서 체결시킬지
        self.fail = set(fail)                 # {"cancel", "get_after_cancel", "place"}
        self.partial = partial                # 부분체결 비율
        self.market_orders: list[dict] = []
        self.placed: list[dict] = []

    def live_limit_ids(self):
        return [i for i, o in self.orders.items()
                if o["type"] == "LIMIT" and o["status"] not in ex.TERMINAL]

    async def signed(self, session, method, path, params, key, secret, offset):
        if path == "/fapi/v1/order" and method == "POST":
            if params.get("type") == "MARKET":
                self.market_orders.append(dict(params))
                return {"orderId": -1, "status": "FILLED",
                        "executedQty": str(params["quantity"])}
            if "place" in self.fail:
                return {"__error__": "500 boom"}
            self.next_id += 1
            self.orders[self.next_id] = {"type": "LIMIT", "status": "NEW", "filled": 0.0,
                                         "qty": float(params["quantity"])}
            self.placed.append(dict(params))
            self.polls[self.next_id] = 0
            return {"orderId": self.next_id, "status": "NEW", "executedQty": "0"}
        if path == "/fapi/v1/order" and method == "GET":
            oid = params.get("orderId")
            o = self.orders.get(oid)
            if o is None:
                return {"__error__": "400 unknown order"}
            if o["status"] == "CANCELED_PENDING_READ" and "get_after_cancel" in self.fail:
                return {"__error__": "500 read fail"}
            self.polls[oid] += 1
            if o["status"] == "NEW" and self.polls[oid] >= self.fill_after:
                if self.partial:
                    o["filled"] = round(o["qty"] * self.partial, 8)
                else:
                    o["filled"] = o["qty"]; o["status"] = "FILLED"
            return {"orderId": oid, "status": o["status"], "executedQty": str(o["filled"])}
        if path == "/fapi/v1/order" and method == "DELETE":
            oid = params.get("orderId")
            if "cancel" in self.fail:
                return {"__error__": "500 cancel fail"}
            o = self.orders.get(oid)
            if o and o["status"] == "NEW":
                o["status"] = "CANCELED"
            return {"orderId": oid, "status": "CANCELED"}
        return {"__error__": f"unexpected {method} {path}"}


def run(plan, fx, *, drift=False, entry=False):
    state: dict = {}
    ex.signed = fx.signed
    ex.POLL_SEC = 0.0
    if drift:
        async def mp(session, side):
            return (2470.02, 2470.00, 2470.00)      # 늘 내 밑으로 -- 매번 리페그 유발
        ex.maker_price = mp
        ex.drifted = lambda *a: True
    else:
        async def mp(session, side):
            return (float(plan.get("price", 2470.01)), 2470.00, 2470.01)
        ex.maker_price = mp
        ex.drifted = lambda *a: False
    fn = ex.run_entry if entry else ex.run_exit
    asyncio.run(fn(None, plan, state))
    return state


BASE = {"symbol": "ETHUSDT", "side": "SELL", "positionSide": "LONG", "type": "LIMIT",
        "timeInForce": "GTX", "price": 2470.01, "quantity": 2.0,
        "fallback_after_sec": 2.0}


def check(name, state, fx, total):
    filled = float(state.get("filled") or 0.0)
    taker = float(state.get("taker_qty") or 0.0)
    placed_qty = sum(float(p["quantity"]) for p in fx.placed) + \
        sum(float(p["quantity"]) for p in fx.market_orders)
    assert filled <= total + 1e-9, f"{name}: 체결 {filled} > 계획 {total}"
    # 거래소가 취소를 거부하면 **하나**는 남을 수 있다 -- 그건 우리가 못 막는다.
    # 막아야 하는 건 «못 막은 채로 또 거는 것»이다. 그래서 상한은 1 이고, 그때는 멈춰야 한다.
    live = fx.live_limit_ids()
    assert len(live) <= 1, f"{name}: 살아남은 지정가 주문 {live} -- 취소 실패 후에도 재호가했다"
    if live:
        assert state.get("phase") == "error", \
            f"{name}: 주문이 살아 있는데 phase={state.get('phase')} -- 사람에게 넘겨야 한다"
    assert state.get("phase") in ("filled_maker", "filled_taker", "rejected", "taker_failed",
                                 "error"), f"{name}: 종료 안 됨 phase={state.get('phase')}"
    return filled, taker, placed_qty


def main() -> int:
    import os
    os.environ.setdefault("BINANCE_API_KEY", "k"); os.environ.setdefault("BINANCE_SECRET_KEY", "s")
    async def no_off(session): return 0
    ex._clock_offset = no_off
    fails = []

    # 1) 정상 체결
    fx = FakeExchange(fill_after=1)
    st = run(dict(BASE), fx)
    check("정상", st, fx, 2.0)
    assert st["phase"] == "filled_maker" and abs(st["filled"] - 2.0) < 1e-9, st

    # 2) 계속 드리프트 -> 리페그 반복 -> 마감 후 테이커
    fx = FakeExchange(fill_after=99)
    st = run(dict(BASE), fx, drift=True)
    try:
        check("리페그", st, fx, 2.0)
    except AssertionError as e:
        fails.append(str(e))

    # 3) 🔴취소가 실패하는데 재호가한다 -> 지정가 주문이 둘 살아있게 되는가
    fx = FakeExchange(fill_after=99, fail=("cancel",))
    st = run(dict(BASE), fx, drift=True)
    try:
        check("취소실패+리페그", st, fx, 2.0)
    except AssertionError as e:
        fails.append(str(e))

    # 4) 부분체결 후 드리프트 -> 잔량만 다시 걸어야 하고 총합이 계획을 넘으면 안 된다
    fx = FakeExchange(fill_after=1, partial=0.5)
    st = run(dict(BASE), fx, drift=True)
    try:
        f, t, placed = check("부분체결+리페그", st, fx, 2.0)
        assert placed <= 2.0 * (ex.REPEG_MAX + 2), f"주문 총량 폭주 {placed}"
    except AssertionError as e:
        fails.append(str(e))

    # 5) 진입 경로 대조 -- 같은 취소 실패에서 어떻게 끝나나
    fx = FakeExchange(fill_after=99, fail=("cancel",))
    ep = {**BASE, "side": "BUY", "positionSide": "LONG"}
    st = run(ep, fx, entry=True)
    try:
        check("진입/취소실패", st, fx, 2.0)
    except AssertionError as e:
        fails.append("진입: " + str(e))

    # 6) 주문 접수 자체가 실패 -> 조용히 재시도하지 않고 멈춰야 한다
    fx = FakeExchange(fill_after=99, fail=("place",))
    st = run(dict(BASE), fx)
    try:
        check("접수실패", st, fx, 2.0)
        assert st["phase"] == "rejected", st
        assert len(fx.placed) == 0 and len(fx.market_orders) == 0, "실패했는데 뭔가 나갔다"
    except AssertionError as e:
        fails.append(str(e))

    # 7) 취소 후 재조회가 실패 -> 체결량을 모르는 채로 잔량을 계산하면 안 된다
    fx = FakeExchange(fill_after=99, fail=("get_after_cancel",))
    st = run(dict(BASE), fx, drift=True)
    try:
        check("재조회실패", st, fx, 2.0)
    except AssertionError as e:
        fails.append(str(e))

    # 8) 부분 청산(비율) 계획도 총량을 넘지 않는다
    fx = FakeExchange(fill_after=1)
    half = {**BASE, "quantity": 1.0}
    st = run(half, fx)
    try:
        f, t, placed = check("부분청산", st, fx, 1.0)
        assert placed <= 1.0 + 1e-9, f"계획 1.0 인데 {placed} 나갔다"
    except AssertionError as e:
        fails.append(str(e))

    # 9) 시장가 계획(극단 변동성)은 지정가를 아예 안 건다
    fx = FakeExchange(fill_after=1)
    mk = {k: v for k, v in BASE.items() if k not in ("price", "timeInForce")}
    mk["type"] = "MARKET"
    st = run(mk, fx)
    try:
        check("시장가계획", st, fx, 2.0)
        assert len(fx.placed) == 0, "MARKET 계획인데 지정가가 나갔다"
        assert len(fx.market_orders) == 1 and st["phase"] == "filled_taker", st
    except AssertionError as e:
        fails.append(str(e))

    if fails:
        print(f"🔴 불변식 위반 {len(fails)}건")
        for f in fails:
            print("   -", f)
        return 1
    print("통과 9/9 — 과청산 없음 · 중복 주문 없음 · 항상 종료 · 실패시 멈춤")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
