"""peg 진입/청산 집행의 **불변식**을 모의 거래소로 검증한다 (2026-09-13, 사용자 요청).

실돈이 오가는 경로라 «돌려보니 되더라»로는 부족하다. 여기서 고정하는 것은 셋이다:
  ① 총 체결이 계획 수량을 **절대 넘지 않는다** (헤지 모드엔 reduceOnly 가 없어 수량이 유일한 방어)
  ② 동시에 살아 있는 지정가 주문이 **둘이 되지 않는다** (취소 실패 후 재호가하면 이게 깨진다)
  ③ 어떤 경로로도 **종료 상태로 끝난다** (phase 가 working 인 채 반환되지 않는다)

네트워크를 타지 않는다 -- `signed`/`maker_price` 를 갈아끼운다.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # PYTHONPATH 없이 돌려도 저장소 뿌리를 찾는다

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402

REAL_MAKER_PRICE = ex.maker_price   # 갈아끼우기 전 원본 -- 가짜의 인자 모양을 이것과 대조한다


class FakeExchange:
    """주문 장부를 들고 있는 최소 모의체. 실패는 `fail` 로 주입한다."""

    def __init__(self, *, fill_after=1, fail=(), ok_first=0, partial=None, post_only_rejects=0):
        self.post_only_rejects = post_only_rejects   # 처음 몇 번의 지정가를 -5022(«지금 걸면 테이커»)로 거부
        self.orders: dict[int, dict] = {}
        self.next_id = 1000
        self.polls: dict[int, int] = {}
        self.fill_after = fill_after          # 몇 번째 조회에서 체결시킬지
        self.fail = set(fail)                 # {"cancel", "get_after_cancel", "place"}
        self.ok_first = ok_first              # 실패 종류마다 처음 몇 번은 성공 -- 재호가 **도는 중**에 터뜨린다
        self.calls: dict[str, int] = {}
        self.partial = partial                # 부분체결 비율
        self.market_orders: list[dict] = []
        self.placed: list[dict] = []

    def _fails(self, kind):
        if kind not in self.fail:
            return False
        self.calls[kind] = self.calls.get(kind, 0) + 1
        return self.calls[kind] > self.ok_first

    def live_limit_ids(self):
        return [i for i, o in self.orders.items()
                if o["type"] == "LIMIT" and o["status"] not in ex.TERMINAL]

    def exposure(self):
        """거래소 쪽 사실: 이미 체결된 양 + 아직 체결될 수 있는 양(살아 있는 지정가 잔량)."""
        filled = sum(o["filled"] for o in self.orders.values()) + \
            sum(float(m["quantity"]) for m in self.market_orders)
        resting = sum(o["qty"] - o["filled"] for i, o in self.orders.items() if i in self.live_limit_ids())
        return filled, resting

    async def signed(self, session, method, path, params, key, secret, offset):
        if path == "/fapi/v1/order" and method == "POST":
            if params.get("type") == "MARKET":
                self.market_orders.append(dict(params))
                return {"orderId": -1, "status": "FILLED",
                        "executedQty": str(params["quantity"])}
            if self.post_only_rejects > 0:
                self.post_only_rejects -= 1
                return {"__error__": "400 [-5022] Due to the order could not be executed as maker"}
            if self._fails("place"):
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
            # 예전 조건은 아무도 안 쓰는 상태값("CANCELED_PENDING_READ")을 봐서 한 번도 안 터졌다
            if o.get("cancel_sent") and self._fails("get_after_cancel"):
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
            o = self.orders.get(oid)
            if o:
                o["cancel_sent"] = True
            if self._fails("cancel"):
                return {"__error__": "500 cancel fail"}
            if o and o["status"] == "NEW":
                o["status"] = "CANCELED"
            return {"orderId": oid, "status": "CANCELED"}
        return {"__error__": f"unexpected {method} {path}"}


def run(plan, fx, *, drift=False, entry=False):
    state: dict = {}
    ex.signed = fx.signed
    ex.POLL_SEC = 0.0
    # 🔴가짜는 진짜와 **같은 인자**를 받아야 한다. 2인자 가짜가 3인자 호출에 TypeError 를 냈고 fill_maker 의
    #   «호가 조회 실패 = 다음 조회에» 처리가 그걸 삼켜, 재호가 시나리오가 한 번도 재호가하지 않은 채 통과했다(2026-11-20 발견).
    async def mp(session, order_side, symbol):
        assert symbol == plan["symbol"], f"호가를 주문 심볼이 아닌 {symbol} 에서 읽는다"
        if drift:
            return (2470.02, 2470.00, 2470.00)      # 늘 내 밑으로 -- 매번 리페그 유발
        return (float(plan.get("price", 2470.01)), 2470.00, 2470.01)
    assert list(inspect.signature(mp).parameters) == list(inspect.signature(REAL_MAKER_PRICE).parameters), \
        "가짜 maker_price 의 인자가 집행 코드와 다르다 -- 시험이 낡았다"
    ex.maker_price = mp
    ex.drifted = lambda *a: drift
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
    # ①을 **거래소 쪽에서** 다시 센다 -- state 는 집행 코드가 스스로 적은 값이라 그것만 보면 자기 채점이다.
    real, resting = fx.exposure()
    assert real + resting <= total + 1e-6, \
        f"{name}: 거래소 체결 {real} + 살아 있는 지정가 잔량 {resting} > 계획 {total}"
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


def fail_cases(name, kind, phase, fails, *, entry=False):
    """실패 `kind` 를 재호가 n 번 **뒤에** 주입한다. n=0 은 첫 주문에서, n>0 은 재호가가 도는 중에 터진다.
    재호가 수가 정확히 n 이어야 한다 -- 0 이면 주입이 재호가 경로를 안 지난 것이다(이 시험이 낡았던 모양)."""
    for n in (0, 1, 3):
        label = f"{name}(재호가 {n}회 뒤)"
        fx = FakeExchange(fill_after=99, fail=(kind,), ok_first=n)
        st = run({**BASE, "side": "BUY"} if entry else dict(BASE), fx, drift=True, entry=entry)
        try:
            check(label, st, fx, 2.0)
            assert st["phase"] == phase, f"{label}: phase={st['phase']} (기대 {phase})"
            assert st["repegs"] == n, f"{label}: 재호가 {st['repegs']}회 (기대 {n})"
            assert len(fx.placed) == n + (kind != "place"), \
                f"{label}: 지정가 {len(fx.placed)}건 -- 실패 뒤에 또 걸었다"
            assert not fx.market_orders, f"{label}: 멈춰야 하는데 시장가가 나갔다"
            assert len(fx.live_limit_ids()) == (kind == "cancel"), f"{label}: 살아남은 지정가 {fx.live_limit_ids()}"
        except AssertionError as e:
            fails.append(str(e))


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
        assert st["repegs"] >= 1, f"리페그: 재호가 0회 -- 드리프트 시나리오가 재호가를 안 했다: {st}"
        assert st["phase"] == "filled_taker" and len(fx.market_orders) == 1, f"리페그: {st}"
    except AssertionError as e:
        fails.append(str(e))

    # 3) 🔴취소가 실패하는데 재호가한다 -> 지정가 주문이 둘 살아있게 되는가
    fail_cases("취소실패+리페그", "cancel", "error", fails)

    # 4) 부분체결 후 드리프트 -> 잔량만 다시 걸어야 하고 총합이 계획을 넘으면 안 된다
    fx = FakeExchange(fill_after=1, partial=0.5)
    st = run(dict(BASE), fx, drift=True)
    try:
        f, t, placed = check("부분체결+리페그", st, fx, 2.0)
        assert st["repegs"] >= 1, f"부분체결+리페그: 재호가 0회: {st}"
        assert placed <= 2.0 * (ex.REPEG_MAX + 2), f"주문 총량 폭주 {placed}"
    except AssertionError as e:
        fails.append(str(e))

    # 5) 진입 경로 대조 -- 같은 취소 실패에서 어떻게 끝나나
    #    (예전엔 드리프트 없이 돌려 취소까지 가지도 않았다: 99번째 조회에서 그냥 체결됐다)
    fail_cases("진입/취소실패", "cancel", "error", fails, entry=True)

    # 6) 주문 접수 자체가 실패 -> 조용히 재시도하지 않고 멈춰야 한다
    fail_cases("접수실패", "place", "rejected", fails)

    # 7) 취소 후 재조회가 실패 -> 체결량을 모르는 채로 잔량을 계산하면 안 된다
    fail_cases("재조회실패", "get_after_cancel", "error", fails)

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

    # 10) 진입도 리페그한다 (2026-09-15). 예전에는 한 번 걸고 마감까지 기다리기만 했다 --
    #     그 모양이 섀도우 static 과 같고 미체결 7.6~14.4% 가 시장가로 끝났다.
    #     여기서 고정하는 것: 리페그를 돌아도 ①계획 수량을 안 넘고 ②지정가가 둘이 안 되고
    #     ③종료 상태로 끝나며 ④적힌 체결량이 거래소 실제 체결과 같다.
    #     ④는 예전엔 «state 에 stop 칸» 이었다 -- run_entry 는 이제 손절을 안 건다(옛 3% 손절 삭제, 집행 모듈 _self_check 가
    #     run_entry 에 STOP_MARKET 이 없음을 고정). 손절은 호출부(dashboard/orders.py)가 state["filled"] > 0 을 보고 건다.
    fx = FakeExchange(fill_after=99)
    ep = {**BASE, "side": "BUY", "positionSide": "LONG"}
    st = run(ep, fx, entry=True, drift=True)
    try:
        f, tk, placed = check("진입/리페그", st, fx, 2.0)
        assert int(st.get("repegs") or 0) >= 1, f"진입이 리페그를 안 했다: {st}"
        assert placed <= 2.0 * (ex.REPEG_MAX + 2), f"주문 총량 폭주 {placed}"
        assert abs(f - fx.exposure()[0]) < 1e-6, f"적힌 체결 {f} != 거래소 체결 {fx.exposure()[0]} -- 호출부가 손절을 잘못 건다"
    except AssertionError as e:
        fails.append("진입리페그: " + str(e))

    # 11) post-only 거부(-5022)는 실패가 아니라 «호가가 움직였다» -> 새 호가를 읽어 다시 건다(2026-09-20 실사고 경로).
    #      fill_maker 가 maker_price 를 부르는 두 곳 중 나머지 하나다. 상한(REPEG_MAX)을 다 쓰면 시장가 없이 rejected 로 멈춘다.
    for k, phase in ((3, "filled_maker"), (ex.REPEG_MAX + 5, "rejected")):
        label = f"post-only 거부 {k}회"
        fx = FakeExchange(fill_after=1, post_only_rejects=k)
        st = run(dict(BASE), fx)
        try:
            check(label, st, fx, 2.0)
            assert st["phase"] == phase, f"{label}: phase={st['phase']} (기대 {phase}) {st.get('error')}"
            assert st["repegs"] == min(k, ex.REPEG_MAX), f"{label}: 재호가 {st['repegs']}회"
            assert len(fx.placed) == (phase == "filled_maker") and not fx.market_orders, \
                f"{label}: 지정가 {len(fx.placed)}건 · 시장가 {len(fx.market_orders)}건"
        except AssertionError as e:
            fails.append(str(e))

    if fails:
        print(f"🔴 불변식 위반 {len(fails)}건")
        for f in fails:
            print("   -", f)
        return 1
    print("통과 11/11 — 과청산 없음 · 중복 주문 없음 · 항상 종료 · 실패시 멈춤(재호가 0·1·3회 뒤) · 진입 리페그 · post-only 재호가")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
