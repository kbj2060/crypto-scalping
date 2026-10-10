"""진입 «기다리기»(2026-10-10) -- 모의 거래소로 집행 경로를 끝까지. 🔴네트워크 0(signed·maker_price·시계 전부 가짜).

① 앞질러도 다시 걸지 않는다(주문 1건) → 마감 뒤 잔량 시장가  ② 취소 = 시장가 0  ③ «지금 시장가» = 잔량 시장가
④ 처음 -5022 면 새 호가로 다시 건다(허용된 유일한 재설정)  ⑤ 거래소에서 사람이 취소 = 시장가 0  ⑥ 첫 체결에 on_fill 한 번
⑦ 서버 재시작: 표식(dbwt) 주문만 지운다  ⑧ 서버 경로: submit mode=wait → 별도 상태 · 대기 중 청산 가능 · 새 진입 409 · 취소 action.
실행: python -m pytest -q test/test_manual_entry_wait_20261010.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.live_manual_peg_execute_20260912 as ex  # noqa: E402

PLAN = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "price": 2500.0, "quantity": 1.0, "target_leverage": 0}


class Fake:
    def __init__(self, fill_at: int = 10 ** 9, partial: float = 0.0, reject_first: int = 0, cancel_at: int | None = None):
        self.limits, self.markets, self.deletes, self.polls = [], [], [], 0
        self.fill_at, self.partial, self.reject_first, self.cancel_at = fill_at, partial, reject_first, cancel_at
        self.status, self.filled = "NEW", 0.0
        self.open_orders = []

    async def signed(self, _s, method, path, params, *_a):
        if path == "/fapi/v1/order" and method == "POST":
            if params["type"] == "MARKET":
                self.markets.append(params)
                return {"orderId": 9, "status": "FILLED", "executedQty": str(params["quantity"])}
            if self.reject_first > 0:
                self.reject_first -= 1
                return {"__error__": "400 [-5022] Due to the order could not be executed as maker"}
            self.limits.append(params)
            return {"orderId": 7, "status": "NEW", "executedQty": "0"}
        if path == "/fapi/v1/order" and method == "GET":
            self.polls += 1
            if self.polls == 1 and self.partial:
                self.filled = self.partial
            if self.polls >= self.fill_at:
                self.status, self.filled = "FILLED", float(PLAN["quantity"])
            if self.cancel_at is not None and self.polls >= self.cancel_at and self.status == "NEW":
                self.status = "CANCELED"
            return {"orderId": 7, "status": self.status, "executedQty": str(self.filled)}
        if path == "/fapi/v1/order" and method == "DELETE":
            self.deletes.append(params)
            if self.status == "NEW":
                self.status = "CANCELED"
            self.open_orders = [o for o in self.open_orders if o["orderId"] != params["orderId"]]
            return {"orderId": params["orderId"]}
        if path == "/fapi/v1/openOrders":
            return [o for o in self.open_orders if o["symbol"] == params["symbol"]]
        return {"__error__": f"unexpected {method} {path}"}


def run(fx: Fake, wait_sec: float = 0.3, action_after: float | None = None, action: str = "cancel", on_fill=None) -> dict:
    state: dict = {}

    async def mp(_s, side, symbol):
        return (2500.5, 2500.5, 2500.6)              # 늘 내 위로 앞지른다 -- 따라가기면 매번 리페그했을 상황

    async def go():
        t = asyncio.create_task(ex.run_entry_wait(None, dict(PLAN), state, wait_sec, on_fill=on_fill))
        if action_after is not None:
            await asyncio.sleep(action_after)
            state["action"] = action
        await t

    with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "maker_price", mp), \
         mock.patch.object(ex, "POLL_SEC", 0.02), mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
         mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
        asyncio.run(go())
    return state


class WaitRunnerTest(unittest.TestCase):
    def test_no_repeg_then_market_at_deadline(self) -> None:
        fx = Fake()
        st = run(fx)
        self.assertEqual(len(fx.limits), 1, "앞질러도 다시 걸지 않는다")
        self.assertTrue(fx.limits[0]["newClientOrderId"].startswith(ex.WAIT_TAG + "L"))
        self.assertEqual(st["phase"], "filled_taker")
        self.assertEqual([m["quantity"] for m in fx.markets], [1.0])
        self.assertEqual(st["taker_reason"], "대기 마감")
        self.assertGreater(fx.polls, 3)

    def test_cancel_sends_no_market(self) -> None:
        fx = Fake(partial=0.3)
        st = run(fx, wait_sec=30, action_after=0.1, action="cancel")
        self.assertEqual(fx.markets, [])
        self.assertEqual(st["phase"], "cancelled")
        self.assertAlmostEqual(st["filled"], 0.3)
        self.assertEqual(len(fx.deletes), 1)

    def test_market_now(self) -> None:
        fx = Fake(partial=0.3)
        st = run(fx, wait_sec=30, action_after=0.1, action="market")
        self.assertEqual([m["quantity"] for m in fx.markets], [0.7])
        self.assertEqual(st["phase"], "filled_taker")
        self.assertEqual(st["taker_reason"], "지금 시장가(사용자)")

    def test_post_only_reject_retries_with_new_price(self) -> None:
        fx = Fake(fill_at=2, reject_first=2)
        st = run(fx)
        self.assertEqual(len(fx.limits), 1)
        self.assertEqual(st["repegs"], 2)
        self.assertEqual(fx.limits[0]["price"], 2500.5)
        self.assertEqual(st["phase"], "filled_maker")
        self.assertEqual(fx.markets, [])

    def test_exchange_side_cancel_is_cancel(self) -> None:
        fx = Fake(cancel_at=2)
        st = run(fx, wait_sec=30)
        self.assertEqual(st["phase"], "cancelled")
        self.assertEqual(fx.markets, [])

    def test_on_fill_once(self) -> None:
        fx, calls = Fake(partial=0.2), []

        async def cb():
            calls.append(1)
        run(fx, wait_sec=0.2, on_fill=cb)
        self.assertEqual(calls, [1])

    def test_restart_cleans_only_tagged(self) -> None:
        fx = Fake()
        fx.open_orders = [{"symbol": "ETHUSDC", "orderId": 1, "clientOrderId": "dbwtL123", "positionSide": "LONG", "executedQty": "0.1", "origQty": "1"},
                          {"symbol": "ETHUSDC", "orderId": 2, "clientOrderId": "dbtpL9", "positionSide": "LONG"},
                          {"symbol": "ETHUSDC", "orderId": 3, "clientOrderId": "aos_user", "positionSide": "SHORT"}]

        async def go():
            return await ex.cancel_wait_orphans(None, ["ETHUSDC", "SOLUSDC"])
        with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            r = asyncio.run(go())
        self.assertEqual([c["order_id"] for c in r["cancelled"]], [1])
        self.assertEqual([o["orderId"] for o in fx.open_orders], [2, 3])      # 익절·사용자 주문은 그대로


class WaitServerTest(unittest.TestCase):
    """서버 경로(가짜 집행): 기다리기는 별도 상태라 청산을 막지 않는다 · 대기 중 새 진입은 409 · 취소 action 이 전달된다."""

    def test_server_flow(self) -> None:
        from aiohttp.test_utils import TestClient, TestServer
        from offline_app import offline_app
        from dashboard import server
        from test_manual_entry_preview_smoke_20260913 import _isolated_dirs
        from test_rec_rule_entry_20261009 import account, long_pos
        acct = account([long_pos(0.2, 2500.0)])
        seen: dict = {}

        async def fake_wait(_s, plan, state, wait_sec, on_fill=None):
            seen["wait_sec"] = wait_sec
            state.update(phase="working", mode="wait", quantity=plan["quantity"], filled=0.0)
            while not state.get("action"):
                await asyncio.sleep(0.01)
            seen["action"] = state["action"]
            state.update(phase="cancelled")
            return state

        async def fake_exit(_s, plan, state):
            state.update(phase="filled_maker", quantity=plan["quantity"], filled=plan["quantity"])
            return state

        async def fake_account(*_a, **_k):
            return acct

        async def exercise():
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "run_entry_wait", fake_wait), \
                 mock.patch.object(server, "run_exit", fake_exit), \
                 mock.patch.object(server, "run_entry", mock.AsyncMock(side_effect=AssertionError("따라가기로 새면 안 된다"))), \
                 mock.patch.object(server, "place_bracket", mock.AsyncMock(side_effect=AssertionError("주문 금지"))):
                c = TestClient(TestServer(offline_app(server)))
                await c.start_server()
                try:
                    self.assertEqual((await c.post("/api/manual-entry/submit?side=LONG&confirm=1&pct=5&lev=20&sltp=0&mode=wait&wait_min=30")).status, 400)
                    b = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&pct=5&lev=20&sltp=0&mode=wait&wait_min=60")).json()
                    self.assertTrue(b.get("ok"), b)
                    await asyncio.sleep(0.05)
                    self.assertEqual(seen["wait_sec"], 3600)
                    st = await (await c.get("/api/manual-entry/status")).json()
                    self.assertEqual(st["wait"]["phase"], "working")
                    self.assertEqual(st["state"].get("phase"), "idle")                    # 주문 칸(청산용)은 비어 있다
                    r = await c.post("/api/manual-entry/submit?side=LONG&confirm=1&pct=5&lev=20&sltp=0")
                    self.assertEqual(r.status, 409)                                         # 대기 중 새 진입(따라가기 포함) 막음
                    r = await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=100")
                    self.assertEqual(r.status, 200, await r.text())                         # 청산은 된다
                    self.assertEqual((await c.post("/api/manual-entry/wait-action?action=market")).status, 400)   # confirm 없음
                    r = await c.post("/api/manual-entry/wait-action?action=cancel&confirm=1")
                    self.assertEqual(r.status, 200)
                    for _ in range(50):
                        await asyncio.sleep(0.01)
                        if seen.get("action"):
                            break
                    self.assertEqual(seen["action"], "cancel")
                    await asyncio.sleep(0.05)
                    self.assertEqual((await c.post("/api/manual-entry/wait-action?action=cancel&confirm=1")).status, 409)
                finally:
                    await c.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_restart_hook_records_cleanup(self) -> None:
        """재시작 기동 훅이 고아 정리를 부르고 결과를 상태 칸(wait)에 남긴다 -- 화면이 «재시작으로 취소»를 말할 수 있게."""
        from aiohttp.test_utils import TestClient, TestServer
        from dashboard import server
        from test_manual_entry_preview_smoke_20260913 import _isolated_dirs
        called = []

        async def fake_cleanup(_s, symbols):
            called.append(symbols)
            return {"cancelled": [{"symbol": "ETHUSDC", "order_id": 1, "side": "LONG", "filled": 0.0, "qty": 1.0}], "errors": []}

        async def exercise():
            with mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "cancel_wait_orphans", fake_cleanup):
                app = server.make_app()
                keep = {"start_http_session", "start_wait_cleanup", "stop_http_session"}
                for sig in (app.on_startup, app.on_cleanup):
                    sig[:] = [f for f in sig if getattr(f, "__name__", "") in keep]
                c = TestClient(TestServer(app))
                await c.start_server()
                try:
                    await asyncio.sleep(0.05)
                    w = (await (await c.get("/api/manual-entry/status")).json())["wait"]
                    self.assertEqual(w["phase"], "restart_cancelled")
                    self.assertEqual(w["cancelled"][0]["order_id"], 1)
                finally:
                    await c.close()
            self.assertIn(server.MANUAL_EXEC_SYMBOLS["eth"], called[0])   # 주문 심볼(서버 .env 는 ETHUSDC)

        with _isolated_dirs():
            asyncio.run(exercise())

if __name__ == "__main__":
    unittest.main()
