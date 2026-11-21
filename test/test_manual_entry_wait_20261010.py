"""진입 «기다리기»(2026-10-10)가 있던 자리 -- 🔴2026-11-21 사용자 «sltp 만 쓴다»로 기다리기(mode=wait · run_entry_wait)를 지워 그 시험(①~⑥·⑧)은 지웠다.
남은 것 = 지정가·감시 대기도 쓰는 재시작 정리: ⑦ 표식(dbwt) 주문만 지운다 · 기동 훅이 결과를 대기 칸에 남긴다. 🔴네트워크 0(signed·시계 가짜).
실행: python -m pytest -q test/test_manual_entry_wait_20261010.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402

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


class WaitRunnerTest(unittest.TestCase):
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

    def test_restart_hook_records_cleanup(self) -> None:
        """재시작 기동 훅이 고아 정리를 부르고 결과를 상태 칸(wait)에 남긴다 -- 화면이 «재시작으로 취소»를 말할 수 있게."""
        from aiohttp.test_utils import TestClient, TestServer
        from dashboard import server
        from dashboard import orders
        from test_manual_entry_preview_smoke_20260913 import _isolated_dirs
        called = []

        async def fake_cleanup(_s, symbols):
            called.append(symbols)
            return {"cancelled": [{"symbol": "ETHUSDC", "order_id": 1, "side": "LONG", "filled": 0.0, "qty": 1.0}], "errors": []}

        async def exercise():
            with mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.object(orders, "cancel_wait_orphans", fake_cleanup):
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
            self.assertIn(orders.MANUAL_EXEC_SYMBOLS["eth"], called[0])   # 주문 심볼(서버 .env 는 ETHUSDC)

        with _isolated_dirs():
            asyncio.run(exercise())

if __name__ == "__main__":
    unittest.main()
