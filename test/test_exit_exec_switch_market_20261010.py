"""스위칭 = 즉시 체결 · 청산 «지정가/즉시» (2026-10-10 사용자 결정 «스위칭은 즉시 체결 기본, 청산은 선택지로»).

진짜 run_exit·run_entry 를 돌리고 거래소 서명 요청(signed)만 가짜로 갈아끼워 실제로 나간 주문 유형을 센다.
(a) 스위칭 = 지정가 0 · 시장가 청산 1 + 시장가 진입 1  (b) 청산 exec=market → 시장가 1·GTX 0 / 기본 → GTX 경로
(c) exec 이상값 400(미리보기·제출, 주문 0)  (d) 스위칭 시장가 청산이 부분 체결이면 반대 진입 0.
🔴실주문 0 -- signed·시계·place_bracket·margin_type 가짜, 계좌·호가는 _isolated_dirs 의 가짜 응답.
실행: python -m pytest -q test/test_exit_exec_switch_market_20261010.py
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

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

import scripts.live_manual_peg_execute_20260912 as ex  # noqa: E402
from dashboard import server  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402
from test_rec_rule_entry_20261009 import account, long_pos  # noqa: E402

SHORT = {**long_pos(2.563, 2500.0), "side": "SHORT"}
SWITCH = "/api/manual-exit/submit?side=SHORT&pct=100&confirm=1&switch_to=LONG&sw_pct=10&sw_lev=20"
EXIT = "/api/manual-exit/submit?side=SHORT&pct=100&confirm=1"


class ExitExecTest(unittest.TestCase):
    def _run(self, fn, market_frac: float = 1.0) -> dict:
        acct = account([dict(SHORT)])
        orders: list[dict] = []

        async def fake_account(*_a, **_k):
            return acct

        async def fake_signed(_s, method, path, params, *_a):     # 거래소 대신 -- 낸 주문만 적는다
            if path == "/fapi/v1/order" and method == "POST":
                orders.append(dict(params))
                q = float(params["quantity"])
                if params["type"] == "MARKET":
                    done = q * market_frac if params["positionSide"] == "SHORT" else q
                    if params["positionSide"] == "SHORT" and done >= q:
                        acct["positions"] = []
                    return {"orderId": len(orders), "status": "FILLED" if done >= q else "EXPIRED", "executedQty": str(done)}
                return {"orderId": len(orders), "status": "NEW", "executedQty": "0"}
            if path == "/fapi/v1/order" and method == "GET":              # 지정가는 첫 조회에 다 찬다(기본 경로를 빨리 끝낸다)
                acct["positions"] = []
                return {"orderId": params.get("orderId"), "status": "FILLED", "executedQty": str(orders[-1]["quantity"])}
            if path == "/fapi/v2/positionRisk":
                return [{"leverage": "20", "marginType": "cross"}]
            return {"__error__": f"가짜 거래소: {method} {path}"}

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "place_bracket", mock.AsyncMock(return_value={"placed": True})), \
                 mock.patch.object(server, "margin_type", mock.AsyncMock(return_value="cross")), \
                 mock.patch.object(ex, "signed", fake_signed), \
                 mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
                 mock.patch.object(ex, "maker_price", mock.AsyncMock(return_value=(2500.0, 2499.9, 2500.0))), \
                 mock.patch.object(ex, "POLL_SEC", 0.02), \
                 mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
                app = offline_app(server)
                app["situation_eth"]["vol_mult"] = {"sigma_bp": 100.0, "mult": 1.0, "bar": time.time()}
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())
        return {"orders": orders}

    @staticmethod
    async def _done(c) -> dict:
        for _ in range(200):
            await asyncio.sleep(0.02)
            st = (await (await c.get("/api/manual-entry/status")).json())["state"]
            sw = (st.get("switch") or {}).get("phase")
            if sw in ("done", "aborted") or (not st.get("switch") and st.get("phase") not in ("submitting", "working")):
                return st
        raise AssertionError("끝나지 않음")

    def test_a_switch_is_market_both_legs(self) -> None:
        async def fn(c):
            self.assertTrue((await (await c.post(SWITCH)).json())["ok"])
            st = await self._done(c)
            self.assertEqual(st["switch"]["phase"], "done", st)
        r = self._run(fn)
        kinds = [(o["positionSide"], o["side"], o["type"]) for o in r["orders"]]
        self.assertEqual(kinds, [("SHORT", "BUY", "MARKET"), ("LONG", "BUY", "MARKET")], kinds)
        self.assertFalse(any("timeInForce" in o or "price" in o for o in r["orders"]), "지정가(GTX)가 나갔다")

    def test_b_exit_market_vs_default(self) -> None:
        async def market(c):
            self.assertTrue((await (await c.post(EXIT + "&exec=market")).json())["ok"])
            await self._done(c)
        r = self._run(market)
        self.assertEqual([o["type"] for o in r["orders"]], ["MARKET"])

        async def default(c):
            self.assertTrue((await (await c.post(EXIT)).json())["ok"])
            await self._done(c)
        r = self._run(default)
        self.assertEqual([(o["type"], o.get("timeInForce")) for o in r["orders"]], [("LIMIT", "GTX")])

        async def preview(c):
            p = (await (await c.get("/api/manual-exit/preview?side=SHORT&pct=100&exec=market")).json())["plan"]
            self.assertEqual(p["type"], "MARKET")
            self.assertNotIn("price", p)
            self.assertIn("즉시", p["market_reason"])
            p = (await (await c.get("/api/manual-exit/preview?side=SHORT&pct=100")).json())["plan"]
            self.assertEqual((p["type"], p["timeInForce"]), ("LIMIT", "GTX"))
        self.assertEqual(self._run(preview)["orders"], [])

    def test_c_bad_exec_400(self) -> None:
        async def fn(c):
            for u in (EXIT + "&exec=taker", SWITCH + "&exec=MARKET"):
                r = await c.post(u)
                self.assertEqual(r.status, 400, u)
                self.assertEqual((await r.json())["error"], "bad_exec")
            r = await c.get("/api/manual-exit/preview?side=SHORT&pct=100&exec=x")
            self.assertEqual(r.status, 400)
        self.assertEqual(self._run(fn)["orders"], [])

    def test_d_partial_market_exit_no_entry(self) -> None:
        async def fn(c):
            self.assertTrue((await (await c.post(SWITCH)).json())["ok"])
            st = await self._done(c)
            self.assertEqual(st["switch"]["phase"], "aborted")
            self.assertIn("다 닫히지 않아", st["switch"]["error"])
        r = self._run(fn, market_frac=0.4)
        self.assertEqual([(o["positionSide"], o["type"]) for o in r["orders"]], [("SHORT", "MARKET")], "반대 진입이 나갔다")


if __name__ == "__main__":
    unittest.main()
