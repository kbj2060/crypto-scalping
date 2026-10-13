"""자동 규칙 «첫 진입 체결 → 첫 진입가 ∓0.75% 물타기 지정가 자동»(2026-10-14 사용자 지시) -- 🔴주문 함수는 전부 가짜(실주문 금지).
① 미리보기가 물타기 가격·수량을 싣는다 ② 첫 진입이 체결되면 손절을 건 뒤 지정가를 **한 번** 건다(가격 = 손절선에서 푼 첫 진입가 × 0.9925,
수량 = 실제 체결 수량으로 다시 잰 남은 여유) · 무장 파일에 남는다 ③ 정리(cancel_bracket)는 우리 TP·물타기 지정가만 지우고 사용자 주문은 남긴다.
python -m pytest -q test/test_rule_auto_add_20261014.py"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from scripts import live_manual_peg_execute_20260912 as ex  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import REC_ADD_DIST, REC_FIRST_SHARE  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = server.MANUAL_EXEC_SYMBOLS["eth"]
L = 4 * 0.303 * 10.26 * 0.5          # 12.435 × 권장배수 0.5 = 6.218
VM = {"sigma_bp": 250.0 / math.sqrt(6), "mult": 0.5}
ACCT = {"ok": True, "balance": {"margin": 1000.0, "initial_margin": 0.0, "available": 1000.0, "unrealized": 0.0},
        "positions": [], "leverage_by_symbol": {SYM: 20.0}, "trips": []}


async def _boom(*_a, **_k):
    raise AssertionError("실주문 경로가 불렸다 -- 시험에서 주문 금지")


class AutoAddTest(unittest.TestCase):
    def _run(self, fn) -> None:
        async def fake_account(*_a, **_k):
            return ACCT

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(server, "REC_AUTO_ADD", True), \
                 mock.patch.object(server, "place_bracket", _boom), \
                 mock.patch.object(server, "place_add_order", _boom), \
                 mock.patch.object(server, "run_entry", _boom), \
                 mock.patch.object(server, "margin_type", mock.AsyncMock(return_value="cross")):
                app = offline_app(server)
                app["situation_eth"]["vol_mult"] = {"bar": time.time(), **VM}
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_preview_carries_add_price_and_size(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            r = p["rule"]
            self.assertTrue(r["first"])
            anchor = r["sl_price"] / 0.97
            self.assertAlmostEqual(r["add"]["price"], math.floor(anchor * (1 - REC_ADD_DIST) / 0.01 + 1e-9) * 0.01, places=6)
            self.assertAlmostEqual(r["add"]["notional"], (1 - REC_FIRST_SHARE) * 1000 * L, delta=6.0)   # 남은 2/3
        self._run(fn)

    def test_fill_places_one_add_after_stop_and_arms_it(self) -> None:
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(("bracket", b.get("backstop_price")))
            return {"placed": True, "backstop": {"placed": True, "price": b.get("backstop_price")}}

        async def fake_add(_s, symbol, pside, price, qty):
            calls.append(("add", price, qty))
            return {"placed": True, "price": price, "qty": qty, "order_id": 1}

        async def fn(c):
            with mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "run_entry", fake_entry), \
                 mock.patch.object(server, "place_bracket", fake_bracket), \
                 mock.patch.object(server, "place_add_order", fake_add):
                body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=c")).json()
                self.assertTrue(body.get("ok"), body)
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if any(k[0] == "add" for k in calls):
                        break
                st = (await (await c.get("/api/manual-entry/status")).json())["state"]
            self.assertEqual([k[0] for k in calls], ["bracket", "add"], calls)          # 손절 먼저, 물타기 지정가 한 번
            sl = calls[0][1]
            self.assertAlmostEqual(calls[1][1], math.floor(sl / 0.97 * (1 - REC_ADD_DIST) / 0.01 + 1e-9) * 0.01, places=6)
            self.assertAlmostEqual(calls[1][1] * calls[1][2], (1 - REC_FIRST_SHARE) * 1000 * L, delta=6.0)
            self.assertTrue(st["add_order"]["placed"], st)
            armed = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
            self.assertTrue(next(iter(armed.values()))["add_order"]["placed"], armed)
        self._run(fn)


    def test_no_add_when_stop_failed(self) -> None:
        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def failed_bracket(_s, b, symbol, pside):
            return {"placed": False, "backstop": {"placed": False, "error": "-4120"}}

        async def fn(c):
            with mock.patch.object(server, "exec_enabled", lambda: True), mock.patch.object(server, "run_entry", fake_entry), \
                 mock.patch.object(server, "place_bracket", failed_bracket):          # place_add_order 는 _boom 그대로 -- 불리면 실패
                self.assertTrue((await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=c")).json()).get("ok"))
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    st = (await (await c.get("/api/manual-entry/status")).json())["state"]
                    if (st.get("add_order") or {}).get("reason"):
                        break
            self.assertFalse(st["add_order"]["placed"])
            self.assertIn("손절 스탑", st["add_order"]["reason"])
        self._run(fn)


class CancelKeepsUserOrdersTest(unittest.TestCase):
    def test_cancel_bracket_deletes_only_ours(self) -> None:
        deleted = []

        async def fake_signed(_s, method, path, params, *_a, **_k):
            if method == "GET" and path == "/fapi/v1/openOrders":
                return [{"orderId": 1, "clientOrderId": "dbadL1"}, {"orderId": 2, "clientOrderId": "dbtpL2"},
                        {"orderId": 3, "clientOrderId": "aos_user"}, {"orderId": 4, "clientOrderId": "dbadS4"}]
            if method == "GET" and path == "/fapi/v1/openAlgoOrders":
                return [{"algoId": 9, "clientAlgoId": "dbslL9"}]
            if method == "DELETE":
                deleted.append(params.get("orderId") or params.get("algoId"))
                return {}
            raise AssertionError(f"예상 밖 호출 {method} {path}")

        with mock.patch.object(ex, "signed", fake_signed):
            out = asyncio.run(ex.cancel_bracket(None, SYM, "LONG", "k", "s", 0))
        self.assertEqual(sorted(deleted), [1, 2, 9], deleted)      # 롱 물타기·TP·스탑만 -- 사용자 주문(3)·숏 쪽(4)은 남김
        self.assertEqual(out["cancelled"], 3)


if __name__ == "__main__":
    unittest.main()
