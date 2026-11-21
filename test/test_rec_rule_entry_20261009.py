"""자동 규칙 진입(2026-10-09)의 시험이 있던 자리 -- 🔴2026-11-21 사용자 «sltp 만 쓴다»로 자동 규칙(rule=c|h)·−5% 규칙 손절을 지워
그 시험은 지웠다(test_stop_removal_20261121 이 «rule= 거절»·옛 규칙 무장의 5분봉 감시를 고정). 남은 것 = 다른 시험이 쓰는 계좌 틀(account·long_pos)
· 무장 없는 포지션을 계좌 API 가 unprotected 로 싣는가 · 청산 칸(코인별). 🔴실주문 0.
실행: python -m pytest -q test/test_rec_rule_entry_20261009.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from dashboard import orders  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = orders.MANUAL_EXEC_SYMBOLS["eth"]


def account(positions: list[dict], equity: float = 1000.0) -> dict:
    return {"ok": True, "balance": {"margin": equity, "initial_margin": 0.0, "available": equity, "unrealized": 0.0},
            "positions": positions, "leverage_by_symbol": {SYM: 20.0}, "trips": []}


def long_pos(qty: float, entry: float) -> dict:
    return {"symbol": SYM, "side": "LONG", "qty": qty, "entry_price": entry, "mark_price": 2500.0,
            "liquidation_price": 1000.0, "leverage": 20.0, "notional": qty * 2500.0, "unrealized_pnl": 0.0, "updated_at": None}


class RecRuleEntryTest(unittest.TestCase):
    def _run(self, acct: dict, fn, armed: dict | None = None) -> None:
        async def fake_account(*_a, **_k):
            return acct

        async def boom(*_a, **_k):
            raise AssertionError("주문 경로가 불렸다 -- 시험에서 실주문 금지")

        async def exercise() -> None:
            with mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "produce_account", fake_account, create=True), \
                 mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(orders, "SIZING_MARGIN_CAP_PCT", 50.0), \
                 mock.patch.object(orders, "place_bracket", boom):
                if armed is not None:
                    (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(armed), "utf-8")
                app = offline_app(server)
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_unarmed_position_listed_unprotected(self) -> None:
        """2026-10-20 XRP 무방비 사고: 계좌 payload 에 무장 없는 포지션 키 · 무장이 있으면 빠진다(2026-11-21 부터 무장 없는 포지션에 넣는 추가 진입은 409 -- test_stop_removal)."""
        async def fn(c):
            a = await (await c.get("/api/binance-account")).json()
            self.assertEqual(a.get("unprotected"), [f"{SYM}:LONG"], a.get("unprotected"))
        self._run(account([long_pos(0.4, 2500.0)]), fn)

        async def fn2(c):
            self.assertEqual((await (await c.get("/api/binance-account")).json()).get("unprotected"), [])
        self._run(account([long_pos(0.4, 2500.0)]), fn2,
                  armed={f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2300.0, "backstop_price": 2300.0, "armed_at": 1.0}})

    def test_exit_slot_per_coin(self) -> None:
        """2026-10-20 사용자 «청산만 코인별»: 주 칸을 다른 코인이 쓰면 이 코인 청산 칸 · 같은 코인이면 막힘 · 칸이 비면 주 칸."""
        import re
        src = (Path(__file__).resolve().parents[1] / "dashboard" / "orders.py").read_text("utf-8")
        i = src.index("    def _busy("); j = src.index("    # 2026-10-16 접수 검사")
        ns: dict = {"Any": object}
        exec(re.sub(r"^    ", "", src[i:j], flags=re.M), ns)
        ns["manual_entry_state"], ns["side_exit_states"] = {"phase": "idle"}, {}
        self.assertIs(ns["exit_slot"]("xrp"), ns["manual_entry_state"])
        ns["manual_entry_state"].update(phase="working", asset="eth")
        self.assertIsNone(ns["exit_slot"]("eth"))
        side = ns["exit_slot"]("xrp")
        self.assertIs(side, ns["side_exit_states"]["xrp"])
        side["phase"] = "working"
        self.assertIsNone(ns["exit_slot"]("xrp"))                      # 이 코인 청산이 도는 중이면 또 못 낸다
        ns["manual_entry_state"]["phase"] = "filled_maker"
        self.assertIsNone(ns["exit_slot"]("xrp"))                      # 주 칸이 비어도 이 코인 칸이 돌면 막힘


if __name__ == "__main__":
    unittest.main()
