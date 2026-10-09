"""권고 규칙 진입(2026-10-09) -- 서버 경로를 끝까지 지나게 하고 숫자를 고정한다. 🔴실주문 0.

규칙: 왕복 최대 명목 = 순자산 × L(1.5·2) · 첫 진입 = 그 75% · 물타기 = 남은 여유 · 손절 = 첫 진입가 ∓5%(물타기해도 유지) ·
SL/TP 체크를 꺼도 규칙 손절은 건다 · L 이 이상하면 400(직접 모드로 조용히 떨어지지 않는다) · ETH 만.
네트워크·계좌는 test_manual_entry_preview_smoke 의 격리 틀(_isolated_dirs: 바이낸스 GET 가짜 응답)을 쓰고,
run_entry·place_bracket 은 갈아끼워 주문 경로가 절대 바깥으로 못 나가게 한다.
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
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = server.MANUAL_EXEC_SYMBOLS["eth"]


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
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "SIZING_MARGIN_CAP_PCT", 50.0), \
                 mock.patch.object(server, "place_bracket", boom):
                if armed is not None:
                    (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(armed), "utf-8")
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_first_entry_is_75pct_of_cap_and_stop_5pct(self) -> None:
        async def fn(c):
            j = await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5")).json()
            self.assertTrue(j.get("ok"), j)
            p = j["plan"]
            self.assertTrue(p["rule"]["first"], p["rule"])
            self.assertEqual(p["rule"]["cap_notional"], 1500.0)
            self.assertAlmostEqual(p["notional_usdt"], 1125.0, delta=2.6)          # 순자산 1000 × 1.5 × 0.75
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)                      # 2500.00 × 0.95
            self.assertIn("첫 진입가", p["rule"]["sl_name"])
            self.assertEqual(p["target_leverage"], 20)
            j2 = await (await c.get("/api/manual-entry/preview?side=SHORT&lev=20&rule=2")).json()
            self.assertAlmostEqual(j2["plan"]["notional_usdt"], 1500.0, delta=2.6)  # × 2 × 0.75
            self.assertAlmostEqual(j2["plan"]["bracket"]["sl_price"], 2625.01, delta=0.02)   # 숏 = 매도호가 × 1.05
        self._run(account([]), fn)

    def test_bad_rule_is_rejected_not_ignored(self) -> None:
        async def fn(c):
            for bad in ("3", "1", "abc", "-2"):
                r = await c.get(f"/api/manual-entry/preview?side=LONG&lev=20&rule={bad}")
                self.assertEqual(r.status, 400, bad)
                self.assertEqual((await r.json())["error"], "bad_rule")
                with mock.patch.object(server, "exec_enabled", lambda: True), \
                     mock.patch.object(server, "run_entry", mock.AsyncMock(side_effect=AssertionError("실주문 금지"))):
                    r = await c.post(f"/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule={bad}")
                self.assertEqual(r.status, 400, bad)                                # 게이트가 열려 있어도 400(직접 모드로 안 떨어진다)
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5&asset=sol")
            self.assertEqual(r.status, 400)
        self._run(account([]), fn)

    def test_rule_stop_survives_sltp_off(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5&sltp=0")).json())["plan"]
            self.assertFalse(p["bracket"].get("disabled"), p["bracket"])
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)
            p0 = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&sltp=0")).json())["plan"]
            self.assertTrue(p0["bracket"].get("disabled"), "직접 모드의 SL/TP 끔은 그대로여야 한다")
        self._run(account([]), fn)

    def test_add_uses_remaining_room_and_keeps_armed_stop(self) -> None:
        armed = {f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2375.0, "sl_level": 2375.0,
                                 "backstop_price": 2363.12, "rule": True, "sl_name": "첫 진입가 −5%",
                                 "armed_at": 0, "placed": {}}}

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5")).json())["plan"]
            self.assertFalse(p["rule"]["first"])
            self.assertAlmostEqual(p["notional_usdt"], 750.0, delta=2.6)            # 1500 − 기존 750
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)                      # 손절선이 안 물러난다
            self.assertEqual(p["bracket"]["backstop_price"], 2363.12)
        self._run(account([long_pos(0.3, 2500.0)]), fn, armed=armed)

    def test_add_without_rule_arming_anchors_on_average(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5")).json())["plan"]
            self.assertEqual(p["bracket"]["sl_price"], 2470.0)                      # 평단 2600 × 0.95
            self.assertIn("평단", p["rule"]["sl_name"])
        self._run(account([long_pos(0.3, 2600.0)]), fn)

    def test_full_cap_blocks_add(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=1.5")).json())["plan"]
            self.assertTrue(p["blocked"], p)
        self._run(account([long_pos(0.6, 2500.0)]), fn)                           # 명목 1500 = 상한

    def test_submit_carries_rule_without_sending_orders(self) -> None:
        fired = []

        async def never_run(_session, plan, state):
            fired.append(plan)
            state.update(phase="filled_maker")   # filled 0 -- 브래킷 단계로 안 간다
            return state

        async def fn(c):
            with mock.patch.object(server, "exec_enabled", lambda: True), mock.patch.object(server, "run_entry", never_run):
                body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=1.5")).json()
                self.assertTrue(body.get("ok"), body)
                await asyncio.sleep(0.05)
            self.assertEqual(len(fired), 1)
            self.assertEqual(fired[0]["rule"]["l"], 1.5)
            self.assertAlmostEqual(fired[0]["notional_usdt"], 1125.0, delta=2.6)
            self.assertEqual(fired[0]["bracket"]["sl_price"], 2375.0)
        self._run(account([]), fn)


if __name__ == "__main__":
    unittest.main()
