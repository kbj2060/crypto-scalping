"""2026-11-09 «감시 진입»(롱 = 지금가 위 · 숏 = 아래 선에 닿으면 peg 따라가기, 시장가 없음). 🔴실주문 0(집행·서명 전부 가짜).
python -m pytest -q test/test_manual_stop_entry_20261109.py"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_chart_lines_entry_20261028 import ChartLinesTest  # noqa: E402

from live.execution import live_manual_peg_execute_20260912 as ex  # noqa: E402

Q = "lev=20&pct=10"


class StopServerTest(ChartLinesTest):
    def test_preview_needs_stop_flag_above_price(self) -> None:
        async def fn(c):
            r = await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&tp=2600&sl=2480&limit=2520")
            self.assertEqual((r.status, (await r.json())["error"]), (400, "limit_marketable"))   # 플래그 없으면 예전 그대로
            r = await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&tp=2600&sl=2480&limit=2520&stop=1")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            self.assertTrue(b["plan"]["stop"])
            self.assertEqual((b["plan"]["bracket"]["tp_level"], b["plan"]["bracket"]["sl_level"]), (2600.0, 2480.0))   # 선 기준 = 감시선
        self._run(fn)

    def test_submit_goes_to_watcher_and_amend_works_without_order(self) -> None:
        seen = []

        async def fake_stop(_s, plan, state, touched, max_sec=0):
            seen.append(plan)
            state.update(phase="working", mode="limit", stop=True, triggered=None, filled=0.0, limit_price=plan["price"], amends=0)
            while not state.get("action"):
                await asyncio.sleep(0.01)
            state.update(phase="cancelled")
            return state

        async def fn(c):
            r = await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2600&sl=2480&limit=2520&mode=limit&stop=1&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            await asyncio.sleep(0.1)
            self.assertTrue(seen and seen[0]["stop"], "감시 진입은 run_entry_stop 이 맡는다")
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2600&sl=2480&limit=2530&confirm=1")   # 주문 없이도(감시 중) 수정
            b = await r.json()
            self.assertEqual((r.status, b["moved"]), (200, True), b)
            st = (await (await c.get("/api/manual-entry/status")).json())["wait"]
            self.assertEqual((st["plan"]["limit"]["usdt"], st["amend"]["price"]), (2530.0, 2530.0))
            r = await c.post("/api/manual-entry/wait-action?action=cancel&confirm=1")
            self.assertEqual(r.status, 200)
            await asyncio.sleep(0.1)
        self._run(fn, run_entry_stop=fake_stop)

    def test_amend_refused_once_triggered(self) -> None:
        async def fake_stop(_s, plan, state, touched, max_sec=0):
            state.update(phase="working", mode="limit", stop=True, triggered="now", order_id=9, filled=0.0, limit_price=plan["price"])
            while not state.get("action"):
                await asyncio.sleep(0.01)
            state.update(phase="cancelled")
            return state

        async def fn(c):
            r = await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2600&sl=2480&limit=2520&mode=limit&stop=1&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            await asyncio.sleep(0.1)
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2600&sl=2480&limit=2530&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "no_limit"))   # 따라가는 중엔 수정 없음
            r = await c.post("/api/manual-entry/wait-action?action=market&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "not_ready"))   # 따라가는 중 TAKER 도 없음
            await c.post("/api/manual-entry/wait-action?action=cancel&confirm=1")
            await asyncio.sleep(0.1)
        self._run(fn, run_entry_stop=fake_stop)


PLAN = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "price": 2520.0, "quantity": 0.4, "limit": {"usdt": 2520.0}}


class StopLoopTest(unittest.TestCase):
    def _go(self, touched, script=None):
        calls = []

        async def fake_entry(_s, plan, state):
            calls.append(plan)
            return state

        async def fake_book(*_a):
            return 2521.5, 2521.5, 2521.6

        async def go():
            st: dict = {}
            with mock.patch.object(ex, "run_entry", fake_entry), mock.patch.object(ex, "maker_price", fake_book):
                t = asyncio.create_task(ex.run_entry_stop(None, dict(PLAN), st, touched, max_sec=30))
                if script:
                    await script(st)
                await asyncio.wait_for(t, 10)
            return st, calls
        return asyncio.run(go())

    def test_touch_starts_maker_only_chase_at_best_bid(self) -> None:
        st, calls = self._go(lambda since: True)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["maker_only"])
        self.assertEqual(calls[0]["price"], 2521.5)
        self.assertTrue(st["triggered"])

    def test_amend_resets_touch_window_and_cancel_places_nothing(self) -> None:
        sinces = []

        def touched(since):
            sinces.append(since)
            return False

        async def script(st):
            await asyncio.sleep(1.2)
            st["amend"] = {"price": 2530.0, "qty": 0.39}
            await asyncio.sleep(1.2)
            st["action"] = "cancel"
        st, calls = self._go(touched, script)
        self.assertEqual((calls, st["phase"], st["amends"], st["limit_price"]), ([], "cancelled", 1, 2530.0))
        self.assertGreater(sinces[-1], sinces[0], "선을 옮기면 그 뒤 가격만 본다")

    def test_market_action_while_watching_goes_market(self) -> None:
        async def script(st):
            st["action"] = "market"
        st, calls = self._go(lambda since: False, script)
        self.assertEqual(calls[0]["type"], "MARKET")


class MakerOnlyTest(unittest.TestCase):
    def test_chase_deadline_drops_remainder_never_market(self) -> None:
        sent = []

        async def fake_signed(_s, method, path, params, *a, **k):
            sent.append((method, path, params.get("type")))
            return {}

        async def fake_fill(*_a, **_k):
            return 0.1, 2521.5, 3, None   # 0.4 중 0.1 만 체결하고 마감

        async def go():
            st: dict = {}
            with mock.patch.dict("os.environ", {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}), \
                 mock.patch.object(ex, "signed", fake_signed), mock.patch.object(ex, "fill_maker", fake_fill), \
                 mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
                 mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value=None)):
                await ex.run_entry(None, {**PLAN, "maker_only": True}, st)
            return st
        st = asyncio.run(go())
        self.assertEqual((st["phase"], st["filled"], st["taker_qty"]), ("cancelled", 0.1, 0.0))
        self.assertFalse([s for s in sent if s[2] == "MARKET"], "지정가 체결은 시장가로 바꾸지 않는다")


if __name__ == "__main__":
    unittest.main()
