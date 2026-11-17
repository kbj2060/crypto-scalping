"""추가 진입 대기 둘(2026-11-18 사용자 «현재가 위에 하나 아래에 하나 · 하나 체결되면 나머지 취소»). 🔴실주문 0(집행 함수 전부 가짜).
① 같은 쪽 추가 진입 지정가(아래)가 걸린 채 감시 진입(위)을 내면 칸 2(wid=2) · 세 번째·추가 아닌 것은 409
② wait-action·status 가 칸 번호로 갈린다(wid=2 취소 = 칸 1 은 그대로)
③ 한쪽이 체결되면 서버가 다른 쪽 대기를 취소(하나만 들어감)
python -m pytest -q test/test_add_entry_two_waits_20261118.py"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_chart_lines_entry_20261028 as CL  # noqa: E402 -- 모듈로 들여 그쪽 시험 클래스가 여기서 다시 모이지 않게

POS = CL.POS

BASE = "/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2550&sl=2480&mode=limit"


class TwoWaitsTest(CL.ChartLinesTest):
    SAVED = CL.LinesAddSizeTest.SAVED
    _armed = CL.LinesArmedGuardTest._armed   # 상속하면 그쪽 시험까지 다시 돈다 -- 도구만 빌린다

    def _fakes(self, flags: dict):
        async def fake_limit(_s, plan, state, max_sec=0, on_fill=None, reprice=None):
            state.update(phase="working", mode="limit", order_id=5, filled=0.0, limit_price=plan["price"])
            while not state.get("action"):
                await asyncio.sleep(0.01)
                if flags.get("fill"):
                    state.update(filled=float(plan["quantity"]), phase="filled_maker")
                    return state
            state.update(phase="cancelled")
            return state

        async def fake_stop(_s, plan, state, touched, max_sec=0):
            state.update(phase="working", mode="limit", stop=True, filled=0.0)
            while not state.get("action"):
                await asyncio.sleep(0.01)
            state.update(phase="cancelled", cancel_reason=state.get("action_reason"))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fake_ladder(*_a, **_k):
            return {"placed": True}
        return {"run_entry_limit": fake_limit, "run_entry_stop": fake_stop, "place_bracket": fake_bracket, "place_tp_ladder": fake_ladder}

    async def _two(self, c) -> None:
        self._armed(self.SAVED)
        b1 = await (await c.post(BASE + "&limit=2490")).json()
        self.assertTrue(b1.get("ok") and b1.get("wid") == 1, b1)
        await asyncio.sleep(0.05)
        b2 = await (await c.post(BASE + "&limit=2510&stop=1")).json()
        self.assertTrue(b2.get("ok") and b2.get("wid") == 2 and b2["plan"].get("add_keep"), b2)
        await asyncio.sleep(0.05)
        r = await c.post(BASE + "&limit=2485")   # 세 번째 = 막힘
        self.assertEqual((r.status, (await r.json())["error"]), (409, "wait_pending"))
        s = await (await c.get("/api/manual-entry/status?asset=eth")).json()
        self.assertEqual((s["wait"]["phase"], s["wait2"]["phase"], s["wait2"]["wid"]), ("working", "working", 2), s)

    def test_second_slot_and_cancel_by_wid(self) -> None:
        async def fn(c):
            await self._two(c)
            r = await c.post("/api/manual-entry/wait-action?action=cancel&asset=eth&wid=2&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            await asyncio.sleep(0.1)
            s = await (await c.get("/api/manual-entry/status?asset=eth")).json()
            self.assertEqual((s["wait"]["phase"], s["wait2"]["phase"]), ("working", "cancelled"), s)
            await c.post("/api/manual-entry/wait-action?action=cancel&asset=eth&confirm=1")
            await asyncio.sleep(0.1)
        self._run(fn, acct=POS, **self._fakes({}))

    def test_free_slot_one_is_reused_while_two_waits(self) -> None:   # 칸 1 을 취소한 뒤 다시 걸면 칸 1(칸 2 는 그대로)
        async def fn(c):
            await self._two(c)
            await c.post("/api/manual-entry/wait-action?action=cancel&asset=eth&confirm=1")
            await asyncio.sleep(0.1)
            b = await (await c.post(BASE + "&limit=2488")).json()
            self.assertTrue(b.get("ok") and b.get("wid") == 1, b)
            await asyncio.sleep(0.05)
            s = await (await c.get("/api/manual-entry/status?asset=eth")).json()
            self.assertEqual((s["wait"]["phase"], s["wait2"]["phase"]), ("working", "working"), s)
            for w in ("1", "2"):
                await c.post(f"/api/manual-entry/wait-action?action=cancel&asset=eth&wid={w}&confirm=1")
            await asyncio.sleep(0.1)
        self._run(fn, acct=POS, **self._fakes({}))

    def test_fill_cancels_the_other(self) -> None:
        flags: dict = {}

        async def fn(c):
            await self._two(c)
            flags["fill"] = True
            for _ in range(60):
                await asyncio.sleep(0.05)
                s = await (await c.get("/api/manual-entry/status?asset=eth")).json()
                if s["wait2"]["phase"] == "cancelled":
                    break
            self.assertEqual((s["wait"]["phase"], s["wait2"]["phase"]), ("filled_maker", "cancelled"), s)
            self.assertIn("하나만", s["wait2"].get("cancel_reason") or "", s["wait2"])
        self._run(fn, acct=POS, **self._fakes(flags))

    def test_non_add_second_is_refused(self) -> None:
        async def fn(c):
            b1 = await (await c.post(BASE + "&limit=2490")).json()   # 포지션 없음 = 추가 진입 아님
            self.assertTrue(b1.get("ok") and b1.get("wid") == 1, b1)
            await asyncio.sleep(0.05)
            r = await c.post(BASE + "&limit=2480")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "wait_pending"))
            await c.post("/api/manual-entry/wait-action?action=cancel&asset=eth&confirm=1")
            await asyncio.sleep(0.1)
        self._run(fn, **self._fakes({}))


if __name__ == "__main__":
    unittest.main()
