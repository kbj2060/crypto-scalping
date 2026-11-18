"""추가 진입 대기 여러 개(2026-11-18 사용자 «제한 없이 여러 개 · 하나 체결돼도 나머지 유지»). 🔴실주문 0(집행 함수 전부 가짜).
① 같은 쪽 추가 진입 지정가·감시는 위·아래 상관없이 여러 칸(wid 1,2,3…) · 같은 종류·같은 선 중복(재시도·두 기기)과 추가 아닌 것은 409
② wait-action·status(wait_list) 가 칸 번호로 갈린다 · 빈 칸 번호는 다시 쓴다
③ 하나가 체결돼도 나머지는 그대로(독립)
④ 새 대기 크기 = 같은 코인 다른 대기의 남은 몫을 상한에서 뺀 값(다 체결돼도 상한 안)
python -m pytest -q test/test_add_entry_many_waits_20261118.py"""
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
STATUS = "/api/manual-entry/status?asset=eth"


class ManyWaitsTest(CL.ChartLinesTest):
    SAVED = CL.LinesAddSizeTest.SAVED
    _armed = CL.LinesArmedGuardTest._armed   # 상속하면 그쪽 시험까지 다시 돈다 -- 도구만 빌린다

    def _fakes(self, flags: dict):
        async def fake_limit(_s, plan, state, max_sec=0, on_fill=None, reprice=None):
            state.update(phase="working", mode="limit", order_id=5, filled=0.0, limit_price=plan["price"])
            while not state.get("action"):
                await asyncio.sleep(0.01)
                if flags.get("fill") == plan["limit"]["usdt"]:
                    state.update(filled=float(plan["quantity"]), phase="filled_maker")
                    return state
            state.update(phase="cancelled")
            return state

        async def fake_stop(_s, plan, state, touched, max_sec=0):
            state.update(phase="working", mode="limit", stop=True, filled=0.0)
            while not state.get("action"):
                await asyncio.sleep(0.01)
            state.update(phase="cancelled")
            return state

        async def fake_bracket(_s, b, symbol, pside):
            return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fake_ladder(*_a, **_k):
            return {"placed": True}
        return {"run_entry_limit": fake_limit, "run_entry_stop": fake_stop, "place_bracket": fake_bracket, "place_tp_ladder": fake_ladder}

    async def _post(self, c, q):
        r = await c.post(BASE + q)
        b = await r.json()
        await asyncio.sleep(0.05)
        return r.status, b

    async def _live(self, c):
        return {w["wid"]: w["phase"] for w in (await (await c.get(STATUS)).json())["wait_list"]}

    async def _cancel_all(self, c):
        for w in range(1, 6):
            await c.post(f"/api/manual-entry/wait-action?action=cancel&asset=eth&wid={w}&confirm=1")
        await asyncio.sleep(0.1)

    def test_many_same_side_and_dup_refused(self) -> None:
        async def fn(c):
            self._armed(self.SAVED)
            got = [await self._post(c, q) for q in ("&limit=2490", "&limit=2485", "&limit=2510&stop=1", "&limit=2520&stop=1")]
            self.assertEqual([(st, b.get("wid")) for st, b in got], [(200, 1), (200, 2), (200, 3), (200, 4)], got)
            st, b = await self._post(c, "&limit=2485")   # 같은 종류·같은 선 = 재시도 중복
            self.assertEqual((st, b["error"]), (409, "wait_pending"))
            self.assertEqual(await self._live(c), {1: "working", 2: "working", 3: "working", 4: "working"})
            r = await c.post("/api/manual-entry/wait-action?action=cancel&asset=eth&wid=2&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            await asyncio.sleep(0.1)
            st, b = await self._post(c, "&limit=2480")   # 빈 칸 2 를 다시 쓴다
            self.assertEqual((st, b.get("wid")), (200, 2), b)
            self.assertEqual(await self._live(c), {1: "working", 2: "working", 3: "working", 4: "working"})
            await self._cancel_all(c)
        self._run(fn, acct=POS, **self._fakes({}))

    def test_fill_keeps_the_others(self) -> None:
        flags: dict = {}

        async def fn(c):
            self._armed(self.SAVED)
            for q in ("&limit=2490", "&limit=2510&stop=1"):
                await self._post(c, q)
            flags["fill"] = 2490.0
            await asyncio.sleep(0.4)
            self.assertEqual(await self._live(c), {1: "filled_maker", 2: "working"})
            await self._cancel_all(c)
        self._run(fn, acct=POS, **self._fakes(flags))

    def test_room_counts_other_waits(self) -> None:   # 다 체결돼도 상한 안 -- 두 번째 대기는 첫 대기 남은 몫만큼 작다
        async def fn(c):
            self._armed(self.SAVED)
            q = "/api/manual-entry/preview?side=LONG&lev=20&pct=100&tp=2550&sl=2480&limit=2485"
            n0 = (await (await c.get(q)).json())["plan"]["notional_usdt"]
            r = await c.post(BASE.replace("pct=10", "pct=100") + "&limit=2490")   # 검토 #6: 100% 로 제출(전엔 replace 가 안 먹어 10%)
            b = await r.json()
            await asyncio.sleep(0.05)
            self.assertEqual(r.status, 200, b)
            r = await c.get(q)
            j = await r.json()
            n1 = (j.get("plan") or {}).get("notional_usdt") or 0.0
            self.assertLess(n1, n0 - 1.0, (n0, j))
            await self._cancel_all(c)
        self._run(fn, acct=POS, **self._fakes({}))

    def test_slot_not_reused_while_old_task_rearms(self) -> None:   # 검토 11-18 #2: 체결 뒤 익절 다시 걸기 중인 칸은 새 대기에 안 준다(복구 파일·상태가 섞이지 않게)
        flags: dict = {}
        fk = self._fakes(flags)

        async def slow_bracket(_s, b, symbol, pside):
            await asyncio.sleep(0.5)
            return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}
        fk["place_bracket"] = slow_bracket

        async def fn(c):
            from dashboard import server
            self._armed({**self.SAVED, "tp_resting": True})
            await self._post(c, "&limit=2490")
            await self._post(c, "&limit=2510&stop=1")
            flags["fill"] = 2490.0
            await asyncio.sleep(0.1)
            st, b = await self._post(c, "&limit=2470")
            self.assertEqual((st, b.get("wid")), (200, 3), b)
            await asyncio.sleep(0.8)
            self.assertTrue((server.LIVE_DIR / f"manual_limit_wait_{CL.SYM}_3.json").exists(), "새 대기 복구 파일이 지워지면 안 된다")
            await self._cancel_all(c)
        self._run(fn, acct=POS, **fk)

    def test_non_add_second_is_refused(self) -> None:
        async def fn(c):
            st, b = await self._post(c, "&limit=2490")   # 포지션 없음 = 추가 진입 아님
            self.assertEqual((st, b.get("wid")), (200, 1), b)
            st, b = await self._post(c, "&limit=2480")
            self.assertEqual((st, b["error"]), (409, "wait_pending"))
            await self._cancel_all(c)
        self._run(fn, **self._fakes({}))


if __name__ == "__main__":
    unittest.main()
