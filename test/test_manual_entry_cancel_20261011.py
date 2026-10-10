"""«따라가기» 진입(peg · 120초 뒤 잔량 시장가)의 대기 취소(2026-10-11) -- 모의 거래소로 집행 경로를 끝까지. 🔴네트워크 0.

(a) working 중 취소 → 주문 취소 1회 · 시장가 0 · phase cancelled  (b) 일부 체결 뒤 취소 → 든 몫 유지 · 손절/TP 경로 한 번(서버)
(c) 취소 확인 실패 → «거래소에서 직접 확인» 오류로 멈춤 · 다시 걸기 0 · 시장가 0  (d) 끝난 주문·청산·confirm 없음 → 409/400(무해)
음성 대조: 취소 없이 같은 상황이면 마감에 잔량 시장가(옛 동작 그대로).
실행: python -m pytest -q test/test_manual_entry_cancel_20261011.py
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

import scripts.live_manual_peg_execute_20260912 as ex  # noqa: E402

PLAN = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "price": 2500.0, "quantity": 1.0, "target_leverage": 0}


class Fake:
    def __init__(self, partial: float = 0.0, stuck: bool = False):
        self.limits, self.markets, self.deletes, self.polls = [], [], [], 0
        self.partial, self.stuck, self.status, self.filled = partial, stuck, "NEW", 0.0

    async def signed(self, _s, method, path, params, *_a):
        if path == "/fapi/v1/order" and method == "POST":
            if params["type"] == "MARKET":
                self.markets.append(params)
                return {"orderId": 9, "status": "FILLED", "executedQty": str(params["quantity"])}
            self.limits.append(params)
            return {"orderId": 7, "status": "NEW", "executedQty": "0"}
        if path == "/fapi/v1/order" and method == "GET":
            self.polls += 1
            if self.partial:
                self.filled = self.partial
            return {"orderId": 7, "status": self.status, "executedQty": str(self.filled)}
        if path == "/fapi/v1/order" and method == "DELETE":
            self.deletes.append(params)
            if self.status == "NEW" and not self.stuck:        # stuck = 취소가 안 먹는 거래소(확인 실패)
                self.status = "CANCELED"
            return {"orderId": params["orderId"]}
        return {"__error__": f"unexpected {method} {path}"}


def patched(fx: Fake):
    async def mp(_s, side, symbol):
        return (2500.0, 2500.0, 2500.1)                         # 앞지르지 않음 -- 리페그 없이 그대로 기다린다
    return [mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "maker_price", mp),
            mock.patch.object(ex, "POLL_SEC", 0.02), mock.patch.object(ex, "FALLBACK_SEC", 0.6),
            mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)),
            mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"})]


def run(fx: Fake, cancel_after: float | None) -> dict:
    state: dict = {}

    async def go():
        t = asyncio.create_task(ex.run_entry(None, dict(PLAN), state))
        if cancel_after is not None:
            await asyncio.sleep(cancel_after)
            state.update(action="cancel", action_reason="사용자 취소")
        await t

    ps = patched(fx)
    for p in ps:
        p.start()
    try:
        asyncio.run(go())
    finally:
        for p in reversed(ps):
            p.stop()
    return state


class EntryCancelRunnerTest(unittest.TestCase):
    def test_a_cancel_while_working(self) -> None:
        fx = Fake()
        st = run(fx, cancel_after=0.1)
        self.assertEqual(st["phase"], "cancelled")
        self.assertEqual(len(fx.limits), 1)
        self.assertEqual(len(fx.deletes), 1)
        self.assertEqual(fx.markets, [])
        self.assertEqual(st["filled"], 0.0)
        self.assertEqual(st["cancel_reason"], "사용자 취소")

    def test_b_partial_fill_kept(self) -> None:
        fx = Fake(partial=0.3)
        st = run(fx, cancel_after=0.1)
        self.assertEqual(st["phase"], "cancelled")
        self.assertAlmostEqual(st["filled"], 0.3)
        self.assertEqual(fx.markets, [])

    def test_c_unconfirmed_cancel_stops_with_error(self) -> None:
        fx = Fake(stuck=True)
        st = run(fx, cancel_after=0.1)
        self.assertEqual(st["phase"], "error")
        self.assertIn("거래소에서 직접 확인", st["error"])
        self.assertEqual(len(fx.limits), 1, "확인 못 한 채 다시 걸면 안 된다")
        self.assertEqual(fx.markets, [])

    def test_negative_control_no_cancel_goes_taker(self) -> None:
        fx = Fake(partial=0.3)
        st = run(fx, cancel_after=None)
        self.assertEqual(st["phase"], "filled_taker")
        self.assertEqual([m["quantity"] for m in fx.markets], [0.7])


class EntryCancelServerTest(unittest.TestCase):
    """서버 경로(실제 run_entry · 가짜 거래소): 취소 엔드포인트 게이트 · 든 몫이 있으면 진입 종료 경로가 손절/TP 를 한 번 다룬다."""

    def _exercise(self, fx: Fake, body, positions=()):
        from aiohttp.test_utils import TestClient, TestServer
        from offline_app import offline_app
        from dashboard import server
        from test_manual_entry_preview_smoke_20260913 import _isolated_dirs
        from test_rec_rule_entry_20261009 import account
        acct = account(list(positions))

        async def fake_account(*_a, **_k):
            return acct

        async def fake_exit(_s, plan, state):
            state.update(phase="working", quantity=plan["quantity"], filled=0.0)
            await asyncio.sleep(0.3)
            state.update(phase="filled_maker", filled=plan["quantity"])
            return state

        async def go():
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "run_exit", fake_exit), \
                 mock.patch.object(server, "place_bracket", mock.AsyncMock(side_effect=AssertionError("주문 금지"))):
                c = TestClient(TestServer(offline_app(server)))
                await c.start_server()
                try:
                    await body(c, server)
                finally:
                    await c.close()

        ps = patched(fx)
        for p in ps:
            p.start()
        try:
            with _isolated_dirs():
                asyncio.run(go())
        finally:
            for p in reversed(ps):
                p.stop()

    async def _wait_done(self, c) -> dict:
        for _ in range(200):
            st = (await (await c.get("/api/manual-entry/status")).json())["state"]
            if st.get("phase") not in ("working", "submitting") and (not float(st.get("filled") or 0) or st.get("bracket")):
                return st
            await asyncio.sleep(0.02)
        return st

    def test_server_cancel_partial_then_bracket_path_once(self) -> None:
        fx = Fake(partial=0.3)

        async def body(c, server):
            self.assertEqual((await c.post("/api/manual-entry/cancel?confirm=1")).status, 409)   # (d) 진입 없음
            b = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&pct=5&lev=20&sltp=0")).json()
            self.assertTrue(b.get("ok"), b)
            await asyncio.sleep(0.1)
            self.assertEqual((await c.post("/api/manual-entry/cancel")).status, 400)              # confirm 없음
            r = await c.post("/api/manual-entry/cancel?confirm=1")
            self.assertEqual(r.status, 200, await r.text())
            st = await self._wait_done(c)
            self.assertEqual(st["phase"], "cancelled")
            self.assertGreater(float(st["filled"]), 0)
            self.assertEqual(st["bracket"].get("disabled"), True)    # 든 몫 → 진입 종료 경로가 손절/TP 를 다뤘다(sltp=0 이라 «안 걸기» 기록)
            self.assertEqual(fx.markets, [])
            self.assertEqual((await c.post("/api/manual-entry/cancel?confirm=1")).status, 409)   # (d) 끝난 뒤 = 무해
        self._exercise(fx, body)

    def test_server_cancel_zero_fill_no_bracket(self) -> None:
        fx = Fake()

        async def body(c, server):
            b = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&pct=5&lev=20&sltp=0")).json()
            self.assertTrue(b.get("ok"), b)
            await asyncio.sleep(0.1)
            self.assertEqual((await c.post("/api/manual-entry/cancel?confirm=1")).status, 200)
            st = await self._wait_done(c)
            self.assertEqual(st["phase"], "cancelled")
            self.assertNotIn("bracket", st)                           # 체결 0 → 아무것도 안 건다
            self.assertEqual(fx.markets, [])
        self._exercise(fx, body)

    def test_server_exit_not_cancellable(self) -> None:
        fx = Fake()

        from test_rec_rule_entry_20261009 import long_pos

        async def body(c, server):
            r = await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=100")
            self.assertEqual(r.status, 200, await r.text())
            await asyncio.sleep(0.05)
            st = (await (await c.get("/api/manual-entry/status")).json())["state"]
            self.assertEqual((st.get("kind"), st.get("phase")), ("exit", "working"))
            self.assertEqual((await c.post("/api/manual-entry/cancel?confirm=1")).status, 409)   # 청산 취소는 안 받는다
            st = (await (await c.get("/api/manual-entry/status")).json())["state"]
            self.assertNotEqual(st.get("action"), "cancel")
        self._exercise(fx, body, positions=[long_pos(0.2, 2500.0)])


if __name__ == "__main__":
    unittest.main()
