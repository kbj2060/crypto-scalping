"""차트 익절·손절 선 진입(2026-10-28 사용자 «선 2개를 끌어 놓고 진입 → peg · 둘 다 메이커로 청산 · 닿으면 닫기 · 비상 스탑 없음 · 자동 2×»).
🔴실주문 0: 주문·취소 함수는 전부 가짜. ① 선 검증(방향·하나만·같은 쪽 포지션) ② 체결 → 익절 지정가만(비상 스탑 없음·물타기 없음) · 무장 파일에 선
③ 감시 한 틱: 1초 체결가가 손절선에 닿으면 익절 지정가를 먼저 지우고 peg 청산 · 무장 전 가격은 무시 · 안 닿으면 아무것도 안 함.
python -m pytest -q test/test_chart_lines_entry_20261028.py"""
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
from dashboard import orders  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _fake_binance, _isolated_dirs  # noqa: E402

SYM = orders.MANUAL_EXEC_SYMBOLS["eth"]
VM = {"sigma_bp": 250.0 / math.sqrt(6), "mult": 0.45}
FLAT = {"ok": True, "balance": {"margin": 1000.0, "initial_margin": 0.0, "available": 1000.0, "unrealized": 0.0},
        "positions": [], "leverage_by_symbol": {SYM: 20.0}, "trips": []}
POS = {**FLAT, "positions": [{"symbol": SYM, "side": "LONG", "qty": 0.5, "entry_price": 2500.0, "mark_price": 2500.0,
                              "liquidation_price": 1500.0, "leverage": 20.0, "notional": 1250.0, "unrealized_pnl": 0.0,
                              "updated_at": None}]}


async def _boom(*_a, **_k):
    raise AssertionError("실주문 경로가 불렸다 -- 시험에서 주문 금지")


class ChartLinesTest(unittest.TestCase):
    def _run(self, fn, acct=FLAT, **patches) -> None:
        async def fake_account(*_a, **_k):
            return acct

        async def exercise() -> None:
            base = {"place_bracket": _boom, "place_add_order": _boom, "run_entry": _boom, "run_exit": _boom, "clear_bracket": _boom, "place_stop": _boom}
            base.update(patches)
            with mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "produce_account", fake_account, create=True), \
                 mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(orders, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(orders, "REC_AUTO_ADD", True), \
                 mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.object(orders, "margin_type", mock.AsyncMock(return_value="cross")), \
                 mock.patch.multiple(orders, **base):
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

    def test_line_validation(self) -> None:
        async def fn(c):
            async def get(q):
                r = await c.get(f"/api/manual-entry/preview?side=LONG&lev=20&rule=c{q}")
                return r.status, await r.json()
            st, b = await get("&tp=2550&sl=2480")
            self.assertEqual(st, 200, b)
            br = b["plan"]["bracket"]
            self.assertTrue(br["lines"] and br["backstop_price"] is None and br["tp_level"] == 2550.0 and br["sl_level"] == 2480.0, br)
            self.assertNotIn("add", b["plan"].get("rule") or {}, "선 진입엔 자동 물타기 지정가가 없다")
            self.assertEqual((await get("&tp=2480&sl=2550"))[0], 400)        # 롱인데 익절이 아래
            self.assertEqual((await get("&tp=2550"))[0], 400)                # 하나만
            self.assertEqual((await get("&tp=abc&sl=2480"))[0], 400)
            self.assertEqual((await get("&tp=2500.5&sl=2480"))[0], 400)      # 0.05% 안쪽
        self._run(fn)

    def test_same_side_position_blocks_lines(self) -> None:
        async def fn(c):
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c&tp=2550&sl=2480")
            self.assertEqual(r.status, 409, await r.json())
        self._run(fn, acct=POS)

    def test_fill_places_only_tp_and_arms_lines(self) -> None:
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(("bracket", dict(b)))
            return {"placed": True, "tp": {"placed": True, "price": b["tp_price"]}}

        async def fn(c):
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=c&tp=2550&sl=2480")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(50):
                await asyncio.sleep(0.02)
                if (server.LIVE_DIR / "manual_bracket_state.json").exists():
                    break
            self.assertEqual([k[0] for k in calls], ["bracket"], calls)   # 물타기 지정가 없음
            b = calls[0][1]
            self.assertIsNone(b["backstop_price"], "비상 스탑을 걸면 안 된다(사용자)")
            self.assertGreaterEqual(b["tp_price"], 2550.0 * 0.999)
            armed = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertTrue(armed["lines"] and armed["tp_resting"], armed)
            self.assertEqual((armed["sl_level"], armed["tp_level"]), (2480.0, 2550.0))
        self._run(fn, run_entry=fake_entry, place_bracket=fake_bracket)


class LinesArmedGuardTest(ChartLinesTest):
    def _armed(self, extra=None):
        (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({f"{SYM}:LONG": {
            "symbol": SYM, "side": "LONG", "lines": True, "sl_level": 2480.0, "tp_level": 2550.0, "tp_resting": True,
            "backstop_price": None, "armed_at": 1000.0, **(extra or {})}}), "utf-8")

    def test_plain_add_on_lines_position_is_blocked(self) -> None:
        async def fn(c):
            self._armed()
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
            self.assertEqual(r.status, 409, await r.json())
            self.assertEqual((await r.json())["error"], "lines_armed")
        self._run(fn, acct=POS)

    def test_rearm_lines_watches_from_now_without_exchange_stop(self) -> None:
        async def fn(c):
            self._armed({"sl_off": {"at": 1.0}, "fired": True})
            t0 = time.time()
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            b = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertNotIn("sl_off", b); self.assertNotIn("fired", b)
            self.assertGreaterEqual(b["watch_from"], t0)
        self._run(fn, acct=POS)   # place_stop = _boom -- 거래소 스탑을 걸려 하면 실패


class LineTickTest(unittest.TestCase):
    """감시 한 틱 -- register 를 가짜 의존성으로 직접 만든다(앱의 수집기·네트워크 없이)."""

    def _tick(self, prices, armed_at, acct=POS, side="LONG", armed=None, clear_err=None, exit_phase="filled_maker"):
        calls = []
        if side == "SHORT":
            acct = {**acct, "positions": [{**acct["positions"][0], "side": "SHORT"}]}   # 계좌 qty 는 절댓값(live_binance_account abs(positionAmt))

        async def fake_account(*_a, **_k):
            return acct

        async def swr(_k, _ttl, produce, **_kw):
            return await produce()

        async def clear(_s, symbol, pside, *a, **k):
            calls.append(("clear", pside))
            return {"errors": [clear_err]} if clear_err else {"cancelled": 1}

        async def exit_(_s, plan, state):
            calls.append(("exit", plan.get("position_side") or plan.get("positionSide")))
            state.update(phase=exit_phase, filled=1.0 if exit_phase.startswith("filled") else 0.0, quantity=1.0)
            return state

        async def hist(_a):
            return []

        async def go():
            with mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.multiple(orders, clear_bracket=clear, run_exit=exit_, run_entry=_boom, place_bracket=_boom,
                                     place_add_order=_boom):
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                      fetch_binance_json=_fake_binance, swr_cached=swr, _timed_produce=None,
                                      load_market_history=hist, situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60,
                                      sec_prices=lambda a: prices if a == "eth" else None)
                base = {"symbol": SYM, "side": side, "lines": True, "sl_level": 2480.0, "tp_level": 2550.0, "tp_resting": True,
                        "sl_price": 2480.0, "tp_price": 2550.0, "backstop_price": None, "armed_at": armed_at}
                if side == "SHORT":
                    base.update(sl_level=2520.0, tp_level=2450.0, sl_price=2520.0, tp_price=2450.0)
                (orders.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({f"{SYM}:{side}": {**base, **(armed or {})}}), "utf-8")
                await api.bracket_tick()
            return calls, json.loads((orders.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))

        with _isolated_dirs():
            return asyncio.run(go())

    def test_touch_clears_tp_then_peg_exits(self) -> None:
        calls, st = self._tick([(2000, 2490.0), (2001, 2479.5), (2002, 2483.0)], armed_at=1999.0)
        self.assertEqual([c[0] for c in calls][:2], ["clear", "exit"], calls)   # 익절 지정가 먼저 지우고 청산
        self.assertEqual(st, {}, "다 닫히면 무장을 내린다")

    def test_pre_arm_prices_and_no_touch_do_nothing(self) -> None:
        calls, st = self._tick([(1000, 2470.0), (2001, 2485.0), (2002, 2549.0)], armed_at=1999.0)
        self.assertEqual(calls, [])
        self.assertIn(f"{SYM}:LONG", st)

    def test_short_side_touch(self) -> None:
        calls, st = self._tick([(2001, 2510.0), (2002, 2520.5)], armed_at=1999.0, side="SHORT")
        self.assertEqual([c for c in calls][:2], [("clear", "SHORT"), ("exit", "SHORT")], calls)
        self.assertEqual(st, {})

    def test_tp_touch_when_tp_order_not_resting(self) -> None:
        calls, _ = self._tick([(2001, 2551.0)], armed_at=1999.0, armed={"tp_resting": False})
        self.assertEqual([c[0] for c in calls][:2], ["clear", "exit"], calls)
        calls, _ = self._tick([(2001, 2551.0)], armed_at=1999.0)   # 익절 지정가가 걸려 있으면 거래소 몫
        self.assertEqual(calls, [])

    def test_tp_cancel_failure_waits_and_does_not_exit(self) -> None:
        calls, st = self._tick([(2001, 2479.0)], armed_at=1999.0, clear_err="-1003 too many")
        self.assertEqual(calls, [("clear", "LONG")], "익절을 못 지웠으면 청산하지 않는다")
        b = st[f"{SYM}:LONG"]
        self.assertFalse(b.get("fired"))
        self.assertGreater(b["retry_after"] - time.time(), 5, "계좌 조회를 2초마다 되풀이하지 않게 10초 쉰다")

    def test_exit_failure_keeps_fired_latch_and_retries_without_touch(self) -> None:
        calls, st = self._tick([(2001, 2479.0)], armed_at=1999.0, exit_phase="error")
        b = st[f"{SYM}:LONG"]
        self.assertTrue(b["fired"] and not b["tp_resting"], b)
        calls, st = self._tick([(2003, 2495.0)], armed_at=1999.0, armed={"fired": True, "tp_resting": False})
        self.assertEqual([c[0] for c in calls][:1], ["exit"], "익절은 이미 지웠다 -- 닿음 창이 지나도 닫힐 때까지 청산")

    def test_prices_before_rearm_are_ignored(self) -> None:
        calls, _ = self._tick([(2001, 2470.0), (2050, 2500.0)], armed_at=1999.0, armed={"watch_from": 2040.0})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
