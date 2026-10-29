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

    def _tick(self, prices, armed_at, acct=POS, side="LONG", armed=None, clear_err=None, exit_phase="filled_maker", on_exit=None):
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
            if on_exit:
                on_exit()
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


class LineModifyTest(ChartLinesTest):
    """2026-10-29 «이미 들어간 선을 끌면 주문 수정»(POST /api/manual-bracket/modify). 시세 = 가짜 bookTicker 2500 · basis 1 · 계좌 POS 진입 2500."""
    _armed = LinesArmedGuardTest._armed

    def _mod(self, q, acct=POS, armed=None, place=None):
        out = {}

        async def fn(c):
            self._armed({"tp_price": 2550.0, "sl_price": 2480.0, **(armed or {})})
            out["t0"] = time.time()
            r = await c.post(f"/api/manual-bracket/modify?side=LONG&symbol={SYM}&confirm=1{q}")
            out["status"], out["body"] = r.status, await r.json()
            out["state"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
        self._run(fn, acct=acct, **({"place_bracket": place} if place else {}))
        return out

    def test_move_tp_replaces_resting_order_and_watches_from_now(self) -> None:
        calls = []

        async def place(_s, b, symbol, pside, **k):
            calls.append((dict(b), symbol, pside, k.get("strict")))
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": True, "price": b["tp_price"]}, "placed": True}
        o = self._mod("&tp=2560&sl=2490", place=place)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [({"tp_price": 2560.0}, SYM, "LONG", True)], "익절 지정가만 다시(비상 스탑·물타기 없음) · strict")
        b = o["state"]
        self.assertEqual((b["tp_level"], b["sl_level"], b["tp_price"], b["tp_resting"]), (2560.0, 2490.0, 2560.0, True), b)
        self.assertGreaterEqual(b["watch_from"], o["t0"], "옮기기 전 가격으로 바로 닫지 않게")
        self.assertEqual(b["armed_at"], 1000.0, "무장 시각은 그대로(감시 틱 병합 기준)")

    def test_move_sl_only_touches_no_exchange_order(self) -> None:
        o = self._mod("&tp=2550&sl=2470")   # place_bracket = _boom
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual((o["state"]["sl_level"], o["state"]["tp_level"], o["state"]["tp_resting"]), (2470.0, 2550.0, True))

    def test_tp_cancel_failure_hands_tp_to_server_watch(self) -> None:
        async def place(*_a, **_k):   # strict: 옛 주문을 다 못 지웠으면 새로 안 건다(place_bracket 실제 동작)
            return {"qty": 0.5, "cleared": {"cancelled": 0, "errors": ["openAlgoOrders: -1003"]}, "placed": False, "reason": "옛 주문 정리 실패"}
        o = self._mod("&tp=2560&sl=2470", place=place)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertIn("정리 실패", o["body"]["warn"])
        b = o["state"]
        self.assertEqual((b["tp_level"], b["sl_level"], b["tp_resting"]), (2560.0, 2470.0, False), "새 익절선은 서버 1초 닿음 감시")

    def test_early_place_failure_changes_nothing(self) -> None:
        async def place(*_a, **_k):
            return {"placed": False, "reason": "포지션 조회 실패"}
        o = self._mod("&tp=2560&sl=2470", place=place)
        self.assertEqual(o["status"], 502, o["body"])
        self.assertEqual((o["state"]["tp_level"], o["state"]["sl_level"], o["state"]["tp_resting"]), (2550.0, 2480.0, True))

    def test_sl_only_with_basis_drift_keeps_resting_tp(self) -> None:
        o = self._mod("&tp=2550&sl=2470", armed={"tp_price": 2549.49})   # 무장 때 basis 0.9998 -- 레벨이 같으면 지정가를 안 건드린다(place = _boom)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual((o["state"]["tp_price"], o["state"]["sl_level"]), (2549.49, 2470.0))

    def test_tp_place_failure_falls_back_to_server_watch(self) -> None:
        async def place(*_a, **_k):
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": False, "error": "-5022"}, "placed": False}
        o = self._mod("&tp=2560&sl=2480", place=place)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertFalse(o["state"]["tp_resting"], "지정가가 없으면 서버가 1초 닿음으로 익절")

    def test_rejects(self) -> None:
        self.assertEqual(self._mod("&tp=2450&sl=2480")["status"], 400)                       # 롱인데 익절이 지금가 아래
        self.assertEqual(self._mod("&tp=2550")["status"], 400)                                # 하나만
        far = {**POS, "positions": [{**POS["positions"][0], "entry_price": 2600.0}]}
        o = self._mod("&tp=2550&sl=2440", acct=far)                                          # 지금가에서 2.4% · 진입가에서 6.2% > 5%
        self.assertEqual((o["status"], o["body"]["error"]), (400, "sl_too_far"), o["body"])
        self.assertEqual(self._mod("&tp=2560&sl=2490", armed={"fired": True})["status"], 409)
        self.assertEqual(self._mod("&tp=2560&sl=2490", armed={"lines": False})["status"], 409)
        self.assertEqual(self._mod("&tp=2560&sl=2490", acct=FLAT)["status"], 409)             # 포지션 없음

    def test_confirm_required(self) -> None:
        async def fn(c):
            self._armed()
            r = await c.post(f"/api/manual-bracket/modify?side=LONG&symbol={SYM}&tp=2560&sl=2490")
            self.assertEqual(r.status, 400)
        self._run(fn, acct=POS)


class FiredPersistAndBarTest(LineTickTest):
    def test_fired_is_saved_before_exit_runs(self) -> None:
        seen = []   # 청산이 도는 동안 파일에 이미 fired 가 있어야 수정이 거절한다(검토 B1)
        calls, _ = self._tick([(2001, 2479.0)], armed_at=1999.0, exit_phase="error",
                              on_exit=lambda: seen.append(json.loads((orders.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"].get("fired")))
        self.assertEqual([c[0] for c in calls], ["clear", "exit"], calls)
        self.assertEqual(seen, [True], "run_exit 전에 fired 저장")

    def test_bar_close_backstop_respects_watch_from(self) -> None:
        from scripts.live_manual_peg_entry_20260912 import bracket_sl_hit
        armed = {"armed_at": 1000.0, "sl_level": 2498.0}
        self.assertTrue(bracket_sl_hit("LONG", (2000.0, 2495.0), armed))
        self.assertFalse(bracket_sl_hit("LONG", (2000.0, 2495.0), {**armed, "watch_from": 2100.0}), "수정 전에 마감된 봉으로 바로 닫지 않는다(검토 F1)")
        self.assertTrue(bracket_sl_hit("LONG", (2400.0, 2495.0), {**armed, "watch_from": 2100.0}))


class BracketMergeOwnershipTest(unittest.TestCase):
    def test_tick_does_not_revert_modified_lines(self) -> None:
        from scripts.live_manual_peg_entry_20260912 import bracket_merge
        k = f"{SYM}:LONG"
        cur = {k: {"armed_at": 1.0, "tp_level": 2560.0, "sl_level": 2490.0, "tp_price": 2560.0, "watch_from": 9.0, "tp_resting": False}}
        mine = {k: {"armed_at": 1.0, "tp_level": 2550.0, "sl_level": 2480.0, "tp_price": 2550.0, "tp_resting": True, "retry_after": 5.0}}
        m = bracket_merge(cur, mine, {k: 1.0})[k]
        self.assertEqual((m["tp_level"], m["sl_level"], m["tp_price"], m["watch_from"], m["tp_resting"], m["retry_after"]),
                         (2560.0, 2490.0, 2560.0, 9.0, False, 5.0), m)
        fired = {k: {**mine[k], "fired": True, "tp_resting": False}}
        m = bracket_merge({k: {**cur[k], "tp_resting": True}}, fired, {k: 1.0})[k]
        self.assertTrue(m["fired"] and m["tp_resting"] is False, "틱이 쏘았으면 그 틱의 tp_resting")


if __name__ == "__main__":
    unittest.main()
