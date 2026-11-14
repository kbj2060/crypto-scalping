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
            base = {"place_bracket": _boom, "place_add_order": _boom, "run_entry": _boom, "run_exit": _boom, "clear_bracket": _boom, "place_stop": _boom, "place_tp_ladder": _boom}
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
            self.assertEqual((await get("&tp=2500.5&sl=2499.5"))[0], 200)    # 2026-11-11 현재가에 붙인 선(0.02%)도 받는다 -- 최소 거리 없앰
            self.assertEqual((await get("&tp=2550&sl=2300"))[0], 200)        # 2026-11-04 손절 8% -- 규칙 손절폭 5% 한도를 뺐다
            self.assertEqual((await get("&tp=2550&sl=2100"))[0], 200)        # 2026-11-14 사용자 «±15% 제한 자체를 제거» -- 손절 16% 도 받는다
            self.assertEqual((await get("&tp=3000&sl=2480"))[0], 200)        # 익절 +20% 도
        self._run(fn)

    def test_direct_pct_lines(self) -> None:   # 2026-10-30 SLTP = 직접(증거금 %) -- rule 없이 선만, 계좌 2× 규칙 상한 없음
        async def fn(c):
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            pl = b["plan"]; br = pl["bracket"]
            self.assertTrue(br["lines"] and br["backstop_price"] is None and br["sl_level"] == 2480.0, br)
            self.assertNotIn("rule", pl, "직접 모드 = 자동 규칙 없음")
            eq = pl["projection"]["equity_usdt"]
            self.assertAlmostEqual(pl["margin_usdt"] / eq, 0.10, delta=0.01, msg=(pl["margin_usdt"], eq))
        self._run(fn)

    def test_same_side_position_ignores_sent_lines(self) -> None:   # 2026-11-14 사용자 «포지션 추가에 라인이 왜 필요하나»: 같은 쪽 포지션(선 무장 없음)엔 보낸 선을 버리고 일반 추가 진입
        async def fn(c):
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c&tp=2550&sl=2480")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            self.assertFalse(b["plan"]["bracket"].get("lines"), b["plan"]["bracket"])   # 무장 없는 포지션 = 평단 −5% 손절(10-20 XRP 무방비 사고 뒤 규칙) 그대로
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

    def test_plain_add_on_lines_position_keeps_lines(self) -> None:
        # 2026-11-12 사용자 «SOXL 추가 숏이 안 된다»: 선 없는 일반 진입도 막지 않고 무장된 선 그대로 크기만 늘린다(거래소 스탑 없음)
        async def fn(c):
            self._armed({"tp_price": 2550.4, "sl_price": 2479.6})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            br = b["plan"]["bracket"]
            self.assertTrue(b["plan"].get("add_lines") and br["lines"] and br["backstop_price"] is None, br)
            self.assertEqual((br["tp_price"], br["sl_price"]), (2550.4, 2479.6), br)
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


class LinesAddSizeTest(LinesArmedGuardTest):
    """2026-11-03 사용자 «SLTP 로 추가 진입 -- 손절/익절 라인은 그대로»: 같은 쪽 선 포지션에 크기만 늘리고 무장된 선(레벨·주문가)을 그대로 쓴다. 실주문 0(전부 가짜)."""
    SAVED = {"tp_price": 2550.4, "sl_price": 2479.6, "sl_name": "선 손절 2480.00"}

    def test_preview_uses_armed_lines_not_the_sent_ones(self) -> None:
        async def fn(c):
            self._armed(self.SAVED)
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2600&sl=2400")   # 보낸 선(2600/2400)은 무시된다
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            br = b["plan"]["bracket"]
            self.assertEqual((br["tp_level"], br["sl_level"], br["tp_price"], br["sl_price"]), (2550.0, 2480.0, 2550.4, 2479.6), br)
            self.assertTrue(b["plan"].get("add_lines"), "추가 진입 표시")
            self.assertTrue(br["lines"] and br["backstop_price"] is None, br)
        self._run(fn, acct=POS)

    def test_refused_when_fired_and_line_tool_on_plain_position_adds(self) -> None:
        async def fn(c):
            self._armed({**self.SAVED, "fired": True})                                           # 청산 진행 중은 막는다
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "exiting"))
            # 2026-11-14 사용자 «포지션 추가에 라인이 왜 필요하나»: 선 없는 포지션(규칙 손절)에 선 도구로 와도 보낸 선은 버리고 일반 추가 진입(기존 손절 그대로)
            self._armed({**self.SAVED, "lines": False, "rule": True})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            self.assertFalse(b["plan"]["bracket"].get("lines"), b["plan"]["bracket"])
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480&limit=2490")   # 선 없는 포지션의 지정가 추가는 아직 안 됨
            self.assertEqual((r.status, (await r.json())["error"]), (409, "add_limit_unsupported"))
        self._run(fn, acct=POS)

    def test_add_on_sl_off_position_keeps_sl_off(self) -> None:
        # 2026-11-13 펜 메모 «손절/익절 없이도 추가 진입»: 손절을 지운 선 포지션(옛 손절선이 지금가 너머여도) -- 체결 뒤에도 sl_off 그대로 · 익절은 다시 건다
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(dict(b))
            return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fn(c):
            self._armed({**self.SAVED, "sl_level": 2510.0, "sl_off": {"at": 1.0}})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10")
            self.assertEqual(r.status, 200, await r.json())
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(60):
                await asyncio.sleep(0.02)
                if calls:
                    break
            self.assertEqual(len(calls), 1, calls)
            self.assertEqual(calls[0]["tp_price"], 2550.4, calls[0])
            self.assertIsNone(calls[0].get("backstop_price"), calls[0])   # 2026-11-14 추가 진입 = 익절 수량만 다시(keep_rearm)
            await asyncio.sleep(0.1)
            armed = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertTrue(armed["lines"] and armed["sl_off"] and armed["tp_resting"], armed)
        self._run(fn, acct=POS, run_entry=fake_entry, place_bracket=fake_bracket)

    # 2026-11-13 서브 에이전트 검증 결함 재현(고친 뒤 통과해야 한다)
    def _bstate(self):
        return json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]

    def test_limit_add_on_sl_off_rearms_tp_on_full_size(self) -> None:
        """지정가 추가 + 손절 지운 포지션: 첫 부분 체결 뒤 최종 재설정을 건너뛰어 TP 가 일부만 덮던 것."""
        calls = []

        async def fake_limit(_s, plan, state, on_fill=None, reprice=None, **_k):
            state.update(filled=0.01); await on_fill()
            state.update(phase="filled_maker", filled=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(dict(b)); return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fn(c):
            self._armed({**self.SAVED, "sl_off": {"at": 1.0}})
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2550&sl=2480&mode=limit&limit=2490")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(60):
                await asyncio.sleep(0.02)
                if len(calls) >= 2:
                    break
            await asyncio.sleep(0.1)
            self.assertEqual(len(calls), 2, "체결 뒤 합친 전량에 TP 를 다시 걸어야 한다")
            self.assertTrue(self._bstate().get("sl_off"), "지운 손절은 꺼진 채")
        self._run(fn, acct=POS, run_entry_limit=fake_limit, place_bracket=fake_bracket)

    def test_wait_plain_add_keeps_user_changes_made_while_waiting(self) -> None:
        """기다리기 일반 진입: 대기 중 사용자가 지운 익절을 진입 때 계획으로 되살리던 것."""
        calls = []

        async def fake_wait(_s, plan, state, wait_sec, on_fill=None):
            state.update(filled=0.01); await on_fill()
            st = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
            st[f"{SYM}:LONG"]["tp_off"] = {"at": 2.0}; st[f"{SYM}:LONG"]["tp_resting"] = False; st[f"{SYM}:LONG"]["sl_level"] = 2470.0
            (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(st), "utf-8")
            state.update(phase="filled_maker", filled=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(dict(b)); return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fn(c):
            self._armed(self.SAVED)
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&mode=wait&wait_min=60")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(60):
                await asyncio.sleep(0.02)
                if len(calls) >= 2:
                    break
            await asyncio.sleep(0.1)
            s = self._bstate()
            self.assertTrue(s.get("tp_off"), s)
            self.assertEqual(s.get("sl_level"), 2470.0, "대기 중 옮긴 손절선이 되돌아가면 안 된다")
            self.assertEqual(len(calls), 1, "지운 익절을 다시 걸면 안 된다 -- 첫 체결 때 수량 맞춤 한 번뿐(2026-11-14 keep_rearm)")
        self._run(fn, acct=POS, run_entry_wait=fake_wait, place_bracket=fake_bracket)

    def test_rule_sizing_refused_on_lines_position(self) -> None:
        async def fn(c):
            self._armed({**self.SAVED, "sl_off": {"at": 1.0}})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
            self.assertEqual((r.status, (await r.json()).get("error")), (409, "rule_on_lines"))
        self._run(fn, acct=POS)

    def test_add_preview_shows_real_sl_distance_and_off_state(self) -> None:
        async def fn(c):
            self._armed({**self.SAVED, "sl_off": {"at": 1.0}})
            b = await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10")).json()
            br = b["plan"]["bracket"]
            self.assertTrue(br["sl_off"] and br["sl_pct"] is None, br)
            self.assertTrue(any("손절 지운 상태" in n for n in b["plan"]["notes"]), b["plan"]["notes"])
            self._armed(self.SAVED)
            br = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10")).json())["plan"]["bracket"]
            self.assertTrue(br["sl_pct"] is not None and br["sl_pct"] < -0.5, br)   # 대역 −1.0 고정이 아니라 실제 2480 거리
        self._run(fn, acct=POS)

    def test_add_on_tp_off_position_keeps_tp_off(self) -> None:
        # 2026-11-12 SOXL: 익절을 지운(tp_off) 선 포지션 -- 옛 익절선이 지금가 너머(2490 < 2500)여도 추가 진입 · 체결 뒤 익절 지정가 안 걸고 tp_off 유지
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(dict(b))
            return {"placed": True}

        async def fn(c):
            self._armed({**self.SAVED, "tp_level": 2490.0, "tp_off": {"at": 1.0}, "tp_resting": False})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10")
            self.assertEqual(r.status, 200, await r.json())
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(60):
                await asyncio.sleep(0.02)
                if calls:
                    break
            self.assertEqual(calls, [], "익절 지운 포지션 = 추가 진입 뒤에도 거래소에 아무것도 안 건다(2026-11-14 keep_rearm)")
            await asyncio.sleep(0.1)
            armed = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertTrue(armed["lines"] and armed["tp_off"] and not armed["tp_resting"], armed)
            self.assertEqual((armed["sl_level"], armed["sl_price"], armed["tp_price"]), (2480.0, 2479.6, 2550.4), armed)
        self._run(fn, acct=POS, run_entry=fake_entry, place_bracket=fake_bracket)

    def test_bracket_cleared_while_wait_add_pending_rearms(self) -> None:   # 검증 11-14(high): 대기 중 무장이 지워진 뒤 체결 = 새 포지션 -- 계획의 선으로 다시 무장(무방비 금지)
        calls = []

        async def fake_wait(_s, plan, state, wait_sec, on_fill=None):
            (server.LIVE_DIR / "manual_bracket_state.json").write_text("{}", "utf-8")   # 감시 루프 clear 가 무장을 지운 직후 peg 체결
            state.update(filled=0.01); await on_fill()
            state.update(phase="filled_maker", filled=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside, **k):
            calls.append(dict(b)); return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def fn(c):
            self._armed(self.SAVED)
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&mode=wait&wait_min=60")).json()
            self.assertTrue(body.get("ok"), body)
            await asyncio.sleep(0.5)
            st = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
            self.assertIn(f"{SYM}:LONG", st, "새로 열린 포지션에 무장이 없다(무방비)")
            self.assertEqual((st[f"{SYM}:LONG"]["sl_level"], st[f"{SYM}:LONG"]["tp_price"]), (2480.0, 2550.4), st)
        self._run(fn, acct=POS, run_entry_wait=fake_wait, place_bracket=fake_bracket)

    def test_line_tool_limit_add_keeps_lines(self) -> None:   # 검증 11-14: 추가 도구가 선을 비워 보내 지정가가 400 이던 것 -- 선(자리값)을 같이 보내고 서버는 무장된 조건을 쓴다
        async def fn(c):
            self._armed(self.SAVED)
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2600&sl=2400&limit=2490")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            self.assertTrue(b["plan"].get("add_keep") and b["plan"].get("limit"), b["plan"])
            self.assertEqual((b["plan"]["bracket"]["tp_level"], b["plan"]["bracket"]["sl_level"]), (2550.0, 2480.0))
        self._run(fn, acct=POS)

    def test_far_armed_tp_does_not_block_add(self) -> None:   # 2026-11-14 SOXL: 걸린 익절이 지금가에서 +17% -- 거리(15%)로 막지 않는다(방향은 아래 시험대로 본다)
        async def fn(c):
            self._armed({**self.SAVED, "tp_level": 2930.0, "tp_price": 2930.6})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480")
            body = await r.json()
            self.assertEqual(r.status, 200, body)
            self.assertTrue(body["plan"].get("add_lines"), body)
            self.assertEqual(body["plan"]["bracket"]["tp_level"], 2930.0, body)
        self._run(fn, acct=POS)

    def test_armed_lines_past_current_price_still_add(self) -> None:   # 2026-11-14 사용자 «추가 진입 = 기존 조건 그대로»: 선 검사(방향 포함)를 하지 않는다
        async def fn(c):
            self._armed({**self.SAVED, "tp_level": 2490.0})                                      # 익절선이 지금가(≈2500) 아래 = 이미 지난 선
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2550&sl=2480")
            b = await r.json()
            self.assertEqual(r.status, 200, b); self.assertEqual(b["plan"]["bracket"]["tp_level"], 2490.0, b)
        self._run(fn, acct=POS)

    def test_fill_rearms_same_lines_on_whole_position(self) -> None:
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            calls.append(dict(b))
            return {"placed": True, "tp": {"placed": True, "price": b["tp_price"]}}

        async def fn(c):
            self._armed(self.SAVED)
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2600&sl=2400")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(60):
                await asyncio.sleep(0.02)
                if calls:
                    break
            self.assertEqual(len(calls), 1, calls)
            self.assertEqual(calls[0], {"tp_price": 2550.4}, calls[0])   # 2026-11-14 추가 진입 = 무장은 그대로 · 익절 지정가만 합친 수량으로
            await asyncio.sleep(0.1)
            armed = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertTrue(armed["lines"] and armed["tp_resting"], armed)
            self.assertEqual((armed["sl_level"], armed["tp_level"], armed["sl_price"], armed["tp_price"]), (2480.0, 2550.0, 2479.6, 2550.4))
        self._run(fn, acct=POS, run_entry=fake_entry, place_bracket=fake_bracket)


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

    def test_move_sl_with_far_armed_tp(self) -> None:   # 2026-11-14: 익절이 +17% 여도 손절만 옮기기는 된다(거리 제한 없음 · 방향만)
        o = self._mod("&tp=2930&sl=2450", armed={"tp_level": 2930.0, "tp_price": 2930.6})
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual((o["state"]["sl_level"], o["state"]["tp_level"]), (2450.0, 2930.0), o["state"])
        o = self._mod("&tp=2930&sl=1900", armed={"tp_level": 2930.0, "tp_price": 2930.6})   # 손절 −24% 도 받는다(사용자 «제한 자체를 제거»)
        self.assertEqual((o["status"], o["state"]["sl_level"]), (200, 1900.0), o["body"])

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

    def test_lines_hugging_price_go_out_as_dragged(self) -> None:   # 2026-11-11 사용자 «현재가에 최대한 붙여도 끌어 놓은 곳으로 바로 지정가»
        calls = []

        async def place(_s, b, symbol, pside, **k):
            calls.append(b["tp_price"])
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": True, "price": b["tp_price"]}, "placed": True}
        o = self._mod("&tp=2500.05&sl=2499.95", place=place)   # 지금가 2500 에서 ±0.002%
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual((calls, o["state"]["tp_level"], o["state"]["sl_level"]), ([2500.05], 2500.05, 2499.95), o["state"])
        self.assertEqual(self._mod("&tp=2499.9&sl=2480")["status"], 400, "익절이 이미 지금가 너머(롱인데 아래)면 거절 -- 방향 검사는 남긴다")

    def test_sl_direction_kept_when_only_tp_moves(self) -> None:   # 2026-11-12 검토: 이미 넘어간 손절(롱 2510 · 지금가 2500)이 익절 수정 한 번에 «올라 닿으면»으로 뒤집혔다
        async def place(_s, b, symbol, pside, **k):
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": True, "price": b["tp_price"]}, "placed": True}
        o = self._mod("&tp=2560&sl=2510", armed={"sl_level": 2510.0}, place=place)
        self.assertEqual((o["status"], o["state"]["sl_up"]), (200, False), o["body"])

    def test_sl_direction_from_where_user_dropped_it(self) -> None:   # 화면이 놓은 순간 차트 기준 방향(slup) -- 서버 지금가가 이미 넘어갔어도 «내려 닿으면»
        self.assertIs(self._mod("&tp=2550&sl=2500.5&slup=0")["state"]["sl_up"], False)
        self.assertIs(self._mod("&tp=2550&sl=2500.5")["state"]["sl_up"], True, "옛 화면(slup 없음) = 서버 지금가 기준 그대로")
        self.assertIs(self._mod("&tp=2550&sl=2520&slup=1")["state"]["sl_up"], True, "롱 손절을 위로 = 감시형(11-10 사용자 기능) 유지")

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
        o = self._mod("&tp=2550&sl=2440", acct=far)                                          # 지금가에서 2.4% · 진입가에서 6.2% -- 2026-11-04 규칙 손절폭 한도를 뺐다
        self.assertNotEqual(o["body"].get("error"), "sl_too_far", o["body"])
        self.assertEqual(self._mod("&tp=2560&sl=2490", armed={"fired": True})["status"], 409)
        self.assertEqual(self._mod("&tp=2560&sl=2490", armed={"lines": False})["status"], 409)
        self.assertEqual(self._mod("&tp=2560&sl=2490", acct=FLAT)["status"], 409)             # 포지션 없음

    def test_confirm_required(self) -> None:
        async def fn(c):
            self._armed()
            r = await c.post(f"/api/manual-bracket/modify?side=LONG&symbol={SYM}&tp=2560&sl=2490")
            self.assertEqual(r.status, 400)
        self._run(fn, acct=POS)


class LineAddTest(ChartLinesTest):
    """2026-11-11 «손절/익절 라인 추가»(POST /api/manual-bracket/add · 진입 팝업 «추가» → 초안 선 끌어 놓기). 시세 = 가짜 bookTicker 2500 · 계좌 POS(롱 2500). 실주문 0."""
    _armed = LinesArmedGuardTest._armed

    def _add(self, q, acct=POS, armed=None, place=None, write=True):
        out = {}

        async def fn(c):
            if write:
                self._armed({"tp_price": 2550.0, "sl_price": 2480.0, **(armed or {})})
            out["t0"] = time.time()
            r = await c.post(f"/api/manual-bracket/add?side=LONG&symbol={SYM}&confirm=1{q}")
            out["status"], out["body"] = r.status, await r.json()
            f = server.LIVE_DIR / "manual_bracket_state.json"
            out["state"] = json.loads(f.read_text("utf-8")).get(f"{SYM}:LONG") if f.exists() else None
        self._run(fn, acct=acct, **({"place_bracket": place} if place else {}))
        return out

    @staticmethod
    def _place(calls):
        async def place(_s, b, symbol, pside, **k):
            calls.append((dict(b), k.get("strict")))
            return {"qty": 0.5, "cleared": {"cancelled": 0, "errors": []}, "tp": {"placed": True, "price": b["tp_price"]}, "placed": True}
        return place

    def test_sl_on_position_without_bracket_makes_new_lines_arm(self) -> None:
        o = self._add("&leg=sl&level=2470", write=False)   # 무장 파일 없음 = 앱으로 들어간 «손절 없음» 포지션 · place_bracket = _boom(거래소 주문 없음)
        self.assertEqual(o["status"], 200, o["body"])
        b = o["state"]
        self.assertTrue(b["lines"] and b["tp_off"]["never"] and "sl_off" not in b, b)
        self.assertEqual((b["sl_level"], b["sl_up"]), (2470.0, False))
        self.assertGreaterEqual(b["watch_from"], o["t0"])

    def test_tp_on_position_without_bracket_places_only_tp(self) -> None:
        calls = []
        o = self._add("&leg=tp&level=2560", write=False, place=self._place(calls))
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [({"tp_price": 2560.0}, True)])
        b = o["state"]
        self.assertTrue(b["sl_off"]["never"] and b["tp_resting"] and b["tp_level"] == 2560.0, b)

    def test_readd_removed_sl_at_new_price(self) -> None:
        o = self._add("&leg=sl&level=2490", armed={"sl_off": {"at": 1.0}})
        self.assertEqual(o["status"], 200, o["body"])
        self.assertTrue("sl_off" not in o["state"] and o["state"]["sl_level"] == 2490.0 and o["state"]["tp_level"] == 2550.0, o["state"])

    def test_refusals(self) -> None:
        o = self._add("&leg=sl&level=2470")                                                                  # 손절선이 이미 있다 = 분할 손절 조각 하나 더(2026-11-11)
        self.assertEqual((o["status"], len(o["state"]["sl_legs"])), (200, 2), o["body"])
        self.assertEqual(self._add("&leg=sl&level=2470", armed={"lines": False})["body"]["error"], "rule_armed")
        self.assertEqual(self._add("&leg=sl&level=2470", armed={"sl_off": {"at": 1.0}, "fired": "tp"})["status"], 409)
        self.assertEqual(self._add("&leg=tp&level=2450", armed={"tp_off": {"at": 1.0}})["status"], 400)       # 롱 익절이 지금가 아래
        self.assertEqual(self._add("&leg=sl&level=2470", acct=FLAT, write=False)["body"]["error"], "no_position")
        self.assertEqual(self._add("&leg=x&level=2470")["status"], 400)
        o = self._add("&leg=sl&level=2470", write=False, armed=None)
        self.assertEqual(o["status"], 200)

    def test_modify_ignores_removed_leg_even_if_out_of_range(self) -> None:
        async def fn(c):
            self._armed({"tp_price": 2550.0, "sl_price": 2000.0, "sl_level": 2000.0, "sl_off": {"at": 1.0}})   # 지운 손절은 지금가에서 20% -- 범위 밖
            r = await c.post(f"/api/manual-bracket/modify?side=LONG&symbol={SYM}&confirm=1&tp=2550&sl=2000")
            self.assertEqual(r.status, 200, await r.json())
            b = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertEqual((b["sl_level"], b["tp_level"]), (2000.0, 2550.0))
        self._run(fn, acct=POS)

    def test_rearm_refuses_never_armed_sl(self) -> None:
        async def fn(c):
            self._armed({"sl_level": None, "sl_off": {"at": 1.0, "never": True}})
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "no_sl_line"))
        self._run(fn, acct=POS)


class TpLadderTest(ChartLinesTest):
    """2026-11-11 분할 익절 1단계: 익절선 하나 더(add) · 조각 끌기(modify tpi) · 조각 지우기·무게(tp-legs) · 추가 진입 막기. 거래소 = 가짜 place_tp_ladder(실주문 0)."""
    _armed = LinesArmedGuardTest._armed

    @staticmethod
    def _ladder(calls, filled=()):
        async def fake(_s, legs, symbol, pside):
            calls.append([(round(x["level"], 2), round(float(x.get("w") or 0), 3)) for x in legs])
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": [], "tp_open": []}, "placed": True,
                    "legs": [{**x, "resting": True, "qty": 0.25, "cid": f"dbtpL{i}", "filled": round(x["level"], 2) in filled} for i, x in enumerate(legs)]}
        return fake

    def _post(self, path, armed, calls, filled=()):
        out = {}

        async def fn(c):
            self._armed({"tp_price": 2550.0, "sl_price": 2480.0, **armed})
            r = await c.post(f"/api/manual-bracket/{path}&side=LONG&symbol={SYM}&confirm=1")
            out["status"], out["body"] = r.status, await r.json()
            out["state"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
        self._run(fn, acct=POS, place_tp_ladder=self._ladder(calls, filled))
        return out

    LEGS = {"tp_legs": [{"level": 2550.0, "price": 2550.0, "w": 1.0, "resting": True, "cid": "dbtpL0"},
                        {"level": 2600.0, "price": 2600.0, "w": 1.0, "resting": True, "cid": "dbtpL1"}]}

    def test_add_second_tp_makes_two_equal_legs(self) -> None:
        calls = []
        o = self._post("add?leg=tp&level=2600", {}, calls)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [[(2550.0, 1.0), (2600.0, 1.0)]])
        b = o["state"]
        self.assertEqual((len(b["tp_legs"]), b["tp_level"], b["tp_price"], b["tp_resting"]), (2, 2550.0, 2550.0, True), b)

    def test_partial_cancel_hands_legs_to_server_watch(self) -> None:   # 2026-11-12 검토: 조각 하나 취소 실패 = 다른 조각은 이미 지워졌는데 «아무것도 안 바뀜» + 전부 resting 으로 남아 익절 무방비
        out = {}

        async def fake(_s, legs, symbol, pside):
            return {"qty": 0.5, "placed": False, "reason": "옛 익절 정리 실패", "cleared": {"cancelled": 1, "errors": ["-2011 Unknown order sent."], "tp_open": ["dbtpL0", "dbtpL1"]}}

        async def fn(c):
            self._armed({"tp_price": 2550.0, "sl_price": 2480.0, **self.LEGS})
            r = await c.post(f"/api/manual-bracket/tp-legs?kind=tp&act=weights&w=1,3&side=LONG&symbol={SYM}&confirm=1")
            out["status"], out["body"] = r.status, await r.json()
            out["state"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
        self._run(fn, acct=POS, place_tp_ladder=fake)
        b = out["state"]
        self.assertEqual((out["status"], out["body"]["error"]), (502, "tp_partial_cancel"), out["body"])
        self.assertEqual((b["tp_resting"], [x["resting"] for x in b["tp_legs"]], [x["w"] for x in b["tp_legs"]]), (False, [False, False], [1.0, 1.0]), b)
        from scripts.live_manual_peg_entry_20260912 import line_hit
        self.assertEqual(line_hit("LONG", b, (2540.0, 2551.0)), "tp", "이제 서버가 닿으면 닫는다")

    def test_add_refusals(self) -> None:
        self.assertEqual(self._post("add?leg=tp&level=2551", {}, [])["body"]["error"], "same_level")
        four = {"tp_legs": [{"level": 2550.0 + 20 * i, "price": 2550.0 + 20 * i, "w": 1.0, "resting": True} for i in range(4)]}
        self.assertEqual(self._post("add?leg=tp&level=2700", four, [])["body"]["error"], "too_many")

    def test_drag_one_leg_replaces_ladder_with_new_price(self) -> None:
        calls = []
        o = self._post("modify?tp=2620&sl=2480&tpi=1", self.LEGS, calls)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [[(2550.0, 1.0), (2620.0, 1.0)]])
        self.assertEqual([x["level"] for x in o["state"]["tp_legs"]], [2550.0, 2620.0])

    def test_drag_sl_only_keeps_ladder(self) -> None:
        calls = []
        o = self._post("modify?tp=2550&sl=2470", self.LEGS, calls)   # tpi 없음 = 손절만 -- 익절 조각은 거래소에서 안 건드린다
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [])
        self.assertEqual((o["state"]["sl_level"], len(o["state"]["tp_legs"])), (2470.0, 2))

    def test_remove_leg_collapses_to_single(self) -> None:
        calls = []
        o = self._post("tp-legs?act=remove&i=0", self.LEGS, calls)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [[(2600.0, 1.0)]])
        b = o["state"]
        self.assertTrue("tp_legs" not in b and b["tp_level"] == 2600.0 and b["tp_resting"], b)

    def test_weights_and_filled_leg_dropped(self) -> None:
        calls = []
        o = self._post("tp-legs?act=weights&w=1,3", self.LEGS, calls, filled=(2550.0,))   # 거래소에 없던 조각(체결) = 빠진다
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual(calls, [[(2550.0, 1.0), (2600.0, 3.0)]])
        self.assertTrue("tp_legs" not in o["state"] and o["state"]["tp_level"] == 2600.0, o["state"])
        self.assertEqual(self._post("tp-legs?act=weights&w=1", self.LEGS, [])["status"], 400)
        self.assertEqual(self._post("tp-legs?act=remove&i=0", {}, [])["body"]["error"], "no_legs")
        self.assertEqual(self._post("tp-legs?act=remove&i=0", {**self.LEGS, "fired": True}, [])["status"], 409)

    def test_same_side_add_keeps_ladder(self) -> None:   # 2026-11-14 «추가 진입 = 기존 조건 그대로»: 나눈 익절 조각 그대로 · 체결 뒤 조각을 합친 수량으로 다시
        calls = []

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fn(c):
            self._armed({"tp_price": 2550.4, "sl_price": 2479.6, "sl_name": "선 손절 2480.00", **self.LEGS})
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=10&tp=2600&sl=2400")   # 보낸 선은 쓰지 않는다
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            self.assertTrue(b["plan"].get("add_keep"), b["plan"])
            body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10")).json()
            self.assertTrue(body.get("ok"), body)
            for _ in range(80):
                await asyncio.sleep(0.02)
                if calls:
                    break
            self.assertEqual(calls, [[(2550.0, 1.0), (2600.0, 1.0)]])
            await asyncio.sleep(0.1)
            st = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
            self.assertEqual((len(st["tp_legs"]), st["sl_level"], st["sl_price"]), (2, 2480.0, 2479.6), st)
        self._run(fn, acct=POS, run_entry=fake_entry, place_tp_ladder=self._ladder(calls))


class SlLadderTest(ChartLinesTest):
    """2026-11-11 분할 손절: 손절 조각 하나 더(add) · 조각 끌기(modify sli) · 조각 지우기·무게(tp-legs kind=sl) -- 서버 감시 레벨뿐이라 거래소 주문 0(place_* = _boom)."""
    _armed = LinesArmedGuardTest._armed
    LEGS = {"sl_legs": [{"level": 2480.0, "price": 2480.0, "up": False, "w": 1.0}, {"level": 2460.0, "price": 2460.0, "up": False, "w": 1.0}]}

    def _post(self, path, armed):
        out = {}

        async def fn(c):
            self._armed({"tp_price": 2550.0, "sl_price": 2480.0, **armed})
            r = await c.post(f"/api/manual-bracket/{path}&side=LONG&symbol={SYM}&confirm=1")
            out["status"], out["body"] = r.status, await r.json()
            out["state"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[f"{SYM}:LONG"]
        self._run(fn, acct=POS)
        return out

    def test_add_second_sl_leg(self) -> None:
        o = self._post("add?leg=sl&level=2460", {})
        self.assertEqual(o["status"], 200, o["body"])
        b = o["state"]
        self.assertEqual(([x["level"] for x in b["sl_legs"]], [x["w"] for x in b["sl_legs"]], b["sl_level"]), ([2480.0, 2460.0], [1.0, 1.0], 2480.0), b)
        self.assertEqual(self._post("add?leg=sl&level=2480.5", {})["body"]["error"], "same_level")

    def test_drag_one_sl_leg(self) -> None:
        o = self._post("modify?tp=2550&sl=2450&sli=1", self.LEGS)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertEqual([x["level"] for x in o["state"]["sl_legs"]], [2480.0, 2450.0])
        o = self._post("modify?tp=2550&sl=2470", self.LEGS)   # sli 없음(익절 꼬리표를 놓은 경우) -- 손절 조각은 그대로
        self.assertEqual([x["level"] for x in o["state"]["sl_legs"]], [2480.0, 2460.0])

    def test_refuse_while_leg_fired(self) -> None:
        fired = {"sl_legs": [{**self.LEGS["sl_legs"][0], "fired": True}, self.LEGS["sl_legs"][1]]}
        self.assertEqual(self._post("modify?tp=2550&sl=2450&sli=1", fired)["status"], 409)
        self.assertEqual(self._post("tp-legs?kind=sl&act=remove&i=1", fired)["status"], 409)

    def test_remove_and_weights(self) -> None:
        o = self._post("tp-legs?kind=sl&act=weights&w=1,3", self.LEGS)
        self.assertEqual((o["status"], [x["w"] for x in o["state"]["sl_legs"]]), (200, [1.0, 3.0]), o["body"])
        o = self._post("tp-legs?kind=sl&act=remove&i=0", self.LEGS)
        self.assertEqual(o["status"], 200, o["body"])
        self.assertTrue("sl_legs" not in o["state"] and o["state"]["sl_level"] == 2460.0, o["state"])


class SlLadderTickTest(LineTickTest):
    """감시 한 틱: 손절 조각 하나 닿음 → 그 몫만 청산(익절 지정가는 안 지움) → 조각을 빼고 남은 손절로 · 청산 실패면 쏜 표시를 남겨 다시."""
    LEGS = {"sl_legs": [{"level": 2480.0, "price": 2480.0, "up": False, "w": 1.0}, {"level": 2460.0, "price": 2460.0, "up": False, "w": 1.0}]}

    def test_one_leg_partial_exit(self) -> None:
        calls, st = self._tick([(2001, 2490.0), (2002, 2479.5)], armed_at=1999.0, armed=self.LEGS)
        self.assertEqual(calls, [("exit", "LONG")], "익절 지정가를 지우지 않고 그 몫만 청산")
        b = st[f"{SYM}:LONG"]
        self.assertTrue("sl_legs" not in b and b["sl_level"] == 2460.0 and not b.get("fired"), b)

    def test_all_legs_full_exit(self) -> None:
        calls, st = self._tick([(2001, 2490.0), (2002, 2455.0)], armed_at=1999.0, armed=self.LEGS)
        self.assertEqual([c[0] for c in calls][:2], ["clear", "exit"], calls)
        self.assertEqual(st, {})

    def test_failed_leg_exit_keeps_fired(self) -> None:
        calls, st = self._tick([(2002, 2479.5)], armed_at=1999.0, armed=self.LEGS, exit_phase="error")
        legs = st[f"{SYM}:LONG"]["sl_legs"]
        self.assertEqual([x.get("fired") for x in legs], [True, None])
        self.assertGreater(st[f"{SYM}:LONG"]["retry_after"], time.time())


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


class WatcherOwnSymbolTest(unittest.TestCase):
    """2026-11-12 검토: 지키던 USDC 롱이 이미 닫혔는데(30초 묵은 계좌엔 남음) 같은 쪽 USDT 롱이 있으면 감시 틱이 그 USDT 포지션을 닫았다 -- 무장한 심볼만 닫는다."""
    def test_does_not_close_other_symbol(self) -> None:
        mine, other = orders.MANUAL_EXEC_SYMBOLS["btc"], orders.MARKET_SYMBOLS["btc"]   # 주문 심볼 ≠ 시세 심볼인 코인(BTCUSDC · BTCUSDT)
        p = POS["positions"][0]
        stale = {**POS, "positions": [{**p, "symbol": mine}], "generated_at": "2000-01-01T00:00:00+00:00"}
        fresh = {**POS, "positions": [{**p, "symbol": other, "qty": 3.0}]}
        calls = []

        async def fresh_acct(*_a, **_k):
            return fresh

        async def swr(_k, _ttl, _produce, **_kw):
            return stale

        async def clear(_s, symbol, _pside, *_a, **_k):
            calls.append(("clear", symbol))
            return {"cancelled": 0}

        async def exit_(_s, plan, state):
            calls.append(("run_exit", plan["symbol"]))
            state.update(phase="filled_maker", filled=plan["quantity"], quantity=plan["quantity"])
            return state

        async def hist(_a):
            return []

        async def go():
            with mock.patch.object(orders, "fetch_account", fresh_acct), mock.patch.object(orders, "produce_account", fresh_acct, create=True), \
                 mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.multiple(orders, clear_bracket=clear, run_exit=exit_, run_entry=_boom, place_bracket=_boom, place_add_order=_boom):
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                      fetch_binance_json=_fake_binance, swr_cached=swr, _timed_produce=None, load_market_history=hist,
                                      situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=lambda a: [(time.time(), 2495.0)])
                (orders.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({f"{mine}:LONG": {
                    "symbol": mine, "side": "LONG", "lines": True, "sl_level": 2480.0, "tp_level": 2550.0, "tp_resting": False,
                    "fired": True, "backstop_price": None, "armed_at": 1999.0}}), "utf-8")
                await api.bracket_tick()
        self.assertNotEqual(mine, other)
        with _isolated_dirs():
            asyncio.run(go())
        self.assertNotIn(("run_exit", other), calls, calls)


if __name__ == "__main__":
    unittest.main()
