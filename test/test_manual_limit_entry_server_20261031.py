"""진입 «지정가 대기» 서버 경로(2026-10-31) -- 접수·수정·취소. 🔴실주문 0(집행 함수 전부 가짜 · 부르면 AssertionError).
① 미리보기: 지정가 = 진입선 · 명목 유지 · 선 기준이 지정가  ② 호가를 넘은 지정가·선 없는 지정가 = 400
③ submit mode=limit → 별도 칸(manual_wait_state, mode=limit) · 집행은 run_entry_limit 하나
④ 수정: 가격이 바뀌면 amend(가격·수량) · 선만 바뀌면 주문 그대로 · 체결 시작 뒤·주문 없음 = 409 · confirm 필수
python -m pytest -q test/test_manual_limit_entry_server_20261031.py"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_chart_lines_entry_20261028 import ChartLinesTest, FLAT  # noqa: E402

from dashboard import orders  # noqa: E402

Q = "lev=20&pct=10"


class LimitServerTest(ChartLinesTest):
    def test_preview_limit_is_entry_line_and_keeps_notional(self) -> None:
        async def fn(c):
            base = (await (await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&tp=2550&sl=2450")).json())["plan"]
            r = await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&tp=2550&sl=2450&limit=2480")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            pl = b["plan"]
            self.assertEqual((pl["price"], pl["limit"]["price"]), (2480.0, 2480.0))
            self.assertEqual((pl["bracket"]["tp_level"], pl["bracket"]["sl_level"]), (2550.0, 2450.0))
            self.assertAlmostEqual(pl["notional_usdt"], base["notional_usdt"], delta=base["notional_usdt"] * 0.01)   # 명목 유지(수량만 가격비로)
            self.assertGreater(pl["quantity"], base["quantity"])   # 더 싼 가격이라 같은 명목에 수량이 많다
        self._run(fn)

    def test_short_limit_rounds_up_and_marketable_is_rejected(self) -> None:
        async def fn(c):
            r = await c.get(f"/api/manual-entry/preview?side=SHORT&{Q}&tp=2450&sl=2550&limit=2520.003")
            b = await r.json()
            self.assertEqual((r.status, b["plan"]["price"]), (200, 2520.01), b)   # 숏 = 올림(메이커 쪽)
            for side, lim, tp, sl in (("LONG", 2501, 2550, 2450), ("LONG", 2500.01, 2550, 2450), ("SHORT", 2499, 2450, 2550)):
                r = await c.get(f"/api/manual-entry/preview?side={side}&{Q}&tp={tp}&sl={sl}&limit={lim}")
                self.assertEqual((r.status, (await r.json())["error"]), (400, "limit_marketable"), (side, lim))
        self._run(fn)

    def test_limit_needs_lines_and_valid_direction(self) -> None:
        async def fn(c):
            r = await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&limit=2480")
            self.assertEqual((r.status, (await r.json())["error"]), (400, "bad_limit"))
            r = await c.get(f"/api/manual-entry/preview?side=LONG&{Q}&tp=2470&sl=2450&limit=2480")   # 익절이 지정가 아래
            self.assertEqual(r.status, 400, await r.json())
        self._run(fn)

    def test_submit_amend_cancel_flow(self) -> None:
        seen, flags = [], {}

        async def fake_limit(_s, plan, state, max_sec=0, on_fill=None, reprice=None):
            seen.append(dict(plan))
            state.update(phase="working", mode="limit", order_id=5, filled=0.0, limit_price=plan["price"])
            while not state.get("action"):
                await asyncio.sleep(0.01)
                if flags.get("consume") and state.get("amend"):   # 집행 루프가 수정을 받아 간 것을 흉내
                    state.pop("amend")
                    flags["consume"] = False
                if flags.get("fill"):
                    state["filled"] = 0.2
                elif state.get("filled"):
                    state["filled"] = 0.0
            state.update(phase="cancelled")
            return state

        async def fn(c):
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2550&sl=2450&limit=2470&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "no_limit"))   # 주문 없음
            r = await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2550&sl=2450&limit=2480&mode=limit&confirm=1")
            b = await r.json()
            self.assertEqual(r.status, 200, b)
            await asyncio.sleep(0.1)
            st = (await (await c.get("/api/manual-entry/status")).json())["wait"]
            self.assertEqual((st["mode"], st["phase"], seen[0]["price"]), ("limit", "working", 2480.0))
            from dashboard import server
            self.assertTrue((server.LIVE_DIR / "manual_limit_wait.json").exists())   # 재시작 복구용 저장(접수 때)
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2550&sl=2450&limit=2470")   # confirm 없음
            self.assertEqual(r.status, 400)
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2560&sl=2440&limit=2470&confirm=1")
            b = await r.json()
            self.assertEqual((r.status, b["moved"], b["price"]), (200, True, 2470.0), b)
            st = (await (await c.get("/api/manual-entry/status")).json())["wait"]
            self.assertEqual((st["amend"]["price"], st["plan"]["bracket"]["tp_level"], st["plan"]["bracket"]["sl_level"]), (2470.0, 2560.0, 2440.0))
            self.assertAlmostEqual(st["amend"]["qty"], seen[0]["quantity"] * 2480 / 2470, delta=seen[0]["quantity"] * 0.01)   # 검증 D1: 접수 때 레버리지(lev=20)로 다시 잰다 -- 모델 기본값으로 떨어지면 수량이 달라진다
            r = await c.post("/api/manual-entry/wait-action?action=market&confirm=1")   # 검증 D5: 지정가 대기는 시장가 전환 없음
            self.assertEqual((r.status, (await r.json())["error"]), (400, "limit_no_market"))
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2565&sl=2440&limit=2470&confirm=1")   # 앞 수정이 아직 반영 전
            self.assertEqual((r.status, (await r.json())["error"]), (409, "amend_pending"))
            flags["consume"] = True
            await asyncio.sleep(0.1)
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2565&sl=2440&limit=2470&confirm=1")   # 가격 그대로 = 주문 안 건드리고 선만
            b = await r.json()
            self.assertEqual((r.status, b["moved"]), (200, False), b)
            self.assertNotIn("amend", (await (await c.get("/api/manual-entry/status")).json())["wait"])
            flags["fill"] = True
            await asyncio.sleep(0.1)
            r = await c.post(f"/api/manual-entry/limit-amend?{Q}&tp=2565&sl=2440&limit=2465&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "partially_filled"))
            flags["fill"] = False
            r = await c.post("/api/manual-entry/wait-action?action=cancel&confirm=1")
            self.assertEqual(r.status, 200)
            await asyncio.sleep(0.1)
            self.assertEqual((await (await c.get("/api/manual-entry/status")).json())["wait"]["phase"], "cancelled")
            self.assertFalse((server.LIVE_DIR / "manual_limit_wait.json").exists())   # 정상 종료하면 지운다
        self._run(fn, run_entry_limit=fake_limit)

    def test_reprice_callback_ignores_unread_basis_and_never_crosses(self) -> None:
        got = []

        async def fake_limit(_s, plan, state, max_sec=0, on_fill=None, reprice=None):
            state.update(phase="working", mode="limit", order_id=5, filled=0.0)
            got.append(await reprice())                  # 모의 호가는 두 심볼이 같아 basis 가 정확히 1.0 = «못 읽음»과 구별 불가 → 건드리지 않는다(검증 S1)
            plan["limit"]["usdt"] = 2501.0               # 선이 호가 위로 간 경우(= 지금 체결) → 건드리지 않는다
            got.append(await reprice())
            return state

        async def fn(c):
            r = await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2550&sl=2450&limit=2480&mode=limit&confirm=1")
            self.assertEqual(r.status, 200, await r.json())
            await asyncio.sleep(0.3)
        self._run(fn, run_entry_limit=fake_limit)
        self.assertEqual(got, [None, None])   # 양의 경로(목표가 계산)는 집행 루프 시험이 가짜 reprice 로 덮는다

    # ── 재시작 복구(검증 D2): 대기 중 서버가 죽은 사이 체결된 몫에 익절·손절을 건다 ─────────────────────────
    def _restart(self, *, saved=True, positions=None, armed=False, account_ok=True):
        import json
        from aiohttp.test_utils import TestClient, TestServer
        from dashboard import server
        from test_manual_entry_preview_smoke_20260913 import _isolated_dirs
        sym = orders.MANUAL_EXEC_SYMBOLS["eth"]
        placed, out = [], {}

        async def fake_orphans(_s, symbols):
            return {"cancelled": [], "errors": []}

        async def fake_account(*_a, **_k):
            return {**FLAT, "ok": account_ok, "positions": positions or []}

        async def fake_bracket(_s, b, symbol, pside, **_k):
            placed.append((symbol, pside, dict(b)))
            return {"placed": True, "tp": {"placed": True, "price": b.get("tp_price")}}

        async def exercise():
            if saved:
                (server.LIVE_DIR / "manual_limit_wait.json").write_text(json.dumps({"saved_at": 1.0, "plan": {
                    "symbol": sym, "positionSide": "LONG", "price": 2480.0, "quantity": 0.5,
                    "bracket": {"available": True, "lines": True, "tp_price": 2550.0, "tp_level": 2550.0, "sl_price": 2450.0, "sl_level": 2450.0,
                                "backstop_price": None, "rule": False, "sl_name": "선 손절 2450.00"}}}), "utf-8")
            if armed:
                (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({f"{sym}:LONG": {"symbol": sym, "side": "LONG", "lines": True,
                                                                                                    "sl_level": 2450.0, "tp_level": 2550.0, "armed_at": 1.0}}), "utf-8")
            from unittest import mock
            with mock.patch.object(orders, "exec_enabled", lambda: True), mock.patch.object(orders, "cancel_wait_orphans", fake_orphans), \
                 mock.patch.object(orders, "fetch_account", fake_account), mock.patch.object(orders, "place_bracket", fake_bracket):
                app = server.make_app()
                keep = {"start_http_session", "start_wait_cleanup", "stop_http_session"}
                for sig in (app.on_startup, app.on_cleanup):
                    sig[:] = [f for f in sig if getattr(f, "__name__", "") in keep]
                c = TestClient(TestServer(app))
                await c.start_server()
                try:
                    await asyncio.sleep(0.3)
                    out["wait"] = (await (await c.get("/api/manual-entry/status")).json())["wait"]
                    out["file"] = (server.LIVE_DIR / "manual_limit_wait.json").exists()
                    out["armed"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8")) if (server.LIVE_DIR / "manual_bracket_state.json").exists() else {}
                finally:
                    await c.close()
        with _isolated_dirs():
            asyncio.run(exercise())
        return placed, out, sym

    def _pos(self, sym):
        return [{"symbol": sym, "side": "LONG", "qty": 0.5, "entry_price": 2480.0, "mark_price": 2480.0, "liquidation_price": 1500.0,
                 "leverage": 20.0, "notional": 1240.0, "unrealized_pnl": 0.0, "updated_at": None}]

    def test_restart_arms_position_filled_while_server_was_down(self) -> None:
        sym = orders.MANUAL_EXEC_SYMBOLS["eth"]
        placed, out, _ = self._restart(positions=self._pos(sym))
        self.assertEqual([(p[0], p[1], p[2]["tp_price"]) for p in placed], [(sym, "LONG", 2550.0)])   # 저장한 선으로 익절 지정가
        self.assertTrue(out["armed"][f"{sym}:LONG"]["lines"])                                          # 손절 감시 무장
        self.assertEqual((out["file"], out["wait"]["recovered"]["armed"]), (False, True))

    def test_restart_without_position_or_with_existing_bracket_places_nothing(self) -> None:
        sym = orders.MANUAL_EXEC_SYMBOLS["eth"]
        placed, out, _ = self._restart(positions=[])
        self.assertEqual((placed, out["file"], out["wait"]["recovered"]["armed"]), ([], False, False))
        placed, out, _ = self._restart(positions=self._pos(sym), armed=True)   # 이미 무장돼 있으면(첫 체결에서 건 것) 다시 안 건다
        self.assertEqual((placed, out["file"]), ([], False))

    def test_restart_with_unreadable_account_keeps_file_and_says_so(self) -> None:
        placed, out, _ = self._restart(positions=[], account_ok=False)
        self.assertEqual((placed, out["file"]), ([], True))                       # 다음 기동에 다시
        self.assertIn("직접 확인", out["wait"]["recovered"]["reason"])

    def test_new_entry_blocked_while_limit_waits(self) -> None:
        async def fake_limit(_s, plan, state, max_sec=0, on_fill=None, reprice=None):
            state.update(phase="working", mode="limit", order_id=5, filled=0.0)
            await asyncio.sleep(1)
            return state

        async def fn(c):
            await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2550&sl=2450&limit=2480&mode=limit&confirm=1")
            await asyncio.sleep(0.05)
            r = await c.post(f"/api/manual-entry/submit?side=LONG&{Q}&tp=2550&sl=2450&limit=2470&mode=limit&confirm=1")
            self.assertEqual((r.status, (await r.json())["error"]), (409, "wait_pending"))   # 대기 몫이 명목에 안 잡히니 겹쳐 넣지 못한다
        self._run(fn, run_entry_limit=fake_limit)


if __name__ == "__main__":
    unittest.main()
