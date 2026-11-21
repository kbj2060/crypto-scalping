"""−5% 거래소 스탑·자동 물타기 빼기(2026-11-21 사용자 질문지) + «sltp(차트 선 주문 + 진입가 팝업 청산)만 쓴다»(같은 날).
🔴실주문 0 -- signed·시계·청산은 전부 가짜. ① 어떤 경로도 STOP_MARKET(algoOrder POST)을 안 건다 · 스탑·물타기·기다리기 집행 코드가 없다
② 선 진입은 익절 GTX 하나만 건다(스탑·물타기 POST 0) ③ 지운 방식(rule=·mode=wait·switch_to·sw_*·exec=market)·선 없는 진입은 주문 없이 거절 ·
선 없는 포지션에 추가 진입 409 · 지운 라우트 404 ④ 옛 규칙 무장(rule True)의 손절 = 서버 5분봉 종가 감시가 실제로 청산을 쏜다
⑤ 배포 전에 걸린 우리 스탑(dbsl)·물타기(dbad)는 포지션이 닫히면 지운다(Rust 와 다른 점) ⑥ 규칙 무장 «손절 다시 켜기» = 거래소 요청 없이 sl_off 만 지운다.
실행: python -m pytest -q test/test_stop_removal_20261121.py"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402
from dashboard import orders, server  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _fake_binance, _isolated_dirs  # noqa: E402

SYM = orders.MANUAL_EXEC_SYMBOLS["eth"]
KEY = f"{SYM}:LONG"
FAKE_ENV = {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}
ACCT = {"ok": True, "balance": {"margin": 1000.0, "initial_margin": 0.0, "available": 1000.0, "unrealized": 0.0},
        "positions": [], "leverage_by_symbol": {SYM: 20.0}, "trips": []}
HELD = {**ACCT, "positions": [{"symbol": SYM, "side": "LONG", "qty": 0.5, "entry_price": 2500.0, "mark_price": 2500.0, "liquidation_price": 1500.0,
                               "leverage": 20.0, "notional": 1250.0, "unrealized_pnl": 0.0, "updated_at": None}]}
RULE_ARM = {"symbol": SYM, "side": "LONG", "sl_price": 2306.98, "sl_level": 2309.25, "backstop_price": 2306.98, "rule": True,
            "sl_name": "첫 진입가 −5%", "armed_at": 1000.0, "placed": {"placed": True}}


async def _boom(*_a, **_k):
    raise AssertionError("실주문 경로가 불렸다 -- 시험에서 주문 금지")


async def _zero(*_a, **_k):
    return 0


class FakeEx:
    """거래소 가짜: 포지션 하나(held) · 배포 전에 걸린 우리 스탑 dbslL1 이 남아 있다 · opens = 걸린 일반 주문(예: 옛 자동 물타기 dbadL)."""

    def __init__(self, held=True, opens=()):
        self.held, self.calls, self.opens = held, [], [dict(o) for o in opens]

    async def signed(self, _s, method, path, params, *_a, **_k):
        self.calls.append((method, path, dict(params)))
        if path == "/fapi/v2/positionRisk":
            return [{"positionSide": "LONG", "positionAmt": "1.0" if self.held else "0"}]
        if path == "/fapi/v1/openOrders":
            return [dict(o) for o in self.opens]
        if path == "/fapi/v1/commissionRate":
            return {"makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"}
        if path == "/fapi/v1/openAlgoOrders":
            return [{"algoId": 9, "clientAlgoId": "dbslL1", "symbol": SYM}]
        if method == "DELETE":
            return {}
        if method == "POST":
            return {"orderId": 1, "algoId": 2}
        raise AssertionError(f"모르는 요청 {method} {path}")

    def stop_posts(self):
        return [c for c in self.calls if c[0] == "POST" and c[1] == "/fapi/v1/algoOrder"]

    def order_posts(self):
        return [c for c in self.calls if c[0] == "POST" and c[1] == "/fapi/v1/order"]

    def order_deletes(self):
        return [c[2].get("orderId") for c in self.calls if c[0] == "DELETE" and c[1] == "/fapi/v1/order"]

    def stop_deletes(self):
        return [c for c in self.calls if c[0] == "DELETE" and c[1] == "/fapi/v1/algoOrder"]


def fake_exchange(fx):
    return mock.patch.multiple(ex, signed=fx.signed, _clock_offset=_zero), mock.patch.dict(os.environ, FAKE_ENV)


class NoStopTest(unittest.TestCase):
    def test_place_bracket_posts_no_stop_but_clears_old_one(self) -> None:
        fx = FakeEx()
        p1, p2 = fake_exchange(fx)
        with p1, p2:
            out = asyncio.run(ex.place_bracket(None, {"tp_price": None, "sl_price": 2306.98, "backstop_price": 2306.98}, SYM, "LONG"))
        self.assertEqual(fx.stop_posts(), [], fx.calls)                 # 스탑 POST 0건
        self.assertEqual(len(fx.stop_deletes()), 1, fx.calls)           # 남은 dbsl 은 지운다(겹침 방지 정리 그대로)
        self.assertTrue(out["placed"], out)                             # «스탑 못 걸었다»로 실패 처리되지 않는다
        self.assertTrue(out["backstop"]["skipped"] and not out["backstop"]["placed"], out)

    def test_no_stop_code_left(self) -> None:
        src = inspect.getsource(ex)
        self.assertNotIn('"POST", "/fapi/v1/algoOrder"', src)            # 스탑을 거는 요청이 모듈 어디에도 없다
        self.assertFalse(hasattr(ex, "place_stop") or hasattr(ex, "_post_stop"))
        self.assertNotIn("place_stop(", inspect.getsource(orders))


def server_run(fn, acct=ACCT, **patches) -> FakeEx:
    """서버 한 벌 -- 진짜 계획·place_bracket·arm_bracket 이 돌고 거래소(signed)만 가짜 · 진입·청산 집행은 기본 _boom(부르면 실패)."""
    fx = FakeEx()

    async def fake_account(*_a, **_k):
        return acct

    async def exercise() -> None:
        p1, p2 = fake_exchange(fx)
        base = {"run_entry": _boom, "run_entry_limit": _boom, "run_entry_stop": _boom, "run_exit": _boom, **patches}
        with p1, p2, mock.patch.object(orders, "fetch_account", fake_account), \
             mock.patch.object(orders, "produce_account", fake_account, create=True), \
             mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
             mock.patch.object(orders, "exec_enabled", lambda: True), \
             mock.patch.multiple(orders, **base):
            client = TestClient(TestServer(offline_app(server)))
            await client.start_server()
            try:
                await fn(client)
            finally:
                await client.close()

    with _isolated_dirs():
        asyncio.run(exercise())
    return fx


async def _filled(_s, plan, state):
    state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
    return state


class LineEntryTest(unittest.TestCase):
    def test_line_entry_places_only_tp(self) -> None:
        box = {}

        async def fn(c):
            b = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2550&sl=2480")).json()
            self.assertTrue(b.get("ok"), b)
            for _ in range(100):
                await asyncio.sleep(0.02)
                if (server.LIVE_DIR / "manual_bracket_state.json").exists():
                    break
            await asyncio.sleep(0.1)
            box["st"] = (await (await c.get("/api/manual-entry/status")).json())["state"]
            box["armed"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
        fx = server_run(fn, run_entry=_filled)
        self.assertEqual(fx.stop_posts(), [], fx.calls)                                     # 스탑 POST 0건
        posts = fx.order_posts()
        self.assertEqual([p[2]["timeInForce"] for p in posts], ["GTX"], posts)              # 익절 지정가 하나뿐 -- 물타기 GTC 0건
        self.assertTrue(posts[0][2]["newClientOrderId"].startswith("dbtpL"), posts)
        st, a = box["st"], box["armed"][KEY]
        self.assertNotIn("add_order", st)
        self.assertNotIn("backstop", st["bracket"])
        self.assertTrue(a["lines"] and a["sl_level"] == 2480.0 and a["tp_resting"] and not a["rule"], a)
        self.assertNotIn("add_order", a)


class RemovedOptionsTest(unittest.TestCase):
    """옛 화면 캐시가 지운 방식을 보내도 «다른 방식으로 조용히» 나가지 않는다 -- 주문 0 으로 거절."""

    def test_entry_rejects_removed_modes_and_no_lines(self) -> None:
        async def fn(c):
            for q in ("&rule=c&tp=2550&sl=2480", "&rule=h&tp=2550&sl=2480", "&mode=wait&wait_min=60&tp=2550&sl=2480", "&mode=wait&wait_min=15"):
                for path in ("preview", "submit"):
                    r = await (c.get if path == "preview" else c.post)(f"/api/manual-entry/{path}?side=LONG&confirm=1&lev=20&pct=10{q}")
                    self.assertEqual((r.status, (await r.json()).get("error")), (400, "removed_option"), (path, q))
            for path in ("preview", "submit"):   # 선 없는 진입(옛 버튼) -- 옛 판은 −5% 규칙 손절로 넣었다
                r = await (c.get if path == "preview" else c.post)(f"/api/manual-entry/{path}?side=LONG&confirm=1&lev=20&pct=10")
                self.assertEqual((r.status, (await r.json()).get("error")), (400, "lines_required"), path)
            self.assertFalse((server.LIVE_DIR / "manual_bracket_state.json").exists())
        fx = server_run(fn)
        self.assertEqual([c for c in fx.calls if c[0] != "GET"], [])                        # 거래소에 쓰는 요청 0

    def test_add_on_unlined_position_refused(self) -> None:
        async def fn(c):
            for q in ("", "&mode=limit&limit=2490"):
                for path in ("preview", "submit"):
                    r = await (c.get if path == "preview" else c.post)(f"/api/manual-entry/{path}?side=LONG&confirm=1&lev=20&pct=10&tp=2550&sl=2480{q}")
                    self.assertEqual((r.status, (await r.json()).get("error")), (409, "add_on_unlined_position"), (path, q))
        fx = server_run(fn, acct=HELD)
        self.assertEqual([c for c in fx.calls if c[0] != "GET"], [])

    def test_exit_rejects_switch_and_market_but_takes_limit(self) -> None:
        seen = []

        async def fake_exit(_s, plan, state):
            seen.append(plan.get("type"))
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fn(c):
            for q in ("&switch_to=SHORT", "&switch_to=SHORT&sw_pct=50&sw_lev=10", "&sw_rule=c", "&exec=market"):
                r = await c.post(f"/api/manual-exit/submit?side=LONG&confirm=1&pct=100{q}")
                self.assertEqual((r.status, (await r.json()).get("error")), (400, "removed_option"), q)
            self.assertEqual(seen, [])
            r = await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=100&exec=limit&src=ep-chip")   # 옛 기본값 exec=limit 은 같은 동작이라 받는다
            self.assertEqual(r.status, 200, await r.text())
            await asyncio.sleep(0.1)
            self.assertEqual(len(seen), 1)
            self.assertEqual((await r.json())["plan"]["type"], "LIMIT")                        # «즉시» 시장가로 바꾸지 않았다(peg 따라가기)
        server_run(fn, acct=HELD, run_exit=fake_exit)

    def test_removed_routes_404(self) -> None:
        async def fn(c):
            self.assertEqual((await c.get("/api/manual-exit/preview?side=LONG")).status, 404)
            self.assertEqual((await c.post("/api/manual-exit/submit-all?confirm=1")).status, 404)
        server_run(fn, acct=HELD)

    def test_removed_code_is_gone(self) -> None:
        from scripts import live_manual_peg_entry_20260912 as pe
        for mod, names in ((ex, ("place_add_order", "run_entry_wait", "WAIT_CHOICES_MIN", "margin_type")),
                           (pe, ("commit_rule_l", "rec_rule_size", "rec_rule_add_order", "rec_rule_bracket", "RULE_STOP_PCT")),
                           (orders, ("REC_AUTO_ADD", "MANUAL_RULES_ENABLED", "RULE_ASSETS"))):
            for n in names:
                self.assertFalse(hasattr(mod, n), f"{mod.__name__}.{n}")
        src = inspect.getsource(orders)
        for frag in ("arm_rule_add", "exit_then_switch", "api_manual_exit_preview", "api_manual_exit_all_submit", "query_rule"):
            self.assertNotIn(frag, src)


def watch(armed: dict, *, held=True, bars=None, opens=()):
    """감시 틱 한 번 -- clear_bracket 은 진짜(거래소만 가짜) · 청산 run_exit 은 가짜. 돌려주기 = (청산 호출, 거래소 가짜, 저장본)."""
    exits, fx = [], FakeEx(held=held, opens=opens)
    pos = {"ok": True, "generated_at": "2100-01-01T00:00:00+00:00",
           "positions": [{"symbol": SYM, "side": "LONG", "qty": 0.5, "entry_price": 2400.0}] if held else []}

    async def fake_account(*_a, **_k):
        return pos

    async def swr(_k, _ttl, produce, **_kw):
        return await produce()

    async def exit_(_s, plan, state):
        exits.append(plan.get("symbol"))
        state.update(phase="filled_maker", filled=1.0, quantity=1.0)
        return state

    async def hist(_a):
        return bars or []

    async def go():
        p1, p2 = fake_exchange(fx)
        with p1, p2, mock.patch.object(orders, "fetch_account", fake_account), \
             mock.patch.object(orders, "exec_enabled", lambda: True), \
             mock.patch.multiple(orders, run_exit=exit_, run_entry=_boom, place_bracket=_boom):
            api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                  fetch_binance_json=_fake_binance, swr_cached=swr, _timed_produce=None,
                                  load_market_history=hist, situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60,
                                  sec_prices=lambda a: None)
            (orders.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({KEY: armed}), "utf-8")
            await api.bracket_tick()
        return exits, fx, json.loads((orders.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))

    with _isolated_dirs():
        return asyncio.run(go())


class RuleWatchTest(unittest.TestCase):
    def test_rule_arm_fires_on_5m_close_below_sl(self) -> None:
        now = time.time()
        closed = now - 2 * orders.FOOTPRINT_BAR_SECONDS
        exits, fx, _ = watch({**RULE_ARM, "armed_at": now - 3600}, bars=[{"time": closed, "close": 2300.0}])
        self.assertEqual(exits, [SYM])                                    # 스탑 없이 서버 감시가 손절 청산을 쏜다
        self.assertEqual(fx.stop_posts(), [])
        self.assertEqual(len(fx.stop_deletes()), 1, fx.calls)             # 다 닫힌 뒤 남은 dbsl 정리

    def test_rule_arm_holds_above_sl(self) -> None:
        now = time.time()
        exits, _, st = watch({**RULE_ARM, "armed_at": now - 3600},
                             bars=[{"time": now - 2 * orders.FOOTPRINT_BAR_SECONDS, "close": 2400.0}])
        self.assertEqual(exits, [])
        self.assertIn(KEY, st)

    def test_closed_position_cancels_leftover_stop(self) -> None:
        exits, fx, st = watch({**RULE_ARM, "armed_at": 1000.0}, held=False,
                              opens=[{"orderId": 7, "clientOrderId": "dbadL7"}, {"orderId": 8, "clientOrderId": "web_user"}])
        self.assertEqual(exits, [])
        self.assertEqual([c[2].get("algoId") for c in fx.stop_deletes()], [9], fx.calls)   # 배포 전 스탑이 다음 포지션을 닫지 않게
        self.assertEqual(fx.order_deletes(), [7], fx.calls)                                 # 배포 전 자동 물타기(dbad)도 · 사용자 주문은 남김
        self.assertEqual(st, {})


class RearmTest(unittest.TestCase):
    def _run(self, fn, armed) -> None:
        async def exercise() -> None:
            with mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.multiple(orders, clear_bracket=_boom, place_bracket=_boom, run_entry=_boom, run_exit=_boom):
                (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps({KEY: armed}), "utf-8")
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_rule_rearm_only_clears_sl_off(self) -> None:
        async def fn(c):
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 200, await r.text())
            j = await r.json()
            self.assertTrue(j["ok"] and j.get("backstop") is None, j)
            st = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[KEY]
            self.assertNotIn("sl_off", st)                                 # 5분봉 감시 다시 켜짐
            self.assertEqual(st["sl_level"], RULE_ARM["sl_level"])
        self._run(fn, {**RULE_ARM, "sl_off": {"at": 5.0, "cancelled": 1}})

    def test_rule_rearm_without_level_refuses(self) -> None:
        async def fn(c):
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 409)
            st = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))[KEY]
            self.assertIn("sl_off", st)                                    # 손절 없이 «켜짐»이 되지 않게
        self._run(fn, {**RULE_ARM, "sl_price": None, "sl_level": None, "backstop_price": None, "sl_off": {"at": 5.0}})


if __name__ == "__main__":
    unittest.main()
