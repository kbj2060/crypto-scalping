"""−5% 거래소 스탑·자동 물타기 빼기(2026-11-21 사용자 질문지 «전부 뺀다 · 파이썬에서도 지금» · «자동 물타기 안 건다»).
🔴실주문 0 -- signed·시계·청산은 전부 가짜. ① 어떤 경로도 STOP_MARKET(algoOrder POST)을 안 건다 ② 자동 규칙 진입이 물타기 지정가를 안 건다
③ 규칙 무장의 손절 = 서버 5분봉 종가 감시가 실제로 청산을 쏜다 ④ 배포 전에 걸린 우리 스탑(dbsl)은 포지션이 닫히면 지운다(Rust 와 다른 점)
⑤ 규칙 무장 «손절 다시 켜기» = 거래소 요청 없이 sl_off 만 지운다.
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
from test_rule_auto_add_20261014 import ACCT, VM  # noqa: E402

SYM = orders.MANUAL_EXEC_SYMBOLS["eth"]
KEY = f"{SYM}:LONG"
FAKE_ENV = {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}
RULE_ARM = {"symbol": SYM, "side": "LONG", "sl_price": 2306.98, "sl_level": 2309.25, "backstop_price": 2306.98, "rule": True,
            "sl_name": "첫 진입가 −5%", "armed_at": 1000.0, "placed": {"placed": True}}


async def _boom(*_a, **_k):
    raise AssertionError("실주문 경로가 불렸다 -- 시험에서 주문 금지")


async def _zero(*_a, **_k):
    return 0


class FakeEx:
    """거래소 가짜: 포지션 하나(held) · 배포 전에 걸린 우리 스탑 dbslL1 이 남아 있다."""

    def __init__(self, held=True):
        self.held, self.calls = held, []

    async def signed(self, _s, method, path, params, *_a, **_k):
        self.calls.append((method, path, dict(params)))
        if path == "/fapi/v2/positionRisk":
            return [{"positionSide": "LONG", "positionAmt": "1.0" if self.held else "0"}]
        if path == "/fapi/v1/openOrders":
            return []
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


class RuleEntryTest(unittest.TestCase):
    """자동 규칙(rule=c) 첫 진입 -- 진짜 place_bracket·arm_rule_add 가 돌고 거래소만 가짜."""

    def _run(self, fn) -> FakeEx:
        fx = FakeEx()

        async def fake_account(*_a, **_k):
            return ACCT

        async def fake_entry(_s, plan, state):
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def exercise() -> None:
            p1, p2 = fake_exchange(fx)
            with p1, p2, mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "produce_account", fake_account, create=True), \
                 mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(orders, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.object(orders, "run_entry", fake_entry), \
                 mock.patch.object(orders, "margin_type", mock.AsyncMock(return_value="cross")):
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
        return fx

    def test_preview_has_no_add(self) -> None:
        async def fn(c):
            r = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]["rule"]
            self.assertTrue(r["first"] and r["sl_price"], r)
            self.assertNotIn("add", r)                                   # 화면이 «물타기 지정가 자동»을 말하지 않게
        self._run(fn)

    def test_fill_arms_watch_without_stop_or_add(self) -> None:
        box = {}

        async def fn(c):
            self.assertTrue((await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=c")).json()).get("ok"))
            for _ in range(100):
                await asyncio.sleep(0.02)
                if (server.LIVE_DIR / "manual_bracket_state.json").exists():
                    break
            await asyncio.sleep(0.1)
            box["st"] = (await (await c.get("/api/manual-entry/status")).json())["state"]
            box["armed"] = json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
        fx = self._run(fn)
        self.assertEqual(fx.stop_posts(), [], fx.calls)                 # 스탑 POST 0건
        self.assertEqual(fx.order_posts(), [], fx.calls)                # 물타기 지정가 POST 0건(익절도 규칙엔 없다)
        st, a = box["st"], box["armed"][KEY]
        self.assertIsNone(st.get("add_order"), st)
        self.assertTrue(st["bracket"]["placed"] and st["bracket"]["backstop"]["skipped"], st["bracket"])
        self.assertTrue(a["rule"] and a["sl_price"] and a["sl_level"] and a["backstop_price"], a)   # 5분봉 감시·«물타기해도 손절 유지»가 보는 값은 남는다
        self.assertIsNone(a["add_order"], a)


def watch(armed: dict, *, held=True, bars=None):
    """감시 틱 한 번 -- clear_bracket 은 진짜(거래소만 가짜) · 청산 run_exit 은 가짜. 돌려주기 = (청산 호출, 거래소 가짜, 저장본)."""
    exits, fx = [], FakeEx(held=held)
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
             mock.patch.multiple(orders, run_exit=exit_, run_entry=_boom, place_bracket=_boom, place_add_order=_boom):
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
        exits, fx, st = watch({**RULE_ARM, "armed_at": 1000.0}, held=False)
        self.assertEqual(exits, [])
        self.assertEqual([c[2].get("algoId") for c in fx.stop_deletes()], [9], fx.calls)   # 배포 전 스탑이 다음 포지션을 닫지 않게
        self.assertEqual(st, {})


class RearmTest(unittest.TestCase):
    def _run(self, fn, armed) -> None:
        async def exercise() -> None:
            with mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.multiple(orders, clear_bracket=_boom, place_bracket=_boom, place_add_order=_boom, run_entry=_boom, run_exit=_boom):
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
