"""«손절 해제» 버튼(2026-10-15 사용자 «2번으로 진행» -- 10-14 사고: 바이낸스에서 스탑만 지웠는데 서버 5분봉 종가 감시가 청산).
🔴실주문 0: 거래소 취소(clear_bracket)·주문 함수는 전부 가짜. ① 방어(confirm·게이트·측면·무장 없음) ② 해제 = 우리 손절 algo 만 취소(only_sl) +
sl_off 남김(물타기 지정가 기록은 그대로) · 두 번째 요청은 거래소를 안 부름 ③ 취소 실패면 아무것도 안 바꿈 ④ 계좌 API 가 무장 손절을 싣는다
⑤ cancel_bracket(only_sl) 은 일반 주문 목록을 안 건드린다 ⑥ «다시 켜기»(/rearm) = place_stop(스탑만) 성공 때만 sl_off 지움.
python -m pytest -q test/test_bracket_disarm_20261015.py"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from scripts import live_manual_peg_execute_20260912 as ex  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = server.MANUAL_EXEC_SYMBOLS["eth"]
KEY = f"{SYM}:LONG"
ARMED = {KEY: {"symbol": SYM, "side": "LONG", "sl_price": 2306.98, "sl_level": 2309.25, "backstop_price": 2306.98, "rule": True,
               "sl_name": "첫 진입가 −5%", "armed_at": 1000.0, "placed": {"placed": True},
               "add_order": {"placed": True, "price": 2410.18, "qty": 0.875, "order_id": 1}}}


async def _boom(*_a, **_k):
    raise AssertionError("실주문 경로가 불렸다 -- 시험에서 주문 금지")


class DisarmTest(unittest.TestCase):
    def _run(self, fn, armed=ARMED, clear=None, enabled=True, stop=None) -> None:
        async def exercise() -> None:
            with mock.patch.object(server, "exec_enabled", lambda: enabled), \
                 mock.patch.object(server, "clear_bracket", clear or _boom), \
                 mock.patch.object(server, "place_stop", stop or _boom), \
                 mock.patch.object(server, "place_bracket", _boom), \
                 mock.patch.object(server, "place_add_order", _boom), \
                 mock.patch.object(server, "run_entry", _boom), \
                 mock.patch.object(server, "run_exit", _boom):
                if armed is not None:
                    (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(armed), "utf-8")
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def state(self) -> dict:
        return json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))

    def test_guards(self) -> None:
        async def fn(c):
            self.assertEqual((await c.post(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}")).status, 400)            # confirm 없음
            self.assertEqual((await c.post(f"/api/manual-bracket/disarm?side=UP&symbol={SYM}&confirm=1")).status, 400)
            self.assertEqual((await c.post("/api/manual-bracket/disarm?side=LONG&symbol=BTCUSDT&confirm=1")).status, 400)
            self.assertEqual((await c.post(f"/api/manual-bracket/disarm?side=SHORT&symbol={SYM}&confirm=1")).status, 409)  # 무장 없음
            self.assertEqual((await c.get(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}&confirm=1")).status, 405)    # GET 금지
        self._run(fn)

        async def off(c):
            self.assertEqual((await c.post(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}&confirm=1")).status, 403)
        self._run(off, enabled=False)

    def test_disarm_cancels_only_stop_and_marks_state(self) -> None:
        calls = []

        async def fake_clear(_s, symbol, pside, *, only_sl=False):
            calls.append((symbol, pside, only_sl))
            return {"cancelled": 1, "errors": []}

        async def fn(c):
            r = await c.post(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 200, await r.text())
            self.assertEqual(calls, [(SYM, "LONG", True)])                       # 손절 algo 만(물타기·TP 지정가는 남김)
            st = self.state()[KEY]
            self.assertEqual(st["sl_off"]["cancelled"], 1)
            self.assertEqual(st["add_order"]["order_id"], 1)                    # 물타기 기록 유지 → 닫히면 감시가 정리
            j = await (await c.post(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}&confirm=1")).json()
            self.assertTrue(j["ok"] and j["already"])
            self.assertEqual(len(calls), 1)                                     # 두 번째는 거래소를 안 부른다
        self._run(fn, clear=fake_clear)

    def test_cancel_failure_changes_nothing(self) -> None:
        async def failing(_s, symbol, pside, *, only_sl=False):
            return {"cancelled": 0, "errors": ["openAlgoOrders: timeout"]}

        async def fn(c):
            r = await c.post(f"/api/manual-bracket/disarm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 502)
            self.assertIn("손절은 그대로", (await r.json())["detail"])
            self.assertNotIn("sl_off", self.state()[KEY])                        # 한 겹만 풀린 상태를 안 만든다
        self._run(fn, clear=failing)

    def test_rearm(self) -> None:
        off = {KEY: {**ARMED[KEY], "sl_off": {"at": 5.0, "cancelled": 1}}}
        calls = []

        async def fake_stop(_s, symbol, pside, price):
            calls.append((symbol, pside, price))
            return {"placed": True, "price": price, "algo_id": 77, "error": None}

        async def fn(c):
            self.assertEqual((await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}")).status, 400)          # confirm 없음
            self.assertEqual((await c.post(f"/api/manual-bracket/rearm?side=SHORT&symbol={SYM}&confirm=1")).status, 409)  # 무장 없음
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 200, await r.text())
            self.assertEqual(calls, [(SYM, "LONG", 2306.98)])                    # 거래소 스탑 = backstop_price
            st = self.state()[KEY]
            self.assertNotIn("sl_off", st)                                       # 5분봉 감시 다시 켜짐
            self.assertEqual(st["placed"]["backstop"]["algo_id"], 77)
            self.assertEqual(st["add_order"]["order_id"], 1)                     # 물타기 기록 그대로
            j = await (await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")).json()
            self.assertTrue(j["ok"] and j["already"]) ; self.assertEqual(len(calls), 1)   # 켜진 상태면 거래소를 안 부른다
        self._run(fn, armed=off, stop=fake_stop)

    def test_rearm_failure_changes_nothing(self) -> None:
        off = {KEY: {**ARMED[KEY], "sl_off": {"at": 5.0, "cancelled": 1}}}

        async def failing(_s, symbol, pside, price):
            return {"placed": False, "error": "Order would immediately trigger."}

        async def fn(c):
            r = await c.post(f"/api/manual-bracket/rearm?side=LONG&symbol={SYM}&confirm=1")
            self.assertEqual(r.status, 502)
            self.assertIn("꺼진 그대로", (await r.json())["detail"])
            self.assertIn("sl_off", self.state()[KEY])                           # 감시만 켜진 반쪽 상태를 안 만든다
        self._run(fn, armed=off, stop=failing)

    def test_account_carries_armed_stop(self) -> None:
        async def fn(c):
            j = await (await c.get("/api/binance-account")).json()
            self.assertEqual(j["bracket_armed"][KEY]["sl_price"], 2306.98)
            self.assertIsNone(j["bracket_armed"][KEY]["sl_off"])
        self._run(fn)


class CancelOnlyStopTest(unittest.TestCase):
    def test_only_sl_skips_regular_orders(self) -> None:
        deleted, seen = [], []

        async def fake_signed(_s, method, path, params, *_a, **_k):
            seen.append((method, path))
            if method == "GET" and path == "/fapi/v1/openAlgoOrders":
                return [{"algoId": 9, "clientAlgoId": "dbslL9"}, {"algoId": 8, "clientAlgoId": "dbslS8"}]
            if method == "DELETE":
                deleted.append(params.get("algoId") or params.get("orderId"))
                return {}
            raise AssertionError(f"예상 밖 호출 {method} {path}")             # openOrders(물타기·TP)는 부르지도 않는다

        with mock.patch.object(ex, "signed", fake_signed):
            out = asyncio.run(ex.cancel_bracket(None, SYM, "LONG", "k", "s", 0, only_sl=True))
        self.assertEqual(deleted, [9])                                          # 롱 손절만 -- 숏 손절(8)은 남김
        self.assertEqual(out, {"cancelled": 1, "errors": []})
        self.assertNotIn(("GET", "/fapi/v1/openOrders"), seen)


class PlaceStopTest(unittest.TestCase):
    def test_places_only_stop_after_clearing_old(self) -> None:
        seen, posted = [], []

        async def fake_signed(_s, method, path, params, *_a, **_k):
            seen.append((method, path))
            if path == "/fapi/v2/positionRisk":
                return [{"positionSide": "LONG", "positionAmt": "0.433"}]
            if method == "GET" and path == "/fapi/v1/openAlgoOrders":
                return [{"algoId": 9, "clientAlgoId": "dbslL1"}]
            if method == "DELETE":
                return {}
            if method == "POST" and path == "/fapi/v1/algoOrder":
                posted.append(params); return {"algoId": 10}
            raise AssertionError(f"예상 밖 호출 {method} {path}")             # TP·물타기 주문(/fapi/v1/order)은 안 건드린다

        async def zero(_s):
            return 0

        with mock.patch.object(ex, "signed", fake_signed), mock.patch.object(ex, "_clock_offset", zero), \
             mock.patch.dict("os.environ", {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            out = asyncio.run(ex.place_stop(None, SYM, "LONG", 2306.98))
            none = (seen.clear(), asyncio.run(ex.place_stop(None, SYM, "SHORT", 2500.0)))[1]
        self.assertTrue(out["placed"]) ; self.assertEqual(out["algo_id"], 10)
        self.assertEqual(len(posted), 1)
        self.assertEqual((posted[0]["side"], posted[0]["type"], posted[0]["triggerPrice"], posted[0]["closePosition"]), ("SELL", "STOP_MARKET", 2306.98, "true"))
        self.assertFalse(none["placed"]) ; self.assertEqual(none["error"], "포지션 없음")   # 숏 포지션 없음 → 안 건다
        self.assertEqual(seen, [("GET", "/fapi/v2/positionRisk")])


if __name__ == "__main__":
    unittest.main()
