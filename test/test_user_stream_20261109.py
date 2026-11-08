"""2026-11-09 체결 알림(바이낸스 사용자 데이터 스트림) -- 분류 · 깨우기 · 연결 한 바퀴(가짜 세션, 네트워크 없음)."""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from unittest import mock

from aiohttp import WSMsgType

from dashboard import orders
from live.execution import live_manual_peg_execute_20260912 as ex


class EventKindTest(unittest.TestCase):
    def test_kinds(self) -> None:
        k = orders.user_event_kind
        self.assertEqual(k({"e": "ORDER_TRADE_UPDATE", "o": {"x": "TRADE"}}), "fill")
        self.assertEqual(k({"e": "ORDER_TRADE_UPDATE", "o": {"x": "NEW"}}), "")        # 우리 리페그마다 계좌를 읽지 않는다
        self.assertEqual(k({"e": "ORDER_TRADE_UPDATE", "o": {"x": "CANCELED"}}), "order")
        self.assertEqual(k({"e": "ACCOUNT_UPDATE"}), "fill")
        self.assertEqual(k({"e": "listenKeyExpired"}), "expired")
        self.assertEqual(k({"e": "MARGIN_CALL"}), "")
        self.assertEqual(k([]), "")


class NapTest(unittest.TestCase):
    def test_wake_cuts_nap_short(self) -> None:
        async def go():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            task = asyncio.create_task(ex.nap(5.0))
            await asyncio.sleep(0.05)
            ex.wake_fills()
            await task
            return loop.time() - t0
        self.assertLess(asyncio.run(go()), 1.0)

    def test_nap_times_out_without_wake(self) -> None:
        async def go():
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await ex.nap(0.1)
            return loop.time() - t0, len(ex._wakers)
        dt, left = asyncio.run(go())
        self.assertGreaterEqual(dt, 0.09)
        self.assertEqual(left, 0, "깨우기 목록에 찌꺼기가 남지 않는다")


class NapCancelTest(unittest.TestCase):
    def test_cancel_not_swallowed_when_woken_same_tick(self) -> None:
        async def go():
            async def loop_():
                while True:
                    await ex.nap(5.0)
            t = asyncio.create_task(loop_())
            await asyncio.sleep(0.01)
            ex.wake_fills()
            t.cancel()
            try:
                await asyncio.wait_for(t, 1.0)
            except asyncio.CancelledError:
                return "cancelled"
            except asyncio.TimeoutError:
                return "hung"
        self.assertEqual(asyncio.run(go()), "cancelled", "종료 때 감시 루프가 멈춘다(3.11 wait_for 취소 삼킴 회피)")


class _Msg:
    def __init__(self, typ, data=""):
        self.type, self.data = typ, data


class _WS:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    async def receive(self, timeout=None):
        if self.msgs:
            return self.msgs.pop(0)
        await asyncio.sleep(3600)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Resp:
    status = 200
    headers: dict = {}

    async def json(self, content_type=None):
        return {"listenKey": "LK"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, msgs):
        self.msgs, self.urls = msgs, []

    def request(self, method, url, headers=None):
        self.urls.append((method, url))
        return _Resp()

    def ws_connect(self, url, heartbeat=None):
        self.urls.append(("WS", url))
        return _WS(self.msgs)

    async def close(self):
        pass


class LoopTest(unittest.TestCase):
    def test_connect_refreshes_account_and_trade_wakes(self) -> None:
        trade = json.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"x": "TRADE"}})
        sess = _Session([_Msg(WSMsgType.TEXT, json.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"x": "NEW"}})), _Msg(WSMsgType.TEXT, trade)])
        produced = []

        async def produce(_name, _fn):
            produced.append(1)
            return {"ok": True}

        async def go():
            tasks: dict = {}
            with mock.patch.object(orders, "exec_enabled", lambda: True), mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k"}), \
                 mock.patch.object(orders, "ClientSession", lambda **kw: sess), mock.patch.object(orders.ban_guard, "ban_remaining", lambda *a: 0.0):
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks=tasks, swr_locks={}, binance_session=lambda: sess,
                                      fetch_binance_json=None, swr_cached=None, _timed_produce=produce,
                                      load_market_history=None, situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60)
                waiter = asyncio.create_task(ex.nap(30.0))
                await api.start_user_stream(None)
                await asyncio.sleep(0.6)
                woke = waiter.done()
                seq = api.acct_seq()
                for t in tasks.values():
                    t.cancel()
                await asyncio.gather(*tasks.values(), return_exceptions=True)
                waiter.cancel()
                return woke, seq
        woke, seq = asyncio.run(go())
        self.assertIn(("POST", "https://" + "fapi.binance.com/fapi/v1/listenKey"), sess.urls)
        self.assertIn(("WS", orders.USER_STREAM_WS + "LK"), sess.urls)
        self.assertTrue(orders.USER_STREAM_WS.endswith("/private/ws/"), "공식 문서: /private 없으면 계좌 알림이 안 온다")
        self.assertTrue(woke, "체결 알림이 기다리던 루프를 깨운다")
        self.assertEqual(len(produced), 1, "붙자마자 + 체결 알림이 0.3초 안에 겹쳐 계좌 조회는 한 번")
        self.assertEqual(seq, 1)

    def test_gate_off_starts_nothing(self) -> None:
        async def go():
            tasks: dict = {}
            with mock.patch.object(orders, "exec_enabled", lambda: False):
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks=tasks, swr_locks={}, binance_session=lambda: None,
                                      fetch_binance_json=None, swr_cached=None, _timed_produce=None,
                                      load_market_history=None, situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60)
                await api.start_user_stream(None)
            return tasks
        self.assertEqual(asyncio.run(go()), {})


if __name__ == "__main__":
    unittest.main()
