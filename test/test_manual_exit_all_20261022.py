"""«전체 청산»(2026-10-22 사용자 «코인 여러 개를 한꺼번에 청산 -- 지금은 다른 청산을 기다려야 한다»).

(a) ETH 숏 + XRP 롱 + 대시보드 밖 BTC → 한 번 접수에 ETH·XRP 청산이 **동시에** 돈다(ETH 가 끝나기 전에 XRP 도 나감), BTC 는 unknown_symbol
(b) 돌고 있는 동안 다시 누르면 둘 다 already_working(두 번 나가지 않는다)
(c) confirm 없음 400 · GET 405 · 게이트 꺼짐 403 · 포지션 없음 400 · exec=market 이면 계획이 MARKET
🔴실주문 0 -- run_exit 를 가짜로, run_entry·place_bracket 은 불리면 실패. 바이낸스 GET 은 _isolated_dirs 의 가짜 응답.
실행: python -m pytest -q test/test_manual_exit_all_20261022.py
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import server  # noqa: E402
from dashboard import orders  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402
from test_rec_rule_entry_20261009 import SYM, account, long_pos  # noqa: E402   SYM = 주문 ETH 심볼(.env 따라 USDC/USDT)

ETH_SHORT = {**long_pos(0.943, 2447.0), "side": "SHORT"}
XRP_LONG = {**long_pos(1904.8, 1.29), "symbol": "XRPUSDC", "mark_price": 1.30, "notional": 2476.0}
BTC_LONG = {**long_pos(0.01, 60000.0), "symbol": "BTCUSDT"}


class ExitAllTest(unittest.TestCase):
    def _run(self, positions, fn, enabled: bool = True) -> dict:
        acct = account([dict(p) for p in positions])
        rec: dict = {"exits": []}

        async def fake_account(*_a, **_k):
            return acct

        async def fake_exit(_s, plan, state):
            rec["exits"].append(plan)
            state.update(phase="working", kind="exit")
            await rec["release"].wait()                    # 앞 청산이 «지정가로 도는 중»
            state.update(phase="filled_maker", filled=plan["quantity"])
            return state

        async def boom(*_a, **_k):
            raise AssertionError("주문 경로가 불렸다 -- 시험에서 실주문 금지")

        async def exercise() -> None:
            rec["release"] = asyncio.Event()
            with mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "produce_account", fake_account, create=True), \
                 mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(orders, "exec_enabled", lambda: enabled), \
                 mock.patch.object(orders, "run_exit", fake_exit), \
                 mock.patch.object(orders, "run_entry", boom), \
                 mock.patch.object(orders, "place_bracket", boom):
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    await fn(client, rec)
                finally:
                    rec["release"].set()
                    await asyncio.sleep(0.05)
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())
        return rec

    def test_all_coins_at_once_and_no_double_submit(self) -> None:
        async def fn(client, rec):
            r = await client.post("/api/manual-exit/submit-all?confirm=1")
            body = await r.json()
            self.assertEqual(r.status, 200, body)
            by = {x["symbol"]: x for x in body["results"]}
            self.assertTrue(by[SYM]["ok"] and by["XRPUSDC"]["ok"], body)
            self.assertEqual(by["BTCUSDT"]["error"], "unknown_symbol")
            await asyncio.sleep(0.05)
            self.assertEqual(sorted(p["symbol"] for p in rec["exits"]), sorted([SYM, "XRPUSDC"]))   # ETH 가 안 끝났는데 XRP 도 나갔다
            self.assertEqual({p["positionSide"] for p in rec["exits"]}, {"SHORT", "LONG"})
            st = await (await client.get("/api/manual-entry/status")).json()
            self.assertEqual(st["state"]["phase"], "working")
            self.assertEqual(st["side_exits"]["xrp"]["phase"], "working")
            again = await client.post("/api/manual-exit/submit-all?confirm=1")
            b2 = await again.json()
            self.assertEqual(again.status, 409, b2)
            self.assertEqual({x["error"] for x in b2["results"] if x["symbol"] != "BTCUSDT"}, {"already_working"})
            self.assertEqual(len(rec["exits"]), 2)                                        # 두 번 나가지 않았다
        self._run([ETH_SHORT, XRP_LONG, BTC_LONG], fn)

    def test_guards(self) -> None:
        async def fn(client, rec):
            self.assertEqual((await client.post("/api/manual-exit/submit-all")).status, 400)
            self.assertEqual((await client.get("/api/manual-exit/submit-all?confirm=1")).status, 405)
            self.assertEqual((await client.post("/api/manual-exit/submit-all?confirm=1&exec=x")).status, 400)
            self.assertEqual(rec["exits"], [])
        self._run([ETH_SHORT, XRP_LONG], fn)

        async def off(client, rec):
            self.assertEqual((await client.post("/api/manual-exit/submit-all?confirm=1")).status, 403)
            self.assertEqual(rec["exits"], [])
        self._run([ETH_SHORT], off, enabled=False)

        async def none(client, rec):
            self.assertEqual((await client.post("/api/manual-exit/submit-all?confirm=1")).status, 400)
        self._run([], none)

    def test_market_exec(self) -> None:
        async def fn(client, rec):
            r = await client.post("/api/manual-exit/submit-all?confirm=1&exec=market")
            self.assertEqual(r.status, 200, await r.text())
            await asyncio.sleep(0.05)
            self.assertEqual({p["type"] for p in rec["exits"]}, {"MARKET"})
        self._run([ETH_SHORT, XRP_LONG], fn)


if __name__ == "__main__":
    unittest.main()
