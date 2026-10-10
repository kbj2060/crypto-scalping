"""스위칭(전량 청산 → 반대 진입)을 서버가 한 작업으로 한다 (2026-10-10 실사고: 브라우저가 이어 주던 반대 진입이 화면이 꺼져 안 나감).

(a) 전량 청산 → 반대 진입 1회  (b) 부분 청산 → 반대 진입 0  (c) 반대 진입 계획 막힘 → 상태에 사유·주문 0
(d) 클라이언트 연결이 끊겨도 서버만으로 완결. 음성 대조: 수정 전 server.py 에서는 (a)·(d) 가 실패한다(switch_to 를 무시).
🔴실주문 0 -- run_exit·run_entry·place_bracket 을 가짜로 갈아끼우고 바이낸스 GET 은 _isolated_dirs 의 가짜 응답.
실행: python -m pytest -q test/test_manual_switch_server_20261010.py
"""
from __future__ import annotations

import asyncio
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
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402
from test_rec_rule_entry_20261009 import account, long_pos  # noqa: E402

SHORT = {**long_pos(2.563, 2500.0), "side": "SHORT"}
SUBMIT = "/api/manual-exit/submit?side=SHORT&pct=100&confirm=1&switch_to=LONG&sw_pct=10&sw_lev=20&sw_sltp=0"


class SwitchServerTest(unittest.TestCase):
    def _run(self, fn, exit_phase: str = "filled_maker", exit_frac: float = 1.0, mtype: str = "cross", fill_entry: bool = False) -> dict:
        acct = account([dict(SHORT)])
        rec: dict = {"exits": [], "entries": [], "placed": []}

        async def fake_account(*_a, **_k):
            return acct

        async def fake_exit(_s, plan, state):
            rec["exits"].append(plan)
            await rec["release"].wait()                       # 청산이 도는 동안 클라이언트가 떠날 수 있다
            q = float(plan["quantity"])
            state.update(phase=exit_phase, kind="exit", quantity=q, filled=round(q * exit_frac, 8))
            if exit_frac >= 1.0:
                acct["positions"] = []                        # 거래소에서 포지션이 사라졌다
            return state

        async def fake_entry(_s, plan, state):
            rec["entries"].append(plan)
            state.update(phase="filled_maker", quantity=plan["quantity"], filled=plan["quantity"] if fill_entry else 0.0)
            return state

        async def fake_place(_s, bracket, symbol, pside):             # 가짜 -- 거래소로 안 간다
            rec["placed"].append((pside, bracket.get("sl_price"), bracket.get("backstop_price")))
            return {"placed": True, "backstop": {"placed": True, "price": bracket.get("backstop_price")}}

        async def exercise() -> None:
            rec["release"] = asyncio.Event()
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "run_exit", fake_exit), \
                 mock.patch.object(server, "run_entry", fake_entry), \
                 mock.patch.object(server, "place_bracket", fake_place if fill_entry else mock.AsyncMock(side_effect=AssertionError("주문 금지"))), \
                 mock.patch.object(server, "margin_type", mock.AsyncMock(return_value=mtype)):
                app = offline_app(server)
                app["situation_eth"]["vol_mult"] = {"sigma_bp": 100.0, "mult": 1.0, "bar": time.time()}
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client, rec)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())
        return rec

    @staticmethod
    async def _settle(rec, client=None) -> dict | None:
        rec["release"].set()
        for _ in range(100):
            await asyncio.sleep(0.02)
            if client is None:
                continue
            st = (await (await client.get("/api/manual-entry/status")).json())["state"]
            if (st.get("switch") or {}).get("phase") in ("done", "aborted"):
                return st
        return None

    def test_a_full_exit_then_one_opposite_entry(self) -> None:
        async def fn(c, rec):
            body = await (await c.post(SUBMIT)).json()
            self.assertTrue(body.get("ok"), body)
            self.assertEqual(body["state"]["switch"]["phase"], "exit")
            st = await self._settle(rec, c)
            self.assertEqual(len(rec["entries"]), 1, "반대 진입이 정확히 한 번")
            p = rec["entries"][0]
            self.assertEqual((p["positionSide"], p["side"]), ("LONG", "BUY"))
            self.assertAlmostEqual(p["notional_usdt"], 1000 * 0.10 * 20, delta=3.0)    # 증거금 10% × 20배
            self.assertFalse(p["bracket"].get("disabled"), p["bracket"])                 # sw_sltp=0 이어도 손절(10-10 사용자 결정)
            self.assertEqual(st["switch"]["phase"], "done")
            self.assertEqual(st["switch"]["from"], "SHORT")
            self.assertEqual(st["switch"]["exit"]["phase"], "filled_maker")              # 진입으로 넘어가도 «스위칭에서 왔다»가 남는다
            self.assertEqual(st["side"], "LONG")
        self._run(fn)

    def test_b_partial_exit_no_entry(self) -> None:
        async def fn(c, rec):
            await c.post(SUBMIT)
            st = await self._settle(rec, c)
            self.assertEqual(rec["entries"], [])
            self.assertEqual(st["switch"]["phase"], "aborted")
            self.assertIn("다 닫히지 않아", st["switch"]["error"])
        self._run(fn, exit_phase="rejected", exit_frac=0.4)

    def test_c_blocked_entry_records_reason_no_order(self) -> None:
        async def fn(c, rec):
            r = await c.post(SUBMIT + "&sw_rule=c")
            self.assertEqual(r.status, 200, await r.text())
            st = await self._settle(rec, c)
            self.assertEqual(rec["entries"], [])
            self.assertEqual(st["phase"], "error")
            self.assertEqual(st["switch"]["phase"], "aborted")
            self.assertIn("격리 마진", st["switch"]["error"])                           # rule_not_cross 의 사유 그대로(10-11 σ 검사는 없어졌다)
            self.assertIn("격리 마진", st["error"])
        self._run(fn, mtype="isolated")

    def test_d_client_gone_server_completes(self) -> None:
        async def fn(c, rec):
            self.assertTrue((await (await c.post(SUBMIT)).json())["ok"])
            await c.session.close()                           # 화면 꺼짐·앱 전환 -- 더는 아무 요청도 안 온다
            await self._settle(rec)
            self.assertEqual(len(rec["exits"]), 1)
            self.assertEqual(len(rec["entries"]), 1)
        self._run(fn)

    def test_direct_mode_switch_entry_gets_stop_even_with_sltp_off(self) -> None:
        """2026-10-10 사용자 결정: 직접 모드 스위칭 반대 진입엔 SL/TP 체크(sw_sltp=0)와 무관하게 손절을 건다."""
        async def fn(c, rec):
            self.assertTrue((await (await c.post(SUBMIT)).json())["ok"])          # SUBMIT 에 sw_sltp=0 이 들어 있다
            st = await self._settle(rec, c)
            self.assertEqual(len(rec["entries"]), 1)
            self.assertEqual(len(rec["placed"]), 1, "반대 진입 체결 뒤 손절·비상 스탑이 걸려야 한다")
            pside, sl, backstop = rec["placed"][0]
            self.assertEqual(pside, "LONG")
            self.assertTrue(sl and backstop, rec["placed"])
            self.assertTrue(st["bracket"]["placed"], st["bracket"])
        self._run(fn, fill_entry=True)

    def test_bad_switch_rejected(self) -> None:
        async def fn(c, rec):
            for q in ("&switch_to=SHORT", "&switch_to=LONG&pct=50", "&switch_to=LONG&sw_pct=0", "&switch_to=LONG&sw_rule=x"):
                r = await c.post("/api/manual-exit/submit?side=SHORT&confirm=1" + q)
                self.assertEqual(r.status, 400, q)
            self.assertEqual(rec["exits"], [])
        self._run(fn)


if __name__ == "__main__":
    unittest.main()
