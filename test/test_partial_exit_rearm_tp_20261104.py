"""부분 청산 뒤 익절 지정가 복구(2026-11-04 사용자 «SLTP 주문 걸린 채 중간 청산 50% 하면 익절 지정가가 사라진다»).

익절 지정가 수량 = 걸 때의 포지션 전량 -> 일부를 닫으면 거래소가 닫는 쪽 주문을 줄이거나 지운다.
(a) 50% 청산이 끝나면 place_bracket 이 **저장된 익절가**로 한 번 불리고 tp_resting 이 갱신된다
(b) 100% 청산 · 익절 지정가가 원래 없던(tp_resting False) 무장은 안 부른다
🔴실주문 0 -- run_exit·place_bracket 은 가짜.
실행: python -m pytest -q test/test_partial_exit_rearm_tp_20261104.py
"""
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
from dashboard import orders  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402
from test_rec_rule_entry_20261009 import SYM, account, long_pos  # noqa: E402

KEY = f"{SYM}:LONG"


def armed(tp_resting: bool) -> dict:
    return {KEY: {"symbol": SYM, "side": "LONG", "lines": True, "tp_price": 2500.5, "tp_level": 2500.0, "sl_level": 2300.0,
                  "tp_resting": tp_resting, "armed_at": 1000.0, "placed": {"tp": {"placed": tp_resting}}}}


class PartialExitRearmTest(unittest.TestCase):
    def _run(self, pct: int, tp_resting: bool) -> tuple[list, dict]:
        acct, calls = account([dict(long_pos(1.0, 2400.0))]), []

        async def fake_account(*_a, **_k):
            return acct

        async def fake_exit(_s, plan, state):
            state.update(phase="filled_maker", filled=plan["quantity"])
            return state

        async def fake_bracket(_s, spec, symbol, pside, **_k):
            calls.append((spec, symbol, pside))
            return {"qty": 0.5, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": True, "price": spec["tp_price"]}, "placed": True}

        async def exercise() -> dict:
            with mock.patch.object(orders, "fetch_account", fake_account), \
                 mock.patch.object(orders, "produce_account", fake_account, create=True), \
                 mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.object(orders, "run_exit", fake_exit), \
                 mock.patch.object(orders, "place_bracket", fake_bracket):
                (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(armed(tp_resting)), "utf-8")
                client = TestClient(TestServer(offline_app(server)))
                await client.start_server()
                try:
                    r = await client.post(f"/api/manual-exit/submit?side=LONG&asset=eth&confirm=1&pct={pct}")
                    self.assertEqual(r.status, 200, await r.text())
                    await asyncio.sleep(0.3)   # 백그라운드 청산 태스크
                    return json.loads((server.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))
                finally:
                    await client.close()

        with _isolated_dirs():
            st = asyncio.run(exercise())
        return calls, st

    def test_partial_exit_replaces_tp_for_remaining_qty(self) -> None:
        calls, st = self._run(50, True)
        self.assertEqual(calls, [({"tp_price": 2500.5}, SYM, "LONG")])
        self.assertTrue(st[KEY]["tp_resting"])

    def test_full_exit_or_no_resting_tp_leaves_orders_alone(self) -> None:
        self.assertEqual(self._run(100, True)[0], [])
        self.assertEqual(self._run(50, False)[0], [])


if __name__ == "__main__":
    unittest.main()
