"""자동 규칙 진입(2026-10-09, 재생 ×15.2 크기식) -- 서버 경로를 끝까지 지나게 하고 숫자를 고정한다. 🔴실주문 0.

규칙: 왕복 최대 명목 = 순자산 × L, L = min(12.4 × 권장배수, 청산 안전, 10) · 첫 진입 = 그 75% · 물타기 = 남은 여유 ·
손절 = 첫 진입가 ∓3σ24 거래소 스탑(물타기해도 유지) · 익절 없음 · SL/TP 체크를 꺼도 건다 · rule=c 밖이면 400 · 변동성 없으면 503 · ETH 만.
심는 값: 권장배수 0.5 · σ(4h) 250/√6 → σ24 250bp → 손절 7.5% · L = 0.075 ÷ 7.5% = 1.0(2026-10-10 동일위험이 묶음).
네트워크·계좌는 test_manual_entry_preview_smoke 의 격리 틀(_isolated_dirs: 바이낸스 GET 가짜 응답)을 쓰고,
run_entry·place_bracket 은 갈아끼워 주문 경로가 절대 바깥으로 못 나가게 한다.
실행: python -m pytest -q test/test_rec_rule_entry_20261009.py
"""
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
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = server.MANUAL_EXEC_SYMBOLS["eth"]
L = 0.075 / 0.075                    # 2026-10-10 동일위험: 손절 시 순자산 7.5% ÷ 손절폭 7.5% = 1.0
VM = {"sigma_bp": 250.0 / math.sqrt(6), "mult": 0.5}


def account(positions: list[dict], equity: float = 1000.0) -> dict:
    return {"ok": True, "balance": {"margin": equity, "initial_margin": 0.0, "available": equity, "unrealized": 0.0},
            "positions": positions, "leverage_by_symbol": {SYM: 20.0}, "trips": []}


def long_pos(qty: float, entry: float) -> dict:
    return {"symbol": SYM, "side": "LONG", "qty": qty, "entry_price": entry, "mark_price": 2500.0,
            "liquidation_price": 1000.0, "leverage": 20.0, "notional": qty * 2500.0, "unrealized_pnl": 0.0, "updated_at": None}


class RecRuleEntryTest(unittest.TestCase):
    def _run(self, acct: dict, fn, armed: dict | None = None, vm: dict | None = VM, mtype: str | None = "cross") -> None:
        async def fake_account(*_a, **_k):
            return acct

        async def boom(*_a, **_k):
            raise AssertionError("주문 경로가 불렸다 -- 시험에서 실주문 금지")

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "SIZING_MARGIN_CAP_PCT", 50.0), \
                 mock.patch.object(server, "MANUAL_RULES_ENABLED", True), \
                 mock.patch.object(server, "place_bracket", boom), \
                 mock.patch.object(server, "margin_type", mock.AsyncMock(return_value=mtype)):
                if armed is not None:
                    (server.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(armed), "utf-8")
                app = offline_app(server)
                app["situation_eth"]["vol_mult"] = {**vm, "bar": time.time()} if vm else None
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_first_entry_is_75pct_of_cap_and_stop_3sigma(self) -> None:
        async def fn(c):
            j = await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json()
            self.assertTrue(j.get("ok"), j)
            p = j["plan"]
            self.assertTrue(p["rule"]["first"], p["rule"])
            self.assertEqual(p["rule"]["binding"], "동일위험")
            self.assertEqual(p["rule"]["loss_at_stop_pct"], 7.5)
            self.assertAlmostEqual(p["rule"]["cap_notional"], 1000 * L, delta=0.01)
            self.assertAlmostEqual(p["notional_usdt"], 0.75 * 1000 * L, delta=2.6)       # 순자산 1000 × L × 0.75
            self.assertEqual(p["bracket"]["sl_price"], 2312.5)                           # 2500.00 × (1 − 7.5%)
            self.assertEqual(p["bracket"]["backstop_price"], 2312.5)                     # 거래소 스탑 = 손절선(닿는 즉시)
            self.assertIsNone(p["bracket"]["tp_price"])                                  # 익절 없음
            self.assertIn("첫 진입가", p["rule"]["sl_name"])
            self.assertEqual(p["target_leverage"], 20)
            j2 = await (await c.get("/api/manual-entry/preview?side=SHORT&lev=20&rule=c")).json()
            self.assertAlmostEqual(j2["plan"]["bracket"]["sl_price"], 2687.52, delta=0.011)   # 매도호가 2500.01 × 1.075 올림
        self._run(account([]), fn)

    def test_bad_rule_is_rejected_not_ignored(self) -> None:
        async def fn(c):
            for bad in ("1.5", "2", "x", "C"):
                r = await c.get(f"/api/manual-entry/preview?side=LONG&lev=20&rule={bad}")
                self.assertEqual(r.status, 400, bad)
                self.assertEqual((await r.json())["error"], "bad_rule")
                with mock.patch.object(server, "exec_enabled", lambda: True), \
                     mock.patch.object(server, "run_entry", mock.AsyncMock(side_effect=AssertionError("실주문 금지"))):
                    r = await c.post(f"/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule={bad}")
                self.assertEqual(r.status, 400, bad)                                # 게이트가 열려 있어도 400(직접 모드로 안 떨어진다)
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c&asset=sol")
            self.assertEqual(r.status, 400)
        self._run(account([]), fn)

    def test_missing_vol_blocks(self) -> None:
        async def fn(c):
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
            self.assertEqual(r.status, 503)
            self.assertEqual((await r.json())["error"], "rule_vol_unavailable")
        self._run(account([]), fn, vm=None)

    def test_isolated_or_unknown_margin_blocks(self) -> None:
        for mt in ("isolated", None):
            async def fn(c, mt=mt):
                r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
                self.assertEqual(r.status, 409, mt)
                self.assertEqual((await r.json())["error"], "rule_not_cross")
                r0 = await c.get("/api/manual-entry/preview?side=LONG&lev=20")          # 직접 모드는 그대로
                self.assertEqual(r0.status, 200)
            self._run(account([]), fn, mtype=mt)

    def test_rule_stop_survives_sltp_off(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c&sltp=0")).json())["plan"]
            self.assertFalse(p["bracket"].get("disabled"), p["bracket"])
            self.assertEqual(p["bracket"]["sl_price"], 2312.5)
            p0 = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&sltp=0")).json())["plan"]
            self.assertTrue(p0["bracket"].get("disabled"), "직접 모드의 SL/TP 끔은 그대로여야 한다")
        self._run(account([]), fn)

    def test_add_uses_remaining_room_and_keeps_armed_stop(self) -> None:
        armed = {f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2312.5, "sl_level": 2312.5,
                                 "backstop_price": 2312.5, "rule": True, "sl_name": "첫 진입가 −7.5%(3σ)",
                                 "armed_at": 0, "placed": {}}}
        qty = round(0.75 * 1000 * L / 2500.0, 3)

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertFalse(p["rule"]["first"])
            self.assertAlmostEqual(p["notional_usdt"], 1000 * L - qty * 2500.0, delta=2.6)   # 남은 여유만
            self.assertEqual(p["bracket"]["sl_price"], 2312.5)                               # 손절선이 안 물러난다
            self.assertIsNone(p["bracket"]["tp_price"])
        self._run(account([long_pos(qty, 2500.0)]), fn, armed=armed)

    def test_add_without_rule_arming_anchors_on_average(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertEqual(p["bracket"]["sl_price"], 2405.0)                      # 평단 2600 × (1 − 7.5%)
            self.assertIn("평단", p["rule"]["sl_name"])
        self._run(account([long_pos(0.3, 2600.0)]), fn)

    def test_opposite_leg_does_not_turn_first_entry_into_add(self) -> None:
        short = {**long_pos(0.04, 2500.0), "side": "SHORT"}                         # 반대 다리 명목 100(상한 1000 의 여유 900 ≥ 첫 750)

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertTrue(p["rule"]["first"], p["rule"])
            self.assertAlmostEqual(p["notional_usdt"], 0.75 * 1000 * L, delta=2.6)       # 남은 여유 전부(900)가 아니라 75%
            self.assertIn("첫 진입가", p["rule"]["sl_name"])
        self._run(account([short]), fn)

    def test_add_after_full_then_drop_keeps_loss_at_stop(self) -> None:
        """다 찬 뒤(명목 1000 = 0.4 ETH @2500) 2400 으로 빠지면 명목 여유 40 이 다시 생기지만, 손절(2312.5)에서 이미 75 를 잃으니 0."""
        armed = {f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2312.5, "sl_level": 2312.5,
                                 "backstop_price": 2312.5, "rule": True, "sl_name": "첫 진입가 −7.5%(3σ)", "armed_at": 0, "placed": {}}}
        pos = {**long_pos(0.4, 2500.0), "mark_price": 2400.0, "notional": 960.0}

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertLess(p["notional_usdt"], 1.0, p["rule"])                                   # 위험 여유 0 → 사실상 0(막힘)
        self._run(account([pos]), fn, armed=armed)

    def test_full_cap_blocks_add(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertTrue(p["blocked"], p)
        self._run(account([long_pos(0.42, 2500.0)]), fn)                          # 명목 1050 > 상한 1000

    def test_submit_carries_rule_without_sending_orders(self) -> None:
        fired = []

        async def never_run(_session, plan, state):
            fired.append(plan)
            state.update(phase="filled_maker")   # filled 0 -- 브래킷 단계로 안 간다
            return state

        async def fn(c):
            with mock.patch.object(server, "exec_enabled", lambda: True), mock.patch.object(server, "run_entry", never_run):
                body = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&rule=c")).json()
                self.assertTrue(body.get("ok"), body)
                await asyncio.sleep(0.05)
            self.assertEqual(len(fired), 1)
            self.assertAlmostEqual(fired[0]["rule"]["l"], round(L, 3), delta=1e-3)
            self.assertAlmostEqual(fired[0]["notional_usdt"], 0.75 * 1000 * L, delta=2.6)
            self.assertEqual(fired[0]["bracket"]["sl_price"], 2312.5)
        self._run(account([]), fn)


class ManualModeTest(unittest.TestCase):
    """2026-10-09 «규칙 모두 제거해 수동으로»: 커밋 기본값(MANUAL_RULES_ENABLED False · 증거금 상한 0) 그대로 --
    큰 기존 포지션이 있어도 상한으로 안 막히고, SL/TP 는 안 걸리고, rule=c 는 400."""

    def test_defaults_are_manual(self) -> None:
        self.assertEqual(server.SIZING_MARGIN_CAP_PCT, 0.0)   # 1번 배포(10-09)에도 증거금 상한은 꺼진 채

        async def fake_account(*_a, **_k):
            return account([long_pos(1.2, 2500.0)], equity=300.0)      # 명목 3,000 = 순자산 10배(옛 50% 상한이 꽉 찬 상태)

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "produce_account", fake_account, create=True), \
                 mock.patch.object(server, "SIZING_RISK_MODEL_ENABLED", False), \
                 mock.patch.object(server, "MANUAL_RULES_ENABLED", False), \
                 mock.patch.object(server, "place_bracket", mock.AsyncMock(side_effect=AssertionError("주문 금지"))):
                c = TestClient(TestServer(offline_app(server)))
                await c.start_server()
                try:
                    p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&pct=25")).json())["plan"]
                    self.assertIsNone(p["blocked"], p)
                    self.assertIsNone(p["cap_notional_usdt"])
                    self.assertAlmostEqual(p["notional_usdt"], 300 * 0.25 * 20, delta=3.0)   # 순자산 × 비율 × 레버리지
                    self.assertTrue(p["bracket"].get("disabled"), p["bracket"])            # sltp 를 안 보내도 끔
                    r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
                    self.assertEqual(r.status, 400)
                finally:
                    await c.close()

        with _isolated_dirs():
            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
