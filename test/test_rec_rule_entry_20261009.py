"""자동 규칙 진입(2026-10-09, 재생 ×15.2 크기식) -- 서버 경로를 끝까지 지나게 하고 숫자를 고정한다. 🔴실주문 0.

규칙(2026-10-14 «권장 B»): 왕복 최대 명목 = 순자산 × L, L = min(12.4 × 권장배수, 청산 안전, 3) · 첫 진입 = 그 1/3(10-14, 옛 75%) ·
물타기 = 남은 여유(손절 시 손실 L×5% 안) · 손절 = 첫 진입가 ∓5% 거래소 스탑(물타기해도 유지) · 익절 없음 · SL/TP 체크를 꺼도 건다 ·
rule=c 밖이면 400 · 변동성 없거나 15분 묵으면 503 · ETH 만. 심는 값: 권장배수 0.45 · σ24 250bp → 12.435 × 0.45 = 5.596 > 상한 3 → L = 3(10-14 권장 B · 상한이 묶는다).
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
from scripts.live_manual_peg_entry_20260912 import REC_FIRST_SHARE  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402

SYM = server.MANUAL_EXEC_SYMBOLS["eth"]
L = 3.0                               # 10-14 권장 B: 12.435 × 권장배수 0.45 = 5.596 > 상한 3 → 상한이 묶는다(청산 안전 10.6)
STOP = 0.05                           # 10-14 권장 B 손절 ∓5%
VM = {"sigma_bp": 250.0 / math.sqrt(6), "mult": 0.45}


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
                app["situation_eth"]["vol_mult"] = {"bar": time.time(), **vm} if vm else None   # vm 의 bar 가 있으면 그게 이긴다(묵은 값 시험)
                client = TestClient(TestServer(app))
                await client.start_server()
                try:
                    await fn(client)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_first_entry_is_third_of_cap_and_stop_5pct(self) -> None:
        async def fn(c):
            j = await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json()
            self.assertTrue(j.get("ok"), j)
            p = j["plan"]
            self.assertTrue(p["rule"]["first"], p["rule"])
            self.assertEqual(p["rule"]["binding"], "상한")
            self.assertAlmostEqual(p["rule"]["loss_at_stop_pct"], 100 * L * STOP, delta=0.01)   # 손절 시 손실 한도 = L × 5% = 15%
            self.assertEqual(p["rule"]["stop_pct"], 5.0)
            self.assertAlmostEqual(p["rule"]["l"], L, places=3)
            self.assertAlmostEqual(p["rule"]["cap_notional"], 1000 * L, delta=0.01)
            self.assertAlmostEqual(p["notional_usdt"], REC_FIRST_SHARE * 1000 * L, delta=2.6)       # 순자산 1000 × L × 첫 몫(10-14 부터 1/3)
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)                           # 2500.00 × (1 − 5%)
            self.assertEqual(p["bracket"]["backstop_price"], 2375.0)                     # 거래소 스탑 = 손절선(닿는 즉시)
            self.assertIsNone(p["bracket"]["tp_price"])                                  # 익절 없음
            self.assertEqual(p["rule"]["sl_name"], "첫 진입가 −5%")
            self.assertEqual(p["target_leverage"], 20)
            j2 = await (await c.get("/api/manual-entry/preview?side=SHORT&lev=20&rule=c")).json()
            self.assertAlmostEqual(j2["plan"]["bracket"]["sl_price"], 2625.02, delta=0.011)   # 매도호가 2500.01 × 1.05 올림
        self._run(account([]), fn)

    def test_hold_mode_caps_at_1_5(self) -> None:
        """2026-10-16 «보유 1.5×» = rule=h -- 같은 식에 상한만 1.5. 손절·첫 1/3 그대로 · 모르는 값은 400."""
        async def fn(c):
            j = await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=h")).json()
            self.assertTrue(j.get("ok"), j)
            p = j["plan"]
            self.assertAlmostEqual(p["rule"]["l"], 1.5, places=3)
            self.assertEqual(p["rule"]["binding"], "상한")
            self.assertAlmostEqual(p["rule"]["cap_notional"], 1000 * 1.5, delta=0.01)
            self.assertAlmostEqual(p["notional_usdt"], REC_FIRST_SHARE * 1000 * 1.5, delta=2.6)
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)
            self.assertEqual((await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=x")).status, 400)
        self._run(account([]), fn)

    def test_low_exchange_leverage_binds_rule_l(self) -> None:
        """2026-10-13 레버리지 20→10: 거래소 레버리지 X 면 실효 L ≤ 0.9X (증거금 90%). 10-14 상한 3 이면 3배에서 2.7 < 3 이라 묶인다."""
        async def fn(c):
            j = await (await c.get("/api/manual-entry/preview?side=LONG&lev=3&rule=c")).json()
            self.assertTrue(j.get("ok"), j)
            p = j["plan"]
            self.assertAlmostEqual(p["rule"]["l"], 2.7, places=6)
            self.assertEqual(p["rule"]["binding"], "레버리지 증거금")
            self.assertAlmostEqual(p["rule"]["loss_at_stop_pct"], 100 * 2.7 * STOP, delta=0.01)
            self.assertAlmostEqual(p["notional_usdt"], REC_FIRST_SHARE * 1000 * 2.7, delta=2.6)
            j10 = await (await c.get("/api/manual-entry/preview?side=LONG&lev=10&rule=c")).json()
            self.assertAlmostEqual(j10["plan"]["rule"]["l"], L, places=3)              # 상한 3 < 9 -- 10배에선 안 묶인다
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

    def test_missing_or_stale_vol_blocks(self) -> None:
        """크기가 권장 배수에 달렸다 -- 없거나 15분 넘게 묵으면 503(10-11 복원)."""
        async def fn(c):
            r = await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")
            self.assertEqual(r.status, 503)
            self.assertEqual((await r.json())["error"], "rule_vol_unavailable")
        self._run(account([]), fn, vm=None)
        self._run(account([]), fn, vm={**VM, "bar": 0})

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
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)
            p0 = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&sltp=0")).json())["plan"]
            self.assertTrue(p0["bracket"].get("disabled"), "직접 모드의 SL/TP 끔은 그대로여야 한다")
        self._run(account([]), fn)

    def test_add_uses_remaining_room_and_keeps_armed_stop(self) -> None:
        armed = {f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2375.0, "sl_level": 2375.0,
                                 "backstop_price": 2375.0, "rule": True, "sl_name": "첫 진입가 −5%",
                                 "armed_at": 0, "placed": {}}}
        qty = round(REC_FIRST_SHARE * 1000 * L / 2500.0, 3)

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertFalse(p["rule"]["first"])
            self.assertAlmostEqual(p["notional_usdt"], 1000 * L - qty * 2500.0, delta=2.6)   # 남은 여유만
            self.assertEqual(p["bracket"]["sl_price"], 2375.0)                               # 손절선이 안 물러난다
            self.assertIsNone(p["bracket"]["tp_price"])
        self._run(account([long_pos(qty, 2500.0)]), fn, armed=armed)

    def test_add_without_rule_arming_anchors_on_average(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertEqual(p["bracket"]["sl_price"], 2470.0)                      # 평단 2600 × (1 − 5%)
            self.assertIn("평단", p["rule"]["sl_name"])
        self._run(account([long_pos(0.3, 2600.0)]), fn)

    def test_opposite_leg_does_not_turn_first_entry_into_add(self) -> None:
        short = {**long_pos(0.04, 2500.0), "side": "SHORT"}                         # 반대 다리 명목 100(상한 3000 의 여유 2900 ≥ 첫 1000)

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertTrue(p["rule"]["first"], p["rule"])
            self.assertAlmostEqual(p["notional_usdt"], REC_FIRST_SHARE * 1000 * L, delta=2.6)       # 남은 여유 전부(2900)가 아니라 첫 몫(10-14 부터 1/3)
            self.assertIn("첫 진입가", p["rule"]["sl_name"])
        self._run(account([short]), fn)

    def test_add_after_full_then_drop_keeps_loss_at_stop(self) -> None:
        """다 찬 뒤(명목 3000 = 1.2 ETH @2500) 2450 으로 빠지면 명목 여유 ≈60 이 다시 생기지만, 손절(2375)에서 이미 L×5%(150)를 잃으니 ≈0."""
        armed = {f"{SYM}:LONG": {"symbol": SYM, "side": "LONG", "sl_price": 2375.0, "sl_level": 2375.0,
                                 "backstop_price": 2375.0, "rule": True, "sl_name": "첫 진입가 −5%", "armed_at": 0, "placed": {}}}
        pos = {**long_pos(round(L * 1000 / 2500, 3), 2500.0), "mark_price": 2450.0, "notional": round(L * 1000 / 2500, 3) * 2450.0}

        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertLess(p["notional_usdt"], 1.0, p["rule"])                                   # 위험 여유 0 → 사실상 0(막힘)
        self._run(account([pos]), fn, armed=armed)

    def test_full_cap_blocks_add(self) -> None:
        async def fn(c):
            p = (await (await c.get("/api/manual-entry/preview?side=LONG&lev=20&rule=c")).json())["plan"]
            self.assertTrue(p["blocked"], p)
        self._run(account([long_pos(2.5, 2500.0)]), fn)                           # 명목 6250 > 상한 3000

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
            self.assertAlmostEqual(fired[0]["notional_usdt"], REC_FIRST_SHARE * 1000 * L, delta=2.6)
            self.assertEqual(fired[0]["bracket"]["sl_price"], 2375.0)
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
