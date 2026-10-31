"""진입 «지정가 대기»(2026-10-31) -- 모의 거래소로 집행 경로를 끝까지. 🔴네트워크 0(signed·시계·레버리지 전부 가짜).

① 고정가 1건 · 리페그 0 · 시장가 0 → 체결 ② 취소 = 시장가 0 ③ 수정 = 취소 확인 뒤 새 가격·새 수량으로 재주문
④ -5022 = 다시 안 걸고 거부 ⑤ 만료 = 시장가 0 ⑥ 거래소에서 취소 = 시장가 0 ⑦ 첫 체결 on_fill 한 번
실행: python -m pytest -q test/test_manual_limit_entry_20261031.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402

PLAN = {"symbol": "ETHUSDC", "side": "BUY", "positionSide": "LONG", "price": 2500.0, "quantity": 1.0, "target_leverage": 0}


class Fake:
    """주문마다 번호를 매긴다. fill_after = 그 주문을 이 횟수째 조회할 때 체결(None = 안 체결)."""

    def __init__(self, fill_after=None, reject=False, partial=0.0, kill_at=None):
        self.limits, self.markets, self.deletes = [], [], []
        self.fill_after, self.reject, self.partial, self.kill_at = fill_after, reject, partial, kill_at
        self.status, self.filled, self.polls, self.oid, self.qty = "NEW", 0.0, 0, 0, 0.0

    async def signed(self, _s, method, path, params, *_a):
        if path != "/fapi/v1/order":
            return {"__error__": f"unexpected {method} {path}"}
        if method == "POST":
            if params["type"] == "MARKET":
                self.markets.append(params)
                return {"orderId": 99, "status": "FILLED", "executedQty": str(params["quantity"])}
            if self.reject:
                return {"__error__": "400 [-5022] Due to the order could not be executed as maker"}
            self.limits.append(params)
            self.oid, self.qty, self.status, self.filled, self.polls = self.oid + 1, float(params["quantity"]), "NEW", 0.0, 0
            return {"orderId": self.oid, "status": "NEW", "executedQty": "0"}
        if method == "GET":
            self.polls += 1
            if self.partial and self.polls == 1:
                self.filled = self.partial
            if self.fill_after and self.polls >= self.fill_after:
                self.status, self.filled = "FILLED", self.qty
            if self.kill_at and self.polls >= self.kill_at and self.status == "NEW":
                self.status = "CANCELED"
            return {"orderId": self.oid, "status": self.status, "executedQty": str(self.filled)}
        if method == "DELETE":
            self.deletes.append(params)
            if self.status == "NEW":
                self.status = "CANCELED"
            return {"orderId": params["orderId"]}


def run(fx, max_sec=0.5, plan=None, drive=None, on_fill=None):
    state: dict = {}

    async def go():
        t = asyncio.create_task(ex.run_entry_limit(None, dict(plan or PLAN), state, max_sec, on_fill=on_fill))
        if drive:
            await drive(state)
        await t

    with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "POLL_SEC", 0.02), \
         mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
         mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value={"ok": True})), \
         mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
        asyncio.run(go())
    return state


class LimitEntry(unittest.TestCase):
    def test_fixed_price_fill_no_repeg_no_market(self):
        fx = Fake(fill_after=3)
        st = run(fx)
        self.assertEqual(st["phase"], "filled_maker")
        self.assertEqual(len(fx.limits), 1)
        self.assertEqual((fx.limits[0]["price"], fx.limits[0]["timeInForce"]), (2500.0, "GTX"))
        self.assertEqual(fx.markets, [])
        self.assertTrue(fx.limits[0]["newClientOrderId"].startswith(ex.WAIT_TAG))   # 재시작 고아 정리 대상

    def test_cancel_no_market(self):
        fx = Fake()

        async def drive(st):
            await asyncio.sleep(0.1)
            st["action"] = "cancel"
        st = run(fx, drive=drive)
        self.assertEqual((st["phase"], st["cancel_reason"]), ("cancelled", "사용자 취소"))
        self.assertEqual((len(fx.deletes), fx.markets), (1, []))

    def test_amend_cancels_then_replaces_with_new_price_and_qty(self):
        fx = Fake()

        async def drive(st):
            await asyncio.sleep(0.1)
            st["amend"] = {"price": 2490.0, "qty": 1.004}
            await asyncio.sleep(0.3)
            fx.fill_after = 1          # 새 주문이 걸린 뒤 체결
            await asyncio.sleep(0.2)
        st = run(fx, max_sec=2, drive=drive)
        self.assertEqual([l["price"] for l in fx.limits], [2500.0, 2490.0])
        self.assertEqual(fx.limits[1]["quantity"], 1.004)
        self.assertEqual((len(fx.deletes), st["amends"], st["phase"], fx.markets), (1, 1, "filled_maker", []))

    def test_auto_reprice_follows_basis_drift_then_stops_after_fill_starts(self):
        fx = Fake()
        targets = iter([2499.4, 2499.4, 2499.4])   # 첫 확인: 2500 → 2499.4 (2.4bp 벗어남) · 뒤는 이미 그 값(1bp 미만)

        async def rp():
            return next(targets, 2499.4)

        async def drive(st):
            await asyncio.sleep(0.5)
            fx.fill_after = 1
            await asyncio.sleep(0.2)
        state: dict = {}

        async def go():
            t = asyncio.create_task(ex.run_entry_limit(None, dict(PLAN), state, 3, reprice=rp))
            await drive(state)
            await t
        with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "POLL_SEC", 0.02), mock.patch.object(ex, "REPRICE_SEC", 0.1), \
             mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value={"ok": True})), \
             mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            asyncio.run(go())
        self.assertEqual([l["price"] for l in fx.limits], [2500.0, 2499.4])   # 한 번만 재조정(두 번째 확인은 변화 1bp 미만)
        self.assertEqual((state["reprices"], state.get("amends", 0), len(fx.deletes), state["phase"], fx.markets), (1, 0, 1, "filled_maker", []))

    def test_no_reprice_once_partially_filled(self):
        fx = Fake(partial=0.4)
        calls = []

        async def rp():
            calls.append(1)
            return 2400.0
        state: dict = {}

        async def go():
            t = asyncio.create_task(ex.run_entry_limit(None, dict(PLAN), state, 0.5, reprice=rp))
            await asyncio.sleep(0.3)
            state["action"] = "cancel"
            await t
        with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "POLL_SEC", 0.02), mock.patch.object(ex, "REPRICE_SEC", 0.05), \
             mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value={"ok": True})), \
             mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            asyncio.run(go())
        self.assertEqual((calls, len(fx.limits), state.get("reprices", 0)), ([], 1, 0))   # 체결이 시작됐으면 확인조차 안 한다

    def test_amend_after_partial_fill_keeps_old_price_and_rests_remainder(self):   # 검증 D4
        fx = Fake(partial=0.4)

        async def drive(st):
            await asyncio.sleep(0.1)
            st["amend"] = {"price": 2490.0, "qty": 1.2}
            await asyncio.sleep(0.2)
            st["action"] = "cancel"
        run(fx, max_sec=2, drive=drive)
        self.assertEqual([(l["price"], l["quantity"]) for l in fx.limits], [(2500.0, 1.0), (2500.0, 0.6)])   # 가격·총량 수정은 버려지고 남은 수량만 옛 가격에

    def test_replace_rejected_after_amend_retries_instead_of_dying(self):   # 검증 D6
        fx = Fake()
        orig = fx.signed
        rejects = {"n": 0}

        async def signed(_s, method, path, params, *a):
            if method == "POST" and path == "/fapi/v1/order" and params["type"] == "LIMIT" and len(fx.limits) == 1 and rejects["n"] < 1:
                rejects["n"] += 1
                return {"__error__": "400 [-5022] Due to the order could not be executed as maker"}
            return await orig(_s, method, path, params, *a)
        fx.signed = signed

        async def drive(st):
            await asyncio.sleep(0.1)
            st["amend"] = {"price": 2490.0, "qty": 1.004}
            await asyncio.sleep(1.6)
            fx.fill_after = 1
            await asyncio.sleep(0.3)
        mp = mock.AsyncMock(return_value=2491.0)

        async def go():
            state: dict = {}
            t = asyncio.create_task(ex.run_entry_limit(None, dict(PLAN), state, 5, reprice=mp))
            await drive(state)
            await t
            return state
        with mock.patch.object(ex, "signed", fx.signed), mock.patch.object(ex, "POLL_SEC", 0.02), \
             mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value={"ok": True})), \
             mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            st = asyncio.run(go())
        self.assertEqual((st["phase"], rejects["n"], [l["price"] for l in fx.limits]), ("filled_maker", 1, [2500.0, 2491.0]))   # 새 목표가로 다시 걸어 살아남음

    def test_unknown_submit_outcome_deletes_by_client_id_and_stops(self):   # 검증 D3
        fx = Fake()
        calls = []

        async def signed(_s, method, path, params, *a):
            calls.append((method, params.get("origClientOrderId") and "cid"))
            if method == "POST":
                return {"__error__": "timeout"}
            return {"__error__": "-2013 Order does not exist"}
        with mock.patch.object(ex, "signed", signed), mock.patch.object(ex, "POLL_SEC", 0.02), mock.patch.object(asyncio, "sleep", mock.AsyncMock()), \
             mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.object(ex, "ensure_leverage", mock.AsyncMock(return_value={"ok": True})), \
             mock.patch.dict(os.environ, {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}):
            st: dict = {}
            asyncio.run(ex.run_entry_limit(None, dict(PLAN), st, 1))
        self.assertEqual(st["phase"], "error")
        self.assertEqual([c for c in calls if c[0] == "GET"].__len__(), 3)          # 세 번 찾아 본다
        self.assertIn(("DELETE", "cid"), calls)                                      # 끝내 모르면 표식으로 지운다

    def test_post_only_reject_does_not_repeg(self):
        fx = Fake(reject=True)
        st = run(fx)
        self.assertEqual((st["phase"], fx.limits), ("rejected", []))
        self.assertIn("메이커", st["error"])

    def test_expiry_cancels_without_market(self):
        fx = Fake()
        st = run(fx, max_sec=0.15)
        self.assertEqual((st["phase"], st["cancel_reason"], fx.markets), ("cancelled", "대기 시간 만료", []))

    def test_exchange_side_cancel_is_cancel(self):
        fx = Fake(kill_at=2)
        st = run(fx, max_sec=2)
        self.assertEqual((st["phase"], st["cancel_reason"], fx.markets, len(fx.limits)), ("cancelled", "거래소에서 취소됨", [], 1))

    def test_on_fill_once_on_partial(self):
        fx = Fake(partial=0.4)
        n = []

        async def cb():
            n.append(1)

        async def drive(st):
            await asyncio.sleep(0.15)
            st["action"] = "cancel"
        st = run(fx, drive=drive, on_fill=cb)
        self.assertEqual((n, st["phase"], st["filled"]), ([1], "cancelled", 0.4))   # 든 몫은 그대로 -- 손절은 on_fill 이 걸었다


if __name__ == "__main__":
    unittest.main()
