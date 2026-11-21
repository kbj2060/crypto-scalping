"""주문 결함 3건(2026-11-21, 사용자 «파이썬부터 지금 고친다»). 🔴실주문 0 -- signed·청산·취소는 전부 가짜.
(1) 익절 GTX·분할 익절·물타기 POST 의 «나갔는지 모름» = 진입과 같은 규칙(표식 조회 3번·1초 → 찾으면 걸림 · 못 찾으면 표식 DELETE 뒤 안 걸림) · 200 빈 본문 = 불명
(2) 망가진 무장 칸 하나가 감시 틱을 죽여 다른 코인 손절이 멈추지 않는다 · 그 칸은 지우지 않고 한 번만 알린다
(3) 오류로 끝난 손절 청산의 30초 재발동 전에 앞선 청산 peg(dbpg+측면 · 닫는 쪽)를 지우고 «없음»을 확인 -- 실패면 쏘지 않는다
실행: python -m pytest -q test/test_order_defect_fixes_20261121.py"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import live.execution.live_manual_peg_execute_20260912 as ex  # noqa: E402
import scripts.live_manual_peg_entry_20260912 as entry_mod  # noqa: E402
from dashboard import orders  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _fake_binance, _isolated_dirs  # noqa: E402

SYM = orders.MANUAL_EXEC_SYMBOLS["eth"]
FAKE_ENV = {"BINANCE_API_KEY": "k", "BINANCE_SECRET_KEY": "s"}


class Fx:
    """주문 POST 를 (오류 문자열, 실제로 접수됐나) 로 만든다 -- 접수됐으면 표식으로 찾을 수 있다."""

    def __init__(self, post=None, opens=None, fail=()):
        self.post, self.opens, self.fail = post, list(opens or []), set(fail)
        self.book: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.next_id = 500

    async def signed(self, _s, method, path, params, *_a):
        self.calls.append((method, path, dict(params)))
        if path == "/fapi/v2/positionRisk":
            return [{"positionSide": "LONG", "positionAmt": "1.0"}]
        if path == "/fapi/v1/openOrders":
            return {"__error__": "TimeoutError: list"} if "list" in self.fail else [dict(o) for o in self.opens]
        if path == "/fapi/v1/openAlgoOrders":
            return []
        if method == "POST" and path == "/fapi/v1/order":
            self.next_id += 1
            err, accepted = self.post if self.post else (None, True)
            if accepted:
                self.book[params["newClientOrderId"]] = {"orderId": self.next_id, "status": "NEW"}
            return {"__error__": err} if err else {"orderId": self.next_id}
        if method == "GET" and "origClientOrderId" in params:
            o = self.book.get(params["origClientOrderId"])
            return dict(o) if o else {"__error__": "400 [-2013] Order does not exist."}
        if method == "DELETE":
            if "keep" not in self.fail:
                self.opens = [o for o in self.opens if o.get("orderId") != params.get("orderId")]
            return {"__error__": "TimeoutError: delete"} if "delete" in self.fail else {"status": "CANCELED"}
        raise AssertionError(f"모르는 요청 {method} {path}")

    def n(self, method, key):
        return sum(1 for m, _p, prm in self.calls if m == method and key in prm)


def run_ex(fx, coro_fn):
    sleeps = []

    async def fast_sleep(sec, *_a):
        sleeps.append(sec)

    async def go():
        with mock.patch.dict(os.environ, FAKE_ENV), \
             mock.patch.object(ex, "signed", fx.signed), \
             mock.patch.object(ex, "_clock_offset", mock.AsyncMock(return_value=0)), \
             mock.patch.object(ex.asyncio, "sleep", fast_sleep), \
             mock.patch.object(entry_mod, "load_filters", mock.AsyncMock(return_value={"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 0.0})):
            return await coro_fn()
    return asyncio.run(go()), sleeps


POSTS = {
    "tp": lambda: ex.place_bracket(None, {"tp_price": 2600.0}, SYM, "LONG"),
    "ladder": lambda: ex.place_tp_ladder(None, [{"price": 2600.0, "w": 1}, {"price": 2650.0, "w": 1}], SYM, "LONG"),
    "add": lambda: ex.place_add_order(None, SYM, "LONG", 2400.0, 0.5),
}


def outcome(path, r):
    """(걸렸나, 주문번호들, 첫 오류)"""
    if path == "tp":
        return r["tp"]["placed"], [r["tp"]["order_id"]], r["tp"]["error"]
    if path == "ladder":
        return all(l["resting"] for l in r["legs"]), [l["order_id"] for l in r["legs"]], r["legs"][0]["error"]
    return r["placed"], [r["order_id"]], r["error"]


class UnknownPostTest(unittest.TestCase):
    def test_unknown_accepted_is_found(self) -> None:
        for path in POSTS:
            for err in ("503 Service Unavailable", "TimeoutError: read", "408 [-1007] Timeout waiting for response from backend server."):
                fx = Fx(post=(err, True))
                r, _ = run_ex(fx, POSTS[path])
                ok, ids, _e = outcome(path, r)
                self.assertTrue(ok, (path, err, r))
                self.assertTrue(all(ids), (path, err, ids))
                self.assertEqual(fx.n("DELETE", "origClientOrderId"), 0, (path, err))

    def test_unknown_not_accepted_lookup_3_then_cid_delete(self) -> None:
        for path in POSTS:
            fx = Fx(post=("TimeoutError: read", False))
            r, sleeps = run_ex(fx, POSTS[path])
            ok, _ids, err = outcome(path, r)
            self.assertFalse(ok, (path, r))
            posts = [p["newClientOrderId"] for m, _x, p in fx.calls if m == "POST"]
            gets = [p["origClientOrderId"] for m, _x, p in fx.calls if m == "GET" and "origClientOrderId" in p]
            dels = [p["origClientOrderId"] for m, _x, p in fx.calls if m == "DELETE" and "origClientOrderId" in p]
            self.assertEqual(gets, [c for c in posts for _ in range(3)], (path, gets))   # 주문마다 3번
            self.assertEqual(dels, posts, (path, dels))                                  # 주문마다 표식 DELETE 1번
            self.assertEqual(sleeps.count(1.0), 3 * len(posts), (path, sleeps))
            self.assertIn("확인하지 못했습니다", str(err), (path, err))

    def test_definite_reject_unchanged(self) -> None:
        for path in POSTS:
            fx = Fx(post=("400 [-5022] Due to the order could not be executed as maker", False))
            r, _ = run_ex(fx, POSTS[path])
            ok, _ids, err = outcome(path, r)
            self.assertFalse(ok, path)
            self.assertEqual(fx.n("GET", "origClientOrderId") + fx.n("DELETE", "origClientOrderId"), 0, path)
            self.assertTrue(str(err).startswith("400 [-5022]"), (path, err))

    def test_empty_200_body_is_unknown_not_typeerror(self) -> None:
        class Resp:
            status, headers = 200, {}

            async def json(self):
                return None   # aiohttp: application/json + 빈 본문 = None

        class Ctx:
            async def __aenter__(self):
                return Resp()

            async def __aexit__(self, *_a):
                return False

        sess = types.SimpleNamespace(request=lambda *_a, **_k: Ctx())
        r = asyncio.run(ex.signed(sess, "POST", "/fapi/v1/order", {"symbol": SYM}, "k", "s", 0))
        self.assertIsInstance(r, dict)
        e = r.get("__error__", "")
        self.assertTrue(e and not e[:1].isdigit(), r)   # 숫자로 시작하면 «확정 거부»로 읽힌다


def peg(cid, side, pside="LONG", oid=0):
    return {"clientOrderId": cid, "side": side, "positionSide": pside, "orderId": oid, "reduceOnly": False}


class CancelExitOrdersTest(unittest.TestCase):
    OPENS = [peg("dbpgL1", "SELL", oid=1), peg("dbpgL2", "BUY", oid=2), peg("dbtpL3", "SELL", oid=3),
             peg("web_user", "SELL", oid=4), peg("dbpgS5", "BUY", "SHORT", oid=5), peg("dbwtL6", "SELL", oid=6)]

    def test_only_our_closing_peg_is_cancelled_then_confirmed(self) -> None:
        fx = Fx(opens=self.OPENS)
        r, _ = run_ex(fx, lambda: ex.cancel_exit_orders(None, SYM, "LONG"))
        self.assertEqual(r["errors"], [], r)
        self.assertEqual([p["orderId"] for m, _x, p in fx.calls if m == "DELETE"], [1])
        self.assertEqual(sum(1 for _m, x, _p in fx.calls if x == "/fapi/v1/openOrders"), 2, "지운 뒤 다시 읽어 확인")

    def test_nothing_to_cancel_one_list(self) -> None:
        fx = Fx(opens=self.OPENS[1:])
        r, _ = run_ex(fx, lambda: ex.cancel_exit_orders(None, SYM, "LONG"))
        self.assertEqual((r["errors"], len(fx.calls)), ([], 1))

    def test_delete_error_but_gone_is_ok(self) -> None:   # 취소 응답이 아니라 다시 읽은 목록을 믿는다(체결·이미 취소)
        fx = Fx(opens=self.OPENS, fail=("delete",))
        r, _ = run_ex(fx, lambda: ex.cancel_exit_orders(None, SYM, "LONG"))
        self.assertEqual(r["errors"], [], r)

    def test_failures_are_errors(self) -> None:
        for fail in (("list",), ("keep",), ("keep", "delete")):
            fx = Fx(opens=self.OPENS, fail=fail)
            r, _ = run_ex(fx, lambda: ex.cancel_exit_orders(None, SYM, "LONG"))
            self.assertTrue(r["errors"], (fail, r))


ETH_TOUCHED = {"symbol": SYM, "side": "LONG", "lines": True, "sl_level": 2480.0, "tp_level": 2550.0, "tp_resting": True,
               "sl_price": 2480.0, "tp_price": 2550.0, "backstop_price": None, "armed_at": 1999.0}
POS = {"ok": True, "generated_at": "1970-01-01T00:00:00+00:00",
       "positions": [{"symbol": SYM, "side": "LONG", "qty": 0.5, "entry_price": 2500.0}]}


def cell(fn, name):
    """register 클로저 안의 값(상태 칸) -- 실주문 없이 «같은 코인 청산 중»을 만든다."""
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


def tick(entries: dict, n=1, cancel=None, exit_phase="filled_maker", sec=None, busy_on_clear=False):
    """감시 틱 n 번 -- ETH 1초 가격이 손절선 2480 에 닿았다. 돌려주기 = (호출, 알림, 저장본).
    busy_on_clear = 익절 지정가 취소(clear_bracket) await 사이에 같은 코인 수동 청산이 시작된 것으로."""
    calls, alerts, box = [], [], {}

    async def fake_account(*_a, **_k):
        return POS

    async def swr(_k, _ttl, produce, **_kw):
        return await produce()

    async def clear(_s, symbol, pside, *a, **k):
        calls.append(("clear", symbol, pside))
        if busy_on_clear:
            cell(cell(box["api"].bracket_tick, "exit_slot"), "manual_entry_state").update(phase="working", kind="exit", asset="eth")
        return {"cancelled": 1, "errors": []}

    async def exit_(_s, plan, state):
        calls.append(("exit", plan.get("symbol")))
        state.update(phase=exit_phase, filled=1.0 if exit_phase.startswith("filled") else 0.0, quantity=1.0)
        return state

    async def cancel_exits(_s, symbol, pside):
        calls.append(("cancel_exits", symbol, pside))
        return cancel or {"cancelled": 1, "errors": []}

    async def boom(*_a, **_k):
        raise AssertionError("실주문 경로")

    async def hist(_a):
        return []

    fake_wd = types.ModuleType("scripts.ops_watchdog")   # 진짜 모듈은 .env 를 읽고 텔레그램으로 보낸다 -- 가짜로
    fake_wd.send_telegram = lambda msg: alerts.append(msg)

    async def go():
        with mock.patch.object(orders, "fetch_account", fake_account), \
             mock.patch.object(orders, "exec_enabled", lambda: True), \
             mock.patch.dict(sys.modules, {"scripts.ops_watchdog": fake_wd}), \
             mock.patch.multiple(orders, clear_bracket=clear, run_exit=exit_, run_entry=boom, place_bracket=boom, place_add_order=boom), \
             mock.patch.object(orders, "cancel_exit_orders", cancel_exits, create=True):
            api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                  fetch_binance_json=_fake_binance, swr_cached=swr, _timed_produce=None,
                                  load_market_history=hist, situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60,
                                  sec_prices=sec or (lambda a: [(2001, 2490.0), (2002, 2479.0)] if a == "eth" else None))
            box["api"] = api
            (orders.LIVE_DIR / "manual_bracket_state.json").write_text(json.dumps(entries), "utf-8")
            for _ in range(n):
                await api.bracket_tick()
        return calls, alerts, json.loads((orders.LIVE_DIR / "manual_bracket_state.json").read_text("utf-8"))

    with _isolated_dirs():
        return asyncio.run(go())


class CorruptEntryTest(unittest.TestCase):
    BAD = {
        "symbol 없음": {"side": "LONG", "lines": True, "sl_level": 1.0, "armed_at": 1.0},
        "armed_at 글자": {"symbol": "XRPUSDC", "side": "LONG", "lines": True, "sl_level": 1.0, "armed_at": "abc"},
        "dict 아님": "garbage",
        "손절 조각이 목록 아님": {"symbol": "XRPUSDC", "side": "LONG", "lines": True, "sl_legs": 5, "armed_at": 1.0},
    }

    def test_bad_entry_first_does_not_stop_other_coin(self) -> None:
        for name, bad in self.BAD.items():
            calls, alerts, st = tick({"XRPUSDC:LONG": bad, f"{SYM}:LONG": dict(ETH_TOUCHED)}, n=3)
            self.assertIn(("exit", SYM), calls, (name, calls))           # 망가진 칸 뒤 ETH 손절은 나간다
            self.assertEqual(st.get("XRPUSDC:LONG"), bad, (name, st))    # 망가진 칸은 지우지 않는다
            self.assertEqual(len(alerts), 1, (name, alerts))             # 틱 3번에 알림 1번
            self.assertIn("XRPUSDC:LONG", alerts[0], name)

    def test_same_kind_changing_message_alerts_once(self) -> None:   # 메시지에 시각·숫자가 바뀌어 들어가도 2초마다 보내지 않는다
        n = []

        def sec(_a):
            n.append(1)
            raise ValueError(f"t={time.time()} n={len(n)}")

        _c, alerts, st = tick({f"{SYM}:LONG": dict(ETH_TOUCHED)}, n=3, sec=sec)
        self.assertEqual((len(n), len(alerts)), (3, 1), alerts)
        self.assertIn(f"{SYM}:LONG", st)


class RetryCancelsPreviousExitTest(unittest.TestCase):
    RETRY = {**ETH_TOUCHED, "fired": True, "tp_resting": False, "retry_after": 1.0, "last_error": "주문 1 취소를 확인하지 못했습니다"}

    def test_retry_cancels_old_peg_then_exits(self) -> None:
        calls, _a, st = tick({f"{SYM}:LONG": dict(self.RETRY)})
        self.assertEqual([c[0] for c in calls][:2], ["cancel_exits", "exit"], calls)
        self.assertEqual(st, {})

    def test_cancel_failure_defers_and_keeps_arm(self) -> None:
        calls, _a, st = tick({f"{SYM}:LONG": dict(self.RETRY)}, cancel={"cancelled": 0, "errors": ["openOrders: TimeoutError"]})
        self.assertEqual([c[0] for c in calls], ["cancel_exits"], calls)   # 새 청산 없음
        b = st[f"{SYM}:LONG"]
        self.assertTrue(b["fired"] and b["retry_after"] - time.time() > 20, b)
        self.assertIn("TimeoutError", b["last_error"])

    def test_sl_leg_retry_also_cancels_first(self) -> None:
        legs = {**ETH_TOUCHED, "sl_level": None, "retry_after": 1.0,
                "sl_legs": [{"level": 2480.0, "w": 1, "fired": True}, {"level": 2400.0, "w": 1}]}
        calls, _a, _st = tick({f"{SYM}:LONG": legs}, cancel={"cancelled": 0, "errors": ["x"]})
        self.assertEqual([c[0] for c in calls], ["cancel_exits"], calls)

    def test_retry_skips_when_same_coin_exit_started_meanwhile(self) -> None:   # 익절 취소 await 틈에 같은 코인 수동 청산 -- 그 따라가기 주문을 지우지 않는다
        calls, _a, st = tick({f"{SYM}:LONG": {**ETH_TOUCHED, "retry_after": 1.0}}, busy_on_clear=True)
        self.assertEqual([c[0] for c in calls], ["clear"], calls)   # cancel_exit_orders 0 · 청산 0
        b = st[f"{SYM}:LONG"]
        self.assertTrue(b["fired"] and b["retry_after"] == 1.0, b)   # 무장 유지 · 다음 틱에 다시

    def test_first_fire_does_not_list_orders(self) -> None:   # 첫 발동은 지금 그대로(조회 하나 늘리지 않는다 · 손절 속도)
        calls, _a, _st = tick({f"{SYM}:LONG": dict(ETH_TOUCHED)})
        self.assertNotIn("cancel_exits", [c[0] for c in calls], calls)
        self.assertIn(("exit", SYM), calls)


if __name__ == "__main__":
    unittest.main()
