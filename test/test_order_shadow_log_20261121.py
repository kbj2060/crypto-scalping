"""주문 경로 그림자 기록(2026-11-21 · live/execution/order_shadow.py · rtorder/SHADOW.md 8절).
🔴실주문 0: 거래소는 127.0.0.1 가짜 aiohttp 앱뿐(FAPI 를 그 주소로 바꾼다) · 키·서명은 합성 값.
① signed() 의 응답 처리(정상·빈 본문 EmptyResponse·400 코드·JSON 아님·연결 끊김)가 기록 켬/끔에서 같다 -- 원문은 json() 이 받아 둔 바이트를 쓴다
② 진입·청산 접수(진짜 offline_app 라우트 + order_request_log) → start · wire · end 줄 + 접수 기록 줄의 chain 이 같다
③ 감시 틱 발동 → start(op=tick) · input account·bars·sec·exit_plan · wire · end · 아무 일 없는 틱은 안 남는다
④ 화면 입력(wait-action) → 그 칸을 도는 작업 chain 의 input handler · 체결 알림 = chain "*"
⑤ 가림(서명·키 머리·listenKey) · chmod 600 · 50MB 돌림 · 쓰기 실패(쓸 수 없는 폴더)여도 주문 결과가 같다
python -m pytest -q test/test_order_shadow_log_20261121.py"""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp import ClientSession, web  # noqa: E402
from aiohttp.test_utils import TestServer, make_mocked_request  # noqa: E402

from dashboard import orders, server  # noqa: E402
from live.execution import live_manual_peg_execute_20260912 as ex  # noqa: E402
from live.execution import order_shadow as shadow  # noqa: E402
from scripts import live_binance_account_20260910 as acct_mod  # noqa: E402
import test_chart_lines_entry_20261028 as cl  # noqa: E402
from test_chart_lines_entry_20261028 import POS, SYM  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import FAKE_STATE, _fake_binance  # noqa: E402
from test_order_defect_fixes_20261121 import ETH_TOUCHED, POS as DPOS  # noqa: E402

KEY, SECRET = "test_key_0000", "test_secret_0000"


def fake_exchange() -> web.Application:
    """127.0.0.1 가짜 거래소 -- 주문 POST 는 NEW · 조회는 FILLED · 몇 가지 깨진 응답."""
    async def order(request: web.Request) -> web.Response:
        assert request.headers.get("X-MBX-APIKEY") == KEY
        return web.json_response({"orderId": 7, "status": "FILLED" if request.method == "GET" else "NEW", "executedQty": "0.5", "listenKey": "lk-secret-1"})

    async def empty(_r):
        return web.Response(status=200, body=b"", content_type="application/json")

    async def bad(_r):
        return web.json_response({"code": -5022, "msg": "Due to the order could not be executed as maker"}, status=400)

    async def text(_r):
        return web.Response(status=200, text="not json", content_type="text/plain")

    async def drop(request):
        request.transport.close()
        return web.Response()

    async def time_(_r):
        return web.json_response({"serverTime": int(time.time() * 1000)})

    async def book(_r):
        return web.json_response({"symbol": SYM, "bidPrice": "2500.00", "askPrice": "2500.01"})

    app = web.Application()
    app.router.add_route("*", "/fapi/v1/order", order)
    app.router.add_get("/fapi/v1/empty", empty)
    app.router.add_get("/fapi/v1/bad", bad)
    app.router.add_get("/fapi/v1/text", text)
    app.router.add_get("/fapi/v1/drop", drop)
    app.router.add_get("/fapi/v1/time", time_)
    app.router.add_get("/fapi/v1/ticker/bookTicker", book)
    return app


def rows(path: Path) -> list[dict]:
    shadow.flush()
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()] if path.exists() else []


class SignedParityTest(unittest.TestCase):
    def test_json_handling_same_with_and_without_record(self) -> None:
        async def go():
            ts = TestServer(fake_exchange(), host="127.0.0.1")
            await ts.start_server()
            base = str(ts.make_url("")).rstrip("/")
            assert base.startswith("http://127.0.0.1:")
            out = {}
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ex, "FAPI", base):
                shadow.set_path(Path(tmp) / "order_shadow.jsonl")
                async with ClientSession() as s:
                    for on in ("0", "1"):
                        with mock.patch.dict(os.environ, {"ORDER_SHADOW_LOG": on}):
                            shadow.new_chain("test")
                            res = []
                            for path in ("/fapi/v1/order", "/fapi/v1/empty", "/fapi/v1/bad", "/fapi/v1/text", "/fapi/v1/drop"):
                                r = await ex.signed(s, "GET", path, {"symbol": SYM}, KEY, SECRET, 0)
                                if isinstance(r, dict) and "__error__" in r and "ClientConnection" in r["__error__"]:
                                    r = {"__error__": "끊김"}   # 끊김 문구는 포트·연결마다 다르다 -- 갈래만 본다
                                res.append(r)
                            out[on] = res
                rs = rows(Path(tmp) / "order_shadow.jsonl")
            await ts.close()
            return out, rs

        out, rs = asyncio.run(go())
        import re
        norm = lambda v: json.loads(re.sub(r"timestamp=\d+&signature=[0-9a-f]+", "T", json.dumps(v, ensure_ascii=False)))   # noqa: E731 -- 예외 문구의 URL 은 호출 시각마다 다르다
        self.assertEqual(norm(out["0"]), norm(out["1"]), "기록 켬/끔에서 signed() 결과가 다르다")
        self.assertEqual(out["1"][0]["status"], "FILLED")   # 가짜 거래소 GET 주문 조회 = FILLED
        self.assertTrue(out["1"][1]["__error__"].startswith("EmptyResponse: HTTP 200 본문 None"), out["1"][1])   # PR 55 갈래 그대로
        self.assertTrue(out["1"][2]["__error__"].startswith("400 [-5022]"), out["1"][2])
        self.assertTrue(out["1"][3]["__error__"].startswith("ContentTypeError"), out["1"][3])
        wires = [r for r in rs if r["rec"] == "wire"]
        self.assertEqual(len(wires), 5, rs)   # 끔(0) 은 0줄
        self.assertEqual([w["status"] for w in wires], [200, 200, 400, 200, None])
        self.assertEqual(wires[1]["body"], "")
        self.assertIn("not json", wires[3]["body"])
        self.assertIsNotNone(wires[4]["error"])
        for w in wires:
            self.assertNotIn("signature", w["target"])
            self.assertIn("timestamp=", w["target"])
        text = json.dumps(rs)
        self.assertNotIn(KEY, text)
        self.assertNotRegex(text, r"signature=[0-9a-f]{16}", "서명 값이 파일에 남으면 안 된다(예외 문구 포함)")
        self.assertNotIn("lk-secret-1", text)
        self.assertIn("<listenKey>", wires[0]["body"])


class ShadowFlowTest(unittest.TestCase):
    """진짜 라우트(offline_app)로 진입·청산 접수 → 가짜 집행이 signed() 로 가짜 거래소에 주문 하나."""
    _run = cl.ChartLinesTest._run   # 같은 격리(폴더·계좌·실행 스위치) -- 그 시험들은 다시 돌리지 않는다

    def _flow(self, base: str, env: dict) -> dict:
        seen: dict = {}

        async def fake_entry(_s, plan, state):
            async with ClientSession() as s:
                r = await ex.signed(s, "POST", "/fapi/v1/order", {"symbol": plan["symbol"], "side": plan["side"], "quantity": plan["quantity"]}, KEY, SECRET, 0)
            seen["entry"] = r
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_exit(_s, plan, state):
            async with ClientSession() as s:
                r = await ex.signed(s, "POST", "/fapi/v1/order", {"symbol": plan["symbol"], "side": plan["side"]}, KEY, SECRET, 0)
            seen["exit"] = r
            state.update(phase="filled_maker", filled=float(plan["quantity"]), quantity=float(plan["quantity"]))
            return state

        async def fake_bracket(_s, b, symbol, pside):
            return {"placed": True, "tp": {"placed": True, "price": b["tp_price"]}}

        async def fn(c):
            for k, v in env.items():
                if k == "path":
                    shadow.set_path(v)
            r = await (await c.post("/api/manual-entry/submit?side=SHORT&confirm=1&lev=20&pct=10&tp=2400&sl=2600")).json()
            seen["submit_ok"] = r.get("ok")
            seen["submit"] = r
            for _ in range(100):
                await asyncio.sleep(0.02)
                if "entry" in seen and (server.LIVE_DIR / "manual_bracket_state.json").exists():
                    break
            await asyncio.sleep(0.05)
            r = await (await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=50&asset=eth")).json()
            seen["exit_ok"] = r.get("ok"), r.get("error")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if "exit" in seen:
                    break
            await asyncio.sleep(0.1)
            seen["live"] = server.LIVE_DIR

        with mock.patch.object(ex, "FAPI", base), mock.patch.dict(os.environ, {k: v for k, v in env.items() if k != "path"}):
            self._run(fn, acct=POS, run_entry=fake_entry, run_exit=fake_exit, place_bracket=fake_bracket, place_tp_ladder=fake_bracket,
                      clear_bracket=lambda *a, **k: _ok())
        return seen

    def test_entry_exit_rows_and_failure_does_not_change_orders(self) -> None:
        import threading
        loop = asyncio.new_event_loop()   # 가짜 거래소는 따로 도는 루프(스레드) -- 시험 본체는 asyncio.run 으로 제 루프를 쓴다
        ts = TestServer(fake_exchange(), host="127.0.0.1")
        loop.run_until_complete(ts.start_server())
        th = threading.Thread(target=loop.run_forever, daemon=True)
        th.start()
        base = str(ts.make_url("")).rstrip("/")
        try:
            captured: dict = {}
            orig = shadow._write

            def spy(path, lines):   # 쓰는 줄을 같이 잡아 둔다(임시 폴더는 시험 끝에 지워진다)
                captured.setdefault("lines", []).extend(lines)
                orig(path, lines)

            with mock.patch.object(shadow, "_write", spy):
                on = self._flow(base, {"ORDER_SHADOW_LOG": "1"})
                shadow.flush()
            off = self._flow(base, {"ORDER_SHADOW_LOG": "0"})
            bad = self._flow(base, {"ORDER_SHADOW_LOG": "1", "path": Path("/proc/rtorder_no_such_dir/order_shadow.jsonl")})   # 쓸 수 없는 곳
            shadow.flush()
        finally:
            asyncio.run_coroutine_threadsafe(ts.close(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            th.join(10)
            loop.close()
        for name, s in (("on", on), ("off", off), ("bad", bad)):
            self.assertTrue(s["submit_ok"], (name, s))
            self.assertEqual(s.get("entry", {}).get("status"), "NEW", (name, s.get("submit")))
            self.assertEqual(s["exit"]["status"], "NEW", (name, s))
        self.assertEqual(on["exit_ok"], off["exit_ok"])
        self.assertEqual(bad["exit_ok"], off["exit_ok"], "기록이 실패해도 주문 결과가 같아야 한다")
        rs = [json.loads(x) for x in captured.get("lines", [])]
        kinds = [(r["rec"], r.get("op") or r.get("what") or r.get("target", "").split("?")[0]) for r in rs]
        self.assertIn(("start", "entry"), kinds, kinds)
        self.assertIn(("start", "exit"), kinds, kinds)
        self.assertIn(("wire", "/fapi/v1/order"), kinds, kinds)
        self.assertGreaterEqual(sum(1 for k in kinds if k[0] == "end"), 2, kinds)
        st = next(r for r in rs if r["rec"] == "start" and r["op"] == "entry")
        for k in ("chain", "t", "mono", "plan", "slot", "limit", "file", "limits"):
            self.assertIn(k, st, st)
        same = [r for r in rs if r["chain"] == st["chain"]]
        self.assertEqual([r["rec"] for r in same][0], "start")
        self.assertIn("wire", [r["rec"] for r in same])
        self.assertEqual([r["rec"] for r in same][-1], "end")
        ex_start = next(r for r in rs if r["rec"] == "start" and r["op"] == "exit")
        self.assertEqual(ex_start["rearm"], True)   # 50% = 부분 청산
        text = "\n".join(captured["lines"])
        self.assertNotIn("signature=", text)
        self.assertNotIn(KEY, text)


async def _ok():
    return {"cancelled": 1, "errors": []}


class ShadowTickTest(unittest.TestCase):
    def _tick(self, entries: dict, sec, run_exit=None):
        calls = []

        async def fake_account(*_a, **_k):
            return DPOS

        async def swr(_k, _ttl, produce, **_kw):
            return await produce()

        async def clear(_s, symbol, pside, *a, **k):
            async with ClientSession() as s:
                await ex.signed(s, "DELETE", "/fapi/v1/order", {"symbol": symbol}, KEY, SECRET, 0)
            calls.append("clear")
            return {"cancelled": 1, "errors": []}

        async def exit_(_s, plan, state):
            calls.append("exit")
            state.update(phase="filled_maker", filled=1.0, quantity=1.0)
            return state

        async def cancel_exits(_s, symbol, pside):
            return {"cancelled": 0, "errors": []}

        async def hist(_a):
            return [{"time": 1000, "close": 2490.0}]

        async def fetch(url, params=None, **kw):   # 호가·클라인 = 시험 공용 가짜 · 펀딩 등 나머지 = 빈 dict(표시용 칸)
            r = await _fake_binance(url, params or {}, **kw)
            return {} if r is None else r

        async def boom(*_a, **_k):
            raise AssertionError("실주문 경로")

        async def filters(_s, _symbol):
            return {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 5.0}

        async def go(base):
            with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ex, "FAPI", base), \
                 mock.patch.object(orders, "LIVE_DIR", Path(tmp)), \
                 mock.patch.object(orders, "fetch_account", fake_account), mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.multiple(orders, clear_bracket=clear, run_exit=run_exit or exit_, run_entry=boom, place_bracket=boom), \
                 mock.patch.object(orders, "cancel_exit_orders", cancel_exits, create=True), mock.patch.object(orders, "load_filters", filters):
                (Path(tmp) / "manual_bracket_state.json").write_text(json.dumps(entries), "utf-8")
                (Path(tmp) / "eth_position_sizing_state.json").write_text(json.dumps(FAKE_STATE), "utf-8")   # 청산 조립이 읽는 사이징 상태(시험 공용 가짜)
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                      fetch_binance_json=fetch, swr_cached=swr, _timed_produce=None, load_market_history=hist,
                                      situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=sec)
                await api.shadow_bracket_tick()
                await api.shadow_bracket_tick()   # 둘째 = 무장이 내려가 아무 일 없음
                return rows(Path(tmp) / "order_shadow.jsonl"), calls

        async def main():
            ts = TestServer(fake_exchange(), host="127.0.0.1")
            await ts.start_server()
            try:
                return await go(str(ts.make_url("")).rstrip("/"))
            finally:
                await ts.close()

        with mock.patch.dict(os.environ, {"ORDER_SHADOW_LOG": "1"}):
            return asyncio.run(main())

    def test_fire_tick_rows_and_quiet_tick_dropped(self) -> None:
        rs, calls = self._tick({f"{SYM}:LONG": dict(ETH_TOUCHED)}, sec=lambda a: [(2001, 2490.0), (2002, 2479.0)] if a == "eth" else None)
        self.assertIn("exit", calls)
        chains = {r["chain"] for r in rs}
        self.assertEqual(len(chains), 1, f"발동 틱 하나만 남는다(무장 내린 뒤 틱은 버림) {chains}")
        seq = [(r["rec"], r.get("op") or r.get("what")) for r in rs]
        self.assertEqual(seq[0], ("start", "tick"))
        self.assertEqual(seq[-1][0], "end")
        for need in [("input", "account"), ("input", "sec"), ("input", "exit_plan"), ("wire", None)]:
            self.assertIn(need, seq, seq)
        self.assertIn(f"{SYM}:LONG", rs[0]["file"], "틱 시작의 무장 파일")
        self.assertIsInstance(rs[0]["slot"], dict)
        self.assertNotIn(f"{SYM}:LONG", rs[-1]["file"], "끝 = 저장한 무장(청산 뒤 내림)")

    def test_quiet_tick_writes_nothing(self) -> None:
        rs, calls = self._tick({f"{SYM}:LONG": {**dict(ETH_TOUCHED), "sl_level": 2000.0, "sl_price": 2000.0}}, sec=lambda a: [(2001, 2490.0)])
        self.assertEqual((rs, calls), ([], []))


def find_cell(fn, name, seen=None):
    """register 클로저를 따라가며 이름이 name 인 칸(상태 dict) -- 실주문 없이 «도는 대기 칸»을 만든다."""
    seen = seen if seen is not None else set()
    if id(fn) in seen or not hasattr(fn, "__code__"):
        return None
    seen.add(id(fn))
    for n, c in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            v = c.cell_contents
        except ValueError:
            continue
        if n == name:
            return v
        got = find_cell(v, name, seen)
        if got is not None:
            return got
    return None


class HandlerWakeAndFileTest(unittest.TestCase):
    def test_handler_input_wake_mode_and_rotation(self) -> None:
        async def go(tmp: Path):
            with mock.patch.object(orders, "LIVE_DIR", tmp), mock.patch.object(orders, "exec_enabled", lambda: True):
                api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                                      fetch_binance_json=None, swr_cached=None, _timed_produce=None, load_market_history=None,
                                      situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60)
                waits = find_cell(api.api_manual_wait_action, "manual_wait_states")
                waits["eth"] = {"phase": "working", "mode": "limit", "side": "LONG"}
                shadow.bind_slot(waits["eth"], "entry-abc")
                r = await api.api_manual_wait_action(make_mocked_request("POST", "/api/manual-entry/wait-action?action=cancel&confirm=1&asset=eth"))
                ex.wake_fills()
                return r.status, rows(tmp / "order_shadow.jsonl")

        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"ORDER_SHADOW_LOG": "1"}):
            st, rs = asyncio.run(go(Path(t)))
            mode = stat.S_IMODE(os.stat(Path(t) / "order_shadow.jsonl").st_mode)
        self.assertEqual(st, 200)
        h = [r for r in rs if r.get("what") == "handler"]
        self.assertEqual(len(h), 1, rs)
        self.assertEqual((h[0]["chain"], h[0]["args"][0], h[0]["value"]["status"]), ("entry-abc", "api_manual_wait_action", 200))
        self.assertTrue(any(r.get("what") == "wake" and r["chain"] == "*" for r in rs), rs)
        self.assertEqual(mode, 0o600)
        # 50MB(여기서는 작게) 넘으면 .1 로 한 번 돌린다
        with tempfile.TemporaryDirectory() as t, mock.patch.object(shadow, "SHADOW_MAX_BYTES", 100), mock.patch.dict(os.environ, {"ORDER_SHADOW_LOG": "1"}):
            p = Path(t) / "order_shadow.jsonl"
            shadow.set_path(p)
            shadow.new_chain("rot")
            for i in range(3):
                shadow.rec("input", what="x", value="y" * 80)
                shadow.flush()
            self.assertTrue((Path(t) / "order_shadow.jsonl.1").exists())
            self.assertTrue(p.exists())

    def test_off_switch_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"ORDER_SHADOW_LOG": "0"}):
            p = Path(t) / "order_shadow.jsonl"
            shadow.set_path(p)
            shadow.new_chain("off")
            shadow.rec("input", what="x")
            shadow.flush()
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()


def stateful_exchange(state: dict) -> web.Application:
    """진짜 집행 함수(run_entry·run_exit·place_bracket·clear_bracket)를 받는 127.0.0.1 가짜 거래소 -- 지정가는 걸리자마자 다음 조회에 체결,
    포지션은 체결로 늘고 줄며, 우리 표식 주문 목록·취소를 받는다. 값은 전부 합성."""
    async def order(request: web.Request) -> web.Response:
        q = request.query
        assert request.headers.get("X-MBX-APIKEY") == KEY
        if request.method == "POST":
            state["n"] += 1
            o = {"orderId": state["n"], "clientOrderId": q.get("newClientOrderId", ""), "symbol": q["symbol"], "side": q["side"], "positionSide": q["positionSide"],
                 "type": q.get("type", "LIMIT"), "price": q.get("price", "0"), "origQty": q["quantity"], "executedQty": "0", "status": "NEW", "avgPrice": "0"}
            if o["type"] == "MARKET":
                o.update(status="FILLED", executedQty=o["origQty"], avgPrice="2500.00")
                fill(o)
            state["orders"][o["orderId"]] = o
            return web.json_response(o)
        o = next((x for x in state["orders"].values() if str(x["orderId"]) == q.get("orderId") or x["clientOrderId"] == q.get("origClientOrderId")), None)
        if o is None:
            return web.json_response({"code": -2013, "msg": "Order does not exist."}, status=400)
        if request.method == "GET" and o["status"] == "NEW" and not o["clientOrderId"].startswith("dbtp"):   # 진입·청산 따라가기 = 다음 조회에 체결 · 익절은 걸린 채
            o.update(status="FILLED", executedQty=o["origQty"], avgPrice=o["price"])
            fill(o)
        if request.method == "DELETE":
            if o["status"] == "NEW":
                o["status"] = "CANCELED"
            else:
                return web.json_response({"code": -2011, "msg": "Unknown order sent."}, status=400)
        return web.json_response(o)

    def fill(o):
        k = (o["symbol"], o["positionSide"])
        sign = 1 if (o["side"] == "BUY") == (o["positionSide"] == "LONG") else -1
        state["pos"][k] = round(state["pos"].get(k, 0.0) + sign * float(o["executedQty"]), 8)

    async def position_risk(request):
        out = []
        for (sym, ps), qty in state["pos"].items():
            if sym == request.query.get("symbol", sym):
                out.append({"symbol": sym, "positionSide": ps, "positionAmt": str(qty if ps == "LONG" else -qty), "entryPrice": "2500.0",
                            "markPrice": "2500.0", "liquidationPrice": "0", "leverage": "20", "notional": str(qty * 2500.0), "unRealizedProfit": "0", "updateTime": 1})
        return web.json_response(out)

    async def open_orders(request):
        return web.json_response([o for o in state["orders"].values() if o["status"] == "NEW" and o["symbol"] == request.query.get("symbol")])

    async def empty_list(_r):
        return web.json_response([])

    async def time_(_r):
        return web.json_response({"serverTime": int(time.time() * 1000)})

    async def book(_r):
        return web.json_response({"symbol": SYM, "bidPrice": "2500.00", "askPrice": "2500.01"})

    async def lev(_r):
        return web.json_response({"leverage": 20, "symbol": SYM})

    app = web.Application()
    app.router.add_route("*", "/fapi/v1/order", order)
    app.router.add_get("/fapi/v2/positionRisk", position_risk)
    app.router.add_get("/fapi/v1/openOrders", open_orders)
    app.router.add_get("/fapi/v1/openAlgoOrders", empty_list)
    app.router.add_get("/fapi/v1/time", time_)
    app.router.add_get("/fapi/v1/ticker/bookTicker", book)
    app.router.add_post("/fapi/v1/leverage", lev)

    async def fee(_r):
        return web.json_response({"symbol": SYM, "makerCommissionRate": "0.0002", "takerCommissionRate": "0.0005"})
    app.router.add_get("/fapi/v1/commissionRate", fee)
    return app


class RealExecRecordTest(unittest.TestCase):
    """진짜 집행 함수로 선 진입(체결 → 익절 걸고 무장) → 100% 청산. 남은 그림자 기록 = Rust 재생(rtorder shadow_replay)의 합성 실기록.
    ORDER_SHADOW_DUMP=<경로> 면 그 기록을 거기에 남긴다(시험 폴더는 끝에 지워진다)."""

    def test_real_entry_and_exit_leave_replayable_record(self) -> None:
        import threading
        st = {"n": 100, "orders": {}, "pos": {}}
        loop = asyncio.new_event_loop()
        ts = TestServer(stateful_exchange(st), host="127.0.0.1")
        loop.run_until_complete(ts.start_server())
        th = threading.Thread(target=loop.run_forever, daemon=True)
        th.start()
        base = str(ts.make_url("")).rstrip("/")
        assert base.startswith("http://127.0.0.1:")
        acct = json.loads(json.dumps(cl.FLAT))
        captured: list = []
        orig = shadow._write

        def spy(path, lines):
            captured.extend(lines)
            orig(path, lines)

        async def fn(c):
            r = await (await c.post("/api/manual-entry/submit?side=LONG&confirm=1&lev=20&pct=10&tp=2600&sl=2400")).json()
            self.assertTrue(r.get("ok"), r)
            for _ in range(300):
                await asyncio.sleep(0.05)
                if (server.LIVE_DIR / "manual_bracket_state.json").exists():
                    break
            await asyncio.sleep(0.3)
            qty = st["pos"].get((SYM, "LONG"), 0.0)
            acct["positions"] = [{**cl.POS["positions"][0], "qty": qty, "notional": qty * 2500.0}]
            r = await (await c.post("/api/manual-exit/submit?side=LONG&confirm=1&pct=100&asset=eth")).json()
            self.assertTrue(r.get("ok"), r)
            for _ in range(300):
                await asyncio.sleep(0.05)
                if st["pos"].get((SYM, "LONG"), 0.0) == 0.0:
                    break
            await asyncio.sleep(0.3)

        try:
            with mock.patch.object(ex, "FAPI", base), mock.patch.object(acct_mod, "FAPI", base), mock.patch.object(shadow, "_write", spy), \
                 mock.patch.dict(os.environ, {"BINANCE_API_KEY": KEY, "BINANCE_SECRET_KEY": SECRET, "ORDER_SHADOW_LOG": "1"}):
                cl.ChartLinesTest._run(self, fn, acct=acct, run_entry=ex.run_entry, run_exit=ex.run_exit, place_bracket=ex.place_bracket,
                                       clear_bracket=ex.clear_bracket, place_tp_ladder=ex.place_tp_ladder)
                shadow.flush()
        finally:
            asyncio.run_coroutine_threadsafe(ts.close(), loop).result(10)
            loop.call_soon_threadsafe(loop.stop)
            th.join(10)
            loop.close()
        rs = [json.loads(x) for x in captured]
        starts = [r for r in rs if r["rec"] == "start"]
        self.assertEqual([s["op"] for s in starts], ["entry", "exit"], [(r["rec"], r.get("op")) for r in rs])
        self.assertEqual(st["pos"].get((SYM, "LONG")), 0.0, st["pos"])
        for s in starts:
            mine = [r for r in rs if r["chain"] == s["chain"]]
            self.assertEqual(mine[-1]["rec"], "end")
            self.assertTrue(any(r["rec"] == "wire" and r["target"].startswith("/fapi/v1/order") for r in mine), mine)
        self.assertNotIn("signature=", "\n".join(captured))
        dump = os.environ.get("ORDER_SHADOW_DUMP")
        if dump:
            Path(dump).write_text("".join(x + "\n" for x in captured), "utf-8")
