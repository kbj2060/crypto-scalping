"""Rust 주문 전환 스위치(2026-11-22 · data/live/rtorder_live.on · rtorder/SHADOW.md §9).
🔴실주문 0: rtorder 문·거래소는 둘 다 127.0.0.1 가짜 aiohttp 앱 · 망 차단(unshare -rn) 안에서 돈다.
① 표식 없음: 14주소가 가짜 문으로 한 번도 안 가고, 스위치 미들웨어를 뗀 앱과 응답이 같다 · 감시 틱이 평소대로 거래소를 부른다
② 표식 있음: 14주소마다 가짜 문이 받은 메서드·경로·쿼리·본문·머리 = 보낸 것 · 돌아온 상태·본문 바이트·머리 = 가짜 문이 낸 것 · 거래소 0 · 무장·대기 파일 그대로 · 파이썬 접수 기록 0
③ 문이 닫혀 있으면 502 rtorder_unavailable · 루프백 아닌 RTORDER_DOOR 거부
④ 실행 중 표식을 만들고 지우면 다음 틱부터 바뀐다(감시·사용자 스트림·계좌 갱신기·시작 정리)
⑤ SSE acct_seq 가 가짜 문 /internal/acct 의 seq 를 따른다
python -m pytest -q test/test_rtorder_switch_20261122.py"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiohttp import ClientSession, web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from offline_app import offline_app  # noqa: E402

from dashboard import orders, server  # noqa: E402
from live.execution import live_manual_peg_execute_20260912 as ex  # noqa: E402
from live.execution import order_shadow as shadow  # noqa: E402
from scripts import live_binance_account_20260910 as acct_mod  # noqa: E402
import test_chart_lines_entry_20261028 as cl  # noqa: E402
from test_chart_lines_entry_20261028 import SYM  # noqa: E402
from test_manual_entry_preview_smoke_20260913 import _isolated_dirs  # noqa: E402
from test_order_defect_fixes_20261121 import ETH_TOUCHED, POS as DPOS  # noqa: E402

# (메서드, 주소, 쿼리 원문, 본문) -- rtorder/src/route.rs ROUTES 순서 · 쿼리는 + 와 %XX 와 겹친 키를 섞는다(풀지 않고 넘겨야 한다)
CASES = [
    ("GET", "/api/binance-account", "fresh=1&x=a+b%20c", b""),
    ("GET", "/api/manual-entry/preview", "side=LONG&lev=20&pct=10&tp=2550&sl=2480", b""),
    ("POST", "/api/manual-entry/submit", "side=SHORT&lev=20&pct=10&tp=2400&sl=2600&side=LONG", b'{"a": 1}'),
    ("GET", "/api/manual-entry/status", "asset=eth", b""),
    ("POST", "/api/manual-entry/wait-action", "action=cancel&asset=eth&wid=2", b""),
    ("POST", "/api/manual-entry/cancel", "", b""),
    ("POST", "/api/manual-entry/limit-amend", "asset=eth&price=2490.5", b""),
    ("POST", "/api/manual-bracket/disarm", "symbol=ETHUSDC&side=LONG", b""),
    ("POST", "/api/manual-bracket/rearm", "symbol=ETHUSDC&side=LONG", b""),
    ("POST", "/api/manual-bracket/tp", "symbol=ETHUSDC&side=LONG&op=remove", b""),
    ("POST", "/api/manual-bracket/tp-legs", "symbol=ETHUSDC&side=LONG&legs=%5B%5D", b""),
    ("POST", "/api/manual-bracket/add", "symbol=ETHUSDC&side=LONG&kind=sl&price=2400", b""),
    ("POST", "/api/manual-bracket/modify", "symbol=ETHUSDC&side=LONG&kind=sl&price=2410", b"x=\xed\x95\x9c"),
    ("POST", "/api/manual-exit/submit", "side=LONG&pct=50&asset=eth", b""),
]
assert [c[1] for c in CASES] == sorted(orders.RTORDER_PATHS, key=[c[1] for c in CASES].index) and len(CASES) == len(orders.RTORDER_PATHS) == 14
STATUSES = [200, 409, 200, 200, 400, 403, 409, 200, 404, 200, 500, 200, 409, 200]


def fake_door(state: dict) -> web.Application:
    """가짜 rtorder 문 -- 받은 것을 적고, 주소마다 다른 상태·본문(한글 바이트 포함)·머리로 답한다."""
    async def any_(request: web.Request) -> web.Response:
        if request.path == "/internal/acct":
            state["acct_reads"] += 1
            return web.json_response({"seq": state["seq"], "user_stream": {"connected": True}, "feed": {}})
        i = [c[1] for c in CASES].index(request.path)
        state["got"].append({"method": request.method, "path": request.path, "qs": request.rel_url.raw_query_string, "body": await request.read(),
                             "ua": request.headers.get("User-Agent"), "cf": request.headers.get("CF-Connecting-IP"),
                             "xff": request.headers.get("X-Forwarded-For"), "ct": request.headers.get("Content-Type")})
        body = json.dumps({"door": request.path, "i": i, "말": "문"}, ensure_ascii=False).encode()
        h = {"Content-Type": "application/json; charset=utf-8"}
        if STATUSES[i] == 200:
            h["Cache-Control"] = "no-cache"
        if i == 5:
            h["Allow"] = "POST"
        return web.Response(status=STATUSES[i], body=body, headers=h)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", any_)
    return app


def counting_exchange(hits: list) -> web.Application:
    async def any_(request: web.Request) -> web.Response:
        hits.append((request.method, request.path))
        return web.json_response({"serverTime": int(time.time() * 1000), "orderId": 1, "status": "NEW"})
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", any_)
    return app


async def _start(app: web.Application) -> TestServer:
    ts = TestServer(app, host="127.0.0.1")
    await ts.start_server()
    return ts


def _door_addr(ts: TestServer) -> str:
    return f"127.0.0.1:{ts.port}"


def _strip(app: web.Application) -> web.Application:
    """스위치 미들웨어를 뗀 앱(= 이 변경 전과 같은 미들웨어 줄) -- 표식 없음 응답의 기준."""
    app.middlewares[:] = [m for m in app.middlewares if getattr(m, "__name__", "") != "rtorder_door"]
    return app


class RouteSwitchTest(unittest.TestCase):
    def _routes(self, fn) -> None:
        async def fake_account(*_a, **_k):
            return cl.FLAT

        async def exercise() -> None:
            door_state = {"got": [], "seq": 0, "acct_reads": 0}
            hits: list = []
            door, exch = await _start(fake_door(door_state)), await _start(counting_exchange(hits))
            base = f"http://127.0.0.1:{exch.port}"
            boom = {k: cl._boom for k in ("place_bracket", "run_entry", "run_exit", "clear_bracket", "place_tp_ladder", "cancel_wait_orphans")}
            try:
                with mock.patch.object(orders, "fetch_account", fake_account), mock.patch.object(orders, "SIZING_RISK_MODEL_ENABLED", False), \
                     mock.patch.object(orders, "exec_enabled", lambda: True), mock.patch.multiple(orders, **boom), \
                     mock.patch.object(ex, "FAPI", base), mock.patch.object(acct_mod, "FAPI", base), \
                     mock.patch.object(orders, "RTORDER_DOOR", _door_addr(door)):
                    apps = []
                    for strip in (False, True):
                        app = offline_app(server)
                        app["situation_eth"]["vol_mult"] = {"bar": time.time(), **cl.VM}
                        c = TestClient(TestServer(_strip(app) if strip else app))
                        await c.start_server()
                        apps.append(c)
                    try:
                        await fn(apps[0], apps[1], door_state, hits, door)
                    finally:
                        for c in apps:
                            await c.close()
            finally:
                await door.close()
                await exch.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    @staticmethod
    async def _send(c: TestClient, method: str, path: str, qs: str, body: bytes, headers: dict | None = None):
        r = await c.request(method, path + (f"?{qs}" if qs else ""), data=body or None, headers=headers or {}, skip_auto_headers=("User-Agent", "Content-Type"))   # 화면이 UA·Content-Type 을 안 보낸 경우도 보려고
        return r.status, await r.read(), r.headers

    def test_off_never_reaches_door_and_same_as_without_switch(self) -> None:
        async def fn(c, ref, door_state, hits, _door):
            self.assertFalse(orders.rtorder_on())
            log = server.LIVE_DIR / "order_requests.jsonl"
            for m, p, qs, body in CASES:
                a = await self._send(c, m, p, qs, body)
                b = await self._send(ref, m, p, qs, body)
                self.assertEqual((a[0], a[1]), (b[0], b[1]), (m, p))
                self.assertNotIn(b'"door"', a[1])
            self.assertEqual(door_state["got"], [], "표식이 없으면 문으로 하나도 안 간다")
            n_post = sum(1 for c_ in CASES if c_[0] == "POST")
            self.assertEqual(len(log.read_text("utf-8").splitlines()), 2 * n_post, "표식 없음 = 파이썬 접수 기록 그대로(두 앱 몫)")
        self._routes(fn)

    def test_on_forwards_bytes_and_touches_nothing(self) -> None:
        async def fn(c, _ref, door_state, hits, _door):
            live = orders.LIVE_DIR
            (live / "manual_bracket_state.json").write_text(json.dumps({f"{SYM}:LONG": ETH_TOUCHED}), "utf-8")
            (live / "manual_limit_wait_ETHUSDC.json").write_text('{"symbol": "ETHUSDC"}', "utf-8")
            shadow.flush()   # 앞 시험이 큐에 남긴 그림자 줄을 먼저 비운다(뒤 스레드가 지금 경로에 쓴다)
            before = {p.name: p.read_bytes() for p in live.iterdir()}
            (live / "rtorder_live.on").write_text("live\n", "utf-8")
            for i, (m, p, qs, body) in enumerate(CASES):
                hdr = {"User-Agent": f"ua-{i}", "CF-Connecting-IP": "203.0.113.9"} if i % 2 else {}
                if body:
                    hdr["Content-Type"] = "application/x-test"
                st, got, h = await self._send(c, m, p, qs, body, hdr)
                sent = door_state["got"][-1]
                self.assertEqual((sent["method"], sent["path"], sent["qs"], sent["body"]), (m, p, qs, body), p)
                self.assertEqual(sent["ua"], hdr.get("User-Agent"), "UA 는 받은 그대로(없으면 없음 -- aiohttp 기본 UA 를 넣지 않는다)")
                self.assertEqual(sent["cf"], hdr.get("CF-Connecting-IP"))
                self.assertEqual(sent["xff"], "127.0.0.1", "X-Forwarded-For 가 없으면 접속 주소")
                self.assertEqual(sent["ct"], hdr.get("Content-Type"))
                want = json.dumps({"door": p, "i": i, "말": "문"}, ensure_ascii=False).encode()
                self.assertEqual((st, got), (STATUSES[i], want), p)
                self.assertTrue(h["Content-Type"].startswith("application/json"), h)
                self.assertEqual(h.get("Cache-Control"), "no-cache" if STATUSES[i] == 200 else None, (p, dict(h)))
                self.assertEqual(h.get("Allow"), "POST" if i == 5 else None)
            self.assertEqual(len(door_state["got"]), 14)
            # 화면이 X-Forwarded-For 를 보냈으면 그대로
            await self._send(c, "GET", "/api/manual-entry/status", "", b"", {"X-Forwarded-For": "198.51.100.7"})
            self.assertEqual(door_state["got"][-1]["xff"], "198.51.100.7")
            await asyncio.sleep(0.2)
            shadow.flush()
            after = {p.name: p.read_bytes() for p in live.iterdir() if p.name != "rtorder_live.on"}
            self.assertEqual(after, before, "무장·대기·접수 기록 파일 그대로(파이썬은 아무것도 안 씀)")
            self.assertEqual(hits, [], "가짜 거래소가 받은 요청 0")
        self._routes(fn)

    def test_door_closed_or_not_loopback_is_502(self) -> None:
        async def fn(c, _ref, door_state, _hits, door):
            (orders.LIVE_DIR / "rtorder_live.on").write_text("live", "utf-8")
            await door.close()   # 문이 닫혔다(죽음·재시작 중)
            for m, p, qs, body in CASES:
                st, got, _h = await self._send(c, m, p, qs, body)
                self.assertEqual((st, json.loads(got)), (502, {"ok": False, "error": "rtorder_unavailable"}), p)
            with mock.patch.object(orders, "RTORDER_DOOR", None):
                st, got, _h = await self._send(c, "GET", "/api/binance-account", "", b"")
                self.assertEqual((st, json.loads(got)["error"]), (502, "rtorder_unavailable"))
            (orders.LIVE_DIR / "rtorder_live.on").unlink()   # 지우면 바로 파이썬이 다시 답한다
            st, got, _h = await self._send(c, "GET", "/api/manual-entry/status", "asset=eth", b"")
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(got)["ok"])
        self._routes(fn)

    def test_switching_blocks_posts_passes_gets(self) -> None:
        """표식 내용 switching·빈 파일·모르는 글자 = POST 11개 503 rtorder_switching(문에 0번) · GET 3개는 넘어감 · live 로 바꾸면(임시 파일 → 바꿔치기) POST 도 넘어감."""
        async def fn(c, _ref, door_state, hits, _door):
            mark = orders.LIVE_DIR / "rtorder_live.on"
            posts = [x for x in CASES if x[0] == "POST"]
            gets = [x for x in CASES if x[0] == "GET"]
            self.assertEqual((len(posts), len(gets)), (11, 3))
            for content in ("switching", "", "LIVE?", "live-ish", "\xff"):
                mark.write_bytes(content.encode("latin-1") if content == "\xff" else content.encode())
                n0 = len(door_state["got"])
                for m, p, qs, body in posts:
                    st, got, _h = await self._send(c, m, p, qs, body)
                    self.assertEqual((st, json.loads(got)), (503, {"ok": False, "error": "rtorder_switching"}), (content, p))
                self.assertEqual(len(door_state["got"]), n0, f"전환 중 POST 는 문에 0번 ({content!r})")
                for m, p, qs, body in gets:
                    st, got, _h = await self._send(c, m, p, qs, body)
                    self.assertEqual(door_state["got"][-1]["path"], p, (content, p))
                self.assertEqual(len(door_state["got"]), n0 + 3)
            tmp = mark.with_suffix(".tmp")
            tmp.write_text("live\n", "utf-8")
            tmp.replace(mark)   # 원자적으로 live
            for m, p, qs, body in posts:
                await self._send(c, m, p, qs, body)
                self.assertEqual(door_state["got"][-1]["path"], p)
            tmp.write_text("switching\n", "utf-8")
            tmp.replace(mark)   # 되돌리기 시작 = 다시 switching
            st, _got, _h = await self._send(c, *posts[0][:2], posts[0][2], posts[0][3])
            self.assertEqual(st, 503)
            self.assertEqual(hits, [])
        self._routes(fn)

    def test_loopback_only(self) -> None:
        self.assertEqual(orders._loopback_door("127.0.0.1:8791"), "127.0.0.1:8791")
        self.assertEqual(orders._loopback_door("[::1]:8791"), "[::1]:8791")
        for bad in ("10.0.0.1:8791", "0.0.0.0:8791", "localhost:8791", "127.0.0.1", "127.0.0.1:0", "fapi.binance.com:443"):
            self.assertIsNone(orders._loopback_door(bad), bad)


class LoopSwitchTest(unittest.TestCase):
    """감시 틱·사용자 스트림·계좌 갱신기·시작 정리가 표식을 회차마다 본다."""

    def _api(self, tmp: Path, *, session, calls: dict, sec=None, fetch=None):
        async def swr(_k, _ttl, produce, **_kw):
            calls["swr"] += 1
            return await produce()

        async def fake_account(*_a, **_k):
            return DPOS

        async def hist(_a):
            return [{"time": 1000, "close": 2490.0}]

        async def timed(_name, produce):
            calls["refresh"] += 1
            return await produce()

        return orders.register(account_trip_state={"seen": {}}, refresh_tasks=calls["tasks"], swr_locks={}, binance_session=session,
                               fetch_binance_json=fetch, swr_cached=swr, _timed_produce=timed, load_market_history=hist,
                               situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=sec or (lambda _a: [(2001, 2490.0)])), fake_account

    def test_watch_tick_toggles_and_on_sends_nothing(self) -> None:
        async def go(tmp: Path):
            hits: list = []
            exch = await _start(counting_exchange(hits))
            base = f"http://127.0.0.1:{exch.port}"
            calls = {"swr": 0, "refresh": 0, "tasks": {}, "clear": 0, "exit": 0}

            async def clear(_s, symbol, pside, *a, **k):
                async with ClientSession() as s:
                    await ex.signed(s, "DELETE", "/fapi/v1/order", {"symbol": symbol}, "k", "s", 0)
                calls["clear"] += 1
                return {"cancelled": 1, "errors": []}

            async def exit_(_s, plan, state):
                calls["exit"] += 1
                state.update(phase="filled_maker", filled=1.0, quantity=1.0)
                return state

            async def no_exits(*_a, **_k):
                return {"cancelled": 0, "errors": []}

            async def filters(_s, _symbol):
                return {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 5.0}

            from test_manual_entry_preview_smoke_20260913 import FAKE_STATE, _fake_binance

            async def fetch(url, params=None, **kw):
                r = await _fake_binance(url, params or {}, **kw)
                return {} if r is None else r

            bfile = tmp / "manual_bracket_state.json"
            try:
                with mock.patch.object(ex, "FAPI", base), mock.patch.object(orders, "LIVE_DIR", tmp), mock.patch.object(orders, "exec_enabled", lambda: True), \
                     mock.patch.multiple(orders, clear_bracket=clear, run_exit=exit_, run_entry=cl._boom, place_bracket=cl._boom, cancel_exit_orders=no_exits,
                                         load_filters=filters, POSITION_SIZING_STATE_PATH=tmp / "eth_position_sizing_state.json"):
                    (tmp / "eth_position_sizing_state.json").write_text(json.dumps(FAKE_STATE), "utf-8")
                    api, fake_account = self._api(tmp, session=lambda: None, calls=calls,
                                                  sec=lambda a: [(2001, 2490.0), (2002, 2479.0)] if a == "eth" else None, fetch=fetch)
                    with mock.patch.object(orders, "fetch_account", fake_account):
                        app: dict = {}
                        # ① 표식 있음: 닿은 손절선(발동할 칸)이 있어도 아무것도 안 한다
                        (tmp / "rtorder_live.on").touch()
                        bfile.write_text(json.dumps({f"{SYM}:LONG": dict(ETH_TOUCHED)}), "utf-8")
                        (tmp / "manual_limit_wait_ETHUSDC.json").write_text('{"symbol": "ETHUSDC"}', "utf-8")
                        shadow.flush()
                        before = snap(tmp)
                        await api.start_bracket_watcher(app)
                        await api.start_wait_cleanup(app)
                        for _ in range(5):
                            ex.wake_fills()
                            await asyncio.sleep(0.05)
                        shadow.flush()
                        on = dict(calls), list(hits), snap(tmp)
                        self.assertNotIn("wait_cleanup", calls["tasks"], "표식 있음 = 시작 정리·대기 복구 안 띄움")
                        # ② 표식 지움 → 다음 틱부터 평소대로(발동: 청산 + 정리 요청이 가짜 거래소로) · 첫 틱은 계좌 캐시를 버린다
                        cache = find_cell(api.api_binance_account, "binance_account_cache")
                        cache["ts"] = time.monotonic()
                        (tmp / "rtorder_live.on").unlink()
                        for _ in range(40):
                            ex.wake_fills()
                            await asyncio.sleep(0.05)
                            if calls["clear"]:
                                break
                        off = dict(calls), list(hits)
                        self.assertEqual(cache["ts"], 0.0, "되돌린 첫 틱 = 계좌 새로 읽기(캐시 시각 0)")
                        # ③ 다시 표식 → 조용한 무장(안 닿는 손절선)이 있어도 틱이 계좌를 안 읽는다
                        bfile.write_text(json.dumps({f"{SYM}:LONG": {**dict(ETH_TOUCHED), "sl_level": 2000.0, "sl_price": 2000.0}}), "utf-8")
                        ex.wake_fills()
                        await asyncio.sleep(0.1)
                        (tmp / "rtorder_live.on").touch()
                        await asyncio.sleep(0.05)
                        n0 = calls["swr"]
                        for _ in range(5):
                            ex.wake_fills()
                            await asyncio.sleep(0.05)
                        again = calls["swr"] - n0
                        (tmp / "rtorder_live.on").unlink()
                        for _ in range(5):
                            ex.wake_fills()
                            await asyncio.sleep(0.05)
                        back = calls["swr"] - n0 - again
                        await api.stop_bracket_watcher(app)
                        await api.start_wait_cleanup(app)   # 표식 없음 = 지금처럼 띄운다
                        self.assertIn("wait_cleanup", calls["tasks"])
                        calls["tasks"]["wait_cleanup"].cancel()
                        return on, off, again, back, before
            finally:
                await exch.close()

        with tempfile.TemporaryDirectory() as t:
            on, off, again, back, before_ = asyncio.run(go(Path(t)))
        (c_on, h_on, same) = on
        self.assertEqual((c_on["swr"], c_on["clear"], c_on["exit"], h_on, same), (0, 0, 0, [], before_), "표식 있음: 계좌·청산·거래소 요청 0 · 파일 그대로")
        c_off, h_off = off
        self.assertGreaterEqual(c_off["swr"], 1)
        self.assertEqual(c_off["exit"], 1, "표식 지운 다음 틱 = 평소처럼 발동")
        self.assertGreaterEqual(c_off["clear"], 1)
        self.assertIn(("DELETE", "/fapi/v1/order"), h_off, "감시 틱이 평소대로 거래소를 부른다")
        self.assertEqual(again, 0, "다시 표식 = 다음 틱부터 계좌도 안 읽음")
        self.assertGreaterEqual(back, 1, "지우면 다시 돈다")

    def test_user_stream_and_refresher_skip_when_on(self) -> None:
        async def go(tmp: Path):
            used: list = []

            class Spy:   # listenKey 가 나가면 여기에 남는다(실제로는 보내지 않고 실패시킨다)
                def request(self, method, url, **_k):
                    used.append((method, url))
                    raise RuntimeError("시험: 보내지 않음")

            calls = {"swr": 0, "refresh": 0, "tasks": {}}
            with mock.patch.object(orders, "LIVE_DIR", tmp), mock.patch.object(orders, "exec_enabled", lambda: True), \
                 mock.patch.dict("os.environ", {"BINANCE_API_KEY": "test_key_0000"}), \
                 mock.patch.object(orders.ban_guard, "ban_remaining", lambda: 0):
                api, fake_account = self._api(tmp, session=lambda: Spy(), calls=calls)
                with mock.patch.object(orders, "fetch_account", fake_account):
                    dirty = find_cell(api.start_user_stream, "acct_dirty")
                    (tmp / "rtorder_live.on").touch()
                    await api.start_user_stream({})
                    await asyncio.sleep(0.3)
                    dirty.set()
                    await asyncio.sleep(0.6)
                    on = ([u for u in used if "listenKey" in u[1]], calls["refresh"], api.acct_seq())
                    (tmp / "rtorder_live.on").unlink()
                    await asyncio.sleep(1.3)   # 스트림 고리는 1초마다 표식을 본다
                    dirty.set()
                    await asyncio.sleep(0.6)
                    off = ([u for u in used if "listenKey" in u[1]], calls["refresh"], api.acct_seq())
                    for t in calls["tasks"].values():
                        t.cancel()
                    await asyncio.gather(*calls["tasks"].values(), return_exceptions=True)
                    return on, off

        with tempfile.TemporaryDirectory() as t:
            on, off = asyncio.run(go(Path(t)))
        self.assertEqual(on, ([], 0, 0), "표식 있음: listenKey 0 · 계좌 갱신 0")
        self.assertTrue(off[0] and off[0][0][0] == "POST" and "listenKey" in off[0][0][1], off)
        self.assertEqual((off[1], off[2]), (1, 1), "표식 지움: 계좌 갱신 · seq 오름(지금 동작)")


class AcctSeqTest(unittest.TestCase):
    def test_sse_acct_seq_follows_door(self) -> None:
        async def go(tmp: Path):
            st = {"got": [], "seq": 41, "acct_reads": 0}
            door = await _start(fake_door(st))
            out = []
            async with ClientSession() as s:
                with mock.patch.object(orders, "LIVE_DIR", tmp), mock.patch.object(orders, "RTORDER_DOOR", _door_addr(door)):
                    api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: s,
                                          fetch_binance_json=None, swr_cached=None, _timed_produce=None, load_market_history=None,
                                          situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60)
                    us = find_cell(api.start_user_stream, "user_stream")
                    us["seq"] = 7
                    out.append(("off", api.acct_seq(), st["acct_reads"]))
                    (tmp / "rtorder_live.on").touch()
                    out.append(("on-first", api.acct_seq()))   # 아직 못 읽음 = 마지막 값(파이썬 것)
                    await asyncio.sleep(0.2)
                    out.append(("on", api.acct_seq()))
                    st["seq"] = 42
                    await asyncio.sleep(0.2)
                    api.acct_seq()
                    await asyncio.sleep(0.2)
                    out.append(("on-next", api.acct_seq()))
                    await door.close()
                    api.acct_seq()
                    await asyncio.sleep(0.3)
                    out.append(("closed", api.acct_seq()))   # 못 읽으면 마지막 값
                    (tmp / "rtorder_live.on").unlink()
                    out.append(("back", api.acct_seq()))
            return out

        with tempfile.TemporaryDirectory() as t:
            out = asyncio.run(go(Path(t)))
        self.assertEqual(out, [("off", 7, 0), ("on-first", 7), ("on", 41), ("on-next", 42), ("closed", 42), ("back", 7)])


def snap(d: Path) -> dict:
    """폴더 파일 바이트 -- 그림자 기록의 «깨움» 줄(시험이 체결 알림 대신 wake_fills 를 부른 것)은 뺀다."""
    out = {}
    for p in d.iterdir():
        b = p.read_bytes()
        if p.name == "order_shadow.jsonl":
            b = b"".join(x for x in b.splitlines(keepends=True) if b'"what": "wake"' not in x)
            if not b:
                continue
        out[p.name] = b
    return out


def find_cell(fn, name, seen=None):
    """register 클로저를 따라가며 이름이 name 인 값."""
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


if __name__ == "__main__":
    unittest.main()
