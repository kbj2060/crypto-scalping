"""rtorder 넷째 조각 정답지(2026-11-21) -- 원본 파이썬을 가짜 의존성·가짜 시계로 돌려 적는다:
  account  : scripts/live_binance_account_20260910.fetch_account(가짜 세션 -- 나간 요청 열 · 스냅숏)
  stream   : orders._user_stream_run · account_refresher(가짜 웹소켓·listenKey·잠 -- 사건 열 · user_stream 칸 · 로그)
  entry/exit: orders.assemble_entry_plan · assemble_exit_plan(가짜 계좌·호가·필터·변동성·펀딩·수수료·마진 -- 계획 dict 전체 · 나간 조회 열)
  api      : orders 의 주문 API 핸들러(가짜 요청 -- 상태코드 · 본문 · 시작된 작업 · 상태 칸 · 무장 파일 · 로그)
Rust `cargo test --test desk_fixture` 가 같은 입력으로 대조한다.

안전: 실주문·네트워크 없음(모든 바깥은 가짜 · 각본이 비면 예외) · .env 안 읽음 · 키는 명백한 가짜 문자열 · data/live 대신 임시 폴더 ·
`unshare -rn`(네트워크 없는 이름공간). 실행(저장소 뿌리):
  ( ulimit -v 4000000; TZ=UTC timeout 300 unshare -rn python3 -I rtorder/tools/gen_desk_fixture.py )  → rtorder/tests/fixtures/desk_fixture.json
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qsl, urlsplit

ROOT = Path(__file__).resolve().parents[2]
# 2026-11-21 원본 파이썬 위치 -- 기본 = 이 저장소(main 판 · PR 61 머지 뒤) · RTORDER_PY_ROOT = 다른 판을 볼 때만(읽기만 · 정답지는 이 저장소에 쓴다)
PY_ROOT = Path(os.environ.get("RTORDER_PY_ROOT") or ROOT)
sys.path.insert(0, str(PY_ROOT))
# 감시가 망가진 칸을 알릴 때 텔레그램 모듈을 부른다(진짜는 서버 비밀 설정을 읽고 보낸다) -- 가짜를 먼저 넣는다
_wd = types.ModuleType("scripts.ops_watchdog")
_wd.send_telegram = lambda *_a, **_k: None
sys.modules["scripts.ops_watchdog"] = _wd
for _k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "DASHBOARD_MANUAL_EXEC_ENABLED", "DASHBOARD_SIZING_CAP_EQUITY_X"):
    os.environ.pop(_k, None)
os.environ["DASHBOARD_MANUAL_EXEC_SYMBOL"] = "ETHUSDC"

from aiohttp import WSMsgType  # noqa: E402
from dashboard import orders  # noqa: E402
import scripts.live_binance_account_20260910 as acct  # noqa: E402
import scripts.binance_ban_guard as ban_guard  # noqa: E402

# 스위치가 지금 값일 때만 이 정답지가 뜻이 있다(Rust 는 같은 값을 상수로 둔다 -- assemble.rs)
assert orders.SIZING_RISK_MODEL_ENABLED is False and orders.SIZING_CAP_MODEL_ONLY is True and orders.SIZING_MARGIN_CAP_PCT == 0.0
assert orders.MANUAL_EXEC_SYMBOL == "ETHUSDC" and not hasattr(orders, "REC_AUTO_ADD")   # 2026-11-21 PR 61 판(자동 규칙·물타기 지움)

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "desk_fixture.json"
T0 = 1_800_000_000.0


class Clock:
    t = T0


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(Clock.t, tz)


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def snap(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


def exc_of(spec):
    return type(spec[0], (Exception,), {})(spec[1])


FAKE_TIME = types.SimpleNamespace(time=lambda: Clock.t, monotonic=lambda: Clock.t, perf_counter=lambda: Clock.t)


# ═══ A. 계좌 조회 ═══════════════════════════════════════════════════════════════
class Resp:
    def __init__(self, status, body):
        self.status, self.body, self.headers = status, body, {}

    async def json(self):
        if isinstance(self.body, dict) and "__raise__" in self.body:
            raise exc_of(self.body["__raise__"])
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None):
        u = urlsplit(url)
        params = {k: v for k, v in parse_qsl(u.query) if k not in ("recvWindow", "timestamp", "signature")}
        self.calls.append([u.path, params])
        key = u.path + ("?" + params["symbol"] if "symbol" in params and u.path in ("/fapi/v1/userTrades", "/fapi/v1/openOrders") else "")
        r = self.routes.get(key, self.routes.get(u.path))
        if r is None:
            raise RuntimeError(f"각본 없음: {key}")
        if isinstance(r, dict) and "__raise__" in r and u.path == "/fapi/v1/time":
            raise exc_of(r["__raise__"])
        status = r.get("__status__", 200) if isinstance(r, dict) else 200
        body = r.get("__body__", r) if isinstance(r, dict) and "__status__" in r else r
        return Resp(status, body)


def run_account(c):
    sess = FakeSession(c["routes"])
    env = {"BINANCE_API_KEY": "fake-key", "BINANCE_SECRET_KEY": "fake-secret"} if c.get("keys", True) else {}
    Clock.t = c.get("t", T0)
    with mock.patch.dict(os.environ, env), mock.patch.object(acct, "datetime", FakeDT), \
            mock.patch.object(ban_guard, "ban_remaining", lambda now=None: c.get("ban", 0.0)), mock.patch.object(ban_guard, "note", lambda *a, **k: 0.0):
        raised = None
        try:
            out = asyncio.run(acct.fetch_account(sess, c["symbols"], quote_asset=c.get("quote", "USDC")))
        except Exception as e:  # noqa: BLE001
            out, raised = None, f"{type(e).__name__}: {e}"
    return {"name": c["name"], "in": c, "calls": sess.calls, "out": snap(out), "raised": raised}


def asset(a, wallet, avail, margin, im=0.0, mm=0.0, up=0.0):
    return {"asset": a, "walletBalance": str(wallet), "availableBalance": str(avail), "marginBalance": str(margin), "initialMargin": str(im),
            "maintMargin": str(mm), "unrealizedProfit": str(up)}


BAL = {"multiAssetsMargin": False, "totalWalletBalance": "111.1", "totalMarginBalance": "112.2", "totalInitialMargin": "3.3", "totalMaintMargin": "1.1",
       "availableBalance": "99.9", "totalUnrealizedProfit": "1.1",
       "assets": [asset("USDC", 1234.5, 1000.25, 1240.75, 200.5, 20.25, 6.25), asset("USDT", 111.1, 99.9, 112.2, 3.3, 1.1, 1.1), asset("BNB", 0, 0, 0)]}
RISK = [{"symbol": "ETHUSDC", "positionSide": "LONG", "positionAmt": "0.500", "entryPrice": "2400.5", "markPrice": "2450.25", "liquidationPrice": "1900.1",
         "leverage": "10", "notional": "1225.125", "unRealizedProfit": "24.875", "updateTime": 1799990000123},
        {"symbol": "ETHUSDC", "positionSide": "SHORT", "positionAmt": "0", "entryPrice": "0", "markPrice": "2450.25", "liquidationPrice": "0",
         "leverage": "10", "notional": "0", "unRealizedProfit": "0", "updateTime": 0},
        {"symbol": "SOXLUSDT", "positionSide": "SHORT", "positionAmt": "-3", "entryPrice": "31.5", "markPrice": "30.25", "liquidationPrice": "60",
         "leverage": "5", "notional": "-90.75", "unRealizedProfit": "3.75", "updateTime": 1799990500000},
        {"symbol": "XRPUSDC", "positionSide": "BOTH", "positionAmt": "0", "entryPrice": "0", "markPrice": "0.5", "liquidationPrice": "0",
         "leverage": "", "notional": "0", "unRealizedProfit": "0", "updateTime": 0}]


def fill(sym, i, t, side, px, q, rpnl="0", fee="0.01", ps="LONG", maker=True):
    return {"symbol": sym, "id": i, "time": t, "side": side, "price": str(px), "qty": str(q), "realizedPnl": str(rpnl), "commission": str(fee),
            "positionSide": ps, "maker": maker}


ETH_FILLS = [fill("ETHUSDC", 1, 1799000000000, "BUY", 2300, 1), fill("ETHUSDC", 2, 1799000100000, "BUY", 2310, 0.5),
             fill("ETHUSDC", 3, 1799000400000, "BUY", 2290, 0.25), fill("ETHUSDC", 4, 1799001000000, "SELL", 2350, 1.75, rpnl="88.75"),
             fill("ETHUSDC", 5, 1799990000000, "BUY", 2400.5, 0.3), fill("ETHUSDC", 6, 1799990000500, "BUY", 2400.5, 0.2, maker=False)]
SOXL_FILLS = [fill("SOXLUSDT", 7, 1799990400000, "SELL", 31.5, 3, ps="SHORT")]
ORDERS_ETH = [{"symbol": "ETHUSDC", "side": "SELL", "type": "LIMIT", "price": "2600", "origQty": "0.5", "executedQty": "0.1", "reduceOnly": False, "clientOrderId": "dbtpL123"},
              {"symbol": "ETHUSDC", "side": "SELL", "type": "STOP_MARKET", "price": "0", "origQty": "0", "executedQty": "0", "reduceOnly": True, "clientOrderId": "x"}]


def acct_routes(**over):
    r = {"/fapi/v1/time": {"serverTime": 1800000000500}, "/fapi/v2/account": BAL, "/fapi/v2/positionRisk": RISK,
         "/fapi/v1/userTrades?ETHUSDC": ETH_FILLS, "/fapi/v1/userTrades?SOXLUSDT": SOXL_FILLS, "/fapi/v1/userTrades": [],
         "/fapi/v1/openOrders?ETHUSDC": ORDERS_ETH, "/fapi/v1/openOrders": []}
    r.update(over)
    return r


SYMS = ["ETHUSDT", "SOLUSDT", "ETHUSDC", "SOXLUSDT"]
ACCOUNT_CASES = [
    {"name": "normal_usdc", "symbols": SYMS, "routes": acct_routes()},
    {"name": "quote_usdt", "symbols": SYMS, "quote": "USDT", "routes": acct_routes()},
    {"name": "multi_assets", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/account": {**BAL, "multiAssetsMargin": True}})},
    {"name": "quote_missing_asset", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/account": {**BAL, "assets": [asset("USDT", 5, 4, 5)]}})},
    {"name": "no_keys", "symbols": SYMS, "keys": False, "routes": acct_routes()},
    {"name": "account_perm", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/account": {"__status__": 401, "__body__": {"code": -2015, "msg": "Invalid API-key, IP, or permissions for action"}}})},
    {"name": "account_clock", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/account": {"__status__": 400, "__body__": {"code": -1021, "msg": "Timestamp for this request is outside of the recvWindow."}}})},
    {"name": "account_5xx", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/account": {"__status__": 503, "__body__": {"msg": "busy"}}})},
    {"name": "risk_ban_418", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/positionRisk": {"__status__": 418, "__body__": {"code": -1003, "msg": "banned until 1800000100000"}}})},
    {"name": "trades_fail_one", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v1/userTrades?SOXLUSDT": {"__status__": 429, "__body__": {"code": -1003, "msg": "too many"}}})},
    {"name": "orders_fail", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v1/openOrders?SOXLUSDT": {"__status__": 500, "__body__": {"msg": "x"}}})},
    {"name": "ban_precheck", "symbols": SYMS, "ban": 42.4, "routes": acct_routes()},
    {"name": "clock_raises", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v1/time": {"__raise__": ["ClientConnectorError", "down"]}})},
    {"name": "keyerror_position", "symbols": SYMS, "routes": acct_routes(**{"/fapi/v2/positionRisk": [{"symbol": "ETHUSDC", "positionAmt": "1"}]})},
    # 단방향(BOTH) 뒤집기 · 헤지 절단(창이 포지션 가운데서 시작) + 재동기 · 재동기 못 함 · 유령 잔여
    {"name": "fold_flip_both", "symbols": ["ETHUSDT"], "routes": acct_routes(**{"/fapi/v1/userTrades?ETHUSDT": [
        fill("ETHUSDT", 1, 1000, "SELL", 2000, 2, ps="BOTH"), fill("ETHUSDT", 2, 2000, "BUY", 1990, 5, rpnl="20", fee="0.5", ps="BOTH"),
        fill("ETHUSDT", 3, 3000, "SELL", 2010, 3, rpnl="60", ps="BOTH")]})},
    {"name": "fold_truncated_resync", "symbols": ["ETHUSDC"], "routes": acct_routes(**{"/fapi/v1/userTrades?ETHUSDC": [
        fill("ETHUSDC", 1, 1000, "SELL", 2000, 1.0), fill("ETHUSDC", 2, 2000, "SELL", 2010, 0.5), fill("ETHUSDC", 3, 400000, "BUY", 2020, 0.4),
        fill("ETHUSDC", 4, 900000, "BUY", 2030, 0.1)]})},
    {"name": "fold_truncated_no_resync", "symbols": ["ETHUSDC"], "routes": acct_routes(**{
        "/fapi/v2/positionRisk": [r for r in RISK if r["symbol"] != "ETHUSDC"],
        "/fapi/v1/userTrades?ETHUSDC": [fill("ETHUSDC", 1, 1000, "SELL", 2000, 1.0), fill("ETHUSDC", 2, 2000, "BUY", 2010, 0.3)]})},
    {"name": "fold_eps_residue", "symbols": ["ETHUSDT"], "routes": acct_routes(**{"/fapi/v1/userTrades?ETHUSDT": [
        fill("ETHUSDT", 1, 1000, "BUY", 2000, 0.3, ps="BOTH"), fill("ETHUSDT", 2, 2000, "SELL", 2001, 0.1, ps="BOTH"),
        fill("ETHUSDT", 3, 3000, "SELL", 2002, 0.2, ps="BOTH"), fill("ETHUSDT", 4, 4000, "BUY", 2003, 0.1, ps="BOTH")]})},
    {"name": "fills_at_limit", "symbols": ["ETHUSDT"], "routes": acct_routes(**{"/fapi/v1/userTrades?ETHUSDT": [fill("ETHUSDT", i, 1000 + i, "BUY" if i % 2 else "SELL", 2000, 1, ps="BOTH") for i in range(1, 7)]}),
     "limit_note": "trade_limit 은 기본 1000 -- 이 사례는 limit 미만"},
]


# ═══ B. 사용자 스트림 ════════════════════════════════════════════════════════════
class SCtl:
    ev: list = []
    script: dict = {}
    ban = 0.0
    sleeps = 0
    max_sleeps = 0
    dirty_at: list = []
    flag = False


def sev(*a):
    SCtl.ev.append([round(Clock.t, 6), *a])


def take(name):
    q = SCtl.script.get(name) or []
    if not q:
        return None
    r = q.pop(0)
    Clock.t += r.get("dt", 0.0)
    if "set_ban" in r:
        SCtl.ban = r["set_ban"]
    return r


async def s_sleep(sec):
    sev("sleep", sec)
    SCtl.sleeps += 1
    if SCtl.sleeps > SCtl.max_sleeps:
        raise asyncio.CancelledError()
    Clock.t += sec
    r = take("sleep")
    if r and "set_ban" in r:
        SCtl.ban = r["set_ban"]


async def s_listen_key(method):
    sev("listen_key", method)
    r = take("listen_key") or {"raise": ["RuntimeError", "각본 없음"]}
    if "raise" in r:
        raise exc_of(r["raise"])
    return r["ret"]


class WS:
    async def receive(self, timeout=None):
        r = take("recv")
        if r is None:
            sev("recv", timeout, "close")
            return types.SimpleNamespace(type=WSMsgType.CLOSE, data=None)
        sev("recv", timeout, r.get("kind", "text"))
        if r.get("kind") == "timeout":
            raise asyncio.TimeoutError()
        if r.get("kind") == "raise":
            raise exc_of(r["raise"])
        if r.get("kind") == "close":
            return types.SimpleNamespace(type=WSMsgType.CLOSE, data=None)
        return types.SimpleNamespace(type=WSMsgType.TEXT, data=r["data"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class WSSession:
    def ws_connect(self, url, heartbeat=None):
        sev("connect", url, heartbeat)
        r = take("connect") or {}
        if "raise" in r:
            raise exc_of(r["raise"])
        return WS()


class FakeEvent:
    def set(self):
        sev("dirty")

    def clear(self):
        sev("clear_dirty")
        SCtl.flag = False

    async def wait(self):
        sev("wait_dirty")
        SCtl.flag = SCtl.flag or any(t <= Clock.t for t in SCtl.dirty_at)
        SCtl.dirty_at = [t for t in SCtl.dirty_at if t > Clock.t]
        if SCtl.flag:
            return True
        if not SCtl.dirty_at:
            raise asyncio.CancelledError()
        Clock.t = SCtl.dirty_at.pop(0)
        SCtl.flag = True
        return True


def closure_map(fns):
    seen, out, todo = set(), {}, list(fns)
    while todo:
        f = todo.pop()
        if id(f) in seen or not hasattr(f, "__code__"):
            continue
        seen.add(id(f))
        for n, c in zip(f.__code__.co_freevars, f.__closure__ or ()):
            out.setdefault(n, c)
            try:
                v = c.cell_contents
            except ValueError:
                continue
            if callable(v) and hasattr(v, "__code__"):
                todo.append(v)
    return out


async def boom(*_a, **_k):
    raise AssertionError("시험에서 거래소·공개 시세 조회 금지")


FAKE_ASYNCIO = types.SimpleNamespace(sleep=s_sleep, CancelledError=asyncio.CancelledError, TimeoutError=asyncio.TimeoutError,
                                     Lock=asyncio.Lock, Event=asyncio.Event, gather=asyncio.gather, to_thread=asyncio.to_thread, create_task=None)


def register(tmp, **kw):
    return orders.register(account_trip_state={"seen": {}}, refresh_tasks=kw.get("tasks", {}), swr_locks={}, binance_session=lambda: None,
                           fetch_binance_json=kw.get("fetch", boom), swr_cached=kw.get("swr", boom), _timed_produce=kw.get("produce", boom),
                           load_market_history=boom, situation_by=kw.get("situation", {}), NOCACHE={}, STALE_GRACE_SECONDS=60,
                           sec_prices=kw.get("sec", lambda _a: None))


def run_stream(c):
    SCtl.ev, SCtl.script, SCtl.ban, SCtl.sleeps = [], json.loads(json.dumps(c.get("script") or {})), c.get("ban", 0.0), 0
    SCtl.max_sleeps, SCtl.dirty_at, SCtl.flag = c.get("max_sleeps", 3), list(c.get("dirty_at") or []), False
    Clock.t = T0
    tmp = Path(tempfile.mkdtemp())
    buf = io.StringIO()

    async def f_produce(key, produce):
        sev("produce")
        r = take("produce") or {}
        if "raise" in r:
            raise exc_of(r["raise"])
        return {"ok": True}

    def f_tp_leg(symbol, side, cid):
        sev("tp_leg_filled", symbol, side, cid)
        r = take("tp_leg") or {}
        if "raise" in r:
            raise exc_of(r["raise"])

    with mock.patch.multiple(orders, LIVE_DIR=tmp, time=FAKE_TIME, asyncio=FAKE_ASYNCIO, wake_fills=lambda: sev("wake"),
                             ban_guard=types.SimpleNamespace(ban_remaining=lambda: SCtl.ban, note=lambda *a, **k: 0.0)):
        api = register(tmp, produce=f_produce)
        cells = closure_map([api.start_user_stream, api.api_binance_account])
        cells["listen_key"].cell_contents = s_listen_key
        cells["acct_dirty"].cell_contents = FakeEvent()
        cells["tp_leg_filled"].cell_contents = f_tp_leg
        us = cells["user_stream"].cell_contents
        fn = cells["_user_stream_run"].cell_contents if c["op"] == "stream" else cells["account_refresher"].cell_contents
        with contextlib.redirect_stdout(buf):
            try:
                asyncio.run(fn(WSSession(), 5.0) if c["op"] == "stream" else fn())
            except asyncio.CancelledError:
                pass
    return {"name": c["name"], "in": c, "events": SCtl.ev, "user_stream": snap(us), "logs": buf.getvalue().splitlines()}


def otu(x, X="NEW", c="dbpgB1", s="ETHUSDC", ps="LONG", o="LIMIT"):
    return json.dumps({"e": "ORDER_TRADE_UPDATE", "o": {"s": s, "x": x, "X": X, "c": c, "ps": ps, "o": o, "S": "BUY", "q": "1", "z": "0", "p": "2400"}})


STREAM_CASES = [
    {"name": "connect_events_close", "op": "stream", "max_sleeps": 1, "script": {"listen_key": [{"ret": "lk1"}], "recv": [
        {"kind": "timeout", "dt": 60}, {"data": otu("NEW")}, {"data": otu("TRADE", "PARTIALLY_FILLED")}, {"data": otu("CANCELED", "CANCELED")},
        {"data": json.dumps({"e": "ACCOUNT_UPDATE", "a": {}})}, {"data": "not json"}, {"data": "[1,2]"}, {"data": otu("TRADE", "FILLED", c="dbtpL9")},
        {"data": otu("AMENDMENT")}, {"kind": "close"}]}},
    {"name": "expired_reconnect_backoff", "op": "stream", "max_sleeps": 3, "script": {
        "listen_key": [{"ret": "a"}, {"raise": ["RuntimeError", "listenKey POST 503 busy"]}, {"ret": ""}, {"ret": "b"}],
        "recv": [{"data": json.dumps({"e": "listenKeyExpired"})}]}},
    {"name": "keepalive_and_long_conn_resets_wait", "op": "stream", "max_sleeps": 2, "script": {
        "listen_key": [{"ret": "a"}, {"raise": ["RuntimeError", "listenKey PUT 400 x"]}, {"ret": "b"}],
        "recv": [{"kind": "timeout", "dt": 1000}, {"kind": "timeout", "dt": 900}, {"data": otu("TRADE"), "dt": 1}, {"kind": "close"}]}},
    {"name": "backoff_then_long_conn_resets_wait", "op": "stream", "max_sleeps": 4, "script": {
        "listen_key": [{"raise": ["RuntimeError", "listenKey POST 502 x"]}, {"raise": ["RuntimeError", "listenKey POST 502 y"]}, {"ret": "c"}],
        "recv": [{"kind": "timeout", "dt": 400}, {"kind": "close"}]}},
    {"name": "ban_waits_then_connects", "op": "stream", "ban": 30.0, "max_sleeps": 2, "script": {"sleep": [{"set_ban": 0.0}], "listen_key": [{"ret": "z"}], "recv": [{"kind": "close"}]}},
    {"name": "connect_fails_and_recv_raises", "op": "stream", "max_sleeps": 3, "script": {
        "listen_key": [{"ret": "a"}, {"ret": "b"}], "connect": [{"raise": ["WSServerHandshakeError", "403, message='Invalid response status'"]}, {}],
        "recv": [{"kind": "raise", "raise": ["ClientError", "x" * 300]}]}},
    {"name": "tp_leg_filled_raises_keeps_stream", "op": "stream", "max_sleeps": 1, "script": {"listen_key": [{"ret": "a"}], "tp_leg": [{"raise": ["KeyError", "'tp_legs'"]}],
        "recv": [{"data": otu("TRADE", "FILLED", c="dbtpS2", ps="SHORT")}, {"data": json.dumps({"e": "ORDER_TRADE_UPDATE", "o": "bad"})}]}},
    {"name": "refresher_coalesce_and_gap", "op": "refresh", "max_sleeps": 20, "dirty_at": [T0 + 1.0, T0 + 1.1, T0 + 1.2, T0 + 2.0, T0 + 10.0, T0 + 10.25],
     "script": {"produce": [{}, {"raise": ["RuntimeError", "418 banned"]}, {}]}},
]


# ═══ C·D. 계획 조립 · 접수 문 ═════════════════════════════════════════════════════
class Ctl:
    calls: list = []
    account = None
    fresh = None
    books: dict = {}
    filters: dict = {}
    vol: dict = {}
    funding: dict = {}
    fees: dict = {}
    margin: dict = {}
    sec: dict = {}
    exec_on = True
    script: dict = {}
    started: list = []


def note(*a):
    Ctl.calls.append(list(a))


def resp_of(name):
    q = Ctl.script.get(name) or []
    if not q:
        raise RuntimeError(f"각본 없음: {name}")
    r = q.pop(0)
    if "patch_file" in r:
        p = Ctl.live / "manual_bracket_state.json"
        st = json.loads(p.read_text("utf-8")) if p.exists() else {}
        for k, v in r["patch_file"].items():
            st[k] = v
        p.write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    if "raise" in r:
        raise exc_of(r["raise"])
    return r.get("ret")


def raise_or(v):
    if isinstance(v, dict) and "__raise__" in v:
        raise exc_of(v["__raise__"])
    return v


async def f_swr(key, ttl, produce, **_kw):
    note("account", False)
    return raise_or(Ctl.account)


async def f_produce():
    note("account", True)
    return raise_or(Ctl.fresh if Ctl.fresh is not None else Ctl.account)


async def f_fetch(url, params=None, **_kw):
    if url.endswith("/ticker/bookTicker"):
        note("book", params["symbol"])
        b = Ctl.books.get(params["symbol"])
        if b is None:
            raise RuntimeError(f"book_ticker_failed {params['symbol']}")
        return raise_or(b)
    raise AssertionError(f"모르는 공개 조회 {url}")


async def f_filters(_s, symbol):
    note("filters", symbol)
    v = Ctl.filters.get(symbol)
    if v is None:
        raise RuntimeError(f"symbol not in exchangeInfo: {symbol}")
    return raise_or(dict(v))


async def f_vol(symbol):
    note("vol", symbol)
    return Ctl.vol.get(symbol)


async def f_funding(symbol):
    note("funding", symbol)
    return Ctl.funding.get(symbol)


async def f_fees(_s, symbol):
    note("fees", symbol)
    return Ctl.fees.get(symbol) or {"symbol": symbol, "maker_bp": 2.0, "taker_bp": 5.0, "source": "fallback"}


async def f_clear(_s, symbol, pside, *, only_sl=False, only_tp=False):
    note("clear_bracket", symbol, pside, only_sl, only_tp)
    return resp_of("clear_bracket")


async def f_place_bracket(_s, bracket, symbol, pside, *, strict=False):
    note("place_bracket", bracket, symbol, pside, strict)
    return resp_of("place_bracket")


async def f_ladder(_s, legs, symbol, pside):
    note("place_tp_ladder", legs, symbol, pside)
    return resp_of("place_tp_ladder")


async def _noop():
    return None


def f_etb(plan, state, limit=False):
    Ctl.started.append({"fn": "entry_then_bracket", "plan": snap(plan), "limit": limit})
    return _noop()


def f_settle(coro, state):
    name = coro.cr_code.co_name
    coro.close()
    Ctl.started.append({"fn": f"settle({name})", "plan": snap(state.get("plan"))})
    return _noop()


async def run_exit(_s, plan, state):   # 이름이 settle 기록에 쓰인다
    return None


async def exit_then_rearm(plan, state):
    return None


class FakeTask:
    def __init__(self, done):
        self._d = done

    def done(self):
        return self._d


class Tasks(dict):
    pass


class Req:
    def __init__(self, query):
        self.query = query


def setup(c, tmp):
    Ctl.calls, Ctl.started = [], []
    Ctl.account, Ctl.fresh = c.get("account"), c.get("fresh_account")
    Ctl.books, Ctl.filters = c.get("books", BOOKS), c.get("filters", FILTERS)
    Ctl.vol, Ctl.funding, Ctl.fees = c.get("vol", {}), c.get("funding", {}), c.get("fees", {})
    Ctl.sec, Ctl.exec_on = c.get("sec", {}), c.get("exec", True)
    Ctl.script = json.loads(json.dumps(c.get("script") or {}))
    Ctl.live = tmp
    Clock.t = c.get("t", T0)
    if "file0" in c:
        (tmp / "manual_bracket_state.json").write_text(json.dumps(c["file0"], ensure_ascii=False), "utf-8")


async def run_desk_case(c):
    tmp = Path(tempfile.mkdtemp())
    setup(c, tmp)
    tasks = Tasks({k: FakeTask(v) for k, v in (c.get("tasks") or {}).items()})
    fa = types.SimpleNamespace(**{**FAKE_ASYNCIO.__dict__, "sleep": asyncio.sleep, "create_task": lambda coro: (coro.close(), FakeTask(False))[1]})
    buf = io.StringIO()
    with mock.patch.multiple(orders, LIVE_DIR=tmp, time=FAKE_TIME, datetime=FakeDT, asyncio=fa, exec_enabled=lambda: Ctl.exec_on,
                             load_filters=f_filters, commission_bp=f_fees, clear_bracket=f_clear,
                             run_exit=run_exit, SIZING_CAP_OVERRIDE_X=c.get("override_x", 0.0),
                             fetch_account=boom, run_entry=boom, place_bracket=f_place_bracket, place_tp_ladder=f_ladder):
        api = register(tmp, tasks=tasks, fetch=f_fetch, swr=f_swr, situation=c.get("situation", {}),
                       sec=lambda a: None if Ctl.sec.get(a) is None else [tuple(x) for x in Ctl.sec[a]])
        cells = closure_map([api.api_manual_entry_submit, api.api_manual_exit_submit, api.api_manual_limit_amend, api.busy_or, api.bracket_tick,
                             api.api_manual_wait_action, api.api_manual_bracket_rearm, api.start_wait_cleanup])
        cells["produce_account"].cell_contents = f_produce
        cells["realized_vol_now"].cell_contents = f_vol
        cells["funding_now"].cell_contents = f_funding
        cells["entry_then_bracket"].cell_contents = f_etb
        cells["settle"].cell_contents = f_settle
        cells["exit_then_rearm"].cell_contents = exit_then_rearm
        sl = {k: cells[k].cell_contents for k in ("manual_entry_state", "side_exit_states", "manual_wait_states", "bracket_exiting")}
        s0 = c.get("slots0") or {}
        sl["manual_entry_state"].clear()
        sl["manual_entry_state"].update(s0.get("entry") or {"phase": "idle"})
        sl["side_exit_states"].update(s0.get("side") or {})
        sl["manual_wait_states"].update(json.loads(json.dumps(s0.get("waits") or {})))
        sl["bracket_exiting"].update(s0.get("exiting") or [])
        out = {"name": c["name"], "in": c}
        with contextlib.redirect_stdout(buf):
            if c["op"] == "entry":
                a = c["args"]
                plan, cap, _sz, err = await cells["assemble_entry_plan"].cell_contents(
                    a["side"], a.get("fraction", 1.0), a.get("want_lev"), a.get("asset", "eth"), fresh=a.get("fresh", False),
                    lines=tuple(a["lines"]) if a.get("lines") else None, limit_usdt=a.get("limit_usdt"), stop=a.get("stop", False), skip_wait=a.get("skip_wait"))
                out.update(plan=snap(plan), cap=snap(cap), err=snap(err))
            elif c["op"] == "exit":
                a = c["args"]
                plan, err = await cells["assemble_exit_plan"].cell_contents(a["side"], a.get("fraction", 1.0), fresh=a.get("fresh", True), asset=a.get("asset", "eth"),
                                                                            for_order=a.get("for_order", False), only_symbol=a.get("only_symbol"))
                out.update(plan=snap(plan), err=snap(err))
            else:
                h = getattr(api, c["op"])
                r = await h(Req(c["query"]))
                body = json.loads(r.text)
                body.pop("server_ms", None)
                out.update(status=r.status, body=snap(body), started=Ctl.started, task_keys=sorted(k for k in tasks if k not in (c.get("tasks") or {})))
        p = tmp / "manual_bracket_state.json"
        out.update(calls=Ctl.calls, logs=[l for l in buf.getvalue().splitlines() if not l.startswith("plan_timing")],
                   slots={"entry": snap(sl["manual_entry_state"]), "side": snap(sl["side_exit_states"]), "waits": snap(sl["manual_wait_states"])},
                   file=json.loads(p.read_text("utf-8")) if p.exists() else None,
                   limits={q.name: json.loads(q.read_text("utf-8")) for q in sorted(tmp.glob("manual_limit_wait*.json"))})
    return out


# ── 사례 재료(명백한 가짜 숫자) ──
BOOKS = {"ETHUSDC": {"bidPrice": "2500.11", "askPrice": "2500.37"}, "ETHUSDT": {"bidPrice": "2499.5", "askPrice": "2499.7"},
         "SOLUSDC": {"bidPrice": "150.123", "askPrice": "150.131"}, "SOLUSDT": {"bidPrice": "150.1", "askPrice": "150.12"},
         "XRPUSDC": {"bidPrice": "0.51234", "askPrice": "0.51241"}, "XRPUSDT": {"bidPrice": "0.5122", "askPrice": "0.5123"},
         "SOXLUSDT": {"bidPrice": "30.11", "askPrice": "30.13"}, "BTCUSDC": {"bidPrice": "90000.1", "askPrice": "90000.3"}, "BTCUSDT": {"bidPrice": "89990", "askPrice": "89990.2"}}
FILTERS = {"ETHUSDC": {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}, "ETHUSDT": {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0},
           "SOLUSDC": {"step": 0.01, "tick": 0.001, "min_qty": 0.01, "min_notional": 5.0}, "SOLUSDT": {"step": 0.01, "tick": 0.01, "min_qty": 0.01, "min_notional": 5.0}, "XRPUSDC": {"step": 0.1, "tick": 0.00001, "min_qty": 0.1, "min_notional": 5.0},
           "SOXLUSDT": {"step": 1.0, "tick": 0.01, "min_qty": 1.0, "min_notional": 5.0}, "BTCUSDC": {"step": 0.001, "tick": 0.1, "min_qty": 0.001, "min_notional": 100.0}}


def pos(sym, side, qty, ep, mark, lev=10.0, liq=0.0, up=0.0, entry_at=None):
    return {"symbol": sym, "side": side, "qty": qty, "entry_price": ep, "mark_price": mark, "liquidation_price": liq, "leverage": lev,
            "notional": abs(qty * mark), "unrealized_pnl": up, "updated_at": iso(T0 - 7200), "entry_at": entry_at or iso(T0 - 3600)}


def ACC(positions=(), orders=(), usdc=(1000.0, 800.0, 150.0), usdt=(300.0, 250.0, 20.0), lev=None, ok=True, multi=False, balances=True):
    a = {"ok": ok, "generated_at": iso(T0 - 1), "positions": list(positions), "open_orders": list(orders),
         "leverage_by_symbol": lev if lev is not None else {"ETHUSDC": 10.0, "SOLUSDC": 5.0, "SOXLUSDT": 3.0, "XRPUSDC": 8.0},
         "multi_assets_margin": multi, "balance": {"margin": usdc[0], "available": usdc[1], "initial_margin": usdc[2]}}
    if balances:
        a["balances"] = {"USDC": {"margin": usdc[0], "available": usdc[1], "initial_margin": usdc[2]}, "USDT": {"margin": usdt[0], "available": usdt[1], "initial_margin": usdt[2]}}
    if not ok:
        a = {"ok": False, "error": "positionRisk: 418 [-1003] banned"}
    return a


def oo(sym, side, px, qty, cid="", ro=False):
    return {"symbol": sym, "side": side, "price": px, "qty": qty, "reduce_only": ro, "cid": cid}


RULE_ARM = {"ETHUSDC:LONG": {"symbol": "ETHUSDC", "side": "LONG", "sl_price": 2375.1, "sl_level": 2374.5, "backstop_price": 2375.1, "rule": True, "auto": True,
                             "sl_name": "첫 진입가 −5%", "armed_at": T0 - 500}}   # 배포 전 옛 규칙 무장(5분봉 종가 감시는 계속 · 선은 못 건다)
LINES_ARM = {"ETHUSDC:LONG": {"symbol": "ETHUSDC", "side": "LONG", "lines": True, "rule": False, "tp_level": 2600.0, "tp_price": 2600.25, "sl_level": 2400.0,
                              "sl_price": 2400.2, "sl_name": "선 손절 2400.00", "sl_up": False, "tp_resting": True, "armed_at": T0 - 500}}
LINES_ARM_S = {"ETHUSDC:SHORT": {**LINES_ARM["ETHUSDC:LONG"], "side": "SHORT", "tp_level": 2300.0, "tp_price": 2300.2, "sl_level": 2600.0, "sl_price": 2600.3, "sl_name": "선 손절 2600.00"}}
SEC = {"eth": [[T0 - 1, 2499.6]], "soxl": [[T0 - 1, 30.12]], "sol": [[T0 - 1, 150.11]], "btc": [[T0 - 1, 89990.1]], "xrp": [[T0 - 1, 0.5122]]}
# 2026-11-21 PR 61: 진입은 선(익절·손절)이 있어야 받는다 -- 크기 조립을 보는 사례는 코인·방향마다 맞는 선을 붙인다
LN = {("eth", "LONG"): [2600.0, 2400.0], ("eth", "SHORT"): [2400.0, 2600.0], ("sol", "LONG"): [160.0, 140.0], ("soxl", "SHORT"): [28.0, 32.0],
      ("btc", "LONG"): [95000.0, 85000.0], ("xrp", "LONG"): [0.55, 0.48]}


def W(asset_sym_plan, **kw):
    return {"phase": "working", **kw, "plan": asset_sym_plan}


E0 = lambda **kw: {"op": "entry", "args": {"side": "LONG", **kw}}   # noqa: E731 -- 선 없이(lines_required)


def E(**kw):
    a = {"side": "LONG", **kw}
    a.setdefault("lines", LN[(a.get("asset", "eth"), a["side"])])
    return {"op": "entry", "args": a}
X = lambda **kw: {"op": "exit", "args": {"side": "LONG", **kw}}   # noqa: E731
P_ETH = [pos("ETHUSDC", "LONG", 0.4, 2400.0, 2450.0, liq=1800.0, up=20.0)]
ENTRY_CASES = [
    ("eth_long_lev10", E(want_lev=10), {"sec": SEC}),
    ("eth_short_half_model_lev", E(side="SHORT", fraction=0.5), {"sec": SEC}),
    ("eth_long_no_balances_field", E(want_lev=10), {"sec": SEC, "account": ACC(balances=False)}),
    ("sol_long", E(asset="sol", want_lev=5, fraction=0.3), {"sec": SEC}),
    ("soxl_short_usdt_pool", E(side="SHORT", asset="soxl", want_lev=3), {"sec": SEC, "account": ACC(positions=P_ETH + [pos("SOXLUSDT", "SHORT", 3, 31.0, 30.12, lev=3.0)]),
                                                                   "file0": {"SOXLUSDT:SHORT": {**LINES_ARM_S["ETHUSDC:SHORT"], "symbol": "SOXLUSDT", "tp_level": 28.0, "tp_price": 28.01, "sl_level": 32.0, "sl_price": 32.01}}}),
    ("btc_long_tiny_blocked", E(asset="btc", want_lev=1, fraction=0.001), {"sec": SEC}),
    ("other_symbol_orders_and_waits", E(want_lev=10),
     {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=P_ETH, orders=[oo("XRPUSDC", "BUY", 0.5, 1000.0, "dbad1"), oo("SOXLUSDT", "BUY", 29.0, 5.0), oo("ETHUSDC", "BUY", 2300.0, 0.2, "dbwtB1"),
                                                                         oo("XRPUSDC", "SELL", 0.6, 500.0, "dbtpS", ro=True), oo("SOLUSDC", "BUY", 140.0, 2.0, "dbwtB2")]),
      "slots0": {"waits": {"sol": W({"quantity": 3.0, "price": 140.0}, filled=0.5), "eth#2": W({"quantity": 0.1, "price": 2300.0}, limit_price=2301.0),
                           "soxl": W({"quantity": 5, "price": 29.0}), "xrp": {"phase": "filled_maker", "plan": {"quantity": 100.0, "price": 0.5}}}}}),
    ("skip_wait_own_order_in_account", E(want_lev=10, skip_wait="eth#2", lines=[2600, 2400], limit_usdt=2450.0),
     {"sec": SEC, "account": ACC(orders=[oo("ETHUSDC", "BUY", 2450.12, 0.3, "dbwtB9")]),
      "slots0": {"waits": {"eth#2": W({"quantity": 0.3, "price": 2450.12}, order_id=55, limit_price=2450.12), "eth": W({"quantity": 0.1, "price": 2440.0}, stop=True)}}}),
    # 2026-11-21 PR 61 새 거절: 선 없는 진입 400 · 선 없는 포지션(옛 규칙 무장·앱 진입)에 추가 진입 409
    ("lines_required_no_lines", E0(want_lev=10), {"sec": SEC}),
    ("lines_required_on_lined_position", E0(want_lev=10), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=P_ETH)}),
    ("add_on_rule_position_409", E(want_lev=10), {"sec": SEC, "file0": RULE_ARM, "account": ACC(positions=P_ETH)}),
    ("add_on_app_position_409", E(want_lev=10), {"sec": SEC, "account": ACC(positions=P_ETH)}),
    ("limit_add_on_app_position_409", E(want_lev=10, lines=[2700.0, 2300.0], limit_usdt=2440.0), {"sec": SEC, "account": ACC(positions=P_ETH)}),
    ("same_side_add_entry_at_old", E(want_lev=10), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=[pos("ETHUSDC", "LONG", 0.4, 2400.0, 2450.0, entry_at=iso(T0 - 4 * 3600 - 60))])}),
    ("same_side_add_entry_at_naive", E(want_lev=10), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=[pos("ETHUSDC", "LONG", 0.4, 2400.0, 2450.0, entry_at="2027-01-15T00:00:00")])}),
    ("hedge_both_short_entry", E(side="SHORT", want_lev=10), {"sec": SEC, "file0": LINES_ARM_S, "account": ACC(positions=P_ETH + [pos("ETHUSDC", "SHORT", 0.2, 2480.0, 2450.0, liq=2900.0)])}),
    ("no_leverage_zero_equity", E(), {"sec": SEC, "account": ACC(usdc=(0.0, 0.0, 0.0), lev={})}),
    ("no_leverage_anywhere", E(), {"sec": SEC, "account": ACC(lev={}, usdc=(1000.0, 800.0, None))}),
    ("account_not_ok", E(), {"account": ACC(ok=False)}),
    ("account_raises", E(), {"account": {"__raise__": ["ClientOSError", "boom"]}}),
    ("book_fails", E(), {"sec": SEC, "books": {k: v for k, v in BOOKS.items() if k != "ETHUSDC"}}),
    ("book_missing_ask", E(), {"sec": SEC, "books": {**BOOKS, "ETHUSDC": {"bidPrice": "2500"}}}),
    ("filters_fail", E(), {"sec": SEC, "filters": {}}),
    ("basis_ref_fails", E(want_lev=10), {"sec": SEC, "books": {k: v for k, v in BOOKS.items() if k != "ETHUSDT"}}),
    ("fresh_account", E(want_lev=10, fresh=True), {"sec": SEC, "fresh_account": ACC(usdc=(500.0, 400.0, 10.0))}),
    ("override_x", E(want_lev=20), {"sec": SEC, "override_x": 3.0}),
    ("lines_long", E(want_lev=10, lines=[2600.0, 2400.0]), {"sec": SEC, "vol": {"ETHUSDC": 4.2}, "funding": {"ETHUSDC": 0.8}, "fees": {"ETHUSDC": {"symbol": "ETHUSDC", "maker_bp": 0.0, "taker_bp": 4.0, "source": "live"}}}),
    ("lines_short", E(side="SHORT", want_lev=10, lines=[2400.0, 2600.0]), {"sec": SEC, "vol": {"ETHUSDC": 35.0}}),
    ("lines_no_feed", E(want_lev=10, lines=[2600.0, 2400.0]), {}),
    ("lines_bad_direction", E(want_lev=10, lines=[2400.0, 2600.0]), {"sec": SEC}),
    ("limit_long", E(want_lev=10, lines=[2600.0, 2300.0], limit_usdt=2450.0), {"sec": SEC}),
    ("limit_short", E(side="SHORT", want_lev=10, lines=[2400.0, 2700.0], limit_usdt=2550.0), {"sec": SEC}),
    ("limit_marketable", E(want_lev=10, lines=[2600.0, 2300.0], limit_usdt=2510.0), {"sec": SEC}),
    ("stop_entry_above", E(want_lev=10, lines=[2700.0, 2500.0], limit_usdt=2560.0, stop=True), {"sec": SEC}),
    ("limit_tiny_blocked", E(want_lev=1, fraction=0.00005, lines=[2600.0, 2300.0], limit_usdt=2450.0), {"sec": SEC}),
    ("add_on_lines_position", E(want_lev=10, lines=[2700.0, 2300.0]), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=P_ETH)}),
    ("add_on_lines_tp_off_sl_off_legs", E(want_lev=10), {"sec": SEC, "account": ACC(positions=P_ETH),
     "file0": {"ETHUSDC:LONG": {**LINES_ARM["ETHUSDC:LONG"], "tp_off": {"at": 1}, "sl_off": {"at": 2}, "tp_legs": [{"level": 2600.0}, {"level": 2650.0}]}}}),
    ("add_on_lines_limit", E(want_lev=10, lines=[2700.0, 2300.0], limit_usdt=2440.0), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=P_ETH)}),
    ("exiting_fired_409", E(want_lev=10), {"sec": SEC, "file0": {"ETHUSDC:LONG": {**LINES_ARM["ETHUSDC:LONG"], "fired": True}}, "account": ACC(positions=P_ETH)}),
    ("exiting_set_409", E(want_lev=10), {"sec": SEC, "file0": LINES_ARM, "account": ACC(positions=P_ETH), "slots0": {"exiting": ["ETHUSDC:LONG"]}}),
    ("multi_assets_pool", E(asset="soxl", side="SHORT", want_lev=3), {"sec": SEC, "account": ACC(positions=P_ETH, multi=True)}),
]
EXIT_CASES = [
    ("exit_long_full", X(), {"account": ACC(positions=P_ETH), "vol": {"ETHUSDC": 5.0}, "funding": {"ETHUSDC": -0.3}}),
    ("exit_short_half", X(side="SHORT", fraction=0.5), {"account": ACC(positions=[pos("ETHUSDC", "SHORT", 0.3, 2480.0, 2450.0)])}),
    ("exit_no_position", X(), {"account": ACC()}),
    ("exit_account_not_ok", X(), {"account": ACC(ok=False)}),
    ("exit_account_raises", X(), {"account": {"__raise__": ["TimeoutError", ""]}}),
    ("exit_cached_account", X(fresh=False), {"account": ACC(positions=P_ETH)}),
    ("exit_both_symbols_leftover", X(), {"account": ACC(positions=P_ETH + [pos("ETHUSDT", "LONG", 0.1, 2390.0, 2449.0)])}),
    ("exit_only_symbol_usdt", X(only_symbol="ETHUSDT"), {"account": ACC(positions=P_ETH + [pos("ETHUSDT", "LONG", 0.1, 2390.0, 2449.0)])}),
    ("exit_only_symbol_missing", X(only_symbol="ETHUSDT"), {"account": ACC(positions=P_ETH)}),
    ("exit_high_vol_m6", X(for_order=True), {"account": ACC(positions=P_ETH), "vol": {"ETHUSDC": 31.0}}),
    ("exit_dust_m7", X(for_order=True), {"account": ACC(positions=[pos("ETHUSDC", "LONG", 0.005, 2400.0, 2450.0)])}),
    ("exit_for_order_skips_display", X(for_order=True), {"account": ACC(positions=P_ETH), "vol": {"ETHUSDC": 3.0}}),
    ("exit_book_fails", X(), {"account": ACC(positions=P_ETH), "books": {}}),
    ("exit_fraction_tiny_blocked", X(fraction=0.001), {"account": ACC(positions=P_ETH)}),
    ("exit_old_entry_hold_60", X(), {"account": ACC(positions=[pos("ETHUSDC", "LONG", 0.4, 2400.0, 2450.0, entry_at=iso(T0 - 5 * 3600))])}),
    ("exit_sol_usdt_position", X(asset="sol"), {"account": ACC(positions=[pos("SOLUSDT", "LONG", 2.0, 140.0, 150.1)])}),
]


def API(op, query, **kw):
    return {"op": op, "query": query, **kw}


Q = lambda **kw: {k: str(v) for k, v in kw.items()}   # noqa: E731
BUSY_E = {"phase": "working", "side": "LONG", "asset": "eth", "plan": {"type": "LIMIT"}}
LIMIT_W = {"phase": "working", "mode": "limit", "side": "LONG", "asset": "eth", "wid": 1, "order_id": 77, "limit_price": 2450.11, "filled": 0.0,
           "plan": {"symbol": "ETHUSDC", "positionSide": "LONG", "price": 2450.11, "quantity": 0.3, "fraction": 0.5, "target_leverage": 10, "wid": 1,
                    "limit": {"usdt": 2450.0, "price": 2450.11}, "bracket": {"tp_level": 2600.0, "sl_level": 2300.0}}}
ADD_W = {**LIMIT_W, "plan": {**LIMIT_W["plan"], "add_keep": True}}

LA = LINES_ARM["ETHUSDC:LONG"]
TPL3 = [{"level": 2600.0, "price": 2600.25, "w": 1.0, "resting": True, "cid": "dbtpL1"}, {"level": 2650.0, "price": 2650.3, "w": 1.0, "resting": True, "cid": "dbtpL2"},
        {"level": 2700.0, "price": 2700.3, "w": 2.0, "resting": True, "cid": "dbtpL3"}]
SLL2 = [{"level": 2400.0, "price": 2400.2, "up": False, "w": 1.0}, {"level": 2380.0, "price": 2380.2, "up": False, "w": 1.0}]
F_LEGS = {"ETHUSDC:LONG": {**LA, "tp_legs": TPL3}}
F_SL = {"ETHUSDC:LONG": {**LA, "sl_legs": SLL2}}
LAD_OK = lambda *lv: {"ret": {"qty": 0.4, "cleared": {"cancelled": 3, "errors": []}, "placed": True,   # noqa: E731
                              "legs": [{"level": l, "price": l + 0.25, "w": 1.0, "qty": 0.1, "resting": True, "merged": False, "cid": f"dbtpL{i}", "order_id": 20 + i, "error": None} for i, l in enumerate(lv)]}}
PB_OK = {"ret": {"qty": 0.4, "cleared": {"cancelled": 1, "errors": []}, "tp": {"placed": True, "price": 2650.3, "order_id": 31, "error": None}, "placed": True}}
PB_FAILTP = {"ret": {"qty": 0.4, "cleared": {"cancelled": 0, "errors": ["400 [-2011] x"]}, "tp": {"placed": False, "price": 2650.3, "order_id": None, "error": "400 [-5022] maker"}, "placed": False}}
ACCP = ACC(positions=P_ETH)
LQ = lambda **kw: Q(confirm=1, side="LONG", symbol="ETHUSDC", **kw)   # noqa: E731
LINE_CASES = [
    ("tl_no_confirm", API("api_manual_bracket_tp_legs", Q(side="LONG", symbol="ETHUSDC", act="remove", i=0))),
    ("tl_bad_args", API("api_manual_bracket_tp_legs", LQ(act="drop"))),
    ("tl_no_legs", API("api_manual_bracket_tp_legs", LQ(act="remove", i=0), file0=LINES_ARM)),
    ("tl_fired", API("api_manual_bracket_tp_legs", LQ(act="remove", i=0), file0={"ETHUSDC:LONG": {**LA, "tp_legs": TPL3, "fired": True}})),
    ("tl_bad_i", API("api_manual_bracket_tp_legs", LQ(act="remove", i=3), file0=F_LEGS)),
    ("tl_bad_w", API("api_manual_bracket_tp_legs", LQ(act="weights", w="1,0,1"), file0=F_LEGS)),
    ("tl_remove_ok", API("api_manual_bracket_tp_legs", LQ(act="remove", i=1), file0=F_LEGS, script={"place_tp_ladder": [LAD_OK(2600.0, 2700.0)]})),
    ("tl_weights_ok", API("api_manual_bracket_tp_legs", LQ(act="weights", w="1,2,3"), file0=F_LEGS, script={"place_tp_ladder": [LAD_OK(2600.0, 2650.0, 2700.0)]})),
    ("tl_place_failed", API("api_manual_bracket_tp_legs", LQ(act="remove", i=0), file0=F_LEGS, script={"place_tp_ladder": [{"ret": {"placed": False, "reason": "포지션 없음"}}]})),
    ("tl_partial_cancel", API("api_manual_bracket_tp_legs", LQ(act="remove", i=0), file0=F_LEGS, script={"place_tp_ladder": [{"ret": {"qty": 0.4, "cleared": {"cancelled": 1, "errors": ["400 [-2011] gone"]}, "legs": [], "placed": False}}]})),
    ("tl_rearmed_race", API("api_manual_bracket_tp_legs", LQ(act="remove", i=0), file0=F_LEGS,
                            script={"place_tp_ladder": [{**LAD_OK(2650.0, 2700.0), "patch_file": {"ETHUSDC:LONG": {**LA, "tp_legs": TPL3, "armed_at": T0}}}], "clear_bracket": [{"ret": {"cancelled": 2, "errors": []}}]})),
    ("tl_sl_weights_ok", API("api_manual_bracket_tp_legs", LQ(act="weights", kind="sl", w="2,1"), file0=F_SL)),
    ("tl_sl_remove_ok", API("api_manual_bracket_tp_legs", LQ(act="remove", kind="sl", i=0), file0=F_SL)),
    ("tp_not_lines", API("api_manual_bracket_tp", LQ(act="off"), file0=RULE_ARM)),
    ("tp_already_on", API("api_manual_bracket_tp", LQ(act="on"), file0=LINES_ARM)),
    ("tp_off_ok", API("api_manual_bracket_tp", LQ(act="off"), file0=LINES_ARM, script={"clear_bracket": [{"ret": {"cancelled": 1, "errors": [], "tp_open": ["dbtpL1"]}}]})),
    ("tp_off_with_legs", API("api_manual_bracket_tp", LQ(act="off"), file0=F_LEGS, script={"clear_bracket": [{"ret": {"cancelled": 3, "errors": []}}]})),
    ("tp_off_cancel_fails", API("api_manual_bracket_tp", LQ(act="off"), file0=F_LEGS, script={"clear_bracket": [{"ret": {"cancelled": 1, "errors": ["400 [-2011] x"]}}]})),
    ("tp_on_ok", API("api_manual_bracket_tp", LQ(act="on"), sec=SEC, file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}, "tp_resting": False, "placed": {"tp": None}}}, script={"place_bracket": [PB_OK]})),
    ("tp_on_past", API("api_manual_bracket_tp", LQ(act="on"), sec={"eth": [[T0 - 1, 2610.0]]}, file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}}})),
    ("tp_on_no_quote", API("api_manual_bracket_tp", LQ(act="on"), sec={"eth": [[T0 - 9, 2500.0]]}, file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}}})),
    ("tp_on_place_failed", API("api_manual_bracket_tp", LQ(act="on"), sec=SEC, file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}}}, script={"place_bracket": [{"ret": {"placed": False, "reason": "포지션 없음"}}]})),
    ("tp_on_rearmed_race", API("api_manual_bracket_tp", LQ(act="on"), sec=SEC, file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}}},
                               script={"place_bracket": [{**PB_OK, "patch_file": {"ETHUSDC:LONG": {**LA, "armed_at": T0}}}], "clear_bracket": [{"ret": {"cancelled": 1, "errors": []}}]})),
    ("add_bad_args", API("api_manual_bracket_add", LQ(leg="sl", level="x"))),
    ("add_rule_armed", API("api_manual_bracket_add", LQ(leg="sl", level=2400), file0=RULE_ARM)),
    ("add_has_line", API("api_manual_bracket_add", LQ(leg="sl", level=2390), file0={"ETHUSDC:LONG": {**{k: v for k, v in LA.items() if k not in ("sl_level",)}}}, account=ACCP, sec=SEC)),
    ("add_no_position", API("api_manual_bracket_add", LQ(leg="sl", level=2400), account=ACC(), sec=SEC)),
    ("add_new_arm_sl", API("api_manual_bracket_add", LQ(leg="sl", level=2400, slup=0), account=ACCP, sec=SEC)),
    ("add_new_arm_tp", API("api_manual_bracket_add", LQ(leg="tp", level=2650), account=ACCP, sec=SEC, script={"place_bracket": [PB_OK]})),
    ("add_new_arm_tp_not_resting", API("api_manual_bracket_add", LQ(leg="tp", level=2650), account=ACCP, sec=SEC, script={"place_bracket": [PB_FAILTP]})),
    ("add_readd_sl_off", API("api_manual_bracket_add", LQ(leg="sl", level=2410), account=ACCP, sec=SEC, file0={"ETHUSDC:LONG": {**LA, "sl_off": {"at": 1}, "sl_legs": SLL2}})),
    ("add_ladder_tp", API("api_manual_bracket_add", LQ(leg="tp", level=2650), account=ACCP, sec=SEC, file0=LINES_ARM, script={"place_tp_ladder": [LAD_OK(2600.0, 2650.0)]})),
    ("add_ladder_sl", API("api_manual_bracket_add", LQ(leg="sl", level=2380), account=ACCP, sec=SEC, file0=LINES_ARM)),
    ("add_same_level", API("api_manual_bracket_add", LQ(leg="sl", level=2400.5), account=ACCP, sec=SEC, file0=LINES_ARM)),
    ("add_too_many", API("api_manual_bracket_add", LQ(leg="tp", level=2800), account=ACCP, sec=SEC, file0={"ETHUSDC:LONG": {**LA, "tp_legs": TPL3 + [{"level": 2750.0, "price": 2750.3, "w": 1.0}]}})),
    ("add_bad_lines", API("api_manual_bracket_add", LQ(leg="tp", level=2400), account=ACCP, sec=SEC)),
    ("add_quote_fail", API("api_manual_bracket_add", LQ(leg="sl", level=2400), account=ACCP, sec=SEC, books={})),
    ("mod_bad_lines", API("api_manual_bracket_modify", LQ(tp=2650))),
    ("mod_not_lines", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0=RULE_ARM)),
    ("mod_fired", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0={"ETHUSDC:LONG": {**LA, "fired": True}})),
    ("mod_no_position", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0=LINES_ARM, account=ACC(), sec=SEC)),
    ("mod_sl_only", API("api_manual_bracket_modify", LQ(tp=2600, sl=2410), file0=LINES_ARM, account=ACCP, sec=SEC)),
    ("mod_tp_only_keeps_sl_up", API("api_manual_bracket_modify", LQ(tp=2650, sl=2400, slup=1), file0=LINES_ARM, account=ACCP, sec=SEC, script={"place_bracket": [PB_OK]})),
    ("mod_tp_move_cleanup_err", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0=LINES_ARM, account=ACCP, sec=SEC, script={"place_bracket": [PB_FAILTP]})),
    ("mod_tp_replace_failed", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0=LINES_ARM, account=ACCP, sec=SEC, script={"place_bracket": [{"ret": {"placed": False, "reason": "포지션 조회 실패: x"}}]})),
    ("mod_bad_direction", API("api_manual_bracket_modify", LQ(tp=2400, sl=2410), file0=LINES_ARM, account=ACCP, sec=SEC)),
    ("mod_tp_off_ignores_tp", API("api_manual_bracket_modify", LQ(tp=2400, sl=2410), file0={"ETHUSDC:LONG": {**LA, "tp_off": {"at": 1}}}, account=ACCP, sec=SEC)),
    ("mod_tp_legs_tpi", API("api_manual_bracket_modify", LQ(tp=2720, sl=2400, tpi=2), file0=F_LEGS, account=ACCP, sec=SEC, script={"place_tp_ladder": [LAD_OK(2600.0, 2650.0, 2720.0)]})),
    ("mod_tp_legs_same_level", API("api_manual_bracket_modify", LQ(tp=2650.5, sl=2400, tpi=0), file0=F_LEGS, account=ACCP, sec=SEC)),
    ("mod_sl_legs_sli", API("api_manual_bracket_modify", LQ(tp=2600, sl=2370, sli=1), file0=F_SL, account=ACCP, sec=SEC)),
    ("mod_rearmed_race", API("api_manual_bracket_modify", LQ(tp=2650, sl=2410), file0=LINES_ARM, account=ACCP, sec=SEC,
                             script={"place_bracket": [{**PB_OK, "patch_file": {"ETHUSDC:LONG": {**LA, "armed_at": T0}}}], "clear_bracket": [{"ret": {"cancelled": 1, "errors": []}}]})),
]

API_CASES = [
    # 진입 제출
    ("sub_bad_side", API("api_manual_entry_submit", Q(side="UP", confirm=1))),
    ("sub_no_confirm", API("api_manual_entry_submit", Q(side="LONG"))),
    ("sub_bad_asset", API("api_manual_entry_submit", Q(side="LONG", confirm=1, asset="mu"))),
    ("sub_bad_pct", API("api_manual_entry_submit", Q(side="LONG", confirm=1, pct=0))),
    ("sub_bad_pct_text", API("api_manual_entry_submit", Q(side="LONG", confirm=1, pct="abc"))),
    ("sub_limit_without_lines", API("api_manual_entry_submit", Q(side="LONG", confirm=1, mode="limit", limit=2450))),
    ("sub_removed_wait", API("api_manual_entry_submit", Q(side="LONG", confirm=1, mode="wait", wait_min=30))),
    ("sub_removed_wait_min_only", API("api_manual_entry_submit", Q(side="LONG", confirm=1, wait_min=15, tp=2600, sl=2400))),
    ("sub_removed_rule_and_mode", API("api_manual_entry_submit", Q(side="LONG", confirm=1, rule="c", mode="peg", tp=2600, sl=2400))),
    ("sub_mode_limit_not_removed", API("api_manual_entry_submit", Q(side="LONG", confirm=1, mode="limit", limit=2450, rule="", tp=2600, sl=2300), sec=SEC)),
    ("sub_lines_required", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, pct=20), sec=SEC)),
    ("sub_add_on_unlined_409", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, pct=20, tp=2600, sl=2400), sec=SEC, account=ACC(positions=P_ETH))),
    ("sub_exec_off", API("api_manual_entry_submit", Q(side="LONG", confirm=1), exec=False)),
    ("sub_entry_busy", API("api_manual_entry_submit", Q(side="LONG", confirm=1), slots0={"entry": BUSY_E})),
    ("sub_side_exit_busy", API("api_manual_entry_submit", Q(side="LONG", confirm=1), slots0={"side": {"eth": {"phase": "submitting"}}})),
    ("sub_wait_pending", API("api_manual_entry_submit", Q(side="LONG", confirm=1), slots0={"waits": {"eth": LIMIT_W}})),
    ("sub_bad_lines", API("api_manual_entry_submit", Q(side="LONG", confirm=1, tp=2600))),
    ("sub_removed_rule_x", API("api_manual_entry_submit", Q(side="LONG", confirm=1, rule="x"))),
    ("sub_removed_rule_sol", API("api_manual_entry_submit", Q(side="LONG", confirm=1, asset="sol", rule="c"))),
    ("sub_blocked", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=1, pct=0.001, tp=2600, sl=2400), sec=SEC)),
    ("sub_peg_ok", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, pct=20, tp=2600, sl=2400), sec=SEC)),
    ("sub_lev_text_and_big", API("api_manual_entry_submit", Q(side="SHORT", confirm=1, lev="abc", pct=10, tp=2400, sl=2600), sec=SEC)),
    ("sub_lev_999", API("api_manual_entry_submit", Q(side="SHORT", confirm=1, lev=999, pct=10, tp=2400, sl=2600), sec=SEC)),
    ("sub_limit_ok", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2450, tp=2600, sl=2300), sec=SEC)),
    ("sub_limit_second_add_ok", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2440, tp=2700, sl=2300), sec=SEC,
                                    file0=LINES_ARM, account=ACC(positions=P_ETH), slots0={"waits": {"eth": ADD_W}})),
    ("sub_limit_dup_refused", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2450, tp=2700, sl=2300), sec=SEC,
                                  file0=LINES_ARM, account=ACC(positions=P_ETH), slots0={"waits": {"eth": ADD_W}})),
    ("sub_limit_second_not_add", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2440, tp=2700, sl=2300), sec=SEC,
                                     slots0={"waits": {"eth": ADD_W}})),
    ("sub_limit_slot_task_running", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2450, tp=2600, sl=2300), sec=SEC,
                                        tasks={"manual_wait_eth": False, "manual_wait_eth#2": True})),
    ("sub_limit_all_slots_running", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2450, tp=2600, sl=2300), sec=SEC,
                                        tasks={f"manual_wait_{'eth' if n == 1 else f'eth#{n}'}": False for n in range(1, 11)})),
    ("sub_stop_ok", API("api_manual_entry_submit", Q(side="LONG", confirm=1, lev=10, mode="limit", limit=2560, tp=2700, sl=2500, stop=1), sec=SEC)),
    ("prev_ok", API("api_manual_entry_preview", Q(side="LONG", lev=10, pct=50, tp=2600, sl=2400), sec=SEC)),
    ("prev_lines_required", API("api_manual_entry_preview", Q(side="LONG", lev=10, pct=50), sec=SEC)),
    ("prev_removed_rule", API("api_manual_entry_preview", Q(side="LONG", rule="c", tp=2600, sl=2400))),
    ("prev_bad_limit", API("api_manual_entry_preview", Q(side="LONG", limit=-1, tp=2600, sl=2300))),
    ("prev_error_503", API("api_manual_entry_preview", Q(side="LONG"), account=ACC(ok=False))),
    # 청산
    ("xs_bad_side", API("api_manual_exit_submit", Q(side="x", confirm=1))),
    ("xs_no_confirm", API("api_manual_exit_submit", Q(side="LONG"))),
    ("xs_removed_exec_fast", API("api_manual_exit_submit", Q(side="LONG", confirm=1, exec="fast"))),
    ("xs_removed_switch_and_sw", API("api_manual_exit_submit", Q(side="LONG", confirm=1, switch_to="SHORT", sw_pct=50, sw_lev=10))),
    ("xs_removed_sw_only", API("api_manual_exit_submit", Q(side="LONG", confirm=1, sw_rule="c"))),
    ("xs_removed_market", API("api_manual_exit_submit", Q(side="LONG", confirm=1, exec="market"), account=ACC(positions=P_ETH))),
    ("xs_exec_limit_ok", API("api_manual_exit_submit", Q(side="LONG", confirm=1, exec="limit"), account=ACC(positions=P_ETH))),
    ("xs_exec_empty_ok", API("api_manual_exit_submit", Q(side="LONG", confirm=1, exec=""), account=ACC(positions=P_ETH))),
    ("xs_exec_off", API("api_manual_exit_submit", Q(side="LONG", confirm=1), exec=False)),
    ("xs_same_coin_busy", API("api_manual_exit_submit", Q(side="LONG", confirm=1), slots0={"entry": BUSY_E}, account=ACC(positions=P_ETH))),
    ("xs_other_coin_busy_side_slot", API("api_manual_exit_submit", Q(side="LONG", confirm=1), slots0={"entry": {**BUSY_E, "asset": "sol"}}, account=ACC(positions=P_ETH))),
    ("xs_full_ok", API("api_manual_exit_submit", Q(side="LONG", confirm=1), account=ACC(positions=P_ETH), vol={"ETHUSDC": 6.0})),
    ("xs_partial_ok", API("api_manual_exit_submit", Q(side="LONG", confirm=1, pct=50), account=ACC(positions=P_ETH))),
    ("xs_no_position", API("api_manual_exit_submit", Q(side="SHORT", confirm=1), account=ACC(positions=P_ETH))),
    ("xs_blocked", API("api_manual_exit_submit", Q(side="LONG", confirm=1, pct=0.1), account=ACC(positions=P_ETH))),
    ("xs_high_vol_m6", API("api_manual_exit_submit", Q(side="LONG", confirm=1), account=ACC(positions=P_ETH), vol={"ETHUSDC": 40.0}), "M6"),
    ("xs_dust_m7", API("api_manual_exit_submit", Q(side="LONG", confirm=1), account=ACC(positions=[pos("ETHUSDC", "LONG", 0.005, 2400.0, 2450.0)])), "M7"),
    # 지정가 수정
    ("am_no_confirm", API("api_manual_limit_amend", Q())),
    ("am_no_limit", API("api_manual_limit_amend", Q(confirm=1, limit=2440, tp=2600, sl=2300))),
    ("am_partial", API("api_manual_limit_amend", Q(confirm=1, limit=2440, tp=2600, sl=2300), slots0={"waits": {"eth": {**LIMIT_W, "filled": 0.1}}})),
    ("am_pending", API("api_manual_limit_amend", Q(confirm=1, limit=2440, tp=2600, sl=2300), slots0={"waits": {"eth": {**LIMIT_W, "amend": {"price": 1}}}})),
    ("am_bad_limit", API("api_manual_limit_amend", Q(confirm=1, limit=2440), slots0={"waits": {"eth": LIMIT_W}})),
    ("am_moved", API("api_manual_limit_amend", Q(confirm=1, limit=2440, tp=2650, sl=2300), slots0={"waits": {"eth": LIMIT_W}}, sec=SEC,
                     account=ACC(orders=[oo("ETHUSDC", "BUY", 2450.11, 0.3, "dbwtB7")]))),
    ("am_lines_only", API("api_manual_limit_amend", Q(confirm=1, limit=2450, tp=2650, sl=2350), slots0={"waits": {"eth": LIMIT_W}}, sec=SEC)),
    ("am_stop_watching_wid2", API("api_manual_limit_amend", Q(confirm=1, wid=2, limit=2570, tp=2700, sl=2500),
                                  slots0={"waits": {"eth#2": {**LIMIT_W, "order_id": None, "stop": True, "wid": 2, "plan": {**LIMIT_W["plan"], "wid": 2, "stop": True}}}}, sec=SEC)),
    ("am_marketable_400", API("api_manual_limit_amend", Q(confirm=1, limit=2520, tp=2650, sl=2300), slots0={"waits": {"eth": LIMIT_W}}, sec=SEC)),
    # 대기 동작
    ("wa_bad", API("api_manual_wait_action", Q(action="stop", confirm=1))),
    ("wa_no_wait", API("api_manual_wait_action", Q(action="cancel", confirm=1))),
    ("wa_peg_on_wait_mode", API("api_manual_wait_action", Q(action="peg", confirm=1), slots0={"waits": {"eth": {**LIMIT_W, "mode": "wait"}}})),
    ("wa_cancel_ok", API("api_manual_wait_action", Q(action="cancel", confirm=1, asset="eth"), slots0={"waits": {"eth": LIMIT_W}})),
    ("wa_peg_ok", API("api_manual_wait_action", Q(action="peg", confirm=1), slots0={"waits": {"eth": LIMIT_W}})),
    ("wa_peg_not_ready", API("api_manual_wait_action", Q(action="peg", confirm=1), slots0={"waits": {"eth": {**LIMIT_W, "pegged": True}}})),
    ("wa_peg_lines_off", API("api_manual_wait_action", Q(action="peg", confirm=1), slots0={"waits": {"eth": {**LIMIT_W, "plan": {**LIMIT_W["plan"], "bracket": {"tp_level": 2400.0, "sl_level": 2300.0}}}}})),
    ("wa_peg_quote_fail", API("api_manual_wait_action", Q(action="peg", confirm=1), books={}, slots0={"waits": {"eth": LIMIT_W}})),
    ("wa_peg_add_keep_skips_lines", API("api_manual_wait_action", Q(action="peg", confirm=1), slots0={"waits": {"eth": {**ADD_W, "plan": {**ADD_W["plan"], "bracket": {"tp_level": None, "sl_level": None}}}}})),
    ("wa_market_m13", API("api_manual_wait_action", Q(action="market", confirm=1), slots0={"waits": {"eth": LIMIT_W}}), "M13/M4"),
    # 진입 취소
    ("ec_no_confirm", API("api_manual_entry_cancel", Q())),
    ("ec_none", API("api_manual_entry_cancel", Q(confirm=1))),
    ("ec_exit_refused", API("api_manual_entry_cancel", Q(confirm=1), slots0={"entry": {**BUSY_E, "kind": "exit"}})),
    ("ec_ok", API("api_manual_entry_cancel", Q(confirm=1), slots0={"entry": BUSY_E})),
    # 손절 해제·다시 켜기
    ("dis_bad", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDT"))),
    ("dis_not_armed", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDC"))),
    ("dis_already", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {**RULE_ARM["ETHUSDC:LONG"], "sl_off": {"at": 5}}})),
    ("dis_cancel_fails", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0=RULE_ARM, script={"clear_bracket": [{"ret": {"cancelled": 0, "errors": ["418 banned", "x"]}}]})),
    ("dis_ok", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0=RULE_ARM, script={"clear_bracket": [{"ret": {"cancelled": 1, "errors": []}}]})),
    ("dis_rearmed_race", API("api_manual_bracket_disarm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0=RULE_ARM,
                             script={"clear_bracket": [{"ret": {"cancelled": 1, "errors": []}, "patch_file": {"ETHUSDC:LONG": {**RULE_ARM["ETHUSDC:LONG"], "armed_at": T0}}}]})),
    ("re_already", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0=RULE_ARM)),
    ("re_lines_no_sl", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {**LINES_ARM["ETHUSDC:LONG"], "sl_level": None, "sl_off": {"never": True}}})),
    ("re_lines_fired_tp", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {**LINES_ARM["ETHUSDC:LONG"], "fired": "tp", "sl_off": {"at": 1}}})),
    ("re_lines_ok", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {**LINES_ARM["ETHUSDC:LONG"], "fired": True, "sl_off": {"at": 1},
                                                                                                   "sl_legs": [{"level": 2400.0, "fired": True, "w": 1}, {"level": 2380.0, "w": 1}]}})),
    ("re_rule_no_price", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {"symbol": "ETHUSDC", "rule": True, "sl_off": {"at": 1}, "armed_at": 1}})),
    ("re_rule_sl_level_only", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {"symbol": "ETHUSDC", "rule": True, "sl_level": 2374.5, "sl_off": {"at": 1}, "armed_at": 1}})),
    # 2026-11-21 여섯째 조각: 선 핸들러 4개(익절 on/off · 분할 조각 · 선 추가 · 선 수정)
    *LINE_CASES,
    ("re_rule_ok", API("api_manual_bracket_rearm", Q(confirm=1, side="LONG", symbol="ETHUSDC"), file0={"ETHUSDC:LONG": {**RULE_ARM["ETHUSDC:LONG"], "sl_off": {"at": 1}, "placed": {"tp": None}}})),
]


def main():
    out = {"account": [run_account(c) for c in ACCOUNT_CASES], "stream": [run_stream(c) for c in STREAM_CASES]}
    desk = []
    for name, spec, extra in ENTRY_CASES + EXIT_CASES:
        c = {"name": name, **spec, **extra}
        c.setdefault("account", ACC())
        desk.append(asyncio.run(run_desk_case(c)))
    for item in API_CASES:
        name, spec = item[0], item[1]
        c = {"name": name, **spec}
        c.setdefault("account", ACC())
        r = asyncio.run(run_desk_case(c))
        if len(item) > 2:
            r["market_branch"] = item[2]   # 시장가 분기 번호(표시용 -- 2026-11-21 부터 Rust 도 원본과 같이 접수)
        desk.append(r)
    out["desk"] = desk
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    print(f"account {len(out['account'])} · stream {len(out['stream'])} · desk {len(desk)} · {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
