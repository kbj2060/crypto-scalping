"""rtorder 둘째 조각 정답지(2026-11-20) -- **원본 파이썬 함수 + 진짜 aiohttp** 를 로컬 가짜 HTTP 서버(127.0.0.1 임의 포트)에 붙여 돌린다.
서버는 각본대로 응답(200·4xx·5xx·끊김·시간초과)하고 받은 요청(메서드·경로+쿼리 전체·X-MBX-APIKEY)을 적는다.
Rust `cargo test --test http_replay` 가 같은 각본을 자기 가짜 서버에 걸고 Rust 함수를 돌려 «요청 열(글자까지)·돌려준 값·밴 기록»을 대조한다.

안전: 키는 test_key/test_secret(가짜) · .env 안 읽음(환경의 BINANCE_* 는 지움) · 주소는 127.0.0.1 만(아니면 시작 전 중단) ·
밴 기록 파일은 임시 경로로 바꾼다(data/live 를 건드리지 않는다) · 시계는 가짜(time.time = T0 + t, 잠은 t 만 늘린다).
실행(저장소 뿌리): ( ulimit -v 4000000; timeout 300 python3 -I rtorder/tools/gen_http_fixture.py )  → rtorder/tests/fixtures/http_fixture.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time as real_time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 2026-11-21 원본 파이썬 위치 -- 기본 = 이 저장소 · RTORDER_PY_ROOT = 결함 수정본 작업 폴더(읽기만 · 정답지는 이 저장소에 쓴다)
PY_ROOT = Path(os.environ.get("RTORDER_PY_ROOT") or ROOT)
sys.path.insert(0, str(PY_ROOT))
# 감시가 망가진 칸을 알릴 때 텔레그램 모듈을 부른다(진짜는 서버 비밀 설정을 읽고 보낸다) -- 가짜를 먼저 넣는다
_wd = types.ModuleType("scripts.ops_watchdog")
_wd.send_telegram = lambda *_a, **_k: None
sys.modules["scripts.ops_watchdog"] = _wd
for _k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY"):
    os.environ.pop(_k, None)

import aiohttp  # noqa: E402

from live.execution import live_manual_peg_execute_20260912 as X  # noqa: E402
from scripts import binance_ban_guard as BG  # noqa: E402
from scripts import live_binance_account_20260910 as A  # noqa: E402
from scripts import live_manual_peg_entry_20260912 as P  # noqa: E402

KEY, SECRET = "test_key", "test_secret"
TIMEOUT, HANG = 0.5, 0.8   # 시간초과 각본: 서버가 HANG 초 늦게 답한다
REAL_BAN = BG.BAN_FILE
BAN = Path(tempfile.mkdtemp()) / "binance_ban_until"
BG.BAN_FILE = BAN
assert BG.BAN_FILE != REAL_BAN


class Clock:
    T0 = 1_800_000_000.0
    t = 0.0


def _time():
    return Clock.T0 + Clock.t


async def fsleep(sec):
    Clock.t += sec


fake_time = types.SimpleNamespace(time=_time, monotonic=lambda: Clock.t, perf_counter=lambda: Clock.t)
A.time = X.time = fake_time
X.nap = fsleep
X.asyncio = types.SimpleNamespace(sleep=fsleep)


class S:
    auto = None
    given: list = []
    queue: list = []
    log: list = []
    conns: set = set()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):   # 시간초과 각본: 손님이 먼저 끊었다
            pass

    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        S.log.append({"method": self.command, "target": self.path, "key": self.headers.get("X-MBX-APIKEY")})
        S.conns.add(self.client_address)
        if S.auto is not None:   # 2026-11-21 집행 각본: 규칙으로 답하고 준 답을 적는다(Rust 재생은 그 열을 차례로 받는다)
            r = S.auto(self.command, self.path)
            S.given.append(r)
        else:
            r = S.queue.pop(0) if S.queue else {"status": 599, "body": {"msg": "각본 소진"}}
        if r.get("act") == "close":
            self.close_connection = True
            return
        if r.get("act") == "hang":
            real_time.sleep(HANG)
        raw = r["raw"] if "raw" in r else json.dumps(r.get("body"), ensure_ascii=False)
        b = raw.encode()
        self.send_response(r.get("status", 200))
        self.send_header("Content-Type", r.get("ctype", "application/json"))
        for k, v in (r.get("headers") or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_GET = do_POST = do_DELETE = do_PUT = _go


SRV = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=SRV.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{SRV.server_address[1]}"
assert BASE.startswith("http://127.0.0.1:")
X.FAPI = A.FAPI = P.FAPI = BASE

SYM = "ETHUSDC"
COMMON = {"symbol": SYM, "side": "BUY", "positionSide": "LONG"}


def ok(body, **kw):
    return {"status": 200, "body": body, **kw}


def err(status, code, msg, **kw):
    return {"status": status, "body": {"code": code, "msg": msg}, **kw}


TIME = ok({"serverTime": 1_800_000_000_037})   # 오프셋 +37ms
CLOSE, HANGR = {"act": "close"}, {"act": "hang"}
NEW = lambda i, q="0": ok({"orderId": i, "status": "NEW", "executedQty": q})  # noqa: E731
FILLED = lambda i, q: ok({"orderId": i, "status": "FILLED", "executedQty": q})  # noqa: E731
BOOK = lambda b, a: ok({"symbol": SYM, "bidPrice": b, "askPrice": a})  # noqa: E731
NOTFOUND = err(400, -2013, "Order does not exist.")
UNKNOWN = err(400, -2011, "Unknown order sent.")
M5022 = err(400, -5022, "Due to the order could not be executed as maker, the Post Only order will be rejected. The order will not be recorded in the order history")
POS = lambda *legs: ok([{"symbol": SYM, "positionSide": ps, "positionAmt": amt, "leverage": "5", "marginType": "cross"} for ps, amt in legs])  # noqa: E731


def jsonable(v):
    if isinstance(v, tuple):
        return [jsonable(x) for x in v]
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v


async def call(session, fn, a):
    if fn == "sign":
        return A._sign(dict(a["params"]), a["secret"], a["offset"])
    if fn == "signed":
        return await X.signed(session, a["method"], a["path"], dict(a["params"]), KEY, SECRET, a["offset"])
    if fn == "clock_offset":
        return await A._clock_offset(session)
    if fn == "maker_price":
        return await X.maker_price(session, a["side"], a["symbol"])
    if fn == "load_filters":
        return await P.load_filters(session, a["symbol"])
    if fn == "ensure_closed":
        return await X.ensure_closed(session, COMMON, a["oid"], KEY, SECRET, a["offset"])
    if fn == "resolve_unknown_post":
        return await X.resolve_unknown_post(session, SYM, a["cid"], a["order"], KEY, SECRET, a["offset"])
    if fn == "fill_maker":
        st = {}
        out = await X.fill_maker(session, common=dict(a.get("common", COMMON)), price=a["price"], total=a["total"],
                                 deadline=Clock.t + 120.0, state=st, key=KEY, secret=SECRET, offset=a["offset"])
        return {"out": out, "state": {k: st.get(k) for k in ("filled", "repegs", "limit_price", "order_id")}}
    if fn in ("current_leverage", "ensure_leverage"):
        extra = {"target": a["target"]} if fn == "ensure_leverage" else {}
        return await getattr(X, fn)(session, SYM, key=KEY, secret=SECRET, offset=a["offset"], **extra)
    if fn == "cancel_bracket":
        return await X.cancel_bracket(session, a.get("symbol", SYM), a["pside"], KEY, SECRET, a["offset"],
                                      only_sl=a.get("only_sl", False), only_tp=a.get("only_tp", False))
    if fn == "clear_bracket":
        return await X.clear_bracket(session, SYM, a["pside"], only_sl=a.get("only_sl", False), only_tp=a.get("only_tp", False))
    if fn == "margin_type":
        return await X.margin_type(session, SYM)
    if fn == "commission_bp":
        return await X.commission_bp(session, SYM)
    if fn == "place_add_order":
        return await X.place_add_order(session, SYM, a["pside"], a["price"], a["qty"])
    if fn == "place_stop":
        return await X.place_stop(session, SYM, a["pside"], a["price"])
    if fn == "place_bracket":
        return await X.place_bracket(session, a["bracket"], SYM, a["pside"], strict=a.get("strict", False))
    if fn == "place_tp_ladder":
        return await X.place_tp_ladder(session, a["legs"], SYM, a["pside"])
    if fn == "cancel_wait_orphans":
        return await X.cancel_wait_orphans(session, a["symbols"])
    if fn == "cancel_exit_orders":
        return await X.cancel_exit_orders(session, SYM, a["pside"])
    if fn in ("run_entry", "run_exit", "run_entry_wait", "run_entry_stop"):
        st = dict(a.get("state0") or {})
        if fn == "run_entry_wait":
            await X.run_entry_wait(session, dict(a["plan"]), st, a["wait_sec"])
        elif fn == "run_entry_stop":
            await X.run_entry_stop(session, dict(a["plan"]), st, lambda since: False, max_sec=a.get("max_sec", 60.0))
        else:
            await getattr(X, fn)(session, dict(a["plan"]), st)
        return {k: st.get(k) for k in EXEC_KEYS}
    raise KeyError(fn)


EXEC_KEYS = ("phase", "filled", "quantity", "taker_qty", "taker_order_id", "taker_reason", "error", "repegs", "cancel_reason")


def auto_book(market, fill_get="0", fill_status="NEW"):
    """집행 각본 규칙: 시계 · 호가(안 움직임) · GTX 접수 NEW · 조회 = fill_status/fill_get · 취소 = CANCELED(같은 체결량) · 시장가 = market."""
    n = {"id": 100}

    def r(method, path):
        p = path.split("?")[0]
        q = dict(x.split("=", 1) for x in path.split("?", 1)[1].split("&")) if "?" in path else {}
        if p.endswith("/time"):
            return TIME
        if p.endswith("/bookTicker"):
            return BOOK("2500.0", "2500.1")
        if method == "POST" and q.get("type") == "MARKET":
            return market
        if method == "POST":
            n["id"] += 1
            return NEW(n["id"])
        oid = int(q.get("orderId") or 0)
        if method == "GET":
            return ok({"orderId": oid, "status": "CANCELED" if oid in n.setdefault("gone", set()) else fill_status, "executedQty": fill_get})
        if method == "DELETE":
            n.setdefault("gone", set()).add(oid)
            return ok({"orderId": oid, "status": "CANCELED", "executedQty": fill_get})
        return {"status": 599, "body": {"msg": "규칙 없음"}}
    return r


SCN: list[dict] = []


def scn(name, calls, responses=(), keys=True, auto=None):
    SCN.append({"name": name, "calls": calls if isinstance(calls, list) else [calls], "responses": list(responses), "keys": keys, "auto": auto})


def c(fn, advance=0.0, **a):
    return {"fn": fn, "args": a, "advance": advance}


# ── 1. 서명(순수) ────────────────────────────────────────────────────────────
for i, (params, secret, t, off) in enumerate([
    ([["symbol", SYM]], SECRET, 0.0, 0),
    ([["symbol", SYM], ["side", "BUY"], ["positionSide", "LONG"], ["type", "LIMIT"], ["timeInForce", "GTX"], ["price", 2500.01], ["quantity", 1.234], ["newClientOrderId", "dbpgL1800000000000"]], SECRET, 0.0, 37),
    ([["price", 0.1 + 0.2], ["quantity", 1e-05], ["x", 1e16], ["y", -0.0], ["z", 123456789012345680.0], ["w", 2.5e-7]], SECRET, 12.3456, -1000),
    ([["orderId", None], ["leverage", 20], ["closePosition", "true"], ["flag", True]], SECRET, 0.001, 5),
    ([["sp ace", "a b+c&d=e/f?g~h_i.j-k"], ["한글", "가격 2,500"], ["pct", "100%"], ["emoji", "✓"]], SECRET, 0.0, 0),
    ([["symbol", SYM]], "다른 비밀값 with space", 99.999, 1),
    ([], SECRET, 0.0, 0),
]):
    scn(f"sign {i}", c("sign", advance=t, params=params, secret=secret, offset=off))

# ── 2. signed 응답 → 호출부가 받는 값 ────────────────────────────────────────
GETP = [["symbol", SYM], ["orderId", 77]]
for name, r in [
    ("200 dict", ok({"orderId": 77, "status": "NEW", "executedQty": "0.000", "price": "2500.01"})),
    ("200 list", ok([{"a": 1}, {"b": [1.5, None, True]}])),
    ("200 charset", ok({"x": 1}, ctype="application/json; charset=utf-8")),
    ("200 empty", {"status": 200, "raw": ""}),
    ("200 bad json", {"status": 200, "raw": "{nope"}),
    ("400 -5022", M5022),
    ("400 no code", {"status": 400, "body": {"msg": "Bad thing"}}),
    ("400 code null", {"status": 400, "body": {"code": None, "msg": "m"}}),
    ("400 no msg", {"status": 400, "body": {"code": -1, "foo": "it's", "bar": [1, 2.5, None, False], "q": "a\"b", "k": "한글\n"}}),
    ("400 msg null", {"status": 400, "body": {"code": -1, "msg": None}}),
    ("400 list", {"status": 400, "body": [1, 2]}),
    ("400 empty", {"status": 400, "raw": ""}),
    ("503 html", {"status": 503, "raw": "<html>busy</html>", "ctype": "text/html"}),
    ("503 json", {"status": 503, "body": {"msg": "Service Unavailable"}}),
    ("408 -1007", err(408, -1007, "Timeout waiting for response from backend server. Send status unknown; execution status unknown.")),
    ("500 -1006", err(500, -1006, "An unexpected response was received from the message bus. Execution status unknown.")),
    ("418 banned", err(418, -1003, "Way too many requests; IP(1.2.3.4) banned until 1800000600000. Please use the websocket for live updates to avoid bans.")),
    ("429 retry-after", err(429, -1003, "Too many requests.", headers={"Retry-After": "5"})),
    ("429 no header", err(429, -1003, "Too many requests.")),
    ("시간초과", HANGR),
]:
    for m in (["GET"] if name not in ("400 -5022",) else ["POST", "DELETE"]):
        scn(f"signed {name} {m}", c("signed", method=m, path="/fapi/v1/order", params=GETP, offset=37), [r])
# aiohttp 는 GET·DELETE·PUT(멱등)이 연결 뒤 끊기면 **한 번 다시 보낸다**(client.py IDEMPOTENT_METHODS) -- POST 는 안 다시 보낸다
for m, rs in [("GET", [CLOSE, CLOSE]), ("DELETE", [CLOSE, CLOSE]), ("PUT", [CLOSE, CLOSE]), ("POST", [CLOSE]), ("GET", [CLOSE, ok({"orderId": 1})]), ("DELETE", [CLOSE, UNKNOWN])]:
    scn(f"signed 끊김 {m} {len(rs)}", c("signed", method=m, path="/fapi/v1/order", params=GETP, offset=0), rs)

scn("clock_offset ok", c("clock_offset"), [TIME])
scn("clock_offset html", c("clock_offset"), [{"status": 502, "raw": "x", "ctype": "text/html"}])
scn("clock_offset 끊김", c("clock_offset"), [CLOSE, CLOSE])
scn("maker_price ok", c("maker_price", side="SELL", symbol=SYM), [BOOK("2500.00", "2500.01")])
scn("maker_price 400", c("maker_price", side="BUY", symbol=SYM), [err(400, -1121, "Invalid symbol.")])
EXI = ok({"symbols": [{"symbol": "ETHUSDT"}, {"symbol": SYM, "filters": [
    {"filterType": "PRICE_FILTER", "tickSize": "0.01"}, {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
    {"filterType": "MIN_NOTIONAL", "notional": "20"}]}]})
scn("load_filters 캐시", [c("load_filters", symbol=SYM), c("load_filters", symbol=SYM)], [EXI])
scn("load_filters 없음", c("load_filters", symbol="NOPE"), [EXI])

# ── 3. 집행 요청 모양(원본 fill_maker·ensure_closed·resolve_unknown_post) ─────────
scn("ensure_closed 두 번째에 확인", c("ensure_closed", oid=77, offset=37), [ok({"orderId": 77, "status": "NEW"}), NEW(77), UNKNOWN, ok({"orderId": 77, "status": "CANCELED", "executedQty": "0.4"})])
scn("ensure_closed 확인 못 함", c("ensure_closed", oid=77, offset=37), [err(500, -1000, "x"), NEW(77)] * 3)
scn("ensure_closed oid None", c("ensure_closed", oid=None, offset=0), [UNKNOWN, NOTFOUND] * 3)
scn("resolve 4xx 그대로", c("resolve_unknown_post", cid="dbpgL1", order={"__error__": "400 [-2019] Margin is insufficient."}, offset=0), [])
scn("resolve 둘째에 찾음", c("resolve_unknown_post", cid="dbpgL1", order={"__error__": "503 Service Unavailable"}, offset=0), [NOTFOUND, NEW(9)])
scn("resolve 끝내 없음", c("resolve_unknown_post", cid="dbwtS1", order={"__error__": "TimeoutError: "}, offset=0), [NOTFOUND, CLOSE, NOTFOUND, NOTFOUND, UNKNOWN])
FM = dict(price=2500.0, total=2.0, offset=37)
scn("fill_maker 정상", c("fill_maker", **FM), [NEW(101), NEW(101), BOOK("2500.00", "2500.01"), FILLED(101, "2.0")])
scn("fill_maker 앞지름 1번", c("fill_maker", **FM), [NEW(101), NEW(101), BOOK("2500.05", "2500.06"), ok({"orderId": 101, "status": "CANCELED"}),
                                                    ok({"orderId": 101, "status": "CANCELED", "executedQty": "0.5"}), NEW(102), FILLED(102, "1.5")])
scn("fill_maker -5022 뒤", c("fill_maker", **FM), [M5022, BOOK("2499.99", "2500.00"), NEW(103), FILLED(103, "2")])
scn("fill_maker 503 접수됨", c("fill_maker", **FM), [ok({"msg": "Service Unavailable"}, status=503), NEW(104), FILLED(104, "2.0")])
scn("fill_maker 끊김 끝내 모름", c("fill_maker", **FM), [CLOSE, NOTFOUND, NOTFOUND, NOTFOUND, UNKNOWN])
scn("fill_maker 시간초과 접수됨", c("fill_maker", **FM), [HANGR, NEW(105), FILLED(105, "2.0")])
scn("fill_maker 취소 확인 못 함", c("fill_maker", **FM), [NEW(101), ok({"orderId": 101, "status": "PARTIALLY_FILLED", "executedQty": "0.3"}),
                                                       BOOK("2500.05", "2500.06")] + [err(500, -1000, "x"), NEW(101, "0.3")] * 3)
scn("fill_maker 매도 조회 실패 재시도", c("fill_maker", common={"symbol": SYM, "side": "SELL", "positionSide": "LONG"}, price=2500.01, total=0.5, offset=0),
    [NEW(106), CLOSE, CLOSE, err(502, -1001, "Internal error"), FILLED(106, "0.5")])

# ── 4. 레버리지·마진·수수료 ─────────────────────────────────────────────────
scn("ensure_leverage 목표 없음", c("ensure_leverage", target=0, offset=0))
scn("ensure_leverage 이미 같음", c("ensure_leverage", target=5, offset=37), [POS(("LONG", "0.0"), ("SHORT", "0.0"))])
scn("ensure_leverage 바꿈", c("ensure_leverage", target=20, offset=37), [POS(("LONG", "0.0")), ok({"symbol": SYM, "leverage": 20, "maxNotionalValue": "1000"})])
scn("ensure_leverage 응답에 값 없음", c("ensure_leverage", target=7, offset=37), [POS(("LONG", "0.0")), ok({"symbol": SYM})])
scn("ensure_leverage 거부", c("ensure_leverage", target=3, offset=37), [POS(("LONG", "1.0")), err(400, -4028, "Leverage 3 is not valid")])
scn("ensure_leverage 조회 실패", c("ensure_leverage", target=10, offset=37), [CLOSE, CLOSE, ok({"leverage": "10"})])
scn("current_leverage 빈 목록", c("current_leverage", offset=0), [ok([])])
scn("current_leverage 0", c("current_leverage", offset=0), [ok([{"leverage": "0"}])])
scn("margin_type 키 없음", c("margin_type"), keys=False)
scn("margin_type cross", c("margin_type"), [TIME, POS(("LONG", "0"), ("SHORT", "0"))])
scn("margin_type 섞임", c("margin_type"), [TIME, ok([{"marginType": "cross"}, {"marginType": "ISOLATED"}])])
scn("margin_type 빈칸", c("margin_type"), [TIME, ok([{"marginType": None}])])
scn("margin_type 실패", c("margin_type"), [TIME, err(401, -2015, "Invalid API-key, IP, or permissions for action.")])
FEE = ok({"symbol": SYM, "makerCommissionRate": "0.000200", "takerCommissionRate": "0.000500"})
scn("commission_bp 캐시·TTL", [c("commission_bp"), c("commission_bp", advance=3599.0), c("commission_bp", advance=1.5),
                               c("commission_bp", advance=299.0), c("commission_bp", advance=2.0)],
    [TIME, FEE, TIME, err(400, -1000, "boom"), TIME, ok({"symbol": SYM, "makerCommissionRate": "0", "takerCommissionRate": "0.0004"})])
scn("commission_bp 처음부터 실패", [c("commission_bp"), c("commission_bp", advance=301.0)], [TIME, CLOSE, TIME, ok({"makerCommissionRate": "x"})])
scn("commission_bp 키 없음", c("commission_bp"), keys=False)

# ── 5. 브래킷 ────────────────────────────────────────────────────────────────
OPENS = ok([
    {"orderId": 1, "clientOrderId": "dbtpL1800000000000", "positionSide": "LONG"},
    {"orderId": 2, "clientOrderId": "dbtpS1800000000000", "positionSide": "SHORT"},
    {"orderId": 3, "clientOrderId": "dbadL1800000000000", "positionSide": "LONG"},
    {"orderId": 4, "clientOrderId": "aos_user_app", "positionSide": "LONG"},
    {"orderId": 5, "clientOrderId": None},
    {"orderId": 6, "clientOrderId": "dbtpL1800000000001", "positionSide": "LONG"},
])
ALGOS = ok([
    {"algoId": 11, "clientAlgoId": "dbslL1", "symbol": SYM},
    {"algoId": 12, "clientAlgoId": "dbslL2", "symbol": "ETHUSDT"},
    {"algoId": 13, "clientAlgoId": "dbslS1", "symbol": SYM},
    {"algoId": 14, "clientAlgoId": "dbslL3"},
])
CB = dict(pside="LONG", offset=37)
scn("cancel_bracket 전부", c("cancel_bracket", **CB), [OPENS, ok({}), ok({}), ok({}), ALGOS, ok({}), ok({})])
scn("cancel_bracket 삭제 실패 섞임", c("cancel_bracket", **CB), [OPENS, UNKNOWN, ok({}), CLOSE, CLOSE, ALGOS, err(400, -20130, "algo gone"), ok({})])
scn("cancel_bracket 목록 실패", c("cancel_bracket", **CB), [err(500, -1001, "x"), {"status": 200, "body": "not a list"}])
scn("cancel_bracket only_sl", c("cancel_bracket", only_sl=True, **CB), [ALGOS, ok({}), ok({})])
scn("cancel_bracket only_tp", c("cancel_bracket", only_tp=True, **CB), [OPENS, ok({}), ok({})])
scn("cancel_bracket only_tp 목록 실패", c("cancel_bracket", only_tp=True, **CB), [CLOSE, CLOSE])
scn("cancel_bracket 숏", c("cancel_bracket", pside="SHORT", offset=0), [OPENS, ok({}), ALGOS, ok({})])
scn("clear_bracket 키 없음", c("clear_bracket", pside="LONG"), keys=False)
scn("clear_bracket", c("clear_bracket", pside="LONG", only_sl=True), [TIME, ok([]), ])
scn("place_add_order 키 없음", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), keys=False)
scn("place_add_order 접수", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, NEW(201)])
scn("place_add_order 숏 거부", c("place_add_order", pside="SHORT", price=2518.75, qty=0.123), [TIME, err(400, -2019, "Margin is insufficient.")])
scn("place_add_order 시간초과", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, HANGR])
# 2026-11-21 결함 수정본: 불명 응답 = 표식으로 3번 찾기 → 찾으면 걸림 · 끝내 못 찾으면 표식 DELETE · 확정 거부는 그대로 · 200 빈 본문 = 불명
scn("place_add_order 빈 본문 → 찾음", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, ok(None), NOTFOUND, ok({"orderId": 77, "status": "NEW", "executedQty": "0"})])
scn("place_add_order 5xx → 끝내 모름", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, err(502, -1001, "x"), NOTFOUND, NOTFOUND, NOTFOUND, UNKNOWN])
scn("place_add_order -1007 → 찾음", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, err(408, -1007, "Timeout waiting for response from backend server."), ok({"orderId": 78, "status": "NEW", "executedQty": "0"})])
scn("place_add_order 4xx 확정 거부(조회 없음)", c("place_add_order", pside="LONG", price=2481.25, qty=0.5), [TIME, err(400, -2019, "Margin is insufficient.")])
scn("place_stop 키 없음", c("place_stop", pside="LONG", price=2375.0), keys=False)
scn("place_stop 조회 실패", c("place_stop", pside="LONG", price=2375.0), [TIME, err(500, -1001, "x")])
scn("place_stop 포지션 없음", c("place_stop", pside="LONG", price=2375.0), [TIME, POS(("LONG", "0.000"), ("SHORT", "-1.0"))])
scn("place_stop 정리 실패", c("place_stop", pside="LONG", price=2375.0), [TIME, POS(("LONG", "1.0")), CLOSE, CLOSE])
scn("place_stop 정상", c("place_stop", pside="LONG", price=2375.0), [TIME, POS(("LONG", "1.0")), ALGOS, ok({}), ok({}), ok({"algoId": 31})])
scn("place_stop 숏 -2021", c("place_stop", pside="SHORT", price=2625.5), [TIME, POS(("SHORT", "-2.0")), ok([]), err(400, -2021, "Order would immediately trigger.")])
BR = {"tp_price": 2600.5, "backstop_price": 2375.0}
scn("place_bracket 키 없음", c("place_bracket", bracket=BR, pside="LONG"), keys=False)
scn("place_bracket 조회 실패", c("place_bracket", bracket=BR, pside="LONG"), [TIME, CLOSE, CLOSE])
scn("place_bracket 포지션 없음", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("SHORT", "-1"))])
scn("place_bracket 정상", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("LONG", "0.7"), ("LONG", "0.3000001")), ok([]), ok([]), NEW(301), ok({"algoId": 41})])
scn("place_bracket 익절 실패", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("LONG", "1")), ok([]), ok([]), M5022, ok({"algoId": 42})])
scn("place_bracket 익절 불명 → 두 번째 조회에서 찾음", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("LONG", "1")), ok([]), ok([]), CLOSE, NOTFOUND,
    ok({"orderId": 301, "status": "NEW", "executedQty": "0"}), ok({"algoId": 42})])   # 2026-11-21 결함 수정본(resolve_resting)
scn("place_bracket 익절 불명 → 끝내 모름 · 표식 취소", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("LONG", "1")), ok([]), ok([]), CLOSE, NOTFOUND, NOTFOUND, NOTFOUND, UNKNOWN, ok({"algoId": 42})])
scn("place_bracket strict 정리 실패", c("place_bracket", bracket=BR, pside="LONG", strict=True), [TIME, POS(("LONG", "1")), OPENS, UNKNOWN, ok({}), ok({}), ok([])])
scn("place_bracket 정리 실패 그래도 건다", c("place_bracket", bracket=BR, pside="LONG"), [TIME, POS(("LONG", "1")), CLOSE, CLOSE, ok([]), NEW(302), ok({"algoId": 43})])
scn("place_bracket 숏 익절만", c("place_bracket", bracket={"tp_price": 2400.25, "backstop_price": None}, pside="SHORT"), [TIME, POS(("SHORT", "-1.5")), ok([]), ok([]), NEW(303)])
scn("place_bracket 스탑만", c("place_bracket", bracket={"tp_price": 0, "backstop_price": 2375}, pside="LONG"), [TIME, POS(("LONG", "1")), ok([]), ok([]), ok({"algoId": 44})])
scn("place_bracket 둘 다 없음", c("place_bracket", bracket={}, pside="LONG"), [TIME, POS(("LONG", "1")), ok([]), ok([])])
LEGS = [{"price": 2600.0, "w": 1.0}, {"price": 2650.0, "w": 1.0}]
TPL = dict(pside="LONG")
scn("place_tp_ladder 키 없음", c("place_tp_ladder", legs=LEGS, **TPL), keys=False)
scn("place_tp_ladder 조회 실패", c("place_tp_ladder", legs=LEGS, **TPL), [TIME, {"status": 200, "body": "x"}])
scn("place_tp_ladder 포지션 없음", c("place_tp_ladder", legs=LEGS, **TPL), [TIME, POS(("LONG", "0"))])
scn("place_tp_ladder 정리 실패", c("place_tp_ladder", legs=LEGS, **TPL), [TIME, POS(("LONG", "1")), OPENS, ok({}), UNKNOWN])
scn("place_tp_ladder 정상", c("place_tp_ladder", legs=LEGS, **TPL), [TIME, POS(("LONG", "1.001")), ok([]), EXI, NEW(401), M5022])
scn("place_tp_ladder 걸린 조각 사라짐(체결·취소)", c("place_tp_ladder", **TPL, legs=[
    {"price": 2600.0, "w": 1, "resting": True, "cid": "dbtpL1800000000000", "tag": "가"},
    {"price": 2650.0, "w": 1, "resting": True, "cid": "dbtpL1800000000009"},
    {"price": 2700.0, "w": 2, "resting": True, "cid": "dbtpL1800000000008"},
    {"price": 2750.0, "w": 1, "filled": True, "cid": "old", "qty": 0.2}]),
    [TIME, POS(("LONG", "0.8")), OPENS, ok({}), ok({}), ok({"status": "FILLED"}), ok({"status": "CANCELED"}), EXI, NEW(402), NEW(403)])
scn("place_tp_ladder 전부 체결", c("place_tp_ladder", **TPL, legs=[{"price": 2600.0, "w": 1, "filled": True}]),
    [TIME, POS(("LONG", "0.5")), ok([]), EXI])
scn("place_tp_ladder 먼지 조각 합침", c("place_tp_ladder", pside="SHORT", legs=[{"price": 2400.0, "w": 0.01}, {"price": 2300.0, "w": 0.99}]),
    [TIME, POS(("SHORT", "-0.5")), ok([]), EXI, NEW(404)])
scn("place_tp_ladder 필터 실패(예외)", c("place_tp_ladder", legs=LEGS, **TPL), [TIME, POS(("LONG", "1")), ok([]), {"status": 200, "raw": "x", "ctype": "text/plain"}])
scn("cancel_wait_orphans 키 없음", c("cancel_wait_orphans", symbols=[SYM]), keys=False)
scn("cancel_wait_orphans", c("cancel_wait_orphans", symbols=[SYM, "ETHUSDT", "SOLUSDC"]), [TIME,
    ok([{"orderId": 7, "clientOrderId": "dbwtL1", "positionSide": "LONG", "executedQty": "0.25", "origQty": "1.0"},
        {"orderId": 8, "clientOrderId": "dbpgS1", "positionSide": "SHORT", "executedQty": None},
        {"orderId": 9, "clientOrderId": "dbtpL1"}, {"orderId": 10, "clientOrderId": "dbwtX"}]),
    ok({}), ok({}), UNKNOWN, err(500, -1001, "x"), ok([{"orderId": 11, "clientOrderId": "dbwtS2", "origQty": "3"}]), CLOSE, CLOSE])


# ── 2026-11-21 결함 수정본: 재발동 전 앞선 청산 따라가기 정리(cancel_exit_orders) ─────────────────────
PG = lambda i, cid, side: {"orderId": i, "clientOrderId": cid, "side": side, "positionSide": "LONG", "type": "LIMIT"}   # noqa: E731
OTHERS = [PG(5, "dbpgL1", "BUY"), PG(6, "dbtpL1", "SELL"), PG(8, "dbwtL1", "SELL"), PG(9, "aos_x", "SELL")]
scn("cancel_exit_orders 없음(조회 1번)", c("cancel_exit_orders", pside="LONG"), [TIME, ok(OTHERS)])
scn("cancel_exit_orders 지우고 확인", c("cancel_exit_orders", pside="LONG"), [TIME, ok([PG(7, "dbpgL18", "SELL"), *OTHERS]), UNKNOWN, ok(OTHERS)])
scn("cancel_exit_orders 다시 읽어도 남음", c("cancel_exit_orders", pside="LONG"), [TIME, ok([PG(7, "dbpgL18", "SELL")]), ok({}), ok([PG(7, "dbpgL18", "SELL")])])
scn("cancel_exit_orders 목록 실패 418", c("cancel_exit_orders", pside="LONG"), [TIME, err(418, -1003, "banned until 1800000999000")])
scn("cancel_exit_orders 숏", c("cancel_exit_orders", pside="SHORT"), [TIME, ok([PG(3, "dbpgS1", "BUY"), PG(4, "dbpgS2", "SELL")]), ok({}), ok([PG(4, "dbpgS2", "SELL")])])
scn("cancel_exit_orders 키 없음", c("cancel_exit_orders", pside="LONG"), keys=False)


# ── 집행 시장가 다리(2026-11-21 사용자 결정 «원본 동작 유지» -- M1·M2·M3·M5·M6·M9) ─────────────────────
EP = {"symbol": SYM, "side": "BUY", "positionSide": "LONG", "type": "LIMIT", "price": 2500.0, "quantity": 2.0, "target_leverage": 0}
XP = {"symbol": SYM, "side": "SELL", "positionSide": "LONG", "type": "LIMIT", "price": 2500.1, "quantity": 2.0, "fallback_after_sec": 30.0, "vol_bpm": 9.0}
MK = lambda q: FILLED(9001, q)   # noqa: E731
scn("집행 M1 따라가기 마감 → 시장가", c("run_entry", plan=EP), auto=auto_book(MK("2")))
scn("집행 M1 부분 체결 뒤 시장가", c("run_entry", plan=EP), auto=auto_book(MK("1.5"), fill_get="0.5", fill_status="PARTIALLY_FILLED"))
scn("집행 M1 시장가 ACK(체결량 0)", c("run_entry", plan=EP), auto=auto_book(MK("0")))
scn("집행 M1 시장가 거부", c("run_entry", plan=EP), auto=auto_book(err(400, -2019, "Margin is insufficient.")))
scn("집행 M1 시장가 불명(끊김)", c("run_entry", plan=EP), auto=auto_book(CLOSE))
scn("집행 M2 시장가 계획", c("run_entry", plan={**EP, "type": "MARKET"}), auto=auto_book(MK("2")))
scn("집행 M6 청산 시장가 계획", c("run_exit", plan={k: v for k, v in {**XP, "type": "MARKET", "market_reason": "변동성"}.items() if k != "price"}), auto=auto_book(MK("2")))
scn("집행 M6 청산 시장가 거부", c("run_exit", plan={k: v for k, v in {**XP, "type": "MARKET"}.items() if k != "price"}), auto=auto_book(err(400, -2022, "ReduceOnly Order is rejected.")))
scn("집행 M9 청산 마감 잔량 시장가", c("run_exit", plan=XP), auto=auto_book(MK("0")))
scn("집행 M9 부분 뒤 잔량 시장가", c("run_exit", plan=XP), auto=auto_book(MK("1.2"), fill_get="0.8", fill_status="PARTIALLY_FILLED"))
scn("집행 M3 기다리기 마감 시장가", c("run_entry_wait", plan=EP, wait_sec=60.0), auto=auto_book(MK("2")))
scn("집행 M3 지금 시장가(사용자)", c("run_entry_wait", plan=EP, wait_sec=60.0, state0={"action": "market"}), auto=auto_book(MK("2")))
scn("집행 M5 감시 진입 지금 시장가", c("run_entry_stop", plan={**EP, "limit": {"usdt": 2600.0}}, state0={"action": "market"}), auto=auto_book(MK("2")))


async def main():
    out = []
    for s in SCN:
        S.queue, S.log, S.conns = [dict(r) for r in s["responses"]], [], set()
        S.auto, S.given = s.pop("auto", None), []
        Clock.t = 0.0
        X._fee_cache.clear()
        P._filters.clear()
        BAN.unlink(missing_ok=True)
        if s["keys"]:
            os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"] = KEY, SECRET
        else:
            os.environ.pop("BINANCE_API_KEY", None)
            os.environ.pop("BINANCE_SECRET_KEY", None)
        results = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            for cl in s["calls"]:
                Clock.t += cl["advance"]
                try:
                    r = await call(session, cl["fn"], cl["args"])
                except Exception as e:  # noqa: BLE001 -- 원본이 예외를 내는 경로도 정답지다(종류·문구)
                    r = {"__raise__": f"{type(e).__name__}: {e}"}
                results.append(jsonable(r))
        assert not S.queue, (s["name"], "각본이 남았다", S.queue)
        if S.auto is not None:
            s["responses"] = S.given
        assert all(r.get("act") or True for r in s["responses"])
        out.append({**s, "requests": S.log, "results": results, "ban": BAN.read_text() if BAN.exists() else None,
                    "conns": len(S.conns), "post_only": [X.is_post_only_reject(r.get("__error__", "")) if isinstance(r, dict) else False for r in results]})
        print(f"{s['name']}: 요청 {len(S.log)} · 결과 {json.dumps(results, ensure_ascii=False)[:150]}", flush=True)
    os.environ.pop("BINANCE_API_KEY", None)
    os.environ.pop("BINANCE_SECRET_KEY", None)
    assert not REAL_BAN.exists() or REAL_BAN.stat().st_mtime < real_time.time() - 5, "진짜 밴 파일을 건드렸다"
    dst = ROOT / "rtorder" / "tests" / "fixtures" / "http_fixture.json"
    dst.write_text(json.dumps({"base_note": "포트는 실행마다 다르다 -- 요청 대상은 경로+쿼리만 적는다", "timeout": TIMEOUT, "hang": HANG,
                               "t0": Clock.T0, "scenarios": out}, ensure_ascii=False, indent=0).replace(BASE, "{BASE}"))
    print(f"시나리오 {len(out)} · 요청 {sum(len(s['requests']) for s in out)} · {dst.stat().st_size} bytes")


asyncio.run(main())
SRV.shutdown()
