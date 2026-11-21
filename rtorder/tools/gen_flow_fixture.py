"""rtorder 일곱째 조각 정답지(2026-11-21) -- **원본 «전체 흐름»**: `dashboard/orders.py` 의 진짜 `entry_then_bracket` · `bracket_tick` 이
진짜 집행·브래킷 함수(live_manual_peg_execute: run_entry·run_entry_limit·run_exit·place_bracket·clear_bracket·cancel_exit_orders·ensure_leverage)와
진짜 aiohttp 로 127.0.0.1 **모의 거래소**(주문·체결·포지션을 규칙으로 흉내)에 붙어 돈다.
각 걸음(진입 · 감시 틱)마다 «나간 요청 열(서명 URL 글자까지) · 모의 거래소가 준 응답 열 · 상태 칸(진입) · 무장 파일 · 지정가 복구 파일 · 가짜 의존성 호출 · 로그»를 적는다.
Rust `cargo test --test flow_fixture` 가 같은 응답 열을 자기 가짜 서버에 걸고 Rust 실행기(runner::entry_then_bracket)·감시(Watcher::bracket_tick)를 돌려 대조한다.

가짜로 두는 것(양쪽 같은 값): 계좌(swr_cached -- 모의 거래소 포지션에서 만든 값) · 5분봉(빈 목록) · 1초 체결가 · 청산 계획 조립(assemble_exit_plan) ·
지정가 재조정의 호가(fetch_binance_json)·필터(load_filters) · 실행 스위치(켬) · 주문 잠금(안 잠김) · 시계(T0 + t, 잠은 t 만 늘림).

안전: 키는 test_key/test_secret(가짜) · .env 안 읽음(BINANCE_* 지움) · 주소는 127.0.0.1 만 · 밴 기록·data/live 는 임시 폴더 · 텔레그램 모듈은 가짜.
실행(저장소 뿌리, 네트워크 없는 이름공간 + loopback 만):
  ( ulimit -v 4000000; RTORDER_PY_ROOT=<main 판> TZ=UTC timeout 300 unshare -rn sh -c 'ip link set lo up && python3 -I rtorder/tools/gen_flow_fixture.py' )
  → rtorder/tests/fixtures/flow_fixture.json
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import types
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qsl

ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = Path(os.environ.get("RTORDER_PY_ROOT") or ROOT)
sys.path.insert(0, str(PY_ROOT))
_wd = types.ModuleType("scripts.ops_watchdog")
_wd.send_telegram = lambda *_a, **_k: None
sys.modules["scripts.ops_watchdog"] = _wd
for _k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "DASHBOARD_MANUAL_EXEC_ENABLED"):
    os.environ.pop(_k, None)
os.environ["DASHBOARD_MANUAL_EXEC_SYMBOL"] = "ETHUSDC"

import aiohttp  # noqa: E402

from dashboard import orders  # noqa: E402
from live.execution import live_manual_peg_execute_20260912 as X  # noqa: E402
from scripts import binance_ban_guard as BG  # noqa: E402
from scripts import live_binance_account_20260910 as A  # noqa: E402
from scripts import live_manual_peg_entry_20260912 as P  # noqa: E402

KEY, SECRET = "test_key", "test_secret"
T0 = 1_800_000_000.0
SYM = "ETHUSDC"
REAL_BAN = BG.BAN_FILE
BG.BAN_FILE = Path(tempfile.mkdtemp()) / "binance_ban_until"
assert BG.BAN_FILE != REAL_BAN


class Clock:
    t = 0.0


def now():
    return T0 + Clock.t


async def fsleep(sec, *_a):
    Clock.t += sec


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(now(), tz)


fake_time = types.SimpleNamespace(time=now, monotonic=lambda: Clock.t, perf_counter=lambda: Clock.t)
A.time = X.time = fake_time
X.nap = fsleep
X.asyncio = types.SimpleNamespace(sleep=fsleep)
X.datetime = FakeDT


def ok(body, status=200):
    return {"status": status, "body": body}


def err(status, code, msg):
    return {"status": status, "body": {"code": code, "msg": msg}}


M5022 = err(400, -5022, "Due to the order could not be executed as maker, the Post Only order will be rejected. The order will not be recorded in the order history")


class Sim:
    """규칙으로 움직이는 모의 거래소(헤지 모드 한 심볼). fill(o) = 주문 조회 때 체결로 바꿀지 · del_fail(o) = 취소를 실패로 답할지."""

    def __init__(self, fill=lambda o: False, del_fail=lambda o: False, tp_reject=False, lev="5", orders=(), algos=()):
        self.fill, self.del_fail, self.tp_reject, self.lev = fill, del_fail, tp_reject, lev
        self.orders: dict[int, dict] = {o["orderId"]: dict(o) for o in orders}   # 배포 전에 걸린 주문(dbad 물타기 등)
        self.algos = [dict(a) for a in algos]   # 배포 전에 걸린 우리 스탑(dbsl · Algo)
        self.next = 100
        self.pos = {"LONG": 0.0, "SHORT": 0.0}

    def _fill(self, o):
        o.update(status="FILLED", executedQty=o["origQty"])
        sgn = 1 if (o["side"] == "BUY") == (o["positionSide"] == "LONG") else -1
        self.pos[o["positionSide"]] = round(self.pos[o["positionSide"]] + sgn * float(o["origQty"]), 8)

    def view(self, o):
        return {k: o[k] for k in ("orderId", "clientOrderId", "symbol", "side", "positionSide", "type", "price", "origQty", "executedQty", "status")}

    def __call__(self, method, target):
        path, _, qs = target.partition("?")
        q = dict(parse_qsl(qs, keep_blank_values=True))
        if path == "/fapi/v1/time":
            return ok({"serverTime": int(now() * 1000) + 37})
        if path == "/fapi/v1/ticker/bookTicker":
            return ok({"symbol": q.get("symbol"), "bidPrice": "2500.00", "askPrice": "2500.10"})
        if path == "/fapi/v2/positionRisk":
            return ok([{"symbol": SYM, "positionSide": ps, "positionAmt": str(self.pos[ps] if ps == "LONG" else -self.pos[ps]), "leverage": self.lev,
                        "marginType": "cross"} for ps in ("LONG", "SHORT")])
        if path == "/fapi/v1/leverage" and method == "POST":
            self.lev = q["leverage"]
            return ok({"symbol": SYM, "leverage": int(q["leverage"]), "maxNotionalValue": "1000000"})
        if path == "/fapi/v1/openOrders":
            return ok([self.view(o) for o in self.orders.values() if o["status"] == "NEW"])
        if path == "/fapi/v1/openAlgoOrders":   # 배포 전 우리 스탑(dbsl) 정리 목록 -- 원본·Rust 둘 다 읽는다(2026-11-21 PR 61)
            return ok(list(self.algos))
        if path == "/fapi/v1/algoOrder" and method == "DELETE":
            a = next((x for x in self.algos if str(x["algoId"]) == q.get("algoId")), None)
            if a is None:
                return err(400, -2011, "Unknown order sent.")
            self.algos.remove(a)
            return ok({"algoId": a["algoId"], "code": "200", "msg": "success"})
        if path == "/fapi/v1/order" and method == "POST":
            self.next += 1
            o = {"orderId": self.next, "clientOrderId": q.get("newClientOrderId") or f"x{self.next}", "symbol": q["symbol"], "side": q["side"],
                 "positionSide": q["positionSide"], "type": q["type"], "price": q.get("price", "0"), "origQty": q["quantity"], "executedQty": "0", "status": "NEW"}
            if q["type"] == "LIMIT" and self.tp_reject and o["clientOrderId"].startswith("dbtp"):
                return M5022
            self.orders[o["orderId"]] = o
            if q["type"] == "MARKET":
                self._fill(o)
            return ok(self.view(o))
        if path == "/fapi/v1/order":
            o = self.orders.get(int(q["orderId"])) if q.get("orderId") else next((x for x in self.orders.values() if x["clientOrderId"] == q.get("origClientOrderId")), None)
            if o is None:
                return err(400, -2013, "Order does not exist.")
            if method == "GET":
                o["gets"] = o.get("gets", 0) + 1
                if o["status"] == "NEW" and self.fill(o):
                    self._fill(o)
                return ok(self.view(o))
            if method == "DELETE":
                if self.del_fail(o):
                    return err(500, -1000, "An unknown error occured while processing the request.")
                if o["status"] == "NEW":
                    o["status"] = "CANCELED"
                    return ok(self.view(o))
                return err(400, -2011, "Unknown order sent.")
        return {"status": 599, "body": {"msg": f"규칙 없음 {method} {path}"}}


class S:
    sim: Sim | None = None
    log: list = []
    given: list = []


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _go(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        S.log.append({"method": self.command, "target": self.path, "key": self.headers.get("X-MBX-APIKEY")})
        r = S.sim(self.command, self.path)
        S.given.append(r)
        b = json.dumps(r.get("body"), ensure_ascii=False).encode()
        self.send_response(r.get("status", 200))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_GET = do_POST = do_DELETE = do_PUT = _go


SRV = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=SRV.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{SRV.server_address[1]}"
assert BASE.startswith("http://127.0.0.1:")
X.FAPI = A.FAPI = P.FAPI = BASE


class Ctl:
    calls: list = []
    sec: list = []
    exit_plans: list = []
    books: list = []   # [(t 이상이면, {심볼: 호가})] -- 지정가 재조정(limit_reprice·basis_for)이 읽는 공개 호가


def note(*a):
    Ctl.calls.append([Clock.t, *a])


def account_now():
    return {"ok": True, "generated_at": FakeDT.now(timezone.utc).isoformat(),
            "positions": [{"symbol": SYM, "side": ps, "qty": q} for ps, q in S.sim.pos.items() if q]}


async def f_swr(key, ttl, produce, **_kw):
    note("account")
    return account_now()


async def f_hist(asset):
    note("bars", asset)
    return []


def f_sec(asset):
    note("sec", asset)
    return [tuple(x) for x in Ctl.sec]


def exit_plan_for(pside, qty, **kw):
    return {"symbol": SYM, "side": "SELL" if pside == "LONG" else "BUY", "positionSide": pside, "type": "LIMIT", "timeInForce": "GTX",
            "price": 2500.1 if pside == "LONG" else 2500.0, "quantity": qty, "fallback_after_sec": 30.0, "vol_bpm": 5.0, "blocked": None, **kw}


async def f_exit_plan(position_side, fraction=1.0, fresh=False, asset="eth", market=False, for_order=False, only_symbol=None):
    note("exit_plan", position_side, fraction, fresh, asset, only_symbol)
    p = exit_plan_for(position_side, S.sim.pos[position_side])
    Ctl.exit_plans.append(p)
    return p, None


async def f_fetch(url, params, error_reason=None, **_kw):
    sym = params["symbol"]
    note("book", sym)
    cur = [b for t, b in Ctl.books if Clock.t >= t][-1]
    return dict(cur[sym])


async def f_filters(_session, symbol):
    note("filters", symbol)
    return {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 5.0}


class FakeLock:
    def locked(self):
        return False


async def boom(*_a, **_k):
    raise AssertionError("이 각본에서 부르면 안 된다")


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


def snap(x):
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))


RULE_ARM = {"symbol": SYM, "side": "LONG", "sl_price": 2375.1, "sl_level": 2374.5, "tp_price": None, "backstop_price": 2375.1, "rule": True, "auto": True,
            "sl_name": "첫 진입가 −5%", "lines": False, "tp_level": None, "sl_up": None, "tp_resting": False, "armed_at": T0 - 600.0,
            "placed": {"placed": True, "backstop": {"placed": True}}, "add_order": None}
LINES = {"available": True, "lines": True, "rule": False, "tp_price": 2600.0, "tp_level": 2600.0, "sl_price": 2400.0, "sl_level": 2400.0,
         "sl_name": "선 손절 2400.00", "sl_up": None, "backstop_price": None}
ENTRY = {"symbol": SYM, "side": "BUY", "positionSide": "LONG", "type": "LIMIT", "timeInForce": "GTX", "price": 2500.0, "quantity": 0.4,
         "target_leverage": 10, "bracket": LINES}
LIMIT = {**ENTRY, "price": 2489.0, "limit": {"usdt": 2490.0}, "wid": 1}
BOOK0 = {SYM: {"symbol": SYM, "bidPrice": "2500.00", "askPrice": "2500.10"}, "ETHUSDT": {"symbol": "ETHUSDT", "bidPrice": "2501.00", "askPrice": "2501.10"}}
BOOK1 = {**BOOK0, "ETHUSDT": {"symbol": "ETHUSDT", "bidPrice": "2503.00", "askPrice": "2503.10"}}

SEC_SL = lambda: [[now() - 1.0, 2401.0], [now(), 2400.0]]   # noqa: E731 -- 손절선 2400 에 정확히
SEC_TP = lambda: [[now(), 2600.5]]   # noqa: E731
peg_in = lambda o: o["clientOrderId"].startswith(("dbpg", "dbwt"))   # noqa: E731

SCN = [
    # 1. 진입 메이커 체결 → 익절 걸림 → 손절선 닿음 → 우리 익절 먼저 지움 → 청산 → 무장 해제
    {"name": "선 진입 체결 → 손절선 닿음 → 청산 → 무장 해제", "sim": lambda: Sim(fill=lambda o: peg_in(o) and o["gets"] >= 1),
     "steps": [{"op": "entry", "plan": ENTRY}, {"op": "tick", "dt": 10.0, "sec": SEC_SL}]},
    # 2. 실패 주입: 익절 지정가 -5022 → 서버 감시(tp_resting False) → 익절선 닿음 → 청산
    {"name": "익절 못 검 → 서버 감시 → 익절선 닿음 → 청산", "sim": lambda: Sim(fill=lambda o: peg_in(o) and o["gets"] >= 1, tp_reject=True),
     "steps": [{"op": "entry", "plan": ENTRY}, {"op": "tick", "dt": 10.0, "sec": SEC_TP}]},
    # 3. 실패 주입: 청산 따라가기가 안 차고 취소 확인도 실패 → error · 30초 → 재발동 전 앞선 청산 주문 정리(cancel_exit_orders) → 청산
    {"name": "청산 취소 확인 실패 → 재발동 전 정리 → 청산", "sim": lambda: Sim(
        fill=lambda o: (o["side"] == "BUY" and o["gets"] >= 1) or (o["side"] == "SELL" and Clock.t >= 100.0 and o["gets"] >= 1),
        del_fail=lambda o: o["clientOrderId"].startswith("dbpg") and o["side"] == "SELL" and Clock.t < 100.0),
     "steps": [{"op": "entry", "plan": ENTRY}, {"op": "tick", "dt": 10.0, "sec": SEC_SL}, {"op": "tick", "dt": 34.0, "sec": SEC_SL},
               {"op": "tick", "dt": 31.0, "sec": SEC_SL}]},
    # 3b. 실패 주입: 손절선 발동 전 익절 지정가 취소 실패 → 10초 뒤 다시(무장 유지) → 그다음엔 지우고 청산
    {"name": "발동 전 익절 취소 실패 → 10초 뒤 다시 → 청산", "sim": lambda: Sim(fill=lambda o: peg_in(o) and o["gets"] >= 1,
                                                          del_fail=lambda o: o["clientOrderId"].startswith("dbtp") and Clock.t < 20.0),
     "steps": [{"op": "entry", "plan": ENTRY}, {"op": "tick", "dt": 10.0, "sec": SEC_SL}, {"op": "tick", "dt": 11.0, "sec": SEC_SL}]},
    # 4. 실패 주입: 따라가기 120초 안 참 → 잔량 시장가(M1) → filled_taker → 무장
    {"name": "따라가기 미체결 → 시장가(M1) → 무장", "sim": lambda: Sim(),
     "steps": [{"op": "entry", "plan": ENTRY}]},
    # 6. 배포 전 옛 규칙 무장(−5% 스탑 dbsl · 자동 물타기 dbad 가 거래소에 남음) → 포지션 닫힘 → 둘 다 지우고 무장 해제(PR 61 이 남긴 정리 경로)
    {"name": "옛 규칙 무장 닫힘 → dbad·dbsl 정리 → 무장 해제", "file0": {"ETHUSDC:LONG": RULE_ARM},
     "sim": lambda: Sim(orders=[{"orderId": 50, "clientOrderId": "dbadL1799990000000", "symbol": SYM, "side": "BUY", "positionSide": "LONG", "type": "LIMIT",
                                 "price": "2481.25", "origQty": "0.5", "executedQty": "0", "status": "NEW"}],
                        algos=[{"algoId": 11, "clientAlgoId": "dbslL1799990000000", "symbol": SYM}, {"algoId": 12, "clientAlgoId": "dbslL1799990000001", "symbol": "ETHUSDT"},
                               {"algoId": 13, "clientAlgoId": "dbslS1799990000000", "symbol": SYM}]),
     "steps": [{"op": "tick", "dt": 5.0, "sec": lambda: []}]},
    # 7. 감시 진입(선 위에서 닿으면 따라가기 · maker_only) → 체결 → 무장
    {"name": "감시 진입 닿음 → 따라가기 체결 → 무장", "sim": lambda: Sim(fill=lambda o: peg_in(o) and o["gets"] >= 1),
     "steps": [{"op": "entry", "plan": {**ENTRY, "price": 2510.0, "limit": {"usdt": 2510.0}, "stop": True, "wid": 1}, "limit": True, "sec": lambda: [[now() + 0.5, 2511.0]]}]},
    # 5. P23 지정가 자동 재조정(20초 각본): basis 가 바뀌면 취소 확인 → 새 가격 → 체결 → 첫 체결 무장 → 끝 무장 → 복구 파일 지움
    {"name": "지정가 대기 자동 재조정(P23) → 체결 → 무장", "books": [(0.0, BOOK0), (15.0, BOOK1)],
     "sim": lambda: Sim(fill=lambda o: o["clientOrderId"].startswith("dbwt") and o["price"] != "2489.0" and o["gets"] >= 2),
     "steps": [{"op": "entry", "plan": LIMIT, "limit": True}]},
]


async def run(sc):
    tmp = Path(tempfile.mkdtemp())
    S.sim, Ctl.books = sc["sim"](), sc.get("books") or [(0.0, BOOK0)]
    Clock.t = 0.0
    X._fee_cache.clear()
    P._filters.clear()
    os.environ["BINANCE_API_KEY"], os.environ["BINANCE_SECRET_KEY"] = KEY, SECRET
    out = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.0)) as session:
        with mock.patch.multiple(orders, LIVE_DIR=tmp, exec_enabled=lambda: True, nap=boom, datetime=FakeDT, time=fake_time, load_filters=f_filters,
                                 fetch_account=boom, commission_bp=boom):
            api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: session,
                                  fetch_binance_json=f_fetch, swr_cached=f_swr, _timed_produce=boom, load_market_history=f_hist,
                                  situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=f_sec)
            cells = closure_map([api.bracket_tick, api.api_manual_entry_submit, api.busy_or, api.api_manual_exit_submit])
            cells["assemble_exit_plan"].cell_contents = f_exit_plan
            cells["produce_account"].cell_contents = boom
            cells["manual_order_lock"].cell_contents = FakeLock()
            entry_state = cells["manual_entry_state"].cell_contents
            etb = cells["entry_then_bracket"].cell_contents
            entry_state.clear()
            entry_state.update(phase="submitting")
            if sc.get("file0"):
                (tmp / "manual_bracket_state.json").write_text(json.dumps(sc["file0"], ensure_ascii=False), "utf-8")
            for stp in sc["steps"]:
                Clock.t += stp.get("dt", 0.0)
                Ctl.sec = stp["sec"]() if "sec" in stp else []
                S.log, S.given, Ctl.calls, Ctl.exit_plans = [], [], [], []
                buf, raised = io.StringIO(), None
                t_start = Clock.t
                with contextlib.redirect_stdout(buf):
                    try:
                        if stp["op"] == "entry":
                            await etb(json.loads(json.dumps(stp["plan"])), entry_state, bool(stp.get("limit")))
                        else:
                            await api.bracket_tick()
                    except Exception as exc:  # noqa: BLE001
                        raised = f"{type(exc).__name__}: {exc}"
                p = tmp / "manual_bracket_state.json"
                out.append({"op": stp["op"], "plan": stp.get("plan"), "limit": bool(stp.get("limit")), "t": t_start, "sec": Ctl.sec,
                            "requests": S.log, "responses": S.given, "calls": snap(Ctl.calls), "exit_plans": snap(Ctl.exit_plans),
                            "entry": snap(entry_state), "file": json.loads(p.read_text("utf-8")) if p.exists() else None,
                            "limits": sorted(q.name for q in tmp.glob("manual_limit_wait*.json")),
                            "logs": buf.getvalue().splitlines(), "raised": raised, "t_end": Clock.t, "pos": dict(S.sim.pos)})
    os.environ.pop("BINANCE_API_KEY", None)
    os.environ.pop("BINANCE_SECRET_KEY", None)
    return out


async def main():
    res = []
    for sc in SCN:
        steps = await run(sc)
        res.append({"name": sc["name"], "file0": sc.get("file0"), "books": [[t, b] for t, b in (sc.get("books") or [(0.0, BOOK0)])], "steps": steps})
        print(sc["name"], [(s["op"], len(s["requests"]), s["entry"].get("phase"), sorted((s["file"] or {}).keys())) for s in steps], flush=True)
    assert not REAL_BAN.exists() or REAL_BAN.stat().st_mtime < __import__("time").time() - 5, "진짜 밴 파일을 건드렸다"
    dst = ROOT / "rtorder" / "tests" / "fixtures" / "flow_fixture.json"
    dst.write_text(json.dumps({"t0": T0, "scenarios": res}, ensure_ascii=False, indent=0).replace(BASE, "{BASE}"))
    print(f"시나리오 {len(res)} · 요청 {sum(len(s['requests']) for r in res for s in r['steps'])} · {dst.stat().st_size} bytes")


asyncio.run(main())
SRV.shutdown()
