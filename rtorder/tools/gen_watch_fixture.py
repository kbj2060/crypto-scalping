"""rtorder 셋째 조각 정답지(2026-11-20) -- **원본 `dashboard/orders.py` 의 감시 클로저**(bracket_tick · watch_brackets · partial_exit_rearm_tp ·
arm_bracket · plan_from_current · keep_rearm · limit_recover · wait_cleanup)를 가짜 의존성·가짜 시계로 돌려 틱마다
«나간 호출 열(이름·인자·시각) · 저장본(manual_bracket_state.json) · 상태 칸(진입·청산·대기) · 로그 문구 · 새어 나온 예외»를 적는다.
Rust `cargo test --test watch_fixture` 가 같은 각본(같은 응답 열 · 같은 저장본 고치기)으로 Rust 감시를 돌려 다섯을 대조한다.

꺼내는 법: `register(...)` 가 받는 의존성(swr_cached·load_market_history·sec_prices·binance_session)은 가짜로 넣고,
모듈 전역(clear_bracket·run_exit·place_bracket·place_tp_ladder·cancel_wait_orphans·exec_enabled·nap·time·datetime)은
patch, 클로저 안 변수(assemble_exit_plan·produce_account·manual_order_lock)는 셀 내용을 바꾼다(rtcore/tools/gen_supply1s_fixture.py 와 같은 방식).
각본의 응답 하나 = {"ret" | "raise": [형, 문구], "dt": 시계 전진, "patch": 저장본 고치기, "lock": 주문 잠금, "exec": 실행 스위치, "slot": run_exit 결과 칸}.

안전: 실주문·네트워크 없음 -- 주문·조회 함수는 전부 가짜(각본이 비면 예외) · .env 안 읽음(BINANCE_* 지움) · data/live 대신 임시 폴더 ·
`unshare -rn`(네트워크 없는 이름공간)으로 돌린다. 실행(저장소 뿌리):
  ( ulimit -v 4000000; TZ=UTC timeout 300 unshare -rn python3 -I rtorder/tools/gen_watch_fixture.py )  → rtorder/tests/fixtures/watch_fixture.json
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

ROOT = Path(__file__).resolve().parents[2]
# 2026-11-21 원본 파이썬 위치 -- 기본 = 이 저장소(main 판 · PR 61 머지 뒤) · RTORDER_PY_ROOT = 다른 판을 볼 때만(읽기만 · 정답지는 이 저장소에 쓴다)
PY_ROOT = Path(os.environ.get("RTORDER_PY_ROOT") or ROOT)
sys.path.insert(0, str(PY_ROOT))
# 감시가 망가진 칸을 알릴 때 텔레그램 모듈을 부른다(진짜는 서버 비밀 설정을 읽고 보낸다) -- 가짜를 먼저 넣는다
_wd = types.ModuleType("scripts.ops_watchdog")
_wd.send_telegram = lambda *_a, **_k: None
sys.modules["scripts.ops_watchdog"] = _wd
for _k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "DASHBOARD_MANUAL_EXEC_ENABLED"):
    os.environ.pop(_k, None)
os.environ["DASHBOARD_MANUAL_EXEC_SYMBOL"] = "ETHUSDC"   # 서버 설정과 같게(코인 eth 의 주문 심볼)

from dashboard import orders  # noqa: E402

assert orders.MANUAL_EXEC_SYMBOL == "ETHUSDC"
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "watch_fixture.json"
T0 = 1_800_000_000.0
ETH, SOL, ETHT = "ETHUSDC", "SOLUSDC", "ETHUSDT"


class Clock:
    t = T0


class FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(Clock.t, tz)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


class Ctl:
    calls: list = []
    script: dict = {}
    exec_on = True
    lock = False
    account = None
    bars: dict = {}
    sec: dict = {}
    wakes: list = []
    naps = 0
    max_naps = 0
    per_nap: dict = {}
    bars_patch: list = []
    live: Path = Path(".")
    cells: dict = {}


class FakeLock:
    def locked(self):
        return Ctl.lock


def note(name, *args):
    Ctl.calls.append([Clock.t, name, *args])


def apply_patch(patch):
    p = Ctl.live / "manual_bracket_state.json"
    try:
        st = json.loads(p.read_text("utf-8"))
    except (OSError, ValueError):
        st = {}
    for item in patch:
        k = item["key"]
        if item.get("remove"):
            st.pop(k, None)
        elif "value" in item:
            st[k] = item["value"]
        else:
            cur = st.setdefault(k, {})
            cur.update(item.get("set") or {})
            for f in item.get("del") or []:
                cur.pop(f, None)
    p.write_text(json.dumps(st, ensure_ascii=False), "utf-8")


def respond(name, state=None):
    q = Ctl.script.get(name)
    if not q:
        raise RuntimeError(f"각본 없음: {name}")
    r = q.pop(0)
    Clock.t += r.get("dt", 0.0)
    if "patch" in r:
        apply_patch(r["patch"])
    if "lock" in r:
        Ctl.lock = r["lock"]
    if "exec" in r:
        Ctl.exec_on = r["exec"]
    if state is not None and "slot" in r:
        state.update(r["slot"])
    if "raise" in r:
        raise type(r["raise"][0], (Exception,), {})(r["raise"][1])
    return r


# ── 가짜 의존성(모두 호출을 적는다) ─────────────────────────────────────────────
async def f_swr(key, ttl, produce, **_kw):
    note("account")
    a = Ctl.account
    if isinstance(a, dict) and "raise" in a:
        raise type(a["raise"][0], (Exception,), {})(a["raise"][1])
    return a


async def f_hist(asset):
    note("bars", asset)
    if Ctl.bars_patch:   # 5분봉 읽기(틱의 첫 await) 동안 들어온 사용자 조작 -- 한 번만
        apply_patch(Ctl.bars_patch)
        Ctl.bars_patch = []
    v = Ctl.bars.get(asset)
    if isinstance(v, dict) and "raise" in v:
        raise type(v["raise"][0], (Exception,), {})(v["raise"][1])
    return v or []


def f_sec(asset):
    note("sec", asset)
    v = Ctl.sec.get(asset)
    return None if v is None else [tuple(x) for x in v]


async def f_clear(_s, symbol, pside, *, only_sl=False, only_tp=False):
    note("clear_bracket", symbol, pside, only_sl, only_tp)
    return respond("clear_bracket").get("ret")


async def f_exit_plan(position_side, fraction=1.0, fresh=False, asset="eth", market=False, for_order=False, only_symbol=None):
    note("exit_plan", position_side, fraction, fresh, asset, market, only_symbol)
    r = respond("exit_plan")
    return (r["plan"], None) if "plan" in r else (None, (r["err"], r.get("status", 503)))


def file_now():
    try:
        return json.loads((Ctl.live / "manual_bracket_state.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None


async def f_run_exit(_s, plan, state):
    note("run_exit", plan, json.loads(json.dumps(state)), file_now())   # 청산을 시작하는 순간의 저장본 -- «fired·쏜 조각을 청산 전에 파일에»(B14·B20)
    respond("run_exit", state)
    return state


async def f_place_bracket(_s, bracket, symbol, pside, *, strict=False):
    note("place_bracket", bracket, symbol, pside, strict)
    return respond("place_bracket").get("ret")


async def f_ladder(_s, legs, symbol, pside):
    note("place_tp_ladder", legs, symbol, pside)
    return respond("place_tp_ladder").get("ret")


async def f_orphans(_s, symbols):
    note("cancel_wait_orphans", symbols)
    return respond("cancel_wait_orphans").get("ret")


async def f_cxo(_s, symbol, pside):
    note("cancel_exit_orders", symbol, pside)
    return respond("cancel_exit_orders").get("ret")


async def f_produce():
    note("produce_account")
    return respond("produce_account").get("ret")


async def f_nap(sec):
    note("nap", sec)
    Ctl.naps += 1
    if Ctl.naps > Ctl.max_naps:
        raise asyncio.CancelledError()
    nxt = [w for w in Ctl.wakes if Clock.t < w < Clock.t + sec]
    Clock.t = min(nxt) if nxt else Clock.t + sec
    for k, v in (Ctl.per_nap.get(str(Ctl.naps)) or {}).items():
        setattr(Ctl, k, v)


async def boom(*_a, **_k):
    raise AssertionError("시험에서 거래소·공개 시세 조회 금지")


def closure_map(fns):
    """함수들의 클로저를 따라가며 이름 → 셀(중복 이름은 첫 것)."""
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


async def run_scenario(sc: dict) -> dict:
    tmp = Path(tempfile.mkdtemp())
    Ctl.live = tmp
    Ctl.script = json.loads(json.dumps(sc.get("script") or {}))
    Ctl.lock, Ctl.exec_on, Ctl.naps = False, True, 0
    Ctl.bars_patch = json.loads(json.dumps(sc.get("bars_patch") or []))
    Clock.t = sc["ticks"][0].get("t", T0)
    p = tmp / "manual_bracket_state.json"
    if "file0_raw" in sc:
        p.write_text(sc["file0_raw"], "utf-8")
    elif "file0" in sc:
        p.write_text(json.dumps(sc["file0"], ensure_ascii=False), "utf-8")
    for name, body in (sc.get("limit_files") or {}).items():
        (tmp / name).write_text(json.dumps(body, ensure_ascii=False), "utf-8")
    extra = {"cancel_exit_orders": f_cxo} if hasattr(orders, "cancel_exit_orders") else {}   # 2026-11-21 결함 수정본만
    with mock.patch.multiple(orders, **extra, LIVE_DIR=tmp, clear_bracket=f_clear, run_exit=f_run_exit, place_bracket=f_place_bracket,
                             place_tp_ladder=f_ladder, cancel_wait_orphans=f_orphans,
                             exec_enabled=lambda: Ctl.exec_on, nap=f_nap, datetime=FakeDT,
                             time=types.SimpleNamespace(time=lambda: Clock.t, monotonic=lambda: Clock.t, perf_counter=lambda: Clock.t),
                             fetch_account=boom, run_entry=boom, run_entry_limit=boom, run_entry_stop=boom,
                             commission_bp=boom, load_filters=boom):
        api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                              fetch_binance_json=boom, swr_cached=f_swr, _timed_produce=boom, load_market_history=f_hist,
                              situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=f_sec)
        cells = closure_map([api.bracket_tick, api.start_wait_cleanup, api.api_manual_entry_submit, api.busy_or,
                             api.stop_bracket_watcher, api.start_bracket_watcher, api.api_manual_exit_submit])
        cells["assemble_exit_plan"].cell_contents = f_exit_plan
        cells["produce_account"].cell_contents = f_produce
        cells["manual_order_lock"].cell_contents = FakeLock()
        Ctl.cells = {k: c.cell_contents for k, c in cells.items() if k in ("manual_entry_state", "side_exit_states", "manual_wait_states", "bracket_exiting")}
        fn = {k: cells[k].cell_contents for k in ("partial_exit_rearm_tp", "arm_bracket", "plan_from_current", "keep_rearm",
                                                   "wait_cleanup", "limit_recover", "limit_save", "exit_then_rearm")}
        fn["watch_brackets"] = cells["watch_brackets"].cell_contents
        s0 = sc.get("slots0") or {}
        Ctl.cells["manual_entry_state"].clear()
        Ctl.cells["manual_entry_state"].update(s0.get("entry") or {"phase": "idle"})
        Ctl.cells["side_exit_states"].update(s0.get("side") or {})
        Ctl.cells["manual_wait_states"].update(s0.get("waits") or {})
        out = []
        for tk in sc["ticks"]:
            Clock.t = tk.get("t", Clock.t)
            Ctl.exec_on = tk.get("exec", Ctl.exec_on)
            Ctl.lock = tk.get("lock", Ctl.lock)
            Ctl.account = tk.get("account", Ctl.account)
            Ctl.bars = tk.get("bars", Ctl.bars)
            Ctl.sec = tk.get("sec", Ctl.sec)
            Ctl.wakes = tk.get("wakes") or []
            Ctl.max_naps = tk.get("naps", 0)
            Ctl.per_nap = tk.get("per_nap") or {}
            Ctl.calls = []
            buf = io.StringIO()
            raised, ret, state = None, None, tk.get("state")
            state = None if state is None else json.loads(json.dumps(state))
            op = tk.get("op", "tick")
            with contextlib.redirect_stdout(buf):
                try:
                    if op == "tick":
                        await api.bracket_tick()
                    elif op == "watch":
                        try:
                            await fn["watch_brackets"](None)
                        except asyncio.CancelledError:
                            pass
                    elif op in ("partial_exit_rearm_tp", "keep_rearm", "exit_then_rearm"):
                        await fn[op](tk["plan"], state)
                    elif op == "arm_bracket":
                        await fn[op](tk["plan"], state)
                    elif op == "plan_from_current":
                        ret = fn[op](tk["plan"])
                    elif op == "limit_save":
                        fn[op](tk["plan"])
                    elif op == "wait_cleanup":
                        await fn[op]()
                    else:
                        raise SystemExit(f"모르는 op {op}")
                except Exception as exc:  # noqa: BLE001 -- 감시 틱 밖으로 새는 예외도 정답이다
                    raised = f"{type(exc).__name__}: {exc}"
            raw = p.read_text("utf-8") if p.exists() else None
            limits = {q.name: json.loads(q.read_text("utf-8")) for q in sorted(tmp.glob("manual_limit_wait*.json"))}
            try:
                parsed = None if raw is None else json.loads(raw)
            except ValueError:
                parsed = "<unparseable>"
            out.append({"calls": snap(Ctl.calls), "file": parsed, "file_raw": raw,
                        "limits": limits,
                        "slots": {"entry": snap(Ctl.cells["manual_entry_state"]), "side": snap(Ctl.cells["side_exit_states"]),
                                  "waits": snap(Ctl.cells["manual_wait_states"])},
                        "exiting": sorted(Ctl.cells["bracket_exiting"]), "logs": buf.getvalue().splitlines(),
                        "raised": raised, "ret": snap(ret), "state": snap(state), "t_end": Clock.t,
                        "left": {k: len(v) for k, v in Ctl.script.items() if v}})
    return out


# ── 사례 ─────────────────────────────────────────────────────────────────────
def acct(*pos, ok=True, gen=T0 + 1.0, err=None):
    a = {"ok": ok, "generated_at": iso(gen), "positions": [{"symbol": s, "side": d, "qty": q} for s, d, q in pos]}
    if err:
        a["error"] = err
    return a


def bars(close, last_open=T0 - 300.0):
    return [{"time": int(last_open - 300), "close": 2500.0}, {"time": int(last_open), "close": close}]


RULE_L = {"symbol": ETH, "side": "LONG", "sl_price": 2376.0, "sl_level": 2375.0, "tp_price": None, "backstop_price": 2376.0, "rule": True,
          "sl_name": "−5% 손절", "auto": False, "lines": False, "tp_level": None, "sl_up": None, "tp_resting": False, "armed_at": T0 - 600.0,
          "placed": {"placed": True, "backstop": {"placed": True}}, "add_order": None}
RULE_S = {**RULE_L, "side": "SHORT", "sl_price": 2624.0, "sl_level": 2625.0, "backstop_price": 2624.0}
LINES_L = {"symbol": ETH, "side": "LONG", "sl_price": 2480.0, "sl_level": 2480.0, "tp_price": 2550.0, "tp_level": 2550.0, "backstop_price": None,
           "rule": False, "sl_name": "선 손절 2480.00", "auto": False, "lines": True, "sl_up": None, "tp_resting": True, "armed_at": T0 - 600.0,
           "placed": {"placed": True, "tp": {"placed": True}}, "add_order": None}
LINES_S = {**LINES_L, "side": "SHORT", "sl_price": 2520.0, "sl_level": 2520.0, "tp_price": 2450.0, "tp_level": 2450.0}
LINES_SOL = {**LINES_L, "symbol": SOL, "sl_price": 180.0, "sl_level": 180.0, "tp_price": 200.0, "tp_level": 200.0}
K_L, K_S, K_SOL = f"{ETH}:LONG", f"{ETH}:SHORT", f"{SOL}:LONG"


def plan(pside="LONG", qty=0.5, sym=ETH, **kw):
    return {"symbol": sym, "side": "SELL" if pside == "LONG" else "BUY", "positionSide": pside, "type": "LIMIT", "timeInForce": "GTX",
            "price": 2479.0 if pside == "LONG" else 2521.0, "quantity": qty, "fallback_after_sec": 60.0, "vol_bpm": 5.0, "blocked": None, **kw}


def filled(q=0.5, dt=20.0, phase="filled_maker", **kw):
    return {"dt": dt, "slot": {"phase": phase, "filled": q if phase.startswith("filled") else 0.0, "quantity": q, "done_at": "x", **kw}}


POS_L = (ETH, "LONG", 0.5)
POS_S = (ETH, "SHORT", 0.5)
SEC_L_HIT = {"eth": [[T0 - 2, 2490.0], [T0 - 1, 2480.0], [T0, 2484.0]]}   # 정확히 손절선
SEC_L_MISS = {"eth": [[T0 - 2, 2490.0], [T0 - 1, 2480.01], [T0, 2484.0]]}   # 한 틱 위
NOBARS = {"eth": [], "sol": []}
OK_CLEAR = {"ret": {"cancelled": 1}}


def tick(t=T0, **kw):
    return {"t": t, **kw}


S = []


def add(name, file0, ticks, script=None, **kw):
    S.append({"name": name, "file0": file0, "ticks": ticks, "script": script or {}, **kw})


# 규칙 무장(비상 스탑 있음) -- 5분봉 종가
add("rule_long_bar_exact", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2375.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("rule_long_bar_one_tick_above", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2375.01)}, sec={})])
add("rule_short_bar_exact", {K_S: RULE_S}, [tick(account=acct(POS_S), bars={"eth": bars(2625.0)}, sec={})],
    {"exit_plan": [{"plan": plan("SHORT")}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("rule_short_bar_one_tick_below", {K_S: RULE_S}, [tick(account=acct(POS_S), bars={"eth": bars(2624.99)}, sec={})])
add("rule_bar_before_arming_ignored", {K_L: {**RULE_L, "armed_at": T0 + 1.0}}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})])
add("rule_wick_only_tick_ignored", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2390.0)}, sec={"eth": [[T0, 2300.0]]})])
add("rule_no_level_uses_sl_price", {K_L: {**RULE_L, "sl_level": None}}, [tick(account=acct(POS_L), bars={"eth": bars(2376.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("rule_sl_off_holds", {K_L: {**RULE_L, "sl_off": True}}, [tick(account=acct(POS_L), bars={"eth": bars(2000.0)}, sec={})])
add("rule_bar_feed_fails", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": {"raise": ["TimeoutError", "klines"]}}, sec={})])
add("rule_bar_unclosed_ignored", {K_L: RULE_L}, [tick(t=T0 - 1.0, account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})])

# 선 무장 -- 1초 체결가 닿음
add("lines_long_touch_exact", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("lines_long_one_tick_above", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_MISS)])
add("lines_short_touch", {K_S: LINES_S}, [tick(account=acct(POS_S), bars=NOBARS, sec={"eth": [[T0, 2520.0]]})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan("SHORT")}], "run_exit": [filled()]})
add("lines_short_one_tick_below", {K_S: LINES_S}, [tick(account=acct(POS_S), bars=NOBARS, sec={"eth": [[T0, 2519.99]]})])
add("lines_prices_before_arming_ignored", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0 - 700, 2400.0], [T0, 2500.0]]})])
add("lines_prices_before_watch_from_ignored", {K_L: {**LINES_L, "watch_from": T0 - 10}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0 - 20, 2400.0], [T0, 2500.0]]})])
add("lines_no_price_feed", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec={})])
add("lines_empty_price_feed", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": []})])
add("lines_bar_close_backstop", {K_L: LINES_L}, [tick(account=acct(POS_L), bars={"eth": bars(2479.0)}, sec={})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("lines_tp_touch_not_resting", {K_L: {**LINES_L, "tp_resting": False}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2550.0]]})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("lines_tp_touch_resting_holds", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2560.0]]})])
add("lines_tp_cancel_fails_waits_10s", {K_L: LINES_L},
    [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT), tick(t=T0 + 5, sec=SEC_L_HIT), tick(t=T0 + 11, sec=SEC_L_HIT)],
    {"clear_bracket": [{"ret": {"cancelled": 0, "errors": ["400 [-2011] Unknown order sent."]}}, OK_CLEAR, OK_CLEAR],
     "exit_plan": [{"plan": plan()}], "run_exit": [filled()], "cancel_exit_orders": [{"ret": {"cancelled": 0, "errors": []}}]})
add("lines_sl_off_tp_watch_fires_tp", {K_L: {**LINES_L, "sl_off": True, "tp_resting": False}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2400.0], [T0, 2551.0]]})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("lines_fired_true_sl_off_holds", {K_L: {**LINES_L, "sl_off": True, "fired": True, "tp_resting": False}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2400.0]]})])
add("lines_fired_latch_after_restart", {K_L: {**LINES_L, "fired": True, "tp_resting": False}}, [tick(account=acct(POS_L), bars=NOBARS, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})

# 분할 손절·분할 익절 · 부분 청산 뒤 익절 다시 걸기
LEGS2 = {**LINES_L, "sl_legs": [{"level": 2480.0, "price": 2480.0, "up": None, "w": 1.0, "fired": None},
                                {"level": 2460.0, "price": 2460.0, "up": None, "w": 1.0, "fired": None}], "sl_name": "선 손절 2480.00 외 1"}
add("sl_legs_one_then_all", {K_L: LEGS2},
    [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]}),
     tick(t=T0 + 30, account=acct((ETH, "LONG", 0.25)), sec={"eth": [[T0 + 29, 2459.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25, fraction=0.5)}, {"plan": plan(qty=0.25)}],
     "run_exit": [filled(0.25), filled(0.25)],
     "place_bracket": [{"ret": {"qty": 0.25, "tp": {"placed": True, "qty": 0.25}, "backstop": None, "placed": True, "cleared": {"cancelled": 1}}}],
     "clear_bracket": [OK_CLEAR, OK_CLEAR]})
add("sl_leg_lock_held_skips", {K_L: LEGS2}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]}, lock=True)])
add("sl_leg_exit_fails_keeps_fired", {K_L: LEGS2},
    [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]}), tick(t=T0 + 10, sec={"eth": [[T0, 2479.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25)}], "run_exit": [filled(0.25, phase="rejected", error="400 [-2019] Margin is insufficient.")]})
add("sl_leg_disarmed_during_plan", {K_L: LEGS2}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25), "patch": [{"key": K_L, "set": {"sl_off": True}}]}]})
TPLEGS = {**LINES_L, "tp_legs": [{"level": 2550.0, "price": 2550.0, "w": 1.0, "resting": True, "merged": None, "cid": "dbtpL1", "order_id": 11, "qty": 0.25},
                                 {"level": 2600.0, "price": 2600.0, "w": 1.0, "resting": False, "merged": None, "cid": None, "order_id": None, "qty": 0.25}]}
add("tp_leg_not_resting_touched_fires_full", {K_L: TPLEGS}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2600.0]]})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("tp_leg_resting_touched_holds", {K_L: TPLEGS}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2560.0]]})])
add("sl_leg_with_tp_ladder_rearm", {K_L: {**LEGS2, "tp_legs": TPLEGS["tp_legs"]}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25)}], "run_exit": [filled(0.25)],
     "place_tp_ladder": [{"ret": {"qty": 0.25, "placed": True, "cleared": {"cancelled": 1},
                                  "legs": [{"level": 2550.0, "price": 2550.0, "w": 1.0, "resting": True, "qty": 0.12, "cid": "dbtpL9", "order_id": 21},
                                           {"level": 2600.0, "price": 2600.0, "w": 1.0, "resting": True, "qty": 0.13, "cid": "dbtpL10", "order_id": 22}]}}]})
add("sl_leg_tp_ladder_partial_cancel_watch_only", {K_L: {**LEGS2, "tp_legs": TPLEGS["tp_legs"]}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25)}], "run_exit": [filled(0.25)],
     "place_tp_ladder": [{"ret": {"qty": 0.25, "placed": False, "reason": "옛 익절 정리 실패", "cleared": {"cancelled": 0, "errors": ["500 x"]}}}]})
add("sl_leg_tp_ladder_raises", {K_L: {**LEGS2, "tp_legs": TPLEGS["tp_legs"]}}, [tick(account=acct(POS_L), bars=NOBARS, sec={"eth": [[T0, 2479.0]]})],
    {"exit_plan": [{"plan": plan(qty=0.25)}], "run_exit": [filled(0.25)], "place_tp_ladder": [{"raise": ["KeyError", "'legs'"]}]})

# 사용자 조작 중 발동 · 잠금(D9) · 스위치
add("line_moved_before_fire_skips", {K_L: {**LINES_L, "mod": {"at": T0 - 100}}}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)],
    {}, bars_patch=[{"key": K_L, "set": {"mod": {"at": T0 - 1, "tp": 2550.0, "sl": 2470.0}, "sl_level": 2470.0}}])
add("disarm_during_plan_rule", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan(), "patch": [{"key": K_L, "set": {"sl_off": True}}]}]})
add("disarm_before_fire_lines", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)], {},
    bars_patch=[{"key": K_L, "set": {"sl_off": True}}])
add("lock_held_rule_D9", {K_L: RULE_L},
    [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={}, lock=True), tick(t=T0 + 2, lock=True), tick(t=T0 + 4, lock=False)],
    {"exit_plan": [{"plan": plan()}, {"plan": plan()}, {"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("lock_held_lines_D9", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT, lock=True), tick(t=T0 + 2, lock=False)],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("lock_taken_during_plan", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan(), "lock": True}]})
add("exec_switch_off", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={}, exec=False)])
add("exec_switch_off_during_tick_still_fires", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan(), "exec": False}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})

# 발동 뒤 청산 실패 → 30초 뒤 다시
for ph, extra in (("rejected", {"error": "400 [-2019] Margin is insufficient."}),
                  ("error", {"error": "주문 접수 여부를 확인하지 못했습니다(TimeoutError: x) — 거래소에서 직접 확인하세요"}),
                  ("taker_failed", {"error": "500 x"})):
    add(f"exit_{ph}_retry_after_30s", {K_L: RULE_L},
        [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={}), tick(t=T0 + 20), tick(t=T0 + 41)],
        {"exit_plan": [{"plan": plan()}, {"plan": plan()}], "run_exit": [filled(dt=10, phase=ph, **extra), filled()], "clear_bracket": [OK_CLEAR],
         "cancel_exit_orders": [{"ret": {"cancelled": 1, "errors": []}}]})   # 2026-11-21 재발동 전 앞선 청산 주문 정리
add("exit_raises_settle_error", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [{"dt": 1.0, "raise": ["TypeError", "'NoneType' object is not subscriptable"]}]})
add("exit_needs_market_python_taker", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled(phase="filled_taker", taker_qty=0.2)], "clear_bracket": [OK_CLEAR]})
add("plan_no_position_clears", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"err": {"error": "no_position", "detail": "LONG 포지션이 없습니다"}, "status": 400}], "clear_bracket": [OK_CLEAR]})
add("plan_no_position_clear_fails", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"err": {"error": "no_position", "detail": "LONG 포지션이 없습니다"}, "status": 400}],
     "clear_bracket": [{"ret": {"cancelled": 0, "errors": ["x"]}}]})
add("plan_account_unavailable", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"err": {"error": "account_unavailable", "detail": "계좌 조회 실패: 418"}}]})
add("plan_exception_502", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"err": {"error": "KeyError: 'bidPrice'"}, "status": 502}]})
add("plan_blocked", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan(blocked="닫을 포지션이 없습니다")}]})
add("filled_but_cleanup_fails_then_clear", {K_L: RULE_L},
    [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={}), tick(t=T0 + 30, account=acct(gen=T0 + 29)), tick(t=T0 + 70)],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()],
     "clear_bracket": [{"ret": {"cancelled": 0, "errors": ["500 x"]}}, {"ret": {"cancelled": 0, "errors": ["500 y"]}}, OK_CLEAR]})

# 여러 코인·측면 (D8) · 헤지 모드
add("two_coins_touch_serial_D8", {K_L: LINES_L, K_SOL: LINES_SOL},
    [tick(account=acct(POS_L, (SOL, "LONG", 3.0)), bars=NOBARS, sec={"eth": [[T0, 2470.0]], "sol": [[T0, 179.0]]})],
    {"clear_bracket": [OK_CLEAR] * 4, "exit_plan": [{"plan": plan()}, {"plan": plan(sym=SOL, qty=3.0)}],
     "run_exit": [filled(dt=60.0), filled(3.0, dt=5.0)]})
add("two_coins_other_entry_busy_uses_side_slot", {K_SOL: LINES_SOL},
    [tick(account=acct((SOL, "LONG", 3.0)), bars=NOBARS, sec={"sol": [[T0, 179.0]]})],
    {"clear_bracket": [OK_CLEAR] * 2, "exit_plan": [{"plan": plan(sym=SOL, qty=3.0)}], "run_exit": [filled(3.0)]},
    slots0={"entry": {"phase": "working", "asset": "eth", "kind": "entry", "side": "LONG"}})
add("same_coin_entry_busy_skips", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})], {},
    slots0={"entry": {"phase": "working", "asset": "eth", "kind": "exit", "side": "LONG"}})
add("same_coin_side_slot_busy_skips", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})], {},
    slots0={"entry": {"phase": "working", "asset": "sol"}, "side": {"eth": {"phase": "submitting", "kind": "exit"}}})
add("hedge_long_short_both_armed_one_fires", {K_L: LINES_L, K_S: LINES_S},
    [tick(account=acct(POS_L, POS_S), bars=NOBARS, sec={"eth": [[T0, 2475.0], [T0, 2500.0]]})],
    {"clear_bracket": [OK_CLEAR] * 2, "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})
add("hedge_long_short_both_fire", {K_L: LINES_L, K_S: LINES_S},
    [tick(account=acct(POS_L, POS_S), bars=NOBARS, sec={"eth": [[T0, 2475.0], [T0, 2525.0]]})],
    {"clear_bracket": [OK_CLEAR] * 4, "exit_plan": [{"plan": plan()}, {"plan": plan("SHORT")}], "run_exit": [filled(), filled()]})

# 포지션이 이미 닫힘(clear) · 계좌
WAITS = {"eth": {"phase": "working", "mode": "limit", "side": "LONG", "plan": {"symbol": ETH}},
         "eth#2": {"phase": "working", "mode": "limit", "side": "SHORT", "plan": {"symbol": ETH}},
         "sol": {"phase": "working", "mode": "limit", "side": "LONG", "plan": {"symbol": SOL}}}
add("position_gone_clears_and_cancels_same_side_wait", {K_L: RULE_L}, [tick(account=acct(), bars={"eth": bars(2500.0)}, sec={})],
    {"clear_bracket": [OK_CLEAR]}, slots0={"waits": WAITS})
add("position_gone_but_account_too_old", {K_L: {**RULE_L, "armed_at": T0 - 2.0}}, [tick(account=acct(gen=T0 + 2.0), bars={"eth": bars(2500.0)}, sec={})])
add("position_gone_clear_fails_retry_30s", {K_L: RULE_L},
    [tick(account=acct(), bars={"eth": bars(2500.0)}, sec={}), tick(t=T0 + 10), tick(t=T0 + 31)],
    {"clear_bracket": [{"ret": {"cancelled": 0, "errors": ["500 boom"]}}, OK_CLEAR]})
add("account_not_ok_no_clear", {K_L: RULE_L}, [tick(account=acct(ok=False, err="418"), bars={"eth": bars(2500.0)}, sec={})])
add("account_not_ok_sl_hit_plan_fails", {K_L: RULE_L}, [tick(account=acct(ok=False, err="418"), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"err": {"error": "account_unavailable", "detail": "계좌 조회 실패: 418"}}]})
add("account_raises_tick_dies", {K_L: RULE_L}, [tick(account={"raise": ["ClientConnectorError", "Cannot connect"]}, bars={"eth": bars(2300.0)}, sec={})])
add("account_bad_generated_at", {K_L: RULE_L}, [tick(account={"ok": True, "generated_at": "nope", "positions": []}, bars={"eth": bars(2500.0)}, sec={})])
add("account_missing_generated_at", {K_L: RULE_L}, [tick(account={"ok": True, "positions": []}, bars={"eth": bars(2500.0)}, sec={})])
add("corrupt_entry_blocks_all_coins", {"BAD:LONG": {"side": "LONG", "armed_at": T0 - 600.0, "lines": True}, K_L: LINES_L},
    [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)])
add("legacy_key_migrates_on_load", {"LONG": {k: v for k, v in RULE_L.items() if k != "side"}}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [{**filled(), "patch": [{"key": K_S, "value": RULE_S}]}], "clear_bracket": [OK_CLEAR]})
add("legacy_and_new_key_collide", {K_L: RULE_L, "LONG": {**{k: v for k, v in RULE_L.items() if k != "side"}, "sl_level": 2600.0, "armed_at": T0 - 500.0}},
    [tick(account=acct(POS_L), bars={"eth": bars(2500.0)}, sec={})],   # 같은 칸이면 파일 순서상 뒤의 것(옛 키 쪽 손절 2600)이 이긴다
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled(phase="rejected", error="400 x")]})
add("no_armed_state_nothing", {}, [tick(account=acct(POS_L), bars=NOBARS, sec={})])
# 2026-11-21 결함 수정본: 재발동 전 앞선 청산 주문 정리(cancel_exit_orders) · 망가진 칸 건너뛰기 · 알림 한 번
RETRY_L = {**RULE_L, "retry_after": T0 - 1.0, "last_error": "주문 1 취소를 확인하지 못했습니다"}
add("refire_cleanup_ok_then_exit", {K_L: RETRY_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"cancel_exit_orders": [{"ret": {"cancelled": 1, "errors": []}}], "exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("refire_cleanup_fails_keeps_arm", {K_L: RETRY_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"cancel_exit_orders": [{"ret": {"cancelled": 1, "errors": ["앞선 청산 주문 [7] 이 아직 살아 있습니다"]}}]})
add("refire_cleanup_list_fails", {K_L: RETRY_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"cancel_exit_orders": [{"ret": {"cancelled": 0, "errors": ["openOrders: 418 [-1003] banned"]}}]})
add("refire_retry_not_yet", {K_L: {**RETRY_L, "retry_after": T0 + 5.0}}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})])
add("refire_lines_after_tp_cancel", {K_L: {**LINES_L, "fired": True, "tp_resting": False, "retry_after": T0 - 1.0}}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)],
    {"cancel_exit_orders": [{"ret": {"cancelled": 0, "errors": []}}], "exit_plan": [{"plan": plan()}], "run_exit": [filled(phase="error", error="x")]})
add("refire_sl_leg_cleanup_fails", {K_L: {**LINES_L, "retry_after": T0 - 1.0, "sl_legs": [{"level": 2480.0, "w": 1}, {"level": 2470.0, "w": 1}]}},
    [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)], {"cancel_exit_orders": [{"ret": {"cancelled": 0, "errors": ["boom"]}}]})
add("first_fire_no_cleanup_query", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("corrupt_slot_alert_once_two_ticks", {"BAD:LONG": {"side": "LONG", "armed_at": T0 - 600.0, "lines": True}, "STR:LONG": "garbage", K_L: LINES_L},
    [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_MISS), tick(t=T0 + 2.0, account=acct(POS_L, gen=T0 + 3.0), bars=NOBARS, sec=SEC_L_MISS)])
add("corrupt_bad_number_skips_only_that", {"ETHUSDC:SHORT": {**RULE_S, "armed_at": "x"}, K_L: RULE_L}, [tick(account=acct(POS_L, POS_S), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}], "run_exit": [filled()], "clear_bracket": [OK_CLEAR]})
add("unreadable_file_nothing", None, [tick(account=acct(POS_L), bars=NOBARS, sec={})], file0_raw="{not json")

# 틱 사이 저장본 경쟁(bracket_merge)
add("new_arming_during_exit_kept", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}],
     "run_exit": [{**filled(phase="rejected", error="400 x"), "patch": [{"key": K_S, "value": {**RULE_S, "armed_at": T0 + 5.0}}]}]})
add("same_side_rearmed_during_exit_not_popped", {K_L: RULE_L}, [tick(account=acct(POS_L), bars={"eth": bars(2300.0)}, sec={})],
    {"exit_plan": [{"plan": plan()}],
     "run_exit": [{**filled(), "patch": [{"key": K_L, "value": {**RULE_L, "armed_at": T0 + 5.0, "sl_price": 2300.0}}]}], "clear_bracket": [OK_CLEAR]})
add("user_moved_tp_during_exit_saved_wins", {K_L: LINES_L}, [tick(account=acct(POS_L), bars=NOBARS, sec=SEC_L_HIT)],
    {"clear_bracket": [OK_CLEAR], "exit_plan": [{"plan": plan()}],
     "run_exit": [{**filled(phase="rejected", error="400 x"), "patch": [{"key": K_L, "set": {"tp_level": 2600.0, "mod": {"at": T0 + 3}}}]}]})

# 감시 루프(2초 · 알림에 일찍 깸 · 예외 삼킴)
add("watch_loop_period_and_wake", {K_L: RULE_L},
    [tick(op="watch", naps=4, account=acct(POS_L), bars={"eth": bars(2500.0)}, sec={}, wakes=[T0 + 3.0],
          per_nap={"2": {"account": {"raise": ["TimeoutError", "acct"]}}, "3": {"account": acct(POS_L, gen=T0 + 5)}})])
add("watch_loop_fires_on_second_tick", {K_L: LINES_L},
    [tick(op="watch", naps=3, account=acct(POS_L), bars=NOBARS, sec={"eth": []}, per_nap={"2": {"sec": {"eth": [[T0 + 3.5, 2470.0]]}}})],
    {"clear_bracket": [OK_CLEAR, OK_CLEAR], "exit_plan": [{"plan": plan()}], "run_exit": [filled()]})

# 무장 쓰기 · 다시 걸기 · 재시작 복구(감시 밖 클로저)
BR_RULE = {"available": True, "rule": True, "sl_price": 2376.0, "sl_level": 2375.0, "tp_price": None, "backstop_price": 2376.0, "sl_name": "−5% 손절"}
BR_LINES = {"available": True, "lines": True, "sl_price": 2480.0, "sl_level": 2480.0, "tp_price": 2550.0, "tp_level": 2550.0, "backstop_price": None,
            "sl_name": "선 손절 2480.00", "sl_up": None}
EPLAN = {"symbol": ETH, "positionSide": "LONG", "side": "BUY", "quantity": 0.5}
PB_OK = {"ret": {"qty": 0.5, "tp": None, "backstop": {"placed": True, "order_id": 7}, "placed": True, "cleared": {"cancelled": 0}}}
PB_LINES = {"ret": {"qty": 0.5, "tp": {"placed": True, "order_id": 8}, "backstop": None, "placed": True, "cleared": {"cancelled": 0}}}
add("arm_bracket_rule_old_plan_no_add", {K_S: RULE_S},   # 2026-11-21 PR 61: 자동 물타기 지움 -- 규칙 계획이 와도 place_add_order 없음
    [tick(op="arm_bracket", plan={**EPLAN, "bracket": BR_RULE, "rule": {"first": True, "stop_pct": 5.0, "l": 2.0, "cap_notional": 3000.0, "room": 3000.0,
                                                                         "lev": 20, "filters": {"tick": 0.01, "step": 0.001, "min_qty": 0.001, "min_notional": 20.0}}},
          state={"phase": "filled_maker", "filled": 0.4})],
    {"place_bracket": [PB_OK]})
add("arm_bracket_rule_stop_failed_no_add", {},
    [tick(op="arm_bracket", plan={**EPLAN, "bracket": BR_RULE, "rule": {"first": True, "stop_pct": 5.0, "l": 2.0, "cap_notional": 3000.0, "room": 3000.0, "lev": 20}},
          state={"filled": 0.4})],
    {"place_bracket": [{"ret": {"qty": 0.5, "tp": None, "backstop": {"placed": False, "reason": "-2021"}, "placed": False, "cleared": {}}}]})
add("arm_bracket_lines_tp_off_and_raises", {K_L: {**LINES_L, "armed_at": T0 - 5000}},
    [tick(op="arm_bracket", plan={**EPLAN, "bracket": {**BR_LINES, "tp_off": True}}, state={"filled": 0.5}),
     tick(op="arm_bracket", t=T0 + 1, plan={**EPLAN, "bracket": BR_LINES}, state={"filled": 0.5})],
    {"place_bracket": [PB_LINES, {"raise": ["TimeoutError", "place"]}]})
add("arm_bracket_unavailable", {}, [tick(op="arm_bracket", plan={**EPLAN, "bracket": {"available": False}}, state={})])
add("plan_from_current_variants", {K_L: {**LINES_L, "tp_level": 2600.0, "tp_price": 2600.0, "tp_off": True, "mod": {"at": T0}}},
    [tick(op="plan_from_current", plan={**EPLAN, "bracket": BR_LINES}),
     tick(op="plan_from_current", plan={**EPLAN, "positionSide": "SHORT", "bracket": BR_LINES}),
     tick(op="plan_from_current", plan={**EPLAN, "bracket": BR_RULE})])
add("plan_from_current_fired_none", {K_L: {**LINES_L, "fired": True}}, [tick(op="plan_from_current", plan={**EPLAN, "bracket": BR_LINES})])
add("partial_exit_rearm_tp_single", {K_L: {**RULE_L, "tp_price": 2600.0, "tp_resting": True}},
    [tick(op="partial_exit_rearm_tp", plan=plan(qty=0.2), state={"phase": "filled_maker"}),
     tick(op="partial_exit_rearm_tp", plan=plan(qty=0.2), state={})],
    {"place_bracket": [{"ret": {"qty": 0.3, "tp": {"placed": True}, "backstop": {"placed": True}, "placed": True, "cleared": {"cancelled": 2}}},
                       {"ret": {"qty": 0.3, "tp": {"placed": False, "reason": "-5022"}, "backstop": {"placed": True}, "placed": False, "cleared": {"cancelled": 1}}},
                       ]})
add("partial_exit_rearm_tp_lookup_fails_keeps_state", {K_L: {**RULE_L, "tp_price": 2600.0, "tp_resting": True}},
    [tick(op="partial_exit_rearm_tp", plan=plan(qty=0.2), state={})], {"place_bracket": [{"ret": {"placed": False, "reason": "포지션 조회 실패"}}]})
add("partial_exit_rearm_tp_skips", {K_L: {**RULE_L, "tp_price": 2600.0, "tp_resting": False}, K_S: {**RULE_S, "tp_price": 2400.0, "tp_resting": True, "fired": True}},
    [tick(op="partial_exit_rearm_tp", plan=plan(), state={}), tick(op="partial_exit_rearm_tp", plan=plan("SHORT"), state={})])
add("partial_exit_rearm_tp_rearmed_meanwhile", {K_L: {**RULE_L, "tp_price": 2600.0, "tp_resting": True, "sl_off": True}},
    [tick(op="partial_exit_rearm_tp", plan=plan(), state={})],
    {"place_bracket": [{"ret": {"qty": 0.3, "tp": {"placed": True}, "placed": True, "cleared": {}},
                        "patch": [{"key": K_L, "set": {"armed_at": T0 + 9.0}}]}]})
add("keep_rearm_paths", {K_L: {**LINES_L, "tp_resting": True}},
    [tick(op="keep_rearm", plan={**EPLAN, "bracket": BR_LINES, "add_keep": True}, state={}),
     tick(op="keep_rearm", plan={**EPLAN, "positionSide": "SHORT", "bracket": BR_LINES, "add_keep": True}, state={"filled": 0.5})],
    {"place_bracket": [{"ret": {"qty": 1.0, "tp": {"placed": True}, "placed": True, "cleared": {"cancelled": 1}}}, PB_LINES]})
add("keep_rearm_sl_leg_fired_skips", {K_L: {**LEGS2, "sl_legs": [{**LEGS2["sl_legs"][0], "fired": True}, LEGS2["sl_legs"][1]]}},
    [tick(op="keep_rearm", plan={**EPLAN, "bracket": BR_LINES, "add_keep": True}, state={})])
add("exit_then_rearm_partial", {K_L: {**RULE_L, "tp_price": 2600.0, "tp_resting": True}},
    [tick(op="exit_then_rearm", plan=plan(qty=0.2), state={})],
    {"run_exit": [filled(0.2)], "place_bracket": [{"ret": {"qty": 0.3, "tp": {"placed": True}, "placed": True, "cleared": {"cancelled": 1}}}]})

LREC = {"saved_at": T0 - 600, "plan": {**EPLAN, "wid": 1, "bracket": BR_LINES, "mode": "limit"}}
add("restart_cleanup_and_recover", {K_S: RULE_S},
    [tick(op="wait_cleanup")],
    {"cancel_wait_orphans": [{"ret": {"cancelled": [{"symbol": ETH, "order_id": 5, "cid": "dbwtL1", "filled": 0.1}], "errors": []}}],
     "produce_account": [{"ret": acct(POS_L)}], "place_bracket": [PB_LINES]},
    limit_files={"manual_limit_wait_ETHUSDC.json": LREC,
                 "manual_limit_wait_SOLUSDC_2.json": {"saved_at": T0 - 30000, "plan": {"symbol": SOL, "positionSide": "LONG", "wid": 2}}})
add("restart_orphan_errors_keep_file", {},
    [tick(op="wait_cleanup")],
    {"cancel_wait_orphans": [{"ret": {"cancelled": [], "errors": ["ETHUSDC: 500 x"]}}], "produce_account": [{"ret": acct(POS_L)}],
     "place_bracket": [{"ret": {"qty": 0.5, "tp": {"placed": False}, "placed": False, "reason": "익절 거부", "cleared": {}}}]},
    limit_files={"manual_limit_wait_ETHUSDC.json": LREC})
add("restart_orphan_errors_keep_file_even_if_armed", {},
    [tick(op="wait_cleanup")],
    {"cancel_wait_orphans": [{"ret": {"cancelled": [{"symbol": SOL, "order_id": 6, "cid": "dbpgL2"}], "errors": ["XRPUSDC: 500 x"]}}],
     "produce_account": [{"ret": acct(POS_L)}], "place_bracket": [PB_LINES]},
    limit_files={"manual_limit_wait_ETHUSDC.json": LREC})
add("restart_account_fails_and_cleanup_raises", {},
    [tick(op="wait_cleanup")],
    {"cancel_wait_orphans": [{"raise": ["TimeoutError", "openOrders"]}], "produce_account": [{"ret": acct(ok=False)}]},
    limit_files={"manual_limit_wait_ETHUSDC.json": LREC})
add("restart_already_armed_or_no_position", {K_L: LINES_L},
    [tick(op="wait_cleanup")],
    {"cancel_wait_orphans": [{"ret": {"cancelled": [], "errors": []}}], "produce_account": [{"ret": acct(POS_L)}, {"ret": acct()}]},
    limit_files={"manual_limit_wait_ETHUSDC.json": LREC, "manual_limit_wait_ETHUSDC_2.json": {**LREC, "plan": {**LREC["plan"], "positionSide": "SHORT", "wid": 2}}})
# 양방향 왕복용 까다로운 값(감시는 꺼 둔다 -- 파이썬이 쓴 원문 그대로 남는다)
TRICKY = {**LINES_L, "armed_at": 1800000000.123456, "tp_level": 0.1 + 0.2, "sl_level": 1e-07, "sl_price": 1e16, "tp_price": -0.0,
          "big": 2 ** 53 + 1, "neg": -7, "ko": "선 손절 «한글» ✓ 🚀", "esc": "따옴표\" 역슬래시\\ 줄\n탭\t", "none": None, "flag": False,
          "nest": {"legs": [{"w": 1.0, "level": 123456789.12345679}, {"w": 2, "level": 5e-324}], "empty": {}, "list": []},
          "mod": {"at": 1800000000.5, "watch_only": True}}
add("roundtrip_tricky_values", {K_L: TRICKY, "SHORT": {**{k: v for k, v in RULE_S.items() if k != "side"}, "x": 1.7976931348623157e308}},
    [tick(exec=False, account=acct(), bars=NOBARS, sec={})])
add("limit_save_writes", {}, [tick(op="limit_save", plan={**EPLAN, "wid": 3, "limit": {"usdt": 2400.5}, "note": "한글"})])


def main() -> int:
    out = []
    for sc in S:
        res = asyncio.run(run_scenario(json.loads(json.dumps(sc))))
        for t in res[:-1]:   # 저장본 원문(왕복 시험용)은 마지막 틱만
            t.pop("file_raw")
        out.append({"scenario": sc, "ticks": res})
    for r in out:   # 각본이 남으면 사례가 뜻한 길을 안 간 것이다
        left = r["ticks"][-1]["left"]
        if left:
            print("🔴각본 남음", r["scenario"]["name"], left)
    OUT.write_text(json.dumps({"t0": T0, "eth_symbol": ETH, "scenarios": out}, ensure_ascii=False, indent=None), "utf-8")
    n_calls = sum(len(t["calls"]) for r in out for t in r["ticks"])
    print(f"사례 {len(out)} · 틱 {sum(len(r['ticks']) for r in out)} · 호출 {n_calls} · {OUT.stat().st_size} 바이트")
    return 0


if __name__ == "__main__":
    sys.exit(main())
