"""rtorder 계획 함수 정답지 -- **원본 파이썬 함수**(scripts/live_manual_peg_entry_20260912.py)를 넓은 입력 격자 + 고정 시드 무작위 입력으로 돌려
입력·출력 쌍을 JSON 으로 쓴다(2026-11-20). Rust `cargo test --test plan_fixture` 가 같은 입력을 Rust 함수에 넣어 필드 단위로 대조한다.

원본 모듈 import 가 건드리는 것: math·os·typing 뿐(네트워크·.env·dashboard 없음). `load_filters` 만 네트워크(호출 안 함).
`exec_enabled()` 가 환경변수를 읽어 계획의 dry_run 칸이 바뀌므로 여기서 지운다(Rust 시험도 같은 전제를 단언한다).
실행: python rtorder/tools/gen_plan_fixture.py  (저장소 뿌리에서. 출력 rtorder/tests/fixtures/plan_fixture.json)
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY_ROOT = Path(os.environ.get("RTORDER_PY_ROOT") or ROOT)   # 2026-11-21 원본 파이썬 위치(읽기만 · 정답지는 이 저장소에)
os.environ.pop("DASHBOARD_MANUAL_EXEC_ENABLED", None)
_spec = importlib.util.spec_from_file_location("peg_plan", PY_ROOT / "scripts" / "live_manual_peg_entry_20260912.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

rng = random.Random(20261120)
ORDER_KEYS = ("price", "quantity", "qty", "sl_price", "tp_price", "backstop_price", "reference_price")
FILTERS = [
    {"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0},   # ETH
    {"step": 1.0, "tick": 0.0001, "min_qty": 1.0, "min_notional": 5.0},     # XRP 류
    {"step": 0.1, "tick": 0.001, "min_qty": 0.1, "min_notional": 5.0},
    {"step": 0.001, "tick": 0.1, "min_qty": 0.001, "min_notional": 100.0},  # BTC 류
    {"step": 1e-05, "tick": 0.5, "min_qty": 0.0, "min_notional": 0.0},
    {"step": 0.0, "tick": 0.0, "min_qty": 0.0, "min_notional": 0.0},        # 격자 없음(grid<=0 분기 · `or` 기본값 분기)
    {"step": 0.01, "tick": 1e-07, "min_qty": 0.01, "min_notional": 5.0},
    {"step": 0.003, "tick": 0.03, "min_qty": 0.0, "min_notional": 1.0},     # 1/3 류 -- 나눗셈이 깨끗하지 않은 격자
]
# 반올림이 갈리는 자리: 내림 `int(v/g + 1e-9)` 의 경계 1-1e-9 양쪽 · 올림 `ceil(v/g - 1e-9)` 의 경계 1e-9 양쪽 · 정확히 격자 · 반 칸
OFFS = [0.0, 1 - 1e-9 - 1e-11, 1 - 1e-9 + 1e-11, 1e-9 - 1e-11, 1e-9 + 1e-11, 0.5, 1e-13, -1e-13, 0.9999999, 0.1 + 0.2]
cases: list[dict] = []


def on_grid(g: float, scale: float) -> float:
    g = g or 0.01
    n = max(1, int(scale * rng.uniform(0.5, 2.0) / g))
    return (n + rng.choice(OFFS)) * g


def reprs(out):
    if isinstance(out, dict):
        return {k: repr(v) for k, v in out.items() if k in ORDER_KEYS and isinstance(v, float)}
    if isinstance(out, list) and all(isinstance(v, float) for v in out):
        return [repr(v) for v in out]
    return None


def add(fn: str, args: dict, call=None, tag: str = ""):
    try:
        out = (call or getattr(P, fn))(**args)
        rec = {"fn": fn, "args": args, "out": out}
        r = reprs(out)
        if r:
            rec["repr"] = r
    except Exception as e:  # noqa: BLE001 -- 거절 경로도 정답지다(종류·사유까지)
        rec = {"fn": fn, "args": args, "err": [type(e).__name__, str(e)]}
    if tag:
        rec["tag"] = tag
    cases.append(rec)


# ── build_entry_plan ─────────────────────────────────────────────────────────
F0 = FILTERS[0]
for kw in (   # 자체점검이 고정한 입력(_self_check · _self_check_rec_rule)
    dict(side="LONG", best_bid=2470.00, best_ask=2470.01, recommended_qty=3.4, cap_notional=None, filters=F0),
    dict(side="SHORT", best_bid=2470.00, best_ask=2470.01, recommended_qty=3.4, cap_notional=None, filters=F0),
    dict(side="LONG", best_bid=2470.00, best_ask=2470.01, recommended_qty=14.273, cap_notional=15201.0, filters=F0),
    dict(side="SHORT", best_bid=2526.08, best_ask=2526.09, recommended_qty=2.693, cap_notional=13579.31, filters=F0, existing_notional=7641.6, equity=1064.69, leverage=30.0),
    dict(side="LONG", best_bid=2000.00, best_ask=2000.01, recommended_qty=99.0, cap_notional=20000.0, filters=F0, existing_notional=8000.0, equity=2000.0, leverage=20.0,
         fraction=0.40, fraction_of_equity=True, order_leverage=20.0, cap_label="증거금 상한 50%"),
    dict(side="LONG", best_bid=2470.00, best_ask=2470.01, recommended_qty=6.14, cap_notional=7299.0, filters=F0, fraction=1 / 3),
):
    add("build_entry_plan", kw, tag="selfcheck")
for bad in (dict(side="long"), dict(best_bid=0.0), dict(best_bid=-1.0), dict(best_ask=2469.0), dict(fraction=0.0), dict(fraction=1.5), dict(fraction=-0.2)):
    add("build_entry_plan", {**dict(side="LONG", best_bid=2470.0, best_ask=2470.01, recommended_qty=1.0, cap_notional=None, filters=F0), **bad}, tag="reject")
for i in range(240):
    f = rng.choice(FILTERS)
    tick = f["tick"]
    bid = on_grid(tick, rng.choice([0.5, 2.5, 150.0, 2470.0, 98000.0, 1e7]))
    ask = bid + (tick or 0.01) * rng.choice([0, 1, 1, 2, 5]) if rng.random() > 0.03 else bid * 0.999
    step = f["step"] or 0.001
    rq = rng.choice([0.0, -1.0, on_grid(step, rng.choice([0.01, 1.0, 50.0])), rng.uniform(0, 20), 1e9, 3.4])
    exist = rng.choice([0.0, 0.0, -5.0, rng.uniform(0, 20000), 13579.31])
    cap = rng.choice([None, None, 0.0, rng.uniform(100, 30000), exist, exist + rng.choice([1e-9, 0.01, 1.0]), -100.0])
    kw = dict(side=rng.choice(["LONG", "SHORT"]), best_bid=bid, best_ask=ask, recommended_qty=rq, cap_notional=cap, filters=f,
              symbol="ETHUSDC", existing_notional=exist,
              equity=rng.choice([0.0, -100.0, 1064.69, 2000.0, rng.uniform(100, 1e5)]),
              leverage=rng.choice([0.0, 3.0, 10.0, 20.0, 30.0]),
              fraction=rng.choice([1.0, 1.0, 1 / 3, 0.5, 0.25, 0.1, 0.999, 1e-9, 0.125, 0.005, 0.015, rng.uniform(0.001, 1)]),
              fraction_of_equity=rng.random() < 0.35,
              order_leverage=rng.choice([0.0, 1.0, 2.0, 10.0, 20.0, 125.0]),
              cap_label=rng.choice(["상한", "증거금 상한 50%"]),
              available_margin=rng.choice([None, None, 0.0, -5.0, rng.uniform(1, 5000), 385.0]))
    add("build_entry_plan", kw)
for big in (1e12, 1e15, 1e300):   # 아주 큰 값
    add("build_entry_plan", dict(side="LONG", best_bid=big, best_ask=big, recommended_qty=big, cap_notional=big, filters=F0, equity=big,
                                 fraction=0.5, fraction_of_equity=True, order_leverage=20.0), tag="huge")
add("build_entry_plan", dict(side="LONG", best_bid=0.004, best_ask=0.004, recommended_qty=1.0, cap_notional=10.0, filters=F0, equity=100.0,
                             fraction=0.5, fraction_of_equity=True, order_leverage=20.0), tag="price_floors_to_zero")
add("build_entry_plan", dict(side="LONG", best_bid=0.004, best_ask=0.004, recommended_qty=1.0, cap_notional=None, filters=F0,
                             order_leverage=20.0, available_margin=50.0), tag="price_floors_to_zero")

# ── build_exit_plan ──────────────────────────────────────────────────────────
for kw in (
    dict(position_side="LONG", position_qty=2.693, best_bid=2470.00, best_ask=2470.01, filters=F0, entry_price=2400.0),
    dict(position_side="SHORT", position_qty=3.025, best_bid=2470.00, best_ask=2470.01, filters=F0, entry_price=2500.0),
    dict(position_side="LONG", position_qty=0.0015, best_bid=2470.00, best_ask=2470.01, filters=F0),
    dict(position_side="LONG", position_qty=2.693, best_bid=2470.00, best_ask=2470.01, filters=F0, fraction=0.5),
    dict(position_side="LONG", position_qty=0.01, best_bid=2470.00, best_ask=2470.01, filters=F0, fraction=0.05),
    dict(position_side="LONG", position_qty=2.0, best_bid=2470.00, best_ask=2470.01, filters=F0, vol_bpm=45.0),
):
    add("build_exit_plan", kw, tag="selfcheck")
for bad in (dict(position_side="BOTH"), dict(best_ask=0.0), dict(fraction=0.0), dict(fraction=1.01), dict(fraction=-0.5)):
    add("build_exit_plan", {**dict(position_side="LONG", position_qty=2.0, best_bid=2470.0, best_ask=2470.01, filters=F0), **bad}, tag="reject")
for i in range(170):
    f = rng.choice(FILTERS)
    bid = on_grid(f["tick"], rng.choice([0.5, 150.0, 2470.0, 98000.0]))
    ask = bid + (f["tick"] or 0.01) * rng.choice([0, 1, 3])
    kw = dict(position_side=rng.choice(["LONG", "SHORT"]), position_qty=rng.choice([0.0, -1.0, on_grid(f["step"] or 0.001, rng.choice([0.002, 1.0, 30.0])), rng.uniform(0, 10), 2.693, 3.025]),
              best_bid=bid, best_ask=ask, filters=f, symbol="ETHUSDC",
              entry_price=rng.choice([0.0, -1.0, bid * rng.uniform(0.9, 1.1)]), mark_price=rng.choice([0.0, bid]),
              vol_bpm=rng.choice([None, None, 0.0, -1.0, 2.0, 5.25, 9.97, 12.0, 17.74, 29.999999, 30.0, 45.0, 1e-3, 1e5, rng.uniform(0.5, 60)]),
              fraction=rng.choice([1.0, 1.0, 0.5, 0.1, 0.25, 0.75, 0.99, 1 / 3, 0.005, 0.015, rng.uniform(0.01, 1)]))
    add("build_exit_plan", kw)
for v in (1e300, 1e-300, 1e-213, 1e-200):   # 아주 큰·작은 변동성(파이썬 ** 의 넘침/0 나누기 경로)
    add("build_exit_plan", dict(position_side="LONG", position_qty=1.0, best_bid=2470.0, best_ask=2470.01, filters=F0, vol_bpm=v), tag="pow_edge")

# ── 변동성·마감 ───────────────────────────────────────────────────────────────
for v in (None, 0.0, -3.0, 2.0, 5.25, 9.97, 17.74, 30.0, 100.0, 1e-3, 1e300, 1e-300, 1e-213, 1e-200, 0.123456789):
    add("exit_deadline_sec", {"vol_bpm": v})
for i in range(80):
    n = rng.choice([0, 2, 9, 10, 11, 31, 32, 35])
    px, out = rng.uniform(1, 100000), []
    for _ in range(n):
        px *= math.exp(rng.gauss(0, rng.choice([1e-5, 1e-4, 1e-3, 5e-3])))
        out.append(rng.choice([px] * 12 + [0.0, -1.0]))
    add("realized_vol_bpm", {"closes": out})
add("realized_vol_bpm", {"closes": [100.0] * 31}, tag="selfcheck")
add("realized_vol_bpm", {"closes": [100.0 * (1.001 ** (i % 2)) for i in range(31)]}, tag="selfcheck")

# ── 자동 규칙(commit_rule_l·rec_rule_size·rec_rule_add_order·rec_rule_bracket)은 2026-11-21 원본 PR 61 에서 지워 정답지에서 뺐다 ──

# 동점·부호 있는 0·repr 지수 경계(양성 대조가 처음에 못 잡은 자리를 메운다)
for kw in (dict(existing_notional=-0.0), dict(recommended_qty=-0.0), dict(available_margin=-0.0, order_leverage=20.0), dict(cap_notional=-0.0),
           dict(best_bid=2.5e16, best_ask=2.5e16), dict(best_bid=9.99e15, best_ask=9.99e15, recommended_qty=1e-5),
           dict(best_bid=1.234e-05, best_ask=1.235e-05, filters=FILTERS[6], recommended_qty=1e6),
           dict(best_bid=0.00012345, best_ask=0.00012346, filters=FILTERS[6], recommended_qty=5e5)):
    add("build_entry_plan", {**dict(side="LONG", best_bid=2470.0, best_ask=2470.01, recommended_qty=1.0, cap_notional=None, filters=F0), **kw}, tag="edge")
for kw in (dict(position_qty=-0.0), dict(position_qty=2.0, entry_price=-0.0), dict(best_bid=2.5e16, best_ask=2.5e16),
           dict(best_bid=1.234e-05, best_ask=1.235e-05, filters=FILTERS[6], position_qty=3e6)):
    add("build_exit_plan", {**dict(position_side="SHORT", position_qty=2.0, best_bid=2470.0, best_ask=2470.01, filters=F0), **kw}, tag="edge")
add("split_tp_qty", dict(qty=-0.0, legs=[{"price": 2500.0, "w": 1}], filters=F0, position_side="LONG"), tag="edge")

# ── 선 브래킷 · 분할 익절 ─────────────────────────────────────────────────────
for kw in (dict(position_side="LONG", ref_level=2500.0, tp_level=2550.0, sl_level=2480.0, basis=0.99951),
           dict(position_side="SHORT", ref_level=2500.0, tp_level=2450.0, sl_level=2520.0, basis=0.99951),
           dict(position_side="LONG", ref_level=2500.0, tp_level=3000.0, sl_level=2370.0, basis=1.0),
           dict(position_side="LONG", ref_level=2500.0, tp_level=2500.01, sl_level=2499.99, basis=1.0),
           dict(position_side="LONG", ref_level=2500.0, tp_level=2550.0, sl_level=2500.0, basis=1.0, sl_watch=True),
           dict(position_side="LONG", ref_level=2500.0, tp_level=2480.0, sl_level=2550.0, basis=1.0),
           dict(position_side="LONG", ref_level=2500.0, tp_level=2500.0, sl_level=2480.0, basis=1.0)):
    add("line_bracket", {**kw, "filters": {"tick": 0.01, "step": 0.001, "min_qty": 0.0, "min_notional": 0.0}}, tag="selfcheck")
for i in range(170):
    f = rng.choice(FILTERS)
    ref = rng.choice([2500.0, 2.5, 98000.0, 0.0])
    d = lambda: ref * rng.choice([0.0, 1e-6, 0.001, 0.01, 0.2, 1.0, -0.01, -0.2])  # noqa: E731
    add("line_bracket", dict(position_side=rng.choice(["LONG", "SHORT"]), ref_level=ref, tp_level=ref + d(), sl_level=ref + d(),
                             basis=rng.choice([1.0, 0.99951, 1.00049, 0.0]), filters=f, sl_watch=rng.random() < 0.3))
for args in ((1.0, [{"price": 2600.0, "w": 1}, {"price": 2550.0, "w": 1}, {"price": 2700.0, "w": 2}], "LONG"),
             (0.012, [{"price": 2550.0, "w": 1}, {"price": 2600.0, "w": 1}], "LONG"),
             (0.5, [{"price": 2450.0, "w": 1}, {"price": 2400.0, "w": 3}], "SHORT")):
    add("split_tp_qty", dict(qty=args[0], legs=args[1], filters={"step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0}, position_side=args[2]), tag="selfcheck")
for i in range(190):
    f = rng.choice(FILTERS)
    n = rng.choice([0, 1, 2, 3, 4, 4])
    legs = [{"price": round(rng.uniform(2000, 3000), 2), **({} if rng.random() < 0.1 else {"w": rng.choice([None, 0, 1, 1, 2, 3, -1, 0.5])})} for _ in range(n)]
    if n >= 2 and rng.random() < 0.2:
        legs[1]["price"] = legs[0]["price"]   # 같은 가격 -- 안정 정렬
    add("split_tp_qty", dict(qty=rng.choice([0.0, 1e-13, on_grid(f["step"] or 0.001, rng.choice([0.005, 1.0, 20.0])), rng.uniform(0, 5), 0.012]),
                             legs=legs, filters=f, position_side=rng.choice(["LONG", "SHORT"])))


# ── 무장 판정(bracket_action · line_hit · sl_legs_hit · bracket_sl_hit · sl_up_of) ──
def armed_suite(armed, position_side, positions, account_ok, account_ts, last_bar, touch):
    t = tuple(touch) if touch else None
    lb = tuple(last_bar) if last_bar else None
    return {"action": P.bracket_action(armed, position_side, positions, account_ok, account_ts, lb, t),
            "line_hit": P.line_hit(position_side, armed, t), "sl_legs_hit": P.sl_legs_hit(position_side, armed, t, lb),
            "sl_hit": P.bracket_sl_hit(position_side, lb, armed), "sl_up": P.sl_up_of(position_side, armed)}


LV = 2480.0
for i in range(280):
    side = rng.choice(["LONG", "SHORT"])
    a = {"symbol": rng.choice(["ETHUSDC", "SOLUSDC"]), "armed_at": rng.choice([0.0, 100.0, 1000.0])}
    for k, vals in (("watch_from", [None, 0.0, 50.0, 150.0, 1000.0]), ("lines", [None, True, False]), ("sl_level", [None, 0.0, LV, 2520.0]),
                    ("sl_price", [None, 0.0, 2479.0]), ("tp_level", [None, 0.0, 2550.0, 2450.0]), ("tp_resting", [None, True, False]),
                    ("tp_off", [None, {"at": 1.0}, {}]), ("sl_off", [None, {"at": 1.0}, {}]), ("fired", [None, False, True, "tp"]),
                    ("sl_up", [None, True, False])):
        v = rng.choice(vals)
        if v is not None or rng.random() < 0.5:
            a[k] = v
    if rng.random() < 0.3:
        a["sl_legs"] = [{"level": rng.choice([2480.0, 2460.0, 2520.0]), **({"up": rng.choice([True, False, None])} if rng.random() < 0.4 else {}),
                         **({"fired": rng.choice([True, False])} if rng.random() < 0.3 else {}), "w": 1} for _ in range(rng.choice([0, 1, 2, 3]))]
    if rng.random() < 0.3:
        a["tp_legs"] = [{"level": rng.choice([2550.0, 2600.0, 2450.0]), "resting": rng.choice([True, False]),
                         **({"merged": True} if rng.random() < 0.2 else {})} for _ in range(rng.choice([0, 1, 2]))]
    pos = [{"symbol": rng.choice(["ETHUSDC", "SOLUSDC"]), "side": rng.choice(["LONG", "SHORT"]), "qty": rng.choice([0.0, 1.0, -0.5, None])}
           for _ in range(rng.choice([0, 1, 2]))]
    c = rng.choice([2440.0, 2459.0, 2470.0, 2479.9, 2480.0, 2485.0, 2500.0, 2519.0, 2520.0, 2550.0, 2560.0, 2600.0, 2650.0])
    touch = rng.choice([None, [c - rng.choice([0.0, 5.0, 30.0]), c + rng.choice([0.0, 5.0, 60.0])]])
    last_bar = rng.choice([None, [rng.choice([50.0, 100.0, 100.5, 150.0, 2000.0]), rng.choice([2470.0, 2479.0, 2480.0, 2490.0, 2521.0])]])
    add("armed_suite", dict(armed=a, position_side=side, positions=pos, account_ok=rng.random() < 0.8,
                            account_ts=rng.choice([0.0, 100.0, 104.0, 105.0, 105.5, 2000.0]), last_bar=last_bar, touch=touch), call=armed_suite)

# ── 저장본 합치기 · 옛 키 옮기기 · 청산 심볼 ─────────────────────────────────────
KEYS = ["ETHUSDC:LONG", "ETHUSDC:SHORT", "SOLUSDC:LONG"]
for i in range(90):
    def arm():
        d = {"armed_at": rng.choice([1.0, 5.0, 9.0])}
        for k in rng.sample(list(P.OWNED_BY_SAVED) + ["tp_resting", "fired", "retry_after", "exiting"], rng.randint(0, 6)):
            d[k] = rng.choice([None, True, False, 1.0, "tp", {"at": 2.0}, [1, 2]])
        return d
    cur = {k: arm() for k in rng.sample(KEYS, rng.randint(0, 3))}
    mine = {k: arm() for k in rng.sample(KEYS, rng.randint(0, 3))}
    seen = {k: rng.choice([1.0, 5.0, 9.0]) for k in rng.sample(KEYS, rng.randint(0, 3))}
    for k in seen:   # 같은 armed_at 을 자주 맞춰 «적용» 경로가 돌게
        if k in cur and rng.random() < 0.7:
            seen[k] = cur[k]["armed_at"]
    add("bracket_merge", dict(current=cur, mine=mine, seen_at=seen))
add("bracket_merge", dict(current={"k": {"armed_at": 5.0, "sl_off": {"at": 9.0}}}, mine={"k": {"armed_at": 5.0, "retry_after": 1}}, seen_at={"k": 5.0}), tag="selfcheck")
add("bracket_merge", dict(current={"k": {"armed_at": 5.0}}, mine={"k": {"armed_at": 5.0, "sl_off": {"at": 9.0}, "retry_after": 1}}, seen_at={"k": 5.0}), tag="selfcheck")
for st in ({"LONG": {"symbol": "ETHUSDC", "armed_at": 1.0}}, {"SHORT": {"armed_at": 1.0}}, {"ETHUSDC:LONG": {"a": 1}, "SHORT": {"symbol": "SOLUSDC"}}, {}):
    add("bracket_rekey", dict(state=st))   # 옛 키·새 키 충돌은 넣지 않는다(plan.rs bracket_rekey 의 ponytail)
POS = [{"symbol": "ETHUSDT", "side": "LONG", "qty": 0.946}, {"symbol": "ETHUSDT", "side": "SHORT", "qty": 1.5}, {"symbol": "ETHUSDC", "side": "LONG", "qty": 2.0},
       {"symbol": "SOLUSDC", "side": "LONG", "qty": None}]
for i in range(60):
    add("resolve_exit_position", dict(positions=rng.sample(POS, rng.randint(0, 4)), position_side=rng.choice(["LONG", "SHORT"]),
                                      candidates=rng.sample(["ETHUSDC", "ETHUSDT", "SOLUSDC"], rng.randint(0, 3))),
        call=lambda **k: list(P.resolve_exit_position(**k)))

out = ROOT / "rtorder" / "tests" / "fixtures" / "plan_fixture.json"
text = json.dumps({"python": sys.version.split()[0], "cases": cases}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
out.write_text(text, encoding="utf-8")
by: dict[str, int] = {}
for c in cases:
    by[c["fn"]] = by.get(c["fn"], 0) + 1
print(f"{len(cases)}건 · {len(text.encode()) / 1e6:.3f}MB · 오류 경로 {sum('err' in c for c in cases)}건 ·", by)
