#!/usr/bin/env python3
"""라운드 넘버가 «뜻하는 대로» 움직이는가 (2026-11-15). 사전등록 docs/prereg/round_numbers_meaning_20261115.md (fc512857, 계산 전).

  selftest   손계산 봉으로 간격·닿음·넘음·고저 몰림 단언
  run        4코인 실제 + 기하 위약 20벌 → tmp/round_numbers_meaning_20261115/{events.parquet,placebo_agg.parquet,result.json}
경주·위약·피벗·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/round_numbers_meaning_20261115"
FRESH, WR, BAND = 16, 100, 0.02
OFFS = (0.3, 0.7)


def steps(c: np.ndarray) -> np.ndarray:
    return 10.0 ** (np.floor(np.log10(c)) - 1)


@njit(cache=True)
def touch_events(h, l, c, atr, st, off, fresh, wr):
    """R1: 앞 fresh 봉 동안 안 닿던 수준(k+off)·간격 에 처음 닿은 봉 → 다가온 반대쪽 1 ATR 먼저면 1."""
    n = len(c); out = np.empty((2 * n, 2)); m = 0
    for t in range(fresh, n - 1):
        s = st[t]
        for k in range(int(math.ceil(l[t] / s - off)), int(math.floor(h[t] / s - off)) + 1):
            L = (k + off) * s
            ok = True
            for j in range(t - fresh, t):
                if l[j] <= L <= h[j]:
                    ok = False; break
            if not ok or c[t - 1] == L:
                continue
            d = -1 if c[t - 1] < L else 1                   # 아래에서 올라와 닿음 → 되돌 쪽 = 아래
            y = SM.first_hit_race(h, l, c, atr, t, d, wr)
            if y >= 0:
                out[m, 0] = t; out[m, 1] = y; m += 1
    return out[:m]


@njit(cache=True)
def cross_events(h, l, c, atr, st, off, fresh, wr):
    """R2: 앞 fresh 봉 종가가 모두 한쪽이다가 종가가 수준을 넘은 봉 → 넘은 방향 1 ATR 먼저면 1."""
    n = len(c); out = np.empty((2 * n, 2)); m = 0
    for t in range(fresh, n - 1):
        s = st[t]; a, b = min(c[t - 1], c[t]), max(c[t - 1], c[t])
        for k in range(int(math.floor(a / s - off)) + 1, int(math.ceil(b / s - off))):
            L = (k + off) * s
            if not (a < L < b):
                continue
            d = 1 if c[t] > L else -1
            ok = True
            for j in range(t - fresh, t):
                if (d == 1 and c[j] >= L) or (d == -1 and c[j] <= L):
                    ok = False; break
            if not ok:
                continue
            y = SM.first_hit_race(h, l, c, atr, t, d, wr)
            if y >= 0:
                out[m, 0] = t; out[m, 1] = y; m += 1
    return out[:m]


def near(x: np.ndarray, st: np.ndarray, off: float) -> np.ndarray:
    f = x / st - off
    return np.abs(f - np.round(f)) <= BAND


def events(k: pd.DataFrame) -> list[dict]:
    B = SM.bars(k, 15); h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    st = steps(c); R = []
    def add(cell, arr):
        R.extend(dict(cell=cell, tk=int(t[int(i)]) + B["ms"], y=float(y)) for i, y in arr)
    for off, pre in ((0.0, ""),) + tuple((o, "P2") for o in OFFS):
        add(pre + "R1", touch_events(h, l, c, atr, st, off, FRESH, WR))
        add(pre + "R2", cross_events(h, l, c, atr, st, off, FRESH, WR))
    for hi in (True, False):                                    # R3: 큰 스윙 고점·저점(앞뒤 50봉)
        x = h if hi else l
        p = np.flatnonzero(SM.pivots(x, 50, hi))
        for off, pre in ((0.0, ""),) + tuple((o, "P2") for o in OFFS):
            add(pre + "R3", np.c_[p, near(x[p], st[p], off)])
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            ev = pd.DataFrame(events(kk)).assign(coin=coin, tf="15m", draw=draw)
            if draw < 0:
                rows.append(ev)
            else:
                ev = ev[~ev.cell.str.startswith("P2")]
                aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {"15m": 15}, ("15m",), ["R1", "R2", "R3"]
    SM.P2OF = {"R1": "P2R1", "R2": "P2R2", "R3": "P2R3"}
    res = {kk.split("|")[1]: v for kk, v in SM.judge(E, PA).items()}
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for c_ in ("R1", "R2", "R3"):
        print(c_, {x: res[c_].get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})


def selftest() -> None:
    assert steps(np.array([2345.0, 61000.0, 150.0, 15.0, 0.52, 2.3])).tolist() == [100.0, 1000.0, 10.0, 1.0, 0.01, 0.1]
    # 닿음: 2,050 → 2,100 에 처음 닿고(아래에서) 1 ATR(=10) 아래로 먼저 → 되돎 1
    n = 40; c = np.full(n, 2050.0); h = c + 2; l = c - 2; atr = np.full(n, 10.0)
    h[20] = 2101; c[20] = 2098; l[21] = 2085; h[21] = 2099; c[21] = 2088
    st = steps(c)
    r = touch_events(h, l, c, atr, st, 0.0, 16, 100)
    assert r.tolist() == [[20.0, 1.0]], r
    assert len(touch_events(h, l, c, atr, st, 0.3, 16, 100)) == 0          # 2,130 은 안 닿음
    # 앞 16봉에 이미 닿았으면 사건 아님
    h2 = h.copy(); h2[10] = 2100.5
    assert len(touch_events(h2, l, c, atr, st, 0.0, 16, 100)) == 0
    # 넘음: 종가 2,095 → 2,105 (앞 16봉 종가 전부 아래) → 위로 1 ATR(2,115) 먼저 → 가속 1
    c3 = np.full(n, 2095.0); h3 = c3 + 2; l3 = c3 - 2
    c3[20] = 2105; h3[20] = 2106; l3[20] = 2094; c3[21] = 2105; h3[21] = 2107; l3[21] = 2103; h3[22] = 2116; c3[22] = 2114; l3[22] = 2104
    r3 = cross_events(h3, l3, c3, atr, steps(c3), 0.0, 16, 100)
    assert r3.tolist() == [[20.0, 1.0]], r3
    # 몰림: 2,001 (2,000 의 ±2 = 간격 100 의 2%) 은 가깝다 · 2,030 은 0.3 대조에 가깝다
    assert near(np.array([2001.0, 2030.0]), np.array([100.0, 100.0]), 0.0).tolist() == [True, False]
    assert near(np.array([2001.0, 2030.0]), np.array([100.0, 100.0]), 0.3).tolist() == [False, True]
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
