#!/usr/bin/env python3
"""피보나치 되돌림이 «뜻하는 대로» 움직이는가 (2026-11-15). 사전등록 docs/prereg/fibonacci_retracement_meaning_20261115.md (1a46a195, 계산 전).

  selftest   손계산 봉으로 다리·이미 닿은 수준 제외·끝(새 고점/깨짐)·F2 띠 단언
  run        4코인 실제 + 기하 위약 20벌 → tmp/fibonacci_meaning_20261115/
위약·경주·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/fibonacci_meaning_20261115"
FIB, NONFIB = (0.382, 0.5, 0.618, 0.786), (0.30, 0.45, 0.55, 0.70)
GRID = np.round(np.arange(0.30, 0.901, 0.02), 2)                     # 보조 곡선(15분 N50)
W_TOUCH, W_F2, MIN_ATR = 100, 200, 3.0
SETS = {"15m": (15, (50, 5)), "1h": (60, (50,))}


def legs(B: dict, N: int) -> list[tuple]:
    """(p, K=p+N, d, lo, hi): d=+1 상승 다리(직전 스윙 고점 q~p 사이 최저 → h[p]), −1 대칭. 크기 ≥ 3 ATR(K)."""
    h, l, atr = B["h"], B["l"], B["atr"]; n = len(h); out = []
    for d, piv, src, opp in ((1, SM.pivots(h, N, True), h, l), (-1, SM.pivots(l, N, False), l, h)):
        ix = np.flatnonzero(piv)
        for q, p in zip(ix[:-1], ix[1:]):
            K = p + N
            if K >= n - 1 or p - q < 2 or not atr[K] > 0:
                continue
            seg = opp[q + 1: p]
            start = seg.min() if d > 0 else seg.max()
            lo, hi = (start, src[p]) if d > 0 else (src[p], start)
            if hi - lo >= MIN_ATR * atr[K]:
                out.append((int(p), int(K), d, float(lo), float(hi)))
    return out


def track(B: dict, p: int, K: int, d: int, lo: float, hi: float) -> tuple[int, bool]:
    """K 뒤 처음 새 극값(다리 방향) 또는 시작점 깨짐 봉 e · 새 극값이면 True. 없으면 (K+W_F2, False)."""
    h, l = B["h"], B["l"]; end = min(K + W_F2, len(h) - 1)
    a, b = (h[K + 1: end + 1] > hi, l[K + 1: end + 1] < lo) if d > 0 else (l[K + 1: end + 1] < lo, h[K + 1: end + 1] > hi)
    ia = int(np.argmax(a)) if a.any() else 10**9; ib = int(np.argmax(b)) if b.any() else 10**9
    if ia == 10**9 and ib == 10**9:
        return end, False
    return K + 1 + min(ia, ib), ia <= ib                              # 같은 봉이면 새 극값 쪽(주장에 불리: 그 봉 터치 제외)


def events(B: dict, N: int, curve: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; R = []
    for p, K, d, lo, hi in legs(B, N):
        e, newx = track(B, p, K, d, lo, hi)
        rng_ = hi - lo; tk = int(t[K]) + B["ms"]
        pre = l[p + 1: K + 1].min() if d > 0 else h[p + 1: K + 1].max()     # 알려지기 전 이미 간 깊이
        lim = min(e if newx else e + 1, K + 1 + W_TOUCH)
        ratios = [(r, "F1") for r in FIB] + [(r, "P2F1") for r in NONFIB] + ([(r, f"G{r:.2f}") for r in GRID] if curve else [])
        for r, cell in ratios:
            lev = hi - r * rng_ if d > 0 else lo + r * rng_
            if (d > 0 and pre <= lev) or (d < 0 and pre >= lev):
                continue
            seg = (l[K + 1: lim] <= lev) if d > 0 else (h[K + 1: lim] >= lev)
            if seg.any():
                s = K + 1 + int(np.argmax(seg))
                y = SM.first_hit_race(h, l, c, atr, s, d, 100)
                if y >= 0:
                    R.append(dict(cell=cell, tk=tk, y=float(y), r=r))
        if newx:                                                          # F2: 새 극값으로 이어진 다리의 가장 깊은 되돌림
            deep = l[p + 1: e].min() if d > 0 else h[p + 1: e].max()
            q = (hi - deep) / rng_ if d > 0 else (deep - lo) / rng_
            if 0 <= q < 1:
                R.append(dict(cell="F2", tk=tk, y=float(any(abs(q - f) <= 0.02 for f in FIB)), r=q))
                R.append(dict(cell="P2F2", tk=tk, y=0.5 * float(any(0.02 < abs(q - f) <= 0.06 for f in FIB)), r=q))
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            for tf, (m, Ns) in SETS.items():
                B = SM.bars(kk, m)
                for N in Ns:
                    ev = pd.DataFrame(events(B, N, curve=(tf == "15m" and N == 50)))
                    if ev.empty:
                        continue
                    ev["cell"] = ev.cell + f"_N{N}"
                    ev = ev.assign(coin=coin, tf=tf, draw=draw)
                    if draw < 0:
                        rows.append(ev)
                    else:
                        aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS = {"15m": 15, "1h": 60}, ("15m",)
    SM.CELLS = ["F1_N50", "F2_N50"]
    SM.P2OF = {f"{a}_N{N}": f"P2{a}_N{N}" for a in ("F1", "F2") for N in (50, 5)}
    main = SM.judge(E, PA)                                                # Holm 2칸 = 15분 N50
    SM.CELLS = ["F1_N5", "F2_N5"]; SM.MAIN_TFS = ()
    aux = SM.judge(E, PA)
    res = dict(main={k: v for k, v in main.items() if k in ("15m|F1_N50", "15m|F2_N50")},
               aux={**{k: v for k, v in aux.items() if k.startswith("15m|")}, **{k: v for k, v in main.items() if k.startswith("1h|")}})
    # 수준별·곡선 (15분 N50)
    r15 = E[(E.tf == "15m") & E.cell.str.endswith("_N50")]; p15 = PA[(PA.tf == "15m") & PA.cell.str.endswith("_N50")]
    def rate_by(cells):
        out = {}
        for cell in cells:
            x = r15[r15.cell == cell]; pz = p15[p15.cell == cell]
            if len(x):
                out[cell] = dict(n=int(len(x)), real=round(x.y.mean(), 4), p1=round(pz["sum"].sum() / pz["count"].sum(), 4))
        return out
    lv = {}
    for r in FIB + NONFIB:
        x = r15[(r15.cell.isin(["F1_N50", "P2F1_N50"])) & (np.isclose(r15.r, r))]
        lv[str(r)] = dict(n=int(len(x)), real=round(x.y.mean(), 4))
    res["per_level_real_15m_N50"] = lv
    res["curve_15m_N50"] = rate_by([f"G{r:.2f}_N50" for r in GRID])
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k, v in {**res["main"], **res["aux"]}.items():
        print(k, {x: v.get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("levels", lv)


def selftest() -> None:
    # 다리: 스윙 고점 둘(봉 5, 20) 사이 최저(봉 12 = 90) → 상승 다리 [90, 120] (N=2, ATR 1)
    n = 40
    h = np.interp(np.arange(n), [0, 5, 12, 20, 39], [100, 110, 91, 120, 101])   # 봉우리 5·20 만(평평한 동점 없음)
    l = h - 1; l[12] = 90
    c = h - 0.5; B = dict(h=h, l=l, c=c, atr=np.ones(n), t=np.arange(n) * 900_000, ms=900_000)
    L = legs(B, 2)
    up = [x for x in L if x[2] == 1 and x[0] == 20]
    assert up and up[0][3] == 90 and up[0][4] == 120 and up[0][1] == 22
    # 끝: 봉 30 에 새 고점 → (30, True)
    h2 = h.copy(); h2[30] = 121; B2 = dict(B, h=h2)
    assert track(B2, 20, 22, 1, 90.0, 120.0) == (30, True)
    l3 = l.copy(); l3[28] = 89; B3 = dict(B, l=l3)
    assert track(B3, 20, 22, 1, 90.0, 120.0) == (28, False)
    # 이미 닿은 수준 제외: p~K 사이 저가가 0.382 수준(120−11.46=108.54) 아래면 F1 0.382 사건 없음
    l4 = l.copy(); l4[21] = 108.0; B4 = dict(B, l=l4)
    ev = events(B4, 2, False)
    assert not any(r["cell"] == "F1" and r["r"] == 0.382 and r["tk"] == 22 * 900_000 + 900_000 for r in ev)
    # F2 띠: 0.62 → 피보나치 안(1), 옆 띠 0 · 0.66 → 피보나치 밖, 옆 띠(0.5)
    fib = lambda q: float(any(abs(q - f) <= 0.02 for f in FIB)); side = lambda q: 0.5 * float(any(0.02 < abs(q - f) <= 0.06 for f in FIB))
    assert (fib(0.62), side(0.62)) == (1.0, 0.0) and (fib(0.66), side(0.66)) == (0.0, 0.5)
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
