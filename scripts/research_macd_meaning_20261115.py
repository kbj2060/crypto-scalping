#!/usr/bin/env python3
"""MACD 가 «뜻하는 대로» 움직이는가 — 일반 모멘텀 대비 (2026-11-15). 사전등록 docs/prereg/macd_meaning_20261115.md (39eedc03).

  selftest   손계산으로 교차·0선·다이버전스 사건과 방향 단언
  run        4코인 × 15분·1시간 실제 + 기하 위약 20벌 + 모멘텀 맞춤 대조(실제) → tmp/macd_meaning_20261115/
경주·위약·피벗·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/macd_meaning_20261115"
TFS = {"15m": 15, "1h": 60}
GAP, NPIV, MOM = 60, 5, 12


def macd(c: np.ndarray):
    s = pd.Series(c)
    m = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).to_numpy()
    sig = pd.Series(m).ewm(span=9, adjust=False).mean().to_numpy()
    m[:60] = np.nan; sig[:60] = np.nan                       # 데움
    return m, sig


def crosses(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """봉 t 에 a 가 b 를 위로(+1)/아래로(−1) 넘음 = a[t−1] ≤ b[t−1] 이고 a[t] > b[t]."""
    d = a - b; out = []
    for t in range(1, len(d)):
        if np.isfinite(d[t - 1]) and np.isfinite(d[t]):
            if d[t - 1] <= 0 < d[t]:
                out.append((t, 1))
            elif d[t - 1] >= 0 > d[t]:
                out.append((t, -1))
    return out


def divergences(x: np.ndarray, m: np.ndarray, high: bool) -> list[tuple[int, str]]:
    """새 스윙(5봉) 고점이 직전 60봉 안 스윙 고점보다 높을 때: MACD 가 낮으면 'D', 아니면 'P3'. 저점 대칭. 알려지는 시각 = p+5."""
    piv = np.flatnonzero(SM.pivots(x, NPIV, high)); out = []
    for i in range(1, len(piv)):
        p, q = piv[i], piv[i - 1]
        if p - q > GAP or not (np.isfinite(m[p]) and np.isfinite(m[q])):
            continue
        newer = x[p] > x[q] if high else x[p] < x[q]
        if not newer:
            continue
        div = m[p] < m[q] if high else m[p] > m[q]
        out.append((int(p), "D" if div else "P3"))
    return out


def events(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    m, sig = macd(c); n = len(c); R = []
    def add(cell, i, d, k=1.0):
        a2 = atr.copy(); a2[i] = k * atr[i]
        y = SM.first_hit_race(h, l, c, a2, i, int(d), 100)
        if y >= 0:
            R.append(dict(cell=cell, tk=int(t[i]) + B["ms"], y=float(y)))
    for i, d in crosses(m, sig):
        if i + 1 < n and atr[i] > 0:
            add("M1", i, d); add("M1_2atr", i, d, 2.0)
    for i, d in crosses(m, np.zeros(n)):
        if i + 1 < n and atr[i] > 0:
            add("M2", i, d); add("M2_2atr", i, d, 2.0)
    for high in (True, False):
        x = h if high else l
        for p, kind in divergences(x, m, high):
            kk = p + NPIV
            if kk + 1 < n and atr[kk] > 0:
                if kind == "D":
                    add("M3", kk, -1 if high else 1)
                elif with_p3:
                    add("P3M3", kk, -1 if high else 1)
    if with_p3:                                               # 모멘텀 맞춤: 모든 봉 · 방향 = 직전 12봉 수익 부호
        mom = np.sign(c[MOM:] - c[:-MOM])
        for i in range(max(MOM, 60), n - 1):
            d = mom[i - MOM]
            if d != 0 and atr[i] > 0:
                add("P3MOM", i, d)
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            for tf, mnt in TFS.items():
                ev = pd.DataFrame(events(SM.bars(kk, mnt), draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                else:
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS, SM.CELLS = TFS, tuple(TFS), ["M1", "M2", "M3"]         # 주 판정 6칸 Holm
    SM.P2OF = {"M1": "P3MOM", "M2": "P3MOM", "M3": "P3M3"}                      # judge 의 «P2» 자리 = 이번 P3(모멘텀 맞춤)
    main = SM.judge(E, PA)
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), ["M1_2atr", "M2_2atr"], {"M1_2atr": "P3MOM", "M2_2atr": "P3MOM"}   # 보조(판정 없음 · P3 는 1 ATR 판이라 참고만)
    aux = SM.judge(E, PA)
    json.dump(dict(main=main, aux=aux), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in {**main, **aux}.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})


def selftest() -> None:
    a = np.array([0, 1, 2, 1, 0, -1.]); b = np.zeros(6)
    assert crosses(a, b) == [(1, 1), (5, -1)]                 # 0→1 위로 · 0→−1 아래로(0 에 닿은 봉은 «아직»)
    # 다이버전스: 고점 10(봉 5) → 더 높은 11(봉 15), MACD 는 2 → 1 (낮아짐) = D
    x = np.full(30, 5.0); x[5] = 10; x[15] = 11; m = np.zeros(30); m[5] = 2; m[15] = 1
    assert divergences(x, m, True) == [(15, "D")]
    m2 = m.copy(); m2[15] = 3
    assert divergences(x, m2, True) == [(15, "P3")]           # MACD 도 높아짐 = 다이버전스 아님
    x3 = x.copy(); x3[15] = 9
    assert divergences(x3, m, True) == []                     # 새 고점이 아님
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
