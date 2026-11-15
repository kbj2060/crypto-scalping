#!/usr/bin/env python3
"""묶음 5 — RSI(2) 추세 속 눌림 · Aroon(25) 교차 · 터틀 수프 (2026-11-16). 사전등록 docs/prereg/connors_aroon_soup_bundle_20261116.md (0966f228).

  selftest   손계산 봉으로 Aroon·터틀 수프·RSI2 사건 단언
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 지표별 실제 가격 대조 → tmp/connors_aroon_soup_bundle_20261116/
실행·판정 뼈대는 research_classic_indicator_bundle_20261115.run 그대로(events·KEYS·OUT 만 바꿈).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402
from research_rsi_divergence_meaning_20261115 import rsi  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/connors_aroon_soup_bundle_20261116"
KEYS = ["G1", "G2", "G3"]


def aroon(h, l, n=25):
    """(Up, Down) 0~100 = 100·(n − 최근 n봉 최고/최저 뒤 지난 봉 수)/n. 같은 값이면 가장 최근."""
    up, dn = np.full(len(h), np.nan), np.full(len(h), np.nan)
    up[n:] = 100 * (n - np.argmax(swv(h, n + 1)[:, ::-1], axis=1)) / n
    dn[n:] = 100 * (n - np.argmin(swv(l, n + 1)[:, ::-1], axis=1)) / n
    return up, dn


def prior_ext(x, n, high):
    """봉 t 의 직전 n봉(t−n..t−1) 최고(최저)값과 그 봉이 t 에서 몇 봉 전인지. 같은 값이면 가장 최근."""
    v, age = np.full(len(x), np.nan), np.full(len(x), -1)
    W = swv(x, n)[:-1][:, ::-1]                               # 행 t−n 의 창 = x[t−n..t−1] 뒤집음
    k = np.argmax(W, axis=1) if high else np.argmin(W, axis=1)
    v[n:] = W[np.arange(len(W)), k]; age[n:] = k + 1
    return v, age


def soup(h, l, c, n=20):
    """(터틀 수프 방향, P3 돈치안 종가 돌파 반대 방향)."""
    lo, alo = prior_ext(l, n, False); hi, ahi = prior_ext(h, n, True)
    with np.errstate(invalid="ignore"):
        S = np.where((l < lo) & (alo >= 4) & (c > lo), 1, np.where((h > hi) & (ahi >= 4) & (c < hi), -1, 0))
        dn, upb = c < lo, c > hi
        P = np.where(dn & ~np.r_[False, dn[:-1]], 1, np.where(upb & ~np.r_[False, upb[:-1]], -1, 0))
    return S, P


def events(B: dict, tf: str, with_p3: bool) -> dict:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; n = len(c)
    R = dict(cell=[], tk=[], y=[])
    def add(cell, i, d):
        if i < CB.WARM or i + 1 >= n or d == 0 or not atr[i] > 0:
            return
        for suf, k in (("", 1.0), ("_2atr", 2.0)):
            y = SM.first_hit_race(h, l, c, atr if k == 1.0 else atr * k, int(i), int(d), 100)
            if y >= 0:
                R["cell"].append(cell + suf); R["tk"].append(int(t[i]) + B["ms"]); R["y"].append(float(y))
    def fire(cell, x):
        for i in np.flatnonzero(x):
            add(cell, i, x[i])
    r2, tr = rsi(c, 2), np.sign(c - CB.sma(c, 200))
    with np.errstate(invalid="ignore"):
        G1 = np.where((tr == 1) & (CB.cross(r2, np.full(n, 10.0)) == -1), 1, np.where((tr == -1) & (CB.cross(r2, np.full(n, 90.0)) == 1), -1, 0))
    up, dn = aroon(h, l); G2 = CB.cross(up, dn)
    G3, P3 = soup(h, l, c)
    fire("G1", G1); fire("G2", G2); fire("G3", G3)
    if with_p3:
        fire("P3G3", P3)
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            if 30 <= r2[i] <= 70:
                add("P3G1", i, tr[i])
            add("P3G2", i, np.sign(c[i] - c[i - 25]))
    return R


def selftest() -> None:
    h = np.r_[np.arange(30.0), 28.5, 28.0]; l = h - 1
    up, dn = aroon(h, l)
    assert up[29] == 100 and up[31] == 92 and dn[29] == 0                         # 봉 29 최고 → 2봉 뒤 100·23/25
    lo = np.r_[np.full(10, 100.0), 95.0, np.full(15, 100.0), 94.0]; hi = lo + 2; cl = lo + 1
    cl[26] = 95.5                                                                # 봉 26 저가 94 < 직전 20봉 최저 95(16봉 전) · 종가 95.5 > 95
    S, P = soup(hi, lo, cl)
    assert S[26] == 1 and (S != 0).sum() == 1 and P[26] == 0
    cl2 = cl.copy(); cl2[26] = 94.5; S2, P2 = soup(hi, lo, cl2)
    assert S2[26] == 0 and P2[26] == 1                                           # 종가도 아래 = 돈치안 돌파 → 대조(반대)
    lo3 = lo.copy(); lo3[24] = 95.0; S3, _ = soup(hi, lo3, cl)
    assert S3[26] == 0                                                           # 직전 최저가 2봉 전(가장 최근) → 4봉 조건 위반
    c = np.r_[np.linspace(100, 200, 260), 190.0]; r = rsi(c, 2)
    assert r[259] > 90 and r[260] < 10
    print("selftest ok")


def run() -> None:
    CB.events, CB.KEYS, CB.OUT = events, KEYS, OUT
    CB.run()


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
