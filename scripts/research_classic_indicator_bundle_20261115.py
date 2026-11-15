#!/usr/bin/env python3
"""고전 지표 6종 묶음 의미 검증 (2026-11-15). 사전등록 docs/prereg/classic_indicator_bundle_20261115.md (a261404e).

  selftest   지표별 손계산 사건·방향 단언
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 지표별 실제 가격 대조 → tmp/classic_indicator_bundle_20261115/
경주·위약·판정은 research_smc_ict_meaning_20261115, 수준 다가옴은 research_anchored_vwap_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_anchored_vwap_meaning_20261115 as AV  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/classic_indicator_bundle_20261115"
TFS = {"15m": 15, "1h": 60, "4h": 240}                     # 4시간 = 보조(Holm 밖)
STEP = {"15m": 4, "1h": 1, "4h": 1}
WARM = 250
KEYS = ["K1", "K2", "K3", "K4", "K5", "K6"]


def sma(x, w):
    return pd.Series(x).rolling(w).mean().to_numpy()


def cross(a, b):
    """봉 t 에 a 가 b 를 위로 넘으면 +1, 아래로 −1, 아니면 0."""
    d = a - b; dp = np.r_[np.nan, d[:-1]]
    with np.errstate(invalid="ignore"):
        return np.where((dp <= 0) & (d > 0), 1, np.where((dp >= 0) & (d < 0), -1, 0))


def stoch(h, l, c):
    LL, HH = pd.Series(l).rolling(14).min().to_numpy(), pd.Series(h).rolling(14).max().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        fk = 100 * (c - LL) / (HH - LL)
    K = sma(fk, 3); return K, sma(K, 3)


def boll(c):
    m, sd = sma(c, 20), pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    lo, hi = m - 2 * sd, m + 2 * sd
    cp, lop, hip = np.r_[np.nan, c[:-1]], np.r_[np.nan, lo[:-1]], np.r_[np.nan, hi[:-1]]
    with np.errstate(invalid="ignore"):
        return np.where((c < lo) & (cp >= lop), 1, np.where((c > hi) & (cp <= hip), -1, 0))


@njit(cache=True)
def ha_color(o, h, l, c):
    n = len(c); hac = (o + h + l + c) / 4; hao = np.empty(n); hao[0] = (o[0] + c[0]) / 2
    for i in range(1, n):
        hao[i] = (hao[i - 1] + hac[i - 1]) / 2
    return np.sign(hac - hao)


def color_flip(col):
    """같은 색 ≥3 뒤 색 바뀜 → 새 색, 아니면 0."""
    out = np.zeros(len(col))
    s = (col[3:] != 0) & (col[3:] != col[2:-1]) & (col[2:-1] != 0) & (col[2:-1] == col[1:-2]) & (col[1:-2] == col[:-3])
    out[3:][s] = col[3:][s]
    return out


@njit(cache=True)
def psar(h, l, af0=0.02, afmax=0.2):
    n = len(h); tr = np.zeros(n); trend = 1; sar = l[0]; ep = h[0]; af = af0
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if trend == 1:
            sar = min(sar, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < sar:
                trend = -1; sar = ep; ep = l[i]; af = af0
            elif h[i] > ep:
                ep = h[i]; af = min(af + af0, afmax)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > sar:
                trend = 1; sar = ep; ep = h[i]; af = af0
            elif l[i] < ep:
                ep = l[i]; af = min(af + af0, afmax)
        tr[i] = trend
    return tr


def pivot_levels(t, h, l, c):
    """봉별 전날 UTC 일간 P·R1·S1 (3, n)."""
    day = t // 86_400_000
    g = pd.DataFrame(dict(day=day, h=h, l=l, c=c)).groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    P = (g.h + g.l + g.c) / 3; R1, S1 = 2 * P - g.l, 2 * P - g.h
    prev = pd.DataFrame(dict(P=P, R1=R1, S1=S1)).shift(1).reindex(day)
    return prev.to_numpy().T


def events(B: dict, tf: str, with_p3: bool) -> list[dict]:
    h, l, c, o, atr, t = B["h"], B["l"], B["c"], B["o"], B["atr"], B["t"]; n = len(c); R = []
    def add(cell, i, d):
        if i < WARM or i + 1 >= n or d == 0 or not atr[i] > 0:
            return
        for suf, k in (("", 1.0), ("_2atr", 2.0)):
            y = SM.first_hit_race(h, l, c, atr if k == 1.0 else atr * k, int(i), int(d), 100)
            if y >= 0:
                R.append(dict(cell=cell + suf, tk=int(t[i]) + B["ms"], y=float(y)))
    K, D = stoch(h, l, c); x = cross(K, D)
    for i in np.flatnonzero(x):
        z = (x[i] == 1 and D[i] < 20) or (x[i] == -1 and D[i] > 80)
        if z:
            add("K1", i, x[i])
        elif with_p3 and 20 <= D[i] <= 80:
            add("P3K1", i, x[i])
    b = boll(c)
    for i in np.flatnonzero(b):
        add("K2", i, b[i])
    f = color_flip(ha_color(o, h, l, c))
    for i in np.flatnonzero(f):
        add("K3", i, f[i])
    ps = psar(h, l); fl = np.r_[0, np.where(ps[1:] != ps[:-1], ps[1:], 0)]
    for i in np.flatnonzero(fl):
        add("K4", i, fl[i])
    g = cross(sma(c, 50), sma(c, 200))
    for i in np.flatnonzero(g):
        add("K5", i, g[i])
    lv = pivot_levels(t, h, l, c); w = (lv[1] - lv[2]) / 4
    for L in lv:
        for i, s in AV.approaches(c, atr, L, np.r_[np.nan, L[:-1]]):
            add("K6", i, s)
        if with_p3:
            for sg in (1, -1):
                Ls = L + sg * w
                for i, s in AV.approaches(c, atr, Ls, np.r_[np.nan, Ls[:-1]]):
                    add("P3K6", i, s)
    if with_p3:
        f2 = color_flip(np.sign(c - o))                       # K3 대조: 보통 봉 색
        for i in np.flatnonzero(f2):
            add("P3K3", i, f2[i])
        for i in range(WARM, n - 1, STEP[tf]):                # K2·K4·K5 대조: 되돌림·모멘텀
            add("P3K2", i, -np.sign(c[i] - c[i - 20])); add("P3K4", i, np.sign(c[i] - c[i - 12])); add("P3K5", i, np.sign(c[i] - c[i - 50]))
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            for tf, mnt in TFS.items():
                ev = pd.DataFrame(events(SM.bars(kk, mnt), tf, draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                elif len(ev):
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS = TFS; main, aux = {}, {}
    for key in KEYS:                                          # 지표별 Holm 2칸(15분·1시간)
        SM.MAIN_TFS, SM.CELLS, SM.P2OF = ("15m", "1h"), [key], {key: "P3" + key}
        main.update(SM.judge(E, PA))
        SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), [key + "_2atr"], {key + "_2atr": "P3" + key + "_2atr"}
        aux.update(SM.judge(E, PA))
    json.dump(dict(main=main, aux=aux), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in {**main, **aux}.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})


def selftest() -> None:
    assert list(cross(np.array([0, 1, 2, 1, -1.]), np.zeros(5))) == [0, 1, 0, 0, -1]
    c = np.r_[np.full(30, 100.0), 90.0, 95.0]
    b = boll(c); assert b[30] == 1 and b[31] == 0                          # 하단 밖 첫 봉만
    col = np.array([1, 1, 1, -1, -1, 1, 1, 1, 1, -1.])
    assert list(color_flip(col)) == [0, 0, 0, -1, 0, 0, 0, 0, 0, -1]
    up = np.r_[np.linspace(100, 120, 40), np.linspace(120, 100, 40)]
    tr = psar(up + 0.2, up - 0.2)
    assert tr[30] == 1 and tr[-1] == -1 and (np.diff(tr[1:]) != 0).sum() == 1     # 꼭대기 뒤 한 번 뒤집힘
    t = np.arange(48, dtype=np.int64) * 3_600_000; hh = np.r_[np.full(24, 110.), np.full(24, 120.)]; ll = hh - 10; cc = hh - 5
    P, R1, S1 = pivot_levels(t, hh, ll, cc)
    assert np.isnan(P[0]) and abs(P[30] - 105) < 1e-9 and abs(R1[30] - 110) < 1e-9 and abs(S1[30] - 100) < 1e-9   # 전날 값
    K, D = stoch(up + 0.2, up - 0.2, up)
    assert K[30] > 80 and K[-1] < 20
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
