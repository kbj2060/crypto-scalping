#!/usr/bin/env python3
"""묶음 7·8 — 남은 지표 13종 (2026-11-16). 사전등록 docs/prereg/remaining_indicators_bundle78_20261116.md (deeca973).

  selftest   손계산 봉으로 지표값·사건 단언
  run7       오실레이터 7종(I1~I7) → tmp/remaining_bundle7_20261116/
  run8       추세·수준 6종(J1~J6)  → tmp/remaining_bundle8_20261116/
실행·판정(4코인 + 새 코인 · 최종 = 두 그룹 둘 다)은 research_trend_pullback_bundle_20261116.run 그대로(events·KEYS·OUT 만 바꿈).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402
import research_anchored_vwap_meaning_20261115 as AV  # noqa: E402
import research_trend_pullback_bundle_20261116 as TP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rsum = lambda x, w: pd.Series(x).rolling(w).sum().to_numpy()
rmax = lambda x, w: pd.Series(x).rolling(w).max().to_numpy()
rmin = lambda x, w: pd.Series(x).rolling(w).min().to_numpy()
lag = lambda x, k=1: np.r_[np.full(k, np.nan), x[:-k]]
ema = TP.ema
X = lambda a, lv: CB.cross(a, np.full(len(a), float(lv)))           # 문턱 교차 +1/−1/0


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float); out = np.full(len(x), np.nan)
    out[n - 1:] = np.convolve(x, w[::-1], "valid") / w.sum()
    return out


def osc(h, l, c):
    """(이름 → 봉별 주장 방향) 묶음 7."""
    cp = lag(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        HH, LL = rmax(h, 14), rmin(l, 14); wr = -100 * (HH - c) / (HH - LL)
        lo, hi = np.fmin(l, cp), np.fmax(h, cp); bp, tr = c - lo, hi - lo          # 첫 봉 cp NaN → fmin/fmax 가 l·h 로
        a = lambda n: rsum(bp, n) / rsum(tr, n); uo = 100 * (4 * a(7) + 2 * a(14) + a(28)) / 7
        d = np.r_[0, np.diff(c)]; su, sd = rsum(np.maximum(d, 0), 14), rsum(np.maximum(-d, 0), 14); cmo = 100 * (su - sd) / (su + sd)
        e3 = ema(ema(ema(c, 15), 15), 15); trix = 100 * (e3 / lag(e3) - 1)
        mp = (h + l) / 2; ao = CB.sma(mp, 5) - CB.sma(mp, 34)
        trr = np.fmax(h, cp) - np.fmin(l, cp)
        vip, vim = rsum(np.abs(h - lag(l)), 14) / rsum(trr, 14), rsum(np.abs(l - lag(h)), 14) / rsum(trr, 14)
        roc = lambda n: 100 * (c / lag(c, n) - 1)
        kst = sum(k * CB.sma(roc(n), m) for k, (n, m) in enumerate(((10, 10), (15, 10), (20, 10), (30, 15)), 1)); sig = CB.sma(kst, 9)
    rev = lambda x, lo_, hi_: np.where(X(x, lo_) == 1, 1, np.where(X(x, hi_) == -1, -1, 0))
    return dict(I1=rev(wr, -80, -20), I2=rev(uo, 30, 70), I3=rev(cmo, -50, 50), I4=X(trix, 0), I5=X(ao, 0),
                I6=CB.cross(vip, vim), I7=CB.cross(kst, sig))


@njit(cache=True)
def chandelier(c, hh, ll, atr):
    """상태 +1/−1 · 뒤집힌 봉에 새 상태, 아니면 0."""
    n = len(c); out = np.zeros(n); st = 0
    for i in range(n):
        if not (hh[i] == hh[i] and atr[i] == atr[i]):
            continue
        L, S = hh[i] - 3 * atr[i], ll[i] + 3 * atr[i]
        if st == 0:
            st = 1 if c[i] > S else -1
        elif st == 1 and c[i] < L:
            st = -1; out[i] = -1
        elif st == -1 and c[i] > S:
            st = 1; out[i] = 1
    return out


def fractal_break(h, l, c):
    """(프랙탈 돌파, P3 직전 5봉 최고/최저 종가 돌파) — 프랙탈 p 는 p+2 에 알려짐."""
    n = len(c); out = np.zeros(n); P = np.zeros(n)
    for x, sg in ((h, 1), (l, -1)):
        s = sg * x
        f = np.zeros(n, bool); f[2:-2] = (s[2:-2] > s[1:-3]) & (s[2:-2] > s[:-4]) & (s[2:-2] > s[3:-1]) & (s[2:-2] > s[4:])
        ps = np.flatnonzero(f); j = np.searchsorted(ps + 2, np.arange(n), "right") - 1
        lev = np.where(j >= 0, x[ps[np.clip(j, 0, None)]] if len(ps) else np.nan, np.nan)
        with np.errstate(invalid="ignore"):
            hit = (sg * c > sg * lev) & ~(sg * lag(c) > sg * lev)
            out[hit] = sg
            m5 = lag(rmax(s, 5)); bp = (sg * c > m5) & ~(lag(sg * c) > lag(m5))
            P[bp] = sg
    return out, P


def hull_turn(c, n=55):
    hm = wma(2 * wma(c, n // 2) - wma(c, n), int(np.sqrt(n)))
    s = np.sign(np.r_[np.nan, np.diff(hm)])
    with np.errstate(invalid="ignore"):
        return np.where((s != 0) & (lag(s) == -s), s, 0)


def cam(t, h, l, c):
    """봉별 전날 UTC 종가 C · 범위 R."""
    day = t // 86_400_000
    g = pd.DataFrame(dict(day=day, h=h, l=l, c=c)).groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    prev = pd.DataFrame(dict(C=g.c, R=g.h - g.l)).shift(1).reindex(day)
    return prev.C.to_numpy(), prev.R.to_numpy()


def _base(B):
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
    return h, l, c, atr, t, n, R, add, fire


def events7(B: dict, tf: str, with_p3: bool) -> dict:
    h, l, c, atr, t, n, R, add, fire = _base(B)
    for k, x in osc(h, l, c).items():
        fire(k, x)
    if with_p3:
        P3 = dict(I1=(14, -1), I2=(28, -1), I3=(14, -1), I4=(30, 1), I5=(34, 1), I6=(14, 1), I7=(20, 1))
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            for k, (m, s) in P3.items():
                add("P3" + k, i, s * np.sign(c[i] - c[i - m]))
    return R


def events8(B: dict, tf: str, with_p3: bool) -> dict:
    h, l, c, atr, t, n, R, add, fire = _base(B)
    ten, kij = (rmax(h, 9) + rmin(l, 9)) / 2, (rmax(h, 26) + rmin(l, 26)) / 2
    fire("J1", CB.cross(ten, kij)); fire("J2", hull_turn(c)); fire("J3", chandelier(c, rmax(h, 22), rmin(l, 22), atr))
    F, PF = fractal_break(h, l, c); fire("J4", F)
    C, Rg = cam(t, h, l, c); w = 1.1 * Rg / 8
    for L in (C + 1.1 * Rg / 4, C - 1.1 * Rg / 4):                       # J5 H3·L3 되밀림
        for i, s in AV.approaches(c, atr, L, lag(L)):
            add("J5", i, s)
        if with_p3:
            for Ls in (L + w, L - w):
                for i, s in AV.approaches(c, atr, Ls, lag(Ls)):
                    add("P3J5", i, s)
    def brk(Lu, Ld):
        with np.errstate(invalid="ignore"):
            return np.where((c > Lu) & ~(lag(c) > lag(Lu)), 1, np.where((c < Ld) & ~(lag(c) < lag(Ld)), -1, 0))
    H4, L4 = C + 1.1 * Rg / 2, C - 1.1 * Rg / 2
    fire("J6", brk(H4, L4))
    if with_p3:
        fire("P3J6", brk(H4 + w, L4 - w)); fire("P3J6", brk(H4 - w, L4 + w)); fire("P3J4", PF)
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            add("P3J1", i, np.sign(c[i] - c[i - 26])); add("P3J2", i, np.sign(c[i] - c[i - 10])); add("P3J3", i, np.sign(c[i] - c[i - 22]))
    return R


def selftest() -> None:
    up = np.r_[np.linspace(100, 80, 40), np.linspace(80, 95, 20)]; h, l = up + 0.5, up - 0.5
    S = osc(h, l, up)
    assert S["I1"][40:44].max() == 1 and (S["I1"][:39] != 1).all()                 # 바닥 뒤 %R −80 위로 = 위
    assert S["I4"][40:].max() == 1 and S["I5"][40:].max() == 1 and S["I6"][40:].max() == 1
    assert abs(wma(np.arange(5.), 3)[4] - (2 * 1 + 3 * 2 + 4 * 3) / 6) < 1e-12
    st = chandelier(np.r_[np.full(5, 100.), 90.], np.full(6, 101.), np.full(6, 95.), np.full(6, 1.))   # 시작: 100 > 95+3 → 롱
    assert st[5] == -1 and (st[:5] == 0).all()                                    # 101−3 = 98 아래로 닫음 → 숏
    hh = np.array([1, 2, 5, 2, 1, 1, 1, 6.]); cc = np.array([1, 1, 1, 1, 1, 4, 5.5, 5.8]); F, _ = fractal_break(hh, hh - 1, cc)
    assert F[6] == 1 and F[7] == 0 and F[5] == 0                                 # 프랙탈 5(봉 2, 봉 4 에 알려짐) 를 봉 6 종가 5.5 가 처음 넘음
    t = np.arange(48, dtype=np.int64) * 3_600_000; C, Rg = cam(t, np.r_[np.full(24, 110.), np.full(24, 120.)], np.r_[np.full(24, 100.), np.full(24, 110.)], np.full(48, 105.))
    assert np.isnan(C[0]) and C[30] == 105 and Rg[30] == 10
    print("selftest ok")


def run7() -> None:
    TP.events, TP.KEYS, TP.OUT = events7, [f"I{i}" for i in range(1, 8)], ROOT / "tmp/remaining_bundle7_20261116"
    TP.run()


def run8() -> None:
    TP.events, TP.KEYS, TP.OUT = events8, [f"J{i}" for i in range(1, 7)], ROOT / "tmp/remaining_bundle8_20261116"
    TP.run()


if __name__ == "__main__":
    {"selftest": selftest, "run7": run7, "run8": run8}[sys.argv[1]]()
