#!/usr/bin/env python3
"""묶음 4 — CCI ±100 · MFI 80/20 · OBV 선행 · 적삼병/흑삼병 (2026-11-16). 사전등록 docs/prereg/volume_osc_bundle_20261116.md (045eef7b).

  selftest   손계산 봉으로 지표값·사건 방향 단언
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 지표별 실제 가격 대조 → tmp/volume_osc_bundle_20261116/
실행·판정 뼈대는 research_classic_indicator_bundle_20261115.run 그대로(events·KEYS·OUT 만 바꿈). 거래량 끝(AV.END) 앞 사건만.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402
import research_anchored_vwap_meaning_20261115 as AV  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/volume_osc_bundle_20261116"
KEYS = ["E1", "E2", "E3", "E4"]
lag = lambda x, k=1: np.r_[np.full(k, np.nan), x[:-k]]
rmax = lambda x, w: pd.Series(x).rolling(w).max().to_numpy()
rsum = lambda x, w: pd.Series(x).rolling(w).sum().to_numpy()


def cci(h, l, c, w=20):
    tp = (h + l + c) / 3; out = np.full(len(c), np.nan)
    W = swv(tp, w); m = W.mean(1); mad = np.abs(W - m[:, None]).mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[w - 1:] = (tp[w - 1:] - m) / (0.015 * mad)
    return out


def mfi(h, l, c, v, w=14):
    tp = (h + l + c) / 3; d = np.sign(tp - lag(tp)); raw = tp * v
    pos, neg = rsum(np.where(d > 0, raw, 0.0), w), rsum(np.where(d < 0, raw, 0.0), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * pos / (pos + neg)


def obv_lead(c, v, w=20):
    """OBV 가 직전 w봉 최고 위로 막 올라섰는데 종가는 직전 w봉 최고 이하 → +1 · 대칭 −1."""
    o = np.cumsum(np.nan_to_num(np.sign(c - lag(c))) * v)
    with np.errstate(invalid="ignore"):
        up, dn = o > lag(rmax(o, w)), o < -lag(rmax(-o, w))
        up &= ~lag(up.astype(float)).astype(bool) & (c <= lag(rmax(c, w)))
        dn &= ~lag(dn.astype(float)).astype(bool) & (c >= -lag(rmax(-c, w)))
    return np.where(up, 1, np.where(dn, -1, 0))


def soldiers(c, o, atr):
    """(본 사건, P3): 직전 12봉 하락 뒤 연속 3 양봉(그 앞 봉은 양봉 아님) → +1 · 대칭. 본 = 세 몸통 다 ≥ 0.5 ATR."""
    col, body = np.sign(c - o), np.abs(c - o)
    with np.errstate(invalid="ignore"):
        big = body >= 0.5 * atr
        allbig = big & lag(big.astype(float)).astype(bool) & lag(big.astype(float), 2).astype(bool)
        mom = lag(c, 3) - lag(c, 15)
        E, P = np.zeros(len(c)), np.zeros(len(c))
        for s in (1, -1):
            run = (col == s) & (lag(col) == s) & (lag(col, 2) == s) & (lag(col, 3) != s) & (np.sign(mom) == -s)
            E[run & allbig] = s; P[run & ~allbig] = s
    return E, P


def events(B: dict, tf: str, with_p3: bool) -> dict:
    h, l, c, o, atr, t = B["h"], B["l"], B["c"], B["o"], B["atr"], B["t"]; n = len(c)
    v = np.nan_to_num(B["v"]); one = np.ones(n)
    R = dict(cell=[], tk=[], y=[])
    def add(cell, i, d):
        if i < CB.WARM or i + 1 >= n or d == 0 or not atr[i] > 0 or t[i] + B["ms"] > AV.END:
            return
        for suf, k in (("", 1.0), ("_2atr", 2.0)):
            y = SM.first_hit_race(h, l, c, atr if k == 1.0 else atr * k, int(i), int(d), 100)
            if y >= 0:
                R["cell"].append(cell + suf); R["tk"].append(int(t[i]) + B["ms"]); R["y"].append(float(y))
    def fire(cell, x):
        for i in np.flatnonzero(x):
            add(cell, i, x[i])
    C = cci(h, l, c)
    fire("E1", np.where(CB.cross(C, np.full(n, 100.0)) == 1, 1, np.where(CB.cross(C, np.full(n, -100.0)) == -1, -1, 0)))
    def mfi_ev(vv):
        m = mfi(h, l, c, vv)
        return np.where(CB.cross(m, np.full(n, 80.0)) == 1, -1, np.where(CB.cross(m, np.full(n, 20.0)) == -1, 1, 0))
    fire("E2", mfi_ev(v)); fire("E3", obv_lead(c, v))
    E4, P4 = soldiers(c, o, atr); fire("E4", E4)
    if with_p3:
        fire("P3E2", mfi_ev(one)); fire("P3E3", obv_lead(c, one)); fire("P3E4", P4)
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            add("P3E1", i, np.sign(c[i] - c[i - 20]))
    return R


def selftest() -> None:
    c = np.r_[100 + 0.2 * (np.arange(30) % 2), 103.0]; C = cci(c + 0.5, c - 0.5, c)
    assert np.isnan(C[18]) and abs(C[29]) < 100 and C[30] > 100                   # 잔물결 → ±100 안 · 튐 → +100 위
    up = np.arange(20, dtype=float)
    assert abs(mfi(up + .5, up - .5, up, np.ones(20))[-1] - 100) < 1e-9            # 오르기만 → 100
    cc = np.r_[np.linspace(110, 100, 25), 100.5, 99.8]; vv = np.r_[np.ones(25), 50.0, 1.0]
    x = obv_lead(cc, vv)
    assert x[25] == 1 and x[26] == 0                                              # 큰 거래량 양봉 → OBV 새 고점, 종가는 고점 아님
    cs = np.r_[np.linspace(120, 100, 20), 101, 102, 103]; os_ = np.r_[cs[0], cs[:-1]]
    E, P = soldiers(cs, os_, np.full(23, 1.0))
    assert E[22] == 1 and P[22] == 0 and (E != 0).sum() == 1
    E2, P2 = soldiers(cs, os_, np.full(23, 3.0))
    assert E2[22] == 0 and P2[22] == 1                                            # 몸통 1 < 1.5 → 대조
    print("selftest ok")


def run() -> None:
    CB.events, CB.KEYS, CB.OUT = events, KEYS, OUT
    CB.run()


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
