#!/usr/bin/env python3
"""캔들 패턴·좁은 봉 돌파 묶음 5종 의미 검증 (2026-11-15). 사전등록 docs/prereg/candle_breakout_bundle_20261115.md (25b61a77).

  selftest   패턴별 손계산 사건·방향 단언
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 같은 모양 아무 봉 대조 → tmp/candle_breakout_bundle_20261115/
실행·판정 뼈대는 research_classic_indicator_bundle_20261115.run 그대로(events·KEYS·OUT 만 바꿈).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/candle_breakout_bundle_20261115"
KEYS = ["C1", "C2", "C3", "C4", "C5"]


def signals(B: dict) -> dict[str, np.ndarray]:
    """봉별 주장 방향(+1/−1/0). 키 = 지표 · P3 지표."""
    h, l, c, o = B["h"], B["l"], B["c"], B["o"]; n = len(c)
    lag = lambda x, k: np.r_[np.full(k, np.nan), x[:-k]]
    mom = np.r_[np.zeros(12), np.sign(c[12:] - c[:-12])]
    col = np.sign(c - o); body = np.abs(c - o); top, bot = np.maximum(o, c), np.minimum(o, c)
    up, lw = h - top, bot - l
    with np.errstate(invalid="ignore"):
        rev = (col == -lag(col, 1)) & (col != 0)
        bull = (mom < 0) & (col == 1) & rev & (o <= lag(bot, 1)) & (c >= lag(top, 1))
        bear = (mom > 0) & (col == -1) & rev & (o >= lag(top, 1)) & (c <= lag(bot, 1))
        C1 = np.where(bull, 1, np.where(bear, -1, 0))
        P3C1 = np.where(~bull & (mom < 0) & (col == 1) & rev, 1, np.where(~bear & (mom > 0) & (col == -1) & rev, -1, 0))
        ham = (mom < 0) & (body > 0) & (lw >= 2 * body) & (up <= body)
        star = (mom > 0) & (body > 0) & (up >= 2 * body) & (lw <= body)
        C2 = np.where(ham, 1, np.where(star, -1, 0))
        P3C2 = np.where(~ham & (mom < 0), 1, np.where(~star & (mom > 0), -1, 0))
        inside = (lag(h, 1) <= lag(h, 2)) & (lag(l, 1) >= lag(l, 2))
        C3 = np.where(inside & (c > lag(h, 2)), 1, np.where(inside & (c < lag(l, 2)), -1, 0))
        brk = np.where(c > lag(h, 1), 1, np.where(c < lag(l, 1), -1, 0))
        P3C3 = np.where(~inside, brk, 0)
        rng = h - l; nr7 = lag(rng, 1) <= lag(pd.Series(rng).rolling(7).min().to_numpy(), 1)
        C4, P3C4 = np.where(nr7, brk, 0), np.where(~nr7, brk, 0)
        dh, dl = lag(pd.Series(h).rolling(20).max().to_numpy(), 1), lag(pd.Series(l).rolling(20).min().to_numpy(), 1)
        cu, cd = c > dh, c < dl
        C5 = np.where(cu & ~lag(cu.astype(float), 1).astype(bool), 1, np.where(cd & ~lag(cd.astype(float), 1).astype(bool), -1, 0))
        P3C5 = np.r_[np.zeros(20), np.sign(c[20:] - c[:-20])]
    return dict(C1=C1, P3C1=P3C1, C2=C2, P3C2=P3C2, C3=C3, P3C3=P3C3, C4=C4, P3C4=P3C4, C5=C5, P3C5=P3C5)


def events(B: dict, tf: str, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; n = len(c); R = []
    sampled = {"P3C2", "P3C5"}                                 # «아무 봉» 대조는 표본 봉만(15분 4봉마다)
    keep = np.zeros(n, bool); keep[CB.WARM:n - 1:CB.STEP[tf]] = True
    for cell, x in signals(B).items():
        if cell.startswith("P3") and not with_p3:
            continue
        idx = np.flatnonzero((x != 0) & (keep if cell in sampled else True))
        for i in idx:
            if i < CB.WARM or i + 1 >= n or not atr[i] > 0:
                continue
            for suf, k in (("", 1.0), ("_2atr", 2.0)):
                y = SM.first_hit_race(h, l, c, atr if k == 1.0 else atr * k, int(i), int(x[i]), 100)
                if y >= 0:
                    R.append(dict(cell=cell + suf, tk=int(t[i]) + B["ms"], y=float(y)))
    return R


def selftest() -> None:
    c = np.r_[np.linspace(120, 100, 20), 103.0]; o = np.r_[c[0], c[:-1]]
    c[19] = 99.0; o[20] = 99.0; c[20] = 101.5                  # 봉 19 음봉(o 100.0x→99) · 봉 20 양봉 99→101 감쌈
    o[19] = c[18]
    h = np.maximum(o, c) + 0.1; l = np.minimum(o, c) - 0.1
    S = signals(dict(h=h, l=l, c=c, o=o))
    assert S["C1"][20] == 1 and S["P3C1"][20] == 0
    c2 = c.copy(); c2[20] = 99.5; o2 = o.copy(); h2 = np.maximum(o2, c2) + 0.1; l2 = np.minimum(o2, c2) - 3.0   # 긴 아랫꼬리 양봉
    S2 = signals(dict(h=h2, l=l2, c=c2, o=o2))
    assert S2["C2"][20] == 1 and S2["C1"][20] == 0 and S2["P3C1"][20] == 1
    hh = np.r_[np.full(21, 10.0), 9.5, 11]; ll = np.r_[np.full(21, 5.0), 6, 5]; cc = np.r_[np.full(21, 7.0), 7, 10.5]
    S3 = signals(dict(h=hh, l=ll, c=cc, o=np.r_[cc[0], cc[:-1]]))
    assert S3["C3"][22] == 1 and S3["P3C3"][22] == 0                   # 봉 21 이 봉 20 안 · 봉 22 종가 > 봉 20 고
    print("selftest ok")


def run() -> None:
    CB.events, CB.KEYS, CB.OUT = events, KEYS, OUT
    CB.run()


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
