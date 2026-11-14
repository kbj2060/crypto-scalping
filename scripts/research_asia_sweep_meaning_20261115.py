#!/usr/bin/env python3
"""아시아 세션 고저 스윕 뒤 반전(ICT «Judas swing») 의미 검증 (2026-11-15). 사전등록 docs/prereg/asia_session_sweep_meaning_20261115.md (b308c983).

  selftest   손계산 하루로 범위·첫 스윕·A1 경주·A2 반대 끝 단언
  run        4코인 실제 + 기하 위약 20벌 → tmp/asia_sweep_meaning_20261115/{events.parquet,placebo_agg.parquet,result.json}
경주·위약·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/asia_sweep_meaning_20261115"
DAY, Q = 86_400_000, 900_000
ASIA, HUNT = 32, 84                 # 00~08 UTC = 15분봉 32개 · 08~21 UTC = 84개


def events(B: dict, rng, with_p2: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    R = []
    add = lambda cell, i, y, **kw: R.append(dict(cell=cell, tk=int(t[i]) + Q, y=float(y), **kw))
    day = t // DAY; cut = np.flatnonzero(np.diff(day)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        if b - a != 96 or t[a] % DAY != 0:
            continue
        ah, al = h[a:a + ASIA].max(), l[a:a + ASIA].min()
        s0, s1 = a + ASIA, a + ASIA + HUNT                          # 사냥 창 [08:00, 21:00)
        a0 = atr[s0 - 1]
        if not a0 > 0:
            continue
        levels = [(ah, 1, "A1"), (al, -1, "A1")]
        if with_p2:
            levels += [(ah + rng.choice([-1, 1]) * a0, 1, "P2A1"), (al + rng.choice([-1, 1]) * a0, -1, "P2A1")]
        first = {}
        for lev, side, cell in levels:
            hit = h[s0:s1] > lev if side > 0 else l[s0:s1] < lev
            if not hit.any():
                continue
            s = s0 + int(np.argmax(hit))
            if c[s0 - 1] * side >= lev * side:                     # 08:00 에 이미 바깥이면 «스윕» 아님
                continue
            y = SM.first_hit_race(h, l, c, atr, s, -side, 100)      # 범위 안쪽 = 반대 방향
            if y >= 0:
                add(cell, s, y, side=side)
            if cell == "A1":
                first[side] = s
        if first:                                                   # A2: 먼저 깬 쪽 → 그날 21:00 전 반대 끝
            side = min(first, key=first.get); s = first[side]
            opp = (l[s + 1:s1] < al).any() if side > 0 else (h[s + 1:s1] > ah).any()
            if len(first) == 2 and first[-side] == s:              # 같은 봉에서 둘 다 깸 = 모름 → 주장에 불리(0)
                opp = False
            add("A2", s, opp, side=side)
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            rng = np.random.default_rng(SM.SEED + 7 * ci + 13 * (draw + 1))
            ev = pd.DataFrame(events(SM.bars(kk, 15), rng, draw < 0)).assign(coin=coin, tf="15m", draw=draw)
            if draw < 0:
                rows.append(ev)
            else:
                aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {"15m": 15}, ("15m",), ["A1", "A2"]
    SM.P2OF = {"A1": "P2A1"}
    res = {kk.split("|")[1]: v for kk, v in SM.judge(E, PA).items()}
    real = E[E.draw < 0]
    res["aux"] = {"A1_side": {int(s): round(g.y.mean(), 4) for s, g in real[real.cell == "A1"].groupby("side")},
                  "A2_n_days_with_sweep": int((real.cell == "A2").sum())}
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for c_ in ("A1", "A2"):
        print(c_, {x: res[c_].get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("aux", res["aux"])


def selftest() -> None:
    # 하루 96봉(+앞 하루 데움): 아시아 100~102 · 08:15 봉 고가 103(고점 스윕) → 다음 봉 저가가 종가−1 ATR 아래 → A1=1 · 이어 99 아래(반대 끝) → A2=1
    n = 192; t = np.arange(n) * Q; c = np.full(n, 101.0); h = c + 1; l = c - 1; atr = np.full(n, 1.0)
    d0 = 96
    s = d0 + ASIA + 1
    h[s] = 103.0; c[s] = 102.5
    l[s + 1] = 101.4; h[s + 1] = 102.9; c[s + 1] = 101.6
    l[s + 5] = 98.5; c[s + 5] = 99.0
    B = dict(t=t, h=h, l=l, c=c, atr=atr)
    R = events(B, np.random.default_rng(0), False)
    a1 = [r for r in R if r["cell"] == "A1" and r["tk"] == t[s] + Q]
    a2 = [r for r in R if r["cell"] == "A2"]
    assert a1 and a1[0]["y"] == 1.0 and a1[0]["side"] == 1, R
    assert len(a2) == 1 and a2[0]["tk"] == t[s] + Q and a2[0]["y"] == 1.0, a2   # 첫날은 평평해 스윕 없음 → 둘째 날 A2 하나
    # 08:00 에 이미 범위 밖이면 스윕 아님
    c2 = c.copy(); c2[d0 + ASIA - 1] = 102.6
    R2 = events(dict(t=t, h=h, l=l, c=c2, atr=atr), np.random.default_rng(0), False)
    assert not [r for r in R2 if r["cell"] == "A1" and r["side"] == 1 and r["tk"] == t[s] + Q]
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
