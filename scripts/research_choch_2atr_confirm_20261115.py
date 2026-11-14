#!/usr/bin/env python3
"""큰 스윙 CHoCH 뒤 «새 방향 2 ATR 먼저» 확인 검정 (2026-11-15). 사전등록 docs/prereg/choch_swing_2atr_confirm_20261115.md (d93b2dee).

  selftest   2 ATR 경주가 ATR 배수만 바꾼 first_hit_race 인지 단언
  run        4코인 × 5분·30분·1시간(주) + 15분·4시간(보조) · 실제 + 기하 위약 20벌
구조·경주·위약·판정은 research_smc_ict_meaning_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/choch_2atr_confirm_20261115"
TFS = {"5m": 5, "30m": 30, "1h": 60, "15m": 15, "4h": 240}
MAIN = ("5m", "30m", "1h")


def race(h, l, c, atr, t, d, k):
    a2 = atr.copy(); a2[t] = k * atr[t]                      # 기준 = t 의 ATR × k (first_hit_race 는 atr[s] 만 읽는다)
    return SM.first_hit_race(h, l, c, a2, t, d, 100)


def events(B: dict) -> list[dict]:
    h, l, c, atr, tt = B["h"], B["l"], B["c"], B["atr"], B["t"]
    R = []
    for N in (50, 5):
        ev = SM.structure(h, l, c, SM.pivots(h, N, True), SM.pivots(l, N, False), N)
        for t, d, typ in ev[:, :3]:
            t, d = int(t), int(d)
            if t + 1 >= len(c) or not atr[t] > 0:
                continue
            name = ("C" if typ == 2 else "B") + ("" if N == 50 else "5")
            for k, suf in ((2.0, ""), (1.0, "_1atr")):
                y = race(h, l, c, atr, t, d, k)
                if y >= 0:
                    R.append(dict(cell=name + suf, tk=int(tt[t]) + B["ms"], y=float(y)))
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            for tf, m in TFS.items():
                ev = pd.DataFrame(events(SM.bars(kk, m))).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                else:
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    allc = ["C", "B", "C_1atr", "B_1atr", "C5", "B5", "C5_1atr", "B5_1atr"]
    SM.TFS, SM.MAIN_TFS, SM.CELLS, SM.P2OF = TFS, MAIN, allc, {}
    res = SM.judge(E, PA)                                          # 통계는 전 칸 · judge 안의 Holm/pass 는 아래에서 주 3칸으로 다시
    keep = {k: v for k, v in res.items() if k.split("|")[1] in allc}
    main = sorted([f"{tf}|C" for tf in MAIN], key=lambda k_: keep[k_]["p_one"]); alive = True
    for i, k_ in enumerate(main):
        o = keep[k_]; alive = alive and o["p_one"] <= .10 / (3 - i); o["holm"] = alive
        cons = sum(v > 0 for v in o["coin_diff_pp"].values()) >= 3 and all(v > 0 for v in o["half_diff_pp"].values())
        o["pass"] = bool(alive and o["lo90_pp"] > 0 and cons); o["candidate"] = bool(o["diff_pp"] > 0 and cons)
    for k_ in keep:
        if k_ not in main:
            for f in ("holm", "pass", "candidate", "reversed"):
                keep[k_].pop(f, None)                                  # 보조 칸엔 판정 없음
    n_pass = sum(bool(keep.get(f"{tf}|C", {}).get("pass")) for tf in MAIN)
    out = dict(result=keep, confirmed=n_pass >= 2, n_pass=n_pass)
    json.dump(out, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for kk_ in sorted(keep):
        v = keep[kk_]
        print(kk_, {x: v.get(x) for x in ("n", "real", "p1", "diff_pp", "lo90_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate")})
    print("confirmed", out["confirmed"], n_pass)


def selftest() -> None:
    # 기준 100, ATR 1 · 2 ATR 경주: 위로 102 를 먼저 찍으면 1, 98 을 먼저면 0
    c = np.full(10, 100.0); h = c + 0.5; l = c - 0.5; atr = np.ones(10)
    h[3] = 101.5; l[4] = 98.9; h[6] = 102.1
    assert race(h, l, c, atr, 0, 1, 2.0) == 1                     # 1 ATR 판이면 101.5 에 먼저 → 1, 2 ATR 판은 102.1 이 98 보다 먼저 → 1
    l2 = l.copy(); l2[4] = 97.9
    assert race(h, l2, c, atr, 0, 1, 2.0) == 0                    # 97.9 ≤ 98 이 102.1 보다 먼저 → 0
    assert race(h, l2, c, atr, 0, 1, 1.0) == 1                    # 1 ATR 판: 101.5(봉 3) 가 99(봉 4) 보다 먼저 → 1
    assert atr[0] == 1.0                                          # 원본 ATR 안 건드림
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
