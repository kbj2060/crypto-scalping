#!/usr/bin/env python3
"""묶음 3 — 쌍봉·쌍바닥 · 주간 시가 편향 · 일간 시가 편향 (2026-11-15). 사전등록 docs/prereg/pattern_open_bundle_20261115.md (d4b6273d).

  selftest   손계산 봉으로 기간 시가 방향·쌍봉 목선 돌파 단언
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 지표별 실제 가격 대조 → tmp/pattern_open_bundle_20261115/
  run2       D1 만 안 본 코인 5종(5분봉 CSV) 재현 — 사전등록 docs/prereg/double_top_r2_newcoins_20261116.md (33edaca0) → tmp/double_top_r2_newcoins_20261116/
실행·판정 뼈대는 research_classic_indicator_bundle_20261115.run 그대로(events·KEYS·OUT 만 바꿈).
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/pattern_open_bundle_20261115"
KEYS = ["D1", "D2", "D3"]
H, DAY = 3_600_000, 86_400_000
MON = 4 * DAY                                               # 1970-01-05 = 월요일(1970-01-01 목요일)
NP, GAP, EQ, LOOK = 10, 50, 0.25, 50
ONLY_D1 = False                                             # run2: D2·D3 건너뜀


def open_dir(t: np.ndarray, c: np.ndarray, P: int, off: int) -> np.ndarray:
    """봉별 sign(c − 기간 시가). 기간 = (t − off) // P · 시가 = 기간 첫 봉 직전 봉 종가 · 첫 봉 자신은 0."""
    pid = (t - off) // P
    first = np.r_[True, pid[1:] != pid[:-1]]
    fi = np.flatnonzero(first)[np.cumsum(first) - 1]
    op = np.where(fi > 0, c[np.clip(fi - 1, 0, None)], np.nan)
    d = np.sign(c - op); d[first] = 0
    return np.nan_to_num(d)


def doubles(h, l, c, atr) -> list[tuple[int, int, str]]:
    """(사건 봉, 방향, 'D1'|'P3D1'). 고점 둘 → 목선 아래 첫 종가 = −1 · 저점 둘 대칭."""
    n = len(c); out = []
    for high in (True, False):
        x = h if high else l
        ps = np.flatnonzero(SM.pivots(x, NP, high))
        for a, b in zip(ps[:-1], ps[1:]):
            if b - a > GAP or not atr[b] > 0:
                continue
            diff = abs(x[b] - x[a]) / atr[b]
            kind = "D1" if diff <= EQ else ("P3D1" if 0.5 <= diff <= 2.0 else None)
            if kind is None:
                continue
            neck = l[a:b + 1].min() if high else h[a:b + 1].max()
            for j in range(b + NP, min(b + NP + LOOK, n - 1)):
                hit = (c[j] < neck <= c[j - 1]) if high else (c[j] > neck >= c[j - 1])
                if hit:
                    out.append((j, -1 if high else 1, kind)); break
    return out


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
    for j, d, kind in doubles(h, l, c, atr):
        if kind == "D1" or with_p3:
            add(kind, j, d)
    if ONLY_D1:
        return R
    keep = np.arange(CB.WARM, n - 1, CB.STEP[tf])
    tt = t + B["ms"]                                        # 봉 마감 시각으로 기간을 가름(마감 봉의 종가 vs 시가)
    sets = {"D2": [(7 * DAY, MON)], "D3": [(DAY, 0)]}
    if with_p3:
        sets["P3D2"] = [(7 * DAY, MON + k * DAY) for k in range(1, 7)]
        sets["P3D3"] = [(DAY, hh * H) for hh in range(1, 24)]
    for cell, anchors in sets.items():
        for P, off in anchors:
            d = open_dir(tt - 1, c, P, off)               # −1: 마감 시각 정각은 끝난 기간에 넣음
            for i in keep:
                add(cell, i, d[i])
    return R


def selftest() -> None:
    t = np.arange(10, dtype=np.int64) * H; c = np.array([10, 11, 12, 9, 8, 10, 13, 7, 7, 15.])
    d = open_dir(t, c, 3 * H, 0)                            # 기간 [0,1,2] [3,4,5] [6,7,8] [9]
    assert list(d) == [0, 0, 0, 0, -1, -1, 0, -1, -1, 0], d     # 시가 = 직전 기간 끝 종가 12 · 10 · 7 → 봉 9 는 새 기간 첫 봉
    x = np.interp(np.arange(120), [0, 30, 40, 50, 70, 119], [100, 110, 104, 110.1, 100, 90])   # 꺾이는 점에만 스윙
    h = x + 0.1; l = x - 0.1; c = x.copy(); atr = np.full(120, 1.0)
    j0 = int(np.flatnonzero((np.arange(120) >= 60) & (c < 103.9))[0])                       # p2+10 뒤 목선(103.9) 아래 첫 종가
    ev = doubles(h, l, c, atr)
    assert ev == [(j0, -1, "D1")], ev
    h2 = h.copy(); h2[50] = 111.2; ev2 = doubles(h2, l, c, atr)
    assert ev2 == [(j0, -1, "P3D1")], ev2                    # 차이 1.1 ATR = «쌍» 아님
    print("selftest ok")


def run() -> None:
    CB.events, CB.KEYS, CB.OUT = events, KEYS, OUT
    CB.run()


def run2() -> None:
    global ONLY_D1
    import research_orb_meaning_20261115 as OR
    ONLY_D1, out = True, OUT.parent / "double_top_r2_newcoins_20261116"; out.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(OR.NEW):
        k = OR.load5(coin)
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 8000 + 1000 * ci + draw)
            k1 = kk.assign(t=kk.t // 5)                       # ponytail: 5분 행을 «1분 행»으로 눈금만 줄여 SM.bars 재사용 → t·ms 다시 ×5
            for tf, mnt in CB.TFS.items():
                B = SM.bars(k1, mnt // 5); B["t"] = B["t"] * 5; B["ms"] *= 5
                ev = pd.DataFrame(events(B, tf, draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                elif len(ev):
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(out / "events.parquet"); PA.to_parquet(out / "placebo_agg.parquet")
    SM.TFS, SM.COINS = CB.TFS, list(OR.NEW)
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = ("15m", "1h"), ["D1"], {"D1": "P3D1"}
    main = SM.judge(E, PA)
    for v in main.values():                                  # 코인 ≥ 4/5 — judge 는 3 고정이라 다시 셈
        ok = sum(x > 0 for x in v["coin_diff_pp"].values()) >= 4 and all(x > 0 for x in v["half_diff_pp"].values())
        v["pass"] = bool(v["holm"] and v["lo90_pp"] > 0 and v.get("lo90_2_pp", -1) > 0 and ok)
        v["candidate"] = bool(v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and ok)
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), ["D1_2atr"], {"D1_2atr": "P3D1_2atr"}
    aux = SM.judge(E, PA)
    json.dump(dict(main=main, aux=aux), open(out / "result.json", "w"), ensure_ascii=False, indent=1, default=float)


def selftest2() -> None:
    import research_orb_meaning_20261115 as OR
    k = OR.load5("ADA").iloc[:600]; k1 = k.assign(t=k.t // 5)
    B = SM.bars(k1, 3); B["t"] = B["t"] * 5; B["ms"] *= 5
    g = k[k.t // 900_000 == B["t"][0] // 900_000]
    assert B["t"][0] % 900_000 == 0 and B["ms"] == 900_000 and len(g) == 3
    assert B["h"][0] == g.h.max() and B["l"][0] == g.l.min() and B["c"][0] == g.c.iloc[-1]
    print("selftest2 ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "run2": run2, "selftest2": selftest2}[sys.argv[1]]()
