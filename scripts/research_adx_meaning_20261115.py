#!/usr/bin/env python3
"""ADX 가 «추세 구간 / 횡보 구간»을 가르는가 (2026-11-15). 사전등록 docs/prereg/adx_meaning_20261115.md (432f10db).

  selftest   손계산 봉으로 ADX·DI 방향 단언(꾸준한 상승 = 높은 ADX·+방향, 지그재그 = 낮은 ADX)
  run        4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + ADX 조건 없는 같은 규칙 대조 → tmp/adx_meaning_20261115/
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

OUT = Path(__file__).resolve().parents[1] / "tmp/adx_meaning_20261115"
TFS = {"15m": 15, "1h": 60, "4h": 240}                     # 4시간 = 보조(Holm 밖)
STEP = {"15m": 4, "1h": 1, "4h": 1}
WARM, HI, LO, HI2 = 100, 25.0, 20.0, 40.0


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray):
    """와일더 ADX(14). ponytail: RMA 를 ewm(1/14) 로 — 첫 값 시작(와일더 원식은 첫 14봉 합) · 100봉 데움으로 차이 무시."""
    up = np.r_[0, h[1:] - h[:-1]]; dn = np.r_[0, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    c1 = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - c1), abs(l - c1)))
    rma = lambda x: pd.Series(x).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    atr = rma(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, mdi = 100 * rma(pdm) / atr, 100 * rma(mdm) / atr
        dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    a = rma(np.nan_to_num(dx)); a[:WARM] = np.nan
    return a, np.sign(pdi - mdi)


def events(B: dict, tf: str, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    A, d = adx(h, l, c); n = len(c); R = []
    def add(cell, i, dd, k=1.0):
        a2 = atr if k == 1.0 else atr * k
        y = SM.first_hit_race(h, l, c, a2, i, int(dd), 100)
        if y >= 0:
            R.append(dict(cell=cell, tk=int(t[i]) + B["ms"], y=float(y)))
    for i in range(WARM, n - 1, STEP[tf]):
        if not (np.isfinite(A[i]) and d[i] != 0 and atr[i] > 0):
            continue
        if A[i] >= HI:
            add("A1", i, d[i]); add("A1_2atr", i, d[i], 2.0)
            if A[i] >= HI2:
                add("A1_40", i, d[i])
            if A[i] > A[i - 1]:
                add("A1_up", i, d[i])
        elif A[i] <= LO:
            add("A2", i, -d[i]); add("A2_2atr", i, -d[i], 2.0)
        if with_p3:                                           # 조건 없는 같은 규칙(모든 표본 봉)
            add("P3A1", i, d[i]); add("P3A2", i, -d[i])
            add("P3A1_2atr", i, d[i], 2.0); add("P3A2_2atr", i, -d[i], 2.0)
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
    SM.TFS, SM.MAIN_TFS, SM.CELLS = TFS, ("15m", "1h"), ["A1", "A2"]          # 주 판정 4칸 Holm(4시간은 표만)
    SM.P2OF = {"A1": "P3A1", "A2": "P3A2"}                                     # judge 의 «P2» 자리 = ADX 조건 없는 같은 규칙
    main = SM.judge(E, PA)
    SM.MAIN_TFS, SM.CELLS = (), ["A1_40", "A1_up", "A1_2atr", "A2_2atr"]
    SM.P2OF = {"A1_40": "P3A1", "A1_up": "P3A1", "A1_2atr": "P3A1_2atr", "A2_2atr": "P3A2_2atr"}
    aux = SM.judge(E, PA)
    share = {tf: {"adx_ge25": round(float((E[(E.tf == tf) & (E.cell == "A1")].shape[0]) / max(1, E[(E.tf == tf) & (E.cell == "P3A1")].shape[0])), 3),
                  "adx_le20": round(float((E[(E.tf == tf) & (E.cell == "A2")].shape[0]) / max(1, E[(E.tf == tf) & (E.cell == "P3A1")].shape[0])), 3)} for tf in TFS}
    json.dump(dict(main=main, aux=aux, share=share), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in {**main, **aux}.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("share", share)


def selftest() -> None:
    n = 300
    c = 100 + np.arange(n) * 0.5; h, l = c + 0.3, c - 0.3                     # 꾸준한 상승
    A, d = adx(h, l, c)
    assert A[-1] > 50 and d[-1] == 1, (A[-1], d[-1])
    c2 = 100 + np.where(np.arange(n) % 2, 1.0, -1.0); h2, l2 = c2 + 0.3, c2 - 0.3   # 지그재그(방향 없음)
    A2, _ = adx(h2, l2, c2)
    assert A2[-1] < 20, A2[-1]
    assert np.isnan(A[WARM - 1]) and np.isfinite(A[WARM])                     # 데움
    c3 = c[::-1].copy(); A3, d3 = adx(c3 + 0.3, c3 - 0.3, c3)
    assert A3[-1] > 50 and d3[-1] == -1                                       # 꾸준한 하락 = −방향
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
