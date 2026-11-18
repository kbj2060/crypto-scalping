#!/usr/bin/env python3
"""앵커드 VWAP 지지·저항 의미 검증 (2026-11-15). 사전등록 docs/prereg/anchored_vwap_meaning_20261115.md (7cf93e52).

  selftest   손계산으로 AVWAP·앵커 확정 시각·다가옴 사건 단언
  run        4코인 × 15분·1시간 실제 + 기하 위약 20벌 + ±1 ATR 옮긴 수준 대조 → tmp/anchored_vwap_meaning_20261115/
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

OUT = Path(__file__).resolve().parents[1] / "tmp/anchored_vwap_meaning_20261115"
END = int(pd.Timestamp("2026-10-30", tz="UTC").value // 10**6)      # 거래량 열이 있는 마지막 날 다음
TFS = {"15m": 15, "1h": 60}
FAR, NEAR = 0.5, 0.25


def avwap(B: dict, N: int, high: bool):
    """봉별 (지금 AVWAP, 같은 앵커로 잰 직전 봉 AVWAP, 앵커 확정 시각 ATR, 앵커 뒤 첫 사건 여부용 앵커 id). 앵커 없으면 NaN."""
    h, l, c, atr = B["h"], B["l"], B["c"], B["atr"]
    v = np.nan_to_num(B["v"]); n = len(c)
    tp = (h + l + c) / 3
    cpv, cv = np.r_[0, np.cumsum(tp * v)], np.r_[0, np.cumsum(v)]
    ps = np.flatnonzero(SM.pivots(h if high else l, N, high)); known = ps + N
    j = np.searchsorted(known, np.arange(n), "right") - 1                # 봉 t 에 알려진 가장 최근 앵커
    p = np.where(j >= 0, ps[np.clip(j, 0, None)], -1); t = np.arange(n)
    ok = (j >= 0) & (t >= 1)
    pp = np.where(ok, p, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(ok, (cpv[t + 1] - cpv[pp]) / (cv[t + 1] - cv[pp]), np.nan)
        Ap = np.where(ok & (t - 1 >= pp), (cpv[t] - cpv[pp]) / (cv[t] - cv[pp]), np.nan)
    off = np.where(ok, atr[np.clip(known[np.clip(j, 0, None)], 0, n - 1)], np.nan)
    return A, Ap, off, np.where(ok, j, -1)


def approaches(c: np.ndarray, atr: np.ndarray, A: np.ndarray, Ap: np.ndarray) -> list[tuple[int, int]]:
    """(t, s): 봉 t−1 종가가 수준에서 s 쪽 ≥0.5 ATR · 봉 t 종가가 ±0.25 ATR 안."""
    dp = np.r_[np.nan, c[:-1]] - Ap
    with np.errstate(invalid="ignore"):
        m = (np.abs(dp) >= FAR * atr) & (np.abs(c - A) <= NEAR * atr) & (atr > 0)
    return [(int(t), int(np.sign(dp[t]))) for t in np.flatnonzero(m)]


def events(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; n = len(c); R = []
    def add(cell, i, s, k=1.0):
        if i + 1 >= n:
            return
        y = SM.first_hit_race(h, l, c, atr if k == 1.0 else atr * k, i, s, 100)
        if y >= 0:
            R.append(dict(cell=cell, tk=int(t[i]) + B["ms"], y=float(y), s=s))
    for high, X in ((True, "H"), (False, "L")):
        for N, suf in ((20, ""), (50, "50")):
            A, Ap, off, aid = avwap(B, N, high)
            seen = set()
            for i, s in approaches(c, atr, A, Ap):
                add(X + suf, i, s)
                if suf == "":
                    add(X + "_2atr", i, s, 2.0)
                    if aid[i] not in seen:
                        seen.add(aid[i]); add(X + "first", i, s)
            if with_p3 and suf == "":
                for sg in (1, -1):
                    for i, s in approaches(c, atr, A + sg * off, Ap + sg * off):
                        add("P3" + X, i, s)
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT"); k = k[k.t < END].reset_index(drop=True)
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            for tf, mnt in TFS.items():
                ev = pd.DataFrame(events(SM.bars(kk, mnt), draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                elif len(ev):
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.TFS, SM.MAIN_TFS, SM.CELLS = TFS, tuple(TFS), ["H", "L"]               # 주 판정 4칸 Holm
    SM.P2OF = {"H": "P3H", "L": "P3L"}                                        # judge 의 «P2» 자리 = 옮긴 수준
    main = SM.judge(E, PA)
    SM.MAIN_TFS, SM.CELLS = (), ["H50", "L50", "H_2atr", "L_2atr", "Hfirst", "Lfirst"]
    SM.P2OF = {"H50": "P3H", "L50": "P3L", "Hfirst": "P3H", "Lfirst": "P3L"}
    aux = SM.judge(E, PA)
    side = {f"{tf}|{x}|s{s}": round(float(g.y.mean()), 4) for (tf, x, s), g in E[E.cell.isin(["H", "L"])].groupby(["tf", "cell", "s"])}
    json.dump(dict(main=main, aux=aux, side=side), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in {**main, **aux}.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("side", side)


def selftest() -> None:
    n = 80; ix = np.arange(n)
    l = 105.0 + 0.01 * abs(ix - 25); l[25] = 100.0; h = l + 1; c = l + 0.5; v = np.ones(n); v[27] = 3.0   # 봉 25 = 유일한 스윙 저점(20봉)
    B = dict(h=h, l=l, c=c, v=v, atr=np.full(n, 1.0))
    A, Ap, off, aid = avwap(B, 20, False)
    assert np.isnan(A[44]) and np.isfinite(A[45])                             # 확정 = 25+20
    tp = (h + l + c) / 3
    assert abs(A[45] - (tp[25:46] * v[25:46]).sum() / v[25:46].sum()) < 1e-12
    assert abs(Ap[45] - (tp[25:45] * v[25:45]).sum() / v[25:45].sum()) < 1e-12   # 같은 앵커 · 직전 봉
    lv = np.full(8, 100.0); cc = np.array([101, 100.8, 100.1, 100.0, 99.0, 99.9, 100.2, 101.0])
    assert approaches(cc, np.ones(8), lv, lv) == [(2, 1), (5, -1)]            # 위에서 다가옴(되튕김 +) · 아래에서(−)
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
