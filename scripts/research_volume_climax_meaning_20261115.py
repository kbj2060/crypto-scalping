#!/usr/bin/env python3
"""거래량 클라이맥스 뒤 반전 의미 검증 (2026-11-15). 사전등록 docs/prereg/volume_climax_meaning_20261115.md (08eb2aea).

  selftest   손계산 봉으로 큰 봉·거래량 폭증·꼬리·방향 단언
  run        4코인 × 15분·1시간 실제 + 기하 위약 20벌 + 거래량 없는 같은 모양 대조 → tmp/volume_climax_meaning_20261115/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/volume_climax_meaning_20261115"
END = int(pd.Timestamp("2026-10-30", tz="UTC").value // 10**6)      # 거래량 열이 있는 마지막 날 다음
TFS = {"15m": 15, "1h": 60}
MOM, VW, BIG, VX, VLOW = 12, 96, 2.0, 3.0, 1.5


def flags(B: dict) -> pd.DataFrame:
    """봉별: 직전 흐름 · 큰 봉 · 거래량 폭증/평범 · 꼬리(종가가 반전 쪽 절반)."""
    h, l, c, atr, v = B["h"], B["l"], B["c"], B["atr"], B["v"]
    n = len(c)
    d = np.zeros(n); d[MOM:] = np.sign(c[MOM:] - c[:-MOM])
    big = np.zeros(n, bool); big[1:] = (h[1:] - l[1:]) >= BIG * atr[:-1]
    med = pd.Series(v).shift(1).rolling(VW, min_periods=VW).median().to_numpy()
    spike, calm = v >= VX * med, v < VLOW * med
    mid = (h + l) / 2
    tail = np.where(d < 0, c > mid, c < mid)                          # 하락 뒤면 종가가 봉 중간 위(되밀림)
    return pd.DataFrame(dict(d=d, big=big & np.isfinite(med), spike=spike, calm=calm, tl=tail))   # «tail» 은 DataFrame 메서드 이름이라 tl


def events(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    F = flags(B); n = len(c); R = []
    rev_all = np.zeros(n); rev_all[MOM:] = -np.sign(c[MOM:] - c[:-MOM])
    for i in np.flatnonzero(F.big.to_numpy() & (F.d.to_numpy() != 0)):
        if i + 1 >= n or not atr[i] > 0:
            continue
        rv = int(-F.d[i]); kinds = []
        if F.spike[i]:
            kinds += ["C1"] + (["C2"] if F.tl[i] else [])
        elif with_p3 and F.calm[i]:
            kinds += ["P3C1"] + (["P3C2"] if F.tl[i] else [])
        if not kinds:
            continue
        y = SM.first_hit_race(h, l, c, atr, int(i), rv, 100)
        if y >= 0:
            for k in kinds:
                R.append(dict(cell=k, tk=int(t[i]) + B["ms"], y=float(y)))
    if with_p3:                                                       # 보조: 일반 되돌림(모든 봉 · 직전 12봉 반대)
        for i in range(max(MOM, VW), n - 1, 4):                       # ponytail: 4봉마다 표본(전수는 느림) -- 보조라 충분
            if rev_all[i] != 0 and atr[i] > 0:
                y = SM.first_hit_race(h, l, c, atr, i, int(rev_all[i]), 100)
                if y >= 0:
                    R.append(dict(cell="REVALL", tk=int(t[i]) + B["ms"], y=float(y)))
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
    SM.TFS, SM.MAIN_TFS, SM.CELLS = TFS, tuple(TFS), ["C1", "C2"]
    SM.P2OF = {"C1": "P3C1", "C2": "P3C2"}                            # judge 의 «P2» 자리 = 거래량 없는 같은 모양
    res = SM.judge(E, PA)
    aux = {f"{tf}|REVALL_rate": round(float(E[(E.tf == tf) & (E.cell == "REVALL")].y.mean()), 4) for tf in TFS}
    json.dump(dict(main=res, aux=aux), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k_, v in res.items():
        print(k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p2_n", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    print("aux", aux)


def selftest() -> None:
    n = 130; c = np.r_[np.linspace(110, 100, 120), np.full(10, 100.0)]; h = c + 0.2; l = c - 0.2; v = np.ones(n)
    i = 115; l[i] = c[i] - 3.0; h[i] = c[i] + 0.5; v[i] = 10                    # 하락 끝 큰 아래꼬리 + 거래량 10배
    B = dict(h=h, l=l, c=c, v=v, atr=np.full(n, 0.5), t=np.arange(n) * 900_000, ms=900_000)
    F = flags(B)
    assert F.d[i] == -1 and F.big[i] and F.spike[i] and F.tl[i]               # 종가가 봉 중간 위
    v2 = v.copy(); v2[i] = 1.2
    F2 = flags(dict(B, v=v2))
    assert F2.big[i] and not F2.spike[i] and F2.calm[i]                         # 같은 모양 · 거래량 평범 = P3
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
