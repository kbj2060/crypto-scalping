#!/usr/bin/env python3
"""«CME 비트코인 갭은 메워진다» 의미 검증 (2026-11-15). 사전등록 docs/prereg/cme_gap_fill_meaning_20261115.md (9e32fe18, 계산 전).

  selftest   손계산 1분 경로로 시각(서머타임)·갭·메움·거울 단언
  run        BTC(주) · ETH(보조) 실제 + 기하 위약 20벌 → tmp/cme_gap_meaning_20261115/{events.parquet,result.json}
대리 갭 = 바이낸스 1분봉 금 16:00 / 일 17:00 America/Chicago 직전 종가(실제 CME 가격 아님).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/cme_gap_meaning_20261115"
CHI = ZoneInfo("America/Chicago")
DAY = 86_400_000
HS = (0.25, 0.5, 1, 2, 3, 5, 7, 14, 30)       # 곡선용 지평(일) · 주 판정 = 1·7
THR = 0.003


def weekends(t0: int, t1: int) -> list[tuple[int, int]]:
    """[t0, t1) 안의 (금 16:00, 일 17:00 시카고) UTC ms 쌍."""
    d = datetime.fromtimestamp(t0 / 1000, CHI).date()
    d += timedelta(days=(4 - d.weekday()) % 7)                # 첫 금요일
    out = []
    while True:
        fri = int(datetime(d.year, d.month, d.day, 16, tzinfo=CHI).timestamp() * 1000)
        s = d + timedelta(days=2)
        sun = int(datetime(s.year, s.month, s.day, 17, tzinfo=CHI).timestamp() * 1000)
        if sun >= t1:
            return out
        if fri >= t0:
            out.append((fri, sun))
        d += timedelta(days=7)


def gap_events(k: pd.DataFrame) -> pd.DataFrame:
    t, h, l, c = (k[x].to_numpy() for x in ("t", "h", "l", "c"))
    rows = []
    for fri, sun in weekends(int(t[0]), int(t[-1]) - 30 * DAY):
        i_f, i_s = np.searchsorted(t, fri - 60_000), np.searchsorted(t, sun - 60_000)
        if i_f >= len(t) or i_s >= len(t) or t[i_f] != fri - 60_000 or t[i_s] != sun - 60_000:
            continue                                             # 그 분이 빠졌으면 건너뜀
        pf, ps = c[i_f], c[i_s]
        g = ps / pf - 1
        if g == 0:
            continue
        mir = 2 * ps - pf
        row = dict(tk=sun, gap=g, pf=pf, ps=ps)
        a = i_s + 1                                              # 일 17:00 시작 분부터
        for H in HS:
            b = np.searchsorted(t, sun + H * DAY)
            lo, hi = l[a:b].min(), h[a:b].max()
            row[f"fill{H}"] = float(lo <= pf if g > 0 else hi >= pf)
            row[f"mir{H}"] = float(hi >= mir if g > 0 else lo <= mir)
        rows.append(row)
    return pd.DataFrame(rows)


def judge(R: pd.DataFrame, P: pd.DataFrame, H: float, rng) -> dict:
    from scipy.stats import norm
    r, f, m = R[f"fill{H}"].to_numpy(), R[f"fill{H}"].mean(), R[f"mir{H}"].mean()
    pk = P.groupby("draw")[f"fill{H}"].mean()
    ix = rng.integers(0, len(R), (2000, len(R)))
    bs1 = r[ix].mean(1)
    dd = (R[f"fill{H}"] - R[f"mir{H}"]).to_numpy(); bs2 = dd[ix].mean(1)
    se = math.sqrt(bs1.var(ddof=1) + pk.var(ddof=1) / len(pk)); d1 = f - pk.mean()
    half = R.tk >= SM.SPLIT
    hd = {h_: dict(minus_p1=float(R[half == h_][f"fill{H}"].mean() - P[(P.tk >= SM.SPLIT) == h_][f"fill{H}"].mean()),
                   minus_p2=float(dd[(half == h_).to_numpy()].mean()), n=int((half == h_).sum())) for h_ in (False, True)}
    return dict(n=len(R), real=f, p1=float(pk.mean()), p2=m, diff1=d1, lo90_1=d1 - 1.2816 * se, p_one=float(1 - norm.cdf(d1 / se)), p_rev=float(norm.cdf(d1 / se)),
                diff2=float(dd.mean()), lo90_2=float(np.quantile(bs2, .10)), hi90_2=float(np.quantile(bs2, .90)), halves=hd,
                ci95_real=[float(np.quantile(bs1, .025)), float(np.quantile(bs1, .975))])


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SM.SEED)
    res, curves, allev = {}, {}, []
    for ci, coin in ((1, "BTC"), (0, "ETH")):                   # 위약 시드는 SMC 규약(코인 순서 = SM.COINS 위치)
        k = SM.load_1m(f"{coin}USDT")
        R = gap_events(k).assign(coin=coin, draw=-1)
        P = pd.concat([gap_events(SM.placebo(k, SM.SEED + 1000 * ci + d)).assign(coin=coin, draw=d) for d in range(SM.NPLACEBO)], ignore_index=True)
        allev += [R, P]
        Rm, Pm = R[R.gap.abs() >= THR], P[P.gap.abs() >= THR]
        for H, cell in ((7, "G1"), (1, "G2")):
            res[f"{coin}|{cell}"] = judge(Rm, Pm, H, rng)
        res[f"{coin}|all_7d"] = judge(R, P, 7, rng)
        res[f"{coin}|thr_3d"] = judge(Rm, Pm, 3, rng); res[f"{coin}|thr_30d"] = judge(Rm, Pm, 30, rng)
        curves[coin] = {str(H): dict(real=float(Rm[f"fill{H}"].mean()), p1=float(Pm[f"fill{H}"].mean()), p2=float(Rm[f"mir{H}"].mean())) for H in HS}
        res[f"{coin}|share_ge_thr"] = float((R.gap.abs() >= THR).mean())
        print(coin, len(R), len(Rm), flush=True)
    # Holm 2칸(BTC)
    main = sorted(["BTC|G1", "BTC|G2"], key=lambda x: res[x]["p_one"]); alive = True
    for i, kk in enumerate(main):
        alive = alive and res[kk]["p_one"] <= .10 / (2 - i); res[kk]["holm"] = alive
    for kk in ("BTC|G1", "BTC|G2", "ETH|G1", "ETH|G2"):
        o = res[kk]; hv = o["halves"].values()
        cons = all(v["minus_p1"] > 0 and v["minus_p2"] > 0 for v in hv)
        o["pass"] = bool(o.get("holm", False) and o["lo90_1"] > 0 and o["lo90_2"] > 0 and cons)
        o["candidate"] = bool(o["diff1"] > 0 and o["diff2"] > 0 and cons)
        o["reversed"] = bool(o["p_rev"] <= .05 and o["diff2"] < 0 and all(v["minus_p1"] < 0 for v in hv))
    pd.concat(allev, ignore_index=True).to_parquet(OUT / "events.parquet")
    json.dump(dict(result=res, curves=curves), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for kk, o in res.items():
        if isinstance(o, dict):
            print(kk, {x: (round(o[x], 4) if isinstance(o.get(x), float) else o.get(x)) for x in ("n", "real", "p1", "p2", "diff1", "lo90_1", "diff2", "lo90_2", "holm", "pass", "candidate", "reversed")}, {h_: {a: round(b, 3) for a, b in v.items()} for h_, v in o["halves"].items()})
    print(json.dumps(curves, indent=1))


def selftest() -> None:
    # 시각: 2026-03-06(금, 서머타임 전 CST = UTC−6) 16:00 → 22:00 UTC · 2026-03-15(일, CDT = UTC−5) 17:00 → 22:00 UTC
    w = weekends(int(datetime(2026, 3, 1, tzinfo=ZoneInfo("UTC")).timestamp() * 1000), int(datetime(2026, 3, 20, tzinfo=ZoneInfo("UTC")).timestamp() * 1000))
    utc = lambda ms: datetime.fromtimestamp(ms / 1000, ZoneInfo("UTC")).strftime("%a %m-%d %H:%M")
    assert [utc(a) for a, _ in w] == ["Fri 03-06 22:00", "Fri 03-13 21:00"], [utc(a) for a, _ in w]
    assert [utc(b) for _, b in w] == ["Sun 03-08 22:00", "Sun 03-15 22:00"], [utc(b) for _, b in w]
    # 경로: 금 마감 100 → 일 개장 102(위 갭 2%) → 다음 날 101.5 → 3일째 99.9(메움) · 거울 104 는 안 닿음
    fri, sun = w[0]
    t0 = fri - 3 * DAY
    t = np.arange(t0, sun + 40 * DAY, 60_000)
    c = np.full(len(t), 100.0)
    c[t >= fri] = 102.0
    c[t >= sun + DAY] = 101.5
    c[t >= sun + 3 * DAY] = 99.9
    k = pd.DataFrame(dict(t=t, h=c + 0.01, l=c - 0.01, c=c))
    E = gap_events(k)
    e = E[E.tk == sun].iloc[0]
    assert abs(e.gap - 0.02) < 1e-12 and e.fill1 == 0 and e.fill3 == 0 and e.fill5 == 1 and e.fill7 == 1 and e.mir30 == 0, e.to_dict()
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
