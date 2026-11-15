#!/usr/bin/env python3
"""묶음 6 — 추세 속 눌림 3종: EMA20 첫 터치 · 스토캐스틱 추세 눌림 · 홀리 그레일 (2026-11-16). 사전등록 docs/prereg/trend_pullback_bundle_20261116.md (969d5255).

  selftest   손계산 봉으로 터치·눌림 사건 단언
  run        orig 4코인(1분 Vision) + new 5코인(5분 CSV) × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 + 지표별 실제 가격 대조
             → tmp/trend_pullback_bundle_20261116/ · 최종 통과 = 같은 칸 두 그룹 둘 다
경주·위약·판정은 research_smc_ict_meaning_20261115, 봉 묶음·표본은 research_classic_indicator_bundle_20261115 그대로.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402
import research_orb_meaning_20261115 as OR  # noqa: E402
from research_adx_meaning_20261115 import adx  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/trend_pullback_bundle_20261116"
KEYS = ["H1", "H2", "H3"]
ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def touch(h, l, c, e):
    """EMA 첫 터치: 위에서(직전 저가 > e, 이번 저가 ≤ e < 종가) +1 · 아래에서 대칭 −1."""
    lp, hp, ep = np.r_[np.nan, l[:-1]], np.r_[np.nan, h[:-1]], np.r_[np.nan, e[:-1]]
    with np.errstate(invalid="ignore"):
        return np.where((lp > ep) & (l <= e) & (c > e), 1, np.where((hp < ep) & (h >= e) & (c < e), -1, 0))


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
    def fire(cell, x):
        for i in np.flatnonzero(x):
            add(cell, i, x[i])
    e20 = ema(c, 20); tr = np.sign(e20 - ema(c, 50)); tc = touch(h, l, c, e20)
    fire("H1", np.where(tc == tr, tc, 0))
    K, _ = CB.stoch(h, l, c); s2 = np.sign(c - CB.sma(c, 200))
    fire("H2", np.where((s2 == 1) & (CB.cross(K, np.full(n, 20.0)) == 1), 1, np.where((s2 == -1) & (CB.cross(K, np.full(n, 80.0)) == -1), -1, 0)))
    a, di = adx(h, l, c)
    with np.errstate(invalid="ignore"):
        hg = np.where(tc == di, tc, 0)
        fire("H3", np.where(a > 30, hg, 0))
        if with_p3:
            fire("P3H3", np.where(a <= 30, hg, 0))
    if with_p3:
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            add("P3H1", i, tr[i])
            if 30 <= K[i] <= 70:
                add("P3H2", i, s2[i])
    return R


def judge_group(E, PA, coins, need):
    SM.TFS, SM.COINS = CB.TFS, list(coins); main, aux = {}, {}
    for key in KEYS:                                          # 지표별 Holm 2칸(15분·1시간)
        SM.MAIN_TFS, SM.CELLS, SM.P2OF = ("15m", "1h"), [key], {key: "P3" + key}
        main.update(SM.judge(E, PA))
        SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), [key + "_2atr"], {key + "_2atr": "P3" + key + "_2atr"}
        aux.update(SM.judge(E, PA))
    for v in main.values():                                   # 코인 문턱(4/5)을 다시 셈 · Holm 밖(4시간) = 보조
        if "holm" in v:
            ok = sum(x > 0 for x in v["coin_diff_pp"].values()) >= need and all(x > 0 for x in v["half_diff_pp"].values())
            v["pass"] = bool(v["holm"] and v["lo90_pp"] > 0 and v.get("lo90_2_pp", -1) > 0 and ok)
            v["candidate"] = bool(v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and ok)
    return dict(main=main, aux=aux)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    for grp, coins, seed0 in (("orig", list(SM.COINS), SM.SEED), ("new", OR.NEW, SM.SEED + 12000)):
        rows, aggs = [], []
        for ci, coin in enumerate(coins):
            k = OR.load5(coin) if grp == "new" else SM.load_1m(f"{coin}USDT")
            for draw in range(-1, SM.NPLACEBO):
                kk = k if draw < 0 else SM.placebo(k, seed0 + 1000 * ci + draw)
                if grp == "new":
                    kk = kk.assign(t=kk.t // 5)               # 5분 행을 «1분» 눈금으로 줄여 SM.bars 재사용(쌍봉 2라운드 selftest2)
                for tf, mnt in CB.TFS.items():
                    B = SM.bars(kk, mnt // 5 if grp == "new" else mnt)
                    if grp == "new":
                        B["t"] = B["t"] * 5; B["ms"] *= 5
                    ev = pd.DataFrame(events(B, tf, draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                    if draw < 0:
                        rows.append(ev)
                    elif len(ev):
                        aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
                print(grp, coin, draw, flush=True)
        E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
        E.to_parquet(OUT / f"events_{grp}.parquet"); PA.to_parquet(OUT / f"placebo_agg_{grp}.parquet")
        res[grp] = judge_group(E, PA, coins, 3 if grp == "orig" else 4)
    SM.COINS = ["ETH", "BTC", "SOL", "XRP"]
    final = {k: bool(v.get("pass") and res["new"]["main"].get(k, {}).get("pass")) for k, v in res["orig"]["main"].items() if "holm" in v}
    json.dump(dict(final_pass=final, **res), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    print("final", final)


def selftest() -> None:
    c = np.r_[np.linspace(100, 120, 60), 119.0, 118.0]; h = c + 0.3; l = c - 0.3
    e = ema(c, 20); l[61] = e[61] - 0.1; c[61] = e[61] + 0.5; h[61] = c[61] + 0.1   # 봉 61 저가가 EMA20 아래로 찍고 위에서 닫음
    tc = touch(h, l, c, e)
    assert tc[61] == 1 and (tc[:61] == 0).all()
    c2 = c.copy(); c2[61] = e[61] - 0.05; assert touch(h, l, c2, e)[61] == 0          # 종가도 아래 = 터치 아님(이탈)
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
