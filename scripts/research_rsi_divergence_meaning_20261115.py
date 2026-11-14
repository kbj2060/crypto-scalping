#!/usr/bin/env python3
"""RSI 다이버전스가 «뜻하는 대로» 움직이는가 (2026-11-15). 사전등록 docs/prereg/rsi_divergence_meaning_20261115.md (473bbd8b, 계산 전).

  selftest   손계산 봉으로 RSI·다이버전스 분류·알려지는 시각·70 첫 진입 단언
  run        4코인 실제 + 기하 위약 20벌 → tmp/rsi_divergence_meaning_20261115/{events,placebo_agg}.parquet, result.json
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

OUT = Path(__file__).resolve().parents[1] / "tmp/rsi_divergence_meaning_20261115"
N, GAP, H24, WR = 5, 60, 24, 100


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100).where(up.notna()).to_numpy()


def classify(x: np.ndarray, r: np.ndarray, piv: np.ndarray, high: bool) -> list[tuple[int, str]]:
    """스윙 피벗 p 마다 직전 같은 쪽 피벗 q(p−q ≤ GAP)와 비교 → (p, 'D' 정규 다이버전스 | 'P2' 같은 구조 비다이버전스 | 'HID' 숨은 다이버전스)."""
    out, q = [], -1
    s = 1 if high else -1
    for p in np.flatnonzero(piv):
        if q >= 0 and p - q <= GAP and np.isfinite(r[p]) and np.isfinite(r[q]):
            if s * (x[p] - x[q]) > 0:                       # 더 높은 고점(더 낮은 저점)
                out.append((int(p), "D" if s * (r[p] - r[q]) < 0 else "P2"))
            elif s * (x[p] - x[q]) < 0 and s * (r[p] - r[q]) > 0:   # 더 낮은 고점 + 더 높은 RSI(고점) = 숨은 약세 / 대칭
                out.append((int(p), "HID"))
        q = p
    return out


def events(k: pd.DataFrame, tf: str, minutes: int) -> list[dict]:
    B = SM.bars(k, minutes); h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    r = rsi(c); n = len(c); R = []
    add = lambda cell, kk, y: R.append(dict(cell=cell, tf=tf, tk=int(t[kk]) + B["ms"], y=float(y)))
    for high in (True, False):
        x = h if high else l
        rev = -1 if high else 1                               # 정규 다이버전스·P2 의 주장 방향 = 반전
        for p, kind in classify(x, r, SM.pivots(x, N, high), high):
            kk = p + N
            if kk + H24 >= n:
                continue
            d = rev if kind in ("D", "P2") else -rev          # 숨은 다이버전스 = 지속(추세 쪽)
            y1 = SM.first_hit_race(h, l, c, atr, kk, d, WR)
            pre = {"D": "", "P2": "P2", "HID": "HID_"}[kind]
            if y1 >= 0:
                add(pre + "D1" if kind != "HID" else "HID", kk, y1)
            if kind != "HID":
                add(pre + "D2", kk, np.sign(c[kk + H24] - c[kk]) == d)
    if tf == "15m":                                           # 보조: RSI 70 위로 / 30 아래로 처음 들어간 봉 → 반전 1 ATR 먼저
        for thr, d in ((70, -1), (30, 1)):
            cross = np.flatnonzero((r[1:] > thr) & (r[:-1] <= thr)) + 1 if d < 0 else np.flatnonzero((r[1:] < thr) & (r[:-1] >= thr)) + 1
            for kk in cross:
                if kk + 1 < n:
                    y = SM.first_hit_race(h, l, c, atr, int(kk), d, WR)
                    if y >= 0:
                        add("A70", int(kk), y)
    return R


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 1000 * ci + draw)
            ev = pd.DataFrame(events(kk, "15m", 15) + events(kk, "1h", 60)).assign(coin=coin, draw=draw)
            if draw < 0:
                rows.append(ev)
            else:
                ev = ev[~ev.cell.str.startswith("P2")]
                aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    SM.P2OF = {"D1": "P2D1", "D2": "P2D2"}
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {"15m": 15}, ("15m",), ["D1", "D2"]       # 주 판정: 15분 2칸 Holm
    main = {kk.split("|")[1]: v for kk, v in SM.judge(E, PA).items()}
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {"15m": 15, "1h": 60}, (), ["D1", "D2", "A70", "HID"]   # 보조(판정 없음)
    aux = SM.judge(E, PA)
    res = dict(main=main, aux={kk: v for kk, v in aux.items() if not (kk.startswith("15m|D"))},
               by_side={s: {c_: round(float(E[(E.tf == "15m") & (E.cell == c_) & ((E.y >= 0))].y.mean()), 4) for c_ in ("D1", "P2D1")} for s in ("all",)})
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for c_ in ("D1", "D2"):
        print(c_, {x: main[c_].get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate", "reversed")})
    for kk, v in res["aux"].items():
        print("aux", kk, {x: v.get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp")})


def selftest() -> None:
    c = np.r_[np.linspace(100, 110, 20), np.linspace(110, 100, 20)]
    r = rsi(c)
    assert np.isnan(r[:13]).all() and r[19] > 90 and r[-1] < 30
    # 다이버전스 분류: 고점 둘(봉 10·30), 둘째가 더 높고 RSI 는 낮음 → D · RSI 도 높음 → P2
    x = np.zeros(40); x[10], x[30] = 5.0, 6.0
    piv = np.zeros(40, bool); piv[[10, 30]] = True
    rr = np.full(40, 50.0); rr[10], rr[30] = 70.0, 60.0
    assert classify(x, rr, piv, True) == [(30, "D")]
    rr[30] = 75.0; assert classify(x, rr, piv, True) == [(30, "P2")]
    x[30] = 4.0; rr[30] = 80.0; assert classify(x, rr, piv, True) == [(30, "HID")]      # 더 낮은 고점 + 더 높은 RSI
    x2 = np.zeros(100); x2[10], x2[80] = 5.0, 6.0; piv2 = np.zeros(100, bool); piv2[[10, 80]] = True
    assert classify(x2, np.full(100, 50.0), piv2, True) == []                           # 간격 70 > 60
    # 저점 쪽: 더 낮은 저점 + 더 높은 RSI → D
    xl = np.full(40, 10.0); xl[10], xl[30] = 5.0, 4.0; rl = np.full(40, 50.0); rl[10], rl[30] = 30.0, 35.0
    assert classify(xl, rl, piv, False) == [(30, "D")]
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
