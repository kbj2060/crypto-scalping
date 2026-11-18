#!/usr/bin/env python3
"""A11(15분 OB·FVG 버팀)·A13(큰 스윙 구조 돌파 2 ATR) 2라운드 — 안 본 코인 5종 + 모멘텀 대조 P3 (2026-11-16).
사전등록 docs/prereg/a11_a13_newcoins_r2_20261116.md (3e6c73d9, 계산 전).

  selftest   5분→봉 묶음 · 2 ATR 경주 · P3m 가중 단언
  run        새 코인 5종 실제 + P1 20벌 (+4코인 실제 P3 보조) → tmp/a11_a13_r2_20261116/result.json
정의·경주·위약·판정 = research_smc_ict_meaning_20261115 (지표 세션 07fea513 그대로).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/a11_a13_r2_20261116"
NEW = ["ADA", "AVAX", "BNB", "DOGE", "LINK"]
TFS = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
A11_TFS = ("15m", "30m", "1h")
P3STEP = {"5m": 12, "15m": 4, "30m": 2, "1h": 1}            # 60분마다 · 1시간은 매 봉
A11_CELLS = ["S3_N5", "S3_N50", "S5", "P2S3_N5", "P2S3_N50", "P2S5"]


def load5(coin: str) -> pd.DataFrame:
    root = Path("/home/kbj20/crypto-scalping/data")
    d = pd.concat([pd.read_csv(root / f"{coin.lower()}_5m_{p}.csv", usecols=["ts", "high", "low", "close"]) for p in ("2022_2023", "2024_2026")])
    d = d.rename(columns={"ts": "t", "high": "h", "low": "l", "close": "c"}).astype({"t": "int64"})
    return d.drop_duplicates("t").sort_values("t").reset_index(drop=True)


def bars5(k5: pd.DataFrame, minutes: int) -> dict:
    """5분봉을 «1분» 눈금처럼 묶음(지표 세션 2라운드와 같은 방식) → minutes 봉."""
    B = SM.bars(k5.assign(t=k5.t // 5), minutes // 5)
    B["t"] = B["t"] * 5; B["ms"] *= 5
    return B


def a13_events(B: dict, with_p3: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    a2 = 2 * atr                                              # first_hit_race 는 atr[s] 만 읽는다 → 2 ATR 경주
    z = np.abs(c - np.r_[np.full(20, np.nan), c[:-20]]) / atr  # |직전 20봉 수익| / ATR
    R = []
    ev = SM.structure(h, l, c, SM.pivots(h, 50, True), SM.pivots(l, 50, False), 50)
    for s, d, typ in ev[:, :3]:
        s = int(s)
        if s + 1 < len(c) and atr[s] > 0:
            y = SM.first_hit_race(h, l, c, a2, s, int(d), 100)
            if y >= 0:
                for cell in ("K", "KC" if typ == 2 else "KB"):
                    R.append(dict(cell=cell, tk=int(t[s]) + B["ms"], y=float(y), z=z[s]))
    if with_p3:
        for i in range(250, len(c) - 1, P3STEP[B["tf"]]):
            d = np.sign(c[i] - c[i - 20])
            if d != 0 and atr[i] > 0:
                y = SM.first_hit_race(h, l, c, a2, i, int(d), 100)
                if y >= 0:
                    R.append(dict(cell="P3K", tk=int(t[i]) + B["ms"], y=float(y), z=z[i]))
    return R


def p3m(ev_z: np.ndarray, p3: pd.DataFrame) -> float:
    """P3 를 실제 사건 z 의 10분위 경계로 나눠 분위별 비율을 같은 비중으로 평균."""
    edges = np.nanquantile(ev_z, np.linspace(0, 1, 11))
    zz = p3.z.to_numpy(); ok = np.isfinite(zz) & (zz >= edges[0]) & (zz <= edges[-1])
    bi = np.clip(np.searchsorted(edges, zz[ok], "right") - 1, 0, 9)
    return float(pd.Series(p3.y.to_numpy()[ok]).groupby(bi).mean().reindex(range(10)).mean())


def p3m_row(E: pd.DataFrame, tf: str, cell: str = "K", nb: int = 500) -> dict:
    r, p = E[(E.tf == tf) & (E.cell == cell)], E[(E.tf == tf) & (E.cell == "P3K")]
    day = lambda x: (x.tk // 86_400_000).to_numpy()
    dr, dp = day(r), day(p); uu = np.union1d(dr, dp)
    ir = {u: np.flatnonzero(dr == u) for u in np.unique(dr)}; ip = {u: np.flatnonzero(dp == u) for u in np.unique(dp)}
    rng = np.random.default_rng(SM.SEED); bs = []
    for _ in range(nb):
        dd = rng.choice(uu, len(uu))
        a = np.concatenate([ir[u] for u in dd if u in ir]); b = np.concatenate([ip[u] for u in dd if u in ip])
        bs.append(r.y.to_numpy()[a].mean() - p3m(r.z.to_numpy()[a], p.iloc[b]))
    m = p3m(r.z.to_numpy(), p)
    return dict(n=int(len(r)), real=round(r.y.mean(), 4), p3=round(p.y.mean(), 4), p3m=round(m, 4),
                diff_p3m_pp=round(100 * (r.y.mean() - m), 2), lo90_p3m_pp=round(100 * float(np.quantile(bs, .10)), 2))


def rejudge_coins(res: dict) -> None:
    """judge 는 코인 ≥3 고정 → 사전등록 ≥4/5 로 다시."""
    for v in res.values():
        if "holm" not in v:
            continue
        ok = sum(x > 0 for x in v["coin_diff_pp"].values()) >= 4 and all(x > 0 for x in v["half_diff_pp"].values())
        v["pass"] = bool(v["holm"] and v["lo90_pp"] > 0 and v.get("lo90_2_pp", -1) > 0 and ok)
        v["candidate"] = bool(v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and ok)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs = [], []
    for ci, coin in enumerate(NEW):
        k = load5(coin)
        for draw in range(-1, SM.NPLACEBO):
            kk = k if draw < 0 else SM.placebo(k, SM.SEED + 9000 + 1000 * ci + draw)
            for tf, mnt in TFS.items():
                B = bars5(kk, mnt); B["tf"] = tf
                rng = np.random.default_rng(SM.SEED + 7 * ci + 13 * (draw + 1) + mnt)
                R = a13_events(B, with_p3=draw < 0)
                if tf in A11_TFS:
                    R += [e for e in SM.events(B, rng, with_p2=draw < 0) if e["cell"] in A11_CELLS]
                ev = pd.DataFrame(R).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev)
                else:
                    aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")

    SM.TFS, SM.COINS = TFS, NEW
    out = {"a11": {}, "a13": {}, "a13_aux": {}, "a13_p3m": {}, "old4_p3": {}}
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = ("15m",), ["S3_N5", "S3_N50", "S5"], {"S3_N5": "P2S3_N5", "S3_N50": "P2S3_N50", "S5": "P2S5"}
    out["a11"] = {k_: v for k_, v in SM.judge(E[E.tf.isin(A11_TFS)], PA).items() if k_.split("|")[0] in A11_TFS}
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = ("30m", "1h"), ["K"], {"K": "P3K"}
    out["a13"] = SM.judge(E, PA)
    SM.MAIN_TFS, SM.CELLS, SM.P2OF = (), ["KC", "KB"], {"KC": "P3K", "KB": "P3K"}
    out["a13_aux"] = SM.judge(E, PA)
    rejudge_coins(out["a11"]); rejudge_coins(out["a13"])
    for tf in TFS:
        for cell in ("K", "KC", "KB"):
            out["a13_p3m"][f"{tf}|{cell}"] = p3m_row(E, tf, cell)
    for coin in SM.COINS_OLD:                                     # 보조: 4코인 1분봉 실제만
        k = SM.load_1m(f"{coin}USDT")
        for tf in ("30m", "1h"):
            B = SM.bars(k, TFS[tf]); B["tf"] = tf
            e = pd.DataFrame(a13_events(B, True)).assign(tf=tf)
            out["old4_p3"][f"{coin}|{tf}"] = p3m_row(e, tf)
        print(coin, "old4", flush=True)
    json.dump(out, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for sec in ("a11", "a13"):
        for k_, v in out[sec].items():
            print(sec, k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "holm", "pass", "candidate")})
    for sec in ("a13_aux", "a13_p3m", "old4_p3"):
        for k_, v in out[sec].items():
            print(sec, k_, {x: v.get(x) for x in ("n", "real", "p1", "p2", "p3", "p3m", "diff_pp", "diff2_pp", "lo90_2_pp", "diff_p3m_pp", "lo90_p3m_pp")})


SM.COINS_OLD = list(SM.COINS)                                     # judge 가 COINS 를 덮기 전에 4코인 보관


def selftest() -> None:
    # 5분 → 15분: 세 봉 묶음 고가·저가·종가
    t = np.arange(6) * 300_000 + 1_700_000_100_000 // 900_000 * 900_000
    k = pd.DataFrame({"t": t, "h": [3, 5, 4, 6, 7, 8.], "l": [1, 2, 1.5, 4, 5, 3.], "c": [2, 4, 3, 5, 6, 7.]})
    B = bars5(k, 15)
    assert B["h"].tolist() == [5, 8] and B["l"].tolist() == [1, 3] and B["c"].tolist() == [3, 7] and B["ms"] == 900_000
    assert (B["t"] == t[[0, 3]]).all()
    # 2 ATR 경주 = atr 두 배 배열
    c = np.full(10, 100.0); h = c + 0.5; l = c - 0.5; atr = np.ones(10)
    h[3] = 101.5; l[4] = 98.9; h[6] = 102.1
    assert SM.first_hit_race(h, l, c, 2 * atr, 0, 1, 100) == 1 and SM.first_hit_race(h, l, c, atr, 0, -1, 100) == 0
    # P3m: 사건 z 가 전부 높은 분위면 높은 z 의 P3 만 반영
    p = pd.DataFrame({"z": np.r_[np.full(50, 0.1), np.linspace(2, 3, 50)], "y": np.r_[np.zeros(50), np.ones(50)]})
    assert p3m(np.linspace(2, 3, 30), p) == 1.0
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
