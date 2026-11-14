#!/usr/bin/env python3
"""추세 거르개 묶음 — 일목 구름·슈퍼트렌드·EMA200 «롱 추세면 숏 금지» (2026-11-15). 사전등록 docs/prereg/trend_filter_bundle_20261115.md (67780d6d).

  selftest        손계산 봉으로 구름 선행 26·슈퍼트렌드 뒤집힘·EMA·마감 봉 정렬 단언
  run             4코인 15분 실제 + 기하 위약 20벌
  ledger <trips>  실원장 숏 상태별(서술)
측정·부트스트랩·원장은 research_choch_state_short_filter_20261115 를 그대로 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_choch_state_short_filter_20261115 as CS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/trend_filter_bundle_20261115"
TFS = {"1h": 60, "4h": 240, "1d": 1440}
IND = ("ICH", "ST", "EMA")
STATES = [f"{i}_{tf}" for i in IND for tf in TFS]
BASE = ("SMA", "N50")


def ichimoku(B: dict) -> np.ndarray:
    h, l, c = (pd.Series(B[x]) for x in "hlc")
    mid = lambda n: (h.rolling(n).max() + l.rolling(n).min()) / 2
    a, b = ((mid(9) + mid(26)) / 2).shift(26), mid(52).shift(26)      # 봉 t 의 구름 = t−26 에 계산된 값
    top, bot = np.maximum(a, b), np.minimum(a, b)               # 둘 다 있어야(fmax 는 NaN 을 건너뛰어 선행B 없이도 구름을 만든다)
    return np.where(c > top, 1.0, np.where(c < bot, -1.0, 0.0)) * (top.notna().to_numpy())


@njit(cache=True)
def _st(h, l, c, atr, m):
    n = len(c); d = np.zeros(n); fu = np.nan; fl = np.nan; cur = 0.0
    for t in range(n):
        if not atr[t] > 0:
            continue
        hl2 = (h[t] + l[t]) / 2; up, lo = hl2 + m * atr[t], hl2 - m * atr[t]
        if t > 0 and fu == fu:
            fu = up if (up < fu or c[t - 1] > fu) else fu
            fl = lo if (lo > fl or c[t - 1] < fl) else fl
        else:
            fu, fl = up, lo
        if cur >= 0 and c[t] < fl:
            cur = -1.0
        elif cur <= 0 and c[t] > fu:
            cur = 1.0
        d[t] = cur
    return d


def supertrend(B: dict, n=10, m=3.0) -> np.ndarray:
    c = B["c"]; cp = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[B["h"] - B["l"], np.abs(B["h"] - cp), np.abs(B["l"] - cp)], axis=1)
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()   # RMA
    return _st(B["h"], B["l"], c, atr, m)


def ema_state(B: dict, n=200) -> np.ndarray:
    e = pd.Series(B["c"]).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()
    return np.where(np.isfinite(e), np.sign(B["c"] - e), 0.0)


def align(Bx: dict, st: np.ndarray, t15_end: np.ndarray) -> np.ndarray:
    """15분봉 마감 시각에 마감한 상위 봉의 상태."""
    ix = np.searchsorted(Bx["t"] + Bx["ms"], t15_end, "right") - 1
    return np.where(ix >= 0, st[np.maximum(ix, 0)], 0.0)


def frame(k: pd.DataFrame) -> pd.DataFrame:
    f = CS.frame(k)                                                    # t(15분 마감) · N5 · N50 · SMA · s1·s4·s24 …
    for tf, m in TFS.items():
        Bx = SM.bars(k, m)
        for name, fn in (("ICH", ichimoku), ("ST", supertrend), ("EMA", ema_state)):
            f[f"{name}_{tf}"] = align(Bx, fn(Bx), f.t.to_numpy())
    return f


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SM.SEED)
    real, plc = [], {d: [] for d in range(SM.NPLACEBO)}
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        real.append(frame(k).assign(coin=coin))
        for d in range(SM.NPLACEBO):
            plc[d].append(frame(SM.placebo(k, SM.SEED + 1000 * ci + d))[["t", "s4"] + STATES])
        print(coin, flush=True)
    F = pd.concat(real, ignore_index=True)
    PL = {d: pd.concat(v) for d, v in plc.items()}
    from scipy.stats import norm
    res = {}
    for col in STATES + list(BASE):
        b = CS.boot(F, col, 4, rng)
        r = dict(diff=CS.diff(F, col, 4), lo90=b[0], ci95=b[1:],
                 short_up=float(F.s4[F[col] == 1].mean()), short_dn=float(F.s4[F[col] == -1].mean()),
                 coins={c: CS.diff(F[F.coin == c], col, 4) for c in SM.COINS},
                 halves={"22-24": CS.diff(F[F.t < SM.SPLIT], col, 4), "25-26": CS.diff(F[F.t >= SM.SPLIT], col, 4)},
                 h1=CS.diff(F, col, 1), h24=CS.diff(F, col, 24), y2026_4h=CS.diff(F[F.t >= SM.SPLIT + 365 * 86_400_000], col, 4))
        if col in STATES:
            P = np.array([CS.diff(PL[d], col, 4) for d in PL]); r["placebo"] = [round(x, 2) for x in P]; r["placebo95"] = float(np.quantile(P, .95))
            g = F.dropna(subset=["s4"]); g = g[(g[col] != 0) & (g.SMA != 0)]       # SMA50 대비 차이 (같은 7일 블록)
            blk = (g.t // CS.WEEK).to_numpy(); ub, inv = np.unique(blk, return_inverse=True); x = g.s4.to_numpy()
            agg = {nm: (np.bincount(inv, x * (g[cc] == sv).to_numpy()), np.bincount(inv, (g[cc] == sv).to_numpy())) for nm, cc, sv in
                   (("dn", col, -1), ("up", col, 1), ("sdn", "SMA", -1), ("sup", "SMA", 1))}
            q = lambda ix, nm: agg[nm][0][ix].sum() / agg[nm][1][ix].sum()
            bs = [(q(ix, "dn") - q(ix, "up")) - (q(ix, "sdn") - q(ix, "sup")) for ix in rng.integers(0, len(ub), (2000, len(ub)))]
            r["minus_sma"] = [float(np.mean(bs)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        res[col] = r
    # Holm 9칸 (단측 p 는 부트스트랩 분포 대신 정규 근사: diff / ((diff − lo90)/1.2816))
    ps = {c: float(1 - norm.cdf(res[c]["diff"] / max((res[c]["diff"] - res[c]["lo90"]) / 1.2816, 1e-9))) for c in STATES}
    alive = True
    for i, c in enumerate(sorted(STATES, key=ps.get)):
        alive = alive and ps[c] <= .10 / (9 - i); res[c]["holm"] = alive; res[c]["p_one"] = ps[c]
    for c in STATES:
        r = res[c]; cons = sum(v > 0 for v in r["coins"].values()) >= 3 and all(v > 0 for v in r["halves"].values())
        r["pass"] = bool(r["holm"] and r["lo90"] > 0 and r["diff"] > r["placebo95"] and cons)
        r["candidate"] = bool(r["diff"] > 0 and cons)
    beh = {}
    for col in STATES + list(BASE):
        per = []
        for c, g in F.groupby("coin"):
            v = g[col].to_numpy(); days = (g.t.iloc[-1] - g.t.iloc[0]) / 86_400_000
            per.append(((v[1:] != v[:-1]).sum() / days, (v == 1).mean(), (v == -1).mean()))
        fl = np.mean([p[0] for p in per])
        beh[col] = dict(flips_per_day=round(fl, 3), mean_hold_h=round(24 / fl, 1) if fl else None, up=round(np.mean([p[1] for p in per]), 3), dn=round(np.mean([p[2] for p in per]), 3))
    out = dict(result=res, behaviour=beh)
    json.dump(out, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for c in STATES + list(BASE):
        r = res[c]
        print(c, round(r["diff"], 2), [round(x, 2) for x in r["ci95"]], "lo90", round(r["lo90"], 2), "p95", round(r.get("placebo95", np.nan), 2),
              {k: round(v, 1) for k, v in r["coins"].items()}, {k: round(v, 1) for k, v in r["halves"].items()},
              "1h", round(r["h1"], 2), "24h", round(r["h24"], 2), "26", round(r["y2026_4h"], 2), "−SMA", [round(x, 2) for x in r.get("minus_sma", [])],
              r.get("pass"), r.get("candidate"), beh[c])


def ledger(path: str) -> None:
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if r.get("closed", True) and r["side"] == "SHORT"]
    base = lambda s: s.replace("USDT", "").replace("USDC", "")
    rows = []
    for coin in sorted({base(r["symbol"]) for r in R}):
        k = SM.load_1m(f"{coin}USDT"); k = k[k.t >= int(pd.Timestamp("2025-06-01", tz="UTC").value // 10**6)].reset_index(drop=True)
        st = {}
        for tf, m in TFS.items():
            Bx = SM.bars(k, m)
            for name, fn in (("ICH", ichimoku), ("ST", supertrend), ("EMA", ema_state)):
                st[f"{name}_{tf}"] = (Bx["t"] + Bx["ms"], fn(Bx))
        t1, c1 = k.t.to_numpy(), k.c.to_numpy()
        for r in [r for r in R if base(r["symbol"]) == coin]:
            t0 = r["entry_time"]; m0 = np.searchsorted(t1, t0 // 60000 * 60000) - 1
            mh = np.searchsorted(t1, t0 // 60000 * 60000 + 240 * 60000)
            ok = m0 >= 0 and mh < len(t1) and t1[mh] == t0 // 60000 * 60000 + 240 * 60000
            row = dict(coin=coin, net=r["net_pnl"], f240=-(c1[mh] / c1[m0] - 1) * 1e4 if ok else np.nan)
            for col, (end, v) in st.items():
                ix = int(np.searchsorted(end, t0, "right")) - 1; row[col] = v[ix] if ix >= 0 else 0
            rows.append(row)
    d = pd.DataFrame(rows)
    out = {col: {int(s): dict(n=int(len(g)), f240=round(g.f240.mean(), 1), net=round(g.net.sum(), 0)) for s, g in d.groupby(col)} for col in STATES}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "ledger.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False))


def selftest() -> None:
    # 구름: 오르는 직선이면 충분히 지나 위 · 내려가면 아래 · 앞 77봉(52+26−1)은 0
    n = 200; up = np.linspace(100, 200, n)
    B = dict(h=up + 0.5, l=up - 0.5, c=up)
    s = ichimoku(B)
    assert (s[:77] == 0).all() and (s[100:] == 1).all()
    s2 = ichimoku(dict(h=up[::-1] + 0.5, l=up[::-1] - 0.5, c=up[::-1])); assert (s2[100:] == -1).all()
    # 슈퍼트렌드: 오르다가 크게 꺾이면 위 → 아래
    c = np.r_[np.linspace(100, 150, 100), np.linspace(150, 100, 100)]
    st = supertrend(dict(h=c + 0.3, l=c - 0.3, c=c))
    assert st[90] == 1 and st[-1] == -1 and (np.diff(st[20:]) != 0).sum() == 1
    # EMA200: 199봉까지 0, 오르는 직선은 위
    e = ema_state(dict(c=up)); assert (e[:199] == 0).all() and e[-1] == 1
    # 정렬: 1시간봉(시작 0, 3.6e6) 상태를 15분 마감 시각에 — 첫 1시간봉은 3.6e6 마감 뒤부터
    Bx = dict(t=np.array([0, 3_600_000]), ms=3_600_000)
    a = align(Bx, np.array([1.0, -1.0]), np.array([900_000, 3_600_000, 4_500_000, 7_200_000]))
    assert a.tolist() == [0.0, 1.0, 1.0, -1.0]
    print("selftest ok")


if __name__ == "__main__":
    cmd, *arg = sys.argv[1:]
    dict(selftest=selftest, run=run, ledger=ledger)[cmd](*arg)
