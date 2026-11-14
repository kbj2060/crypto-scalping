#!/usr/bin/env python3
"""CHoCH 구조 상태를 «숏 금지» 거르개로 (2026-11-15). 사전등록 docs/prereg/choch_state_short_filter_20261115.md (d52e97b1).

  selftest        상태 앞채움·SMA50 알려지는 시각·차이 부호 단언
  run             4코인 15분 실제 + 기하 위약 20벌 → 상태별 숏 수익 차이
  ledger <trips>  실원장 숏 왕복 상태별
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "tmp/choch_state_short_filter_20261115"
HB = {1: 4, 4: 16, 24: 96}                 # 시간 → 15분봉 수
WEEK = 7 * 86_400_000
SPLIT = SM.SPLIT


def struct_state(B: dict, N: int) -> np.ndarray:
    """봉 t 마감까지 마지막 구조 돌파 방향(+1 위 / −1 아래 / 0 아직 없음)."""
    ev = SM.structure(B["h"], B["l"], B["c"], SM.pivots(B["h"], N, True), SM.pivots(B["l"], N, False), N)
    st = np.full(len(B["c"]), np.nan)
    for t, d in ev[:, :2]:                 # 같은 봉에 둘이면 나중 것(구조 함수 순서 = 위 → 아래)
        st[int(t)] = d
    return pd.Series(st).ffill().fillna(0).to_numpy()


def sma_state(B: dict) -> np.ndarray:
    """전날 마감까지 일봉 종가 vs SMA50. 봉 t 의 마감 시각 기준으로 마감된 날만."""
    end = B["t"] + B["ms"]
    day = pd.Series(B["c"], index=pd.to_datetime(B["t"], unit="ms")).resample("D").last().ffill()
    s = np.sign(day - day.rolling(50).mean()).fillna(0)
    known = s.index.values.astype("datetime64[ms]").astype(np.int64) + 86_400_000
    ix = np.searchsorted(known, end, "right") - 1
    return np.where(ix >= 0, s.to_numpy()[np.maximum(ix, 0)], 0)


def frame(k: pd.DataFrame) -> pd.DataFrame:
    B = SM.bars(k, 15); c = B["c"]; n = len(c)
    f = pd.DataFrame(dict(t=B["t"] + B["ms"], N5=struct_state(B, 5), N50=struct_state(B, 50), SMA=sma_state(B)))
    B1 = SM.bars(k, 60); s1 = struct_state(B1, 50); e1 = B1["t"] + B1["ms"]
    ix = np.searchsorted(e1, f.t.to_numpy(), "right") - 1      # 마감한 1시간봉만
    f["H1N50"] = np.where(ix >= 0, s1[np.maximum(ix, 0)], 0)
    for H, b in HB.items():
        fw = np.full(n, np.nan); fw[: n - b] = -(c[b:] / c[: n - b] - 1) * 1e4
        f[f"s{H}"] = fw
    for col in ("N5", "N50"):              # 상태가 바뀐 뒤 지난 봉 수
        chg = np.r_[True, f[col].to_numpy()[1:] != f[col].to_numpy()[:-1]]
        grp = np.cumsum(chg); f[f"age_{col}"] = f.groupby(grp).cumcount().to_numpy()
    return f


def diff(f: pd.DataFrame, col: str, H: int) -> float:
    """숏 수익(아래 상태) − 숏 수익(위 상태). 양수 = 위 상태에서 숏을 막는 게 맞음."""
    x = f[f"s{H}"]
    return float(x[f[col] == -1].mean() - x[f[col] == 1].mean())


def boot(f: pd.DataFrame, col: str, H: int, rng, B=2000) -> list[float]:
    g = f.dropna(subset=[f"s{H}"]); g = g[g[col] != 0]
    blk = (g.t // WEEK).to_numpy(); ub, inv = np.unique(blk, return_inverse=True)
    dn, up = (g[col] == -1).to_numpy(), (g[col] == 1).to_numpy(); x = g[f"s{H}"].to_numpy()
    sd, nd = np.bincount(inv, x * dn), np.bincount(inv, dn); su, nu = np.bincount(inv, x * up), np.bincount(inv, up)
    bs = []
    for ix in rng.integers(0, len(ub), (B, len(ub))):
        bs.append(sd[ix].sum() / nd[ix].sum() - su[ix].sum() / nu[ix].sum())
    return [float(np.quantile(bs, .10)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SM.SEED)
    real, plc = [], {d: [] for d in range(SM.NPLACEBO)}
    for ci, coin in enumerate(SM.COINS):
        k = SM.load_1m(f"{coin}USDT")
        real.append(frame(k).assign(coin=coin))
        for d in range(SM.NPLACEBO):
            plc[d].append(frame(SM.placebo(k, SM.SEED + 1000 * ci + d))[["t", "N50", "s4"]])
        print(coin, flush=True)
    F = pd.concat(real, ignore_index=True)
    P = [diff(pd.concat(v), "N50", 4) for v in plc.values()]
    main = dict(diff_bp=diff(F, "N50", 4), boot=boot(F, "N50", 4, rng), placebo=[round(p, 2) for p in P],
                coins={c: diff(F[F.coin == c], "N50", 4) for c in SM.COINS},
                halves={"22-24": diff(F[F.t < SPLIT], "N50", 4), "25-26": diff(F[F.t >= SPLIT], "N50", 4)})
    main["pass"] = bool(main["boot"][0] > 0 and main["diff_bp"] > np.quantile(P, .95) and sum(v > 0 for v in main["coins"].values()) >= 3
                        and all(v > 0 for v in main["halves"].values()))
    main["candidate"] = bool(main["diff_bp"] > 0 and sum(v > 0 for v in main["coins"].values()) >= 3 and all(v > 0 for v in main["halves"].values()))
    aux = {}
    for col in ("N5", "N50", "SMA", "H1N50"):
        for H in HB:
            x = F[f"s{H}"]
            aux[f"{col}|{H}h"] = dict(diff=round(diff(F, col, H), 2), ci95=[round(v, 2) for v in boot(F, col, H, rng)[1:]],
                                      short_up=round(x[F[col] == 1].mean(), 2), short_dn=round(x[F[col] == -1].mean(), 2),
                                      y2026=round(diff(F[F.t >= SM.SPLIT + 365 * 86_400_000], col, H), 2))
    sma_up = F[F.SMA == 1]; sma_dn = F[F.SMA == -1]
    aux["N50_within_SMA_up|4h"] = round(diff(sma_up, "N50", 4), 2); aux["N50_within_SMA_dn|4h"] = round(diff(sma_dn, "N50", 4), 2)
    beh = {}
    for col in ("N5", "N50", "SMA", "H1N50"):
        per = []
        for c, g in F.groupby("coin"):
            v = g[col].to_numpy(); flips = (v[1:] != v[:-1]).sum(); days = (g.t.iloc[-1] - g.t.iloc[0]) / 86_400_000
            per.append((flips / days, (v == 1).mean()))
        beh[col] = dict(flips_per_day=round(np.mean([p[0] for p in per]), 2), mean_hold_h=round(24 / np.mean([p[0] for p in per]), 1), up_share=round(np.mean([p[1] for p in per]), 3))
    fresh = {}
    for col in ("N5", "N50"):
        for lab, m in (("first_4h", F[f"age_{col}"] < 16), ("after_4h", F[f"age_{col}"] >= 16)):
            fresh[f"{col}|{lab}"] = round(diff(F[m], col, 4), 2)
    res = dict(main=main, aux=aux, behaviour=beh, fresh_vs_old_4h=fresh)
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))


def ledger(path: str) -> None:
    R = [json.loads(x) for x in open(path) if x.strip()]
    R = [r for r in R if r.get("closed", True) and r["side"] == "SHORT"]
    base = lambda s: s.replace("USDT", "").replace("USDC", "")
    rows = []
    for coin in sorted({base(r["symbol"]) for r in R}):
        k = SM.load_1m(f"{coin}USDT"); k = k[k.t >= int(pd.Timestamp("2025-06-01", tz="UTC").value // 10**6)].reset_index(drop=True)
        B = SM.bars(k, 15); st, ss = struct_state(B, 50), sma_state(B); end = B["t"] + B["ms"]; t1, c1 = k.t.to_numpy(), k.c.to_numpy()
        for r in [r for r in R if base(r["symbol"]) == coin]:
            t0 = r["entry_time"]; b = int(np.searchsorted(end, t0, "right")) - 1
            m0 = np.searchsorted(t1, t0 // 60000 * 60000) - 1; row = dict(coin=coin, t0=t0, net=r["net_pnl"], N50=st[b] if b >= 0 else 0, SMA=ss[b] if b >= 0 else 0)
            for Hm in (60, 240):
                mh = np.searchsorted(t1, t0 // 60000 * 60000 + Hm * 60000)
                ok = m0 >= 0 and mh < len(t1) and t1[mh] == t0 // 60000 * 60000 + Hm * 60000
                row[f"f{Hm}"] = -(c1[mh] / c1[m0] - 1) * 1e4 if ok else np.nan
            rows.append(row)
    d = pd.DataFrame(rows); d["day"] = (d.t0 + 9 * 3_600_000) // 86_400_000
    out = {col: {int(s): dict(n=int(len(g)), f60=round(g.f60.mean(), 1), f240=round(g.f240.mean(), 1), net=round(g.net.sum(), 0), win=round((g.net > 0).mean(), 3))
                 for s, g in d.groupby(col)} for col in ("N50", "SMA")}
    x = d.dropna(subset=["f240"]); x = x[x.N50 != 0]; dn = (x.N50 == -1).to_numpy(); f = x.f240.to_numpy(); days = x.day.to_numpy()
    rng = np.random.default_rng(SM.SEED); ud = np.unique(days); idx = {u: np.flatnonzero(days == u) for u in ud}; bs = []
    for _ in range(2000):
        ix = np.concatenate([idx[u] for u in rng.choice(ud, len(ud))])
        if dn[ix].any() and (~dn[ix]).any():
            bs.append(f[ix][dn[ix]].mean() - f[ix][~dn[ix]].mean())
    out["test_dn_minus_up_f240"] = dict(n_dn=int(dn.sum()), n_up=int((~dn).sum()), diff=round(f[dn].mean() - f[~dn].mean(), 1), ci95=[round(v, 1) for v in np.percentile(bs, [2.5, 97.5])])
    out["by_coin"] = {c: {int(s): [int(len(h)), round(h.f240.mean(), 1), round(h.net.sum(), 0)] for s, h in g.groupby("N50")} for c, g in d.groupby("coin")}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "ledger.json", "w"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


def selftest() -> None:
    # 상태: 위로 돌파(봉 6) 뒤 앞채움, 아래 돌파(봉 9) 뒤 −1
    h = np.array([5, 8, 10, 7, 6, 7, 12, 11, 9, 4, 3, 5.]); l = h - 1; c = h - 0.5; l[4] = 4.0
    B = dict(h=h, l=l, c=c)
    st = struct_state(B, 2)
    ev = SM.structure(h, l, c, SM.pivots(h, 2, True), SM.pivots(l, 2, False), 2)
    t_up, t_dn = int(ev[ev[:, 1] == 1][0, 0]), int(ev[ev[:, 1] == -1][0, 0])
    assert (st[:t_up] == 0).all() and (st[t_up:t_dn] == 1).all() and st[t_dn] == -1
    # SMA50: 날 d 의 값은 d+1 00:00 이후 봉부터
    t = np.arange(60 * 96) * 900_000; cc = np.r_[np.full(55 * 96, 100.0), np.full(5 * 96, 200.0)]
    s = sma_state(dict(t=t, c=cc, ms=900_000))
    jump = 55 * 96                                         # 200 이 시작하는 날
    assert s[jump] == 0 and s[jump + 94] == 0 and s[jump + 95] == 1   # 그날 마지막 봉(마감 = 다음 날 00:00)부터 그날 종가를 앎
    # 차이 부호
    f = pd.DataFrame(dict(N50=[1, 1, -1, -1], s4=[-5.0, -3.0, 4.0, 6.0]))
    assert diff(f, "N50", 4) == 9.0
    print("selftest ok")


if __name__ == "__main__":
    cmd, *arg = sys.argv[1:]
    dict(selftest=selftest, run=run, ledger=ledger)[cmd](*arg)
