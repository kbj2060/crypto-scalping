#!/usr/bin/env python3
"""청산맵/지지·저항 레벨 스윕 거절 페이드 × 변동성 확장 게이트 (2026-11-02). 사전등록 docs/prereg/levels_vol_20261102.md. 파라미터 전부 고정.
  python scripts/research_levels_vol_20261102.py   (4코인 스냅샷 tmp/levels_4coin_20261102/levels_{COIN}.pkl 필요)
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "research_regime_f_20261102.py")
rf = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rf)
OUT = ROOT / "tmp" / "levels_vol_20261102"
SL_K, TP_K, COOL, MIN_N = 0.5, 1.0, 48, 100        # 손절 0.5σ4h · 익절 1.0σ4h · 쿨다운 48봉 · 최소 건수
CI_LO, CI_HI, PLA = 0.5, 99.5, 99.0                # 양측 99% · 위약 99분위 (이 세션 같은 데이터 다중 시행 보정)


def events(coin):
    df = rf.load(coin)
    snaps = pickle.load(open(ROOT / "tmp" / "levels_4coin_20261102" / f"levels_{coin}.pkl", "rb"))
    o, h, l, c = (df[x].to_numpy() for x in ("o", "h", "l", "c"))
    ts = df.index.values.astype("datetime64[s]").astype("int64")
    r1 = np.diff(np.log(c), prepend=np.nan)
    sig = np.sqrt(pd.Series(r1 ** 2).rolling(48).sum().to_numpy())
    rows, last = [], -10**9
    for i in range(2016, len(c) - rf.H - 2):
        if i - last < COOL:
            continue
        sn = snaps.get((ts[i] + 300) // 3600 * 3600)
        if sn is None:
            continue
        _, sup, res = sn
        up = [x for x in res if h[i] > x >= c[i] and o[i] <= x]            # 저항을 꼬리로 뚫고 안으로 닫음
        dn = [x for x in sup if l[i] < x <= c[i] and o[i] >= x]
        if not (up or dn):
            continue
        side = -1 if up else 1
        rows.append((i, side))
        last = i
    ev = pd.DataFrame(rows, columns=["i", "side"])
    ev["coin"], ev["ts"] = coin, df.index[ev.i.to_numpy() + 1]
    return df, ev, sig


def trades(df, i, side, sig):
    """진입 i+1 시가. SL = max(진입∓0.5σ, 신호 봉 극단) · TP = 진입 ±1.0σ(σ = 직전 48봉 RV). rf.barrier 재사용."""
    o, h, l = df.o.to_numpy(), df.h.to_numpy(), df.l.to_numpy()
    e, s = o[i + 1], sig[i]
    sl = np.where(side < 0, np.maximum(e * (1 + SL_K * s), h[i] * 1.0001), np.minimum(e * (1 - SL_K * s), l[i] * 0.9999))
    tp = e * (1 - side * TP_K * s)
    return rf.barrier(df, i, side, tp, sl)


def main():
    rng = np.random.default_rng(rf.SEED)
    parts, dfs, sigs = [], {}, {}
    for coin in rf.COINS:
        df, ev, sig = events(coin)
        n, g = trades(df, ev.i.to_numpy(), ev.side.to_numpy(), sig)
        ev["net"], ev["gross"], ev["sig"] = n, g, sig[ev.i.to_numpy()]
        ev["gate"] = np.log(sig[ev.i.to_numpy()] / np.sqrt(pd.Series(np.diff(np.log(df.c.to_numpy()), prepend=np.nan) ** 2).rolling(288).sum().to_numpy())[ev.i.to_numpy()] * np.sqrt(288 / 48))
        dfs[coin], sigs[coin] = df, sig
        parts.append(ev)
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.net != 0) | (d.gross != 0)].replace([np.inf, -np.inf], np.nan).dropna(subset=["gate"]).reset_index(drop=True)   # 유효 거래만
    d["day"] = d.ts.dt.date
    tr = d[rf.window(d.ts, *rf.SPLITS["학습"])]
    q1, q2 = tr.gate.quantile([1 / 3, 2 / 3])
    d["gq"] = np.where(d.gate <= q1, 0, np.where(d.gate <= q2, 1, 2))
    pol = {k: bool(tr[d.loc[tr.index, "gq"] == k].net.mean() > 0) for k in range(3)}     # 학습에서 순평균 > 0 인 변동성 등급만 거래
    out = {"gate_terciles": [q1, q2], "policy_trade": pol, "n_train": len(tr)}
    for w in ("학습", "검증", "시험"):
        x = d[rf.window(d.ts, *rf.SPLITS[w])].reset_index(drop=True)
        res = {"n": len(x), "mean_net": float(x.net.mean()), "mean_gross": float(x.gross.mean()), "tp_hit": float((x.gross > 0).mean()),
               "by_gate": {k: dict(n=int((x.gq == k).sum()), net=float(x[x.gq == k].net.mean())) for k in range(3)}}
        gx = x[x.gq.map(pol)]
        res["gated_n"], res["gated_net"] = len(gx), float(gx.net.mean()) if len(gx) else None
        if w != "학습" and len(gx) >= MIN_N:
            res["gated_ci99"] = rf.day_ci(gx.day, gx.net, rng, CI_LO, CI_HI).tolist()
            res["all_ci99"] = rf.day_ci(x.day, x.net, rng, CI_LO, CI_HI).tolist()
            sel = d[rf.window(d.ts, *rf.SPLITS[w])]
            null = np.empty(rf.NPLA)
            for k in range(rf.NPLA):                                         # 위약: 같은 규칙(같은 방향 구성·σ 배리어)을 무작위 봉에서
                m = []
                for coin, g in gx.groupby("coin"):
                    df = dfs[coin]
                    lo_, hi_ = df.index.searchsorted(pd.Timestamp(rf.SPLITS[w][0])), df.index.searchsorted(pd.Timestamp(rf.SPLITS[w][1]))
                    j = rng.integers(max(lo_, 2016), min(hi_, len(df) - rf.H - 3), len(g))
                    s = rng.permutation(g.side.to_numpy())
                    m.append(trades(df, j, s, sigs[coin])[0])
                null[k] = np.concatenate(m).mean()
            res["placebo_pct_gated"] = float((null < gx.net.mean()).mean() * 100)
            res["placebo_mean"] = float(null.mean())
        out[w] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "result.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
