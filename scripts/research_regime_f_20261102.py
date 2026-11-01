#!/usr/bin/env python3
"""횡보/추세 구분(F)·전환(T) 모델 검정 (2026-11-02). 사전등록: docs/prereg/regime_f_20261102.md. 파라미터 전부 고정.
  python scripts/research_regime_f_20261102.py s0 | s2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = "/home/kbj20/crypto-scalping/tmp/meta_label_race_20261101/{}USDT_5m.parquet"
OUT = ROOT / "tmp" / "regime_f_20261102"
COINS = ("BTC", "ETH", "SOL", "XRP")
H, STEP, WARM = 48, 12, 2016            # 지평 4h · 결정 간격 1h · 워밍업 7일
COST = 8.0                               # bp 왕복(진입·청산 테이커 4+4) -- 엄격
TRAIN_END, VAL_END = "2026-01-01", "2026-06-01"
SPLITS = {"전체": ("2024-01-15", "2100-01-01"), "학습": ("2024-01-15", TRAIN_END), "검증": (TRAIN_END, VAL_END), "시험": (VAL_END, "2100-01-01")}
NBOOT, NPLA, SEED = 5000, 1000, 20261102


def load(coin):
    return pd.read_parquet(PANEL.format(coin), columns=["o", "h", "l", "c", "qv", "tbqv", "n"])


def decisions(df):
    """결정 인덱스: 매 시 정각 마감 봉 i. 인과 피쳐(≤i)·진입 i+1 시가·청산 i+H 종가."""
    i = np.arange(len(df))
    ok = (i % STEP == STEP - 1) & (i >= WARM) & (i + H < len(df)) & (df.index.minute.values == 55)
    return i[ok]


def er_fwd(c, i):
    ad = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(c)))])
    return np.abs(c[i + H] - c[i]) / np.maximum(ad[i + H] - ad[i], 1e-12)


def strategies(df, i):
    """R(범위 페이드: 24h 범위 위 20% 숏·아래 20% 롱) · T(추세 추종: 직전 24h 방향) -- 방향 부호, 0 = 쉼."""
    o, c, h, l = (df[x].to_numpy() for x in ("o", "c", "h", "l"))
    hi = pd.Series(h).rolling(288).max().to_numpy()[i]
    lo = pd.Series(l).rolling(288).min().to_numpy()[i]
    pos = (c[i] - lo) / np.maximum(hi - lo, 1e-12)
    r_side = np.where(pos >= 0.8, -1, np.where(pos <= 0.2, 1, 0))
    t_side = np.sign(c[i] - c[i - 288]).astype(int)
    ret = (c[i + H] / o[i + 1] - 1) * 1e4
    return r_side, t_side, ret


def net(side, ret, cost=True):
    return np.where(side != 0, side * ret - (COST if cost else 0.0), 0.0)


def day_ci(day, x, rng, lo=2.5, hi=97.5):
    """일 블록 부트스트랩: 결정당 평균의 CI."""
    g = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(g), (NBOOT, len(g)))
    return np.percentile(g[idx, 0].sum(1) / g[idx, 1].sum(1), [lo, hi])


def window(ts, a, b):
    return (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b))


def s0():
    rng = np.random.default_rng(SEED)
    rows = []
    for coin in COINS:
        df = load(coin)
        i = decisions(df)
        r_side, t_side, ret = strategies(df, i)
        rows.append(pd.DataFrame({"coin": coin, "ts": df.index[i + 1], "er": er_fwd(df.c.to_numpy(), i), "r": net(r_side, ret),
                                  "t": net(t_side, ret), "rg": net(r_side, ret, False), "tg": net(t_side, ret, False)}))
    d = pd.concat(rows, ignore_index=True)
    tr = d[window(d.ts, *SPLITS["학습"])]
    med = tr.er.median()
    d["day"] = d.ts.dt.date
    out = {"er_median_train": med}
    for w, (a, b) in SPLITS.items():
        x = d[window(d.ts, a, b)].reset_index(drop=True)
        for tag, rc, tc in (("net", "r", "t"), ("gross", "rg", "tg")):
            gate = np.where(x.er < med, x[rc], x[tc])                 # 오라클: 미래 횡보 → R · 추세 → T
            fixed = {"R": x[rc].mean(), "T": x[tc].mean()}
            best = max(fixed, key=fixed.get)
            bcol = rc if best == "R" else tc
            gain = gate - x[bcol].to_numpy()
            res = dict(n=len(x), gate=float(gate.mean()), R=fixed["R"], T=fixed["T"], best=best, gain=float(gain.mean()))
            if w == "전체" and tag == "net":
                res["gain_ci"] = day_ci(x.day, gain, rng).tolist()
                null = np.empty(NPLA)                                   # 위약: 코인별 오라클 라벨 순환이동(≥7일)
                for k in range(NPLA):
                    er = np.concatenate([np.roll(g.er.to_numpy(), int(rng.integers(168, len(g) - 168))) for _, g in x.groupby("coin", sort=False)])
                    null[k] = (np.where(er < med, x[rc], x[tc]) - x[bcol]).mean()
                res["placebo_pct"] = float((null < gain.mean()).mean() * 100)
                res["placebo_mean"] = float(null.mean())
            out[f"{w}|{tag}"] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "s0.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)


# ---- S0 v2: TP/SL 이 있는 전략으로 오라클 재측정 (사전등록 docs/prereg/regime_f_v2_20261102.md) ----
ATR_N, SL_ATR, TP_BRK, SL_BRK = 48, 0.5, 3.0, 1.5
COST_IN, COST_TP, COST_SL = 4.0, 2.0, 4.0


def barrier(df, i, side, tp, sl):
    """진입 i+1 시가 · TP/SL 은 i+1 봉부터 intrabar · 같은 봉 둘 다면 SL 먼저 · 시간청산 i+H 종가. 반환 (순bp, 총bp), side==0·무효는 0."""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    h, l, c, o = (df[x].to_numpy() for x in ("h", "l", "c", "o"))
    HW, LW = sw(h, H)[i + 1], sw(l, H)[i + 1]
    e = o[i + 1]
    long_ = side > 0
    hit_tp = np.where(long_[:, None], HW >= tp[:, None], LW <= tp[:, None])
    hit_sl = np.where(long_[:, None], LW <= sl[:, None], HW >= sl[:, None])
    big = H + 1
    a = np.where(hit_tp.any(1), hit_tp.argmax(1), big)
    b = np.where(hit_sl.any(1), hit_sl.argmax(1), big)
    kind = np.where(b <= a, np.where(b == big, 2, 1), 0)             # 0 TP · 1 SL · 2 시간
    px = np.where(kind == 0, tp, np.where(kind == 1, sl, c[i + H]))
    gross = side * (px / e - 1) * 1e4
    cost = COST_IN + np.where(kind == 0, COST_TP, COST_SL)
    valid = (side != 0) & np.where(long_, (sl < e) & (tp > e), (sl > e) & (tp < e))
    return np.where(valid, gross - cost, 0.0), np.where(valid, gross, 0.0)


def strategies2(df, i):
    o, c, h, l = (df[x].to_numpy() for x in ("o", "c", "h", "l"))
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(ATR_N).mean().to_numpy()[i]
    hi = pd.Series(h).rolling(288).max().to_numpy()[i]
    lo = pd.Series(l).rolling(288).min().to_numpy()[i]
    pos = (c[i] - lo) / np.maximum(hi - lo, 1e-12)
    e = o[i + 1]
    rs = np.where(pos >= 0.8, -1, np.where(pos <= 0.2, 1, 0))
    mid = (hi + lo) / 2
    rn = barrier(df, i, rs, mid, np.where(rs < 0, hi + SL_ATR * atr, lo - SL_ATR * atr))
    hp = pd.Series(h).rolling(288).max().shift(1).to_numpy()[i]
    lp = pd.Series(l).rolling(288).min().shift(1).to_numpy()[i]
    ts = np.where(c[i] > hp, 1, np.where(c[i] < lp, -1, 0))
    tn = barrier(df, i, ts, e + ts * TP_BRK * atr, e - ts * SL_BRK * atr)
    return rn, tn


def s0v2():
    rng = np.random.default_rng(SEED)
    rows = []
    for coin in COINS:
        df = load(coin)
        i = decisions(df)
        (rn, rg), (tn, tg) = strategies2(df, i)
        rows.append(pd.DataFrame({"coin": coin, "ts": df.index[i + 1], "er": er_fwd(df.c.to_numpy(), i), "r": rn, "t": tn, "rg": rg, "tg": tg}))
    d = pd.concat(rows, ignore_index=True)
    med = d[window(d.ts, *SPLITS["학습"])].er.median()
    d["day"] = d.ts.dt.date
    out = {"er_median_train": med}
    for w, (a, b) in SPLITS.items():
        x = d[window(d.ts, a, b)].reset_index(drop=True)
        for tag, rc, tc in (("net", "r", "t"), ("gross", "rg", "tg")):
            rng_ = x.er < med
            sw_ = np.where(rng_, x[rc], x[tc])                          # 오라클 전환: 횡보 → R2 · 추세 → T2
            ro_ = np.where(rng_, x[rc], 0.0)                            # 보조: 횡보에서만 R2
            fixed = {"R": x[rc].mean(), "T": x[tc].mean()}
            best = max(fixed, key=fixed.get)
            bcol = rc if best == "R" else tc
            gain = sw_ - x[bcol].to_numpy()
            res = dict(n=len(x), switch=float(sw_.mean()), range_only=float(ro_.mean()), R=fixed["R"], T=fixed["T"], best=best, gain=float(gain.mean()),
                       act_R=float((x[rc] != 0).mean()), act_T=float((x[tc] != 0).mean()))
            if w == "전체" and tag == "net":
                res["gain_ci975"] = day_ci(x.day, gain, rng, 1.25, 98.75).tolist()
                null = np.empty(NPLA)
                for k in range(NPLA):
                    er = np.concatenate([np.roll(g.er.to_numpy(), int(rng.integers(168, len(g) - 168))) for _, g in x.groupby("coin", sort=False)])
                    null[k] = (np.where(er < med, x[rc], x[tc]) - x[bcol]).mean()
                res["placebo_pct"] = float((null < gain.mean()).mean() * 100)
                res["placebo_mean"] = float(null.mean())
            out[f"{w}|{tag}"] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "s0v2.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    {"s0": s0, "s0v2": s0v2}[sys.argv[1]]()
