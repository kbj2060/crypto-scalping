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


def s0v3():
    """v3: 학습에서 ER 3분위별 최선 정책(R2/T2/쉼)을 적합 → 검증·시험에서 오라클 라벨로 평가 (사전등록 regime_f_v3_20261102.md)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for coin in COINS:
        df = load(coin)
        i = decisions(df)
        (rn, rg), (tn, tg) = strategies2(df, i)
        rows.append(pd.DataFrame({"coin": coin, "ts": df.index[i + 1], "er": er_fwd(df.c.to_numpy(), i), "r": rn, "t": tn, "rg": rg, "tg": tg}))
    d = pd.concat(rows, ignore_index=True)
    d["day"] = d.ts.dt.date
    tr = d[window(d.ts, *SPLITS["학습"])]
    q1, q2 = tr.er.quantile([1 / 3, 2 / 3])
    d["cls"] = np.where(d.er <= q1, 0, np.where(d.er <= q2, 1, 2))
    out = {"er_terciles": [q1, q2]}
    for tag, rc, tc in (("net", "r", "t"), ("gross", "rg", "tg")):
        pol = {}
        for k in range(3):
            g = d[(d.cls == k) & window(d.ts, *SPLITS["학습"])]
            cand = {"R": g[rc].mean(), "T": g[tc].mean(), "flat": 0.0}
            pol[k] = max(cand, key=cand.get)
        out[f"policy_{tag}"] = pol
        val = lambda x, p: np.select([x.cls.map(p) == "R", x.cls.map(p) == "T"], [x[rc], x[tc]], 0.0)   # noqa: E731
        for w in ("학습", "검증", "시험"):
            x = d[window(d.ts, *SPLITS[w])].reset_index(drop=True)
            g = val(x, pol)
            fixed = {"R": x[rc].mean(), "T": x[tc].mean(), "flat": 0.0}
            best = max(fixed, key=fixed.get)
            base = {"R": x[rc].to_numpy(), "T": x[tc].to_numpy(), "flat": np.zeros(len(x))}[best]
            res = dict(n=len(x), policy=float(g.mean()), best=best, best_val=fixed[best], gain=float((g - base).mean()),
                       by_class={k: dict(R=float(x[x.cls == k][rc].mean()), T=float(x[x.cls == k][tc].mean()), n=int((x.cls == k).sum())) for k in range(3)})
            if tag == "net" and w != "학습":
                res["gain_ci983"] = day_ci(x.day, g - base, rng, 0.85, 99.15).tolist()
                null = np.empty(NPLA)
                for k in range(NPLA):
                    c2 = np.concatenate([np.roll(gg.cls.to_numpy(), int(rng.integers(168, len(gg) - 168))) for _, gg in x.groupby("coin", sort=False)])
                    xx = x.assign(cls=c2)
                    null[k] = (val(xx, pol) - base).mean()
                res["placebo_pct"] = float((null < (g - base).mean()).mean() * 100)
            out[f"{w}|{tag}"] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "s0v3.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)


# ---- 진단 S2d: 큰 움직임(ER 상위 1/3) 예측 가능성 + R2 정책 쓰임 (보고 전용, 승격 근거 아님; regime_f_diag_20261102.md) ----
def features(df, i, btc):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    c, h, l, qv, tb = (df[x] for x in ("c", "h", "l", "qv", "tbqv"))
    lc = np.log(c)
    r1 = lc.diff()
    rv = {w: np.sqrt((r1 ** 2).rolling(w).sum()) for w in (12, 48, 288, 2016)}
    ad = c.diff().abs()
    er = {w: (c - c.shift(w)).abs() / ad.rolling(w).sum() for w in (48, 288)}
    vr = (lc.diff(12).rolling(288).var()) / (12 * r1.rolling(288).var())
    hi, lo = h.rolling(288).max(), l.rolling(288).min()
    s12 = qv.rolling(12).sum()
    X = pd.DataFrame({
        "rv12_288": np.log(rv[12] / rv[288]), "rv48_288": np.log(rv[48] / rv[288]), "rv288_2016": np.log(rv[288] / rv[2016]), "lrv288": np.log(rv[288]),
        "er48": er[48], "er288": er[288], "vr": vr, "pos": (c - lo) / (hi - lo), "rng_rv": np.log((hi - lo) / c) - np.log(rv[288]),
        "qv_ratio": np.log(s12 / s12.rolling(2016).mean()), "tk12": (2 * tb - qv).rolling(12).sum() / s12, "tk48": (2 * tb - qv).rolling(48).sum() / qv.rolling(48).sum(),
    }).iloc[i].reset_index(drop=True)
    hh, ll = sw(h.to_numpy(), 288)[i - 287], sw(l.to_numpy(), 288)[i - 287]
    X["since_hi"], X["since_lo"] = 287 - hh.argmax(1), 287 - ll.argmin(1)
    hr = (df.index[i] + pd.Timedelta(minutes=5)).hour.values
    X["hs"], X["hc"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    X["btc_er48"], X["btc_rv"] = btc["er48"].iloc[i].values, btc["rv"].iloc[i].values
    return X


def s2d():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    b = load("BTC")
    bc = b.c
    btc = pd.DataFrame({"er48": (bc - bc.shift(48)).abs() / bc.diff().abs().rolling(48).sum(),
                        "rv": np.log(np.sqrt((np.log(bc).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(bc).diff() ** 2).rolling(288).sum()))})
    parts = []
    for coin in COINS:
        df = load(coin)
        i = decisions(df)
        (rn, rg), _ = strategies2(df, i)
        X = features(df, i, btc)
        X["coin"], X["ts"], X["er"], X["r"], X["rg"] = coin, df.index[i + 1], er_fwd(df.c.to_numpy(), i), rn, rg
        parts.append(X)
    d = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    d["day"] = d.ts.dt.date
    FE = [c for c in d.columns if c not in ("coin", "ts", "er", "r", "rg", "day")]
    GEO = ["pos", "rng_rv", "since_hi", "since_lo"]
    VOL = ["rv12_288", "rv48_288", "rv288_2016", "lrv288"]
    PERS = ["er48", "er288"]
    tr = d[window(d.ts, *SPLITS["학습"])]
    purge = pd.Timestamp(TRAIN_END) - pd.Timedelta(minutes=5 * H)
    tr = tr[tr.ts < purge].dropna(subset=FE)
    q2 = tr.er.quantile(2 / 3)
    d["y"] = (d.er >= q2).astype(int)
    tr = tr.assign(y=(tr.er >= q2).astype(int))
    mods = {"M_HGB": (HGB(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, min_samples_leaf=200, early_stopping=False, random_state=0), FE),
            "B1_pers": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), PERS),
            "B2_vol": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), PERS + VOL if False else VOL),
            "B3_geo": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), VOL + GEO)}
    out = {"q2_train": q2, "base_rate_train": float(tr.y.mean())}
    pol_r = {}
    for name, (m, cols) in mods.items():
        m.fit(tr[cols], tr.y)
        pol_r[name] = (m, cols)
    for w in ("검증", "시험"):
        x = d[window(d.ts, *SPLITS[w])].dropna(subset=FE).reset_index(drop=True)
        res = {"n": len(x), "base_rate": float(x.y.mean())}
        P = {n: m.predict_proba(x[cols])[:, 1] for n, (m, cols) in pol_r.items()}
        for n, p in P.items():
            res[f"auc_{n}"] = float(roc_auc_score(x.y, p))
            res[f"brier_{n}"] = float(((p - x.y) ** 2).mean())
        res["brier_base"] = float(((x.y.mean() - x.y) ** 2).mean())
        bl = min(("B1_pers", "B2_vol", "B3_geo"), key=lambda n: res[f"brier_{n}"])
        dq = ((P["M_HGB"] - x.y) ** 2 - (P[bl] - x.y) ** 2).to_numpy()
        res["best_baseline"], res["dBrier_M_vs_best"] = bl, float(dq.mean())
        res["dBrier_ci90_upper"] = float(day_ci(x.day, dq, rng, 0, 90)[1])
        for thr in (0.40, 0.50):
            for n in ("M_HGB", "B2_vol", "B3_geo"):
                act = P[n] >= thr
                res[f"R2_net_{n}_p{int(thr * 100)}"] = float(np.where(act, x.r, 0.0).mean())
                res[f"R2_act_{n}_p{int(thr * 100)}"] = float(act.mean())
        res["R2_net_always"] = float(x.r.mean())
        res["R2_net_oracle_high"] = float(np.where(x.y == 1, x.r, 0.0).mean())
        act = P["M_HGB"] >= 0.40
        gain = np.where(act, x.r, 0.0)
        res["gain_M_p40_ci95"] = day_ci(x.day, gain, rng).tolist()
        null = np.array([np.where(rng.random(len(x)) < act.mean(), x.r, 0.0).mean() for _ in range(NPLA)])
        res["gain_M_p40_vs_random_pct"] = float((null < gain.mean()).mean() * 100)
        out[w] = res
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "s2d.json", "w"), indent=1, default=float)
    for k, v in out.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    {"s0": s0, "s0v2": s0v2, "s0v3": s0v3, "s2d": s2d}[sys.argv[1]]()
