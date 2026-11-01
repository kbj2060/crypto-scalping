#!/usr/bin/env python3
"""횡보/추세 분별 정확도 (2026-11-02). 라벨 A = 앞으로 H 동안 가장 가까운 지지~저항 띠를 종가로 이탈(=추세) vs 머묾(=횡보).
사전등록 docs/prereg/regime_acc_20261102.md.   python scripts/research_regime_acc_20261102.py build | eval
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "research_regime_f_20261102.py")
rf = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rf)
OUT = ROOT / "tmp" / "regime_acc_20261102"
HS = (12, 48, 144)                      # 1h · 4h · 12h
SPL = rf.SPLITS


def frame_for(df, i, btc, snaps):
    """결정 인덱스 i 의 피쳐 프레임(학습·서빙 공용). snaps = {경계 시각(초): (가격, 지지[], 저항[])}."""
    c = df.c.to_numpy()
    ts = df.index.values.astype("datetime64[s]").astype("int64")
    R1, S1, nR, nS, R2, S2 = (np.full(len(i), np.nan) for _ in range(6))
    for k, ix in enumerate(i):
        sn = snaps.get(ts[ix] + 300)
        if sn is None:
            continue
        _, sup, res = sn
        nR[k], nS[k] = len(res), len(sup)
        if res:
            R1[k] = min(res)
            R2[k] = max(res)
        if sup:
            S1[k] = max(sup)
            S2[k] = min(sup)
    X = rf.features(df, i, btc)
    p = c[i]
    sig = np.sqrt(pd.Series(np.diff(np.log(c), prepend=np.nan) ** 2).rolling(48).sum().to_numpy())[i]
    X["lw"] = np.log((R1 - S1) / p)
    X["lw_sig"] = np.log((R1 - S1) / p / sig)
    X["dR"], X["dS"] = np.log(R1 / p) - np.log(sig), np.log(p / S1) - np.log(sig)
    X["bpos"] = (p - S1) / (R1 - S1)
    X["nR"], X["nS"] = nR, nS
    X["wide_R"], X["wide_S"] = np.log(R2 / p) - np.log(sig), np.log(p / S2) - np.log(sig)
    hh, ll = df.h.to_numpy(), df.l.to_numpy()
    for k in (12, 48, 144):                                  # 과거 k봉이 현재 띠 바깥/가장자리에 얼마나 닿았나
        Wc, Wh, Wl = sw(c, k + 1)[i - k], sw(hh, k + 1)[i - k], sw(ll, k + 1)[i - k]
        X[f"ex_up{k}"], X[f"ex_dn{k}"] = np.log(Wc.max(1) / R1), np.log(S1 / Wc.min(1))
        X[f"touchR{k}"], X[f"touchS{k}"] = (Wh >= R1[:, None] * 0.999).mean(1), (Wl <= S1[:, None] * 1.001).mean(1)
    X["rng_band"] = np.log((pd.Series(hh).rolling(288).max().to_numpy()[i] - pd.Series(ll).rolling(288).min().to_numpy()[i]) / (R1 - S1))
    return X, R1, S1, sig, p


def build():
    b = rf.load("BTC")
    bc = b.c
    btc = pd.DataFrame({"er48": (bc - bc.shift(48)).abs() / bc.diff().abs().rolling(48).sum(),
                        "rv": np.log(np.sqrt((np.log(bc).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(bc).diff() ** 2).rolling(288).sum()))})
    parts = []
    for coin in rf.COINS:
        df = rf.load(coin)
        i = rf.decisions(df)
        c = df.c.to_numpy()
        snaps = pickle.load(open(ROOT / "tmp" / "levels_4coin_20261102" / f"levels_{coin}.pkl", "rb"))
        ts = df.index.values.astype("datetime64[s]").astype("int64")
        X, R1, S1, sig, p = frame_for(df, i, btc, snaps)
        for H in HS:
            ok = i + H < len(c)
            W = np.full((len(i), H), np.nan)
            W[ok] = sw(c, H)[i[ok] + 1]
            X[f"yA{H}"] = np.where(ok, ((W > R1[:, None]) | (W < S1[:, None])).any(1), np.nan)
            X[f"yB{H}"] = np.where(ok, (np.abs(W / p[:, None] - 1) > sig[:, None]).any(1), np.nan)      # 보조 B: ±1σ4h 이탈
        X["coin"], X["ts"] = coin, df.index[i + 1]
        parts.append(X)
    d = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "data.parquet")
    for H in HS:
        for w, (a, bb) in SPL.items():
            x = d[rf.window(d.ts, a, bb)]
            print(f"H{H} {w}: n={len(x)} A추세율={x[f'yA{H}'].mean():.3f} B이탈율={x[f'yB{H}'].mean():.3f} 레벨결측={x.bpos.isna().mean():.3f}")


GEO = ["lw", "lw_sig", "dR", "dS", "bpos", "nR", "nS", "wide_R", "wide_S", "rng_band"]
CROSS = ["btc_er48", "btc_rv"]
HIST = [c for c in ("ex_up12", "ex_dn12", "ex_up48", "ex_dn48", "ex_up144", "ex_dn144", "touchR12", "touchS12", "touchR48", "touchS48", "touchR144", "touchS144", "rng_band")]
META = {"coin", "ts"} | {f"y{t}{H}" for t in "AB" for H in HS}


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    rng = np.random.default_rng(rf.SEED)
    d = pd.read_parquet(OUT / "data.parquet")
    ALL = [c for c in d.columns if c not in META]
    d["day"] = d.ts.dt.date
    purge = pd.Timedelta(minutes=5 * 48)

    def split(y):
        x = d.dropna(subset=[y])
        return (x[x.ts < pd.Timestamp(rf.TRAIN_END) - purge], x[rf.window(x.ts, rf.TRAIN_END, rf.VAL_END)], x[x.ts >= pd.Timestamp(rf.VAL_END)],
                x[x.ts < pd.Timestamp(rf.VAL_END) - purge])

    def mk(kind, depth=4, lr=0.05, it=300):
        if kind == "hgb":
            return HGB(max_depth=depth, learning_rate=lr, max_iter=it, l2_regularization=1.0, min_samples_leaf=200, early_stopping=False, random_state=0)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))

    cands = {"geo_logit": ("logit", GEO), "all_logit": ("logit", ALL)}
    for depth, lr, it in ((3, 0.05, 300), (5, 0.05, 300), (3, 0.03, 600), (5, 0.03, 600)):
        for nm, cols in (("ALL", ALL), ("noCross", [c for c in ALL if c not in CROSS]), ("noHist", [c for c in ALL if c not in HIST])):
            cands[f"hgb_d{depth}_lr{lr}_{nm}"] = ("hgb", cols, dict(depth=depth, lr=lr, it=it))

    def fit_pred(spec, tr, tests):
        kind, cols, *kw = spec
        m = mk(kind, **(kw[0] if kw else {}))
        m.fit(tr[cols], tr.yA48.astype(int))
        return [m.predict_proba(t[cols])[:, 1] for t in tests], m

    def metrics(y, p, base):
        yy = y.to_numpy().astype(int)
        pred = (p >= 0.5).astype(int)
        acc = float((pred == yy).mean())
        bal = float((np.mean(pred[yy == 1] == 1) + np.mean(pred[yy == 0] == 0)) / 2)
        return dict(acc=acc, bal_acc=bal, auc=float(roc_auc_score(yy, p)), brier_skill=float(1 - ((p - yy) ** 2).mean() / ((base - yy) ** 2).mean()))

    tr, va, te, trva = split("yA48")
    base_va = tr.yA48.mean()
    out = {"n": dict(train=len(tr), val=len(va), test=len(te)), "base_rate": dict(train=float(tr.yA48.mean()), val=float(va.yA48.mean()), test=float(te.yA48.mean()))}
    out["majority_acc"] = dict(val=float(max(va.yA48.mean(), 1 - va.yA48.mean()) if False else (va.yA48 == round(tr.yA48.mean())).mean()),
                               test=float((te.yA48 == round(trva.yA48.mean())).mean()))
    val_res, val_p = {}, {}
    for name, spec in cands.items():
        (pv,), _ = fit_pred(spec, tr, [va])
        val_res[name] = metrics(va.yA48, pv, base_va)
        val_p[name] = pv
    top = sorted(val_res, key=lambda n: -val_res[n]["auc"])[:3]
    ens = np.mean([val_p[n] for n in top if n.startswith("hgb")][:3] or [val_p[top[0]]], 0)
    val_res["ensemble_top3hgb"] = metrics(va.yA48, ens, base_va)
    out["val"] = val_res
    best = max(val_res, key=lambda n: (val_res[n]["acc"], val_res[n]["auc"]))
    out["selected"] = best
    out["selected_top3hgb"] = [n for n in top if n.startswith("hgb")][:3]
    # 시험 1회: 선택 구성을 학습+검증(purge)으로 재적합
    if best == "ensemble_top3hgb":
        specs = [cands[n] for n in out["selected_top3hgb"]]
    else:
        specs = [cands[best]]
    pt = np.mean([fit_pred(sp_, trva, [te])[0][0] for sp_ in specs], 0)
    pg = fit_pred(cands["geo_logit"], trva, [te])[0][0]
    base_te = trva.yA48.mean()
    out["test"] = {"selected": metrics(te.yA48, pt, base_te), "geo_logit_baseline": metrics(te.yA48, pg, base_te)}
    yy = te.yA48.to_numpy().astype(int)
    for nm, d_ in (("acc_gain_vs_geo", ((pt >= .5) == yy).astype(float) - ((pg >= .5) == yy).astype(float)), ("acc_gain_vs_majority", ((pt >= .5) == yy).astype(float) - (yy == round(base_te)).astype(float))):
        out["test"][nm] = dict(mean=float(d_.mean()), ci95=rf.day_ci(te.day.to_numpy(), d_, rng).tolist())
    out["test"]["by_coin_acc"] = {c: float(((pt >= .5) == yy)[(te.coin == c).to_numpy()].mean()) for c in rf.COINS}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "eval.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("n", "base_rate", "majority_acc", "selected", "selected_top3hgb")}, indent=1))
    print("== 검증 상위:"); [print(n, {a: round(b, 4) for a, b in val_res[n].items()}) for n in sorted(val_res, key=lambda n: -val_res[n]["acc"])[:6]]
    print("== 시험 (1회):", json.dumps(out["test"], indent=1))


def final():
    """배포용 최종 모델: 선택 구성(HGB depth3 lr.03 600회, 레벨 과거이력 제외)을 라벨이 있는 전 행으로 적합해 저장(재현 가능)."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    d = pd.read_parquet(OUT / "data.parquet")
    ALL = [c for c in d.columns if c not in META]
    cols = [c for c in ALL if c not in HIST]
    x = d.dropna(subset=["yA48"])
    m = HGB(max_depth=3, learning_rate=0.03, max_iter=600, l2_regularization=1.0, min_samples_leaf=200, early_stopping=False, random_state=0).fit(x[cols], x.yA48.astype(int))
    joblib.dump({"model": m, "features": cols, "label": "yA48: 앞으로 4h 종가가 가장 가까운 지지~저항 띠를 이탈하면 1(추세) 아니면 0(횡보)", "threshold": 0.5,
                 "trained_rows": len(x), "trained_until": str(x.ts.max())}, OUT / "regime_band_h48_model.joblib")
    print("saved", len(cols), "features", len(x), "rows")


if __name__ == "__main__":
    {"build": build, "eval": evaluate, "final": final}[sys.argv[1]]()
