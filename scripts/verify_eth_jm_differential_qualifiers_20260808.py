"""Decisive follow-up to the ETH regime-differential diagnostic (2026-08-08).

The main pass found that under the jm_lam32 gate ETH has 26 TRAIN-stable differential features
versus a permutation null of 12.2 (max 23) -- the first time in this project that the
"persistent differential features exist" test passed. Two things must be checked before that can
be called a finding:

  A. PERSISTENCE   do those 26 keep their delta sign on VAL and on OOS? Train-internal stability
                   is not out-of-sample persistence; BTC's whole failure was the gap between the
                   two. Compared against 200 matched random 26-feature draws.
  B. CONFOUND      the qualifiers are dominated by volatility features, and the JM detector is
                   FIT ON volatility (EWM downside deviation). A regime defined by volatility can
                   induce a mechanical differential in volatility features without any change in
                   the payoff relationship. Re-run selection on a gate that does NOT use
                   volatility (czz4, pure price-path) and on d2_rule, and report the qualifier
                   overlap plus the volatility share of each.

Descriptive diagnostic; OOS read as a persistence measurement only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from test_statistical_jump_model_regimes_20260808 import jm_features, fit_jm, causal_decode, causal_zigzag_regime  # noqa: E402
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, HORIZON_BARS, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
OUT_PATH = ROOT / "tmp/regime_feature_analysis_eth_20260808/qualifier_verification.json"
SEED, N_FOLDS, MIN_ABS_DELTA, N_RANDOM = 903174, 4, 0.01, 200
VOL_PAT = ("vol", "bb_width", "garman", "parkinson", "rogers", "atr", "amihud", "wick",
           "realized", "squeeze", "compression", "range")


def is_vol(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in VOL_PAT)


def auc_matrix(xf, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return None
    r = rankdata(xf, axis=0)
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def delta_on(xf, action, idx, regime):
    parts = []
    for r in (2, 0):
        sub = idx[regime[idx] == r]
        a = action[sub]
        nz = a != 0
        if nz.sum() < 200:
            return None
        m = auc_matrix(xf[sub][nz], (a[nz] == 1).astype(int))
        if m is None:
            return None
        parts.append(m)
    return parts[0] - parts[1]


def select_qualifiers(xf, action, tr_idx, regime):
    folds = np.array_split(tr_idx, N_FOLDS)
    deltas = []
    for fold in folds:
        d = delta_on(xf, action, fold, regime)
        if d is None:
            return np.array([], dtype=int), None
        deltas.append(d)
    D = np.stack(deltas, axis=1)
    same = (np.sign(D) == np.sign(D[:, [0]])).all(axis=1)
    big = np.abs(np.median(D, axis=1)) >= MIN_ABS_DELTA
    return np.flatnonzero(same & big), np.median(D, axis=1)


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    n = min(len(panel), len(labels))
    panel, labels = panel.iloc[:n].reset_index(drop=True), labels.iloc[:n]
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    ts = panel["timestamp"]
    close = panel["close"].to_numpy(dtype=np.float64)

    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    tr_idx, v_idx, o_idx = (np.flatnonzero(m) for m in (train_mask, val_mask, oos_mask))
    med = np.nanmedian(x[tr_idx], axis=0)
    xf = np.where(np.isfinite(x), x, med).astype(np.float64)

    # gates
    r288 = np.full(n, np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    d2 = np.full(n, 1, dtype=np.int8)
    d2[r288 > 0.04] = 2
    d2[r288 < -0.04] = 0
    xj = jm_features(close)
    valid = np.isfinite(xj).all(axis=1)
    z = np.zeros_like(xj)
    z[valid] = RobustScaler().fit(xj[train_mask & valid]).transform(xj[valid])
    mu = fit_jm(z[train_mask & valid], 3, 32.0, SEED)
    st = causal_decode(z[valid], mu, 32.0)
    jm = np.full(n, -1, dtype=int)
    jm[valid] = st
    lr288 = np.full(n, np.nan)
    lr288[288:] = np.log(close[288:] / close[:-288])
    means = [np.nanmean(lr288[train_mask & (jm == s)]) for s in range(3)]
    order = np.argsort(means)
    jm = np.array([{int(order[0]): 0, int(order[1]): 1, int(order[2]): 2}.get(s, 1) for s in jm], dtype=np.int8)
    cdir = causal_zigzag_regime(close, 0.04)
    czz = np.where(cdir > 0, 2, np.where(cdir < 0, 0, 1)).astype(np.int8)
    gates = {"jm_lam32": jm, "czz4": czz, "d2_rule": d2}

    rng = np.random.default_rng(SEED)
    report = {"asset": "ETHUSDT", "n_features": len(feat_cols)}

    for gname, regime in gates.items():
        qual, d_med_tr = select_qualifiers(xf, action, tr_idx, regime)
        d_v = delta_on(xf, action, v_idx, regime)
        d_o = delta_on(xf, action, o_idx, regime)
        if d_v is None or d_o is None or len(qual) == 0:
            report[gname] = {"n_qualifiers": int(len(qual)), "note": "insufficient data"}
            continue
        s_tr = np.sign(d_med_tr[qual])
        keep_v = float(np.mean(s_tr == np.sign(d_v[qual])))
        keep_o = float(np.mean(s_tr == np.sign(d_o[qual])))
        keep_both = float(np.mean((s_tr == np.sign(d_v[qual])) & (s_tr == np.sign(d_o[qual]))))
        rv, ro = [], []
        for _ in range(N_RANDOM):
            sel = rng.choice(len(feat_cols), size=len(qual), replace=False)
            ss = np.sign(d_med_tr[sel])
            rv.append(float(np.mean(ss == np.sign(d_v[sel]))))
            ro.append(float(np.mean(ss == np.sign(d_o[sel]))))
        vol_share = float(np.mean([is_vol(feat_cols[i]) for i in qual]))
        report[gname] = {
            "n_qualifiers": int(len(qual)),
            "qualifier_sign_kept_val": round(keep_v, 3),
            "qualifier_sign_kept_oos": round(keep_o, 3),
            "qualifier_sign_kept_both": round(keep_both, 3),
            "random_matched_baseline": {
                "val_mean": round(float(np.mean(rv)), 3), "val_p95": round(float(np.percentile(rv, 95)), 3),
                "oos_mean": round(float(np.mean(ro)), 3), "oos_p95": round(float(np.percentile(ro, 95)), 3)},
            "beats_random_oos": bool(keep_o > np.percentile(ro, 95)),
            "volatility_share_of_qualifiers": round(vol_share, 3),
            "volatility_share_of_all_features": round(float(np.mean([is_vol(c) for c in feat_cols])), 3),
            "qualifiers": [feat_cols[i] for i in qual],
        }
        print(json.dumps({k: report[gname][k] for k in
                          ("n_qualifiers", "qualifier_sign_kept_val", "qualifier_sign_kept_oos",
                           "beats_random_oos", "volatility_share_of_qualifiers")} | {"gate": gname}), flush=True)

    jm_q = set(report.get("jm_lam32", {}).get("qualifiers", []))
    czz_q = set(report.get("czz4", {}).get("qualifiers", []))
    report["jm_czz_qualifier_overlap"] = sorted(jm_q & czz_q)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({"jm_czz_overlap": report["jm_czz_qualifier_overlap"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
