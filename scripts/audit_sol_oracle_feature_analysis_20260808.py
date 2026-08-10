"""Post-hoc per-feature analysis against the SOL oracle (TB trade-outcome) label (2026-08-08).

For every model feature, compute the univariate Mann-Whitney AUC for two oracle tasks:
  - direction:    oracle LONG vs SHORT (among non-CASH bars)
  - tradeability: non-CASH vs CASH
separately on TRAIN and VAL, plus three train sub-windows for sign-stability. Then retrain the
cheap-gate LGBM on only the top-K train-selected direction features (selection uses TRAIN only --
honest) and replay VAL with the frozen parent rule set.

Outputs: tmp/sol_dl_rl_survey_20260807/oracle_feature_analysis.json (+ printed summary).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_eval_sol_tripbarrier_lgbm_cheapgate_20260807 import (  # noqa: E402
    RAW_LEVEL_COLS, PANEL_PATH, LABEL_PATH, SEED, HORIZON_BARS,
    TRAIN_END, VAL_START, VAL_END, replay, side_state_from_proba,
)

OUT_PATH = ROOT / "tmp/sol_dl_rl_survey_20260807/oracle_feature_analysis.json"
TOP_K = 20
THRESHOLDS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60]


def auc_binary(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC of feature x for binary y (1 vs 0), NaN-safe."""
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    labels = pd.read_parquet(LABEL_PATH)
    action = labels["trade_outcome_action"].to_numpy()
    tp_moves = labels["tp_move"].to_numpy(dtype=np.float64)
    sl_moves = labels["sl_move"].to_numpy(dtype=np.float64)
    feat_cols = [c for c in panel.columns if c != "timestamp" and c not in RAW_LEVEL_COLS]
    x = panel[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)

    ts = panel["timestamp"]
    train_mask = (ts <= TRAIN_END).to_numpy()
    purge_cut = np.flatnonzero(train_mask)[-HORIZON_BARS:]
    train_mask[purge_cut] = False
    train_mask &= np.isfinite(tp_moves)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()

    tr_idx = np.flatnonzero(train_mask)
    sub_bounds = np.array_split(tr_idx, 3)
    windows = {"train": tr_idx, "val": np.flatnonzero(val_mask),
               "tr_sub1": sub_bounds[0], "tr_sub2": sub_bounds[1], "tr_sub3": sub_bounds[2]}

    rows = []
    for f_i, name in enumerate(feat_cols):
        rec = {"feature": name}
        for wname, idx in windows.items():
            a = action[idx]
            xv = x[idx, f_i].astype(np.float64)
            nz = a != 0
            rec[f"dir_auc_{wname}"] = auc_binary(xv[nz], (a[nz] == 1).astype(int))
            rec[f"trade_auc_{wname}"] = auc_binary(xv, nz.astype(int))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["dir_edge_train"] = (df["dir_auc_train"] - 0.5).abs()
    df["dir_edge_val"] = (df["dir_auc_val"] - 0.5).abs()
    df = df.sort_values("dir_edge_train", ascending=False).reset_index(drop=True)

    sign_tr = np.sign(df["dir_auc_train"] - 0.5)
    sign_val = np.sign(df["dir_auc_val"] - 0.5)
    sub_signs = np.stack([np.sign(df[f"dir_auc_tr_sub{k}"] - 0.5) for k in (1, 2, 3)], axis=1)
    stable_in_train = (sub_signs == sign_tr.to_numpy()[:, None]).all(axis=1)
    rho_all = float(spearmanr(df["dir_auc_train"], df["dir_auc_val"], nan_policy="omit").statistic)
    top = df.head(TOP_K)
    sign_agree_top = float((np.sign(top["dir_auc_train"] - 0.5) == np.sign(top["dir_auc_val"] - 0.5)).mean())

    summary = {
        "n_features": len(df),
        "max_dir_auc_train_dev": float(df["dir_edge_train"].max()),
        "max_dir_auc_val_dev": float(df["dir_edge_val"].max()),
        "spearman_dir_auc_train_vs_val": rho_all,
        "topK_sign_agreement_train_to_val": sign_agree_top,
        "n_features_sign_stable_across_3_train_subwindows": int(stable_in_train.sum()),
        "n_topK_sign_stable_in_train_AND_val_agree": int(((sub_signs[:TOP_K] == sign_tr.to_numpy()[:TOP_K, None]).all(axis=1) & (sign_tr[:TOP_K].to_numpy() == sign_val[:TOP_K].to_numpy())).sum()),
        "top10_direction": top.head(10)[["feature", "dir_auc_train", "dir_auc_val", "dir_auc_tr_sub1", "dir_auc_tr_sub2", "dir_auc_tr_sub3"]].round(4).to_dict("records"),
        "top5_tradeability": df.sort_values("trade_auc_train", key=lambda s: (s - 0.5).abs(), ascending=False).head(5)[["feature", "trade_auc_train", "trade_auc_val"]].round(4).to_dict("records"),
    }
    print(json.dumps(summary, indent=2), flush=True)

    # honest top-K filtered retrain: selection on TRAIN only
    top_features = top["feature"].tolist()
    sel_idx = [feat_cols.index(f) for f in top_features]
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
                             num_leaves=63, min_child_samples=200, feature_fraction=0.8,
                             bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                             random_state=SEED, n_jobs=-1, verbosity=-1)
    clf.fit(x[train_mask][:, sel_idx], action[train_mask])
    proba_val = clf.predict_proba(x[val_mask][:, sel_idx])
    acc = float((proba_val.argmax(axis=1) == action[val_mask]).mean())
    filt_results = []
    for thr in THRESHOLDS:
        side_state = np.zeros(len(panel), dtype=np.int64)
        side_state[val_mask] = side_state_from_proba(proba_val, thr)
        r = replay(panel, side_state, tp_moves, sl_moves, val_mask)
        filt_results.append({"threshold": thr, **{k: r.get(k) for k in ("n_trades", "pnl_pct", "win_rate", "mdd_pct")}})
    filtered = {"top_features": top_features, "val_accuracy": acc, "val_replay": filt_results}
    print(json.dumps(filtered, indent=2), flush=True)

    OUT_PATH.write_text(json.dumps({"summary": summary, "per_feature": df.round(5).to_dict("records"), "filtered_model": filtered}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
