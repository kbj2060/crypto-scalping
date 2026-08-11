"""Per-feature analysis against the ETH zig075 oracle (zigzag pivot) direction label, covering
the full ETH candidate feature panel plus the new regime3-jmredesign columns (2026-08-11,
Step B of the zig075 jmredesign component-tuning session).

For every candidate feature, compute the univariate Mann-Whitney AUC for two oracle tasks:
  - direction:    zigzag LONG vs SHORT (among non-CASH bars)
  - tradeability: non-CASH vs CASH
on TRAIN/VAL/OOS_CHECK windows plus three TRAIN sub-windows for sign-stability. A feature
"passes" iff its direction-task edge (|AUC-0.5|) is sign-stable across all 3 train sub-windows,
agrees in sign between TRAIN and VAL, and clears a minimum edge of MIN_EDGE in both TRAIN and VAL.

Then, using only the honestly (TRAIN-only) selected passing features, retrain a 3-class LGBM
gate and report VAL/OOS_CHECK accuracy as a sanity check (methodology mirrors
scripts/audit_sol_oracle_feature_analysis_20260808.py).

Note: the zigzag_action label's 2026 coverage currently ends 2026-02-28 (not the full 2026-03-31
OOS boundary), so OOS_CHECK below is a partial-OOS diagnostic window, not the full fresh-forward
OOS split.

Outputs: tmp/eth_zig075_oracle_label_check_20260811/oracle_feature_analysis.json (+ printed summary).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]

TECH_PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
REGIME3_PATHS = [
    ROOT / f"data/ensemble/supervised/eth_regime3_current_hmm_jmredesign_20260810_{yr}_maskedname.csv"
    for yr in ("2024", "2025", "2026")
]
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
LABEL_PATHS = [LABEL_DIR / f"zigzag_action_labels_{yr}.csv" for yr in ("2024", "2025", "2026")]

OUT_PATH = ROOT / "tmp/eth_zig075_oracle_label_check_20260811/oracle_feature_analysis.json"

RAW_LEVEL_COLS = {
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "close_btc", "volume_btc", "quote_volume_btc",
}
DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-06-30")
VAL_START, VAL_END = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28")
MIN_EDGE = 0.02
SEED = 20260811


def _forbidden(name: str) -> bool:
    low = name.lower()
    return name.startswith(DENY_PREFIXES) or any(tok in low for tok in DENY_TOKENS)


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
    tech = pd.read_csv(TECH_PANEL_PATH, low_memory=False)
    tech["timestamp"] = pd.to_datetime(tech["timestamp"])

    regime3 = pd.concat([pd.read_csv(p) for p in REGIME3_PATHS], ignore_index=True)
    regime3["timestamp"] = pd.to_datetime(regime3["timestamp"])

    labels = pd.concat(
        [pd.read_csv(p, usecols=["timestamp", "zigzag_action"]) for p in LABEL_PATHS], ignore_index=True
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])

    df = labels.merge(tech, on="timestamp", how="inner").merge(regime3, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    feat_cols = [c for c in tech.columns if c != "timestamp" and c not in RAW_LEVEL_COLS and not _forbidden(c)]
    feat_cols += [c for c in regime3.columns if c != "timestamp" and not _forbidden(c)]

    action = df["zigzag_action"].to_numpy()
    x = df[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    ts = df["timestamp"]

    train_mask = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    tr_idx = np.flatnonzero(train_mask)
    sub_bounds = np.array_split(tr_idx, 3)
    windows = {
        "train": tr_idx, "val": np.flatnonzero(val_mask), "oos_check": np.flatnonzero(oos_mask),
        "tr_sub1": sub_bounds[0], "tr_sub2": sub_bounds[1], "tr_sub3": sub_bounds[2],
    }

    print(f"rows: total={len(df)} train={len(tr_idx)} val={int(val_mask.sum())} oos_check={int(oos_mask.sum())}",
          flush=True)
    print(f"n_candidate_features={len(feat_cols)}", flush=True)

    rows = []
    for f_i, name in enumerate(feat_cols):
        rec = {"feature": name}
        for wname, idx in windows.items():
            a = action[idx]
            xv = x[idx, f_i]
            nz = a != 0
            rec[f"dir_auc_{wname}"] = auc_binary(xv[nz], (a[nz] == 1).astype(int))
            rec[f"trade_auc_{wname}"] = auc_binary(xv, nz.astype(int))
        rows.append(rec)
    df_feat = pd.DataFrame(rows)

    df_feat["dir_edge_train"] = (df_feat["dir_auc_train"] - 0.5).abs()
    df_feat["dir_edge_val"] = (df_feat["dir_auc_val"] - 0.5).abs()
    df_feat = df_feat.sort_values("dir_edge_train", ascending=False).reset_index(drop=True)

    sign_tr = np.sign(df_feat["dir_auc_train"] - 0.5)
    sign_val = np.sign(df_feat["dir_auc_val"] - 0.5)
    sign_oos = np.sign(df_feat["dir_auc_oos_check"] - 0.5)
    sub_signs = np.stack([np.sign(df_feat[f"dir_auc_tr_sub{k}"] - 0.5) for k in (1, 2, 3)], axis=1)
    sign_stable_train = (sub_signs == sign_tr.to_numpy()[:, None]).all(axis=1)
    train_val_agree = (sign_tr == sign_val).to_numpy()
    train_oos_agree = (sign_tr == sign_oos).to_numpy()

    passed = (
        sign_stable_train
        & train_val_agree
        & (df_feat["dir_edge_train"] >= MIN_EDGE).to_numpy()
        & (df_feat["dir_edge_val"] >= MIN_EDGE).to_numpy()
    )
    df_feat["sign_stable_train"] = sign_stable_train
    df_feat["train_val_agree"] = train_val_agree
    df_feat["train_oos_agree"] = train_oos_agree
    df_feat["passed"] = passed

    rho_all = float(spearmanr(df_feat["dir_auc_train"], df_feat["dir_auc_val"], nan_policy="omit").statistic)

    summary = {
        "n_features_tested": len(df_feat),
        "n_passed": int(passed.sum()),
        "pass_criterion": (
            f"sign_stable_across_3_train_subwindows AND sign(train)==sign(val) "
            f"AND |auc-0.5|>={MIN_EDGE} in train AND val"
        ),
        "spearman_dir_auc_train_vs_val": rho_all,
        "n_also_oos_check_sign_agree_among_passed": int((passed & train_oos_agree).sum()),
        "passed_features": df_feat.loc[passed, "feature"].tolist(),
        "top15_all": df_feat.head(15)[
            ["feature", "dir_auc_train", "dir_auc_val", "dir_auc_oos_check", "passed"]
        ].round(4).to_dict("records"),
    }
    print(json.dumps(summary, indent=2), flush=True)

    passed_features = df_feat.loc[passed, "feature"].tolist()
    filtered: dict = {}
    if len(passed_features) >= 2:
        sel_idx = [feat_cols.index(f) for f in passed_features]
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=300, learning_rate=0.05,
            num_leaves=63, min_child_samples=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
            random_state=SEED, n_jobs=4, verbosity=-1,
        )
        x_tr, y_tr = x[train_mask][:, sel_idx], action[train_mask]
        finite_tr = np.isfinite(x_tr).all(axis=1)
        clf.fit(x_tr[finite_tr], y_tr[finite_tr])
        for wname, mask in (("val", val_mask), ("oos_check", oos_mask)):
            x_w, y_w = x[mask][:, sel_idx], action[mask]
            finite_w = np.isfinite(x_w).all(axis=1)
            proba = clf.predict_proba(x_w[finite_w])
            filtered[f"{wname}_accuracy"] = float((proba.argmax(axis=1) == y_w[finite_w]).mean())
            filtered[f"{wname}_n_rows"] = int(finite_w.sum())
        filtered["passed_features_used"] = passed_features
    else:
        filtered["note"] = "fewer than 2 passed features -- skipped honest retrain"
    print(json.dumps(filtered, indent=2), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(
        {"summary": summary, "per_feature": df_feat.round(5).to_dict("records"), "filtered_model": filtered},
        indent=2,
    ))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
