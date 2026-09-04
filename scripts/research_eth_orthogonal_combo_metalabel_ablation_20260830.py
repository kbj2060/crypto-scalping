#!/usr/bin/env python3
"""Ablation for orthogonal_combo v2-exclude-middle (HORIZON=24/GAP=12, K_lo=1.786/K_hi=3.571,
VAL/OOS/HOLDOUT AUC 0.723/0.716/0.708 -- the strongest result of any Homer signal so far):
permutation importance found atr_percentile_864 alone at +0.110, ~4.5x the #2 feature
(nyse_open_flag +0.025) and far more extreme than any prior signal's top feature (taker's own
atr_percentile_864 dominance was +0.035; that ablation found it was NOT primarily a regime proxy,
removing all 3 vol-regime features only cost ~0.01-0.012 AUC). Does the same hold here, or is this
signal's excellent AUC substantially just "are we in an unusual volatility regime"?

4 configs: full 23, no-vol-regime (atr_pct/atr_percentile_864/realized_vol_ratio/bb_width_pctile --
bb_width_pctile included here since it's also a volatility-percentile measure and showed up at
rank 6 in the full permutation importance), no-session-timing (nyse_open_flag/hour_utc/weekday),
no-own-signal-vars (p_fast/p_slow/delta_z -- this signal's own trigger-defining variables).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/eth_5m_orthogonal_combo_metalabel_features.csv"
REPORT_DIR = ROOT / "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830"

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]

FULL_FEATURES = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
]
VOL_REGIME_FEATURES = ["atr_pct", "atr_percentile_864", "realized_vol_ratio", "bb_width_pctile"]
SESSION_TIMING_FEATURES = ["nyse_open_flag", "hour_utc", "weekday"]
OWN_SIGNAL_FEATURES = ["p_fast", "p_slow", "delta_z"]

ABLATIONS = {
    "full_23_features": FULL_FEATURES,
    "ablated_no_vol_regime_19_features": [f for f in FULL_FEATURES if f not in VOL_REGIME_FEATURES],
    "ablated_no_session_timing_20_features": [f for f in FULL_FEATURES if f not in SESSION_TIMING_FEATURES],
    "ablated_no_own_signal_vars_20_features": [f for f in FULL_FEATURES if f not in OWN_SIGNAL_FEATURES],
}


def log(msg: str) -> None:
    print(f"[orthogonal_ablation] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
    table = pd.DataFrame(seed_rows)
    return {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def main() -> int:
    fires = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    log(f"loaded {len(fires)} fires from {CSV_PATH}")
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT n={len(holdout)}")

    results = {}
    for label, feats in ABLATIONS.items():
        log(f"=== {label} ({len(feats)} features) ===")
        results[label] = {
            "feature_columns": feats,
            "val": run_panel(train, val, feats, f"{label}/VAL"),
            "oos": run_panel(train, oos, feats, f"{label}/OOS"),
            "holdout": run_panel(train, holdout, feats, f"{label}/HOLDOUT"),
        }

    out_path = REPORT_DIR / "ablation_report.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log(f"saved -> {out_path}")

    log("")
    log("=== SUMMARY (VAL / OOS / HOLDOUT AUC) ===")
    for label in results:
        r = results[label]
        log(f"  {label}: VAL={r['val']['auc_mean']:.4f}  OOS={r['oos']['auc_mean']:.4f}  HOLDOUT={r['holdout']['auc_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
