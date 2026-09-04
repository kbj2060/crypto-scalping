#!/usr/bin/env python3
"""Ablation for volume_wick_climax v1 (research_eth_volume_wick_climax_metalabel_tabpfn_20260830.py,
HORIZON=2h/GAP=3/K=1.90 label, VAL/OOS/HOLDOUT AUC 0.612/0.563/0.565): permutation importance found
nyse_open_flag (+0.034) and atr_percentile_864 (+0.024) dominating the top-2, while the signal's OWN
defining features (vol_z -0.003, lower_wick_ratio +0.0002, upper_wick_ratio -0.005) were ~0 or
NEGATIVE -- the opposite pattern from taker_delta_z_climax, where an equivalent vol-regime ablation
found the signal was NOT primarily a regime proxy (removing 3 vol-regime features only cost
~0.01-0.012 AUC). Does volume_wick_climax's AUC survive without session-timing features, without
vol-regime features, or without its own defining wick/volume features? 3 separate ablations
(session/vol-regime/own-signal) rather than just 1, since this signal's permutation importance
implicated 2 different feature groups AND raised a specific "own trigger vars near-zero" question
taker's ablation never needed to ask.

Reuses the already-built v1 feature CSV (cluster-anchored on vol_z, GAP=3, HORIZON=24 -- no rebuild
needed). Same TRAIN(<2025-09-01)/VAL(2025-09-01~2025-12-31)/OOS(2026-01-01~2026-03-31)/
HOLDOUT(2026-04-01~) split, same 4 seeds, full-23-feature config re-run alongside each ablated
config as an internal consistency check (full should reproduce v1's original report.json numbers
almost exactly).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/eth_5m_volume_wick_climax_metalabel_20260830/eth_5m_volume_wick_climax_metalabel_features.csv"
REPORT_DIR = ROOT / "tmp/eth_volume_wick_climax_metalabel_tabpfn_20260830"

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
SESSION_TIMING_FEATURES = ["hour_utc", "weekday", "nyse_open_flag"]  # perm importance #1 (nyse_open_flag) lives here
VOL_REGIME_FEATURES = ["atr_pct", "atr_percentile_864", "realized_vol_ratio"]  # same grouping as taker's ablation, for cross-signal comparability
OWN_SIGNAL_FEATURES = ["vol_z", "lower_wick_ratio", "upper_wick_ratio"]  # the fire condition's own defining variables

ABLATIONS = {
    "full_23_features": FULL_FEATURES,
    "ablated_no_session_timing_20_features": [f for f in FULL_FEATURES if f not in SESSION_TIMING_FEATURES],
    "ablated_no_vol_regime_20_features": [f for f in FULL_FEATURES if f not in VOL_REGIME_FEATURES],
    "ablated_no_own_signal_vars_20_features": [f for f in FULL_FEATURES if f not in OWN_SIGNAL_FEATURES],
}


def log(msg: str) -> None:
    print(f"[vwc_ablation] {msg}", flush=True)


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
