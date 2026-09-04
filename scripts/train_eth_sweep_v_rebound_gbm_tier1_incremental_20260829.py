#!/usr/bin/env python3
"""Incremental Tier 1 feature evaluation for the liquidity_sweep -> V_REBOUND model: trains the
SAME baseline GBM architecture (train_eth_sweep_v_rebound_gbm_baseline_20260829.py) with Tier 0
alone vs Tier0+each Tier1 group vs Tier0+all Tier1 groups, so each group's individual contribution
over the Tier-0-only baseline is visible rather than one combined number that hides which group
(if any) actually helped. Same Fresh-Forward split, same reserved 2026-04-01+ holdout untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0_tier1.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_gbm_tier1_incremental_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
TIER1_GROUPS = {
    "funding": ["t1_funding_last_funding_rate", "t1_funding_funding_z_score", "t1_funding_funding_abs"],
    "oi": ["t1_oi_oi_change_rate"],
    "btc": ["t1_btc_btc_ret_z_48", "t1_btc_eth_btc_ret_spread_12", "t1_btc_btc_volume_impulse_z", "t1_btc_btc_corr_60"],
    "liqmap": ["t1_liqmap_relevant_side_dist_atr"],
}
ALL_TIER1 = [c for cols in TIER1_GROUPS.values() for c in cols]


def split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    return {
        "train": df.loc[ts < VAL_START],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
        "reserved_untouched": df.loc[ts > OOS_END],
    }


def evaluate(model, frame: pd.DataFrame, feature_cols: list) -> dict:
    X, y = frame[feature_cols], frame["label"].to_numpy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4) if len(np.unique(y)) > 1 else None,
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def run_variant(parts: dict, feature_cols: list) -> dict:
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=20260829,
    )
    model.fit(parts["train"][feature_cols], parts["train"]["label"])
    return {name: evaluate(model, parts[name], feature_cols) for name in ("train", "val", "oos")}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    variants = {"tier0_only": TIER0}
    for name, cols in TIER1_GROUPS.items():
        variants[f"tier0+{name}"] = TIER0 + cols
    variants["tier0+all_tier1"] = TIER0 + ALL_TIER1

    report = {}
    for variant_name, feature_cols in variants.items():
        sub = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
        parts = split(sub)
        result = run_variant(parts, feature_cols)
        report[variant_name] = {"n_features": len(feature_cols), "n_train_after_dropna": len(parts["train"]), **result}
        print(f"\n=== {variant_name} ({len(feature_cols)} features, train n={len(parts['train'])}) ===")
        for split_name in ("train", "val", "oos"):
            r = result[split_name]
            print(f"  {split_name:5s} n={r['n']:5d} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
                  f"auc={r['auc']:.4f} naive={r['naive_majority_class_accuracy']:.4f} beats_naive={r['beats_naive_accuracy']}")

    print("\n\n=== SUMMARY: VAL/OOS AUC delta vs tier0_only ===")
    base_val_auc = report["tier0_only"]["val"]["auc"]
    base_oos_auc = report["tier0_only"]["oos"]["auc"]
    for variant_name in variants:
        v = report[variant_name]
        print(f"  {variant_name:20s} VAL AUC {v['val']['auc']:.4f} ({v['val']['auc']-base_val_auc:+.4f})   "
              f"OOS AUC {v['oos']['auc']:.4f} ({v['oos']['auc']-base_oos_auc:+.4f})")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
