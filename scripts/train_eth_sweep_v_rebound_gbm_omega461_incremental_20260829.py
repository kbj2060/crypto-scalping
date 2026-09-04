#!/usr/bin/env python3
"""Incremental test of the top Omega4.6.1-sourced candidates from
research_eth_sweep_omega461_feature_correlation_20260829.py's decile-spread screen, same
methodology as the Tier1 incremental test (train_eth_sweep_v_rebound_gbm_tier1_incremental_20260829.py):
Tier0 alone vs Tier0+each candidate vs Tier0+all candidates, same Fresh-Forward split, same
reserved 2026-04-01+ holdout untouched.

Candidates (picked from the correlation screen's decile-spread ranking, skipping mtf_trend_4h as
likely redundant with mtf_trend_1h -- same family, shorter list to isolate contribution cleanly):
  mtf_trend_1h, fvg_dist, rsi, cvp_cluster_position, breakout_strength, sig_volume_confirm
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
TRAINING_FEATURES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_gbm_omega461_incremental_20260829"

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
CANDIDATES = ["mtf_trend_1h", "fvg_dist", "rsi", "cvp_cluster_position", "breakout_strength", "sig_volume_confirm"]


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
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)

    frames = []
    for path in TRAINING_FEATURES:
        f = pd.read_csv(path, usecols=["timestamp"] + CANDIDATES)
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    omega_feats = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")

    df = tier0.merge(omega_feats, on="timestamp", how="left")

    variants = {"tier0_only": TIER0}
    for c in CANDIDATES:
        variants[f"tier0+{c}"] = TIER0 + [c]
    variants["tier0+all_omega461_candidates"] = TIER0 + CANDIDATES

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
        print(f"  {variant_name:30s} VAL AUC {v['val']['auc']:.4f} ({v['val']['auc']-base_val_auc:+.4f})   "
              f"OOS AUC {v['oos']['auc']:.4f} ({v['oos']['auc']-base_oos_auc:+.4f})")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
