#!/usr/bin/env python3
"""v7b precision-by-call, same pattern as research_eth_sweep_v_rebound_precision_by_call_20260829.py
but pointed at the v7b train/tier0 files (fuzzy-middle-excluded population)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=60)
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def report_for(y_true, y_pred, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    base_rate_1 = y_true.mean()
    n1, n0 = (tp + fp), (tn + fn)
    prec1 = tp / n1 if n1 else float("nan")
    prec0 = tn / n0 if n0 else float("nan")
    print(f"\n=== {label} (n={n}, base rate V자반등=1: {base_rate_1:.4f}) ===")
    print(f"  called V자반등: {n1} ({n1/n:.1%}) -> precision {prec1:.4f} (lift {prec1/base_rate_1:.3f}x)")
    print(f"  called 지지횡보: {n0} ({n0/n:.1%}) -> precision {prec0:.4f} (lift {prec0/(1-base_rate_1):.3f}x)")
    print(f"  overall accuracy: {(tp+tn)/n:.4f}")


def main() -> int:
    train = pd.read_csv(LABEL_DIR / "tabpfn_train_context_frozen_v7b_20260830.csv")
    tier0 = pd.read_csv(LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0_v7b_20260830.csv")
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    frames = []
    for y in ("2024", "2025", "2026_rebuilt"):
        f = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    val = df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)]
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)]
    print(f"train n={len(train)}  val n={len(val)}  oos n={len(oos)}")

    val_preds, oos_preds = [], []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        val_preds.append(clf.predict_proba(val[FEATURES])[:, 1])
        oos_preds.append(clf.predict_proba(oos[FEATURES])[:, 1])
        print(f"  seed={seed} done")

    val_proba = np.mean(val_preds, axis=0)
    oos_proba = np.mean(oos_preds, axis=0)
    report_for(val["label"].to_numpy(), (val_proba >= 0.5).astype(int), "VAL (2025-09~12)")
    report_for(oos["label"].to_numpy(), (oos_proba >= 0.5).astype(int), "OOS (2026-01~03)")
    combined_true = np.concatenate([val["label"].to_numpy(), oos["label"].to_numpy()])
    combined_pred = np.concatenate([(val_proba >= 0.5).astype(int), (oos_proba >= 0.5).astype(int)])
    report_for(combined_true, combined_pred, "VAL+OOS combined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
