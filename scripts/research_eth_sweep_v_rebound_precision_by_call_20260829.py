#!/usr/bin/env python3
"""User asked a discretionary-use question, distinct from the automated cost-gate question: "is
this signal's accuracy good enough for ME to look at and trade on, manually?" The right number
for that is PRECISION PER PREDICTED CLASS ("when it says rebound, how often is it actually
right?"), not the blended overall accuracy (61.6%) reported so far -- those can differ a lot
under class imbalance, and the discretionary reader sees one call at a time, not an average.

VAL+OOS only (2,961 events combined) -- large enough sample on its own; deliberately does NOT
re-touch the reserved holdout (already spent) for this extra resolution, since VAL+OOS alone give
a statistically solid picture and the holdout's own aggregate accuracy (0.647) already confirms
the same overall story without needing a fresh per-class slice of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]  # same 4-seed panel as the original stability check

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def report_for(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    base_rate_1 = y_true.mean()
    n_called_rebound, n_called_cont = (tp + fp), (tn + fn)
    prec_rebound = tp / n_called_rebound if n_called_rebound else float("nan")
    prec_cont = tn / n_called_cont if n_called_cont else float("nan")
    recall_rebound = tp / (tp + fn) if (tp + fn) else float("nan")
    recall_cont = tn / (tn + fp) if (tn + fp) else float("nan")
    print(f"\n=== {label} (n={n}, base rate V_REBOUND=1: {base_rate_1:.4f}) ===")
    print(f"  called REBOUND: {n_called_rebound} ({n_called_rebound/n:.1%}) -> precision {prec_rebound:.4f} "
          f"(vs base rate {base_rate_1:.4f}, lift {prec_rebound/base_rate_1:.3f}x)  recall {recall_rebound:.4f}")
    print(f"  called CONTINUATION: {n_called_cont} ({n_called_cont/n:.1%}) -> precision {prec_cont:.4f} "
          f"(vs base rate {1-base_rate_1:.4f}, lift {prec_cont/(1-base_rate_1):.3f}x)  recall {recall_cont:.4f}")
    print(f"  overall accuracy: {(tp+tn)/n:.4f}")


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    train = df.loc[df["timestamp"] < VAL_START]
    val = df.loc[(df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)]
    oos = df.loc[(df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)]
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
