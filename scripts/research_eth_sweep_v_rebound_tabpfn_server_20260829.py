#!/usr/bin/env python3
"""Single cheap_gate run: TabPFN v2 (Prior Labs, arXiv:2511.08667 / Nature 2024) on the
liquidity_sweep -> V_REBOUND task, for direct comparison against:
  - GBM Tier0 baseline: VAL AUC 0.6222 / OOS AUC 0.6425
  - TabM cheap_gate:     VAL AUC 0.6108 / OOS AUC 0.6232 (rejected)

Why TabPFN specifically: it is an in-context-learning tabular foundation model with no
per-dataset gradient training -- designed and benchmarked for <=10,000-row classification
datasets, where published results show a 100% win rate against default XGBoost. This
project's train set (9,136 rows after embargo) sits inside that designed range, and TabPFN
sidesteps the small-sample training-instability failure mode that sank the TabM cheap_gate
(no random init / seed-sensitive gradient descent to overfit with).

Must run in the `quant_ai` conda env on the SERVER (GPU required for a train set this size --
Prior Labs' own guidance: CPU is only realistic under ~1,000 rows). Requires TABPFN_TOKEN set
in the environment (license accepted at https://ux.priorlabs.ai) for the first model-weight
download; cached under ~/.cache after that.

Same purge/embargo + Fresh-Forward split as the GBM/TabM scripts, for exact comparability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_cheap_gate_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
SEED = 20260829

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
    }


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)), "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")
    if len(parts["train"]) > 10000:
        print(f"WARNING: train n={len(parts['train'])} exceeds TabPFN's designed <=10000-row range "
              f"(ignore_pretraining_limits would be needed) -- not expected for this dataset, check split.")

    print("\nfitting TabPFNClassifier (device=cuda)...")
    clf = TabPFNClassifier(device="cuda", random_state=SEED)
    clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
    print("fit complete.")

    results = {}
    for name in ("train", "val", "oos"):
        proba = clf.predict_proba(parts[name][FEATURE_COLUMNS])[:, 1]
        results[name] = evaluate(proba, parts[name]["label"].to_numpy())
        r = results[name]
        print(f"  {name:5s} n={r['n']:5d} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
              f"auc={r['auc']:.4f} naive={r['naive_majority_class_accuracy']:.4f} beats_naive={r['beats_naive_accuracy']}")

    comparison = {
        "gbm_tier0_baseline": {"val_auc": 0.6222, "oos_auc": 0.6425},
        "tabm_cheap_gate": {"val_auc": 0.6108, "oos_auc": 0.6232},
    }
    print("\n=== vs prior candidates ===")
    for label, base in comparison.items():
        print(f"  {label}: VAL AUC delta {results['val']['auc'] - base['val_auc']:+.4f}, "
              f"OOS AUC delta {results['oos']['auc'] - base['oos_auc']:+.4f}")

    report = {
        "seed": SEED, "device": "cuda", "results": results, "comparison": comparison,
        "note": "single-run cheap_gate -- TabPFN has no gradient-training seed sensitivity like TabM, "
                "but still a single train/val/oos split; not yet cross-validated.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
