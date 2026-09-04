#!/usr/bin/env python3
"""TabPFN cheap_gate for 추세지속(trend continuation) EXTEND/REVERT -- the confirmatory check
after the GBM proxy screen came back flat (5-point + 24-cell H x GAP grid, both 0.48-0.55,
regime-feature addition and 8-feature microstructure augmentation both failed to move it), per
the same reasoning applied to breakout continuation (research_eth_breakout_continuation_tabpfn_
cheap_gate_20260831.py): weak/flat GBM doesn't strictly prove TabPFN can't separate within the
population using nonlinear feature interactions GBM misses (V자반등's own raw trigger lift was
similarly weak before TabPFN made it this project's best classifier). Adapted directly from that
script's structure and methodology (embargoed Fresh-Forward TRAIN/VAL/OOS split, single-seed
TabPFN, HOLDOUT untouched).

Must run in the quant_ai conda env on the SERVER (GPU required, TABPFN_TOKEN already
authenticated there).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_trend_continuation_20260831/eth_5m_trend_continuation_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_trend_continuation_tabpfn_cheap_gate_20260831"

VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")
LABEL_WINDOW = pd.Timedelta(minutes=120)  # H=24 bars
SEED = 20260831

FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")
    if len(parts["train"]) > 10000:
        print(f"WARNING: train n={len(parts['train'])} exceeds TabPFN's designed <=10000-row range")

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
        "gbm_proxy_5point_h24_gap12": {"val_auc": 0.5188, "oos_auc": 0.5164},
        "gbm_proxy_24cell_grid_best": {"val_auc": 0.5474, "oos_auc": 0.5327},
        "gbm_proxy_augmented_31feat_best": {"val_auc": 0.5359, "oos_auc": 0.5152},
        "v_rebound_9trigger_cheap_gate": {"val_auc": 0.8296, "oos_auc": 0.8119},
    }
    print("\n=== vs this axis's own GBM-proxy history + V자반등 (different task/population, informative not strict) ===")
    for label, base in comparison.items():
        print(f"  {label}: VAL AUC delta {results['val']['auc'] - base['val_auc']:+.4f}, "
              f"OOS AUC delta {results['oos']['auc'] - base['oos_auc']:+.4f}")

    report = {"seed": SEED, "device": "cuda", "results": results, "comparison": comparison,
              "label": "EXTEND(1) vs REVERT(0), H=24/GAP=12/K=2.059, union-8-signal pure-direction",
              "note": "single-run cheap_gate, HOLDOUT untouched."}
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
