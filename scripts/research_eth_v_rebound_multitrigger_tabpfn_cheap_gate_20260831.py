#!/usr/bin/env python3
"""Single cheap_gate run: TabPFN on the new 9-trigger V자반등 candidate pool (data/labels/eth_5m_
v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv), for direct
comparison against the ORIGINAL sweep-only v7b numbers (VAL AUC 0.7342, OOS AUC 0.7621, HOLDOUT
AUC 0.7788 -- docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md lines
1086-1088). Question this answers: does widening the trigger pool from sweep-only to 9 triggers
(sweep + 5 existing evidence signals + local_extreme + demarker_extreme + kalman_deviation_meanrev)
keep the outcome learnable at a comparable level, or does the added heterogeneity dilute it?

Adapted from research_eth_sweep_v_rebound_tabpfn_server_20260829.py (the original sweep-only
cheap_gate, same methodology: embargoed Fresh-Forward TRAIN/VAL/OOS split, single-seed TabPFN,
HOLDOUT untouched). Differences from that script:
  - label: outcome column (V자반등/지지횡보/애매) -> binary (V자반등=1, 지지횡보=0), 애매 dropped
    (same exclude-middle convention v7b itself uses -- not a new design choice).
  - LABEL_WINDOW=60min (not 30) -- this label's own outcome window is 30min(fast)/60min(full);
    embargo purge must cover the full resolution window, not just the fast one.
  - FEATURE_COLUMNS: 22 Tier0 + rsi (23 total), matching the live v7b feature contract exactly.
  - TRAIN n=18,087 (checked locally before this ran) exceeds TabPFN's designed <=10,000-row
    guideline -- ignore_pretraining_limits=True set explicitly rather than silently subsampling,
    so the model sees the full trigger-type diversity (subsampling risks under-representing the
    rarer triggers, e.g. orthogonal_combo n=3,211 candidates project-wide).

Must run in the quant_ai conda env on the SERVER (GPU required, TABPFN_TOKEN already authenticated
there per docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md's own note).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_tabpfn_cheap_gate_20260831"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # untouched by this script -- cheap_gate only
LABEL_WINDOW = pd.Timedelta(minutes=60)
SEED = 20260831

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
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
    df = df[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")

    over_limit = len(parts["train"]) > 10000
    print(f"\ntrain n={len(parts['train'])} {'EXCEEDS' if over_limit else 'within'} TabPFN's "
          f"designed <=10000-row range -- ignore_pretraining_limits={over_limit}")

    print("\nfitting TabPFNClassifier (device=cuda)...")
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=over_limit)
    clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
    print("fit complete.")

    results = {}
    for name in ("train", "val", "oos"):
        proba = clf.predict_proba(parts[name][FEATURE_COLUMNS])[:, 1]
        results[name] = evaluate(proba, parts[name]["label"].to_numpy())
        r = results[name]
        print(f"  {name:5s} n={r['n']:5d} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
              f"auc={r['auc']:.4f} naive={r['naive_majority_class_accuracy']:.4f} beats_naive={r['beats_naive_accuracy']}")

    comparison = {"v7b_sweep_only": {"val_auc": 0.7342, "oos_auc": 0.7621, "holdout_auc": 0.7788}}
    print("\n=== vs v7b (sweep-only, narrower/easier population) ===")
    for label, base in comparison.items():
        print(f"  {label}: VAL AUC delta {results['val']['auc'] - base['val_auc']:+.4f}, "
              f"OOS AUC delta {results['oos']['auc'] - base['oos_auc']:+.4f}")

    report = {
        "seed": SEED, "device": "cuda", "ignore_pretraining_limits": over_limit,
        "feature_columns": FEATURE_COLUMNS, "results": results, "comparison": comparison,
        "note": "single-run cheap_gate, HOLDOUT untouched. Population is 9-trigger (heterogeneous), "
                "not directly the same task as v7b's sweep-only population -- comparison is "
                "informative, not a strict apples-to-apples re-test.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
