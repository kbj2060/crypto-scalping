#!/usr/bin/env python3
"""Single, final look at the 2026-04-01~2026-08-28 RESERVED holdout (2,086 events) --
deliberately untouched all session (never used for feature selection, model selection, or
hyperparameter tuning; VAL/OOS were reused up to 14x this session, this period was reused 0x).

Trains on the EXACT SAME train set used for every prior evaluation this session
(ts < VAL_START, 9,136 rows post-embargo) -- no retraining on VAL+OOS -- so this isolates one
question only: does the already-finalized config (Tier0, 22 features, TabPFN) generalize
further forward in time, or was it quietly overfit via repeated VAL/OOS exposure?

Two variants only:
  - tier0_only: the recommended final config (see eth_liquidity_sweep_v_rebound_feature_plan_
    20260829.md)
  - tier0+rsi: the one Omega4.6.1 candidate that showed a consistent (if weak) positive signal
    in the 3-seed re-verification (VAL +0.0012 / OOS +0.0010) -- resolved here, one way or the
    other, with the full 4-seed panel instead of leaving it as an open thread.

4 seeds (20260829/141592/271828/577215), matching the original seed-stability check's rigor
(the incremental Tier1/Omega461 screens used only 3) since this holdout gets exactly one shot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
TRAINING_FEATURES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_reserved_holdout_20260829"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_END = pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)

    frames = []
    for path in TRAINING_FEATURES:
        f = pd.read_csv(path, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    df = tier0.merge(rsi, on="timestamp", how="left")

    variants = {"tier0_only": TIER0, "tier0+rsi": TIER0 + ["rsi"]}

    all_results = {}
    for variant_name, feature_cols in variants.items():
        sub = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
        ts = sub["timestamp"]
        train = sub.loc[ts < VAL_START]
        reserved = sub.loc[ts > OOS_END]
        print(f"\n=== {variant_name} ({len(feature_cols)} features, train n={len(train)}, "
              f"reserved n={len(reserved)}) ===")
        seed_rows = []
        for seed in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=seed)
            clf.fit(train[feature_cols], train["label"].to_numpy())
            proba = clf.predict_proba(reserved[feature_cols])[:, 1]
            r = evaluate(proba, reserved["label"].to_numpy())
            r["seed"] = seed
            seed_rows.append(r)
            print(f"  seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
                  f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
        table = pd.DataFrame(seed_rows)
        summary = {
            "n_features": len(feature_cols), "n_train": len(train), "n_reserved": len(reserved),
            "auc_mean": round(float(table["auc"].mean()), 4),
            "auc_std": round(float(table["auc"].std(ddof=1)), 4),
            "accuracy_mean": round(float(table["accuracy"].mean()), 4),
            "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
            "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
            "per_seed": seed_rows,
        }
        all_results[variant_name] = summary
        print(f"  -> AUC {summary['auc_mean']:.4f}+/-{summary['auc_std']:.4f}  "
              f"acc {summary['accuracy_mean']:.4f}  bal_acc {summary['balanced_accuracy_mean']:.4f}")

    print("\n\n=== SUMMARY (reserved holdout 2026-04-01~2026-08-28, single exposure) ===")
    for variant_name, s in all_results.items():
        print(f"  {variant_name:12s} AUC {s['auc_mean']:.4f}+/-{s['auc_std']:.4f}  "
              f"acc {s['accuracy_mean']:.4f} (naive {s['naive_majority_accuracy']:.4f})  "
              f"bal_acc {s['balanced_accuracy_mean']:.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    print("RESERVED_HOLDOUT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
