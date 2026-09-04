#!/usr/bin/env python3
"""Seed-stability check for the TabPFN cheap_gate win
(research_eth_sweep_v_rebound_tabpfn_server_20260829.py: VAL AUC 0.6414 / OOS AUC 0.6565,
seed=20260829, +0.0192/+0.0140 over the GBM Tier0 baseline). Runs the identical fit/eval across
4 additional distinct seeds (not fixed-increment, matching this project's own seed-diversity
convention) to check whether the win reproduces or was a single favorable draw.
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
SEEDS = [20260829, 141592, 271828, 577215]  # first already run; keep + 3 more, non-sequential

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
    }


def main() -> int:
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    parts = embargoed_split(df)
    print(f"train n={len(parts['train'])} val n={len(parts['val'])} oos n={len(parts['oos'])}\n")

    per_seed = []
    for seed in SEEDS:
        print(f"=== seed={seed} ===")
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        row = {"seed": seed}
        for name in ("val", "oos"):
            proba = clf.predict_proba(parts[name][FEATURE_COLUMNS])[:, 1]
            r = evaluate(proba, parts[name]["label"].to_numpy())
            row[f"{name}_auc"] = r["auc"]
            row[f"{name}_acc"] = r["accuracy"]
            row[f"{name}_bal_acc"] = r["balanced_accuracy"]
            print(f"  {name:4s} auc={r['auc']:.4f} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f}")
        per_seed.append(row)

    table = pd.DataFrame(per_seed)
    gbm = {"val_auc": 0.6222, "oos_auc": 0.6425}
    print("\n=== summary across seeds ===")
    for col in ("val_auc", "oos_auc"):
        vals = table[col].to_numpy()
        print(f"  {col}: mean={vals.mean():.4f} std={vals.std(ddof=1):.4f} min={vals.min():.4f} max={vals.max():.4f} "
              f"(vs GBM {gbm[col]:.4f}, {int((vals > gbm[col]).sum())}/{len(vals)} seeds beat it)")

    (OUT_DIR / "seed_stability.json").write_text(
        json.dumps({"seeds": SEEDS, "per_seed": per_seed, "gbm_baseline": gbm}, indent=2)
    )
    print(f"\nWrote {OUT_DIR / 'seed_stability.json'}")
    print("SEED_CHECK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
