#!/usr/bin/env python3
"""Re-verify the Omega4.6.1-sourced candidate features (mtf_trend_1h/fvg_dist/rsi/
cvp_cluster_position/breakout_strength/sig_volume_confirm -- all REJECTED under GBM, see
train_eth_sweep_v_rebound_gbm_omega461_incremental_20260829.py) under TabPFN instead, same
rationale as the Tier1 TabPFN re-verification (research_eth_sweep_v_rebound_tabpfn_tier1_
incremental_20260829.py, which found GBM's rejections did NOT carry over -- TabPFN was simply
neutral to those features, neither helped nor hurt).

3 seeds per variant (non-sequential), reads training_features_{2024,2025,2026_rebuilt}.csv
directly from the server's own copy (already present, not synced from dev).
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
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_omega461_incremental_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
SEEDS = [20260829, 141592, 271828]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
CANDIDATES = ["mtf_trend_1h", "fvg_dist", "rsi", "cvp_cluster_position", "breakout_strength", "sig_volume_confirm"]


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
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
    }


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

    all_results = {}
    for variant_name, feature_cols in variants.items():
        sub = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
        parts = embargoed_split(sub)
        print(f"\n=== {variant_name} ({len(feature_cols)} features, train n={len(parts['train'])}) ===")
        seed_rows = []
        for seed in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=seed)
            clf.fit(parts["train"][feature_cols], parts["train"]["label"].to_numpy())
            row = {"seed": seed}
            for split_name in ("val", "oos"):
                proba = clf.predict_proba(parts[split_name][feature_cols])[:, 1]
                r = evaluate(proba, parts[split_name]["label"].to_numpy())
                row[f"{split_name}_auc"] = r["auc"]
            seed_rows.append(row)
            print(f"  seed={seed}: val_auc={row['val_auc']:.4f} oos_auc={row['oos_auc']:.4f}")
        table = pd.DataFrame(seed_rows)
        summary = {
            "n_features": len(feature_cols), "n_train": len(parts["train"]),
            "val_auc_mean": round(float(table["val_auc"].mean()), 4),
            "val_auc_std": round(float(table["val_auc"].std(ddof=1)), 4),
            "oos_auc_mean": round(float(table["oos_auc"].mean()), 4),
            "oos_auc_std": round(float(table["oos_auc"].std(ddof=1)), 4),
            "per_seed": seed_rows,
        }
        all_results[variant_name] = summary
        print(f"  -> VAL {summary['val_auc_mean']:.4f}+/-{summary['val_auc_std']:.4f}  "
              f"OOS {summary['oos_auc_mean']:.4f}+/-{summary['oos_auc_std']:.4f}")

    base = all_results["tier0_only"]
    print("\n\n=== SUMMARY: delta vs tier0_only ===")
    for variant_name, s in all_results.items():
        print(f"  {variant_name:30s} VAL {s['val_auc_mean']:.4f} ({s['val_auc_mean']-base['val_auc_mean']:+.4f})   "
              f"OOS {s['oos_auc_mean']:.4f} ({s['oos_auc_mean']-base['oos_auc_mean']:+.4f})")

    (OUT_DIR / "report.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    print("OMEGA461_TABPFN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
