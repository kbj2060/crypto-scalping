#!/usr/bin/env python3
"""Re-verify the Tier1 feature groups (funding/OI/BTC/liqmap -- all REJECTED under the GBM
baseline, see docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md) under
TabPFN instead, since TabPFN is a structurally different model (in-context foundation model,
not gradient-boosted trees) and GBM's rejection doesn't automatically carry over.

3 seeds per variant (non-sequential, matching this project's seed-diversity convention) --
TabPFN's own seed-to-seed std was measured at 0.0008 (VAL) / 0.0002 (OOS) on Tier0 alone, far
tighter than GBM/TabM ever were, so 3 seeds is enough to detect anything above that noise floor
without the full 5-seed budget every variant.

Tier0-only reference (already measured, 4 seeds): VAL AUC 0.6423+/-0.0008, OOS AUC 0.6566+/-0.0002.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0_tier1.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_tier1_incremental_20260829"

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
TIER1_GROUPS = {
    "funding": ["t1_funding_last_funding_rate", "t1_funding_funding_z_score", "t1_funding_funding_abs"],
    "oi": ["t1_oi_oi_change_rate"],
    "btc": ["t1_btc_btc_ret_z_48", "t1_btc_eth_btc_ret_spread_12", "t1_btc_btc_volume_impulse_z", "t1_btc_btc_corr_60"],
    "liqmap": ["t1_liqmap_relevant_side_dist_atr"],
}
ALL_TIER1 = [c for cols in TIER1_GROUPS.values() for c in cols]


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
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    variants = {"tier0_only": TIER0}
    for name, cols in TIER1_GROUPS.items():
        variants[f"tier0+{name}"] = TIER0 + cols
    variants["tier0+all_tier1"] = TIER0 + ALL_TIER1

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
            for name in ("val", "oos"):
                proba = clf.predict_proba(parts[name][feature_cols])[:, 1]
                r = evaluate(proba, parts[name]["label"].to_numpy())
                row[f"{name}_auc"] = r["auc"]
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
        print(f"  {variant_name:20s} VAL {s['val_auc_mean']:.4f} ({s['val_auc_mean']-base['val_auc_mean']:+.4f})   "
              f"OOS {s['oos_auc_mean']:.4f} ({s['oos_auc_mean']-base['oos_auc_mean']:+.4f})")

    (OUT_DIR / "report.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    print("TIER1_TABPFN_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
