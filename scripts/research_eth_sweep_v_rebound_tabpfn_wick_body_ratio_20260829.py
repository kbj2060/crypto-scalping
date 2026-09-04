#!/usr/bin/env python3
"""Test wick_body_ratio (candle wick-vs-BODY ratio, not Tier0's existing wick-vs-RANGE ratio) as
a 24th feature -- same formula as the liquidation cascade pilot's extract_features(), but applied
to every V_REBOUND sweep event directly, no liquidation-cascade gating (pure OHLC, so it exists
for the FULL 2024-2026 history, unlike nif_whale_rel/the cascade features themselves).

VAL/OOS only, same TRAIN<2025-09-01 as every other test in this lineage -- the reserved holdout
(2026-04-01..08-28) is already spent (see eth_liquidity_sweep_v_rebound_feature_plan_20260829.md)
and stays untouched here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
RSI_SOURCES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_tabpfn_wick_body_ratio_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
BASE = TIER0 + ["rsi"]


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)

    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.drop_duplicates("timestamp")

    frames = []
    for path in RSI_SOURCES:
        f = pd.read_csv(path, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")

    df = tier0.merge(kl, on="timestamp", how="left").merge(rsi, on="timestamp", how="left")

    body = (df["close"] - df["open"]).abs().clip(lower=1e-9)
    wick_in_direction = np.where(
        df["is_downside"] == 1,
        np.minimum(df["open"], df["close"]) - df["low"],
        df["high"] - np.maximum(df["open"], df["close"]),
    )
    df["wick_body_ratio"] = wick_in_direction / body

    variants = {"tier0+rsi (baseline)": BASE, "tier0+rsi+wick_body_ratio": BASE + ["wick_body_ratio"]}

    def embargoed_split(sub: pd.DataFrame) -> dict:
        ts = sub["timestamp"]
        window_end = ts + pd.Timedelta(minutes=30)
        return {
            "train": sub.loc[(ts < VAL_START) & (window_end < VAL_START)],
            "val": sub.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
            "oos": sub.loc[(ts >= OOS_START) & (ts <= OOS_END)],
        }

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
                row[f"{split_name}_auc"] = round(float(roc_auc_score(parts[split_name]["label"], proba)), 4)
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

    base = all_results["tier0+rsi (baseline)"]
    print("\n=== SUMMARY: delta vs tier0+rsi baseline ===")
    for name, s in all_results.items():
        print(f"  {name:30s} VAL {s['val_auc_mean']:.4f} ({s['val_auc_mean']-base['val_auc_mean']:+.4f})  "
              f"OOS {s['oos_auc_mean']:.4f} ({s['oos_auc_mean']-base['oos_auc_mean']:+.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    print("WICK_BODY_RATIO_TEST_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
