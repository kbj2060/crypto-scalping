#!/usr/bin/env python3
"""Fair test of the user's hypothesis ("looser label -> higher accuracy -> better model"): trains
the SAME TabPFN + Tier0+rsi features on the loose-touch-1h label (base rate 78.2%, vs the current
V_REBOUND label's 43.9%) and compares via AUC + lift-over-THIS-LABEL'S-OWN-naive-baseline, not
raw accuracy (which isn't comparable across labels with very different base rates).

Same TRAIN(<2025-09-01)/VAL(2025-09-01..12-31)/OOS(2026-01-01..03-31) split, same 4-seed panel,
VAL+OOS only (reserved holdout untouched, same discipline as every other test in this lineage).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
LOOSE_LABELS_CSV = ROOT / "data/labels/eth_5m_sweep_loose_touch_1h_20260829/eth_5m_sweep_loose_touch_1h_labels.csv"
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=60)  # this label's own forward window -- embargo must match it, not 30min
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END)],
    }


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    tier0 = tier0.drop(columns=["label"])  # drop the OLD V_REBOUND label -- replaced below

    loose = pd.read_csv(LOOSE_LABELS_CSV, usecols=["timestamp", "side", "label"])
    loose["timestamp"] = pd.to_datetime(loose["timestamp"], utc=True)

    df = tier0.merge(loose, on=["timestamp", "side"], how="inner")
    print(f"merged rows: {len(df)} (tier0 had {len(tier0)}, loose labels had {len(loose)})")

    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    rsi = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
    df = df.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    print(f"train n={len(parts['train'])}  val n={len(parts['val'])}  oos n={len(parts['oos'])}")
    for name in ("train", "val", "oos"):
        r = parts[name]["label"].mean()
        print(f"  {name} label rate: {r:.4f}  naive baseline: {max(r, 1-r):.4f}")

    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(parts["train"][FEATURES], parts["train"]["label"].to_numpy())
        row = {"seed": seed}
        for split_name in ("val", "oos"):
            proba = clf.predict_proba(parts[split_name][FEATURES])[:, 1]
            y = parts[split_name]["label"].to_numpy()
            pred = (proba >= 0.5).astype(int)
            row[f"{split_name}_auc"] = round(float(roc_auc_score(y, proba)), 4)
            row[f"{split_name}_accuracy"] = round(float((pred == y).mean()), 4)
        seed_rows.append(row)
        print(f"  seed={seed}: val_auc={row['val_auc']:.4f} val_acc={row['val_accuracy']:.4f}  "
              f"oos_auc={row['oos_auc']:.4f} oos_acc={row['oos_accuracy']:.4f}")

    table = pd.DataFrame(seed_rows)
    val_naive = max(parts["val"]["label"].mean(), 1 - parts["val"]["label"].mean())
    oos_naive = max(parts["oos"]["label"].mean(), 1 - parts["oos"]["label"].mean())
    print(f"\n=== SUMMARY: loose-touch-1h label ===")
    print(f"  VAL  AUC {table['val_auc'].mean():.4f}+/-{table['val_auc'].std(ddof=1):.4f}   "
          f"acc {table['val_accuracy'].mean():.4f} (naive {val_naive:.4f}, "
          f"lift {table['val_accuracy'].mean()-val_naive:+.4f})")
    print(f"  OOS  AUC {table['oos_auc'].mean():.4f}+/-{table['oos_auc'].std(ddof=1):.4f}   "
          f"acc {table['oos_accuracy'].mean():.4f} (naive {oos_naive:.4f}, "
          f"lift {table['oos_accuracy'].mean()-oos_naive:+.4f})")
    print(f"\n=== FOR COMPARISON: current V_REBOUND label (Tier0+rsi, already validated) ===")
    print(f"  VAL  AUC 0.6423+/-0.0008   acc ~0.61 (naive ~0.546-0.576, lift ~+0.05-0.06)")
    print(f"  OOS  AUC 0.6566+/-0.0002   acc ~0.616 (naive ~0.536-0.561, lift ~+0.06-0.07)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
