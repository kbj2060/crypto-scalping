#!/usr/bin/env python3
"""v4 (current production, all 14,259 events, binary) vs v7b (2026-08-30: V자반등=close-confirmed
1.5x-ATR/30min + giveback<=0.20/60min, 지지/횡보=fast_move<1.0x-ATR/30min, everything between
EXCLUDED -- 5,933/14,259 events retained, 41.6%). Same TabPFN(Tier0+rsi), 4 seeds, VAL/OOS/holdout.
v7b's val/oos/holdout are built from its OWN filtered population (excluded events have no
well-defined ground truth under this design, so they cannot appear in any split).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START, HOLDOUT_END = pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-08-28 23:59:59", tz="UTC")
LABEL_WINDOW = pd.Timedelta(minutes=30)
SEEDS = [20260829, 141592, 271828, 577215]

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]
RSI_SOURCES = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]


def load_rsi() -> pd.DataFrame:
    frames = []
    for p in RSI_SOURCES:
        f = pd.read_csv(p, usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")


def build_splits(tier0_path: Path, rsi: pd.DataFrame, label_window: pd.Timedelta) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tier0 = pd.read_csv(tier0_path)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]
    window_end = ts + label_window
    val = df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)]
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)]
    holdout = df.loc[(ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)]
    return val, oos, holdout


def run(name: str, train_path: Path, tier0_path_for_splits: Path, rsi: pd.DataFrame, label_window: pd.Timedelta) -> None:
    train = pd.read_csv(train_path)
    val, oos, holdout = build_splits(tier0_path_for_splits, rsi, label_window)
    print(f"\n=== {name} (train n={len(train)}, val n={len(val)}, oos n={len(oos)}, holdout n={len(holdout)}, "
          f"train label_rate={train['label'].mean():.4f}, val label_rate={val['label'].mean():.4f}, "
          f"oos label_rate={oos['label'].mean():.4f}, holdout label_rate={holdout['label'].mean():.4f}) ===")
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        row = {"seed": seed}
        for split_name, split in (("val", val), ("oos", oos), ("holdout", holdout)):
            proba = clf.predict_proba(split[FEATURES])[:, 1]
            row[f"{split_name}_auc"] = round(float(roc_auc_score(split["label"], proba)), 4)
        seed_rows.append(row)
        print(f"  seed={seed}: val_auc={row['val_auc']:.4f} oos_auc={row['oos_auc']:.4f} holdout_auc={row['holdout_auc']:.4f}")
    table = pd.DataFrame(seed_rows)
    print(f"  -> VAL {table['val_auc'].mean():.4f}+/-{table['val_auc'].std(ddof=1):.4f}  "
          f"OOS {table['oos_auc'].mean():.4f}+/-{table['oos_auc'].std(ddof=1):.4f}  "
          f"HOLDOUT {table['holdout_auc'].mean():.4f}+/-{table['holdout_auc'].std(ddof=1):.4f}")


def main() -> int:
    rsi = load_rsi()
    run("v4 (current production label, all 14,259 events)",
        LABEL_DIR / "tabpfn_train_context_frozen_v4_20260830.csv",
        LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv",
        rsi, label_window=pd.Timedelta(minutes=30))
    run("v7b (V자반등/지지횡보 2-class, 41.6% retained, fuzzy middle excluded)",
        LABEL_DIR / "tabpfn_train_context_v7b_20260830.csv",
        LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0_v7b_20260830.csv",
        rsi, label_window=pd.Timedelta(minutes=60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
