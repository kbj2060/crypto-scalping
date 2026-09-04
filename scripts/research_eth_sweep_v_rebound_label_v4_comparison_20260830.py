#!/usr/bin/env python3
"""Compare the v3 (original) vs v4 (2026-08-30 user code review) V_REBOUND label definitions,
same TabPFN(Tier0+rsi) model, same event population (14,259 sweeps, identical candidate_index/
side set verified before this script ran) -- only the ground-truth label changed:

v4 fixes two issues in build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py:
  (a) ATR was self-inclusive (included the sweep bar's own, typically outsized, true range) ->
      now pre-sweep ATR (14 bars strictly before the sweep bar).
  (b) The 1.5x-ATR move had no timing requirement (could arrive any time in the 30-minute
      window) -> now must arrive within the first 15 minutes (3 of 6 bars); the 30-minute hold
      check is unchanged.
Net effect on the raw label rate: 43.9% (v3) -> 39.7% (v4), 646/14259 events flipped (619 of
those 1->0 from the now-required speed, only 31 0->1 from the now-smaller ATR denominator).

VAL/OOS only (matches every other test in this lineage) -- the reserved holdout
(2026-04-01..08-28) stays untouched pending the user's review of these VAL/OOS numbers first.
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


def build_splits(tier0_path: Path, rsi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tier0 = pd.read_csv(tier0_path)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    val = df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)]
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)]
    holdout = df.loc[(ts >= HOLDOUT_START) & (ts <= HOLDOUT_END)]
    return val, oos, holdout


def run(name: str, train_path: Path, tier0_path_for_splits: Path, rsi: pd.DataFrame) -> None:
    train = pd.read_csv(train_path)
    val, oos, holdout = build_splits(tier0_path_for_splits, rsi)
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
    run("v3 (original label)",
        LABEL_DIR / "tabpfn_train_context_frozen_20260829.csv",
        LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv.bak_pre_v4_20260830",
        rsi)
    run("v4 (pre-sweep ATR + fast-window fix)",
        LABEL_DIR / "tabpfn_train_context_v4_20260830.csv",
        LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0.csv",
        rsi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
