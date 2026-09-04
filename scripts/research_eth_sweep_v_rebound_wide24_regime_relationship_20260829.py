#!/usr/bin/env python3
"""Empirically test whether the wide24 HMM regime (regime3_current) carries any
relationship with the V_REBOUND label, as one candidate input to compare against GBM3.

Joins the confirmed regime3_current_sensitive_wide24_{bull,bear,chop}_prob columns
(2024-01-01..2026-08-19, three year-split files concatenated) onto the 14,259 sweep
events by timestamp, then checks V_REBOUND rate by dominant regime and by a
direction-relative "regime supports the rebound direction" framing (bull_prob for a
downside sweep, bear_prob for an upside sweep -- rebound direction is opposite the sweep).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIME_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
REGIME_FILES = [
    REGIME_DIR / "training_features_2024_regime3_current_sensitive_hmm_wide24.csv",
    REGIME_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
    REGIME_DIR / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
]
REGIME_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"


def load_regime() -> pd.DataFrame:
    frames = []
    for path in REGIME_FILES:
        df = pd.read_csv(path, usecols=["timestamp"] + REGIME_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")


def main() -> int:
    regime = load_regime()
    print(f"regime rows: {len(regime)}  range: {regime['timestamp'].min()} .. {regime['timestamp'].max()}")

    labels = pd.read_csv(LABEL_CSV)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    merged = labels.merge(regime, on="timestamp", how="left")
    matched = merged[REGIME_COLS[0]].notna()
    print(f"label events: {len(labels)}  matched to regime: {int(matched.sum())} ({100*matched.mean():.1f}%)")
    merged = merged[matched].copy()

    merged["dominant"] = merged[REGIME_COLS].idxmax(axis=1).str.replace(
        "regime3_current_sensitive_wide24_", "").str.replace("_prob", "")

    print("\n=== V_REBOUND rate by dominant regime x side ===")
    print(merged.groupby(["dominant", "side"])["label"].agg(["mean", "count"]).round(4))

    print("\n=== V_REBOUND rate by dominant regime (both sides pooled) ===")
    print(merged.groupby("dominant")["label"].agg(["mean", "count"]).round(4))

    merged["rebound_aligned_prob"] = np.where(
        merged["side"] == "downside",
        merged["regime3_current_sensitive_wide24_bull_prob"],
        merged["regime3_current_sensitive_wide24_bear_prob"],
    )
    merged["against_trend"] = merged["rebound_aligned_prob"] >= 0.5
    print("\n=== V_REBOUND rate: is the sweep AGAINST the dominant trend (rebound-aligned prob >= 0.5)? ===")
    print(merged.groupby(["against_trend", "side"])["label"].agg(["mean", "count"]).round(4))
    print("\npooled:")
    print(merged.groupby("against_trend")["label"].agg(["mean", "count"]).round(4))

    corr = merged[["rebound_aligned_prob", "label"]].corr().iloc[0, 1]
    chop_corr = merged[["regime3_current_sensitive_wide24_chop_prob", "label"]].corr().iloc[0, 1]
    print(f"\npoint-biserial corr(rebound_aligned_prob, label) = {corr:.4f}")
    print(f"point-biserial corr(chop_prob, label) = {chop_corr:.4f}")

    print("\n=== V_REBOUND rate by chop_prob quintile ===")
    merged["chop_q"] = pd.qcut(merged["regime3_current_sensitive_wide24_chop_prob"], 5, duplicates="drop")
    print(merged.groupby("chop_q")["label"].agg(["mean", "count"]).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
