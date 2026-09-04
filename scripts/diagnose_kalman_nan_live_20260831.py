#!/usr/bin/env python3
"""Diagnose why kalman_deviation_meanrev's live model_proba is showing None (persisting across
multiple polls, not resolving) -- check exactly which feature(s) are NaN in the current live
indicator_frame's last row, and how many bars are actually available."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_candidate_pool_raw_lift_check_20260831 import kalman_level_and_velocity, rolling_zscore
from live_evidence_signal_dashboard_20260823 import fetch_klines
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS, build_indicator_frame

df = fetch_klines()
print(f"{len(df)} bars fetched, latest={df['timestamp'].iloc[-1]}")

indicator_frame = build_indicator_frame(df)
indicator_frame["dem"] = compute_demarker(df["high"], df["low"]).to_numpy()
levels, _ = kalman_level_and_velocity(df["close"].to_numpy())
kalman_dev = (df["close"].to_numpy() - levels) / levels
indicator_frame["kalman_dev_z"] = rolling_zscore(__import__("pandas").Series(kalman_dev)).to_numpy()

row = indicator_frame.iloc[-1]
all_cols = [c for c in FEATURE_COLUMNS if c != "is_bottom"] + ["dem", "kalman_dev_z"]
print("\nNaN check on last row, per feature:")
for c in all_cols:
    v = row[c]
    print(f"  {c:>22s} = {v!r}" + ("  <-- NaN!" if __import__("pandas").isna(v) else ""))
