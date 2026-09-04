#!/usr/bin/env python3
"""Tier 1 candidate features for the liquidity_sweep -> V_REBOUND model, added incrementally
on top of the Tier 0 feature set (docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md).

Funding/OI/BTC-cross-asset columns are pulled directly from the canonical
data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv files (2024-01-01..2026-08-19,
already causally engineered by features/engineering.py and used live by GBM3/GBM2 -- reused, not
rebuilt from raw sources). The liquidation-map proximity feature has no such precomputed column
(it is a live-dashboard function, not part of the standard feature pipeline) and is built here by
calling compute_spliced_levels() unmodified over a causal 24-hourly-bar resample.

Groups (each independently toggleable in the training script, per-group prefix):
  t1_funding_*  <- last_funding_rate, funding_z_score, funding_abs
  t1_oi_*       <- oi_change_rate
  t1_btc_*      <- btc_ret_z_48, eth_btc_ret_spread_12, btc_volume_impulse_z, btc_corr_60
  t1_liqmap_*   <- distance (in ATR units) from the sweep bar's close to the nearest same-side
                   (support for downside sweeps, resistance for upside) compute_spliced_levels()
                   level, causal 24x1h window ending at the last fully-closed hour before the
                   sweep bar (matches the live dashboard's own convention)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402

TIER0_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
TRAINING_FEATURES = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0_tier1.csv"

FUNDING_COLS = ["last_funding_rate", "funding_z_score", "funding_abs"]
OI_COLS = ["oi_change_rate"]
BTC_COLS = ["btc_ret_z_48", "eth_btc_ret_spread_12", "btc_volume_impulse_z", "btc_corr_60"]


def load_training_features_slice() -> pd.DataFrame:
    frames = []
    for path in TRAINING_FEATURES:
        df = pd.read_csv(path, usecols=["timestamp"] + FUNDING_COLS + OI_COLS + BTC_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")


def build_liqmap_feature(events: pd.DataFrame) -> pd.Series:
    raw = pd.read_csv(SOURCE, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    hourly = raw.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    hourly_reset = hourly.reset_index()

    out = np.full(len(events), np.nan)
    for i, (ts, side, atr) in enumerate(zip(events["timestamp"], events["side"], events["atr"])):
        last_closed_hour_start = ts.floor("h") - pd.Timedelta(hours=1)
        window = hourly_reset[hourly_reset["timestamp"] <= last_closed_hour_start].tail(24)
        if len(window) < 20 or not np.isfinite(atr) or atr <= 0:
            continue
        current_price = float(window["close"].iloc[-1])
        levels = compute_spliced_levels(window, current_price)
        if not levels.get("warmed_up"):
            continue
        relevant = levels["support_levels"] if side == "downside" else levels["resistance_levels"]
        if not relevant:
            continue
        nearest = min(relevant, key=lambda lv: abs(lv["distance_pct"]))
        out[i] = abs(nearest["distance_pct"]) / 100.0 * current_price / atr
    return pd.Series(out, index=events.index)


def main() -> int:
    tier0 = pd.read_csv(TIER0_CSV)
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)

    training = load_training_features_slice()
    merged = tier0.merge(training, on="timestamp", how="left")
    rename = {c: f"t1_funding_{c}" for c in FUNDING_COLS}
    rename.update({c: f"t1_oi_{c}" for c in OI_COLS})
    rename.update({c: f"t1_btc_{c}" for c in BTC_COLS})
    merged = merged.rename(columns=rename)

    print("building liquidation-map proximity feature (loops over events, ~1 min)...")
    merged["t1_liqmap_relevant_side_dist_atr"] = build_liqmap_feature(merged)

    new_cols = list(rename.values()) + ["t1_liqmap_relevant_side_dist_atr"]
    coverage = merged[new_cols].notna().mean().round(4)
    print("coverage (fraction non-NaN) per new Tier1 column:")
    print(coverage.to_string())

    merged.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} shape={merged.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
