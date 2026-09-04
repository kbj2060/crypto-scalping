#!/usr/bin/env python3
"""Re-测 the liquidation-map proximity question properly: how close is the sweep's own WICK
EXTREME (the low for a downside sweep, the high for an upside sweep -- the point that actually
pierces through) to the nearest same-side compute_spliced_levels() level, not the post-reclaim
close used by build_eth_5m_sweep_v_rebound_features_tier1_20260829.py's t1_liqmap_relevant_side_dist_atr.

Also reports the ORIGINAL near/mid/far in-window-tertile framing from the 2026-08-27 confluence
research (eth_evidence_signal_liquidation_confluence_20260827.md), which found a real effect
using tertiles rather than a raw continuous distance -- tests whether that functional form does
better than the continuous ATR-distance version already tried.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from live_liquidation_map_20260824 import compute_spliced_levels  # noqa: E402

SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
FEATURES_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0_tier1.csv"


def main() -> int:
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    raw = pd.read_csv(SOURCE, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    hourly = raw.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()

    dist_atr = np.full(len(df), np.nan)
    dist_pct = np.full(len(df), np.nan)
    n_levels_side = np.zeros(len(df), dtype=int)

    # low/high aren't in the tier0/tier1 features file -- pull them from the raw 5m source directly
    raw5 = raw.reset_index()[["timestamp", "low", "high"]]
    df2 = df.merge(raw5, on="timestamp", how="left")

    for i, row in df2.iterrows():
        ts, side, atr = row["timestamp"], row["side"], row["atr"]
        last_closed_hour_start = ts.floor("h") - pd.Timedelta(hours=1)
        window = hourly[hourly["timestamp"] <= last_closed_hour_start].tail(24)
        if len(window) < 20 or not np.isfinite(atr) or atr <= 0:
            continue
        extreme_price = float(row["low"]) if side == "downside" else float(row["high"])
        levels = compute_spliced_levels(window, extreme_price)
        if not levels.get("warmed_up"):
            continue
        relevant = levels["support_levels"] if side == "downside" else levels["resistance_levels"]
        n_levels_side[i] = len(relevant)
        if not relevant:
            continue
        nearest = min(relevant, key=lambda lv: abs(lv["distance_pct"]))
        dist_pct[i] = abs(nearest["distance_pct"])
        dist_atr[i] = abs(nearest["distance_pct"]) / 100.0 * extreme_price / atr

    df2["dist_atr_from_extreme"] = dist_atr
    df2["dist_pct_from_extreme"] = dist_pct
    df2["n_relevant_levels"] = n_levels_side

    print(f"NaN (no level found on relevant side): {np.isnan(dist_atr).sum()} ({100*np.isnan(dist_atr).mean():.1f}%)")
    print(f"n_relevant_levels distribution:\n{pd.Series(n_levels_side).value_counts().sort_index()}\n")
    valid = df2.dropna(subset=["dist_atr_from_extreme"])
    print("dist_atr_from_extreme describe:")
    print(valid["dist_atr_from_extreme"].describe())
    print()
    for thresh in [0.1, 0.25, 0.5, 1.0, 2.0]:
        print(f"  fraction with extreme within {thresh} ATR of nearest level: {100*(valid['dist_atr_from_extreme']<=thresh).mean():.1f}%")
    print()
    for thresh_pct in [0.1, 0.25, 0.5, 1.0]:
        print(f"  fraction with extreme within {thresh_pct}% price of nearest level: {100*(valid['dist_pct_from_extreme']<=thresh_pct).mean():.1f}%")

    print("\ncorr(dist_atr_from_extreme, label) =", round(valid[["dist_atr_from_extreme", "label"]].corr().iloc[0, 1], 4))

    print("\nV_REBOUND rate by tertile of dist_atr_from_extreme (in-sample tertile, descriptive):")
    valid = valid.copy()
    valid["tertile"] = pd.qcut(valid["dist_atr_from_extreme"], 3, labels=["near", "mid", "far"])
    print(valid.groupby("tertile", observed=True)["label"].agg(["mean", "count"]).round(4))

    out_path = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/liqmap_extreme_proximity_diagnostic.csv"
    df2.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
