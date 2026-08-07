"""Build a continuous, causal 1h ETH OHLCV+feature dataset for the PatchTST standalone
1h entry-signal architecture (new candidate to complement live Omega4.6.1, per user request
2026-07-31 -- an intentionally different architecture family from the tree/HGB-based
Sigma3-1h/Sigma6 lines, which were both closed as un-retrainable/VAL-overfit).

Deliberately price/volume-only (no funding/OI/cross-asset features): this project's history
shows richer feature sets (125 secondary features on Sigma3-1h) made fold-instability WORSE,
not better, so v1 here stays minimal to isolate whether the architecture itself adds anything.

All features are computed directly on 1h-resampled OHLCV (proper open/high/low/close/volume
aggregation, not last-5m-value sampling) so window multiples are meaningful at 1h granularity.
Concatenates the full available history before resampling (no per-year cold-start reset, unlike
the original build_1h_trendscan_dataset_20260705.py which had this bug, fixed in
build_1h_trendscan_dataset_continuous_20260801.py for the Sigma3-1h line).

Output: data/research/eth_1h_patchtst_dataset_20260731.parquet
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KLINES_1M = os.path.join(_ROOT, "binance_data", "klines", "ETHUSDT", "ETHUSDT-1m-api.csv")
OUT_PATH = os.path.join(_ROOT, "data", "research", "eth_1h_patchtst_dataset_20260731.parquet")

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "atr14_pct", "realized_vol_24", "vwap_dev_48", "volume_z_48",
    "upper_wick_ratio", "lower_wick_ratio", "compression_ratio",
    "hour_sin", "hour_cos",
]


def resample_1h(kl_1m: pd.DataFrame) -> pd.DataFrame:
    kl = kl_1m.set_index("timestamp")
    agg = kl.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "volume": "sum", "quote_volume": "sum", "trades": "sum"}
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    return agg


def add_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    for w in [1, 3, 6, 12, 24]:
        df[f"ret_{w}"] = np.log(c / c.shift(w))

    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df["atr14_pct"] = (tr.rolling(14, min_periods=7).mean() / c)

    ret_1h = np.log(c / c.shift(1))
    df["realized_vol_24"] = ret_1h.rolling(24, min_periods=12).std()

    dollar_vol = c * v
    vwap_48 = dollar_vol.rolling(48, min_periods=24).sum() / v.rolling(48, min_periods=24).sum().replace(0, np.nan)
    df["vwap_dev_48"] = c / vwap_48 - 1.0

    vol_mu_48 = v.rolling(48, min_periods=24).mean()
    vol_sd_48 = v.rolling(48, min_periods=24).std()
    df["volume_z_48"] = (v - vol_mu_48) / vol_sd_48.replace(0, np.nan)

    rng = (h - l).replace(0, np.nan)
    df["upper_wick_ratio"] = (h - np.maximum(c, df["open"])) / rng
    df["lower_wick_ratio"] = (np.minimum(c, df["open"]) - l) / rng

    range_6 = (h.rolling(6, min_periods=3).max() - l.rolling(6, min_periods=3).min())
    range_48 = (h.rolling(48, min_periods=24).max() - l.rolling(48, min_periods=24).min()).replace(0, np.nan)
    df["compression_ratio"] = range_6 / range_48

    hour_of_day = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)

    return df


def main() -> None:
    print("Loading 1m klines...")
    kl_1m = pd.read_csv(KLINES_1M, parse_dates=["timestamp"],
                         usecols=["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"])
    print(f"  {len(kl_1m):,} rows, {kl_1m['timestamp'].min()} -> {kl_1m['timestamp'].max()}")

    bars = resample_1h(kl_1m)
    print(f"1h bars: {len(bars):,}, {bars.index.min()} -> {bars.index.max()}")

    df = add_features(bars)
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"after dropna on {len(FEATURE_COLS)} features: {len(df):,} rows (dropped {n_before - len(df):,} warmup rows)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.reset_index().rename(columns={"index": "timestamp"}).to_parquet(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}")
    print(df[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
