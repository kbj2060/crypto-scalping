"""
Full 1h feature pass for BTC: resample the raw 5m frame to 1h (same
resampling convention as build_1h_trendscan_dataset_btc_full_20260801.py's
resample_1h(), extended to cover all columns FeatureEngineer needs), then run
the SAME FeatureEngineer().process() pipeline used for the 5m base features
(build_btc_features_20260708.py) on the 1h-resampled frame -- giving the full
feature set (CVP, flow_whale, state_compression, volatility, trend_structure,
etc, not just the 11 trend-scan/RSI/vol columns previously merged) natively
at 1h resolution.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"

PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]


def resample_1h_full(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.set_index("timestamp").sort_index()
    sum_cols = ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "volume_btc", "quote_volume_btc"]
    last_cols = ["last_funding_rate", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                 "count_long_short_ratio", "close_btc"]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    agg.update({c: "sum" for c in sum_cols if c in f.columns})
    agg.update({c: "last" for c in last_cols if c in f.columns})
    r = f.resample("1h", label="left", closed="left").agg(agg)
    r = r.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return r


def main():
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    missing = [c for c in PRIMARY_RAW_COLS + CROSS_RAW_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing required raw columns: {missing}")

    hourly = resample_1h_full(raw[list(dict.fromkeys(PRIMARY_RAW_COLS + CROSS_RAW_COLS))])
    print(f"1h resampled: {len(hourly)} rows, {hourly['timestamp'].iloc[0]}..{hourly['timestamp'].iloc[-1]}")

    primary_df = hourly[PRIMARY_RAW_COLS].copy()
    cross_df = hourly[CROSS_RAW_COLS].copy()

    fe = FeatureEngineer()
    features = fe.process(primary_df, cross_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"1h engineered features: {len(features)} rows, {len(features.columns)} columns")

    features.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
