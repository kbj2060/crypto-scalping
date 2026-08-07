"""BTC replication, step 3: run the identical FeatureEngineer().process() pipeline on BTC's merged
raw frame (ETH as the cross-asset reference, see build_btc_raw_frame_20260708.py). Mirrors
build_sol_features_20260707.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"

PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]


def main() -> int:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    print(f"BTC raw frame: {len(raw)} rows {raw['timestamp'].iloc[0]}..{raw['timestamp'].iloc[-1]}", flush=True)

    missing = [c for c in PRIMARY_RAW_COLS + CROSS_RAW_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing required raw columns: {missing}")

    primary_df = raw[PRIMARY_RAW_COLS].copy()
    cross_df = raw[CROSS_RAW_COLS].copy()

    fe = FeatureEngineer()
    features = fe.process(primary_df, cross_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"\nBTC engineered features: {len(features)} rows, {len(features.columns)} columns", flush=True)
    print(f"range: {features['timestamp'].iloc[0]}..{features['timestamp'].iloc[-1]}", flush=True)

    n_all_nan_cols = int((features.isna().all(axis=0)).sum())
    print(f"columns that are entirely NaN: {n_all_nan_cols}", flush=True)
    if n_all_nan_cols:
        print([c for c in features.columns if features[c].isna().all()], flush=True)

    features.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
