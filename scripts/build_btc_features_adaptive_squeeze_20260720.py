"""Rebuild BTC's feature frame with FeatureEngineer(adaptive_squeeze=True) -- the only change vs
build_btc_features_20260708.py. Writes to a SEPARATE directory
(data/splits/year_oos_adaptive_squeeze_btc_20260720/) so the existing live-model-supporting
btc_features_*.csv files are untouched.

Motivation: SOL's zig075 v2 fix (docs/model_contracts/sol_adaptive_squeeze_v2_20260720.md) found
the fixed 0.0002 funding-rate divisor in FundingRateMomentum was ETH-calibrated and hurt SOL, whose
funding-rate std is ~3.5x ETH's. BTC's own funding-rate std (0.000041) is already close to ETH's
(0.000044), so the statistical argument says this fix shouldn't matter for BTC -- this script
exists to test that empirically via a real fresh-forward retrain rather than rely on the
distributional argument alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"
OUT_DIR = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720"
OUT_COMBINED = OUT_DIR / "btc_features_2024_2026.csv"

PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    print(f"BTC raw frame: {len(raw)} rows {raw['timestamp'].iloc[0]}..{raw['timestamp'].iloc[-1]}", flush=True)

    missing = [c for c in PRIMARY_RAW_COLS + CROSS_RAW_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing required raw columns: {missing}")

    primary_df = raw[PRIMARY_RAW_COLS].copy()
    cross_df = raw[CROSS_RAW_COLS].copy()

    fe = FeatureEngineer(adaptive_squeeze=True)
    features = fe.process(primary_df, cross_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"\nBTC engineered features (adaptive_squeeze): {len(features)} rows, {len(features.columns)} columns", flush=True)

    n_all_nan_cols = int((features.isna().all(axis=0)).sum())
    print(f"columns that are entirely NaN: {n_all_nan_cols}", flush=True)
    if n_all_nan_cols:
        print([c for c in features.columns if features[c].isna().all()], flush=True)

    features.to_csv(OUT_COMBINED, index=False)
    print(f"\nWrote {OUT_COMBINED}", flush=True)

    for year in (2024, 2025, 2026):
        seg = features[features["timestamp"].dt.year == year].reset_index(drop=True)
        if seg.empty:
            continue
        out = OUT_DIR / f"btc_features_{year}.csv"
        seg.to_csv(out, index=False)
        print(f"{year}: {len(seg)} rows {seg['timestamp'].iloc[0]}..{seg['timestamp'].iloc[-1]} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
