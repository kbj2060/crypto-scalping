"""Rebuild SOL's feature frame with the new session-open dummy features
(session_europe, session_japan, session_{europe,us,japan}_open, added to
features/engineering.py and features/schema.py 2026-07-30) on top of the
adaptive_squeeze base (same as build_sol_features_adaptive_squeeze_20260720.py).

Writes to a SEPARATE directory so the existing live-model-supporting
data/splits/year_oos_adaptive_squeeze_sol_20260720/ files are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/sol_raw_frame_2024_2026.csv"
OUT_DIR = ROOT / "data/splits/session_open_dummy_sol_20260730"
OUT_COMBINED = OUT_DIR / "sol_features_2024_2026.csv"

ETH_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                "trades", "taker_buy_base", "taker_buy_quote",
                "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                "count_long_short_ratio", "last_funding_rate"]
BTC_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    print(f"SOL raw frame: {len(raw)} rows {raw['timestamp'].iloc[0]}..{raw['timestamp'].iloc[-1]}", flush=True)

    missing = [c for c in ETH_RAW_COLS + BTC_RAW_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing required raw columns: {missing}")

    sol_df = raw[ETH_RAW_COLS].copy()
    btc_df = raw[BTC_RAW_COLS].copy()

    fe = FeatureEngineer(adaptive_squeeze=True)
    features = fe.process(sol_df, btc_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"\nSOL engineered features (session_open): {len(features)} rows, {len(features.columns)} columns", flush=True)

    new_cols = ["session_europe", "session_japan", "session_europe_open", "session_us_open", "session_japan_open"]
    missing_new = [c for c in new_cols if c not in features.columns]
    if missing_new:
        raise RuntimeError(f"new session-open columns missing after pruning: {missing_new}")

    n_all_nan_cols = int((features.isna().all(axis=0)).sum())
    print(f"columns that are entirely NaN: {n_all_nan_cols}", flush=True)

    features.to_csv(OUT_COMBINED, index=False)
    print(f"\nWrote {OUT_COMBINED}", flush=True)

    for year in (2024, 2025, 2026):
        seg = features[features["timestamp"].dt.year == year].reset_index(drop=True)
        if seg.empty:
            continue
        out = OUT_DIR / f"sol_features_{year}.csv"
        seg.to_csv(out, index=False)
        print(f"{year}: {len(seg)} rows {seg['timestamp'].iloc[0]}..{seg['timestamp'].iloc[-1]} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
