"""BTC metrics4 experiment, step 2: run the identical FeatureEngineer().process() pipeline BTC's
baseline uses (mirrors build_btc_features_20260708.py exactly, same PRIMARY_RAW_COLS/CROSS_RAW_COLS,
same FeatureEngineer -- this reproduces the baseline feature set bit-for-bit as a sanity check), then
APPENDS 4 new engineered features computed from the 2 new raw metrics columns
(sum_taker_long_short_vol_ratio, count_toptrader_long_short_ratio) sourced from
btc_raw_frame_metrics4_2024_2026.csv:

  1. taker_vol_ratio_z          -- rolling(288) z-score of sum_taker_long_short_vol_ratio
                                    (exchange-wide taker buy/sell vol ratio; distinct from the
                                    already-used net_taker_ratio which is derived from kline
                                    taker_buy_quote/quote_volume)
  2. count_toptrader_ratio_z    -- rolling(288) z-score of count_toptrader_long_short_ratio
                                    (top-trader ACCOUNT-COUNT ratio; distinct from already-used
                                    sum_toptrader_long_short_ratio which is POSITION-SIZE ratio)
  3. toptrader_count_size_divergence -- count_toptrader_ratio_z - z-score(sum_toptrader_long_short_ratio),
                                    the count-vs-size gap (informative part per task spec)
  4. sig_whale                  -- EXACT formula from strategies/elite_strategies.py
                                    WhaleSentimentDivergence.generate_signal, vectorized, using
                                    already-engineered whale_retail_ratio/whale_conviction/close
  5. sig_oi_divergence          -- EXACT formula from strategies/elite_strategies.py
                                    OITrendDivergence.generate_signal, vectorized, using
                                    already-engineered oi_change_rate/log_return/trade_intensity

(5) and (4) count as the task's items 3-4 ("sig_whale and sig_oi_divergence"); (1)-(3) cover items
1-2 (taker vol ratio, count-vs-size divergence). Writes to an ISOLATED path -- does NOT overwrite
data/splits/year_oos/btc_features_2024_2026.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_metrics4_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026_metrics4_20260802.csv"
BASELINE_FEATURES_PATH = ROOT / "data/splits/year_oos/btc_features_2024_2026.csv"

# Identical to build_btc_features_20260708.py -- keeps the baseline FeatureEngineer contract untouched.
PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
NEW_RAW_COLS = ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio"]

ZWIN = 288  # 1 day of 5m bars, matches amihud_illiquidity_z convention in features/engineering.py


def _zscore(s: pd.Series, window: int = ZWIN) -> pd.Series:
    mean = s.rolling(window=window, min_periods=1).mean()
    std = s.rolling(window=window, min_periods=1).std().replace(0, 1e-8)
    return ((s - mean) / std).fillna(0)


def _sig_whale(df: pd.DataFrame) -> pd.Series:
    """Vectorized replica of strategies/elite_strategies.py WhaleSentimentDivergence.generate_signal."""
    ratio = df["whale_retail_ratio"].astype(float)
    conviction = df["whale_conviction"].astype(float)
    close = df["close"].astype(float)
    price_dir = np.sign(close.diff()).fillna(0.0)
    whale_strength = (ratio - 1.48) * 5.0
    whale_dir = whale_strength * (1.0 + conviction.abs())
    disagree = (price_dir * whale_dir) < 0
    sig = np.where(disagree, whale_dir.clip(-1, 1), (whale_dir * 0.3).clip(-1, 1))
    sig = pd.Series(sig, index=df.index)
    sig.iloc[0] = 0.0
    return sig.fillna(0.0)


def _sig_oi_divergence(df: pd.DataFrame) -> pd.Series:
    """Vectorized replica of strategies/elite_strategies.py OITrendDivergence.generate_signal."""
    oi_change = df["oi_change_rate"].astype(float)
    log_ret = df["log_return"].astype(float)
    trade_int = df["trade_intensity"].astype(float)

    active = oi_change.abs() > 0.002
    case_short_squeeze = active & (log_ret < -0.0005) & (oi_change > 0)
    case_long_squeeze = active & (log_ret > 0.0005) & (oi_change > 0)
    case_other = active & ~case_short_squeeze & ~case_long_squeeze

    sig = pd.Series(0.0, index=df.index)
    sig[case_short_squeeze] = (0.5 * (oi_change * 100.0) * trade_int)[case_short_squeeze].clip(0, 1)
    sig[case_long_squeeze] = (-0.5 * (oi_change * 100.0) * trade_int)[case_long_squeeze].clip(-1, 0)
    sig[case_other] = np.sign(log_ret[case_other]) * 0.2
    return sig.fillna(0.0)


def main() -> int:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    print(f"BTC raw frame (metrics4): {len(raw)} rows {raw['timestamp'].iloc[0]}..{raw['timestamp'].iloc[-1]}", flush=True)

    missing = [c for c in PRIMARY_RAW_COLS + CROSS_RAW_COLS + NEW_RAW_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"missing required raw columns: {missing}")

    primary_df = raw[PRIMARY_RAW_COLS].copy()
    cross_df = raw[CROSS_RAW_COLS].copy()

    fe = FeatureEngineer()
    features = fe.process(primary_df, cross_df)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    print(f"BTC baseline-contract engineered features: {len(features)} rows, {len(features.columns)} columns", flush=True)

    # --- sanity check: baseline reproduction bit-for-bit (if baseline file exists) ---
    if BASELINE_FEATURES_PATH.exists():
        baseline = pd.read_csv(BASELINE_FEATURES_PATH, low_memory=False)
        baseline["timestamp"] = pd.to_datetime(baseline["timestamp"])
        common_cols = [c for c in features.columns if c in baseline.columns]
        merged = features[["timestamp"] + [c for c in common_cols if c != "timestamp"]].merge(
            baseline[["timestamp"] + [c for c in common_cols if c != "timestamp"]],
            on="timestamp", suffixes=("_new", "_base"), how="inner")
        n_mismatch_cols = 0
        for c in common_cols:
            if c == "timestamp":
                continue
            a = merged[f"{c}_new"]
            b = merged[f"{c}_base"]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                diff = (a - b).abs()
                bad = (diff > 1e-6) & ~(a.isna() & b.isna())
                if bad.sum() > 0:
                    n_mismatch_cols += 1
                    print(f"  MISMATCH col={c} n_bad={bad.sum()} max_diff={diff.max()}", flush=True)
        print(f"Baseline reproduction check: {len(common_cols)} common cols, {n_mismatch_cols} with mismatches, "
              f"{len(merged)} overlapping rows", flush=True)
    else:
        print("Baseline features file not found, skipping reproduction check.", flush=True)

    # --- append raw new metrics (ffilled onto engineered frame via timestamp merge) ---
    new_raw = raw[["timestamp"] + NEW_RAW_COLS].copy()
    features = features.merge(new_raw, on="timestamp", how="left")
    features[NEW_RAW_COLS] = features[NEW_RAW_COLS].ffill()

    # --- 4 new engineered features ---
    features["taker_vol_ratio_z"] = _zscore(features["sum_taker_long_short_vol_ratio"])
    features["count_toptrader_ratio_z"] = _zscore(features["count_toptrader_long_short_ratio"])
    sum_toptrader_z = _zscore(features["sum_toptrader_long_short_ratio"])
    features["toptrader_count_size_divergence"] = features["count_toptrader_ratio_z"] - sum_toptrader_z
    features["sig_whale"] = _sig_whale(features)
    features["sig_oi_divergence"] = _sig_oi_divergence(features)

    new_cols = ["taker_vol_ratio_z", "count_toptrader_ratio_z", "toptrader_count_size_divergence",
                "sig_whale", "sig_oi_divergence"]
    print("\nNew feature summary stats:")
    print(features[new_cols].describe())

    features.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(features)} rows, {len(features.columns)} columns", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
