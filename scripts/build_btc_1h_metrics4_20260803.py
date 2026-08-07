"""
metrics4 features computed NATIVELY at 1h resolution (user's hypothesis:
5m-level taker/toptrader-ratio noise may wash out real signal that survives
at 1h aggregation -- metrics4 failed decisively at 5m both in h48qual
(project-btc-metrics4-features-standalone-weak-20260802) and in this
session's cluster-importance audit, negative R2 contribution).

Reuses the EXACT derivation formulas from build_btc_features_metrics4_20260802.py
(_zscore, _sig_whale, _sig_oi_divergence), applied to a 1h-resampled raw frame
+ 1h FeatureEngineer pass (whale_retail_ratio/whale_conviction/oi_change_rate/
log_return/trade_intensity all recomputed natively at 1h, not just resampled
from 5m z-scores).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.engineering import FeatureEngineer  # noqa: E402

RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_metrics4_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_1h_metrics4_2024_2026.csv"

PRIMARY_RAW_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
                     "trades", "taker_buy_base", "taker_buy_quote",
                     "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio", "last_funding_rate"]
CROSS_RAW_COLS = ["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]
NEW_RAW_COLS = ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio"]

ZWIN_1H = 24  # 1 day of 1h bars (matches the 5m version's 288 = 1 day of 5m bars)


def _zscore(s: pd.Series, window: int = ZWIN_1H) -> pd.Series:
    mean = s.rolling(window=window, min_periods=1).mean()
    std = s.rolling(window=window, min_periods=1).std().replace(0, 1e-8)
    return ((s - mean) / std).fillna(0)


def _sig_whale(df: pd.DataFrame) -> pd.Series:
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


def resample_1h(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f.set_index("timestamp").sort_index()
    sum_cols = ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "volume_btc", "quote_volume_btc"]
    last_cols = ["last_funding_rate", "sum_open_interest_value", "sum_toptrader_long_short_ratio",
                 "count_long_short_ratio", "close_btc"]
    mean_cols = ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio"]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    agg.update({c: "sum" for c in sum_cols if c in f.columns})
    agg.update({c: "last" for c in last_cols if c in f.columns})
    agg.update({c: "mean" for c in mean_cols if c in f.columns})
    r = f.resample("1h", label="left", closed="left").agg(agg)
    r = r.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return r


def main():
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    cols = list(dict.fromkeys(PRIMARY_RAW_COLS + CROSS_RAW_COLS + NEW_RAW_COLS))
    hourly = resample_1h(raw[cols])
    print(f"1h resampled: {len(hourly)} rows")

    fe = FeatureEngineer()
    features = fe.process(hourly[PRIMARY_RAW_COLS].copy(), hourly[CROSS_RAW_COLS].copy())
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.merge(hourly[["timestamp"] + NEW_RAW_COLS], on="timestamp", how="left")
    features = features.sort_values("timestamp").reset_index(drop=True)

    features["taker_vol_ratio_z"] = _zscore(features["sum_taker_long_short_vol_ratio"])
    features["count_toptrader_ratio_z"] = _zscore(features["count_toptrader_long_short_ratio"])
    features["toptrader_count_size_divergence"] = features["count_toptrader_ratio_z"] - _zscore(features["sum_toptrader_long_short_ratio"])
    features["sig_whale"] = _sig_whale(features)
    features["sig_oi_divergence"] = _sig_oi_divergence(features)

    out_cols = ["timestamp", "sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio",
                "taker_vol_ratio_z", "count_toptrader_ratio_z", "toptrader_count_size_divergence",
                "sig_whale", "sig_oi_divergence"]
    features[out_cols].to_csv(OUT_PATH, index=False)
    print(f"1h metrics4: {len(features)} rows, wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
