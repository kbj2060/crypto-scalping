"""Build the BTC 1h-native swing feature dataset for the btc_1h_native_swing_entry line.

Resamples the pinned 5m raw frame (data/splits/year_oos/btc_raw_frame_2024_2026.csv,
bar-open timestamps) into completed 1h bars and computes causal 1h features only from
bars that are fully closed at each row's own bar close. No forward-looking operations.

Output: data/splits/year_oos/btc_features_1h_swing_20260807.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_features_1h_swing_20260807.parquet"
BARS_PER_HOUR = 12  # complete 1h bucket requires all twelve 5m bars


def resample_1h(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.set_index("timestamp").sort_index()
    agg = raw.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        oi_value=("sum_open_interest_value", "last"),
        lsr_top=("sum_toptrader_long_short_ratio", "last"),
        funding=("last_funding_rate", "last"),
        eth_close=("close_btc", "last"),
        n_bars=("close", "size"),
    )
    complete = agg[agg["n_bars"] == BARS_PER_HOUR].drop(columns="n_bars")
    return complete


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=bars.index)
    close = bars["close"]
    log_close = np.log(close)
    ret_1h = log_close.diff()

    for horizon in (1, 3, 6, 12, 24, 48, 96, 168):
        out[f"ret_{horizon}h"] = log_close.diff(horizon)
    for window in (6, 24, 96, 168):
        out[f"vol_{window}h"] = ret_1h.rolling(window, min_periods=window).std()
    out["vol_ratio_6_96"] = out["vol_6h"] / out["vol_96h"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    out["rsi_14h"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist_1h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    mid = close.rolling(20, min_periods=20).mean()
    band = close.rolling(20, min_periods=20).std()
    out["bb_pctb_20h"] = (close - (mid - 2 * band)) / (4 * band).replace(0, np.nan)
    out["bb_bw_20h"] = (4 * band) / mid

    prev_close = close.shift()
    true_range = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr_pct_14h"] = true_range.rolling(14, min_periods=14).mean() / close

    for window in (24, 96, 168):
        out[f"dist_high_{window}h"] = close / bars["high"].rolling(window, min_periods=window).max() - 1
        out[f"dist_low_{window}h"] = close / bars["low"].rolling(window, min_periods=window).min() - 1

    vol_mean = bars["volume"].rolling(168, min_periods=168).mean()
    vol_std = bars["volume"].rolling(168, min_periods=168).std()
    out["volume_z_168h"] = (bars["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    trades_mean = bars["trades"].rolling(168, min_periods=168).mean()
    trades_std = bars["trades"].rolling(168, min_periods=168).std()
    out["trades_z_168h"] = (bars["trades"] - trades_mean) / trades_std.replace(0, np.nan)

    out["taker_buy_ratio_1h"] = bars["taker_buy_base"] / bars["volume"].replace(0, np.nan)
    out["taker_buy_ratio_24h"] = out["taker_buy_ratio_1h"].rolling(24, min_periods=24).mean()

    out["funding_last"] = bars["funding"]
    out["funding_mean_24h"] = bars["funding"].rolling(24, min_periods=24).mean()
    out["funding_std_168h"] = bars["funding"].rolling(168, min_periods=168).std()

    out["oi_chg_1h"] = bars["oi_value"].pct_change()
    out["oi_chg_24h"] = bars["oi_value"].pct_change(24)
    out["lsr_top_level"] = bars["lsr_top"]
    out["lsr_top_chg_24h"] = bars["lsr_top"].diff(24)

    eth_log = np.log(bars["eth_close"])
    out["eth_ret_1h"] = eth_log.diff()
    out["eth_ret_24h"] = eth_log.diff(24)
    out["eth_ret_168h"] = eth_log.diff(168)
    out["btc_eth_ret_spread_24h"] = out["ret_24h"] - out["eth_ret_24h"]

    hour = out.index.hour
    dow = out.index.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = bars[col]
    return out


def main() -> None:
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    bars = resample_1h(raw)
    features = build_features(bars)
    warm = features.dropna(subset=[c for c in features.columns if c not in ("funding_last",)])
    frame = warm.reset_index().rename(columns={"index": "timestamp"})
    assert frame["timestamp"].is_monotonic_increasing and frame["timestamp"].is_unique
    frame.to_parquet(OUT_PATH, index=False)
    print(f"rows={len(frame)} cols={len(frame.columns)} "
          f"span={frame['timestamp'].iloc[0]} .. {frame['timestamp'].iloc[-1]}")
    print(f"written: {OUT_PATH}")


if __name__ == "__main__":
    main()
