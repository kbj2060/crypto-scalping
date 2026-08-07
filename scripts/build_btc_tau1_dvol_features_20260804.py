"""Build the BTC counterpart of ETH Tau1's 1h feature contract plus causal Deribit DVOL.

The ETH Tau1 contract has 38 features.  Four ETH-specific cross-asset fields are made
explicitly symmetric for BTC: ETH is the external asset and the spread is BTC minus ETH.
DVOL is available only one hour after a Deribit candle timestamp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_1h_trendscan_dataset_btc_full_20260801 as tau1  # noqa: E402

BTC_SOURCES = [ROOT / f"data/splits/year_oos/btc_features_{year}.csv" for year in (2024, 2025, 2026)]
ETH_SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
DVOL_SOURCE = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_tau1_dvol_features_20260804.parquet"

TAU1_BTC_FEATURE_COLS = [
    "logret_1", "logret_2", "logret_3", "logret_6", "logret_12", "logret_24",
    "rvol_6", "rvol_12", "rvol_24", "rvol_48", "atr_pct", "rsi_14", "macd_hist",
    "bb_width", "bb_pos", "eth_logret_1", "eth_logret_6", "eth_logret_24",
    "btc_eth_spread_6", "funding", "funding_z_48", "funding_roc_6", "oi_change_6",
    "oi_z_48", "toptrader_z_48", "vol_z_48", "taker_imb", "body_ratio", "upper_wick",
    "lower_wick", "skew_24", "kurt_24", "dist_sma50", "hurst_proxy", "hour_sin",
    "hour_cos", "dow_sin", "dow_cos", "dvol_btc",
]


def _load_btc() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_base", "last_funding_rate", "sum_open_interest_value",
            "sum_toptrader_long_short_ratio"]
    frame = pd.concat([pd.read_csv(path, usecols=cols) for path in BTC_SOURCES], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def _load_eth_close() -> pd.DataFrame:
    eth = pd.read_csv(ETH_SOURCE, usecols=["timestamp", "close"])
    eth["timestamp"] = pd.to_datetime(eth["timestamp"])
    eth = eth.set_index("timestamp").sort_index()["close"].resample("1h", label="left", closed="left").last()
    return eth.rename("close_btc").reset_index()


def _load_dvol() -> pd.DataFrame:
    dvol = pd.read_csv(DVOL_SOURCE, usecols=["timestamp", "close"])
    dvol["timestamp"] = pd.to_datetime(dvol["timestamp"]) + pd.Timedelta(hours=1)
    return dvol.rename(columns={"close": "dvol_btc"}).sort_values("timestamp")


def build_features() -> pd.DataFrame:
    btc_1h = tau1.resample_1h(_load_btc())
    btc_1h = btc_1h.merge(_load_eth_close(), on="timestamp", how="left", validate="one_to_one")
    features = tau1.compute_features(btc_1h).rename(columns={
        "btc_logret_1": "eth_logret_1", "btc_logret_6": "eth_logret_6",
        "btc_logret_24": "eth_logret_24", "eth_btc_spread_6": "btc_eth_spread_6",
    })
    features = pd.merge_asof(features.sort_values("timestamp"), _load_dvol(), on="timestamp", direction="backward")
    missing = [col for col in TAU1_BTC_FEATURE_COLS if col not in features]
    if missing:
        raise RuntimeError(f"Tau1 BTC feature contract incomplete: {missing}")
    return features[["timestamp", "open", "high", "low", "close", *TAU1_BTC_FEATURE_COLS]]


def main() -> int:
    features = build_features()
    features.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} rows={len(features)} n_features={len(TAU1_BTC_FEATURE_COLS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
