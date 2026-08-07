"""Phase A pilot (SOL), step 3: merge SOL's raw 5m klines + daily OI/top-trader metrics + monthly
funding rate (all downloaded from Binance's public data.binance.vision archive, steps 1-2) into a
single raw input frame matching the exact column contract `features/engineering.py::FeatureEngineer.process()`
expects (eth_raw_cols in verify_live_feature_pipeline_parity_20260706.py): timestamp, open, high,
low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote,
sum_open_interest_value, sum_toptrader_long_short_ratio, count_long_short_ratio, last_funding_rate,
plus close_btc/volume_btc/quote_volume_btc (BTC as the cross-asset secondary series, same role BTC
plays in the ETH-primary pipeline). Output is RAW inputs only -- feature engineering happens in the
next step.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "SOLUSDT"
KLINE_PATH = ROOT / f"binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
BTC_KLINE_PATH = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
OUT_PATH = ROOT / "data/splits/year_oos/sol_raw_frame_2024_2026.csv"


def load_metrics() -> pd.DataFrame:
    frames = []
    for p in sorted(METRICS_DIR.glob(f"{SYMBOL}-metrics-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["create_time"])
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]


def load_funding() -> pd.DataFrame:
    frames = []
    for p in sorted(FUNDING_DIR.glob(f"{SYMBOL}-fundingRate-*.zip")):
        with zipfile.ZipFile(p) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df = pd.read_csv(f)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["calc_time"], unit="ms")
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[["timestamp", "last_funding_rate"]]


def main() -> int:
    kline = pd.read_csv(KLINE_PATH, low_memory=False)
    kline["timestamp"] = pd.to_datetime(kline["timestamp"])
    kline = kline.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"SOL klines: {len(kline)} rows {kline['timestamp'].iloc[0]}..{kline['timestamp'].iloc[-1]}", flush=True)

    metrics = load_metrics()
    print(f"SOL metrics: {len(metrics)} rows {metrics['timestamp'].iloc[0]}..{metrics['timestamp'].iloc[-1]}", flush=True)
    funding = load_funding()
    print(f"SOL funding: {len(funding)} rows {funding['timestamp'].iloc[0]}..{funding['timestamp'].iloc[-1]}", flush=True)

    btc = pd.read_csv(BTC_KLINE_PATH, low_memory=False, usecols=["timestamp", "close", "volume", "quote_volume"])
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    btc = btc.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})

    frame = kline.merge(metrics, on="timestamp", how="left")
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    frame = frame.merge(btc[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")

    # small (<1%), scattered gaps in the exchange-recorded metrics/BTC-cross series are normal
    # (a given 5m timestamp occasionally has no metrics tick) -- forward-fill last known value,
    # the same causal treatment already applied to funding rate above (never fills from the future)
    ffill_cols = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
                  "close_btc", "volume_btc", "quote_volume_btc"]
    frame[ffill_cols] = frame[ffill_cols].ffill()

    required = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate",
                "close_btc", "volume_btc", "quote_volume_btc"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"missing required columns after merge: {missing}")

    # trim to the region where ALL required auxiliary series have coverage (metrics/funding/btc
    # start dates may lag the raw kline start slightly)
    coverage_ok = frame[required].notna().all(axis=1)
    first_ok = coverage_ok.idxmax()
    frame = frame.iloc[first_ok:].reset_index(drop=True)
    na_counts = frame[required].isna().sum()
    print("\nNA counts per required column after trimming to full-coverage region:")
    print(na_counts[na_counts > 0] if na_counts.any() else "  (none)")

    frame.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {len(frame)} rows {frame['timestamp'].iloc[0]}..{frame['timestamp'].iloc[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
