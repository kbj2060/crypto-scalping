"""Isolated variant of build_btc_raw_frame_20260708.py that additionally pulls in 3 previously-unused
Binance BTC metrics columns: sum_taker_long_short_vol_ratio, count_toptrader_long_short_ratio (raw),
plus sum_toptrader_long_short_ratio/count_long_short_ratio are already present in the baseline frame.
Writes to an ISOLATED output path -- does NOT touch data/splits/year_oos/btc_raw_frame_2024_2026.csv
which other pipelines depend on unchanged. BTC-specific copy (build_btc_raw_frame_20260708.py is
BTC-only already, but this is kept as a separate file per task instructions to avoid touching the
original at all).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "BTCUSDT"
KLINE_PATH = ROOT / f"binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv"
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
CROSS_KLINE_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_raw_frame_metrics4_2024_2026.csv"

NEW_METRICS_COLS = ["sum_taker_long_short_vol_ratio", "count_toptrader_long_short_ratio"]
BASE_METRICS_COLS = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]


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
    return out[["timestamp"] + BASE_METRICS_COLS + NEW_METRICS_COLS]


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
    print(f"BTC klines: {len(kline)} rows {kline['timestamp'].iloc[0]}..{kline['timestamp'].iloc[-1]}", flush=True)

    metrics = load_metrics()
    print(f"BTC metrics: {len(metrics)} rows {metrics['timestamp'].iloc[0]}..{metrics['timestamp'].iloc[-1]}", flush=True)
    funding = load_funding()
    print(f"BTC funding: {len(funding)} rows {funding['timestamp'].iloc[0]}..{funding['timestamp'].iloc[-1]}", flush=True)

    cross = pd.read_csv(CROSS_KLINE_PATH, low_memory=False, usecols=["timestamp", "close", "volume", "quote_volume"])
    cross["timestamp"] = pd.to_datetime(cross["timestamp"])
    cross = cross.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    cross = cross.rename(columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})

    frame = kline.merge(metrics, on="timestamp", how="left")
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding.sort_values("timestamp"),
                           on="timestamp", direction="backward")
    frame = frame.merge(cross[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]], on="timestamp", how="left")

    ffill_cols = BASE_METRICS_COLS + NEW_METRICS_COLS + ["close_btc", "volume_btc", "quote_volume_btc"]
    frame[ffill_cols] = frame[ffill_cols].ffill()

    required = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote"] + BASE_METRICS_COLS + NEW_METRICS_COLS + [
                "last_funding_rate", "close_btc", "volume_btc", "quote_volume_btc"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"missing required columns after merge: {missing}")

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
