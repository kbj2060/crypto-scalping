"""Phase A pilot (SOL), step 1: download SOLUSDT 5m futures klines from Binance's PUBLIC REST API
(no account credentials needed -- deliberately does NOT reuse core/binance_client.py, which is
tied to the live trading account and calls futures_change_leverage() on init; using it here would
risk touching live trading settings for a pure historical-data download). Output format matches
the existing binance_data/klines/{BTC,ETH}USDT-5m-api.csv files exactly (same 12 columns) so all
downstream scripts (feature engineering, label building) can treat SOL identically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "SOLUSDT"
INTERVAL = "5m"
START = "2024-06-01"
END = None  # None = up to now
OUT_DIR = ROOT / "binance_data/klines" / SYMBOL
OUT_PATH = OUT_DIR / f"{SYMBOL}-5m-api.csv"
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1500}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text[:500]}")
        batch = resp.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Binance API response: {batch}")
        if not batch:
            break
        rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + 1
        if len(batch) < 1500:
            break
        time.sleep(0.25)  # stay well under Binance's public rate limit
    return rows


def main() -> int:
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) if END is None else int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    print(f"Fetching {SYMBOL} {INTERVAL} klines from {START} to now...", flush=True)

    raw = fetch_klines(SYMBOL, INTERVAL, start_ms, end_ms)
    if not raw:
        raise RuntimeError(f"no klines returned for {SYMBOL}")
    df = pd.DataFrame(raw, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"rows={len(df)} range={df['timestamp'].iloc[0]}..{df['timestamp'].iloc[-1]}", flush=True)
    n_missing_finite = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_missing_finite:
        raise RuntimeError(f"{n_missing_finite} rows have non-finite OHLC values -- refusing to write a corrupt kline file")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
