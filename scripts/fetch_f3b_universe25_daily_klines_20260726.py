#!/usr/bin/env python3
"""Fetch daily klines for the 25-coin F3-B universe (locked in
data/ensemble/metrics/f3b_universe25_selection_20260726.json) from Binance's
public futures REST API. No API key, no account state -- same deliberate
choice as scripts/download_klines_1m_20260716.py.

Writes binance_data/klines_daily/{SYMBOL}/{SYMBOL}-1d-api.csv.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
START = "2023-10-01"  # buffer before the 2024-01-01 exploration start for 60d lookback warm-up
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": "1d", "startTime": cursor, "endTime": end_ms, "limit": 1500}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"{symbol}: Binance API error {resp.status_code}: {resp.text[:300]}")
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1500:
            break
        time.sleep(0.15)
    return rows


def main() -> int:
    sel = json.loads((ROOT / "data/ensemble/metrics/f3b_universe25_selection_20260726.json").read_text())
    universe = sel["universe"]

    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    for symbol in universe:
        out_path = ROOT / f"binance_data/klines_daily/{symbol}/{symbol}-1d-api.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = fetch_klines(symbol, start_ms, end_ms)
        if not raw:
            print(f"{symbol}: NO DATA RETURNED", flush=True)
            continue
        df = pd.DataFrame(raw, columns=COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        n_bad = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        if n_bad:
            raise RuntimeError(f"{symbol}: {n_bad} rows with non-finite OHLC -- refusing to write")
        df[["timestamp", "open", "high", "low", "close", "volume"]].to_csv(out_path, index=False)
        print(f"{symbol}: wrote {out_path}: {len(df)} rows, range {df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}", flush=True)
        time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
