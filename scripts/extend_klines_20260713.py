"""Incrementally extend an existing binance_data/klines/{SYMBOL}/{SYMBOL}-5m-api.csv file with
new bars from Binance's public REST API (fapi, no account credentials -- does not use
core/binance_client.py, matching scripts/download_klines_sol_20260707.py's deliberate choice to
avoid touching live trading account state). Only fetches from the existing file's last timestamp
forward, instead of re-downloading the whole history.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = "5m"
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
        time.sleep(0.25)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["ETHUSDT", "SOLUSDT", "BTCUSDT"])
    args = ap.parse_args()
    symbol = args.symbol

    out_path = ROOT / f"binance_data/klines/{symbol}/{symbol}-5m-api.csv"
    existing = pd.read_csv(out_path, low_memory=False)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"])
    last_ts = existing["timestamp"].max()
    print(f"{symbol}: existing rows={len(existing)}, last_ts={last_ts}", flush=True)

    start_ms = int((last_ts + pd.Timedelta(minutes=5)).tz_localize("UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if start_ms >= end_ms:
        print(f"{symbol}: already up to date, nothing to fetch", flush=True)
        return 0

    raw = fetch_klines(symbol, INTERVAL, start_ms, end_ms)
    if not raw:
        print(f"{symbol}: no new klines returned", flush=True)
        return 0
    new_df = pd.DataFrame(raw, columns=COLUMNS)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    new_df["trades"] = pd.to_numeric(new_df["trades"], errors="coerce").astype("Int64")

    combined = pd.concat([existing, new_df[existing.columns]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    n_missing_finite = int(combined[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_missing_finite:
        raise RuntimeError(f"{n_missing_finite} rows have non-finite OHLC values -- refusing to write a corrupt kline file")

    combined.to_csv(out_path, index=False)
    print(f"{symbol}: wrote {out_path}: {len(combined)} rows, new range end={combined['timestamp'].iloc[-1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
