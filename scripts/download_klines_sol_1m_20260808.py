"""Rev7 maker-execution audit data: SOLUSDT 1m futures klines from Binance's public REST API
(same public-API pattern as scripts/download_klines_sol_20260707.py -- deliberately NOT
core/binance_client.py, which touches live account settings). Window covers VAL + OOS plus the
288x5m horizon tail: 2025-08-25 .. 2026-05-01.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "SOLUSDT"
INTERVAL = "1m"
START = "2025-08-25"
END = "2026-05-01"
OUT_DIR = ROOT / "binance_data/klines" / SYMBOL
OUT_PATH = OUT_DIR / f"{SYMBOL}-1m-api.csv"
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": cursor, "endTime": end_ms, "limit": 1500}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text[:500]}")
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1500:
            break
        time.sleep(0.25)
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_ms = int(pd.Timestamp(START).timestamp() * 1000)
    end_ms = int(pd.Timestamp(END).timestamp() * 1000)
    rows = fetch_klines(start_ms, end_ms)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: {len(df)} rows {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
