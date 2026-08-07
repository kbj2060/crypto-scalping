"""Stage 0 (Rho1 panel design): download 5m futures klines for the full panel universe
from data/splits/panel_universe_symbols_20260804.json.

Same public REST API pattern as scripts/download_klines_sol_20260707.py /
download_klines_1m_20260716.py (fapi.binance.com, no account credentials -- avoids touching
live trading account state). Resumable: if binance_data/klines/{SYM}/{SYM}-5m-api.csv already
exists, only fetches bars after its last timestamp.

Writes one file per symbol; does not touch the RAW_SOURCE_MANIFEST.json hash-pinning scheme
(that scheme covers the immutable daily/monthly zip files from data.binance.vision, not these
REST-assembled, intentionally-growing combined CSVs -- same as how the existing BTC/ETH/SOL
*-5m-api.csv files are not manifest-pinned either).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "data/splits/panel_universe_symbols_20260804.json"
KLINES_DIR = ROOT / "binance_data/klines"
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
INTERVAL = "5m"
DEFAULT_START = "2024-01-01"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": INTERVAL, "startTime": cursor, "endTime": end_ms, "limit": 1500}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code} for {symbol}: {resp.text[:300]}")
        batch = resp.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected response for {symbol}: {batch}")
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1500:
            break
        time.sleep(0.2)
    return rows


def download_symbol(symbol: str, start: str) -> tuple[int, str]:
    out_dir = KLINES_DIR / symbol
    out_path = out_dir / f"{symbol}-5m-api.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path, low_memory=False)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        last_ts = existing["timestamp"].max()
        start_ms = int((last_ts + pd.Timedelta(minutes=5)).tz_localize("UTC").timestamp() * 1000)
    else:
        existing = pd.DataFrame(columns=COLUMNS)
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)

    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if start_ms >= end_ms:
        return len(existing), "up-to-date"

    try:
        raw = fetch_klines(symbol, start_ms, end_ms)
    except RuntimeError as e:
        return len(existing), f"ERROR: {e}"

    if not raw:
        return len(existing), "no-new-data"

    new_df = pd.DataFrame(raw, columns=COLUMNS)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
    for c in NUMERIC_COLS:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce")
    new_df["trades"] = pd.to_numeric(new_df["trades"], errors="coerce").astype("Int64")

    combined = pd.concat([existing, new_df[COLUMNS]], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_bad = int(combined[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{symbol}: {n_bad} rows have non-finite OHLC -- refusing to write corrupt file")

    combined.to_csv(out_path, index=False)
    return len(combined), f"+{len(new_df)} new"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--sleep", type=float, default=0.3, help="seconds between symbols")
    args = ap.parse_args()

    universe = json.loads(UNIVERSE_PATH.read_text())
    symbols = [row["symbol"] for row in universe["symbols"]]
    print(f"downloading 5m klines for {len(symbols)} symbols from {args.start}...", flush=True)

    results = {}
    for i, sym in enumerate(symbols, 1):
        n, status = download_symbol(sym, args.start)
        results[sym] = {"rows": n, "status": status}
        print(f"[{i}/{len(symbols)}] {sym:16s} rows={n:>8d}  {status}", flush=True)
        time.sleep(args.sleep)

    errors = {k: v for k, v in results.items() if "ERROR" in v["status"]}
    print(f"\ndone. {len(symbols) - len(errors)}/{len(symbols)} ok, {len(errors)} errors", flush=True)
    if errors:
        for k, v in errors.items():
            print(f"  {k}: {v['status']}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
