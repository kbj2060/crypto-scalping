"""Fetch the current (in-progress) month's funding-rate history via the public Binance futures
REST endpoint (/fapi/v1/fundingRate) and package it into the same monthly-zip format the
data.binance.vision archive uses, since data.binance.vision only publishes a month's fundingRate
zip after the month completes. Writes into the same `binance_data/funding_rate/` directory
update_features.py's ensure_funding()/load path already reads, using the exact filename it expects
(`{symbol}-fundingRate-{YYYY-MM}.zip` containing a CSV with columns
`calc_time,funding_interval_hours,last_funding_rate`, matching the archived format's schema).
No live-account client is touched -- this is the same public, unauthenticated REST endpoint
`extend_klines_20260713.py` already uses for klines.
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FUNDING_DIR = ROOT / "binance_data/funding_rate"
URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_month(symbol: str, year: int, month: int) -> list[dict]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(URL, params={
            "symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 1000:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, default=FUNDING_DIR,
                     help="defaults to binance_data/funding_rate (ETH's live pipeline dir); "
                          "SOL/BTC's own raw-frame builders read from binance_data/funding_rate_other instead")
    args = ap.parse_args()

    rows = fetch_month(args.symbol, args.year, args.month)
    print(f"{args.symbol} {args.year}-{args.month:02d}: fetched {len(rows)} funding events", flush=True)
    if not rows:
        return 1

    ym = f"{args.year:04d}-{args.month:02d}"
    csv_name = f"{args.symbol}-fundingRate-{ym}.csv"
    out_dir = Path(args.out_dir)
    zip_path = out_dir / f"{args.symbol}-fundingRate-{ym}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["calc_time", "funding_interval_hours", "last_funding_rate"])
    prev_ts = None
    for r in rows:
        ts = int(r["fundingTime"])
        interval_hours = 8
        if prev_ts is not None:
            interval_hours = round((ts - prev_ts) / 3600000.0)
        writer.writerow([ts, interval_hours, r["fundingRate"]])
        prev_ts = ts

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, buf.getvalue())
    print(f"wrote {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
