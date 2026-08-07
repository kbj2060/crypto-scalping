"""Generic (symbol-as-CLI-arg) funding/metrics downloader for extending existing
binance_data/metrics and binance_data/funding_rate coverage forward in time.
Mirrors download_metrics_funding_sol_20260707.py / download_metrics_funding_btc_20260708.py
without hardcoding one symbol.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "binance_data/metrics"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"
METRICS_URL = "https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{d}.zip"
FUNDING_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"


def _get_with_retry(url: str, *, retries: int = 5, timeout: int = 30) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            time.sleep(min(2.0 * (attempt + 1), 10.0))
    raise RuntimeError(f"failed after {retries} retries: {url}") from last_exc


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_metrics(symbol: str, start: date, end: date) -> tuple[int, int]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, 0
    for d in daterange(start, end):
        out = METRICS_DIR / f"{symbol}-metrics-{d.isoformat()}.zip"
        if out.exists():
            ok += 1
            continue
        url = METRICS_URL.format(sym=symbol, d=d.isoformat())
        resp = _get_with_retry(url)
        if resp.status_code == 200 and resp.content:
            out.write_bytes(resp.content)
            ok += 1
        else:
            missing += 1
            print(f"  missing metrics for {d}: HTTP {resp.status_code}", flush=True)
        time.sleep(0.05)
    return ok, missing


def download_funding(symbol: str, start: date, end: date) -> tuple[int, int]:
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, 0
    for ym in month_range(start, end):
        out = FUNDING_DIR / f"{symbol}-fundingRate-{ym}.zip"
        if out.exists():
            ok += 1
            continue
        url = FUNDING_URL.format(sym=symbol, ym=ym)
        resp = _get_with_retry(url)
        if resp.status_code == 200 and resp.content:
            out.write_bytes(resp.content)
            ok += 1
        else:
            missing += 1
            print(f"  missing funding for {ym}: HTTP {resp.status_code}", flush=True)
        time.sleep(0.1)
    return ok, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"Downloading {args.symbol} daily metrics {start}..{end}...", flush=True)
    m_ok, m_missing = download_metrics(args.symbol, start, end)
    print(f"metrics: {m_ok} downloaded/cached, {m_missing} missing", flush=True)

    print(f"\nDownloading {args.symbol} monthly funding rate...", flush=True)
    f_ok, f_missing = download_funding(args.symbol, start, end)
    print(f"funding: {f_ok} downloaded/cached, {f_missing} missing", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
