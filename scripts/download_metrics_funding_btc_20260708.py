"""BTC replication, step 1: download BTCUSDT historical open-interest/top-trader metrics (daily
zips) and funding rate (monthly zips) from Binance's public data.binance.vision archive -- mirrors
scripts/download_metrics_funding_sol_20260707.py. BTC 5m klines already exist locally
(binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv, 2023-12-31..2026-07-01) but funding/OI metrics
have not been downloaded for BTC yet (only ETHUSDT/SOLUSDT are cached).
"""
from __future__ import annotations

import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = "BTCUSDT"
START_DATE = date(2024, 1, 1)
END_DATE = date(2026, 8, 2)
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


def download_metrics() -> tuple[int, int]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, 0
    for d in daterange(START_DATE, END_DATE):
        out = METRICS_DIR / f"{SYMBOL}-metrics-{d.isoformat()}.zip"
        if out.exists():
            ok += 1
            continue
        url = METRICS_URL.format(sym=SYMBOL, d=d.isoformat())
        resp = _get_with_retry(url)
        if resp.status_code == 200 and resp.content:
            out.write_bytes(resp.content)
            ok += 1
        else:
            missing += 1
            print(f"  missing metrics for {d}: HTTP {resp.status_code}", flush=True)
        time.sleep(0.05)
    return ok, missing


def download_funding() -> tuple[int, int]:
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    ok, missing = 0, 0
    for ym in month_range(START_DATE, END_DATE):
        out = FUNDING_DIR / f"{SYMBOL}-fundingRate-{ym}.zip"
        if out.exists():
            ok += 1
            continue
        url = FUNDING_URL.format(sym=SYMBOL, ym=ym)
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
    print(f"Downloading {SYMBOL} daily metrics {START_DATE}..{END_DATE}...", flush=True)
    m_ok, m_missing = download_metrics()
    print(f"metrics: {m_ok} downloaded/cached, {m_missing} missing", flush=True)

    print(f"\nDownloading {SYMBOL} monthly funding rate...", flush=True)
    f_ok, f_missing = download_funding()
    print(f"funding: {f_ok} downloaded/cached, {f_missing} missing", flush=True)

    if m_missing > 30:
        raise RuntimeError(f"too many missing metrics days ({m_missing}) -- BTC metrics history may not extend this far back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
