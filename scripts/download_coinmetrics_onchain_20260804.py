"""New-data-source Stage 0 (on-chain axis, per user request 2026-08-04 following the closed Rho1
panel and Deribit DVOL axes): download BTC on-chain metrics from CoinMetrics' free Community API
(no account/API key required -- same deliberate choice as every other raw downloader in this repo).

Feasibility checked live 2026-08-04: Glassnode has no free API tier at all (Professional-plan
add-on only); CryptoQuant's Data API starts at the Professional plan ($109/mo) with 24h resolution;
CoinMetrics Community is the only genuinely free option with real historical depth -- confirmed via
its catalog-v2 endpoint that BTC daily community=true metrics go back to 2009-2011 with no auth.

Metrics selected (community=true, daily, BTC): AdrActCnt (active addresses), CapMVRVCur (MVRV),
FlowInExNtv/FlowOutExNtv (exchange in/out flow, native units), SplyExNtv (supply held on exchanges,
native units), HashRate, TxCnt. SOPR is NOT available on the free tier (paid-only) -- noted as a
known gap, not worth $109/mo to close for a cheap falsification test.

Endpoint quirk found while building this: the default page_size (100) silently truncates a
multi-year request to the most recent ~100 rows with no error and no next_page_token -- NOT a
forward-pagination-from-start issue like DVOL's backward continuation, just an undocumented default
cap. Fixed by passing an explicit page_size large enough to cover the whole range in one request
(daily cadence over a few years is at most a few thousand rows).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/onchain/coinmetrics"
OUT_PATH = OUT_DIR / "btc_onchain_daily.csv"
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["AdrActCnt", "CapMVRVCur", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "HashRate", "TxCnt"]
START_DEFAULT = pd.Timestamp("2024-01-01", tz="UTC")
PAGE_SIZE = 10000  # covers the full multi-year daily range in one request -- see module docstring


def fetch(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    resp = requests.get(URL, params={
        "assets": "btc",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "page_size": PAGE_SIZE,
    }, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinMetrics API error {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    if "next_page_token" in payload:
        raise RuntimeError("unexpected pagination -- range exceeded PAGE_SIZE, widen PAGE_SIZE or add a pagination loop")
    return payload["data"]


def main() -> int:
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing["time"] = pd.to_datetime(existing["time"])
        start = existing["time"].max() + pd.Timedelta(days=1)
    else:
        existing = pd.DataFrame()
        start = START_DEFAULT

    end = pd.Timestamp.now(tz="UTC")
    if start >= end:
        print(f"up-to-date: rows={len(existing)}")
        return 0

    raw = fetch(start, end)
    if not raw:
        print(f"no-new-data: rows={len(existing)}")
        return 0

    new_df = pd.DataFrame(raw)
    new_df["time"] = pd.to_datetime(new_df["time"])
    for col in METRICS:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
    new_df = new_df[["time"] + METRICS]

    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    n_bad = int(combined[METRICS].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} rows have non-finite on-chain values -- refusing to write")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: rows={len(combined)} (+{len(new_df)} new), "
          f"range={combined['time'].min()}..{combined['time'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
