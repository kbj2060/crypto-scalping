"""New-data-source Stage 0 (per user request 2026-08-04, following the Rho1 close-out research):
download Deribit's DVOL (30-day forward-looking implied volatility index) for BTC and ETH via
Deribit's public REST API (no account credentials -- same deliberate choice as every other raw
downloader in this repo). Hourly resolution, confirmed the only sensible native resolution for
history this deep (1s/1min resolutions only retain the last few days; 3600s/1h has full history
with zero gaps, confirmed by spot-check against a full January 2024 month before committing to
this design).

This is genuinely new information for this project: everything used so far (spot/futures OHLCV,
funding, OI/positioning) comes from the same underlying spot/perp market DYNAMICS. DVOL is priced
by a DIFFERENT market (options) that is directly betting on future realized volatility -- it is
not derivable from the spot/perp data already in binance_data/. It's also the natural next step
after the BTC event gate (project-btc-event-gate-stage1-stable-lift-20260804.md), whose own
closing note flagged options/derivatives data as the one avenue not yet tried for getting REAL
volatility exposure instead of an estimated one.

API caps each response at 1000 rows; paginate using the 'continuation' timestamp Deribit returns.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/derivatives/deribit_dvol"
URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
RESOLUTION = "3600"  # 1h -- the only resolution with full multi-year history (1s/60s only keep recent days)
START_DEFAULT = pd.Timestamp("2024-01-01", tz="UTC")
COLUMNS = ["timestamp", "open", "high", "low", "close"]


def fetch_dvol(currency: str, start_ms: int, end_ms: int) -> list[list]:
    """Deribit caps each response at 1000 rows and, when the requested range exceeds that, fills
    from the END of the range backward -- 'continuation' is the next EARLIER end_timestamp to
    request, not a forward cursor (confirmed empirically 2026-08-04: requesting the full
    2024-01-01..now range returned only the most recent ~42 days, with continuation pointing to a
    timestamp BEFORE the first returned row). So we paginate backward, keeping start_timestamp
    fixed and walking end_timestamp down via continuation until it's no longer newer than start.
    """
    rows: list[list] = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        resp = requests.get(URL, params={"currency": currency, "start_timestamp": start_ms,
                                          "end_timestamp": cursor_end, "resolution": RESOLUTION}, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Deribit API error {resp.status_code} for {currency}: {resp.text[:300]}")
        result = resp.json().get("result")
        if result is None:
            raise RuntimeError(f"Unexpected Deribit response for {currency}: {resp.text[:300]}")
        batch = result["data"]
        if not batch:
            break
        rows.extend(batch)
        continuation = result.get("continuation")
        if continuation is None or continuation >= cursor_end:
            break
        cursor_end = continuation
        time.sleep(0.1)
    return rows


def download_symbol(currency: str, start: pd.Timestamp) -> tuple[int, str]:
    out_path = OUT_DIR / f"{currency}_dvol_hourly.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        last_ts = existing["timestamp"].max()
        start_ms = int((last_ts + pd.Timedelta(hours=1)).timestamp() * 1000)
    else:
        existing = pd.DataFrame(columns=COLUMNS)
        start_ms = int(start.timestamp() * 1000)

    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if start_ms >= end_ms:
        return len(existing), "up-to-date"

    raw = fetch_dvol(currency, start_ms, end_ms)
    if not raw:
        return len(existing), "no-new-data"

    new_df = pd.DataFrame(raw, columns=COLUMNS)
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")
    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_bad = int(combined[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{currency}: {n_bad} rows have non-finite DVOL values -- refusing to write")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return len(combined), f"+{len(new_df)} new"


def main() -> int:
    for currency in ["BTC", "ETH"]:
        n, status = download_symbol(currency, START_DEFAULT)
        print(f"{currency}: rows={n} {status}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
