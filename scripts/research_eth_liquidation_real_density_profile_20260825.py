#!/usr/bin/env python3
"""Descriptive-only real-liquidation price density profile, 2026-08-25.

Not a statistical test -- the underlying window (real @forceOrder data only exists from
2026-07-18 15:03 UTC onward, ~36 days) is far too short for one, see
eth_liquidation_map_volume_liquidation_concentration_20260825 for why. This is a plain "where has
real liquidation $ actually clustered by price over the available window" profile, built the same
way the live estimate bins things (BIN_WIDTH_PCT-wide price buckets) but from REAL long_usd_1m/
short_usd_1m dollars instead of hypothetical leveraged-entry bins -- meant to be eyeballed against
the current deployed support/resistance levels (from /api/liquidation-map), not scored.

tail_risk_1m has no price column, so each minute's liquidation $ is joined to the nearest 5-min
ETHUSDT close (fetched fresh from Binance for just this window) as the best available proxy for
"what price was in effect when this liquidation happened". Server-only: needs duckdb + a fresh
tail_risk.duckdb (local dev has neither, see eth_liquidation_map_volume_liquidation_concentration_
20260825's note on this).
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TAIL_RISK_DB = ROOT / "data" / "live" / "tail_risk.duckdb"
LIQ_VALID_SINCE_UTC = "2026-07-18 15:03:00+00"
BIN_WIDTH_PCT = 0.0025  # matches live_liquidation_map_20260824.py's BIN_WIDTH_PCT, for visual comparability
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_real_density_profile_20260825.json"


def fetch_5m_klines(start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": "ETHUSDT", "interval": "5m", "startTime": cursor, "limit": 1500},
            timeout=10,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        if len(batch) < 1500:
            break
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "qv", "trades", "tbb", "tbq", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    return df[["timestamp", "close"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    con = duckdb.connect(str(TAIL_RISK_DB), read_only=True)
    try:
        liq = con.execute(f"""
            SELECT ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale
            FROM tail_risk_1m
            WHERE ts >= TIMESTAMPTZ '{LIQ_VALID_SINCE_UTC}'
            ORDER BY ts
        """).df()
    finally:
        con.close()
    liq["ts"] = liq["ts"].dt.tz_convert("UTC")
    liq = liq[(liq["valid_liq_stream"] == True) & (liq["ws_stale"] != True)]  # noqa: E712
    print(f"real liquidation rows: {len(liq)}, {liq['ts'].min()} -> {liq['ts'].max()}", flush=True)

    start_ms = int(liq["ts"].min().timestamp() * 1000)
    end_ms = int(liq["ts"].max().timestamp() * 1000) + 5 * 60 * 1000
    px = fetch_5m_klines(start_ms, end_ms)
    print(f"fetched {len(px)} 5m price bars from Binance", flush=True)

    liq["bucket"] = liq["ts"].dt.floor("5min")
    liq_5m = liq.groupby("bucket")[["long_usd_1m", "short_usd_1m"]].sum().reset_index()
    liq_5m["liq_usd"] = liq_5m["long_usd_1m"] + liq_5m["short_usd_1m"]

    merged = liq_5m.merge(px, left_on="bucket", right_on="timestamp", how="inner")
    print(f"merged (price+liquidation) rows: {len(merged)}", flush=True)

    current_price = float(px["close"].iloc[-1])
    bin_width = current_price * BIN_WIDTH_PCT
    merged["bin"] = np.round(merged["close"] / bin_width).astype(int)
    density = merged.groupby("bin")["liq_usd"].sum().reset_index()
    density["price"] = density["bin"] * bin_width
    density["distance_pct"] = (density["price"] - current_price) / current_price * 100
    density = density.sort_values("liq_usd", ascending=False).reset_index(drop=True)

    print(f"\ncurrent price: {current_price:.2f}")
    print(f"total real liquidation $ in window: {merged['liq_usd'].sum():,.0f}")
    print("\nTop 15 real-liquidation price bins (most $ concentrated there over the window):")
    for _, row in density.head(15).iterrows():
        print(f"  price={row['price']:8.2f}  ({row['distance_pct']:+6.2f}%)  liq_usd={row['liq_usd']:>14,.0f}")

    out = {
        "current_price": current_price, "n_liq_rows": int(len(liq)), "n_5m_bars_merged": int(len(merged)),
        "total_liq_usd": float(merged["liq_usd"].sum()),
        "top_bins": density.head(30).to_dict("records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
