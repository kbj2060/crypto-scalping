#!/usr/bin/env python3
"""Renders the CURRENT live event-driven liquidation map exactly as dashboard/server.py::
load_liquidation_map() computes it -- same symbol/interval/fetch depth (ETHUSDT, 1h,
LIQUIDATION_MAP_FETCH_LIMIT=1000 bars ~41.6 days) and the same compute_event_driven_levels() call
-- then charts the last 7 days (168h) of price with the resulting support/resistance levels and
heatmap density overlaid, 2026-08-25 user request ("이벤트 드리븐 1주일치 청산지도 만들어서
보여줘") ahead of discussing where accuracy could be improved.

Not a backtest -- a single live snapshot, matplotlib PNG. Level SET can be (and typically is)
older than the 7-day chart window itself (median reset gap 44-54h, p90 262h per compute_event_
driven_levels()'s own docstring) -- support_window_hours/resistance_window_hours are reported in
the title so it's clear how stale each side's current level set is, exactly the number the dashboard
badge shows.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

import scripts.live_liquidation_map_20260824 as liqmap

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "research" / "eth_liquidation_map_current_1week_20260825.png"
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
FETCH_LIMIT = 1000  # matches dashboard/server.py LIQUIDATION_MAP_FETCH_LIMIT
CHART_DAYS = 7


def fetch_klines() -> pd.DataFrame:
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": SYMBOL, "interval": INTERVAL, "limit": FETCH_LIMIT},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df["close_time"] = df["close_time"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)  # drop still-forming bar, matches load_liquidation_map()
    return df


def main() -> None:
    df = fetch_klines()
    current_price = float(df["close"].iloc[-1])
    print(f"fetched {len(df)} bars, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"current_price={current_price:.2f}", flush=True)

    payload = liqmap.compute_event_driven_levels(df, current_price)
    if not payload["warmed_up"]:
        print(f"NOT WARMED UP: {payload.get('error')}")
        return

    print(f"support_window_hours={payload['support_window_hours']:.0f} "
          f"resistance_window_hours={payload['resistance_window_hours']:.0f}", flush=True)
    print(f"support_levels: {payload['support_levels']}", flush=True)
    print(f"resistance_levels: {payload['resistance_levels']}", flush=True)

    window = df.tail(CHART_DAYS * 24).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    w = pd.Timedelta(hours=0.35)
    for _, row in window.iterrows():
        color = "#2e7d32" if row["close"] >= row["open"] else "#c62828"
        ax.plot([row["timestamp"], row["timestamp"]], [row["low"], row["high"]], color=color, linewidth=0.9)
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(row["timestamp"] - w), min(row["open"], row["close"])),
            mdates.date2num(row["timestamp"] + w) - mdates.date2num(row["timestamp"] - w),
            max(abs(row["close"] - row["open"]), 1e-6), color=color))

    x0, x1 = window["timestamp"].iloc[0], window["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    # heatmap density strip -- same field the dashboard's liquidationDensityProfile() reads
    max_w = max((b["weight_pct"] for b in payload["heatmap_bins"]), default=0.0)
    bw = payload["bin_width"]
    for b in payload["heatmap_bins"]:
        is_support = b["price"] < current_price
        color = "#1565c0" if is_support else "#ef6c00"
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(x0), b["price"] - bw / 2), mdates.date2num(x1) - mdates.date2num(x0), bw,
            color=color, alpha=0.10 + 0.35 * (b["weight_pct"] / max_w if max_w else 0), zorder=0))

    def declutter(prices: list[float], min_gap: float) -> list[float]:
        """Greedy min-separation: walk price-sorted order, push each label at least min_gap
        (data units) from the previous one so text doesn't overlap when levels sit close together."""
        out = list(prices)
        for i in range(1, len(out)):
            if out[i] - out[i - 1] < min_gap:
                out[i] = out[i - 1] + min_gap
        return out

    price_span = window["high"].max() - window["low"].min()
    min_gap = price_span * 0.045

    for side, levels, color, tag in (("support", payload["support_levels"], "#1565c0", "S"),
                                     ("resistance", payload["resistance_levels"], "#ef6c00", "R")):
        order = sorted(range(len(levels)), key=lambda k: levels[k]["price"])
        sorted_prices = [levels[k]["price"] for k in order]
        label_y = declutter(sorted_prices, min_gap)
        label_y_by_idx = {order[j]: label_y[j] for j in range(len(order))}
        for i, lv in enumerate(levels):
            ax.axhline(lv["price"], color=color, linewidth=1 + 3 * lv["weight_pct"], linestyle="--", alpha=0.85)
            ty = label_y_by_idx[i]
            if abs(ty - lv["price"]) > 1e-9:
                ax.plot([x1, x1 + pd.Timedelta(hours=4)], [lv["price"], ty], color=color, linewidth=0.6, alpha=0.6)
            ax.text(x1 + pd.Timedelta(hours=4), ty, f" {tag}{i+1} ${lv['price']:,.0f} ({lv['distance_pct']:+.2f}%)",
                    color=color, fontsize=8, va="center")

    ax.axhline(current_price, color="black", linewidth=1.3)
    ax.text(x0, current_price, f"current ${current_price:,.2f} ", color="black", fontsize=9,
            va="center", ha="right", fontweight="bold")

    ax.set_xlim(x0, x1 + pd.Timedelta(hours=32))  # room for the level labels on the right
    ax.set_title(f"ETHUSDT event-driven liquidation map (live formula, as of {df['timestamp'].iloc[-1]:%Y-%m-%d %H:%M} UTC)\n"
                 f"support side reset {payload['support_window_hours']:.0f}h ago  |  "
                 f"resistance side reset {payload['resistance_window_hours']:.0f}h ago  |  chart window=last {CHART_DAYS}d "
                 f"(S/R blue=support, orange=resistance)")
    ax.set_ylabel("ETHUSDT")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    plt.close(fig)
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
