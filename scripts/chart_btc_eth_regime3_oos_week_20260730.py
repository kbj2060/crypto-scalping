#!/usr/bin/env python3
"""Render the final OOS week as price with causal HMM regime bands only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/causal_regen_20260516/btc_eth_regime3_fresh_forward_oos_month_20260730/btc_eth_regime3_fresh_forward_oos_march_2026.csv"
OUT = ROOT / "tmp/causal_regen_20260516/btc_eth_regime3_fresh_forward_oos_month_20260730/btc_eth_regime3_oos_week_march25_31_2026.png"
START = pd.Timestamp("2026-03-25 00:00:00")
END = pd.Timestamp("2026-04-01 00:00:00")
COLORS = {"bull": "#20c997", "bear": "#ff5c77", "chop": "#f4c95d"}


def _shade(ax: plt.Axes, frame: pd.DataFrame) -> None:
    regimes = frame["regime"].to_numpy()
    times = frame["timestamp"].to_numpy()
    starts = np.r_[0, np.flatnonzero(regimes[1:] != regimes[:-1]) + 1]
    stops = np.r_[starts[1:], len(frame)]
    for start, stop in zip(starts, stops, strict=True):
        ax.axvspan(pd.Timestamp(times[start]), pd.Timestamp(times[stop - 1]) + pd.Timedelta(minutes=5),
                   color=COLORS[str(regimes[start])], alpha=0.25, lw=0)


def main() -> int:
    frame = pd.read_csv(SOURCE, parse_dates=["timestamp"])
    frame = frame[(frame["timestamp"] >= START) & (frame["timestamp"] < END)].copy()
    if len(frame) != 2 * 7 * 24 * 12:
        raise RuntimeError(f"expected 4,032 BTC/ETH rows, found {len(frame):,}")
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(20, 8), dpi=170, sharex=True, gridspec_kw={"hspace": 0.15})
    fig.patch.set_facecolor("#0b1018")
    for ax, asset in zip(axes, ("BTC", "ETH"), strict=True):
        part = frame[frame["asset"] == asset]
        ax.set_facecolor("#101824")
        ax.grid(True, color="#718096", alpha=0.16, lw=0.6)
        _shade(ax, part)
        ax.plot(part["timestamp"], part["close"], color="#f4f7fb", lw=1.0)
        ax.set_ylabel(f"{asset} close\n(USDT)")
        ax.set_title(f"{asset} — OOS 2026-03-25 to 03-31 · causal 5m HMM regime", loc="left", fontsize=14, pad=9)
        for spine in ax.spines.values():
            spine.set_color("#263449")
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("Mar %d"))
    axes[1].set_xlabel("2026 UTC timestamp")
    fig.text(0.99, 0.008, "Background: bull (green) · bear (red) · chop (yellow)", ha="right", va="bottom", color="#b8c4d4", fontsize=9)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.075)
    fig.savefig(OUT, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
