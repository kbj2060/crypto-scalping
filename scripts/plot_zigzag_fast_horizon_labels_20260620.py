#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_fast_horizon_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_fast_horizon_20260620/charts"


def _plot_year(year: int, *, start: str | None = None, end: str | None = None, suffix: str) -> Path:
    df = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", parse_dates=["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"empty chart frame for {year} {start} {end}")

    active_long = df["zigzag_action"].astype(int) == 1
    active_short = df["zigzag_action"].astype(int) == 2
    cash = df["zigzag_action"].astype(int) == 0

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True, gridspec_kw={"height_ratios": [5, 1.35, 1.35]})
    ax = axes[0]
    ax.plot(df["timestamp"], df["close"], color="#1f2937", linewidth=0.75, label="close")
    ax.scatter(df.loc[active_long, "timestamp"], df.loc[active_long, "close"], s=4, color="#16a34a", alpha=0.55, label="LONG label")
    ax.scatter(df.loc[active_short, "timestamp"], df.loc[active_short, "close"], s=4, color="#dc2626", alpha=0.55, label="SHORT label")
    ax.set_title(f"Fast-horizon zigzag labels {year} {suffix}")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncols=3, fontsize=9)

    state = axes[1]
    state.fill_between(df["timestamp"], 0, 1, where=active_long.to_numpy(), color="#16a34a", alpha=0.55, step="pre", label="LONG")
    state.fill_between(df["timestamp"], 0, 1, where=active_short.to_numpy(), color="#dc2626", alpha=0.55, step="pre", label="SHORT")
    state.fill_between(df["timestamp"], 0, 1, where=cash.to_numpy(), color="#9ca3af", alpha=0.35, step="pre", label="CASH")
    state.set_yticks([])
    state.set_ylabel("Label")
    state.grid(True, axis="x", alpha=0.18)

    hold = axes[2]
    active = ~cash
    hold.plot(df["timestamp"], df["zigzag_fast_hold_bars"], color="#2563eb", linewidth=0.65)
    hold.scatter(df.loc[active, "timestamp"], df.loc[active, "zigzag_fast_hold_bars"], s=3, color="#2563eb", alpha=0.35)
    hold.set_ylabel("Target hold bars")
    hold.set_xlabel("Time")
    hold.grid(True, alpha=0.22)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_suffix = suffix.lower().replace(" ", "_").replace("/", "_")
    out = OUT_DIR / f"zigzag_fast_horizon_labels_{year}_{safe_suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    outputs = []
    outputs.append(_plot_year(2025, suffix="full"))
    outputs.append(_plot_year(2026, suffix="full"))
    outputs.append(_plot_year(2025, start="2025-10-01", end="2025-10-14", suffix="zoom_2025_oct01_oct14"))
    outputs.append(_plot_year(2026, start="2026-01-01", end="2026-01-14", suffix="zoom_2026_jan01_jan14"))
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
