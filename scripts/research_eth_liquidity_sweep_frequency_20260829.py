#!/usr/bin/env python3
"""ETH 5m dashboard evidence signal `liquidity_sweep` — raw occurrence frequency, 2024-01-01 to latest.

Counts only the raw sweep condition (no outcome classification). The formula is reused
unmodified (via direct import, not reimplemented) from
scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py::add_causal_columns, which
itself restates the live dashboard's evidence-signal `liquidity_sweep` definition:
  - downside sweep: bar low < prior 48-bar (causal, shifted) swing low, and close reclaims above it
  - upside sweep:   bar high > prior 48-bar (causal, shifted) swing high, and close reclaims below it
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_liquidity_sweep_frequency_20260829"
KST_OFFSET = pd.Timedelta(hours=9)


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_followthrough_v2_impl", V2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    impl = load_sweep_impl()
    frame = impl.add_causal_columns(impl.load_5m(SOURCE))

    frame["sweep_low_hit"] = (frame["low"] < frame["sweep_level_low"]) & (frame["close"] > frame["sweep_level_low"])
    frame["sweep_high_hit"] = (frame["high"] > frame["sweep_level_high"]) & (frame["close"] < frame["sweep_level_high"])
    frame["any_sweep"] = frame["sweep_low_hit"] | frame["sweep_high_hit"]

    total_bars = int(len(frame))
    total_low = int(frame["sweep_low_hit"].sum())
    total_high = int(frame["sweep_high_hit"].sum())
    total_any = int(frame["any_sweep"].sum())
    both_same_bar = int((frame["sweep_low_hit"] & frame["sweep_high_hit"]).sum())
    start, end = frame["timestamp"].min(), frame["timestamp"].max()
    days = (end - start).total_seconds() / 86400

    kst_ts = frame["timestamp"] + KST_OFFSET
    monthly = (
        frame.assign(month=kst_ts.dt.to_period("M").astype(str))
        .groupby("month")[["sweep_low_hit", "sweep_high_hit", "any_sweep"]]
        .sum()
        .astype(int)
        .reset_index()
    )
    last_month_mask = kst_ts.dt.to_period("M").astype(str) == monthly["month"].iloc[-1]
    last_month_days = int(kst_ts.loc[last_month_mask].dt.date.nunique())

    summary = {
        "sweep_definition": "dashboard liquidity_sweep, reused unmodified from build_eth_5m_sweep_followthrough_v2_labels_20260829.py::add_causal_columns (48-bar causal swing wick + close reclaim)",
        "source": str(SOURCE.relative_to(ROOT)),
        "period_utc": {
            "start": str(start),
            "end": str(end),
            "total_closed_5m_bars": total_bars,
            "approx_days": round(days, 1),
        },
        "raw_sweep_counts": {
            "sweep_low (bottom / support-side stop-hunt)": total_low,
            "sweep_high (top / resistance-side stop-hunt)": total_high,
            "any_sweep (union)": total_any,
            "both_same_bar (rare simultaneous)": both_same_bar,
        },
        "rate": {
            "any_sweep_pct_of_bars": round(100 * total_any / total_bars, 2),
            "any_sweep_per_day": round(total_any / days, 2),
            "sweep_low_per_day": round(total_low / days, 2),
            "sweep_high_per_day": round(total_high / days, 2),
        },
        "last_month_partial_days": last_month_days,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    monthly.to_csv(OUT_DIR / "monthly_counts.csv", index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(monthly.to_string(index=False))

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 27,
        "axes.labelsize": 23,
        "xtick.labelsize": 16,
        "ytick.labelsize": 19,
        "legend.fontsize": 19,
    })
    fig, ax = plt.subplots(figsize=(32, 18), dpi=150)
    x = list(range(len(monthly)))
    width = 0.4
    low_colors = ["#2E86AB"] * len(monthly)
    high_colors = ["#C73E1D"] * len(monthly)
    low_colors[-1] = "#9FC8DE"
    high_colors[-1] = "#E8A99A"
    ax.bar([i - width / 2 for i in x], monthly["sweep_low_hit"], width=width,
           label=f"sweep_low (bottom, total {total_low:,})", color=low_colors)
    ax.bar([i + width / 2 for i in x], monthly["sweep_high_hit"], width=width,
           label=f"sweep_high (top, total {total_high:,})", color=high_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month"], rotation=45, ha="right")
    ax.set_title(
        f"ETH 5m dashboard liquidity_sweep — monthly occurrences, {monthly['month'].iloc[0]} to "
        f"{monthly['month'].iloc[-1]} (KST, total {total_any:,} events over {days:.0f} days)"
    )
    ax.set_xlabel("month (KST)")
    ax.set_ylabel("event count")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(
        f"partial month\n(first {last_month_days}d only)",
        xy=(x[-1], max(monthly["sweep_low_hit"].iloc[-1], monthly["sweep_high_hit"].iloc[-1])),
        xytext=(x[-1] - 3, monthly[["sweep_low_hit", "sweep_high_hit"]].to_numpy().max() * 0.85),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=16, color="dimgray",
    )
    fig.tight_layout()
    chart_path = OUT_DIR / "monthly_sweep_counts.png"
    fig.savefig(chart_path)
    print(f"chart saved: {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
