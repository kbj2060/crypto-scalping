#!/usr/bin/env python3
"""RESEARCH ONLY -- verifies the SAVED label contract
(tmp/causal_regen_20260516/eth_triple_barrier_maxdensity_20260728/label_contracts/
triple_barrier_direction_maxdensity_20260728/) rather than recomputing labels, so the chart
reflects the actual artifact that would be passed to a retrain.

Chart 1: reconstructs trades from the saved per-bar zigzag_action codes over the same 2-week
window used throughout this session (2025-01-06..01-20) -- contiguous runs of the same non-CASH
code become one "trade" span, matching how the trainer would see the label. Sanity-checks against
build_eth_triple_barrier_direction_labels_maxdensity_20260728.py's own console counts.

Chart 2: monthly trade-event density across the full saved TRAIN+OOS range (2024-01..2026-02),
to check for any period-specific gaps or anomalies before committing to a multi-seed retrain.

Chart only -- reads the saved label CSVs, writes no data, retrains nothing, touches no live file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_2024tape_20260727 as tape2024  # noqa: E402

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_triple_barrier_maxdensity_20260728/label_contracts/triple_barrier_direction_maxdensity_20260728"
WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"
OUT_PNG1 = ROOT / "tmp/research_20260728/chart_verify_saved_labels_2week.png"
OUT_PNG2 = ROOT / "tmp/research_20260728/chart_verify_saved_labels_monthly_density.png"

COLOR_PRICE, COLOR_LONG, COLOR_SHORT = "#9AA5B1", "#2C6FBB", "#B5651D"


def load_saved_labels() -> pd.DataFrame:
    parts = []
    for f in sorted(LABEL_DIR.glob("zigzag_action_labels_*.csv")):
        df = pd.read_csv(f, parse_dates=["timestamp"])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"loaded saved labels: {len(out)} rows [{out['timestamp'].min()}..{out['timestamp'].max()}]", flush=True)
    return out


def reconstruct_trades(labels: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    merged = labels.merge(prices[["timestamp", "close"]], on="timestamp", how="inner").reset_index(drop=True)
    action = merged["zigzag_action"].to_numpy()
    trades = []
    i = 0
    n = len(merged)
    while i < n:
        code = action[i]
        if code == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and action[j + 1] == code:
            j += 1
        trades.append({
            "entry_ts": merged["timestamp"].iloc[i], "exit_ts": merged["timestamp"].iloc[j],
            "entry_price": float(merged["close"].iloc[i]), "exit_price": float(merged["close"].iloc[j]),
            "label": "LONG" if code == 1 else "SHORT",
        })
        i = j + 1
    return pd.DataFrame(trades)


def chart1(labels: pd.DataFrame) -> None:
    train_all, _eval_df, _overlay = tape2024._load_omega_frames_2024tape()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= WINDOW_START) & (train_all["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    win_labels = labels[(labels["timestamp"] >= WINDOW_START) & (labels["timestamp"] <= WINDOW_END)]
    trades = reconstruct_trades(win_labels, frame)
    print(f"reconstructed {len(trades)} trades in {WINDOW_START}..{WINDOW_END} from the SAVED file "
          f"(LONG={int((trades['label']=='LONG').sum())} SHORT={int((trades['label']=='SHORT').sum())})", flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(frame["timestamp"], frame["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN)")
    long_t = trades[trades["label"] == "LONG"]
    short_t = trades[trades["label"] == "SHORT"]
    ax.scatter(long_t["entry_ts"], long_t["entry_price"], marker="^", s=10, color=COLOR_LONG, alpha=0.5, zorder=3, linewidth=0, label=f"LONG (n={len(long_t)})")
    ax.scatter(short_t["entry_ts"], short_t["entry_price"], marker="v", s=10, color=COLOR_SHORT, alpha=0.5, zorder=3, linewidth=0, label=f"SHORT (n={len(short_t)})")
    ax.set_title(f"VERIFICATION: trades reconstructed from the SAVED label file -- {WINDOW_START}..{WINDOW_END}\n"
                 f"{len(trades)} trades (should match build script's console output for this window)", fontsize=11)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG1)
    print(f"saved {OUT_PNG1}", flush=True)


def chart2(labels: pd.DataFrame) -> None:
    action = labels["zigzag_action"].to_numpy()
    ts = labels["timestamp"].to_numpy()
    trade_starts = []
    i = 0
    n = len(labels)
    while i < n:
        code = action[i]
        if code != 0:
            trade_starts.append(labels["timestamp"].iloc[i])
            j = i
            while j + 1 < n and action[j + 1] == code:
                j += 1
            i = j + 1
        else:
            i += 1
    starts = pd.Series(pd.to_datetime(trade_starts))
    monthly = starts.dt.to_period("M").value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    x = range(len(monthly))
    ax.bar(x, monthly.to_numpy(), color=COLOR_LONG, width=0.65, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha="right", fontsize=8)
    for i, v in enumerate(monthly.to_numpy()):
        ax.text(i, v + max(monthly) * 0.01, str(v), ha="center", fontsize=8)
    ax.set_ylabel("Trade-label events started")
    ax.set_title(f"Monthly trade-event density, SAVED label file (TRAIN 2024-2025 + OOS 2026) -- total {int(monthly.sum())} trades", fontsize=11)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PNG2)
    print(f"saved {OUT_PNG2}", flush=True)


def main() -> None:
    OUT_PNG1.parent.mkdir(parents=True, exist_ok=True)
    labels = load_saved_labels()
    chart1(labels)
    chart2(labels)


if __name__ == "__main__":
    main()
