#!/usr/bin/env python3
"""RESEARCH ONLY -- corrected verification chart. The previous verification script
(chart_verify_saved_triple_barrier_labels_20260728.py) reconstructed trades by scanning the
flattened per-bar zigzag_action array for contiguous same-code runs, which silently MERGES two
separate consecutive same-direction events into one whenever they touch with no CASH bar between
them (proven 2026-07-28: raw event count for 2025-01-06..01-20 on the 2024tape source is
LONG=196/SHORT=231/total=427, but the merge-based reconstruction undercounts it to 296/300).

This script instead re-runs the SAME build_events() used to produce the saved label file, scoped
to the chart window, so individual trade markers reflect the true event count -- no merge-based
reconstruction from the flattened array.

Chart only -- reads/recomputes against the same source as the saved file, writes no data,
retrains nothing, touches no live file.
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

import build_eth_triple_barrier_direction_labels_maxdensity_20260728 as build  # noqa: E402
import train_eval_omega4_3head_parent72_pinned102_2024tape_20260727 as tape2024  # noqa: E402

WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"
OUT_PNG = ROOT / "tmp/research_20260728/chart_verify_saved_labels_fixed.png"

COLOR_PRICE, COLOR_LONG, COLOR_SHORT = "#9AA5B1", "#2C6FBB", "#B5651D"


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    train_all, _eval_df, _overlay = tape2024._load_omega_frames_2024tape()
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"])
    frame = train_all[(train_all["timestamp"] >= WINDOW_START) & (train_all["timestamp"] <= WINDOW_END)].reset_index(drop=True)

    events = build.build_events(frame)
    edf = pd.DataFrame(events)
    edf["entry_ts"] = frame["timestamp"].iloc[edf["t"]].to_numpy()
    edf["entry_price"] = frame["close"].astype(float).iloc[edf["t"]].to_numpy()
    trades = edf[edf["label"] != "CASH"].reset_index(drop=True)
    n_long = int((trades["label"] == "LONG").sum())
    n_short = int((trades["label"] == "SHORT").sum())
    print(f"true event count (same source/config as the saved file): LONG={n_long} SHORT={n_short} total={len(trades)}", flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(frame["timestamp"], frame["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN, 2024+2025 tape)")
    long_t = trades[trades["label"] == "LONG"]
    short_t = trades[trades["label"] == "SHORT"]
    ax.scatter(long_t["entry_ts"], long_t["entry_price"], marker="^", s=11, color=COLOR_LONG, alpha=0.55, zorder=3, linewidth=0, label=f"LONG (n={n_long})")
    ax.scatter(short_t["entry_ts"], short_t["entry_price"], marker="v", s=11, color=COLOR_SHORT, alpha=0.55, zorder=3, linewidth=0, label=f"SHORT (n={n_short})")
    ax.set_title(f"Final label answer key (CORRECTED count) -- {WINDOW_START}..{WINDOW_END}\n"
                 f"min_tp=0.006, min_sl=0.0032, vertical=1h, sequential -- {len(trades)} true trades "
                 f"(this is the file actually saved for retraining)", fontsize=11)
    ax.set_ylabel("ETH price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.15, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, markerscale=2.5)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print(f"saved {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
