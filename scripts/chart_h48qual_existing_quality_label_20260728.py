#!/usr/bin/env python3
"""RESEARCH ONLY -- chart the EXISTING h48qual QUALITY label (h48_conservative triple-barrier
config from scripts/build_omega1_2_triple_barrier_labels_20260619.py, built 2026-06-19, already
in production as h48qual's --quality-label-dir), for direct visual comparison against today's new
DIRECTION label charts.

Uses the RAW pre-padding build output (train_triple_barrier_labels.csv), not the padded/collapsed
production file, because the raw output keeps each row's own tb_long_bars/tb_short_bars/
tb_long_reason/tb_short_reason -- letting individual trade brackets be drawn cleanly, the same way
chart_eth_triple_barrier_label_ground_truth_20260728.py drew v1's independent-per-bar events.
Config: horizon=48 bars (4h, NOT 48h -- corrected naming understanding from this session),
tp_mult=1.2, sl_mult=0.8, min_tp=0.006, min_sl=0.004, ATR-adaptive. Same window as every other
chart this session (2025-01-06..01-20) for direct comparability.

Chart only -- reads the existing saved file, writes no data, retrains nothing, touches no live file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
SRC = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619/train_triple_barrier_labels.csv"
WINDOW_START, WINDOW_END = "2025-01-06", "2025-01-20"
OUT_PNG = ROOT / "tmp/research_20260728/chart_h48qual_existing_quality_label.png"

CFG = "h48_conservative"
COLOR_PRICE, COLOR_LONG, COLOR_SHORT = "#9AA5B1", "#2C6FBB", "#B5651D"


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp", f"entry_timestamp_{CFG}", f"tb_action_{CFG}", f"tb_long_bars_{CFG}", f"tb_short_bars_{CFG}",
            f"tb_long_reason_{CFG}", f"tb_short_reason_{CFG}", f"tb_tp_price_move_{CFG}", f"tb_sl_price_move_{CFG}"]
    print("loading (large file, filtering to window)...", flush=True)
    chunks = []
    for chunk in pd.read_csv(SRC, usecols=cols, parse_dates=["timestamp", f"entry_timestamp_{CFG}"], chunksize=500_000):
        sub = chunk[(chunk["timestamp"] >= WINDOW_START) & (chunk["timestamp"] <= WINDOW_END)]
        if len(sub):
            chunks.append(sub)
        if chunk["timestamp"].iloc[-1] > pd.Timestamp(WINDOW_END):
            break
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols)
    df = df.rename(columns={f"entry_timestamp_{CFG}": "entry_timestamp"})
    print(f"rows in window: {len(df)}", flush=True)

    # Reload a price frame for plotting the close line and entry prices.
    price = pd.read_csv(ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv",
                         usecols=["timestamp", "close"], parse_dates=["timestamp"])
    price = price[(price["timestamp"] >= WINDOW_START) & (price["timestamp"] <= WINDOW_END)].reset_index(drop=True)
    df = df.merge(price.rename(columns={"timestamp": "entry_timestamp", "close": "entry_price"}), on="entry_timestamp", how="inner")

    trades = df[df[f"tb_action_{CFG}"] != 0].copy()
    trades["label"] = trades[f"tb_action_{CFG}"].map({1: "LONG", 2: "SHORT"})
    trades["bars"] = trades.apply(lambda r: r[f"tb_long_bars_{CFG}"] if r["label"] == "LONG" else r[f"tb_short_bars_{CFG}"], axis=1)
    trades["reason"] = trades.apply(lambda r: r[f"tb_long_reason_{CFG}"] if r["label"] == "LONG" else r[f"tb_short_reason_{CFG}"], axis=1)
    price_idx = price.set_index("timestamp")["close"]
    trades["exit_ts"] = trades["entry_timestamp"] + pd.to_timedelta(trades["bars"] * 5, unit="min")
    trades["exit_ts"] = trades["exit_ts"].apply(lambda t: price_idx.index[price_idx.index.searchsorted(t)] if t <= price_idx.index[-1] else price_idx.index[-1])

    n_long = int((trades["label"] == "LONG").sum())
    n_short = int((trades["label"] == "SHORT").sum())
    print(f"trades: LONG={n_long} SHORT={n_short} total={len(trades)}", flush=True)
    print(trades["reason"].value_counts().to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    ax.plot(price["timestamp"], price["close"].astype(float), color=COLOR_PRICE, linewidth=1.0, zorder=1, label="ETH close (TRAIN)")
    long_t = trades[trades["label"] == "LONG"]
    short_t = trades[trades["label"] == "SHORT"]
    ax.scatter(long_t["entry_timestamp"], long_t["entry_price"], marker="^", s=11, color=COLOR_LONG, alpha=0.5, zorder=3, linewidth=0, label=f"LONG (n={n_long})")
    ax.scatter(short_t["entry_timestamp"], short_t["entry_price"], marker="v", s=11, color=COLOR_SHORT, alpha=0.5, zorder=3, linewidth=0, label=f"SHORT (n={n_short})")

    ax.set_title(f"EXISTING h48qual QUALITY label (h48_conservative, built 2026-06-19) -- {WINDOW_START}..{WINDOW_END}\n"
                 f"horizon=48 bars (4h), tp_mult=1.2, sl_mult=0.8, min_tp=0.006, min_sl=0.004 (ATR-adaptive), "
                 f"independent per-bar (not sequential) -- LONG={n_long} SHORT={n_short}", fontsize=10.5)
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
