#!/usr/bin/env python3
"""Visual sanity check for build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py's labels.

Renders 10 random examples per label (V_REBOUND / NO_V_REBOUND) as small candlestick panels
covering +/-30 minutes around the sweep bar, with the swept level and sweep bar marked, so
the label can be checked by eye against the actual price action -- one combined PNG.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_PATH = ROOT / "tmp/eth_liquidity_sweep_frequency_20260829/v_rebound_label_examples.png"
WINDOW_BARS = 6  # 30 minutes each side at 5m bars
N_PER_LABEL = 10
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_NAMES = {0: "NO_V_REBOUND", 1: "V_REBOUND"}
LABEL_TINT = {0: "#fdf1ef", 1: "#eef8f0"}


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_render_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, N_PER_LABEL // 2, figsize=(38, 22), dpi=140)
    fig.suptitle(
        "ETH 5m liquidity_sweep -> V_REBOUND label sanity check "
        f"(random {N_PER_LABEL} examples per class, +/-30min around sweep bar, seed={SEED})",
        fontsize=24, y=0.995,
    )

    for label, row_offset in ((0, 0), (1, 2)):
        sample = labels[labels["label"] == label].sample(n=N_PER_LABEL, random_state=SEED)
        for panel, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row_offset + panel // (N_PER_LABEL // 2)][panel % (N_PER_LABEL // 2)]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
            ax.set_facecolor(LABEL_TINT[label])
            ticks = list(range(0, len(sub), 2))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
            ax.set_title(
                f"{LABEL_NAMES[label]} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST | "
                f"reb={event['rebound_atr_multiple']:.2f}x ATR",
                fontsize=12,
            )
            ax.grid(alpha=0.25)

    for row in axes[:, 0]:
        row.set_ylabel("price", fontsize=11)
    for row in axes[-1]:
        row.set_xlabel("minutes from sweep bar", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
