#!/usr/bin/env python3
"""Visual sanity check for the v4 label fix (build_eth_5m_liquidity_sweep_v_rebound_labels_
20260829.py, 2026-08-30 user code review): pre-sweep ATR (not self-inclusive) + the 1.5x-ATR
move must arrive within the first V_REBOUND_FAST_BARS (15min), not anywhere in the 30min window.

Extends the original render_eth_5m_sweep_v_rebound_label_examples_20260829.py (which only drew
the swept level + sweep bar) with the two new things the fix actually changed, drawn directly on
each panel so the label can be checked by eye against the mechanism, not just the outcome:
  - a shaded band over the first V_REBOUND_FAST_BARS bars (the ONLY window the ATR-move is
    allowed to arrive in)
  - a target price line at sweep_extreme +/- 1.5x pre-sweep ATR (event's own stored "atr" column,
    which after the v4 fix already IS the pre-sweep value -- read directly, not recomputed)

10 total examples (5 V_REBOUND + 5 NO_V_REBOUND, random, matching the original script's
per-class sampling spirit), one combined PNG.
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
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v4_examples_20260830/v_rebound_label_v4_examples.png"
WINDOW_BARS = 6            # 30 minutes each side at 5m bars, same viewing window as the v3 script
FAST_BARS = 3              # v4: V_REBOUND_FAST_BARS -- the move must land inside this many bars
N_PER_LABEL = 5            # 5+5 = 10 total, per user request
SEED = 20260830
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_NAMES = {0: "NO_V_REBOUND", 1: "V_REBOUND"}
LABEL_TINT = {0: "#fdf1ef", 1: "#eef8f0"}


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_render_v4_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float,
                  target_level: float, is_downside: bool) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"])
        highs.append(bar["high"])

    # v4: shaded band = the ONLY window the 1.5x-ATR move is allowed to arrive in (15 minutes)
    ax.axvspan(sweep_pos + 0.5, sweep_pos + FAST_BARS + 0.5, color="#f2c14e", alpha=0.22, zorder=0,
               label="fast window (15min, v4)")
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1, label="swept level")
    ax.axhline(target_level, color="#7b2cbf", linestyle="-.", linewidth=1.3, zorder=1,
               label="1.5x pre-sweep ATR target")
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs + [target_level]) - min(lows + [target_level])) * 0.08 or 1.0
    ax.set_ylim(min(lows + [target_level]) - pad, max(highs + [target_level]) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, N_PER_LABEL, figsize=(34, 15), dpi=145)
    fig.suptitle(
        "ETH 5m liquidity_sweep -> V_REBOUND label v4 sanity check (2026-08-30 fix: pre-sweep "
        f"ATR + move must land in first {FAST_BARS*5}min) -- random {N_PER_LABEL} examples per "
        f"class, +/-30min around sweep bar, seed={SEED}",
        fontsize=22, y=0.998,
    )

    for label, row in ((1, 0), (0, 1)):
        sample = labels[labels["label"] == label].sample(n=N_PER_LABEL, random_state=SEED)
        for col, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row][col]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            is_down = event["side"] == "downside"
            sweep_row = frame.iloc[idx]
            target_level = (
                float(sweep_row["low"]) + 1.5 * float(event["atr"]) if is_down
                else float(sweep_row["high"]) - 1.5 * float(event["atr"])
            )
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]), target_level, is_down)
            ax.set_facecolor(LABEL_TINT[label])
            ticks = list(range(0, len(sub), 2))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
            ax.set_title(
                f"{LABEL_NAMES[label]} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST\n"
                f"reb={event['rebound_atr_multiple']:.2f}x pre-sweep ATR",
                fontsize=11,
            )
            ax.grid(alpha=0.25)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    for row in axes[:, 0]:
        row.set_ylabel("price", fontsize=11)
    for row in axes[-1]:
        row.set_xlabel("minutes from sweep bar", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
