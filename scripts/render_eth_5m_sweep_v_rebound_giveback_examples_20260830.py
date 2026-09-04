#!/usr/bin/env python3
"""Visual check for the giveback_ratio split proposed in research_eth_sweep_v_rebound_giveback_
pattern_20260830.py: does LOW giveback_ratio actually look like the user's pattern-1 sketch
(sustained V, keeps extending) and HIGH giveback_ratio look like pattern-3 (spike then plateau/
support), both drawn from the CURRENTLY-MERGED V_REBOUND=1 population? Diagnostic only, no label
file touched.
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
EVENTS_CSV = ROOT / "tmp/eth_sweep_v_rebound_giveback_pattern_20260830/events_with_giveback_ratio.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_giveback_pattern_20260830/giveback_pattern_examples.png"
WINDOW_BARS = 6
N_PER_GROUP = 5
SEED = 20260830
KST_OFFSET = pd.Timedelta(hours=9)
LOW_MAX = 0.08     # candidate "pattern 1, sustained" cutoff
HIGH_MIN = 0.55    # candidate "pattern 3, plateau" cutoff


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_giveback_render_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
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
    events = pd.read_csv(EVENTS_CSV)
    v1 = events[events["label"] == 1]

    low_group = v1[v1["giveback_ratio"] <= LOW_MAX].sample(n=N_PER_GROUP, random_state=SEED)
    high_group = v1[v1["giveback_ratio"] >= HIGH_MIN].sample(n=N_PER_GROUP, random_state=SEED)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, N_PER_GROUP, figsize=(34, 15), dpi=145)
    fig.suptitle(
        "V_REBOUND=1 events split by giveback_ratio, same 30min window as before -- top row: "
        f"giveback<={LOW_MAX} ('pattern 1' candidate, sustained) | bottom row: giveback>={HIGH_MIN} "
        "('pattern 3' candidate, spike-then-plateau)",
        fontsize=20, y=0.998,
    )

    for row_i, (group, tint) in enumerate(((low_group, "#eef8f0"), (high_group, "#fff6e0"))):
        for col, (_, event) in enumerate(group.iterrows()):
            ax = axes[row_i][col]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
            ax.set_facecolor(tint)
            ticks = list(range(0, len(sub), 2))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
            ax.set_title(
                f"{event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST\ngiveback={event['giveback_ratio']:.2f}",
                fontsize=12,
            )
            ax.grid(alpha=0.25)

    for row in axes[:, 0]:
        row.set_ylabel("price", fontsize=11)
    for row in axes[-1]:
        row.set_xlabel("minutes from sweep bar", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
