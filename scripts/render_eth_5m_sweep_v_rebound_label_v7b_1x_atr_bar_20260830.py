#!/usr/bin/env python3
"""v7b (2026-08-30, user proposal): tighten 지지/횡보(0)'s bar from "<1.5x ATR within 30min" to
"<1.0x ATR within 30min" -- pulls the near-miss cases (1.0x-1.5x ATR, e.g. the three boundary
cases found at 87-96% of the 1.5x target) OUT of 지지/횡보 and into the excluded middle, leaving
지지/횡보 as an even more clearly "genuinely nothing happened" population.

V자반등(1) unchanged: close-confirmed 1.5x-ATR within 30min AND giveback<=0.20 over 60min.
지지/횡보(0): fast_move_atr_mult < 1.0 (NEW, was < 1.5 via close_attempted==False).
Excluded: everything else (1.0x-1.5x reached but not held per giveback, or 1.5x+ reached but
giveback>0.20).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
EVENTS_CSV = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/events_with_v7b_inputs.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/v7b_1x_atr_examples_20.png"
WINDOW_BARS = 12
T_SUSTAIN = 0.20


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v7b_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def pick_by_percentile(pool: pd.DataFrame, sort_col: str, pcts: list[float]) -> pd.DataFrame:
    sorted_pool = pool.sort_values(sort_col).reset_index(drop=True)
    idxs = sorted({min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts})
    return sorted_pool.iloc[idxs]


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(EVENTS_CSV)

    v_rebound = events[events["close_attempted"] & (events["giveback_ratio_v6"] <= T_SUSTAIN)]
    support_chop = events[events["fast_move_atr_mult"] < 1.0]

    n = len(events)
    print(f"V자반등: {len(v_rebound)} ({len(v_rebound)/n:.1%})  지지/횡보(<1.0x ATR): {len(support_chop)} "
          f"({len(support_chop)/n:.1%})  제외: {n-len(v_rebound)-len(support_chop)} "
          f"({(n-len(v_rebound)-len(support_chop))/n:.1%})")

    v_sample = pick_by_percentile(v_rebound, "giveback_ratio_v6",
                                   [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    chop_sample = pick_by_percentile(support_chop, "fast_move_atr_mult",
                                      [0.02, 0.15, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99])

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 5, figsize=(34, 22), dpi=145)
    fig.suptitle(
        "v7b (지지/횡보 문턱 1.5x->1.0x ATR로 강화, 그 사이는 제외): "
        "row1-2=V자반등(1, 10건 percentile순, 불변) | row3-4=지지/횡보(0, 10건, fast_move/ATR순)",
        fontsize=18, y=0.995,
    )

    flat_axes = axes.flatten()
    for i, (_, event) in enumerate(pd.concat([v_sample, chop_sample]).iterrows()):
        ax = flat_axes[i]
        idx = int(event["candidate_index"])
        sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
        draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
        is_v = i < len(v_sample)
        ax.set_facecolor("#eef8f0" if is_v else "#fdf1ef")
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        if is_v:
            label_txt = f"V자반등 | {event['side']}\ngiveback={event['giveback_ratio_v6']:.2f}"
        else:
            label_txt = f"지지/횡보 | {event['side']}\nfast_move={event['fast_move_atr_mult']:.2f}x ATR"
        ax.set_title(label_txt, fontsize=10)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
