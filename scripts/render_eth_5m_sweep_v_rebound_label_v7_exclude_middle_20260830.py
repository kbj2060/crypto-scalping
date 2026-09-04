#!/usr/bin/env python3
"""v7 (2026-08-30, user proposal): drop the fuzzy middle entirely instead of drawing a hard line
through it. label=1 (V자반등) unchanged from v6 (close-confirmed 30min attempt + giveback<=0.20
over 60min). label=0 (지지/횡보) is now ONLY the "never close-attempted at all" population --
the "attempted but gave back too much" middle (41.5% of all events) is EXCLUDED from training
entirely, not forced into either class. Matches this repo's existing near/mid/far tertile
precedent for genuinely ambiguous cases (don't force a binary split through fuzzy territory).

20 examples: 10 V자반등 (unchanged percentile sample from v6) + 10 지지/횡보 (도달X), the latter
stratified by how close the fast-window close move got to the 1.5x-ATR target (as a fraction of
target), from 0% (flat/no attempt) to just under 100% (barely missed), so the sample represents
the full range of "never attempted" rather than a lucky/unlucky random draw.
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
EVENTS_CSV = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/events_with_v6_inputs.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/v7_exclude_middle_examples_20.png"
WINDOW_BARS = 12
T_SUSTAIN = 0.20
FAST_BARS = 6
ATR_MULT = 1.5


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v7_20260830", IMPL_SCRIPT)
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
    never = events[~events["close_attempted"]].copy()

    # how close did the fast-window close move get to the 1.5x-ATR target, as a fraction (0=flat, ~1=barely missed)
    frac = []
    for _, ev in never.iterrows():
        idx = int(ev["candidate_index"])
        row = frame.iloc[idx]
        fast_future = frame.iloc[idx + 1: idx + FAST_BARS + 1]
        atr = ev["atr"]
        if ev["side"] == "downside":
            move = fast_future["close"].max() - row["low"]
        else:
            move = row["high"] - fast_future["close"].min()
        frac.append(float(move / (ATR_MULT * atr)))
    never["fast_move_frac_of_target"] = frac

    v_sample = pick_by_percentile(v_rebound, "giveback_ratio_v6",
                                   [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    never_sample = pick_by_percentile(never, "fast_move_frac_of_target",
                                       [0.05, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.99])

    n = len(events)
    print(f"V자반등: {len(v_rebound)} ({len(v_rebound)/n:.1%})  지지/횡보(도달X만): {len(never)} ({len(never)/n:.1%})  "
          f"제외: {n-len(v_rebound)-len(never)} ({(n-len(v_rebound)-len(never))/n:.1%})")

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 5, figsize=(34, 22), dpi=145)
    fig.suptitle(
        "v7 (지지/횡보=도달X만 사용, 도달O+giveback>0.2는 학습에서 제외): "
        "row1-2=V자반등(1, 10건 percentile순, v6와 동일) | row3-4=지지/횡보(0, 도달X 10건, 목표가근접도순)",
        fontsize=18, y=0.995,
    )

    flat_axes = axes.flatten()
    for i, (_, event) in enumerate(pd.concat([v_sample, never_sample]).iterrows()):
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
            label_txt = f"지지/횡보(도달X) | {event['side']}\n목표가근접도={event['fast_move_frac_of_target']:.2f}"
        ax.set_title(label_txt, fontsize=10)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
