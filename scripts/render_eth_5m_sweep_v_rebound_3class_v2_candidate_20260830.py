#!/usr/bin/env python3
"""Adjusted-threshold validation (2026-08-30, user feedback on the first candidate chart): raises
T_sustain 0.15->0.22 and lowers T_fail 0.70->0.62, based on the ordered giveback-spectrum chart
showing the real "clearly sustained" vs "clearly plateaued" transition sits higher than 0.15, and
0.65+ already looks failure-like.

Also fixes the SAMPLING methodology the user's catch exposed: the previous chart drew 4 random
examples per class and, by bad luck, all 4 "plateau" draws landed in the bottom quartile of that
class's range, making it look identical to "sustained". This version picks each class's examples
at fixed PERCENTILES of its own giveback_ratio range (10/40/70/90) instead of random.sample, so
every render of this script shows the class's actual spread, not a lucky/unlucky random draw.

60min horizon (events_with_3class_inputs_1h.csv, already computed).
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
EVENTS_CSV = ROOT / "tmp/eth_sweep_v_rebound_3class_design_1h_20260830/events_with_3class_inputs_1h.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_3class_design_1h_20260830/3class_v2_candidate_examples.png"
WINDOW_BARS = 12
T_SUSTAIN, T_FAIL = 0.22, 0.62
PCTS = [0.10, 0.40, 0.70, 0.90]  # stratified sampling points within each class's own range


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_3class_v2_20260830", IMPL_SCRIPT)
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


def pick_by_percentile(pool: pd.DataFrame, pcts: list[float]) -> pd.DataFrame:
    sorted_pool = pool.sort_values("giveback_ratio").reset_index(drop=True)
    idxs = [min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts]
    return sorted_pool.iloc[idxs]


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(EVENTS_CSV)

    sustained = events[events["attempted"] & (events["giveback_ratio"] <= T_SUSTAIN)]
    plateau = events[events["attempted"] & (events["giveback_ratio"] > T_SUSTAIN) & (events["giveback_ratio"] <= T_FAIL)]
    failed_never = events[~events["attempted"]]
    failed_gaveback = events[events["attempted"] & (events["giveback_ratio"] > T_FAIL)]

    n = len(events)
    print(f"new thresholds (T_sustain={T_SUSTAIN}, T_fail={T_FAIL}): "
          f"지속 {len(sustained)/n:.1%} ({len(sustained)}) | 횡보/지지 {len(plateau)/n:.1%} ({len(plateau)}) | "
          f"실패 {(len(failed_never)+len(failed_gaveback))/n:.1%} ({len(failed_never)+len(failed_gaveback)})")

    groups = [
        (f"지속 (giveback<={T_SUSTAIN})", pick_by_percentile(sustained, PCTS), "#eef8f0"),
        (f"횡보/지지 ({T_SUSTAIN}<giveback<={T_FAIL})", pick_by_percentile(plateau, PCTS), "#fff6e0"),
        ("실패-미도달 (fast target never hit)", pick_by_percentile(failed_never.assign(giveback_ratio=failed_never["giveback_ratio"].fillna(0)), [0.25, 0.75]), "#fdf1ef"),
        (f"실패-반납 (giveback>{T_FAIL})", pick_by_percentile(failed_gaveback, [0.25, 0.75]), "#fdf1ef"),
    ]

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 4, figsize=(30, 22), dpi=145)
    fig.suptitle(
        f"조정된 3-class 후보 (T_sustain={T_SUSTAIN}, T_fail={T_FAIL}, 60min) -- 각 클래스 percentile "
        "10/40/70/90 지점에서 뽑음(랜덤 아님, 전 구간 대표성 확보)",
        fontsize=19, y=0.995,
    )

    for row, (title, sample, tint) in enumerate(groups):
        for col, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row][col]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
            ax.set_facecolor(tint)
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            gb = event["giveback_ratio"]
            gb_str = f"{gb:.2f}" if pd.notna(gb) else "n/a"
            ax.set_title(f"{title}\n{event['side']} | giveback={gb_str}", fontsize=10)
            ax.grid(alpha=0.25)
        for col in range(len(sample), 4):
            axes[row][col].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
