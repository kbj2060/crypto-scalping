#!/usr/bin/env python3
"""1-hour-horizon version of render_eth_5m_sweep_v_rebound_3class_candidate_20260830.py -- same
(T_sustain=0.15, T_fail=0.70) candidate thresholds, same 4-group layout, but giveback_ratio/peak/
end are now measured over 12 bars (60min) instead of 6 (30min), and the viewing window widens to
+/-60min so the viewer sees the exact window the classification used.
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
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_3class_design_1h_20260830/3class_candidate_examples_1h.png"
WINDOW_BARS = 12  # +/-60min view, matches the 60min lookahead used for the classification
SEED = 20260830
T_SUSTAIN, T_FAIL = 0.15, 0.70


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_3class_1h_render_20260830", IMPL_SCRIPT)
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


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(EVENTS_CSV)

    sustained = events[events["attempted"] & (events["giveback_ratio"] <= T_SUSTAIN)]
    plateau = events[events["attempted"] & (events["giveback_ratio"] > T_SUSTAIN) & (events["giveback_ratio"] <= T_FAIL)]
    failed_never = events[~events["attempted"]]
    failed_gaveback = events[events["attempted"] & (events["giveback_ratio"] > T_FAIL)]

    groups = [
        ("지속 (giveback<=0.15, 60min)", sustained.sample(n=4, random_state=SEED), "#eef8f0"),
        ("횡보/지지 (0.15<giveback<=0.70, 60min)", plateau.sample(n=4, random_state=SEED), "#fff6e0"),
        ("실패-미도달 (fast target never hit)", failed_never.sample(n=2, random_state=SEED), "#fdf1ef"),
        ("실패-반납 (giveback>0.70, 60min)", failed_gaveback.sample(n=2, random_state=SEED), "#fdf1ef"),
    ]

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 4, figsize=(30, 22), dpi=145)
    fig.suptitle(
        f"1시간(60min) 지평 3-class 후보 (T_sustain={T_SUSTAIN}, T_fail={T_FAIL}): row1=지속, "
        "row2=횡보/지지, row3-4=실패 (미도달 2건 + 반납 2건) -- 30분 버전과 같은 임계값, 관찰창만 확장",
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

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
