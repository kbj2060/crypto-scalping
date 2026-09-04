#!/usr/bin/env python3
"""User caught a real problem in render_eth_5m_sweep_v_rebound_3class_1h_candidate_20260830.py's
output: random sampling (seed=20260830) happened to draw all 4 "횡보/지지" examples from the
bottom of its 0.15-0.70 range (0.15-0.30), which looks visually identical to "지속" (0.02-0.15)
since they're adjacent on a smooth continuum -- not evidence the whole bucket is wrong, but not
a fair look at it either. This renders ONE STRATIFIED row spanning the giveback spectrum evenly
(by percentile, not randomly) across BOTH the sustained and plateau ranges together, so the
question "is there an actual visual transition somewhere, or is it a smooth blur end-to-end" can
be answered directly by eye. 60min horizon, same data already computed.
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
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_3class_design_1h_20260830/giveback_spectrum_stratified.png"
WINDOW_BARS = 12
SEED = 20260830
# evenly spaced target giveback values spanning the whole "attempted" range -- picks the CLOSEST
# real event to each target instead of a random draw, so the row shows a true ordered spectrum
TARGETS = [0.02, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.85]
T_SUSTAIN, T_FAIL = 0.15, 0.70


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_spectrum_20260830", IMPL_SCRIPT)
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
    att = events[events["attempted"]].copy()

    picked = []
    used = set()
    for target in TARGETS:
        cand = att.loc[~att.index.isin(used)].assign(dist=(att["giveback_ratio"] - target).abs())
        row = cand.sort_values("dist").iloc[0]
        used.add(row.name)
        picked.append(row)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 6.5), dpi=145)
    fig.suptitle(
        "giveback_ratio 스펙트럼(60min): 낮은값→높은값 순 정렬, 실제로 어디서 '모양'이 바뀌는지 육안 확인용 "
        f"(점선=T_sustain {T_SUSTAIN} / T_fail {T_FAIL})",
        fontsize=17, y=1.02,
    )

    for col, row in enumerate(picked):
        ax = axes[col]
        idx = int(row["candidate_index"])
        sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
        draw_candles(ax, sub, WINDOW_BARS, float(row["sweep_level"]))
        gb = row["giveback_ratio"]
        zone = "지속" if gb <= T_SUSTAIN else ("횡보/지지" if gb <= T_FAIL else "실패권")
        tint = {"지속": "#eef8f0", "횡보/지지": "#fff6e0", "실패권": "#fdf1ef"}[zone]
        ax.set_facecolor(tint)
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(f"[{zone}]\n{row['side']} | giveback={gb:.2f}", fontsize=12)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
