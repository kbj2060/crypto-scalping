#!/usr/bin/env python3
"""Visual validation for v6 binary label: T_sustain=0.20, 30min fast window / 60min full window.
label=1 (V자반등) sampled by percentile across its OWN range; label=0 (지지/횡보) sampled to
cover its full heterogeneity: close-attempted-but-gave-back-too-much (by percentile) AND
never-close-attempted (random) -- learned from the earlier sampling-luck mistake, no plain
random.sample() on a wide range this time.
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
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/v6_label_examples.png"
WINDOW_BARS = 12  # +/-60min view
SEED = 20260830
T_SUSTAIN = 0.20


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v6_render_20260830", IMPL_SCRIPT)
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
    not_v_attempted = events[events["close_attempted"] & (events["giveback_ratio_v6"] > T_SUSTAIN)]
    not_v_never = events[~events["close_attempted"]]

    v_sample = pick_by_percentile(v_rebound, "giveback_ratio_v6", [0.1, 0.3, 0.5, 0.7, 0.9])
    notv_sample = pd.concat([
        pick_by_percentile(not_v_attempted, "giveback_ratio_v6", [0.2, 0.5, 0.8]),
        not_v_never.sample(n=2, random_state=SEED),
    ])

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 5, figsize=(34, 15), dpi=145)
    fig.suptitle(
        f"v6 binary label (T_sustain={T_SUSTAIN}, 30min fast-window/60min full-window): "
        "row1=V자반등(1, percentile 10/30/50/70/90) | row2=지지/횡보(0, 3 gave back too much + 2 never attempted)",
        fontsize=18, y=0.998,
    )

    for row_i, (sample, tint, name) in enumerate(((v_sample, "#eef8f0", "V자반등"), (notv_sample, "#fdf1ef", "지지/횡보"))):
        for col, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row_i][col]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
            ax.set_facecolor(tint)
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            gb = event["giveback_ratio_v6"]
            gb_str = f"{gb:.2f}" if pd.notna(gb) else "n/a"
            att = "30분내종가도달O" if event["close_attempted"] else "30분내종가도달X"
            ax.set_title(f"{name} | {event['side']} | {att}\ngiveback={gb_str}", fontsize=10)
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
