#!/usr/bin/env python3
"""Visual validation for the v5 binary label candidate (T=0.55): close-confirmed fast attempt +
giveback_ratio<=0.55 over the full 30min window -> V_REBOUND=1, else 0. 5+5 random examples,
same style as render_eth_5m_sweep_v_rebound_label_v4_examples_20260830.py.
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
EVENTS_CSV = ROOT / "tmp/eth_sweep_v_rebound_label_v5_binary_20260830/events_with_v5_inputs.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v5_binary_20260830/v5_label_examples.png"
WINDOW_BARS = 6
N_PER_LABEL = 5
SEED = 20260830
T = 0.55
PCTS = [0.10, 0.35, 0.65, 0.90, 0.99]


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v5_render_20260830", IMPL_SCRIPT)
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
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def pick_by_percentile(pool: pd.DataFrame, sort_col: str, pcts: list[float]) -> pd.DataFrame:
    sorted_pool = pool.sort_values(sort_col).reset_index(drop=True)
    idxs = sorted([min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts])
    return sorted_pool.iloc[idxs]


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    events = pd.read_csv(EVENTS_CSV)

    new_label = (events["close_attempted"] & (events["giveback_ratio_v5"] <= T)).astype(int)
    events["v5_label"] = new_label

    pos = events[events["v5_label"] == 1]
    neg = events[events["v5_label"] == 0]
    pos_sample = pick_by_percentile(pos, "giveback_ratio_v5", PCTS)
    # neg population mixes close_attempted-but-failed (has a giveback value) and never-close-attempted
    # (giveback may be noisy/irrelevant) -- sample half from each so both failure modes are shown
    neg_attempted = neg[neg["close_attempted"]]
    neg_never = neg[~neg["close_attempted"]]
    neg_sample = pd.concat([
        pick_by_percentile(neg_attempted, "giveback_ratio_v5", [0.3, 0.6, 0.9]),
        neg_never.sample(n=2, random_state=SEED),
    ])

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, N_PER_LABEL, figsize=(34, 15), dpi=145)
    fig.suptitle(
        f"v5 binary label candidate (T={T}, close-confirmed 15min attempt + giveback<=T over 30min) "
        "-- row1: V_REBOUND=1 | row2: V_REBOUND=0 (3 gave back too much + 2 never really attempted)",
        fontsize=19, y=0.998,
    )

    for row_i, (sample, tint, label_name) in enumerate((
        (pos_sample, "#eef8f0", "V_REBOUND=1"),
        (neg_sample, "#fdf1ef", "V_REBOUND=0"),
    )):
        for col, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row_i][col]
            idx = int(event["candidate_index"])
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["sweep_level"]))
            ax.set_facecolor(tint)
            ticks = list(range(0, len(sub), 2))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            gb = event["giveback_ratio_v5"]
            gb_str = f"{gb:.2f}" if pd.notna(gb) else "n/a"
            att = "close-attempt O" if event["close_attempted"] else "close-attempt X"
            ax.set_title(f"{label_name} | {event['side']} | {att}\ngiveback={gb_str}", fontsize=11)
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
