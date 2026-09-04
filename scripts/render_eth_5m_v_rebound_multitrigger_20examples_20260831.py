#!/usr/bin/env python3
"""Visual verification for the new 7-trigger V자반등 label (data/labels/eth_5m_v_rebound_
multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv). Per project convention
(feedback_visual_verification_chart_gate_explain_before_proceed), this MUST be reviewed and
explicitly approved by the user before any feature-building/TabPFN-training step proceeds.

Candlestick draw function (draw_candles) reused verbatim from render_eth_5m_sweep_v_rebound_
label_v6_final_20examples_20260830.py -- only the event marker's semantics changed (any of the 7
triggers' extreme price, not always a sweep level). Sampling scheme adapted from that script's
percentile-of-giveback_ratio convention, but stratified PER TRIGGER (3 examples each, low/mid/high
giveback within that trigger's own V자반등 subset) instead of one pooled sample -- the entire point
of this chart is to check each of the 7 trigger types' V자반등 calls individually, especially
local_extreme (the one genuinely new, never-before-visually-checked trigger, and the single
largest + highest-hit-rate contributor per the label build report).
"""
from __future__ import annotations

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

ROOT = Path("/home/kbj20/crypto-scalping")
ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABELS_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_labels.csv"
OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/multitrigger_v_rebound_21examples.png"
WINDOW_BARS = 12
SEED = 20260831

TRIGGER_ORDER = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
                  "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion", "local_extreme"]
TRIGGER_LABEL_KR = {
    "liquidity_sweep": "liquidity_sweep (기존)",
    "taker_delta_z_climax": "taker_delta_z_climax (기존신호)",
    "short_term_return_z": "short_term_return_z (기존신호)",
    "orthogonal_combo": "orthogonal_combo (기존신호)",
    "smt_divergence": "smt_divergence (기존신호)",
    "fib_extension_exhaustion": "fib_extension_exhaustion (기존신호)",
    "local_extreme": "local_extreme (신규, 무조건)",
}


def draw_candles(ax, sub: pd.DataFrame, event_pos: int, event_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def pick_per_trigger(labels: pd.DataFrame, trigger: str, n: int = 3) -> pd.DataFrame:
    pool = labels[(labels["triggers"].str.contains(trigger)) & (labels["outcome"] == "V자반등")]
    pool = pool.sort_values("giveback_ratio").reset_index(drop=True)
    if len(pool) <= n:
        return pool
    pcts = [0.15, 0.5, 0.85][:n]
    idxs = sorted({min(int(p * (len(pool) - 1)), len(pool) - 1) for p in pcts})
    return pool.iloc[idxs]


def main() -> int:
    frame = pd.read_csv(ETH_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    labels = pd.read_csv(LABELS_CSV)

    samples = []
    for trig in TRIGGER_ORDER:
        picked = pick_per_trigger(labels, trig, n=3)
        for _, row in picked.iterrows():
            samples.append((trig, row))

    n_cols = 3
    n_rows = len(TRIGGER_ORDER)
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.2 * n_rows), dpi=145)
    fig.suptitle(
        "V자반등 7트리거 통합라벨 육안검증 (트리거별 3건, giveback percentile 15/50/85) — "
        "라벨식은 v7b(30min 1.5xATR 종가도달 AND 60min giveback<=0.20) 그대로 재사용, "
        "바뀐 건 후보(트리거) 정의뿐. 점선=이벤트시점, 회색가로선=이벤트 극값",
        fontsize=17, y=0.998,
    )

    for r, trig in enumerate(TRIGGER_ORDER):
        row_samples = [s for t, s in samples if t == trig]
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(row_samples):
                ax.axis("off")
                continue
            event = row_samples[c]
            idx = int(event["idx"])
            is_down = event["direction"] == "downside"
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            level = float(frame["low"].iloc[idx]) if is_down else float(frame["high"].iloc[idx])
            draw_candles(ax, sub, WINDOW_BARS, level)
            ax.set_facecolor("#eef8f0")
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            dir_kr = "상승반등(downside sweep류)" if is_down else "하락반전(upside류)"
            title = (f"{TRIGGER_LABEL_KR[trig] if c == 0 else trig}\n"
                     f"{dir_kr} | fast={event['fast_move_atr_mult']:.2f}x | giveback={event['giveback_ratio']:.3f}\n"
                     f"other_triggers={event['triggers']}")
            ax.set_title(title, fontsize=9.5)
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}  ({len(samples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
