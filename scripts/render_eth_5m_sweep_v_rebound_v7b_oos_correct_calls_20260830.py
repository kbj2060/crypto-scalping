#!/usr/bin/env python3
"""10 OOS examples where the v7b model CALLED V자반등 (proba>=0.5) and the ground-truth label
agreed (label==1, i.e. a correct/true-positive call) -- visual sanity check requested after the
liquidation-spike/V-rebound literature review raised the "bid-ask-bounce artifact" hypothesis for
why classification improved (v4->v7b) but economics stayed FAILED. Reuses the exact candle-chart
convention from render_eth_5m_sweep_v_rebound_label_v7b_1x_atr_bar_20260830.py (gray dashed
sweep_level line, gray dotted sweep-bar marker, WINDOW_BARS=12 each side).

Population source: data/research/eth_sweep_v_rebound_v7b_costgate_20260830/v7b_costgate_candidates.pkl
(already-computed OOS model_proba from the v7b costgate re-test -- no new TabPFN inference needed).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_sweep_v_rebound_v7b_costgate_20260830/v7b_costgate_candidates.pkl"
LABELS_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830/v7b_oos_correct_calls_10.png"
WINDOW_BARS = 12


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.3, zorder=1)
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.3, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    oos_correct = df[(df["split"] == "oos") & (df["label"] == 1)].copy()
    print(f"OOS 정콜(label==1) 후보: {len(oos_correct)}건")

    sorted_pool = oos_correct.sort_values("model_proba").reset_index(drop=True)
    pcts = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    idxs = sorted({min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts})
    sample = sorted_pool.iloc[idxs].reset_index(drop=True)
    print(f"모델확신도(model_proba) 5~95 percentile로 {len(sample)}건 선택")

    labels = pd.read_csv(LABELS_CSV, usecols=["candidate_index", "side", "sweep_level"])

    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 5, figsize=(32, 14), dpi=145)
    fig.suptitle(
        "v7b OOS 정콜(모델 V자반등 콜=proba≥0.5 AND 실제 라벨도 V자반등) 10건 -- "
        "모델확신도(proba) 5~95 percentile 순",
        fontsize=20, y=1.0,
    )

    side_to_sweep = {"long": "downside", "short": "upside"}
    for ax, (_, ev) in zip(axes.flatten(), sample.iterrows()):
        sweep_side = side_to_sweep[ev["side"]]
        row = labels[(labels["candidate_index"] == ev["candidate_index"]) & (labels["side"] == sweep_side)]
        sweep_level = float(row["sweep_level"].iloc[0])
        idx = int(ts_to_idx[ev["sweep_ts"]])
        sub = kl.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
        draw_candles(ax, sub, WINDOW_BARS, sweep_level)
        ax.set_facecolor("#eef8f0")
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(f"{ev['side']} | proba={ev['model_proba']:.2f}", fontsize=13)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
