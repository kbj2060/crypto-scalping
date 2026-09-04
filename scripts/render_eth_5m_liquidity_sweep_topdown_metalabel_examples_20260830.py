#!/usr/bin/env python3
"""Chart-based visual verification for liquidity_sweep "top/down" metalabel (Homer signal #2
redo) -- MANDATORY step per docs/homer/README.md methodology template item 6 ("예시 10개를 캔들
이미지로 시각 검증... 숫자만으론 놓치는 라벨 버그를 실제로 여러 차례 잡아냄"). Renders 10 HIT +
10 NO_HIT examples (percentile-sampled by the hit ratio, not just extremes) for the WINNING
(horizon, gap) candidate picked from the TabPFN confirmation pass
(research_eth_liquidity_sweep_topdown_metalabel_tabpfn_confirm_20260830.py).

Charting utility (draw_candles/figsize/dpi) adapted verbatim from
render_eth_5m_sweep_v_rebound_label_v7b_1x_atr_bar_20260830.py -- same conventions
(feedback_large_chart_images.md: large figures, 140-150dpi, 18pt+ titles).

Set TAG below to the winning candidate (must match a
data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/
eth_5m_liquidity_sweep_topdown_metalabel_features_H{h}_GAP{g}.csv file, pulled from the server).
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
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_DIR = ROOT / "data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830"
OUT_DIR = ROOT / "tmp/eth_liquidity_sweep_topdown_metalabel_20260830"

# Final config, confirmed by TabPFN: HORIZON=30/GAP=12/K=4.0 clearly beat K=1.5/2.5 and the other
# (horizon,gap) grid candidates -- research_eth_liquidity_sweep_topdown_metalabel_ksweep_tabpfn_
# confirm_20260830.py (VAL 0.6587/OOS 0.6377, smallest VAL-OOS gap of all candidates tried).
TAG = "H30_GAP12_K4.0"
HORIZON = 30
GAP = 12
K = 4.0
SWEEP_LOOKBACK = 48
WINDOW_BARS = max(24, HORIZON + 6)  # a bit wider than the horizon itself so the outcome is visible


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def draw_candles(ax, sub: pd.DataFrame, fire_pos: int, level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(fire_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def pick_by_percentile(pool: pd.DataFrame, sort_col: str, pcts: list[float]) -> pd.DataFrame:
    sorted_pool = pool.sort_values(sort_col).reset_index(drop=True)
    idxs = sorted({min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts})
    return sorted_pool.iloc[idxs]


def main() -> int:
    klines = load_klines()
    fires_path = LABEL_DIR / f"eth_5m_liquidity_sweep_topdown_metalabel_features_{TAG}.csv"
    fires = pd.read_csv(fires_path, parse_dates=["timestamp"])
    print(f"loaded {len(fires)} fires from {fires_path} ({TAG})")

    high, low, close = klines["high"].to_numpy(), klines["low"].to_numpy(), klines["close"].to_numpy()
    swing_low_prior = pd.Series(low).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1).to_numpy()
    swing_high_prior = pd.Series(high).rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1).to_numpy()

    ratios, levels = [], []
    for _, r in fires.iterrows():
        i, side = int(r["pos"]), r["side"]
        entry = close[i]
        if side == "bottom":
            fut_ext = high[i + 1:i + HORIZON + 1].max()
            pred_dir_ret = (fut_ext - entry) / entry
            level = swing_low_prior[i]
        else:
            fut_ext = low[i + 1:i + HORIZON + 1].min()
            pred_dir_ret = (entry - fut_ext) / entry
            level = swing_high_prior[i]
        ratios.append(pred_dir_ret / max(r["atr_pct"], 1e-12))
        levels.append(level)
    fires = fires.assign(ratio=ratios, level=levels)

    hit_pool = fires[fires["hit"] == 1.0]
    miss_pool = fires[fires["hit"] == 0.0]
    print(f"HIT n={len(hit_pool)} ({len(hit_pool)/len(fires):.1%})  NO_HIT n={len(miss_pool)} ({len(miss_pool)/len(fires):.1%})")

    pcts = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    hit_sample = pick_by_percentile(hit_pool, "ratio", pcts)
    miss_sample = pick_by_percentile(miss_pool, "ratio", pcts)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 5, figsize=(34, 22), dpi=145)
    fig.suptitle(
        f"liquidity_sweep top/down metalabel ({TAG}, K={K}xATR touch-based MFE, no persistence): "
        f"row1-2=HIT(1, 10건 ratio순) | row3-4=NO_HIT(0, 10건, ratio순)",
        fontsize=18, y=0.995,
    )
    flat_axes = axes.flatten()
    for i, (_, ev) in enumerate(pd.concat([hit_sample, miss_sample]).iterrows()):
        ax = flat_axes[i]
        pos = int(ev["pos"])
        sub = klines.iloc[max(0, pos - WINDOW_BARS): pos + WINDOW_BARS + 1].reset_index(drop=True)
        fire_rel = pos - max(0, pos - WINDOW_BARS)
        draw_candles(ax, sub, fire_rel, float(ev["level"]))
        is_hit = i < len(hit_sample)
        ax.set_facecolor("#eef8f0" if is_hit else "#fdf1ef")
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - fire_rel) * 5:+d}" for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        label_txt = f"{'HIT' if is_hit else 'NO_HIT'} | {ev['side']}\nMFE/ATR ratio={ev['ratio']:.2f} (need>={K})"
        ax.set_title(label_txt, fontsize=10)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / f"liquidity_sweep_topdown_{TAG}_examples_20.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
