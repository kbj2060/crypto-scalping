#!/usr/bin/env python3
"""Chart-based visual verification for BTC liquidity_sweep grid-screen labels (H=15/K=2.0,
research_btc_liquidity_sweep_gridscreen_20260901.py's chosen point). MANDATORY step per this
project's convention (docs/homer/README.md methodology template item 6) before any further
decision on this signal -- renders 10 HIT + 10 NO_HIT examples (percentile-sampled by MFE/ATR
ratio, not just extremes) in one image.

Charting convention (figsize/dpi/layout) copied verbatim from
render_eth_5m_liquidity_sweep_topdown_metalabel_examples_20260830.py (feedback_large_chart_images:
large figures, 140-150dpi, 18pt+ titles) -- same 4x5 grid, same percentile-sampling, same
HIT(green bg)/NO_HIT(red bg) framing. Only the data source and hit formula differ (BTC's own
Tier0 CSV + the H=15/K=2.0 point research_btc_liquidity_sweep_gridscreen_20260901.py actually
selected, not re-derived here).
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
DATA_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "tmp/btc_liquidity_sweep_gridscreen_20260901"

HORIZON = 15
K = 2.0
WINDOW_BARS = max(24, HORIZON + 6)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
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
    frame = load_frame()
    # TRAIN+VAL only -- same scope research_btc_liquidity_sweep_gridscreen_20260901.py used to pick
    # this (H,K); OOS/HOLDOUT stay untouched here too.
    frame = frame[frame["timestamp"] < "2026-04-01"].reset_index(drop=True)
    n = len(frame)
    close = frame["close"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    atr = frame["atr"].to_numpy()
    sweep_low = frame["sweep_level_low"].to_numpy()
    sweep_high = frame["sweep_level_high"].to_numpy()

    rows = []
    for side, trig_col, level_arr in (("bottom", "bottom_liquidity_sweep", sweep_low),
                                       ("top", "top_liquidity_sweep", sweep_high)):
        fired = np.flatnonzero(frame[trig_col].to_numpy())
        for i in fired:
            if i + HORIZON >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            if side == "bottom":
                fut_ext = high[i + 1:i + HORIZON + 1].max()
                fut_move = fut_ext - close[i]
            else:
                fut_ext = low[i + 1:i + HORIZON + 1].min()
                fut_move = close[i] - fut_ext
            ratio = fut_move / atr[i]
            rows.append({
                "pos": i, "side": side, "level": level_arr[i],
                "ratio": ratio, "hit": 1 if ratio >= K else 0,
            })
    fires = pd.DataFrame(rows)
    print(f"loaded {len(fires)} candidates (H={HORIZON}, K={K}, TRAIN+VAL only)")

    hit_pool = fires[fires["hit"] == 1]
    miss_pool = fires[fires["hit"] == 0]
    print(f"HIT n={len(hit_pool)} ({len(hit_pool)/len(fires):.1%})  NO_HIT n={len(miss_pool)} ({len(miss_pool)/len(fires):.1%})")

    pcts = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    hit_sample = pick_by_percentile(hit_pool, "ratio", pcts)
    miss_sample = pick_by_percentile(miss_pool, "ratio", pcts)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, 5, figsize=(34, 22), dpi=145)
    fig.suptitle(
        f"BTC liquidity_sweep 그리드스크리닝 라벨 (H={HORIZON}/K={K}xATR 터치기반 MFE): "
        f"1~2행=HIT(10건, ratio순) | 3~4행=NO_HIT(10건, ratio순)",
        fontsize=18, y=0.995,
    )
    flat_axes = axes.flatten()
    for i, (_, ev) in enumerate(pd.concat([hit_sample, miss_sample]).iterrows()):
        ax = flat_axes[i]
        pos = int(ev["pos"])
        sub = frame.iloc[max(0, pos - WINDOW_BARS): pos + WINDOW_BARS + 1].reset_index(drop=True)
        fire_rel = pos - max(0, pos - WINDOW_BARS)
        draw_candles(ax, sub, fire_rel, float(ev["level"]))
        is_hit = i < len(hit_sample)
        ax.set_facecolor("#eef8f0" if is_hit else "#fdf1ef")
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - fire_rel) * 5:+d}" for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        side_ko = "바닥(상승기대)" if ev["side"] == "bottom" else "천장(하락기대)"
        label_txt = f"{'HIT' if is_hit else 'NO_HIT'} | {side_ko}\nMFE/ATR ratio={ev['ratio']:.2f} (기준>={K})"
        ax.set_title(label_txt, fontsize=10)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / f"btc_liquidity_sweep_H{HORIZON}_K{K}_examples_20.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
