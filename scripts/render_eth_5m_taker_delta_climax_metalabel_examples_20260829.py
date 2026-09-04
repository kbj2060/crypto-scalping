#!/usr/bin/env python3
"""Visual sanity check for the taker_delta_z_climax meta-label v4 -- ADOPTED FINAL VERSION (see
research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py and docs/experiments/eth_taker_
delta_climax_metalabel_20260829.md for the full version history). HORIZON=24/2h, hit = touched
(MFE_pct, intrabar high/low, >= 2.0*atr_pct_at_fire), fires cluster-anchored (same-side bursts
within 3 bars collapsed to their single most-extreme-delta_z bar). A v5 persistence check (require
the window-end close to also still be net favorable) was tried and REJECTED -- it made VAL/OOS/
HOLDOUT AUC worse, not better, so v4 (touch-only, no persistence) is what ships. The "end=" value
in each panel title is shown for context only (how much of the peak was retained by window end),
it does NOT gate hit/no-hit in this v4 version.

Same style as render_eth_5m_sweep_v_rebound_label_examples_20260829.py -- 10 random examples per
class (HIT / NO_HIT), candlestick panels, +/-2h around the fire bar, one combined PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from live_evidence_signal_dashboard_20260823 import compute_signals

SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_PATH = ROOT / "tmp/eth_taker_delta_climax_metalabel_20260829/label_examples_v4_final.png"
START = pd.Timestamp("2024-01-01")
HORIZON = 24
ATR_HIT_MULT = 2.0
CLUSTER_GAP_MERGE = 3
WINDOW_BARS = 24  # 2 hours each side -- matches HORIZON so the outcome window sits at the right edge
N_PER_LABEL = 10
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_NAMES = {0: "NO_HIT", 1: "HIT"}
LABEL_TINT = {0: "#fdf1ef", 1: "#eef8f0"}


def cluster_dedup(idx: np.ndarray, delta_z_at_idx: np.ndarray, most_negative: bool) -> np.ndarray:
    order = np.argsort(idx)
    idx_sorted, dz_sorted = idx[order], delta_z_at_idx[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > CLUSTER_GAP_MERGE:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "dz": dz_sorted})
    keep = df.loc[df.groupby("cluster")["dz"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["dz"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires() -> pd.DataFrame:
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    close = sig["close"].to_numpy()
    atr_pct = sig["atr_pct"].to_numpy()
    delta_z = sig["delta_z"].to_numpy()
    ts = sig["timestamp"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_taker_delta_z_climax"), ("top", "top_taker_delta_z_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[(idx >= WINDOW_BARS) & (idx < n - max(HORIZON, WINDOW_BARS)) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup(idx, delta_z[idx], most_negative=(side == "bottom"))
        entry = close[idx]
        a = atr_pct[idx]
        end_close = close[idx + HORIZON]  # context only, not used for hit in v4
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            move_pct = (fut_ext - entry) / entry
            end_ret_pct = (end_close - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            move_pct = (entry - fut_ext) / entry
            end_ret_pct = (entry - end_close) / entry
        hit = (move_pct >= ATR_HIT_MULT * a).astype(int)  # v4: touch-only, no persistence gate
        target_level = entry * (1 + ATR_HIT_MULT * a) if side == "bottom" else entry * (1 - ATR_HIT_MULT * a)
        rows.append(pd.DataFrame({
            "pos": idx, "timestamp": ts[idx], "side": side, "hit": hit,
            "entry": entry, "target_level": target_level,
            "move_atr_mult": move_pct / a, "end_ret_atr_mult": end_ret_pct / a,
        }))
    return pd.concat(rows, ignore_index=True)


def draw_candles(ax, sub: pd.DataFrame, fire_pos: int, target_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])
    ax.axhline(target_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(fire_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    fires = build_fires()

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, N_PER_LABEL // 2, figsize=(38, 22), dpi=140)
    fig.suptitle(
        "ETH 5m taker_delta_z_climax meta-label v4 FINAL sanity check (HORIZON=2h, hit = touched "
        "MFE>=2.0xATR, cluster-anchored, no persistence gate) -- "
        f"random {N_PER_LABEL} examples per class, +/-2h around fire bar, seed={SEED}",
        fontsize=24, y=0.995,
    )

    for label, row_offset in ((0, 0), (1, 2)):
        sample = fires[fires["hit"] == label].sample(n=N_PER_LABEL, random_state=SEED)
        for panel, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row_offset + panel // (N_PER_LABEL // 2)][panel % (N_PER_LABEL // 2)]
            idx = int(event["pos"])
            sub = klines.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            draw_candles(ax, sub, WINDOW_BARS, float(event["target_level"]))
            ax.set_facecolor(LABEL_TINT[label])
            ticks = list(range(0, len(sub), 2))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
            ax.set_title(
                f"{LABEL_NAMES[label]} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST | "
                f"peak={event['move_atr_mult']:.2f}x end={event['end_ret_atr_mult']:.2f}x ATR",
                fontsize=12,
            )
            ax.grid(alpha=0.25)

    for row in axes[:, 0]:
        row.set_ylabel("price", fontsize=11)
    for row in axes[-1]:
        row.set_xlabel("minutes from fire bar", fontsize=11)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
