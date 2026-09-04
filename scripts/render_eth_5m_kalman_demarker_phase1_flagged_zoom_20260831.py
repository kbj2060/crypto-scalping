#!/usr/bin/env python3
"""Follow-up to render_eth_5m_kalman_demarker_phase1_examples_20260831.py -- the user flagged (1)
DeMarker's NO_HIT row2/col4 panel as looking ambiguous and (2) "most" of the Kalman HIT panels as
"not looking right". This re-renders those specific examples LARGER, with an added ENTRY-level
reference line (the original render only drew the target line, so a touch-then-fully-reverse-past-
entry pattern wasn't visually distinguishable from a touch-then-partial-fade) and explicit
retention-% annotation (end_ret_atr_mult / move_atr_mult), to make the underlying data verifiable
by eye rather than just trusting the peak/end numbers in the title.

Independent data re-derivation (see chat) confirmed BOTH are NOT labeling bugs:
  - DeMarker row2col4 (2024-01-26 16:50 fire, top, entry=2279.40): the window's true low over the
    next 12 bars is 2265.61 (at 17:25), target was 2264.55 -- missed by $1.06 (peak=1.76x vs
    required 1.9x). A real, correctly-labeled near-miss; visually easy to mistake for a touch at
    this chart's resolution because the miss margin is tiny.
  - Kalman's 10 sampled HIT events all independently re-verified: peaks range 1.96x-4.93x, all
    clear of the 1.65x threshold with no ambiguity. What IS real: average RETENTION of the peak
    favorable move by hour's end is only ~22% for these 10 (vs ~70% for DeMarker's HIT sample),
    and 3/10 fully reverse PAST entry (negative end) -- Kalman's trigger tends to fire mid-spike
    (the filter lags a sudden move, so a fast wick pushes deviation past threshold), which is
    inherently more prone to snapping back than DeMarker's requirement of 14 bars of accumulated
    directional bars. This is a real characteristic, not a bug -- touch-based/no-persistence
    labeling (this project's established default) is mathematically unaffected by it, but it's
    worth flagging before the economics/exit-structure stage later.
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

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_phase1_diagnostics_20260831 import (  # noqa: E402
    START,
    cluster_dedup,
    compute_atr_pct,
)

SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_kalman_demarker_phase1_20260831"
HORIZON = 12
CLUSTER_GAP_MERGE = 3
WINDOW_BARS = 12
N_PER_LABEL = 10
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)


def build_fires(klines, trigger_top, trigger_bottom, extremeness, atr_hit_mult):
    high, low, close = klines["high"].to_numpy(), klines["low"].to_numpy(), klines["close"].to_numpy()
    atr_pct = compute_atr_pct(klines["high"], klines["low"], klines["close"]).to_numpy()
    ts = klines["timestamp"].to_numpy()
    n = len(klines)
    rows = []
    for side, trig in (("bottom", trigger_bottom), ("top", trigger_top)):
        idx = np.flatnonzero(trig.fillna(False).to_numpy())
        idx = idx[(idx >= WINDOW_BARS) & (idx < n - max(HORIZON, WINDOW_BARS)) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup(idx, extremeness[idx], most_negative=(side == "bottom"), gap=CLUSTER_GAP_MERGE)
        entry = close[idx]
        a = atr_pct[idx]
        end_close = close[idx + HORIZON]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            move_pct = (fut_ext - entry) / entry
            end_ret_pct = (end_close - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            move_pct = (entry - fut_ext) / entry
            end_ret_pct = (entry - end_close) / entry
        hit = (move_pct >= atr_hit_mult * a).astype(int)
        target_level = entry * (1 + atr_hit_mult * a) if side == "bottom" else entry * (1 - atr_hit_mult * a)
        rows.append(pd.DataFrame({
            "pos": idx, "timestamp": ts[idx], "side": side, "hit": hit,
            "entry": entry, "atr_pct": a, "target_level": target_level,
            "move_atr_mult": move_pct / a, "end_ret_atr_mult": end_ret_pct / a,
        }))
    return pd.concat(rows, ignore_index=True)


def draw_candles(ax, sub, fire_pos, entry_level, target_level):
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=2.0, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])
    ax.axhline(target_level, color="#7A0EBF", linestyle="--", linewidth=2.0, zorder=1, label=f"target={target_level:.2f}")
    ax.axhline(entry_level, color="dimgray", linestyle="-.", linewidth=1.6, zorder=1, label=f"entry={entry_level:.2f}")
    ax.axvline(fire_pos, color="dimgray", linestyle=":", linewidth=1.3, zorder=1)
    pad = (max(highs) - min(lows)) * 0.10 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.9)


def render_panel(ax, klines, event, title_prefix):
    idx = int(event["pos"])
    sub = klines.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
    draw_candles(ax, sub, WINDOW_BARS, float(event["entry"]), float(event["target_level"]))
    ticks = list(range(0, len(sub), 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
    retention = event["end_ret_atr_mult"] / event["move_atr_mult"] * 100 if event["move_atr_mult"] else float("nan")
    hit_word = "HIT" if event["hit"] else "NO_HIT"
    ax.set_title(
        f"{title_prefix} | {hit_word} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST\n"
        f"peak={event['move_atr_mult']:.2f}x  end={event['end_ret_atr_mult']:.2f}x ATR  "
        f"(retention={retention:.0f}%)  entry={event['entry']:.2f}  target={event['target_level']:.2f}",
        fontsize=15,
    )
    ax.grid(alpha=0.25)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    high, low, close = klines["high"], klines["low"], klines["close"]

    dem = compute_demarker(high, low)
    dem_extremeness = dem.fillna(0.5).to_numpy()
    dem_fires = build_fires(klines, dem >= 0.90, dem <= 0.10, dem_extremeness, atr_hit_mult=1.9)
    dem_no_hit = dem_fires[dem_fires["hit"] == 0].sample(n=N_PER_LABEL, random_state=SEED).reset_index(drop=True)
    flagged_demarker = dem_no_hit.iloc[8]  # row2(idx1) col4(idx3) of the 2x5 NO_HIT block -> panel 5+3=8

    levels, _ = kalman_level_and_velocity(close.to_numpy())
    kalman_dev = pd.Series((close.to_numpy() - levels) / levels, index=close.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    kalman_extremeness = kalman_dev_z.fillna(0.0).to_numpy()
    kalman_fires = build_fires(klines, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0, kalman_extremeness, atr_hit_mult=1.65)
    kalman_hit = kalman_fires[kalman_fires["hit"] == 1].sample(n=N_PER_LABEL, random_state=SEED).reset_index(drop=True)
    kalman_hit = kalman_hit.assign(retention=kalman_hit["end_ret_atr_mult"] / kalman_hit["move_atr_mult"])
    print("Kalman HIT sample retention (end/peak):")
    print(kalman_hit[["timestamp", "side", "move_atr_mult", "end_ret_atr_mult", "retention"]].to_string())
    print(f"mean retention: {kalman_hit['retention'].mean() * 100:.1f}%  "
          f"n with negative end (past entry): {(kalman_hit['end_ret_atr_mult'] < 0).sum()}/10")

    fig, axes = plt.subplots(2, 3, figsize=(30, 18), dpi=140)
    fig.suptitle(
        "Follow-up zoom: flagged DeMarker NO_HIT (top-left) + 5 representative Kalman HIT examples "
        "with entry-level line added (dash-dot gray) alongside target (dashed purple)",
        fontsize=18, y=0.995,
    )
    render_panel(axes[0][0], klines, flagged_demarker, "DeMarker (flagged row2col4)")

    kalman_order = kalman_hit.sort_values("retention").reset_index(drop=True)
    pick = [0, 2, 4, 7, 9]  # spread across the retention range: worst, low, mid, high, best
    for slot, k in enumerate(pick):
        r, c = divmod(slot + 1, 3)
        render_panel(axes[r][c], klines, kalman_order.iloc[k], "Kalman")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "flagged_examples_zoom.png"
    fig.savefig(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
