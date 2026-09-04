#!/usr/bin/env python3
"""kalman_deviation_meanrev, v2: ambiguous-middle exclusion -- same design (2026-08-31, "Option A")
the user first approved for demarker_extreme (render_eth_5m_demarker_metalabel_v2_exclude_middle_
examples_20260831.py), now extended to Kalman on explicit request. Same rule, same rationale (see
that script's docstring for the full explanation and the orthogonal_combo kept-only-AUC-inflation
caveat that applies here too):
  EXCLUDE = |peak - K| < PEAK_BAND  OR  (peak >= K AND end < 0)
  HIT     = peak >= K AND NOT excluded
  NO_HIT  = peak <  K AND NOT excluded
K=1.65 (Kalman's own draft threshold, NOT DeMarker's 1.9 -- each signal keeps its own calibration).
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
ATR_HIT_MULT = 1.65
PEAK_BAND = 0.2
N_PER_LABEL = 10
N_EXCLUDED_EACH = 4
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_TINT = {"NO_HIT": "#fdf1ef", "HIT": "#eef8f0", "EXCLUDE": "#f5f0e8"}


def build_fires(klines: pd.DataFrame, dev_z: pd.Series) -> pd.DataFrame:
    high, low, close = klines["high"].to_numpy(), klines["low"].to_numpy(), klines["close"].to_numpy()
    atr_pct = compute_atr_pct(klines["high"], klines["low"], klines["close"]).to_numpy()
    ts = klines["timestamp"].to_numpy()
    n = len(klines)
    z_arr = dev_z.fillna(0.0).to_numpy()
    rows = []
    for side, trig in (("bottom", dev_z <= -2.0), ("top", dev_z >= 2.0)):
        idx = np.flatnonzero(trig.fillna(False).to_numpy())
        idx = idx[(idx >= WINDOW_BARS) & (idx < n - max(HORIZON, WINDOW_BARS)) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup(idx, z_arr[idx], most_negative=(side == "bottom"), gap=CLUSTER_GAP_MERGE)
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
        peak = move_pct / a
        end = end_ret_pct / a
        near_miss = np.abs(peak - ATR_HIT_MULT) < PEAK_BAND
        reversal = (peak >= ATR_HIT_MULT) & (end < 0)
        exclude = near_miss | reversal
        exclude_reason = np.where(exclude, np.where(near_miss & reversal, "both", np.where(near_miss, "near_miss", "reversal")), "")
        label = np.where(exclude, "EXCLUDE", np.where(peak >= ATR_HIT_MULT, "HIT", "NO_HIT"))
        target_level = entry * (1 + ATR_HIT_MULT * a) if side == "bottom" else entry * (1 - ATR_HIT_MULT * a)
        rows.append(pd.DataFrame({
            "pos": idx, "timestamp": ts[idx], "side": side, "label": label, "exclude_reason": exclude_reason,
            "entry": entry, "target_level": target_level, "peak": peak, "end": end,
        }))
    fires = pd.concat(rows, ignore_index=True)
    counts = fires["label"].value_counts()
    print(f"population: {dict(counts)}  ({counts.get('EXCLUDE', 0) / len(fires) * 100:.1f}% excluded)")
    return fires


def draw_candles(ax, sub: pd.DataFrame, fire_pos: int, entry_level: float, target_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=2)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=3))
        lows.append(bar["low"])
        highs.append(bar["high"])
    ax.axhline(target_level, color="#7A0EBF", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhline(entry_level, color="dimgray", linestyle="-.", linewidth=1.0, zorder=1)
    ax.axvline(fire_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def render_grid(klines, panels, n_rows, n_cols, suptitle, out_path, figsize):
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=140, squeeze=False)
    fig.suptitle(suptitle, fontsize=20, y=0.995)
    for panel, (r, c, event, subtitle) in enumerate(panels):
        ax = axes[r][c]
        idx = int(event["pos"])
        sub = klines.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
        draw_candles(ax, sub, WINDOW_BARS, float(event["entry"]), float(event["target_level"]))
        ax.set_facecolor(LABEL_TINT[event["label"]])
        ticks = list(range(0, len(sub), 2))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    print(f"saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    close = klines["close"].to_numpy()
    levels, _ = kalman_level_and_velocity(close)
    kalman_dev = pd.Series((close - levels) / levels, index=klines.index)
    dev_z = rolling_zscore(kalman_dev)
    fires = build_fires(klines, dev_z)

    def subtitle(event):
        kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
        return (f"{event['label']} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST\n"
                f"peak={event['peak']:.2f}x end={event['end']:.2f}x ATR")

    panels = []
    for label, row_offset in (("NO_HIT", 0), ("HIT", 2)):
        pool = fires[fires["label"] == label]
        sample = pool.sample(n=min(N_PER_LABEL, len(pool)), random_state=SEED)
        for panel, (_, event) in enumerate(sample.iterrows()):
            r, c = row_offset + panel // 5, panel % 5
            panels.append((r, c, event, subtitle(event)))
    render_grid(
        klines, panels, 4, 5,
        "ETH 5m kalman_deviation_meanrev v2 sanity check (HORIZON=1h, K=1.65xATR, ambiguous-middle "
        f"EXCLUDED: |peak-K|<{PEAK_BAND} OR touch-then-reverse -- surviving population only) -- "
        f"10 examples per class, +/-1h, seed={SEED}",
        OUT_DIR / "kalman_v2_hit_nohit_examples.png", (38, 22),
    )

    panels = []
    for reason, row in (("near_miss", 0), ("reversal", 1)):
        pool = fires[(fires["label"] == "EXCLUDE") & (fires["exclude_reason"].isin([reason, "both"]))]
        sample = pool.sample(n=min(N_EXCLUDED_EACH, len(pool)), random_state=SEED)
        for panel, (_, event) in enumerate(sample.iterrows()):
            panels.append((row, panel, event, f"[{reason}] " + subtitle(event)))
    render_grid(
        klines, panels, 2, N_EXCLUDED_EACH,
        "ETH 5m kalman_deviation_meanrev v2 EXCLUDED-middle sanity check -- row1: near-miss "
        "(|peak-1.65|<0.2), row2: touch-then-reverse (peak>=1.65 but end<0)",
        OUT_DIR / "kalman_v2_excluded_examples.png", (24, 12),
    )


if __name__ == "__main__":
    main()
