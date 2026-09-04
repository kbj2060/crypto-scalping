#!/usr/bin/env python3
"""Visual sanity check (20 examples: 10 HIT + 10 NO_HIT) for the DRAFT v1 label of the 2 Homer
candidate-pool signals confirmed to proceed: demarker_extreme and kalman_deviation_meanrev
(2026-08-31 narrowing). Same style as render_eth_5m_taker_delta_climax_metalabel_examples_20260829.py
(candlestick panels, +/-1h around the fire bar, one combined PNG per signal) -- reused verbatim
(draw_candles, cluster_dedup, figure layout, KST display offset).

Draft label parameters (chosen from research_eth_kalman_demarker_phase1_diagnostics_20260831.py's
findings -- BOTH signals' sign-only hit-rate decays monotonically from 15m to 4h, so HORIZON=12(1h)
is used for both as a reasonable starting point, NOT yet a swept/finalized choice):
  demarker_extreme: dem>=0.90 (top) / dem<=0.10 (bottom), K=1.9 (~median 1h MFE/ATR at this
    cluster-anchored population, i.e. close to a 50/50 split)
  kalman_deviation_meanrev: dev_z>=2.0 (top) / dev_z<=-2.0 (bottom), K=1.65 (same rationale)
Both HORIZON=12, CLUSTER_GAP_MERGE=3 (DeMarker's own post-anchor clustering was already thin at
gap=3; Kalman's was NOT -- 17-18% of anchored fires still within 12 bars of each other -- so
CLUSTER_GAP_MERGE may need widening later, same open question phase1 flagged, not resolved here).

These are DRAFT parameters for the visual-check gate only, not a finalized label design -- the
formal HORIZON/K screen (docs/homer/README.md section 5.5's "6~8+ point grid, not 3") still needs
to run before any TabPFN training.
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
WINDOW_BARS = 12  # +-1h each side, matches HORIZON so the outcome window sits at the right edge
N_PER_LABEL = 10
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_NAMES = {0: "NO_HIT", 1: "HIT"}
LABEL_TINT = {0: "#fdf1ef", 1: "#eef8f0"}


def build_fires(klines: pd.DataFrame, trigger_top: pd.Series, trigger_bottom: pd.Series,
                 extremeness: np.ndarray, atr_hit_mult: float) -> pd.DataFrame:
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


def render(name: str, title: str, klines: pd.DataFrame, fires: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, N_PER_LABEL // 2, figsize=(38, 22), dpi=140)
    fig.suptitle(title, fontsize=22, y=0.995)

    for label, row_offset in ((0, 0), (1, 2)):
        pool = fires[fires["hit"] == label]
        sample = pool.sample(n=min(N_PER_LABEL, len(pool)), random_state=SEED)
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
    fig.savefig(out_path)
    print(f"saved: {out_path}  (HIT pool n={len(fires[fires['hit']==1])}, NO_HIT pool n={len(fires[fires['hit']==0])})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    klines = pd.read_csv(SOURCE, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    high, low, close = klines["high"], klines["low"], klines["close"]

    dem = compute_demarker(high, low)
    dem_extremeness = dem.fillna(0.5).to_numpy()
    dem_fires = build_fires(klines, dem >= 0.90, dem <= 0.10, dem_extremeness, atr_hit_mult=1.9)
    render(
        "demarker_extreme",
        "ETH 5m demarker_extreme DRAFT v1 phase1 sanity check (HORIZON=1h, hit = touched "
        "MFE>=1.9xATR, cluster-anchored gap=3, top=dem>=0.90/bottom=dem<=0.10, NOT a finalized "
        "label -- HORIZON/K not yet swept) -- random 10 examples per class, +/-1h around fire bar, "
        f"seed={SEED}",
        klines, dem_fires, OUT_DIR / "demarker_extreme_phase1_examples.png",
    )

    levels, _ = kalman_level_and_velocity(close.to_numpy())
    kalman_dev = pd.Series((close.to_numpy() - levels) / levels, index=close.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    kalman_extremeness = kalman_dev_z.fillna(0.0).to_numpy()
    kalman_fires = build_fires(klines, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0, kalman_extremeness, atr_hit_mult=1.65)
    render(
        "kalman_deviation_meanrev",
        "ETH 5m kalman_deviation_meanrev DRAFT v1 phase1 sanity check (HORIZON=1h, hit = touched "
        "MFE>=1.65xATR, cluster-anchored gap=3, top=dev_z>=2.0/bottom=dev_z<=-2.0, NOT a finalized "
        "label -- HORIZON/K/cluster-gap not yet swept) -- random 10 examples per class, +/-1h "
        f"around fire bar, seed={SEED}",
        klines, kalman_fires, OUT_DIR / "kalman_deviation_meanrev_phase1_examples.png",
    )


if __name__ == "__main__":
    main()
