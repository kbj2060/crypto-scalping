#!/usr/bin/env python3
"""Visual re-verification at the FINAL, TabPFN-confirmed HORIZON/GAP/K (mandatory whenever a label
definition changes meaningfully -- docs/homer/README.md's established V자반등 precedent: "재라벨링
확정시 재진단 필수"). research_eth_kalman_demarker_phase1_examples_20260831.py checked draft
params (both H=12/GAP=3, K=1.9/1.65); the confirmed grid+K-sweep (research_eth_kalman_demarker_
gridscreen_20260831.py / _ksweep_20260831.py / memory eth_kalman_demarker_horizon_gap_k_screening_
20260831) moved these substantially: demarker_extreme H=8/GAP=12/K=0.70 (K nearly 3x lower than
draft, hit_rate now 88.6% -- a much easier bar), kalman_deviation_meanrev H=12/GAP=12/K=2.5 (GAP 4x
wider than draft). Both are different enough label populations that the earlier chart doesn't
verify these -- this one does.

Same candlestick style as render_eth_5m_kalman_demarker_phase1_flagged_zoom_20260831.py (entry line
added alongside target, clearer than the original phase1 chart which only drew target).
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
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    START,
    cluster_dedup,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

OUT_DIR = ROOT / "tmp/eth_kalman_demarker_phase1_20260831"
N_PER_LABEL = 10
SEED = 42
KST_OFFSET = pd.Timedelta(hours=9)
LABEL_TINT = {"NO_HIT": "#fdf1ef", "HIT": "#eef8f0"}

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70, "window_bars": 12},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5, "window_bars": 16},
}


def build_fires_for_chart(klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                           trigger_bottom: pd.Series, extremeness: np.ndarray,
                           horizon: int, gap: int, K: float) -> pd.DataFrame:
    """Same fire/label construction as build_fires() in the gridscreen script, but also keeps
    entry/target_level/peak/end for chart annotation (the grid screen only needed hit_plain)."""
    high, low, close = klines["high"].to_numpy(), klines["low"].to_numpy(), klines["close"].to_numpy()
    atr_pct = ind["atr_pct"].to_numpy()
    ts = klines["timestamp"].to_numpy()
    n = len(klines)
    rows = []
    for side, trig in (("bottom", trigger_bottom), ("top", trigger_top)):
        idx = np.flatnonzero(trig.fillna(False).to_numpy())
        idx = idx[(idx >= 30) & (idx < n - horizon) & (ts[idx] >= np.datetime64(START))]
        idx = cluster_dedup(idx, extremeness[idx], most_negative=(side == "bottom"), gap=gap)
        entry = close[idx]
        a = atr_pct[idx]
        end_close = close[idx + horizon]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            move_pct = (fut_ext - entry) / entry
            end_ret_pct = (end_close - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            move_pct = (entry - fut_ext) / entry
            end_ret_pct = (entry - end_close) / entry
        peak = move_pct / a
        end = end_ret_pct / a
        target_level = entry * (1 + K * a) if side == "bottom" else entry * (1 - K * a)
        rows.append(pd.DataFrame({
            "pos": idx, "timestamp": ts[idx], "side": side,
            "label": np.where(peak >= K, "HIT", "NO_HIT"),
            "entry": entry, "target_level": target_level, "peak": peak, "end": end,
        }))
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


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


def render_signal(name: str, klines: pd.DataFrame, fires: pd.DataFrame) -> None:
    cfg = SIGNAL_CONFIG[name]
    window_bars = cfg["window_bars"]
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(4, N_PER_LABEL // 2, figsize=(38, 22), dpi=140)
    fig.suptitle(
        f"ETH 5m {name} FINAL (TabPFN-confirmed) sanity check -- HORIZON={cfg['horizon']} bars, "
        f"GAP={cfg['gap']}, K={cfg['K']}xATR, hit=touched MFE (no persistence) -- "
        f"random {N_PER_LABEL} examples per class, +/-{window_bars*5}min around fire bar, seed={SEED}",
        fontsize=20, y=0.995,
    )
    for label, row_offset in (("NO_HIT", 0), ("HIT", 2)):
        pool = fires[fires["label"] == label]
        sample = pool.sample(n=min(N_PER_LABEL, len(pool)), random_state=SEED)
        for panel, (_, event) in enumerate(sample.iterrows()):
            ax = axes[row_offset + panel // 5][panel % 5]
            idx = int(event["pos"])
            sub = klines.iloc[idx - window_bars: idx + window_bars + 1].reset_index(drop=True)
            draw_candles(ax, sub, window_bars, float(event["entry"]), float(event["target_level"]))
            ax.set_facecolor(LABEL_TINT[label])
            ticks = list(range(0, len(sub), max(2, window_bars // 6)))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - window_bars) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            kst_ts = pd.Timestamp(event["timestamp"]) + KST_OFFSET
            ax.set_title(
                f"{label} | {event['side']} | {kst_ts:%Y-%m-%d %H:%M} KST\n"
                f"peak={event['peak']:.2f}x end={event['end']:.2f}x ATR",
                fontsize=11,
            )
            ax.grid(alpha=0.25)
    for row in axes[:, 0]:
        row.set_ylabel("price", fontsize=11)
    for row in axes[-1]:
        row.set_xlabel("minutes from fire bar", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / f"{name}_final_examples.png"
    fig.savefig(out_path)
    print(f"saved: {out_path}  (HIT pool n={len(fires[fires['label']=='HIT'])}, "
          f"NO_HIT pool n={len(fires[fires['label']=='NO_HIT'])})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    klines = load_klines()
    ind = build_indicator_frame(klines)

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    cfg = SIGNAL_CONFIG["demarker_extreme"]
    fires_dem = build_fires_for_chart(klines, ind_dem, dem >= 0.90, dem <= 0.10, dem.fillna(0.5).to_numpy(),
                                       cfg["horizon"], cfg["gap"], cfg["K"])
    render_signal("demarker_extreme", klines, fires_dem)

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    cfg = SIGNAL_CONFIG["kalman_deviation_meanrev"]
    fires_kal = build_fires_for_chart(klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                                       kalman_dev_z.fillna(0.0).to_numpy(), cfg["horizon"], cfg["gap"], cfg["K"])
    render_signal("kalman_deviation_meanrev", klines, fires_kal)


if __name__ == "__main__":
    main()
