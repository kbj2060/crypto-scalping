"""Plot one representative OOS week of the ETH zigzag exit-oracle layer
(tmp/eth_zigzag_exit_layer_richfeatures_softlabel_20260810/dataset.csv): price with zigzag
direction-wave shading (long/short, same oracle as the direction label) plus markers at every bar
where the exit-oracle's cost-aware "exit now" target (y_exit_now) fires -- i.e. bars where holding
another HORIZON_BARS would not have cleared round-trip cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tmp/eth_zigzag_exit_layer_richfeatures_softlabel_20260810/dataset.csv"
OUT_PNG = ROOT / "tmp/eth_zigzag_exit_layer_richfeatures_softlabel_20260810/oos_1week_exit_oracle.png"

THETA = 0.02
C_BULL, C_BEAR = "#2563EB", "#D9542B"


def zigzag_oracle(close: np.ndarray, threshold: float = THETA) -> np.ndarray:
    """Verbatim from scripts/build_eth_zigzag_exit_layer_richfeatures_softlabel_20260810.py."""
    n = len(close)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
    pivots: list[int] = []
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        if up is None:
            if close[t] >= close[lo_i] * (1 + threshold):
                up, ext_i = True, t
                pivots.append(lo_i)
            elif close[t] <= close[hi_i] * (1 - threshold):
                up, ext_i = False, t
                pivots.append(hi_i)
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - threshold):
                pivots.append(ext_i)
                up, ext_i = False, t
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + threshold):
                pivots.append(ext_i)
                up, ext_i = True, t
    direction = np.zeros(n, dtype=np.int8)
    if len(pivots) >= 2:
        first_up = close[pivots[1]] > close[pivots[0]]
        bounds = pivots + [n - 1]
        d = 1 if first_up else -1
        for i in range(len(bounds) - 1):
            direction[bounds[i]: bounds[i + 1] + 1] = d
            d = -d
    return direction


def contiguous_runs(vals: np.ndarray):
    n = len(vals)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        yield i, j, vals[i]
        i = j + 1


def main() -> int:
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    oos = df[df["split"] == "oos"].copy()
    if oos.empty:
        raise RuntimeError("no OOS rows in dataset")

    direction_full = zigzag_oracle(df["close"].to_numpy(dtype=np.float64))
    df["direction"] = direction_full

    exit_fires = oos[oos["y_exit_now"] == 1.0]
    week_key = exit_fires["timestamp"].dt.to_period("W-SUN")
    best_week = week_key.value_counts().idxmax()
    week_start, week_end = best_week.start_time, best_week.end_time
    print(f"selected week: {week_start} .. {week_end}, "
          f"n_exit_fires_this_week={int((week_key == best_week).sum())}", flush=True)

    wk = df[(df["timestamp"] >= week_start) & (df["timestamp"] <= week_end + pd.Timedelta(hours=6))].copy()
    wk_idx = wk.index.to_numpy()
    ts = wk["timestamp"].to_numpy()
    close = wk["close"].to_numpy(dtype=np.float64)
    direction = wk["direction"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    for s, e, d in contiguous_runs(direction):
        if d == 0:
            continue
        ax.axvspan(ts[s], ts[min(e, len(ts) - 1)], color=C_BULL if d > 0 else C_BEAR, alpha=0.12, linewidth=0)
    ax.plot(ts, close, color="#333333", linewidth=1.0, zorder=2)

    exit_mask = (wk["y_exit_now"] == 1.0).to_numpy()
    ax.scatter(ts[exit_mask], close[exit_mask], marker="x", color="#111111", s=55, linewidths=1.8,
               zorder=4, label="exit-oracle fires (y_exit_now=1)")

    n_exit = int(exit_mask.sum())
    n_long_bars = int((direction > 0).sum())
    n_short_bars = int((direction < 0).sum())
    ax.set_title(f"ETH 5m -- zigzag exit-oracle, OOS week {week_start.date()} ~ {week_end.date()}\n"
                 f"(shading=zigzag direction wave, x=exit-now-vs-hold oracle firing)")
    ax.set_ylabel("ETH close (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    fig.autofmt_xdate()
    ax.grid(alpha=0.25)
    ax.legend(handles=[
        Patch(facecolor=C_BULL, alpha=0.3, label="long wave"),
        Patch(facecolor=C_BEAR, alpha=0.3, label="short wave"),
        plt.Line2D([0], [0], marker="x", color="#111111", linestyle="", label="exit fires"),
    ], loc="upper left", frameon=False, fontsize=9)
    ax.text(0.01, 0.02, f"exit fires={n_exit}  long_bars={n_long_bars}  short_bars={n_short_bars}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
