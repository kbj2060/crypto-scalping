"""Chart Layer 1 (CUSUM event gate) firing pattern for one OOS week
(2026-01-01..01-08), showing which 5m bars pass the gate and in which
direction (cumulative up-move vs down-move breach)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_final97_2024_2026.parquet"
WEEK_START = pd.Timestamp("2026-01-01")
WEEK_END = pd.Timestamp("2026-01-08")
OUT_PNG = ROOT / "tmp/btc_cusum_gate_oos_week_20260803.png"


def cusum_events_with_direction(frame: pd.DataFrame, atr: np.ndarray, mult: float = 2.0):
    close = frame["close"].to_numpy(dtype=np.float64)
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    s_pos = s_neg = 0.0
    events, directions = [], []
    for i in range(1, len(close)):
        thresh = max(float(atr[i]), 0.001) * mult
        s_pos = max(0.0, s_pos + logret[i])
        s_neg = min(0.0, s_neg + logret[i])
        if s_pos > thresh:
            events.append(i); directions.append(1)
            s_pos = s_neg = 0.0
        elif s_neg < -thresh:
            events.append(i); directions.append(-1)
            s_pos = s_neg = 0.0
    return np.array(events), np.array(directions)


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    atr = _atr_price_move(frame)
    events, directions = cusum_events_with_direction(frame, atr, mult=2.0)

    ev_ts = frame["timestamp"].to_numpy()[events]
    ev_px = frame["close"].to_numpy()[events]
    mask = (ev_ts >= WEEK_START.to_datetime64()) & (ev_ts < WEEK_END.to_datetime64())
    week_ts, week_px, week_dir = ev_ts[mask], ev_px[mask], directions[mask]

    price = frame[(frame["timestamp"] >= WEEK_START) & (frame["timestamp"] < WEEK_END)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price["timestamp"], price["close"], color="#888", linewidth=0.9, label="BTC close (5m)")
    up_mask = week_dir == 1
    ax.scatter(week_ts[up_mask], week_px[up_mask], marker="^", color="#1a9850", s=45, zorder=5, label="CUSUM up-event")
    ax.scatter(week_ts[~up_mask], week_px[~up_mask], marker="v", color="#d73027", s=45, zorder=5, label="CUSUM down-event")

    ax.set_title(f"Layer 1 — CUSUM event gate, OOS week {WEEK_START.date()}..{WEEK_END.date()} "
                 f"({mask.sum()} events fired this week, mult=2.0)")
    ax.set_ylabel("price (USDT)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}, events_this_week={mask.sum()} (up={up_mask.sum()}, down={(~up_mask).sum()})")


if __name__ == "__main__":
    main()
