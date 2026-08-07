"""
Chart CUSUM-filtered triple-barrier entries/exits for one OOS week of BTC
(2026-01-01 to 2026-01-08, the start of the official Fresh-Forward OOS window).
Diagnostic visualization only, not a promotion/backtest claim.
"""
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
from compare_btc_label_schemes_20260803 import build_triple_barrier, cusum_events  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
WEEK_START = pd.Timestamp("2026-01-01")
WEEK_END = pd.Timestamp("2026-01-08")
OUT_PNG = ROOT / "tmp/btc_cusum_tb_oos_week_20260803.png"


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    atr = _atr_price_move(frame)
    events = cusum_events(frame, atr, mult=2.0)
    events = events[events < len(frame) - 48 - 2]
    tb = build_triple_barrier(frame, candidate_idx=events)

    tb["entry_ts"] = frame["timestamp"].to_numpy()[tb["i"].to_numpy() + 1]
    tb["exit_ts"] = tb.apply(
        lambda r: frame["timestamp"].iloc[min(int(r["i"]) + 1 + int(r["bars"]), len(frame) - 1)], axis=1
    )
    tb["entry_px"] = frame["open"].to_numpy()[tb["i"].to_numpy() + 1]
    tb["exit_px"] = tb["entry_px"] * (1 + tb["ret"] * np.where(tb["action"] == 2, -1, 1))

    week = tb[(tb["entry_ts"] >= WEEK_START) & (tb["entry_ts"] < WEEK_END) & (tb["action"] != 0)]
    price = frame[(frame["timestamp"] >= WEEK_START) & (frame["timestamp"] < WEEK_END + pd.Timedelta(hours=4))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price["timestamp"], price["close"], color="#888", linewidth=0.9, label="BTC close (5m)")

    color_map = {"tp": "#1a9850", "sl": "#d73027", "timeout": "#999900"}
    for _, r in week.iterrows():
        marker = "^" if r["action"] == 1 else "v"
        ax.scatter(r["entry_ts"], r["entry_px"], marker=marker, color="#2166ac", s=70, zorder=5)
        ax.scatter(r["exit_ts"], r["exit_px"], marker="o", color=color_map.get(r["reason"], "gray"), s=40, zorder=5)
        ax.plot([r["entry_ts"], r["exit_ts"]], [r["entry_px"], r["exit_px"]],
                color=color_map.get(r["reason"], "gray"), linewidth=1.0, alpha=0.7)

    ax.set_title(f"BTC CUSUM-filtered triple-barrier — OOS week {WEEK_START.date()}..{WEEK_END.date()} "
                 f"({len(week)} entries; ▲long ▼short, green=TP red=SL yellow=timeout)")
    ax.set_ylabel("price (USDT)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}, entries={len(week)}")
    print(week[["entry_ts", "action", "reason", "ret", "bars"]].to_string(index=False))


if __name__ == "__main__":
    main()
