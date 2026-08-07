"""Layer 1-4 Fresh-Forward walk-forward performance charts: VAL/OOS equity
curve + a one-week entry/exit zoom (OOS week 1, matching the earlier
CUSUM-TB chart convention)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_btc_exit_stopping_rl_20260803 as M  # noqa: E402

TRADES_CSV = ROOT / "tmp/btc_fresh_forward_walkforward_trades_20260803.csv"
FRAME_PATH = M.FRAME_PATH
OOS_WEEK_START = pd.Timestamp("2026-01-01")
OOS_WEEK_END = pd.Timestamp("2026-01-08")
OUT_EQUITY = ROOT / "tmp/btc_layer1to4_equity_curve_20260803.png"
OUT_WEEK = ROOT / "tmp/btc_layer1to4_oos_week_20260803.png"


def main():
    t = pd.read_csv(TRADES_CSV, parse_dates=["entry_ts", "exit_ts"]).sort_values("entry_ts").reset_index(drop=True)
    t["cum_net"] = t["net"].cumsum()
    t["period"] = np.where(t["entry_ts"] < M.OOS_START, "VAL", "OOS")

    # --- 1. equity curve (cumulative sum of trade net returns), VAL vs OOS shaded ---
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(t["exit_ts"], 100 * t["cum_net"], color="#2166ac", linewidth=1.6)
    ax.axvspan(M.VAL_START, M.OOS_START, color="#f4a582", alpha=0.15, label="VAL")
    ax.axvspan(M.OOS_START, M.OOS_END, color="#92c5de", alpha=0.25, label="OOS")
    ax.axhline(0, color="#999", linewidth=0.7)
    val_n = (t["period"] == "VAL").sum(); oos_n = (t["period"] == "OOS").sum()
    val_mean = t.loc[t.period == "VAL", "net"].mean() * 100
    oos_mean = t.loc[t.period == "OOS", "net"].mean() * 100
    ax.set_title(f"Layer1-4 Fresh-Forward walk-forward cumulative net return\n"
                 f"VAL n={val_n} mean={val_mean:.3f}%  |  OOS n={oos_n} mean={oos_mean:.3f}%")
    ax.set_ylabel("cumulative net return (%)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_EQUITY, dpi=140)
    print(f"wrote {OUT_EQUITY}")

    # --- 2. one-week zoom, OOS week 1, entries/exits over price ---
    frame = pd.read_parquet(FRAME_PATH, columns=["timestamp", "close"])
    frame = frame[(frame["timestamp"] >= OOS_WEEK_START) & (frame["timestamp"] < OOS_WEEK_END + pd.Timedelta(hours=6))]
    week = t[(t["entry_ts"] >= OOS_WEEK_START) & (t["entry_ts"] < OOS_WEEK_END)]

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(frame["timestamp"], frame["close"], color="#888", linewidth=0.9, label="BTC close (5m)")
    for _, r in week.iterrows():
        entry_px = frame.loc[frame["timestamp"] == r["entry_ts"], "close"]
        exit_px = frame.loc[frame["timestamp"] == r["exit_ts"], "close"]
        entry_px = float(entry_px.iloc[0]) if len(entry_px) else np.nan
        exit_px = float(exit_px.iloc[0]) if len(exit_px) else np.nan
        marker = "^" if r["side"] == 1 else "v"
        color = "#1a9850" if r["net"] > 0 else "#d73027"
        ax2.scatter(r["entry_ts"], entry_px, marker=marker, color="#2166ac", s=70, zorder=5)
        ax2.scatter(r["exit_ts"], exit_px, marker="o", color=color, s=45, zorder=5)
        ax2.plot([r["entry_ts"], r["exit_ts"]], [entry_px, exit_px], color=color, linewidth=1.1, alpha=0.75)

    ax2.set_title(f"Layer1-4 full stack — OOS week {OOS_WEEK_START.date()}..{OOS_WEEK_END.date()} "
                  f"({len(week)} trades; ▲long ▼short entry, green=profit red=loss exit)")
    ax2.set_ylabel("price (USDT)")
    ax2.legend(loc="upper left")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(OUT_WEEK, dpi=140)
    print(f"wrote {OUT_WEEK}, trades_this_week={len(week)}")


if __name__ == "__main__":
    main()
