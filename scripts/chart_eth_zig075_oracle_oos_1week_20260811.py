"""Plot one representative week of ORACLE zigzag wave trades for ETH zig075's direction label
(zigzag_action_labels_20260531), within the label's available OOS coverage (2026-01-01..02-28).

Each contiguous LONG/SHORT run of zigzag_action is one oracle trade (entry=run start, exit=run
end); CASH/transition-buffer rows (zigzag_action==0) are skipped. Sanity check for Step A
(oracle label inspection) before the zig075 jmredesign parent/sidecar tuning.
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

ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2026.csv"
OUT_PNG = ROOT / "tmp/eth_zig075_oracle_label_check_20260811/oos_1week_zigzag_oracle_trades.png"


def _contiguous_runs(action: np.ndarray):
    """Yield (start_idx, end_idx, action_value) for each maximal run of a constant nonzero action."""
    n = len(action)
    i = 0
    while i < n:
        a = action[i]
        j = i
        while j + 1 < n and action[j + 1] == a:
            j += 1
        if a != 0:
            yield i, j, a
        i = j + 1


def main() -> int:
    df = pd.read_csv(LABEL_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    action = df["zigzag_action"].to_numpy()
    ts = df["timestamp"]
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)

    trades = []
    for i, j, a in _contiguous_runs(action):
        side = 1 if a == 1 else -1
        entry_px, exit_px = open_[i], close[j]
        ret = side * (exit_px / entry_px - 1.0)
        trades.append({"entry_ts": ts.iloc[i], "exit_ts": ts.iloc[j], "side": side,
                        "entry_px": entry_px, "exit_px": exit_px, "trade_return": ret})
    ledger = pd.DataFrame(trades)

    week_key = ledger["entry_ts"].dt.to_period("W-SUN")
    best_week = week_key.value_counts().idxmax()
    week_start, week_end = best_week.start_time, best_week.end_time
    print(f"selected week: {week_start} .. {week_end}, "
          f"n_trades_entered_this_week={int((week_key == best_week).sum())}")

    week_ledger = ledger[(ledger["entry_ts"] >= week_start) & (ledger["entry_ts"] <= week_end)].copy()
    price_week = df[(df["timestamp"] >= week_start) & (df["timestamp"] <= week_end + pd.Timedelta(hours=6))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price_week["timestamp"], price_week["close"], color="#888888", linewidth=0.8, zorder=1)

    for _, row in week_ledger.iterrows():
        marker = "^" if row["side"] == 1 else "v"
        c = "#2ca02c" if row["trade_return"] > 0 else "#d62728"
        ax.scatter(row["entry_ts"], row["entry_px"], marker=marker, color=c, s=70,
                   zorder=3, edgecolors="black", linewidths=0.5)
        ax.plot([row["entry_ts"], row["exit_ts"]], [row["entry_px"], row["exit_px"]],
               color=c, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

    win_rate = float((week_ledger["trade_return"] > 0).mean()) if len(week_ledger) else 0.0
    sum_ret = float(week_ledger["trade_return"].sum() * 100)
    n_long = int((week_ledger["side"] == 1).sum())
    n_short = int((week_ledger["side"] == -1).sum())

    ax.set_title(f"ETH 5m -- zig075 ORACLE (zigzag pivot) trades, direction label\n"
                 f"{week_start.date()} ~ {week_end.date()} "
                 f"(up-tri=long down-tri=short, green=win red=loss)")
    ax.set_ylabel("ETH close (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    fig.autofmt_xdate()
    ax.grid(alpha=0.25)
    ax.text(
        0.01, 0.02,
        f"n_trades={len(week_ledger)} (long={n_long} short={n_short})  "
        f"win_rate={win_rate:.1%}  sum_ret={sum_ret:+.2f}%",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
