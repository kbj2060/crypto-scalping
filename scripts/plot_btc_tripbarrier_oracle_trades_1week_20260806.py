"""Plot one representative OOS week of ORACLE (perfect-foresight) trades from the causal
triple-barrier label -- i.e. trade exactly the true trade_outcome_action label, not any model
prediction. Every trade wins by construction (see eval_btc_tripbarrier_label_oracle_quality_20260806.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"
OUT_PNG = ROOT / "tmp/btc_deepfeat_tripbarrier_backtest_20260806/oos_1week_oracle_trades.png"

CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS = 12, 288, 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")


def _fresh_entry_mask(side_state):
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(panel)

    labels = pd.read_parquet(LABEL_PATH, columns=["timestamp", "trade_outcome_action"])
    labels = labels.sort_values("timestamp").reset_index(drop=True)
    true_hard = labels["trade_outcome_action"].to_numpy()
    side_state_true = np.where(true_hard == 1, 1, np.where(true_hard == 2, -1, 0))

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol

    ts = panel["timestamp"]
    row_idx = np.flatnonzero((ts >= OOS_START).to_numpy() & (ts <= OOS_END).to_numpy())

    side_state = side_state_true[row_idx]
    fresh = _fresh_entry_mask(side_state)
    idx, side = row_idx[fresh], side_state[fresh]
    tp, sl = tp_moves_all[idx], sl_moves_all[idx]
    finite = np.isfinite(tp) & np.isfinite(sl)
    idx, side, tp, sl = idx[finite], side[finite], tp[finite], sl[finite]

    result = simulate_single_position(
        timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
        high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
        close=close, decision_indices=idx, scores=side.astype(np.float64), tp_moves=tp, sl_moves=sl,
        upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
    ledger["exit_timestamp"] = pd.to_datetime(ledger["exit_timestamp"])

    week_key = ledger["entry_timestamp"].dt.to_period("W-SUN")
    best_week = week_key.value_counts().idxmax()
    week_start, week_end = best_week.start_time, best_week.end_time
    print(f"selected week: {week_start} .. {week_end}, n_trades_entered_this_week={int((week_key == best_week).sum())}")

    week_ledger = ledger[(ledger["entry_timestamp"] >= week_start) & (ledger["entry_timestamp"] <= week_end)].copy()
    price_week = panel[(panel["timestamp"] >= week_start) & (panel["timestamp"] <= week_end + pd.Timedelta(hours=6))]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(price_week["timestamp"], price_week["close"], color="#888888", linewidth=0.8, zorder=1)

    color_map = {"tp": "#2ca02c", "sl": "#d62728", "timeout": "#7f7f7f"}
    for _, row in week_ledger.iterrows():
        marker = "^" if row["side"] == 1 else "v"
        entry_price = panel.loc[panel["timestamp"] == row["entry_timestamp"], "open"]
        entry_price = float(entry_price.iloc[0]) if len(entry_price) else np.nan
        exit_price = panel.loc[panel["timestamp"] == row["exit_timestamp"], "close"]
        exit_price = float(exit_price.iloc[0]) if len(exit_price) else np.nan
        c = color_map.get(row["reason"], "#7f7f7f")
        ax.scatter(row["entry_timestamp"], entry_price, marker=marker, color=c, s=70, zorder=3, edgecolors="black", linewidths=0.5)
        ax.plot([row["entry_timestamp"], row["exit_timestamp"]], [entry_price, exit_price], color=c, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

    ax.set_title(f"BTC 5m -- OOS ORACLE (perfect-foresight) trades, true label\n{week_start.date()} ~ {week_end.date()} (up-tri=long down-tri=short, green=TP red=SL gray=timeout)")
    ax.set_ylabel("BTC close (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    fig.autofmt_xdate()
    ax.grid(alpha=0.25)

    n_tp = int((week_ledger["reason"] == "tp").sum())
    n_sl = int((week_ledger["reason"] == "sl").sum())
    n_to = int((week_ledger["reason"] == "timeout").sum())
    win_rate = float((week_ledger["trade_return"] > 0).mean()) if len(week_ledger) else 0.0
    sum_ret = float(week_ledger["trade_return"].sum() * 100)
    ax.text(
        0.01, 0.02,
        f"n_trades={len(week_ledger)}  tp={n_tp} sl={n_sl} timeout={n_to}  win_rate={win_rate:.1%}  sum_ret={sum_ret:+.2f}%",
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
