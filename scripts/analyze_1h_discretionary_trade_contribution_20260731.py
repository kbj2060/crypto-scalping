"""
Trade-level breakdown of the 1h_native discretionary strategy from
scripts/backtest_discretionary_ichimoku_cvd_oi_mtf_20260731.py, to check whether
its VAL/OOS positive result is a broad-based edge or driven by a handful of
outsized trades (concentration risk / luck check before trusting the aggregate
numbers).
"""
import numpy as np
import pandas as pd

from backtest_discretionary_ichimoku_cvd_oi_mtf_20260731 import (
    resample_ohlcv, compute_ichimoku, compute_atr, compute_volume_oscillator,
    compute_cvd, compute_oi_change_rate, build_signals,
    FEE_RATE, SLIP_RATE, MIN_R_ATR_MULT,
    VAL_START, VAL_END, OOS_START, OOS_END,
)


def run_backtest_with_trades(df: pd.DataFrame, cost_mult: float, time_stop_bars: int) -> pd.DataFrame:
    fee = FEE_RATE * cost_mult
    slip = SLIP_RATE * cost_mult
    round_trip_cost = 2 * (fee + slip)

    ts = df["timestamp"].values
    opens = df["open"].values
    closes = df["close"].values
    cloud_top = df["cloud_top"].values
    cloud_bottom = df["cloud_bottom"].values
    atr = df["atr"].values
    long_confirm = df["long_confirm"].values
    short_confirm = df["short_confirm"].values

    n = len(df)
    pos = 0
    entry_price = entry_tp = 0.0
    entry_idx = -1
    side = 0
    trades = []

    i = 0
    while i < n:
        if pos != 0:
            if pos == 1:
                hit_stop = closes[i] < cloud_bottom[i]
                hit_tp = closes[i] >= entry_tp
            else:
                hit_stop = closes[i] > cloud_top[i]
                hit_tp = closes[i] <= entry_tp
            hit_time = (i - entry_idx) >= time_stop_bars
            if hit_stop or hit_tp or hit_time:
                if pos == 1:
                    ret = (closes[i] / entry_price - 1.0) - round_trip_cost
                else:
                    ret = (1.0 - closes[i] / entry_price) - round_trip_cost
                trades.append({
                    "entry_time": ts[entry_idx], "exit_time": ts[i],
                    "side": "long" if side == 1 else "short",
                    "bars_held": i - entry_idx,
                    "exit_reason": "tp" if hit_tp else ("stop" if hit_stop else "time"),
                    "return_pct": ret * 100,
                })
                pos = 0

        if pos == 0 and i + 1 < n:
            if long_confirm[i]:
                pos, side = 1, 1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                atr_floor = MIN_R_ATR_MULT * max(atr[i], 1e-9)
                r = max(entry_price - cloud_bottom[i], atr_floor, 1e-9)
                entry_tp = entry_price + 2 * r
                i += 1
            elif short_confirm[i]:
                pos, side = -1, -1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                atr_floor = MIN_R_ATR_MULT * max(atr[i], 1e-9)
                r = max(cloud_top[i] - entry_price, atr_floor, 1e-9)
                entry_tp = entry_price - 2 * r
                i += 1
        i += 1

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, label: str):
    if trades.empty:
        print(f"\n=== {label}: no trades ===")
        return
    trades = trades.sort_values("return_pct", ascending=False).reset_index(drop=True)
    total_simple_sum = trades["return_pct"].sum()
    n = len(trades)

    print(f"\n=== {label}: {n} trades, sum of simple returns = {total_simple_sum:.2f}% ===")
    for top_k in (1, 3, 5, 10):
        if top_k > n:
            continue
        top_sum = trades["return_pct"].head(top_k).sum()
        share = (top_sum / total_simple_sum * 100) if total_simple_sum != 0 else float("nan")
        print(f"  top {top_k:>2} winning trades contribute {top_sum:7.2f}% of {total_simple_sum:.2f}% total ({share:5.1f}%)")

    worst_sum = trades["return_pct"].tail(5).sum()
    print(f"  worst 5 trades sum: {worst_sum:.2f}%")
    print(f"  win rate: {(trades['return_pct'] > 0).mean()*100:.1f}%  "
          f"exit_reason counts: {trades['exit_reason'].value_counts().to_dict()}")
    print("\n  Top 5 individual trades:")
    print(trades[["entry_time", "exit_time", "side", "bars_held", "exit_reason", "return_pct"]].head(5).to_string(index=False))


def main():
    df5m = pd.read_csv("data/training_features_5m.csv")
    df5m["timestamp"] = pd.to_datetime(df5m["timestamp"])
    df5m = df5m.sort_values("timestamp").reset_index(drop=True)

    df = resample_ohlcv(df5m, "1h")
    df = compute_ichimoku(df)
    df = compute_atr(df)
    df = compute_volume_oscillator(df)
    df = compute_cvd(df, slope_window=1)
    df = compute_oi_change_rate(df)
    df = build_signals(df, use_mtf_filter=False)

    windows = {
        "VAL (2025-09-01..2025-12-31)": (pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)),
        "OOS (2026-01-01..2026-03-31)": (pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)),
    }

    for label, (start, end) in windows.items():
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
        trades = run_backtest_with_trades(sub, cost_mult=1.0, time_stop_bars=168)
        summarize(trades, label)


if __name__ == "__main__":
    main()
