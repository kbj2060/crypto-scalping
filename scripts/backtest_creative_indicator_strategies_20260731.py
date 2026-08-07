"""
Three original rule-based strategy ideas on ETH 1h bars, built from the same
auxiliary indicators explored in this session (CVD, OI, ATR) but with distinct
trading logic, not parameter variants of the earlier Ichimoku breakout template:

  A. CVD-Price Divergence Fade (mean reversion)
     Price makes a new N-bar high while CVD fails to make a new N-bar high
     (buying momentum not confirming the price high) -> fade short.
     Mirror for bullish divergence -> fade long.

  B. OI Crowding Unwind (contrarian, event-triggered)
     Rapid OI buildup (z-scored 10-bar sum of oi_change_rate) aligned with a
     strong same-direction price move over the last 10 bars = a crowded
     positioning event. Enter against the crowd only on the FIRST bar the
     move starts reversing (confirmation, not anticipation).

  C. CVD-OI Momentum Alignment (trend continuation, trailing exit)
     CVD slope, OI buildup, and price return all aligned over the last 3 bars
     -> enter with the move. No fixed take-profit; ATR chandelier trailing
     stop lets winners run instead of capping at a fixed R multiple.

All three use 1h bars (resampled from data/training_features_5m.csv, since 5m/15m
were already rejected this session), close-only exits, project-standard
cost1/2/3, and the same VAL/OOS fresh-forward split as the rest of this session's
research. Research-only — none of these go through the Omega promotion gates.
"""
import numpy as np
import pandas as pd

from backtest_discretionary_ichimoku_cvd_oi_mtf_20260731 import (
    resample_ohlcv, compute_atr, compute_cvd, compute_oi_change_rate,
    FEE_RATE, SLIP_RATE, VAL_START, VAL_END, OOS_START, OOS_END,
)


def compute_shared(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_atr(df)
    df = compute_cvd(df, slope_window=3)
    df = compute_oi_change_rate(df)
    df["oi_buildup10"] = df["oi_change_rate"].rolling(10).sum()
    df["oi_buildup10_z"] = (df["oi_buildup10"] - df["oi_buildup10"].rolling(100).mean()) / df["oi_buildup10"].rolling(100).std()
    df["price_ret10"] = df["close"].pct_change(10)
    return df


def signals_divergence_fade(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    close, cvd = df["close"], df["cvd"]
    price_new_high = close >= close.rolling(window).max()
    price_new_low = close <= close.rolling(window).min()
    cvd_at_high = cvd >= cvd.rolling(window).max()
    cvd_at_low = cvd <= cvd.rolling(window).min()
    df["short_confirm"] = price_new_high & (~cvd_at_high)
    df["long_confirm"] = price_new_low & (~cvd_at_low)
    return df


def signals_oi_crowding_unwind(df: pd.DataFrame, z_thresh: float = 1.5) -> pd.DataFrame:
    crowded_long = (df["oi_buildup10_z"] > z_thresh) & (df["price_ret10"] > 0)
    crowded_short = (df["oi_buildup10_z"] > z_thresh) & (df["price_ret10"] < 0)
    first_down_close = df["close"] < df["close"].shift(1)
    first_up_close = df["close"] > df["close"].shift(1)
    df["short_confirm"] = crowded_long.shift(1).fillna(False) & first_down_close
    df["long_confirm"] = crowded_short.shift(1).fillna(False) & first_up_close
    return df


def signals_momentum_alignment(df: pd.DataFrame) -> pd.DataFrame:
    cvd_slope_up = df["cvd_slope"] > 0
    cvd_slope_down = df["cvd_slope"] < 0
    oi_up = df["oi_buildup10"] > 0
    ret3 = df["close"].pct_change(3)
    df["long_confirm"] = cvd_slope_up & oi_up & (ret3 > 0)
    df["short_confirm"] = cvd_slope_down & oi_up & (ret3 < 0)
    return df


def run_fixed_rr(df: pd.DataFrame, cost_mult: float, sl_atr_mult: float,
                  tp_atr_mult: float, time_stop_bars: int) -> np.ndarray:
    fee, slip = FEE_RATE * cost_mult, SLIP_RATE * cost_mult
    round_trip_cost = 2 * (fee + slip)
    opens, closes, atr = df["open"].values, df["close"].values, df["atr"].values
    long_confirm, short_confirm = df["long_confirm"].values, df["short_confirm"].values

    n = len(df)
    pos = 0
    entry_price = sl_price = tp_price = 0.0
    entry_idx = -1
    trade_returns = []

    i = 0
    while i < n:
        if pos == 1:
            hit = closes[i] <= sl_price or closes[i] >= tp_price or (i - entry_idx) >= time_stop_bars
            if hit:
                trade_returns.append((closes[i] / entry_price - 1.0) - round_trip_cost)
                pos = 0
        elif pos == -1:
            hit = closes[i] >= sl_price or closes[i] <= tp_price or (i - entry_idx) >= time_stop_bars
            if hit:
                trade_returns.append((1.0 - closes[i] / entry_price) - round_trip_cost)
                pos = 0

        if pos == 0 and i + 1 < n:
            a = max(atr[i], 1e-9)
            if long_confirm[i]:
                pos = 1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                sl_price = entry_price - sl_atr_mult * a
                tp_price = entry_price + tp_atr_mult * a
                i += 1
            elif short_confirm[i]:
                pos = -1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                sl_price = entry_price + sl_atr_mult * a
                tp_price = entry_price - tp_atr_mult * a
                i += 1
        i += 1
    return np.array(trade_returns)


def run_trailing(df: pd.DataFrame, cost_mult: float, chandelier_atr_mult: float,
                  time_stop_bars: int) -> np.ndarray:
    fee, slip = FEE_RATE * cost_mult, SLIP_RATE * cost_mult
    round_trip_cost = 2 * (fee + slip)
    opens, closes, atr = df["open"].values, df["close"].values, df["atr"].values
    long_confirm, short_confirm = df["long_confirm"].values, df["short_confirm"].values

    n = len(df)
    pos = 0
    entry_price = trail_stop = extreme = 0.0
    entry_idx = -1
    trade_returns = []

    i = 0
    while i < n:
        if pos == 1:
            extreme = max(extreme, closes[i])
            trail_stop = extreme - chandelier_atr_mult * max(atr[i], 1e-9)
            hit = closes[i] <= trail_stop or (i - entry_idx) >= time_stop_bars
            if hit:
                trade_returns.append((closes[i] / entry_price - 1.0) - round_trip_cost)
                pos = 0
        elif pos == -1:
            extreme = min(extreme, closes[i])
            trail_stop = extreme + chandelier_atr_mult * max(atr[i], 1e-9)
            hit = closes[i] >= trail_stop or (i - entry_idx) >= time_stop_bars
            if hit:
                trade_returns.append((1.0 - closes[i] / entry_price) - round_trip_cost)
                pos = 0

        if pos == 0 and i + 1 < n:
            a = max(atr[i], 1e-9)
            if long_confirm[i]:
                pos = 1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                extreme = entry_price
                trail_stop = entry_price - chandelier_atr_mult * a
                i += 1
            elif short_confirm[i]:
                pos = -1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                extreme = entry_price
                trail_stop = entry_price + chandelier_atr_mult * a
                i += 1
        i += 1
    return np.array(trade_returns)


def stats(trade_returns: np.ndarray) -> dict:
    if len(trade_returns) == 0:
        return {"trades": 0, "total_return_pct": 0.0, "win_rate_pct": 0.0, "mdd_pct": 0.0, "avg_trade_pct": 0.0}
    eq = np.cumprod(1 + trade_returns)
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {
        "trades": int(len(trade_returns)),
        "total_return_pct": round((eq[-1] - 1.0) * 100, 2),
        "win_rate_pct": round((trade_returns > 0).mean() * 100, 2),
        "mdd_pct": round(dd.min() * 100, 2),
        "avg_trade_pct": round(trade_returns.mean() * 100, 3),
    }


def main():
    df5m = pd.read_csv("data/training_features_5m.csv")
    df5m["timestamp"] = pd.to_datetime(df5m["timestamp"])
    df5m = df5m.sort_values("timestamp").reset_index(drop=True)

    base = resample_ohlcv(df5m, "1h")
    base = compute_shared(base)

    strategies = {
        "A_cvd_divergence_fade": (signals_divergence_fade, run_fixed_rr,
                                   dict(sl_atr_mult=1.5, tp_atr_mult=1.5, time_stop_bars=24)),
        "B_oi_crowding_unwind": (signals_oi_crowding_unwind, run_fixed_rr,
                                  dict(sl_atr_mult=2.0, tp_atr_mult=3.0, time_stop_bars=48)),
        "C_momentum_alignment_trailing": (signals_momentum_alignment, run_trailing,
                                           dict(chandelier_atr_mult=3.0, time_stop_bars=120)),
    }

    windows = {
        "VAL (2025-09-01..2025-12-31)": (pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)),
        "OOS (2026-01-01..2026-03-31)": (pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)),
    }

    rows = []
    for name, (sig_fn, run_fn, kwargs) in strategies.items():
        df = sig_fn(base.copy())
        for wlabel, (start, end) in windows.items():
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
            for cost_mult in (1, 2, 3):
                tr = run_fn(sub, cost_mult=cost_mult, **kwargs)
                res = stats(tr)
                res["strategy"] = name
                res["window"] = wlabel
                res["cost_mult"] = cost_mult
                rows.append(res)

    out = pd.DataFrame(rows)[["strategy", "window", "cost_mult", "trades",
                               "total_return_pct", "win_rate_pct", "mdd_pct", "avg_trade_pct"]]
    print(out.to_string(index=False))
    out.to_csv("data/ensemble/reports/creative_indicator_strategies_20260731.csv", index=False)


if __name__ == "__main__":
    main()
