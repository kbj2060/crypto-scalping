"""
Same discretionary Ichimoku/CVD/OI/volume strategy as
scripts/backtest_discretionary_ichimoku_cvd_oi_20260731.py, but run natively on
15-minute and 1-hour bars instead of 5-minute bars (per user follow-up request).

This repo has no 15m/1h feature pipeline, so raw OHLCV + taker_buy_base +
sum_open_interest_value are resampled from data/training_features_5m.csv and
every derived indicator (Ichimoku, ATR, volume oscillator, CVD, OI change rate)
is recomputed natively at the target bar cadence — none of the 5m-native
feature columns (e.g. cvd_slope_12) are reused, since their lookback windows
are meaningless once the bar size changes.

Two runs:
  - 15min bars, gated by a 1h EMA50 trend filter (mirrors the 5m+1h design)
  - 1h bars, no higher-timeframe filter (1h is already the top of this repo's
    available data)

Same VAL/OOS fresh-forward split, same cost1/2/3 convention, same 2R TP with
ATR-floored R, same close-only exits, and no artificial short timeout (only a
1-day-equivalent safety-net cap) as the fixed 5m version.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
MIN_R_ATR_MULT = 1.5
ATR_PERIOD = 14

VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59"


def resample_ohlcv(df5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df5m.set_index("timestamp").resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "taker_buy_base": g["taker_buy_base"].sum(),
        "sum_open_interest_value": g["sum_open_interest_value"].last(),
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["tenkan"] = tenkan
    df["kijun"] = kijun
    df["cloud_top"] = np.maximum(senkou_a, senkou_b)
    df["cloud_bottom"] = np.minimum(senkou_a, senkou_b)
    return df


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()
    return df


def compute_volume_oscillator(df: pd.DataFrame) -> pd.DataFrame:
    vol_ema5 = df["volume"].ewm(span=5, adjust=False).mean()
    vol_ema10 = df["volume"].ewm(span=10, adjust=False).mean()
    df["vol_osc"] = (vol_ema5 - vol_ema10) / vol_ema10.replace(0, np.nan) * 100.0
    return df


def compute_cvd(df: pd.DataFrame, slope_window: int) -> pd.DataFrame:
    taker_sell = df["volume"] - df["taker_buy_base"]
    delta = df["taker_buy_base"] - taker_sell
    cvd = delta.cumsum()
    df["cvd"] = cvd
    df["cvd_slope"] = cvd.diff(slope_window)
    return df


def compute_oi_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["oi_change_rate"] = df["sum_open_interest_value"].pct_change(1)
    return df


def compute_1h_trend_filter(df_native: pd.DataFrame, df5m: pd.DataFrame) -> pd.DataFrame:
    hourly = df5m.set_index("timestamp")["close"].resample("1h").last().dropna()
    hourly_ema50 = hourly.ewm(span=50, adjust=False).mean()
    mtf_up = (hourly > hourly_ema50) & (hourly_ema50.diff() > 0)
    mtf_down = (hourly < hourly_ema50) & (hourly_ema50.diff() < 0)
    mtf = pd.DataFrame({"known_at": hourly.index + pd.Timedelta(hours=1),
                         "mtf_up": mtf_up.values, "mtf_down": mtf_down.values})
    merged = pd.merge_asof(df_native[["timestamp"]], mtf, left_on="timestamp",
                            right_on="known_at", direction="backward")
    df_native["mtf_up"] = merged["mtf_up"].fillna(False).values
    df_native["mtf_down"] = merged["mtf_down"].fillna(False).values
    return df_native


def build_signals(df: pd.DataFrame, use_mtf_filter: bool) -> pd.DataFrame:
    close = df["close"]

    long_raw = (close > df["cloud_top"])
    long_confirmed_next = (close.shift(-1) > df["cloud_bottom"].shift(-1))
    long_volosc_ok = df["vol_osc"].shift(-1) > 0
    long_cvd_oi_ok = (df["cvd_slope"].shift(-1) > 0) & (df["oi_change_rate"].shift(-1) > 0)
    long_confirm = long_raw & long_confirmed_next & long_volosc_ok & long_cvd_oi_ok
    if use_mtf_filter:
        long_confirm = long_confirm & df["mtf_up"].shift(-1).fillna(False)
    df["long_confirm"] = long_confirm

    vol_increase = df["volume"] > df["volume"].shift(1)
    short_raw = (close < df["tenkan"]) & (close < df["cloud_bottom"])
    short_cvd_oi_ok = (df["cvd_slope"] < 0) & (df["oi_change_rate"] > 0)
    short_confirm = short_raw & vol_increase & short_cvd_oi_ok
    if use_mtf_filter:
        short_confirm = short_confirm & df["mtf_down"]
    df["short_confirm"] = short_confirm

    return df


def run_backtest(df: pd.DataFrame, cost_mult: float, time_stop_bars: int) -> dict:
    fee = FEE_RATE * cost_mult
    slip = SLIP_RATE * cost_mult
    round_trip_cost = 2 * (fee + slip)

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
    trade_returns = []
    equity = [1.0]

    i = 0
    while i < n:
        if pos == 1:
            hit_stop = closes[i] < cloud_bottom[i]
            hit_tp = closes[i] >= entry_tp
            hit_time = (i - entry_idx) >= time_stop_bars
            if hit_stop or hit_tp or hit_time:
                ret = (closes[i] / entry_price - 1.0) - round_trip_cost
                trade_returns.append(ret)
                equity.append(equity[-1] * (1 + ret))
                pos = 0
        elif pos == -1:
            hit_stop = closes[i] > cloud_top[i]
            hit_tp = closes[i] <= entry_tp
            hit_time = (i - entry_idx) >= time_stop_bars
            if hit_stop or hit_tp or hit_time:
                ret = (1.0 - closes[i] / entry_price) - round_trip_cost
                trade_returns.append(ret)
                equity.append(equity[-1] * (1 + ret))
                pos = 0

        if pos == 0 and i + 1 < n:
            if long_confirm[i]:
                pos = 1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                atr_floor = MIN_R_ATR_MULT * max(atr[i], 1e-9)
                r = max(entry_price - cloud_bottom[i], atr_floor, 1e-9)
                entry_tp = entry_price + 2 * r
                i += 1
            elif short_confirm[i]:
                pos = -1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                atr_floor = MIN_R_ATR_MULT * max(atr[i], 1e-9)
                r = max(cloud_top[i] - entry_price, atr_floor, 1e-9)
                entry_tp = entry_price - 2 * r
                i += 1
        i += 1

    trade_returns = np.array(trade_returns)
    if len(trade_returns) == 0:
        return {"trades": 0, "total_return_pct": 0.0, "win_rate_pct": 0.0,
                "mdd_pct": 0.0, "avg_trade_pct": 0.0}

    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0

    return {
        "trades": int(len(trade_returns)),
        "total_return_pct": round((eq[-1] - 1.0) * 100, 2),
        "win_rate_pct": round((trade_returns > 0).mean() * 100, 2),
        "mdd_pct": round(dd.min() * 100, 2),
        "avg_trade_pct": round(trade_returns.mean() * 100, 3),
    }


def run_for_timeframe(df5m: pd.DataFrame, rule: str, label: str,
                       slope_window: int, use_mtf_filter: bool, time_stop_bars: int) -> pd.DataFrame:
    df = resample_ohlcv(df5m, rule)
    df = compute_ichimoku(df)
    df = compute_atr(df)
    df = compute_volume_oscillator(df)
    df = compute_cvd(df, slope_window)
    df = compute_oi_change_rate(df)
    if use_mtf_filter:
        df = compute_1h_trend_filter(df, df5m)
    df = build_signals(df, use_mtf_filter)

    windows = {
        "FULL": (df["timestamp"].min(), df["timestamp"].max()),
        "VAL (2025-09-01..2025-12-31)": (pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)),
        "OOS (2026-01-01..2026-03-31)": (pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)),
    }

    rows = []
    for wlabel, (start, end) in windows.items():
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
        for cost_mult in (1, 2, 3):
            res = run_backtest(sub, cost_mult=cost_mult, time_stop_bars=time_stop_bars)
            res["timeframe"] = label
            res["window"] = wlabel
            res["cost_mult"] = cost_mult
            rows.append(res)
    return pd.DataFrame(rows)


def main():
    df5m = pd.read_csv("data/training_features_5m.csv")
    df5m["timestamp"] = pd.to_datetime(df5m["timestamp"])
    df5m = df5m.sort_values("timestamp").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume", "taker_buy_base", "sum_open_interest_value"]
    missing = [c for c in required if c not in df5m.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    # 15min: slope window 4 bars = 1h CVD trend (same concept as 5m's 12-bar/1h window),
    # 1h MTF filter still applies. Safety-net time-stop ~1 day = 96 bars of 15min.
    res_15m = run_for_timeframe(df5m, "15min", "15min+1h_filter",
                                 slope_window=4, use_mtf_filter=True, time_stop_bars=96)

    # 1h: native bars, no higher-timeframe filter (1h is already the top available
    # timeframe in this dataset). slope window=1 bar = 1h CVD trend. Safety-net
    # time-stop ~1 week = 168 bars of 1h.
    res_1h = run_for_timeframe(df5m, "1h", "1h_native",
                                slope_window=1, use_mtf_filter=False, time_stop_bars=168)

    out = pd.concat([res_15m, res_1h], ignore_index=True)
    out = out[["timeframe", "window", "cost_mult", "trades", "total_return_pct",
               "win_rate_pct", "mdd_pct", "avg_trade_pct"]]
    print(out.to_string(index=False))
    out.to_csv("data/ensemble/reports/discretionary_ichimoku_cvd_oi_mtf_backtest_20260731.csv", index=False)


if __name__ == "__main__":
    main()
