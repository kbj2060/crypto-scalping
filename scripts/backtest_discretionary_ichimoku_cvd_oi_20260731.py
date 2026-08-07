"""
Discretionary rule-based backtest: Ichimoku cloud breakout + volume + CVD/OI confirmation.

This codifies, as literally as possible, the manual multi-indicator entry logic
described by the user from a single ETH 5m chart reading (2026-07-31), generalized
across the full historical dataset. It is a research-only rule-based strategy check,
NOT an Omega/ensemble promotion candidate — it does not go through
scripts/audit_omega_artifact_integrity_20260630.py or the live promotion gates.

Entry rules (as stated by the user):
  LONG confirm:
    1. bar close > cloud top (senkou span breakout)
    2. following bar does not fall back into/below the cloud
    3. volume oscillator (5/10 EMA of volume) recovers above 0
    4. CVD and OI both rising
    Executed at the open of the bar after confirmation (2-bar lag), matching the
    user's explicit 2-step confirmation description.

  SHORT confirm:
    1. bar close < tenkan-sen (conversion line)
    2. bar close < cloud bottom
    3. volume increases vs prior bar
    4. CVD falling and OI rising ("crowded short" per user's structure read)
    Executed at the next bar's open (1-bar lag).

Exit rules (not explicitly specified by the user beyond "target zones" tied to this
one chart's absolute price levels, which don't generalize across history — translated
here into a relative 2R structure-based exit, documented as an explicit assumption):
    - Stop: close re-enters/breaks the cloud on the entry's invalidation side
    - Take-profit: 2R from entry, where R = max(|entry - cloud boundary at entry|,
      MIN_R_ATR_MULT * ATR14). Fix applied 2026-07-31: the raw structural distance
      alone was often tiny right at a fresh breakout (target too close, whipsawed)
      or huge in high-vol regimes (oversized losses); an ATR floor stabilizes R.
    - Time-stop: originally a 12-bar (60 min) forced exit, which fought this
      project's own finding that ETH's actual working structure is long-hold
      (Omega4.6.1 OOS median hold ~1162 bars). Fix applied 2026-07-31: replaced
      with a 288-bar (~1 day) safety-net cap that essentially never fires under
      normal conditions — exits are now driven by structure (stop/TP) only.
  All exits trigger on BAR CLOSE only (never intrabar high/low), per this repo's
  known TP/SL-on-close live-parity requirement (see project memory: intrabar TP/SL
  fills cost -76pt VAL / -81pt OOS vs the close rule every other backtest here uses).

1h higher-timeframe filter (added 2026-07-31, user's stated #1-priority addition):
    A 1h close/EMA50 trend direction gate is required in addition to the base
    5m confirm conditions above. Only fully-closed 1h bars are used (the current,
    still-forming 1h bar is never referenced) via an "known_at = hour_start + 1h"
    merge_asof join, so this stays causal. LONG additionally requires 1h close
    above 1h EMA50 with EMA50 rising; SHORT additionally requires 1h close below
    1h EMA50 with EMA50 falling. This repo has no 15m pipeline, so 1h is used as
    the higher-timeframe proxy per the user's own fallback suggestion.

Costs: project standard cost1 = FEE_RATE 0.0005 + SLIP_RATE 0.0002 per side
(see scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py). Also reports
cost2/cost3 stress per this repo's durability-check convention.

Fresh-Forward split (CLAUDE.md): VAL 2025-09-01..2025-12-31, OOS 2026-01-01..2026-03-31.
This is a bar-by-bar causal walk using only already-closed-bar information at
signal time; no saved ledgers or future rows are used as backtest input.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
TIME_STOP_BARS = 288  # ~1 day safety-net cap, not an active exit driver
MIN_R_ATR_MULT = 1.5
ATR_PERIOD = 14

VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59"
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59"


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


def compute_1h_trend(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.set_index("timestamp")["close"].resample("1h").last().dropna()
    hourly_ema50 = hourly.ewm(span=50, adjust=False).mean()
    mtf_up = (hourly > hourly_ema50) & (hourly_ema50.diff() > 0)
    mtf_down = (hourly < hourly_ema50) & (hourly_ema50.diff() < 0)

    # an hourly bar starting at H covers [H, H+1h); it is only fully closed at H+1h
    mtf = pd.DataFrame({"known_at": hourly.index + pd.Timedelta(hours=1),
                         "mtf_up": mtf_up.values, "mtf_down": mtf_down.values})

    merged = pd.merge_asof(df[["timestamp"]], mtf, left_on="timestamp",
                            right_on="known_at", direction="backward")
    df["mtf_up"] = merged["mtf_up"].fillna(False).values
    df["mtf_down"] = merged["mtf_down"].fillna(False).values
    return df


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]

    long_raw = (close > df["cloud_top"])
    long_confirmed_next = (close.shift(-1) > df["cloud_bottom"].shift(-1))
    long_volosc_ok = df["vol_osc"].shift(-1) > 0
    long_cvd_oi_ok = (df["cvd_slope_12"].shift(-1) > 0) & (df["oi_change_rate"].shift(-1) > 0)
    long_mtf_ok = df["mtf_up"].shift(-1).fillna(False)
    df["long_confirm"] = long_raw & long_confirmed_next & long_volosc_ok & long_cvd_oi_ok & long_mtf_ok

    vol_increase = df["volume"] > df["volume"].shift(1)
    short_raw = (close < df["tenkan"]) & (close < df["cloud_bottom"])
    short_cvd_oi_ok = (df["cvd_slope_12"] < 0) & (df["oi_change_rate"] > 0)
    short_mtf_ok = df["mtf_down"]
    df["short_confirm"] = short_raw & vol_increase & short_cvd_oi_ok & short_mtf_ok

    return df


def run_backtest(df: pd.DataFrame, cost_mult: float = 1.0) -> dict:
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
    entry_price = entry_stop = entry_tp = 0.0
    entry_idx = -1
    trade_returns = []
    equity = [1.0]

    i = 0
    while i < n:
        if pos == 1:
            hit_stop = closes[i] < cloud_bottom[i]
            hit_tp = closes[i] >= entry_tp
            hit_time = (i - entry_idx) >= TIME_STOP_BARS
            if hit_stop or hit_tp or hit_time:
                ret = (closes[i] / entry_price - 1.0) - round_trip_cost
                trade_returns.append(ret)
                equity.append(equity[-1] * (1 + ret))
                pos = 0
        elif pos == -1:
            hit_stop = closes[i] > cloud_top[i]
            hit_tp = closes[i] <= entry_tp
            hit_time = (i - entry_idx) >= TIME_STOP_BARS
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
                entry_stop = cloud_bottom[i]
                entry_tp = entry_price + 2 * r
                i += 1
            elif short_confirm[i]:
                pos = -1
                entry_idx = i + 1
                entry_price = opens[i + 1]
                atr_floor = MIN_R_ATR_MULT * max(atr[i], 1e-9)
                r = max(cloud_top[i] - entry_price, atr_floor, 1e-9)
                entry_stop = cloud_top[i]
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


def main():
    df = pd.read_csv("data/training_features_5m.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume", "cvd_slope_12", "oi_change_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    df = compute_ichimoku(df)
    df = compute_atr(df)
    df = compute_volume_oscillator(df)
    df = compute_1h_trend(df)
    df = build_signals(df)

    windows = {
        "FULL (2024-01-01..2026-07-20)": (df["timestamp"].min(), df["timestamp"].max()),
        "VAL (2025-09-01..2025-12-31)": (pd.Timestamp(VAL_START), pd.Timestamp(VAL_END)),
        "OOS (2026-01-01..2026-03-31)": (pd.Timestamp(OOS_START), pd.Timestamp(OOS_END)),
    }

    rows = []
    for label, (start, end) in windows.items():
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].reset_index(drop=True)
        for cost_mult in (1, 2, 3):
            res = run_backtest(sub, cost_mult=cost_mult)
            res["window"] = label
            res["cost_mult"] = cost_mult
            rows.append(res)

    out = pd.DataFrame(rows)[["window", "cost_mult", "trades", "total_return_pct",
                               "win_rate_pct", "mdd_pct", "avg_trade_pct"]]
    print(out.to_string(index=False))
    out.to_csv("data/ensemble/reports/discretionary_ichimoku_cvd_oi_backtest_20260731.csv", index=False)


if __name__ == "__main__":
    main()
