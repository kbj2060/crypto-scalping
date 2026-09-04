#!/usr/bin/env python3
"""How many minutes before/after the US equity open (NYSE/Nasdaq, 9:30am ET) does ETH realized
volatility actually elevate, and by how much? User question 2026-08-26: "보통 장 시작 몇 분 전부터
몇 분 후까지 변동성이 크고 위험한지" -- this is a pure descriptive/risk question (bar-to-bar
range/return magnitude), NOT a directional-edge question, so it is a different axis from
eth_session_split_edge_2023utc_20260817 (which found 15-16 UTC momentum / 20-23 UTC mean-reversion
IC around NYSE open/close, then cost-gate REJECTED the 20-23 UTC entry rule at 5.49bp breakeven <
10bp needed) -- that memory's rejection was about an ENTRY RULE's economics, not about whether
volatility itself is elevated, so it does not block this analysis; its "NYSE open" IC finding is
used below only as an independent cross-check that something real happens in this window.

Data: data/splits/year_oos/training_features_{2024,2025,2026_rebuilt}.csv (canonical per
reference_clean_data_locations_20260823 memory), OHLCV columns unaudited-but-never-flagged-bad
per that memory -- fine for a purely descriptive volatility profile (no metrics-column integrity
claim needed here). ~2.6 years of 5-minute ETH bars.

Method: tz-aware conversion to America/New_York (handles EDT/EST automatically), restricted to
actual NYSE trading days via pandas_market_calendars (excludes weekends AND US market holidays --
on a holiday there is no real "open" event, so including it would just dilute the signal with
noise). Per-bar realized range (high-low)/close and |close/open-1| in %, binned into 5-minute
buckets of minutes-since-9:30-open, mean + 95% CI (normal approx, n = trading days per bucket)
compared against the all-day baseline mean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data" / "splits" / "year_oos" / "training_features_2024.csv",
    ROOT / "data" / "splits" / "year_oos" / "training_features_2025.csv",
    ROOT / "data" / "splits" / "year_oos" / "training_features_2026_rebuilt.csv",
]
BUCKET_MIN = 5
WINDOW_BEFORE_MIN = 90   # how far before 9:30 ET to look
WINDOW_AFTER_MIN = 150   # how far after 9:30 ET to look
FUNDING_HOURS_UTC = (0, 8, 16)  # secondary/bonus check: perp funding settlement


def load_eth_5m() -> pd.DataFrame:
    frames = []
    for f in FILES:
        df = pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    return out


def add_vol_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100.0
    df["abs_ret_pct"] = (df["close"] / df["open"] - 1.0).abs() * 100.0
    return df


def nyse_trading_days(start, end) -> set:
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start, end_date=end)
    return set(sched.index.date)


def bucket_table(df: pd.DataFrame, minutes_col: str, metric: str) -> pd.DataFrame:
    lo = -((WINDOW_BEFORE_MIN // BUCKET_MIN)) * BUCKET_MIN
    hi = (WINDOW_AFTER_MIN // BUCKET_MIN) * BUCKET_MIN
    edges = np.arange(lo, hi + BUCKET_MIN, BUCKET_MIN)
    df = df[(df[minutes_col] >= lo) & (df[minutes_col] < hi)].copy()
    df["bucket"] = (np.floor(df[minutes_col] / BUCKET_MIN) * BUCKET_MIN).astype(int)
    g = df.groupby("bucket")[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = 1.959964 * g["std"] / np.sqrt(g["count"])
    return g.sort_values("bucket").reset_index(drop=True)


def summarize_window(g: pd.DataFrame, baseline: float, label: str, metric_name: str) -> None:
    g = g.copy()
    g["ratio"] = g["mean"] / baseline
    g["sig_elevated"] = (g["mean"] - g["ci95"]) > baseline
    print(f"\n=== {label} :: {metric_name} (baseline={baseline:.4f}) ===")
    print(g.to_string(index=False, formatters={
        "mean": "{:.4f}".format, "std": "{:.4f}".format, "ci95": "{:.4f}".format, "ratio": "{:.2f}x".format,
    }))
    sig = g[g["sig_elevated"]]
    if len(sig):
        print(f"  -> significantly elevated (mean-95%CI > baseline) buckets: "
              f"{int(sig['bucket'].min())} to {int(sig['bucket'].max()) + BUCKET_MIN} min, "
              f"peak ratio {g['ratio'].max():.2f}x at bucket {int(g.loc[g['ratio'].idxmax(), 'bucket'])}")
    else:
        print("  -> no bucket significantly above baseline")


def main() -> None:
    raw = load_eth_5m()
    print(f"Loaded {len(raw)} bars, {raw['timestamp'].min()} .. {raw['timestamp'].max()}")
    df = add_vol_metrics(raw)

    baseline_range = df["range_pct"].mean()
    baseline_absret = df["abs_ret_pct"].mean()
    print(f"All-bar baseline: range_pct={baseline_range:.4f}%  abs_ret_pct={baseline_absret:.4f}%  (n={len(df)})")

    # --- NYSE open (9:30 ET), restricted to actual NYSE trading days ---
    ny = df.copy()
    ny["ts_ny"] = ny["timestamp"].dt.tz_convert("America/New_York")
    ny["date_ny"] = ny["ts_ny"].dt.date
    trading_days = nyse_trading_days(ny["date_ny"].min(), ny["date_ny"].max())
    ny = ny[ny["date_ny"].isin(trading_days)].copy()
    open_dt = pd.to_datetime(ny["date_ny"].astype(str)) + pd.Timedelta(hours=9, minutes=30)
    open_dt = open_dt.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    ny["minutes_from_open"] = (ny["ts_ny"] - open_dt).dt.total_seconds() / 60.0
    ny = ny.dropna(subset=["minutes_from_open"])
    n_days = ny["date_ny"].nunique()
    print(f"\nNYSE trading days covered: {n_days} (holidays/weekends excluded via pandas_market_calendars)")

    g_range = bucket_table(ny, "minutes_from_open", "range_pct")
    g_absret = bucket_table(ny, "minutes_from_open", "abs_ret_pct")
    summarize_window(g_range, baseline_range, "NYSE open (9:30 ET)", "range_pct (high-low)/close")
    summarize_window(g_absret, baseline_absret, "NYSE open (9:30 ET)", "abs_ret_pct |close/open-1|")

    out_dir = ROOT / "tmp" / "eth_nyse_open_volatility_window_20260826"
    out_dir.mkdir(parents=True, exist_ok=True)
    g_range.to_csv(out_dir / "nyse_open_range_pct_buckets.csv", index=False)
    g_absret.to_csv(out_dir / "nyse_open_absret_pct_buckets.csv", index=False)

    # --- Bonus/secondary: crypto-native funding settlement times (00/08/16 UTC) ---
    print("\n\n### Bonus check: perp funding settlement times (00:00/08:00/16:00 UTC) ###")
    fd = df.copy()
    fd["date_utc"] = fd["timestamp"].dt.date
    fd["hour_utc"] = fd["timestamp"].dt.hour
    fd["minute_utc"] = fd["timestamp"].dt.minute
    fd["min_of_day"] = fd["hour_utc"] * 60 + fd["minute_utc"]
    for fh in FUNDING_HOURS_UTC:
        fd[f"minutes_from_funding_{fh}"] = fd["min_of_day"] - fh * 60
        # wrap so bars just before 00:00 read as small negative rather than ~1435
        fd.loc[fd[f"minutes_from_funding_{fh}"] > 720, f"minutes_from_funding_{fh}"] -= 1440
        fd.loc[fd[f"minutes_from_funding_{fh}"] < -720, f"minutes_from_funding_{fh}"] += 1440
    for fh in FUNDING_HOURS_UTC:
        col = f"minutes_from_funding_{fh}"
        sub = fd[(fd[col] >= -60) & (fd[col] < 60)]
        g = bucket_table(sub.rename(columns={col: "minutes_from_open"}), "minutes_from_open", "range_pct")
        summarize_window(g, baseline_range, f"Funding {fh:02d}:00 UTC", "range_pct")

    print(f"\nWrote bucket tables to {out_dir}")


if __name__ == "__main__":
    main()
