#!/usr/bin/env python3
"""Follow-up to research_eth_nyse_open_volatility_window_20260826.py -- user asked to extend the
same ETH realized-volatility-by-time-of-day analysis to Japan (JPX) and Europe (LSE) opens too,
focused on a tighter +-60min window around all three, for direct side-by-side comparison.

Uses pandas_market_calendars' actual per-day schedule (market_open, already UTC + DST-aware for
LSE, no-DST for JPX) rather than hand-building local wall-clock times -- more robust than the
previous script's manual America/New_York string construction, same idea eth_session_split_edge_
2023utc_20260817 used (us=NYSE / europe=LSE / asia=JPX mcal) for calendar choice consistency.

Same data/metric/baseline as the NYSE-only script (see that file's docstring for the "why this is
a different axis from the session-split IC/cost-gate work" note -- unchanged here)."""
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
WINDOW_MIN = 60  # +-1h, per user request

MARKETS = [
    ("JPX", "일본 (JPX, 09:00 JST)"),
    ("LSE", "유럽 (LSE, 08:00 London)"),
    ("NYSE", "미국 (NYSE, 09:30 ET)"),
]


def load_eth_5m() -> pd.DataFrame:
    frames = [pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
              for f in FILES]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    out["timestamp"] = out["timestamp"].dt.tz_localize("UTC")
    out["range_pct"] = (out["high"] - out["low"]) / out["close"] * 100.0
    out["abs_ret_pct"] = (out["close"] / out["open"] - 1.0).abs() * 100.0
    return out


def bucket_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    edges_lo, edges_hi = -WINDOW_MIN, WINDOW_MIN
    d = df[(df["minutes_from_open"] >= edges_lo) & (df["minutes_from_open"] < edges_hi)].copy()
    d["bucket"] = (np.floor(d["minutes_from_open"] / BUCKET_MIN) * BUCKET_MIN).astype(int)
    g = d.groupby("bucket")[metric].agg(["mean", "std", "count"]).reset_index()
    g["ci95"] = 1.959964 * g["std"] / np.sqrt(g["count"])
    return g.sort_values("bucket").reset_index(drop=True)


def attach_minutes_from_open(df: pd.DataFrame, calendar_name: str) -> pd.DataFrame:
    cal = mcal.get_calendar(calendar_name)
    sched = cal.schedule(start_date=df["timestamp"].min().date(), end_date=df["timestamp"].max().date())
    opens = sched[["market_open"]].reset_index(drop=True).rename(columns={"market_open": "open_ts"})
    opens = opens.sort_values("open_ts").reset_index(drop=True)
    merged = pd.merge_asof(df.sort_values("timestamp"), opens, left_on="timestamp", right_on="open_ts",
                            direction="nearest")
    merged["minutes_from_open"] = (merged["timestamp"] - merged["open_ts"]).dt.total_seconds() / 60.0
    return merged[merged["minutes_from_open"].abs() <= WINDOW_MIN].copy(), len(opens)


def summarize(g: pd.DataFrame, baseline: float) -> dict:
    g = g.copy()
    g["ratio"] = g["mean"] / baseline
    elevated = g[g["ratio"] >= 1.5]
    return {
        "peak_bucket": int(g.loc[g["ratio"].idxmax(), "bucket"]),
        "peak_ratio": float(g["ratio"].max()),
        "at_open_ratio": float(g.loc[g["bucket"] == 0, "ratio"].iloc[0]) if 0 in g["bucket"].values else float("nan"),
        "elevated_1p5x_range": (int(elevated["bucket"].min()), int(elevated["bucket"].max()) + BUCKET_MIN) if len(elevated) else None,
        "table": g,
    }


def main() -> None:
    df = load_eth_5m()
    baseline_range = df["range_pct"].mean()
    baseline_absret = df["abs_ret_pct"].mean()
    print(f"Loaded {len(df)} bars, {df['timestamp'].min()} .. {df['timestamp'].max()}")
    print(f"Baseline: range_pct={baseline_range:.4f}%  abs_ret_pct={baseline_absret:.4f}%\n")

    results = {}
    for cal_name, label in MARKETS:
        tagged, n_days = attach_minutes_from_open(df, cal_name)
        g_range = bucket_table(tagged, "range_pct")
        g_absret = bucket_table(tagged, "abs_ret_pct")
        s_range = summarize(g_range, baseline_range)
        s_absret = summarize(g_absret, baseline_absret)
        results[cal_name] = (label, n_days, s_range, s_absret)

        print(f"=== {label} -- {n_days} trading days ===")
        print("-- range_pct (high-low)/close, ratio vs baseline --")
        t = s_range["table"].copy()
        print(t.to_string(index=False, formatters={"mean": "{:.4f}".format, "std": "{:.4f}".format,
                                                     "ci95": "{:.4f}".format, "ratio": "{:.2f}x".format}))
        print(f"  at-open(bucket0) ratio={s_range['at_open_ratio']:.2f}x | "
              f"peak={s_range['peak_ratio']:.2f}x @ bucket {s_range['peak_bucket']} | "
              f">=1.5x window: {s_range['elevated_1p5x_range']}")
        print()

    print("\n=== Cross-market summary (range_pct) ===")
    print(f"{'market':<28}{'at-open ratio':>14}{'peak ratio':>12}{'peak @':>10}{'>=1.5x window (min)':>22}")
    for cal_name, label in MARKETS:
        _, n_days, s_range, _ = results[cal_name]
        win = s_range["elevated_1p5x_range"]
        win_str = f"{win[0]:+d} to {win[1]:+d}" if win else "none"
        print(f"{label:<28}{s_range['at_open_ratio']:>13.2f}x{s_range['peak_ratio']:>11.2f}x"
              f"{s_range['peak_bucket']:>+10d}{win_str:>22}")

    out_dir = ROOT / "tmp" / "eth_multi_market_open_volatility_20260826"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cal_name, label in MARKETS:
        _, _, s_range, s_absret = results[cal_name]
        s_range["table"].to_csv(out_dir / f"{cal_name.lower()}_range_pct_buckets.csv", index=False)
        s_absret["table"].to_csv(out_dir / f"{cal_name.lower()}_absret_pct_buckets.csv", index=False)
    print(f"\nWrote per-market bucket tables to {out_dir}")


if __name__ == "__main__":
    main()
