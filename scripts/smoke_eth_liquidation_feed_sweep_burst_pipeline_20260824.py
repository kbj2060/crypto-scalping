#!/usr/bin/env python3
"""Pipeline SMOKE TEST for §14 of docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md
("스윕×청산버스트 임팩트-크기 arm"), 2026-08-24 -- user asked whether a smoke test is possible with
the data collected so far. tail_risk_1m's valid epoch only started 2026-07-18 (WS endpoint bug
before that -- see that doc's §9); the §14 decision gate itself is pre-registered for >=2026-09-15
(8 weeks of valid data) and MUST NOT be run early, or the pre-registration is contaminated.

*** THIS SCRIPT COMPUTES EVENT COUNTS ONLY. IT DOES NOT COMPUTE RETURNS, LIFT, bp IMPACT, OR ANY
    OTHER OUTCOME METRIC. That is deliberate -- counting how many qualifying events exist and
    verifying the pipeline runs without error is pipeline validation (same category as
    eth_candidate_lob_ofi_pipeline_smoke_test_20260822), not a look at the hypothesis's outcome, so
    it does not contaminate the pre-registered §14 gate. Do NOT extend this script to compute
    forward returns before 2026-09-15 -- that IS the pre-registered test and must wait. ***

=== What this verifies ===
1. The pipeline runs end-to-end without error on real current data: tail_risk_1m (pulled from the
   server, valid window only) + fresh ETH 5m klines, causal liq_net_z_12 computation, expanding
   causal decile ranking, and the deployed liquidity_sweep_low/high formula (reused verbatim from
   scripts/live_evidence_signal_dashboard_20260823.py -- no new formula invented, matching §14's
   "새 수식 발명 없음" requirement).
2. How many qualifying events exist in the ~37 days of valid data so far, for ALL FOUR sweep x
   liquidation-decile combinations -- not just the two §14 names as "bottom"/"top", because of an
   ambiguity flagged below.

=== Ambiguity flagged, not silently resolved ===
§14's text pairs bottom=sweep_low AND short-liq-dominant(lower decile), top=sweep_high AND
long-liq-dominant(upper decile). But §11/§12's own established CONTRARIAN direction (long-liq
clustering -> price up next, short-liq clustering -> price down next) and simple liquidation
mechanics (a long liquidation IS a forced sell, driving price down INTO a sweep_low; a short
liquidation IS a forced buy, driving price up INTO a sweep_high) both point the OTHER way: sweep_low
co-occurring with a LONG-liq-dominant (upper decile) burst, sweep_high with a SHORT-liq-dominant
(lower decile) burst. This script does not decide which pairing §14 meant -- it reports all four
combination counts so a human can resolve this before the real 09-15 run, rather than silently
locking in one interpretation now.

=== liq_net_z_12 (verbatim from §4 item 3) ===
(trailing 12min long liq USD sum - short liq USD sum) / (trailing 2880min total liq USD rolling
mean + 1%epsilon), computed causally at 1-minute resolution, existence-rate gated (window's 1m
presence must be >=80% or the point is NaN, never gap-filled with zero).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.live_evidence_signal_dashboard_20260823 import compute_signals

ROOT = Path(__file__).resolve().parents[1]
TAIL_RISK_CSV = ROOT / "data" / "research" / "tail_risk_1m_valid_window_20260824.csv"
PRICE_CSV = ROOT / "data" / "research" / "eth_5m_20260715_to_now_20260824.csv"

NET_WINDOW_MIN = 12
TOTAL_TRAILING_MIN = 2880  # 2 days
EXISTENCE_RATE_MIN = 0.80
DECILE_THRESHOLD = 0.10  # top/bottom 10%
COMBINE_TOLERANCE_BARS = 1  # "same 5m bar or the immediately preceding bar"
VALID_EPOCH_START = pd.Timestamp("2026-07-18 15:03:00", tz="UTC")  # first real nonzero minute


def load_tail_risk() -> pd.DataFrame:
    df = pd.read_csv(TAIL_RISK_CSV, parse_dates=["ts"])
    # server stores KST (UTC+9) -- convert to UTC to match the price series and the doc's UTC dates
    df["ts"] = df["ts"].dt.tz_convert("UTC") if df["ts"].dt.tz is not None else df["ts"].dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    full_index = pd.date_range(df["ts"].iloc[0], df["ts"].iloc[-1], freq="1min", tz="UTC")
    df = df.set_index("ts").reindex(full_index)  # explicit gap detection, no silent skip
    df.index.name = "ts"
    n_missing = df["long_usd_1m"].isna().sum()
    print(f"tail_risk_1m: {len(df)} minutes {df.index[0]} -> {df.index[-1]}, {n_missing} missing minutes ({n_missing/len(df)*100:.1f}%)")
    return df


def compute_liq_net_z_12(tr: pd.DataFrame) -> pd.Series:
    long_usd = tr["long_usd_1m"]
    short_usd = tr["short_usd_1m"]
    total = (long_usd.fillna(0) + short_usd.fillna(0)).where(long_usd.notna() | short_usd.notna())

    long_12 = long_usd.rolling(NET_WINDOW_MIN, min_periods=NET_WINDOW_MIN).sum()
    short_12 = short_usd.rolling(NET_WINDOW_MIN, min_periods=NET_WINDOW_MIN).sum()
    min_periods_2880 = int(np.ceil(TOTAL_TRAILING_MIN * EXISTENCE_RATE_MIN))
    trailing_mean = total.rolling(TOTAL_TRAILING_MIN, min_periods=min_periods_2880).mean()
    eps = trailing_mean * 0.01
    net_z_12 = (long_12 - short_12) / (trailing_mean + eps)
    return net_z_12


def expanding_causal_decile(x: pd.Series) -> pd.Series:
    """out[i] = percentile rank of x[i] among x[0..i] (causal, expanding, NaN-safe). Pandas'
    Expanding.rank() covers this directly (pct=True -> [0,1])."""
    return x.expanding(min_periods=1).rank(pct=True)


def load_price() -> pd.DataFrame:
    df = pd.read_csv(PRICE_CSV, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC") if df["timestamp"].dt.tz is None else df["timestamp"]
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    tr = load_tail_risk()
    net_z_12 = compute_liq_net_z_12(tr)
    decile_pct = expanding_causal_decile(net_z_12.dropna()).reindex(net_z_12.index)
    upper_flag = decile_pct >= (1 - DECILE_THRESHOLD)
    lower_flag = decile_pct <= DECILE_THRESHOLD
    print(f"liq_net_z_12 warmed up: {net_z_12.notna().sum()}/{len(net_z_12)} minutes "
          f"(first valid {net_z_12.dropna().index[0] if net_z_12.notna().any() else 'N/A'})")
    print(f"upper-decile (long-liq-dominant) minutes: {int(upper_flag.sum())}, "
          f"lower-decile (short-liq-dominant) minutes: {int(lower_flag.sum())}")

    price = load_price()
    sig = compute_signals(price, btc_df=None)  # btc_df unused by sweep -- only degrades smt_* signals
    sweep_low = sig["bottom_liquidity_sweep"].fillna(False).to_numpy()
    sweep_high = sig["top_liquidity_sweep"].fillna(False).to_numpy()
    bar_ts = sig["timestamp"].to_numpy()
    print(f"sweep_low fires: {int(sweep_low.sum())}, sweep_high fires: {int(sweep_high.sum())} "
          f"(of {len(sig)} 5m bars, full price series incl. warmup before the valid epoch)")

    # As-of upper/lower decile flags onto each 5m bar's close-minute, causal (<=), then OR against
    # the immediately preceding 5m bar's flag too (COMBINE_TOLERANCE_BARS=1, per §14's "same 5m
    # bar or immediately preceding bar").
    upper_at_bar = upper_flag.reindex(pd.to_datetime(bar_ts), method="ffill", tolerance=pd.Timedelta(minutes=5))
    lower_at_bar = lower_flag.reindex(pd.to_datetime(bar_ts), method="ffill", tolerance=pd.Timedelta(minutes=5))
    upper_at_bar = upper_at_bar.fillna(False).to_numpy()
    lower_at_bar = lower_at_bar.fillna(False).to_numpy()
    upper_or_prev = upper_at_bar | np.r_[[False] * COMBINE_TOLERANCE_BARS, upper_at_bar[:-COMBINE_TOLERANCE_BARS]]
    lower_or_prev = lower_at_bar | np.r_[[False] * COMBINE_TOLERANCE_BARS, lower_at_bar[:-COMBINE_TOLERANCE_BARS]]

    # Restrict to bars fully inside the valid tail_risk epoch (no partial-warmup contamination).
    valid_mask = pd.to_datetime(bar_ts) >= VALID_EPOCH_START

    def count(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sum(a & b & valid_mask))

    print(f"\n=== Bars fully inside valid epoch ({VALID_EPOCH_START} onward): {int(valid_mask.sum())} of {len(sig)} ===")
    print("All 4 combinations (raw counts, ambiguity in §14's bottom/top labeling NOT resolved -- see docstring):")
    print(f"  sweep_low  AND upper-decile(long-liq-dominant) : {count(sweep_low, upper_or_prev)}")
    print(f"  sweep_low  AND lower-decile(short-liq-dominant): {count(sweep_low, lower_or_prev)}   <- §14 literal 'bottom'")
    print(f"  sweep_high AND upper-decile(long-liq-dominant) : {count(sweep_high, upper_or_prev)}   <- §14 literal 'top'")
    print(f"  sweep_high AND lower-decile(short-liq-dominant): {count(sweep_high, lower_or_prev)}")
    print(f"\n§13's own minimum-sample threshold (registered precedent): 15 events per direction before any directional claim.")


if __name__ == "__main__":
    main()
