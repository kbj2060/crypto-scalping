#!/usr/bin/env python3
"""Read-only live reading of the §14 sweep x liquidation-burst COMBO condition (docs/experiments/
eth_candidate_liquidation_feed_features_cheap_gate_20260817.md section 14, corrected pairing), for
the dashboard's "청산확인 스윕" model-indicator chip, 2026-08-25.

User explicitly asked for a §13/§14-DATA-based chip (distinct from live_liquidation_direction_
signal_20260825.py, which is really the plainer §12-item1-level liq_net_z_12 alone -- not §13's
crowding-conditional version, which is currently uncomputable, see below, and not §14's sweep-
paired version, which this module IS). §13 is NOT included here -- its crowding_flag needs
top_pos_ls_ratio, and data/live/oi_lsratio.duckdb only has ~3 days of that column (poller started
2026-08-22) against a pre-registered 15-day-per-direction minimum; still physically impossible as
of this write, unrelated to today's date.

Definitions reused verbatim, no new formulas (matches the exact minimal-early-peek script this
session already validated, scripts/s14_sweep_liq_burst_impact.py, itself sourced from the
pre-registration cited above):
  - sweep leg: scripts/live_evidence_signal_dashboard_20260823.py's liquidity_sweep formula
    (SWEEP_LOOKBACK=48 5m bars), fetched fresh here (that module deliberately never touches
    tail_risk.duckdb, so this is a separate small fetch, not an import).
  - liquidation leg: liq_net_z_12 off data/live/tail_risk.duckdb::tail_risk_1m (same formula as
    live_liquidation_direction_signal_20260825.py), expanding-trailing DECILE flag.
  - combo (corrected pairing): bottom = sweep_low & (top-decile burst same-or-prev 5m bar), top =
    sweep_high & (bottom-decile burst same-or-prev 5m bar).

THIS IS NOT A VALIDATED TRADING SIGNAL -- the same-day minimal peek (scripts/
s14_sweep_liq_burst_impact.py, 37-day sample) found the bottom-side (primary, per pre-
registration) net NEGATIVE at every one of 4 horizons after standard costs (-9.19bp to -5.00bp,
|t| up to 3.67), and no consistent impact amplification over an unconditional-sweep control. The
frontend caveat text must carry that result explicitly, not just "unvalidated" in the abstract.
"""
from __future__ import annotations

import time
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "live" / "tail_risk.duckdb"
TABLE = "tail_risk_1m"
VALID_SINCE_UTC = "2026-07-18 15:03:00+00"

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
FETCH_LIMIT = 1500  # ~5.2 days -- matches live_evidence_signal_dashboard_20260823.py's own budget

SWEEP_LOOKBACK = 48
NET_WIN, NET_MINP = 12, 10
TRAIL_WIN, TRAIL_MINP = 2880, 2304
DECILE_WARMUP = 200
SUSTAIN_BARS = 4  # matches live_evidence_signal_dashboard_20260823.py's 15-min sustain convention


def _empty(error: str) -> dict:
    return {"warmed_up": False, "error": error, "bottom_active": None, "top_active": None,
            "bottom_bars_ago": None, "top_bars_ago": None, "latest_bar_utc": None, "price": None}


def _fetch_eth_klines() -> pd.DataFrame | None:
    try:
        resp = requests.get(FUTURES_KLINES_URL,
                             params={"symbol": SYMBOL, "interval": INTERVAL, "limit": FETCH_LIMIT},
                             timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except Exception:
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "tb", "tq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def _read_tail_risk_1m() -> pd.DataFrame | None:
    if not DB_PATH.exists():
        return None
    import duckdb
    df = None
    for attempt, delay in enumerate((0.0, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                df = con.execute(
                    f"""
                    SELECT ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale
                    FROM {TABLE}
                    WHERE ts >= TIMESTAMPTZ '{VALID_SINCE_UTC}'
                    ORDER BY ts
                    """
                ).df()
            finally:
                con.close()
            break
        except Exception:
            df = None
    return df


def compute_sweep_liq_combo_signal() -> dict:
    """Returns {"warmed_up", "error", "bottom_active", "top_active", "bottom_bars_ago",
    "top_bars_ago", "latest_bar_utc", "price"}. Never raises."""
    px = _fetch_eth_klines()
    if px is None or px.empty:
        return _empty("price_fetch_failed")
    tr = _read_tail_risk_1m()
    if tr is None or tr.empty:
        return _empty("tail_risk_read_failed")

    tr["ts"] = tr["ts"].dt.tz_convert("UTC")
    tr = tr[(tr["valid_liq_stream"] == True) & (tr["ws_stale"] != True)]  # noqa: E712
    if tr.empty:
        return _empty("no_valid_tail_risk_rows")
    tr = tr.drop_duplicates("ts").set_index("ts")
    full_idx = pd.date_range(tr.index.min(), tr.index.max(), freq="1min", tz="UTC")
    tr = tr.reindex(full_idx)

    long_ = tr["long_usd_1m"].fillna(0.0)
    short_ = tr["short_usd_1m"].fillna(0.0)
    net_12 = long_.rolling(NET_WIN, min_periods=NET_MINP).sum() - short_.rolling(NET_WIN, min_periods=NET_MINP).sum()
    total = long_ + short_
    trail_mean = total.rolling(TRAIL_WIN, min_periods=TRAIL_MINP).mean()
    eps = 0.01 * trail_mean
    liq_net_z_12 = (net_12 / (trail_mean + eps)).rename("liq_net_z_12")

    liq_df = liq_net_z_12.reset_index().rename(columns={"index": "ts"})
    merged = pd.merge_asof(px[["timestamp"]], liq_df, left_on="timestamp", right_on="ts", direction="backward")
    px = px.copy()
    px["liq_net_z_12"] = merged["liq_net_z_12"].to_numpy()

    vals = px["liq_net_z_12"].to_numpy()
    n = len(px)
    top_decile = np.zeros(n, dtype=bool)
    bottom_decile = np.zeros(n, dtype=bool)
    valid_hist: list[float] = []
    for i in range(n):
        v = vals[i]
        if not np.isnan(v):
            valid_hist.append(v)
        if len(valid_hist) >= DECILE_WARMUP and not np.isnan(v):
            arr = np.array(valid_hist)
            pct = (arr <= v).mean()
            top_decile[i] = pct >= 0.90
            bottom_decile[i] = pct <= 0.10

    high, low, close = px["high"], px["low"], px["close"]
    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    sweep_low = ((low < swing_low_prior) & (close > swing_low_prior)).to_numpy()
    sweep_high = ((high > swing_high_prior) & (close < swing_high_prior)).to_numpy()

    top_decile_or_prev = top_decile | np.roll(top_decile, 1)
    top_decile_or_prev[0] = top_decile[0]
    bottom_decile_or_prev = bottom_decile | np.roll(bottom_decile, 1)
    bottom_decile_or_prev[0] = bottom_decile[0]

    bottom_combo = sweep_low & top_decile_or_prev
    top_combo = sweep_high & bottom_decile_or_prev

    bottom_active_series = pd.Series(bottom_combo).rolling(SUSTAIN_BARS, min_periods=1).max().astype(bool)
    top_active_series = pd.Series(top_combo).rolling(SUSTAIN_BARS, min_periods=1).max().astype(bool)

    def _bars_ago(flags: np.ndarray) -> int | None:
        idx = np.flatnonzero(flags)
        if len(idx) == 0:
            return None
        return int(n - 1 - idx[-1])

    return {
        "warmed_up": True,
        "error": None,
        "bottom_active": bool(bottom_active_series.iloc[-1]),
        "top_active": bool(top_active_series.iloc[-1]),
        "bottom_bars_ago": _bars_ago(bottom_combo),
        "top_bars_ago": _bars_ago(top_combo),
        "latest_bar_utc": px["timestamp"].iloc[-1].isoformat(),
        "price": float(px["close"].iloc[-1]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(compute_sweep_liq_combo_signal(), indent=2, default=str))
