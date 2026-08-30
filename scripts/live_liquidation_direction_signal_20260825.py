#!/usr/bin/env python3
"""Read-only live directional-tilt reading from the real @forceOrder liquidation stream, 2026-08-25.

User request: not a PnL/economic claim (that axis was tested same-day for the sweep-confirmed
variant -- scripts/s14_sweep_liq_burst_impact.py equivalent, net negative on every horizon, see
docs/experiments/eth_candidate_liquidation_feed_features_cheap_gate_20260817.md section 14) -- just
a frequently-refreshing DIRECTIONAL reading the user reads themselves, same "model indicator, not an
evidence signal" tier as the existing 수급 흐름/고래 포지션 chips (DIRECTIONAL_MODEL_CHIP_KEYS),
not the 7 VAL+OOS-validated evidence-signal chips.

Formula is `liq_net_z_12` verbatim from the section-17-pre-registered design (docs/experiments/
eth_candidate_liquidation_feed_features_cheap_gate_20260817.md section 4 item 3): (trailing-12min
long-liquidation-USD sum minus short-liquidation-USD sum) / (trailing-2day total-liquidation-USD
rolling mean + 1% epsilon), read causally off data/live/tail_risk.duckdb::tail_risk_1m -- the same
table tail_risk_interceptor.py itself writes, read-only, no new collector. Sign convention is the
one that same design doc's section 11 exploratory scan (2026-08-17, 24-day non-decisive scan, its
own "exploratory=true, no promotion/kill decision from this" label) found sign-consistent across two
independent sub-windows: net LONG-liquidation dominance (positive) => contrarian BULLISH tilt
(capitulation-exhaustion logic -- forced long selling exhausting itself), net SHORT-liquidation
dominance (negative) => contrarian BEARISH tilt. That scan is exploratory only, not a validated
edge -- this reading is deliberately NOT badged at the same confidence as the 7 evidence-signal
chips, and the frontend must carry an explicit "방향성 탐색 신호, PnL 미검증" caveat, matching how
live_oi_delta_signal_20260824.py's OI-delta chip was similarly badged "잠정등급" rather than
promoted to evidence-signal status on exploratory-only lift.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 2026-08-31: DB_PATH/TABLE moved into per-coin COIN_CONFIG (see that module's docstring for why
# BTC's tail-risk data lives in a SEPARATE file, not a same-file suffixed table) so this signal can
# be computed for coins other than ETH -- see
# docs/eth_dashboard_multicoin_expansion_design_20260831.md section 6.2.
from coin_config import COIN_CONFIG  # noqa: E402

VALID_SINCE_UTC = "2026-07-18 15:03:00+00"  # forceOrder WS endpoint fix (7fbfd30/984ed8a); pre this
                                              # is fake-always-zero, see design doc section 9.
NET_WIN, NET_MINP = 12, 10       # 80% of 12
TRAIL_WIN, TRAIL_MINP = 2880, 2304  # 80% of 2880 (2 days)
DECILE_WARMUP = 200               # 5m-equivalent-scale warmup before a percentile rank is reported
# 2026-08-25 perf pass: the query below used to fetch the ENTIRE table since VALID_SINCE_UTC (~54.7k
# rows and growing by ~1,440/day forever) every cache refresh (60s), then reindexed+rolled the whole
# span. QUERY_WINDOW_DAYS bounds that -- comfortably covers TRAIL_WIN (2 days) with wide margin, so
# liq_net_z_12/direction/tone_history are unaffected. percentile_rank/intensity (computed below
# against the full fetched history) DO change meaning as a result -- from "vs. all-time since
# VALID_SINCE_UTC" to "vs. the trailing QUERY_WINDOW_DAYS" -- a disclosed, user-confirmed tradeoff
# (neither field is currently rendered anywhere in app.js; grepped to confirm before making this
# change). No-op today (table is ~38 days old, well under 90) -- starts mattering once the table
# passes 90 days old, at which point a trailing-90d percentile is arguably more useful than a
# slowly-staling all-time one anyway.
QUERY_WINDOW_DAYS = 90
# 2026-08-25 user request: uniform 48-slot cap across all model-indicator strips (was 240, sized to
# match oi_delta's ~4h span at this table's 1-min cadence). Now matches MICRO_HISTORY_MAX/
# EVIDENCE_SIGNAL_HISTORY_BARS's slot count instead -- since this table is 1-min-cadence (not
# oi_delta's 5-min oi_lsratio_5m), 48 slots span ~48min, not 4h.
HISTORY_BARS = 48


def _tone(v: float) -> str:
    """Same bullish/bearish/neutral -> good/bad/neutral mapping app.js applies to the latest
    reading (liqDirTone in render()), just run pointwise over the whole history."""
    if v > 0:
        return "good"
    if v < 0:
        return "bad"
    return "neutral"


def compute_liquidation_direction_signal(coin: str = "eth") -> dict:
    """Returns {"warmed_up", "error", "liq_net_z_12", "percentile_rank", "direction",
    "intensity", "n_valid_minutes", "latest_ts_utc", "tone_history"}. tone_history is
    oldest-to-newest (same convention as live_oi_delta_signal_20260824.py's tone_history) so the
    Snapshot tab's activity strip survives a page refresh instead of resetting like the
    client-accumulated toneHistory the other model indicators use -- the full liq_net_z_12 series
    is already computed below, so returning the tail is free. Never raises -- degrades to
    warmed_up=False on any read/compute problem, same contract as the other live_*_signal
    modules in this file's family.

    coin: key into COIN_CONFIG (2026-08-31, BTC added) -- resolves which duckdb file/table this
    reads, since VALID_SINCE_UTC/formula/sign-convention are shared across coins but the
    underlying tail-risk stream lives in a different file for BTC (see coin_config.py)."""
    cfg = COIN_CONFIG[coin]
    db_path, table = cfg["tail_risk_db_path"], cfg["tail_risk_table"]
    if not db_path.exists():
        return _empty("db_missing")
    import duckdb
    df = None
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.0, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                df = con.execute(
                    f"""
                    SELECT ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale
                    FROM {table}
                    WHERE ts >= GREATEST(TIMESTAMPTZ '{VALID_SINCE_UTC}', now() - INTERVAL '{QUERY_WINDOW_DAYS} days')
                    ORDER BY ts
                    """
                ).df()
            finally:
                con.close()
            last_error = None
            break
        except Exception as e:  # noqa: BLE001 -- table-not-yet-created, lock contention, etc.
            last_error = e
    if last_error is not None:
        return _empty(f"db_read_error: {last_error}")
    if df is None or df.empty:
        return _empty("no_rows")

    # duckdb TIMESTAMPTZ comes back in the connection's local session timezone (KST on this
    # server) -- tz_convert before any comparison/formatting, same fix this file's sibling
    # modules (live_oi_delta_signal_20260824.py, live_liquidation_5m_signal_20260825.py) needed.
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df[(df["valid_liq_stream"] == True) & (df["ws_stale"] != True)]  # noqa: E712
    if df.empty:
        return _empty("no_valid_rows")
    df = df.drop_duplicates("ts").set_index("ts")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)
    n_valid = int(df["long_usd_1m"].notna().sum())

    long_ = df["long_usd_1m"].fillna(0.0)
    short_ = df["short_usd_1m"].fillna(0.0)
    net_12 = long_.rolling(NET_WIN, min_periods=NET_MINP).sum() - short_.rolling(NET_WIN, min_periods=NET_MINP).sum()
    total = long_ + short_
    trail_mean = total.rolling(TRAIL_WIN, min_periods=TRAIL_MINP).mean()
    eps = 0.01 * trail_mean
    liq_net_z_12 = net_12 / (trail_mean + eps)

    latest_val = liq_net_z_12.iloc[-1]
    latest_ts = df.index[-1]
    if pd.isna(latest_val):
        return _empty("not_warmed_up_yet", latest_ts_utc=latest_ts.isoformat(), n_valid_minutes=n_valid)

    # hist's population is now bounded by QUERY_WINDOW_DAYS (see that constant's comment) --
    # percentile_rank reads as "vs. the trailing QUERY_WINDOW_DAYS", not "vs. all-time".
    hist = liq_net_z_12.dropna().to_numpy()
    if len(hist) >= DECILE_WARMUP:
        pct = float((hist <= latest_val).mean())
    else:
        pct = None

    if latest_val > 0:
        direction = "bullish"  # net long-liquidation dominance -> contrarian bullish tilt
    elif latest_val < 0:
        direction = "bearish"
    else:
        direction = "neutral"

    if pct is None:
        intensity = "warming_up"
    elif pct >= 0.90 or pct <= 0.10:
        intensity = "strong"
    elif pct >= 0.75 or pct <= 0.25:
        intensity = "moderate"
    else:
        intensity = "weak"

    tone_series = liq_net_z_12.dropna().tail(HISTORY_BARS).apply(_tone)
    return {
        "warmed_up": True,
        "error": None,
        "liq_net_z_12": float(latest_val),
        "percentile_rank": pct,
        "direction": direction,
        "intensity": intensity,
        "n_valid_minutes": n_valid,
        "latest_ts_utc": latest_ts.isoformat(),
        "tone_history": tone_series.tolist(),
    }


def _empty(error: str, **extra) -> dict:
    out = {"warmed_up": False, "error": error, "liq_net_z_12": None, "percentile_rank": None,
           "direction": None, "intensity": None, "n_valid_minutes": 0, "latest_ts_utc": None,
           "tone_history": []}
    out.update(extra)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(compute_liquidation_direction_signal(), indent=2, default=str))
