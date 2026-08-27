#!/usr/bin/env python3
"""Read-only clock-aligned-bar liquidation $ accumulator for the Snapshot tab's liquidation gauge,
2026-08-25 (bar length widened 5min -> 15min, same day, see note below).

tail_risk_interceptor.py (the live bot's real @forceOrder liquidation stream) only tracks a
TRAILING 1-minute sum (long_usd_1m/short_usd_1m in state.tail_risk, recomputed every ~10s). Summing
successive ticks of that trailing value on the client would NOT equal a true total -- each tick's
1m window overlaps the others depending on sampling cadence, so it would over- or under-count.

Two designs were considered: a SLIDING window (always "sum of the trailing N completed minutes")
vs a DISCRETE, clock-aligned BAR that fills from 0 at each boundary and resets at the next one --
same accumulate-then-reset semantics as an OHLCV candle bar, which is what the user asked for ("매
5분봉마다 누적", after a first "sliding window" version). This implements the bar version: sums rows
from data/live/tail_risk.duckdb::tail_risk_1m (one discrete, non-overlapping row inserted per
completed minute by TailRiskInterceptor._db_insert()) whose timestamp falls within the CURRENT bar,
so bars_used/elapsed naturally grows from 1 toward BAR_MINUTES over the life of the bar rather than
always reading a full trailing window. Read-only; does not touch trading_bot.py or
tail_risk_interceptor.py, and is not consumed by the live bot's actual decisions (same
"dashboard-only, not bot-wired" category as the OI-delta indicator -- see
live_oi_delta_signal_20260824.py for the twin of this pattern).

2026-08-25 follow-up (same day): BAR_MINUTES widened from 5 to 15 per user request ("누적 15m 으로
하는건 어때?"), raised after a real ~$176K long-liquidation burst (13:15-13:23 UTC) spanned two
5-min bars and had already fully reset out of the gauge by the time it was checked ~2 minutes later
-- a 5-min bar's boundary comes around often enough that a burst landing late in one can vanish
almost immediately. Widening to 15min doesn't eliminate that edge (a burst seconds before ANY
boundary still resets away instantly), but cuts how often it happens roughly 3x, and -- as a side
effect -- also directly addresses an earlier "gauge looks frozen at 안정" question from the same
session: a live duckdb check that day found liquidation $ nonzero in only ~16-21% of individual
minutes (24h sample), so a 5-min bar was all-zero much more often than a 15-min bar will be. Endpoint path
(/api/liquidation-5m-signal), this module's filename, compute_liquidation_5m_signal()'s name, and
the long_usd_5m/short_usd_5m payload field names were all deliberately NOT renamed -- narrow,
contained change (BAR_MINUTES + FETCH_ROWS + user-facing Korean labels only), consistent with this
repo's "defer wide-blast-radius cleanup" convention rather than a rename sweep across this file +
server.py + app.js for a naming-purity concern nobody asked to fix.

2026-08-27 follow-up: BAR_MINUTES widened again, 15 -> 30, per user request while recreating the
Snapshot tab's long/short liquidation volume gauge (the frontend consumer of this endpoint had been
removed at some point after 2026-08-25's feature-stripping rounds; this backend script kept running
unconsumed the whole time -- server.py's /api/liquidation-5m-signal route and its 60s cache never
stopped working). Same "don't rename, just widen the window" convention as the first bump above.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "live" / "tail_risk.duckdb"
TABLE = "tail_risk_1m"
BAR_MINUTES = 30  # widened from 15 2026-08-27 (previously 5->15 on 2026-08-25), see docstring
FETCH_ROWS = 60  # 2x BAR_MINUTES of buffer -- filtering to the current bar happens in Python
                  # below (against already tz_convert'd timestamps), not in the SQL WHERE clause,
                  # to sidestep the duckdb-session-local-timezone quirk this repo has been bitten
                  # by more than once rather than trust a parameterized UTC comparison against it.


def _bar_start(now_utc: datetime) -> datetime:
    floored_minute = (now_utc.minute // BAR_MINUTES) * BAR_MINUTES
    return now_utc.replace(minute=floored_minute, second=0, microsecond=0)


def compute_liquidation_5m_signal() -> dict:
    """Returns {"warmed_up", "error", "long_usd_5m", "short_usd_5m", "bars_used",
    "bar_start_utc", "bar_elapsed_sec", "latest_ts_utc"}. bars_used grows from ~0 to BAR_MINUTES
    (15) and bar_elapsed_sec from ~0 to BAR_MINUTES*60 (900) over the life of the current bar, then
    both reset at the next boundary -- reported honestly (never padded to look like a full bar),
    same "never raises, degrade to warmed_up=False" contract as compute_oi_delta_signal()."""
    if not DB_PATH.exists():
        return {"warmed_up": False, "error": "db_missing", "long_usd_5m": None, "short_usd_5m": None,
                "bars_used": 0, "bar_start_utc": None, "bar_elapsed_sec": None, "latest_ts_utc": None}
    import duckdb
    # Same brief read-vs-write lock contention window ops_watchdog.py already retries for this
    # exact db file -- see live_oi_delta_signal_20260824.py's identical retry comment/pattern.
    df = None
    last_error: Exception | None = None
    for attempt, delay in enumerate((0.0, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                df = con.execute(
                    f"""
                    SELECT ts, long_usd_1m, short_usd_1m
                    FROM {TABLE}
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    [FETCH_ROWS],
                ).df()
            finally:
                con.close()
            last_error = None
            break
        except Exception as e:  # noqa: BLE001 -- table-not-yet-created, lock contention, etc.
            last_error = e
    if last_error is not None:
        return {"warmed_up": False, "error": f"db_read_error: {last_error}", "long_usd_5m": None,
                "short_usd_5m": None, "bars_used": 0, "bar_start_utc": None, "bar_elapsed_sec": None,
                "latest_ts_utc": None}
    if df.empty:
        return {"warmed_up": False, "error": "no_rows", "long_usd_5m": None, "short_usd_5m": None,
                "bars_used": 0, "bar_start_utc": None, "bar_elapsed_sec": None, "latest_ts_utc": None}

    # duckdb TIMESTAMPTZ comes back in the connection's LOCAL session timezone (KST on this
    # server), not UTC -- tz_convert here before comparing/formatting, same fix
    # live_oi_delta_signal_20260824.py already needed for this exact class of bug.
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    now_utc = datetime.now(timezone.utc)
    bar_start = _bar_start(now_utc)
    in_bar = df[df["ts"] >= bar_start]

    latest_ts = df["ts"].iloc[0]
    if in_bar.empty:
        # Between the boundary tick and the interceptor's own next insert -- genuinely zero rows
        # in the new bar yet, not an error.
        return {
            "warmed_up": True, "error": None, "long_usd_5m": 0.0, "short_usd_5m": 0.0,
            "bars_used": 0, "bar_start_utc": bar_start.isoformat(),
            "bar_elapsed_sec": (now_utc - bar_start).total_seconds(),
            "latest_ts_utc": latest_ts.isoformat() if pd.notna(latest_ts) else None,
        }

    long_sum = float(in_bar["long_usd_1m"].fillna(0.0).sum())
    short_sum = float(in_bar["short_usd_1m"].fillna(0.0).sum())
    return {
        "warmed_up": True,
        "error": None,
        "long_usd_5m": long_sum,
        "short_usd_5m": short_sum,
        "bars_used": int(len(in_bar)),
        "bar_start_utc": bar_start.isoformat(),
        "bar_elapsed_sec": (now_utc - bar_start).total_seconds(),
        "latest_ts_utc": latest_ts.isoformat() if pd.notna(latest_ts) else None,
    }
