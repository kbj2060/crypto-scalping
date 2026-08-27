#!/usr/bin/env python3
"""Read-only live computation for the "OI 급변" (OI rapid-change) model indicator: replaces the
호가 불균형(OBI) slot on the Snapshot tab's model-indicator panel, 2026-08-24.

*** RISK/VOLATILITY CONTEXT ONLY -- NOT A DIRECTION SIGNAL. ***
Unlike OBI (which claimed direction and failed 45/45 economic tests, see
docs/experiments/eth_candidate_microstructure_scalp_horizon_screen_20260824.md), this indicator
makes NO direction claim. It answers one question only: "is open interest moving unusually fast
right now" -- a precursor-to-volatility read, same framing as the tail-risk (aftershock) indicator,
just from an independent data source (derivatives positioning, not liquidations).

=== Validation (2026-08-24, ad hoc analysis for this feature, not a full pre-registered line) ===
Computed on data/eth_5m_1year.csv (price) joined with data/TOTAL_ETHUSDT_metrics_2024_2026.csv
(OI, 2024-01..2026-02 overlap, ~224k bars, +5min archive-label correction applied and verified via
sum_open_interest_value ~= sum_open_interest * close residual check, median resid 0.006% at the
correct offset vs 0.096% at zero offset). Signal: 5-min OI delta, z-scored against its own rolling
1-day (288-bar) history -- same convention as delta_z/vol_z elsewhere in this repo. Forward
realized-range lift vs unconditional baseline:
    |z|>=1.0: 15.2% of bars, 1h lift 1.21x, 4h lift 1.13x
    |z|>=2.0: 5.0% of bars,  1h lift 1.35x, 4h lift 1.19x
    |z|>=3.0: 2.3% of bars,  1h lift 1.45x, 4h lift 1.22x
Monotonic with threshold (not noise). Symmetric across direction (OI surge z>=+2: 1.33x/1.16x vs
OI drop z<=-2: 1.37x/1.21x) -- confirms this is a magnitude/risk read, not a direction read, and
matches the literature-grounded hypothesis that BOTH rapid position buildup and rapid deleveraging
precede continued volatility. Redundancy check against the existing EAI indicator (also OI-derived,
formula = |oi_delta|/price_range): corr(|oi_delta_z|, eai_analog) = 0.148 -- NOT a near-duplicate
(compare shadow_absorption_score's disqualifying 96.7% duplicate-of-toxicity finding). This was a
same-day exploratory check, not a pre-registered/OOS-gated research line -- treat the lift numbers
as suggestive, re-validate with a proper walk-forward split before leaning on this harder.

=== Data source ===
data/live/oi_lsratio.duckdb::oi_lsratio_5m (written by oi_lsratio_collector.py, a REST poller
independent of trading_bot.py -- 5-min cadence, live since 2026-08-22). This script reads that
table READ-ONLY and computes the z-score itself; it does NOT touch trading_bot.py and is NOT
consumed by the live bot's actual decisions (unlike the other 5 model indicators, which
microstructure_scanner.py computes and playbook_router etc. actually consume). The dashboard
labels this distinction explicitly so "model indicator" doesn't silently imply "bot-wired" for
this one field -- see MODEL_INDICATOR_MEANING/DETAIL copy in app.js.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "live" / "oi_lsratio.duckdb"
TABLE = "oi_lsratio_5m"
SYMBOL = "ETHUSDT"

ZSCORE_WINDOW = 288  # 1 day of 5m bars, matches this repo's standard z-score window elsewhere
FETCH_ROWS = 500     # buffer beyond ZSCORE_WINDOW so the latest reading always has full warmup
HISTORY_BARS = 48    # 4h strip, matches EVIDENCE_SIGNAL_HISTORY_BARS's convention
WARN_TH = 1.0
DANGER_TH = 2.0       # matches this analysis's confirmed vol-lift regime (1.35x+ at 1h)


def _tone(z: float) -> str:
    az = abs(z)
    return "bad" if az >= DANGER_TH else "warn" if az >= WARN_TH else "good"


def compute_oi_delta_signal(symbol: str = SYMBOL) -> dict:
    """Returns {"warmed_up": bool, "oi_delta_z": float|None, "tone": str, "tone_history": list[str],
    "bars_loaded": int, "latest_ts_utc": str|None}. tone_history is oldest-to-newest (same
    convention as evidence signals' bottom_history/top_history) so the Snapshot tab's activity
    strip survives a page refresh instead of resetting like the other 5 model indicators'
    client-accumulated toneHistory does -- this signal already has the full rolling series
    computed here, so returning the tail is free. Never raises -- any failure (missing db/table,
    insufficient rows) yields warmed_up=False so the caller can render a "warming up" state, same
    contract as load_evidence_signals()."""
    if not DB_PATH.exists():
        return {"warmed_up": False, "error": "db_missing", "oi_delta_z": None, "tone": "neutral",
                "tone_history": [], "bars_loaded": 0, "latest_ts_utc": None}
    import duckdb
    # oi_lsratio_collector.py writes one table per symbol (ETHUSDT -> oi_lsratio_5m, others ->
    # oi_lsratio_5m_<suffix>, same branching as its own self._table) -- this was hardcoded to the
    # ETH table only, so symbol="BTCUSDT"/"SOLUSDT" silently returned no_rows instead of reading
    # oi_lsratio_5m_btc/_sol.
    table = TABLE if symbol.upper() == "ETHUSDT" else f"{TABLE}_{symbol.lower().replace('usdt', '')}"
    # oi_lsratio_collector.py polls every 5 min via short-lived write connections, but DuckDB
    # briefly refuses a NEW read-only connection while that writer's connection is open across
    # its insert -- same lock-contention window ops_watchdog.py::check_duckdb_table_freshness()
    # already retries for tail_risk.duckdb/microstructure.duckdb. Retryable because this is
    # read-only (see feedback_duckdb_single_writer_per_file memory: only write-vs-write conflicts
    # are structural, read-only-vs-write is a brief race absorbed by a few short retries).
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
                    SELECT ts, sum_open_interest
                    FROM {table}
                    WHERE symbol = ? AND sum_open_interest IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    [symbol, FETCH_ROWS],
                ).df()
            finally:
                con.close()
            last_error = None
            break
        except Exception as e:  # noqa: BLE001 -- table-not-yet-created, lock contention, etc.
            last_error = e
    if last_error is not None:
        return {"warmed_up": False, "error": f"db_read_error: {last_error}", "oi_delta_z": None,
                "tone": "neutral", "tone_history": [], "bars_loaded": 0, "latest_ts_utc": None}

    if df.empty:
        return {"warmed_up": False, "error": "no_rows", "oi_delta_z": None, "tone": "neutral",
                "tone_history": [], "bars_loaded": 0, "latest_ts_utc": None}

    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    oi_delta = df["sum_open_interest"].diff()
    oi_delta_z = (oi_delta - oi_delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        oi_delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)

    latest_z = oi_delta_z.iloc[-1]
    warmed_up = bool(pd.notna(latest_z))
    if not warmed_up:
        return {"warmed_up": False, "error": None, "oi_delta_z": None, "tone": "neutral",
                "tone_history": [], "bars_loaded": int(len(df)), "latest_ts_utc": None}

    z = float(latest_z)
    tone_series = oi_delta_z.dropna().tail(HISTORY_BARS).apply(_tone)
    return {
        "warmed_up": True,
        "error": None,
        "oi_delta_z": z,
        "tone": _tone(z),
        "tone_history": tone_series.tolist(),
        "bars_loaded": int(len(df)),
        # duckdb returns TIMESTAMPTZ via the connection's LOCAL session timezone (KST on this
        # server), not UTC -- tz_convert here before formatting so this field's name is honest
        # (2026-08-24 audit: found via live spot-check, the raw value was previously KST-offset
        # despite the "_utc" name; zero live consumers today so no display was ever wrong, but
        # this is the exact duckdb-local-tz pattern this repo has been bitten by before).
        "latest_ts_utc": df["ts"].iloc[-1].tz_convert("UTC").isoformat() if pd.notna(df["ts"].iloc[-1]) else None,
    }
