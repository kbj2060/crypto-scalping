#!/usr/bin/env python3
"""Read-only live event-triggered signal: after a genuine ETH liquidation cascade (one that
actually breached a technical swing level, not just a pure liquidation-$ volume burst), does the
candle shape + whale flow reaction look more like a sweep-and-reverse or a trend-continuation?
2026-08-28, replaces the "OI 급변" model-indicator slot at user request (they don't watch that
chip) -- and built as an event-triggered SIGNAL (active/call/minutes_ago), not a continuous
z-score indicator, since the underlying rule only has something to say right after a cascade, not
at every moment -- same shape as live_liquidation_sweep_combo_signal_20260825.py's bottom_active/
top_active/bars_ago, not live_oi_delta_signal_20260824.py's always-on z-score.

*** DISCRETIONARY READING AID -- NOT EVIDENCE-SIGNAL TIER, NOT WIRED INTO trading_bot.py. ***
Same confidence tier as liq_direction/liq_pressure/liq_cascade (dev/holdout-validated pilot
finding, not this dashboard's formal pre-registered TRAIN/VAL/OOS + cost-gate evidence-signal
bar). Full methodology: docs/experiments/
eth_liquidation_cascade_sweep_vs_trend_pilot_design_20260828.md (see that doc's "📌 확정 규칙"
box, kept up to date as this rule gets re-validated on more data).

=== Rule (chronological dev70%/holdout30% split over N=121 genuine-breach cascades, 41 days) ===
- CONTINUATION call: cascade candle's wick/body ratio < 0.5 (body-dominant candle) AND
  direction-relative whale net-inflow (nif_whale, sign-aligned to the cascade direction) <= 0
  (whale flow confirms/doesn't fight the cascade) -- holdout precision 88.9% (n=9 predictions).
- SWEEP call: wick/body ratio > 2.0 (tail-dominant candle) alone -- holdout precision 75.0%
  (n=8 predictions). Adding other axes did not improve this side (see design doc §13-15).
- Neither condition -> "no_read" (the common case even right after a genuine-breach cascade).
- "Genuine breach" itself matters a lot: ~73-81% of raw liquidation-cascade triggers (z-score+$
  threshold) never actually touch a 4h swing level at all -- those are excluded entirely rather
  than shown as a call, matching the pilot's own tested population exactly (see design doc §12.2).

Absolute call volume is low by design (order of a few per week) -- this is a rare-but-higher-
confidence read, not a constant stream.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH_TAIL_RISK = ROOT / "data" / "live" / "tail_risk.duckdb"
DB_PATH_MICRO = ROOT / "data" / "live" / "microstructure.duckdb"
VALID_SINCE_UTC = "2026-07-18 15:03:00+00"  # tail_risk_1m valid epoch, matches sibling live_* scripts

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
FETCH_LIMIT = 1500  # ~5.2 days at 5m, matches live_liquidation_sweep_combo_signal_20260825.py's budget

SWEEP_LOOKBACK_BARS = 48        # 4h at 5m -- swept_level lookback, matches the pilot + the existing
                                 # liquidity_sweep formula in live_evidence_signal_dashboard_20260823.py
BUF_PCT = 0.005                 # unused here (no forward labeling needed live) -- kept for reference
SCAN_WINDOW_MINUTES = 360       # 6h -- how far back to look for the most recent hawkes onset
TAIL_RISK_LOOKBACK_MINUTES = SCAN_WINDOW_MINUTES + 60  # +1h buffer for the 30min hawkes rolling warmup
ACTIVE_WINDOW_MINUTES = 120     # 2h -- how long a found cascade's call stays "active" for display
FEATURE_WINDOW_MINUTES = 15     # matches the pilot's [t0, t0+15min] feature window
HISTORY_BARS = 48               # 4h strip at 5m, matches this dashboard's HISTORY_BARS convention

WICK_SWEEP_THRESHOLD = 2.0
WICK_CONTINUATION_THRESHOLD = 0.5


def _empty(error: str) -> dict:
    return {"warmed_up": False, "error": error, "event_active": False, "call": None,
            "direction": None, "wick_body_ratio": None, "nif_whale_rel": None,
            "minutes_ago": None, "cascade_ts_utc": None, "price": None,
            "tone": "neutral", "history": [], "times": []}


def _fetch_klines() -> pd.DataFrame | None:
    try:
        resp = requests.get(FUTURES_KLINES_URL,
                             params={"symbol": SYMBOL, "interval": "5m", "limit": FETCH_LIMIT},
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
        df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar
    return df


def _read_duckdb_window(db_path: Path, table: str, cols_sql: str, lookback_minutes: int,
                         extra_where: str = "") -> pd.DataFrame | None:
    if not db_path.exists():
        return None
    import duckdb
    df = None
    for attempt, delay in enumerate((0.0, 0.4, 0.8, 1.6)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                df = con.execute(
                    f"""
                    SELECT {cols_sql} FROM {table}
                    WHERE ts >= now() - INTERVAL '{lookback_minutes} minutes' {extra_where}
                    ORDER BY ts
                    """
                ).df()
            finally:
                con.close()
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return None
    df["ts"] = df["ts"].dt.tz_convert("UTC")
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _read_tail_risk_1m() -> pd.DataFrame | None:
    df = _read_duckdb_window(
        DB_PATH_TAIL_RISK, "tail_risk_1m", "ts, long_usd_1m, short_usd_1m, valid_liq_stream, ws_stale",
        TAIL_RISK_LOOKBACK_MINUTES, f"AND ts >= TIMESTAMPTZ '{VALID_SINCE_UTC}'")
    if df is None:
        return None
    df = df[(df["valid_liq_stream"] == True) & (df["ws_stale"] != True)]  # noqa: E712
    return df.reset_index(drop=True) if len(df) else None


def _read_microstructure() -> pd.DataFrame | None:
    return _read_duckdb_window(DB_PATH_MICRO, "microstructure_1m", "ts, nif_whale",
                                TAIL_RISK_LOOKBACK_MINUTES + FEATURE_WINDOW_MINUTES)


def _replay_hawkes_onsets(tail_risk: pd.DataFrame) -> list[dict]:
    """Causally replays the real TailRiskInterceptor._update_hawkes_state (not a reimplementation)
    by monkeypatching time.time() to each historical minute -- identical technique to the pilot's
    research_eth_liquidation_cascade_sweep_vs_trend_pilot_20260828.py. Returns onset events
    (z_peak>=3.5 & peak_usd>=$10k, new activation) in chronological order."""
    import tail_risk_interceptor as tri
    interceptor = tri.TailRiskInterceptor(symbol="ethusdt")
    onsets = []
    for row in tail_risk.itertuples():
        ts_epoch = row.ts.timestamp()
        prev_active, prev_crisis = interceptor._hawkes_active, interceptor._crisis_type
        with patch.object(tri.time, "time", return_value=ts_epoch):
            interceptor._update_hawkes_state(float(row.long_usd_1m), float(row.short_usd_1m))
        if interceptor._hawkes_active and ((not prev_active) or (interceptor._crisis_type != prev_crisis)):
            onsets.append({"t0": row.ts, "crisis_type": interceptor._crisis_type})
        interceptor._history_long.append(float(row.long_usd_1m))
        interceptor._history_short.append(float(row.short_usd_1m))
        interceptor._recalculate_stats()
    return onsets


def _evaluate_onset(t0: pd.Timestamp, crisis_type: str, kl: pd.DataFrame,
                     micro: pd.DataFrame | None) -> dict | None:
    """genuine-breach filter + wick_body_ratio + nif_whale_rel + call for one onset. Returns None
    if this onset never actually broke a 4h swing level (the ~73-81% majority case, see module
    docstring) or there isn't enough kline history yet."""
    idx = int(kl["timestamp"].searchsorted(t0, side="right")) - 1
    if idx < SWEEP_LOOKBACK_BARS or idx < 0 or idx >= len(kl):
        return None
    direction = "down" if crisis_type == "LONG_CRISIS" else "up"
    pre = kl.iloc[idx - SWEEP_LOOKBACK_BARS: idx]
    swept_level = pre["low"].min() if direction == "down" else pre["high"].max()
    cascade_extreme = kl["low"].iloc[idx] if direction == "down" else kl["high"].iloc[idx]
    genuine_breach = (cascade_extreme < swept_level) if direction == "down" else (cascade_extreme > swept_level)
    if not genuine_breach:
        return None

    bar = kl.iloc[idx]
    body = max(abs(bar["close"] - bar["open"]), 1e-9)
    wick_in_direction = ((min(bar["open"], bar["close"]) - bar["low"]) if direction == "down"
                         else (bar["high"] - max(bar["open"], bar["close"])))
    wick_body_ratio = float(wick_in_direction / body)

    nif_whale_rel = None
    if micro is not None:
        win = micro[(micro["ts"] > t0) & (micro["ts"] <= t0 + pd.Timedelta(minutes=FEATURE_WINDOW_MINUTES))]
        if len(win):
            m = win["nif_whale"].mean()
            if pd.notna(m):
                nif_whale_rel = float(m if direction == "down" else -m)

    if wick_body_ratio < WICK_CONTINUATION_THRESHOLD and nif_whale_rel is not None and nif_whale_rel <= 0:
        call = "continuation"
    elif wick_body_ratio > WICK_SWEEP_THRESHOLD:
        call = "sweep"
    else:
        call = "no_read"

    return {"t0": t0, "direction": direction, "wick_body_ratio": round(wick_body_ratio, 3),
            "nif_whale_rel": round(nif_whale_rel, 4) if nif_whale_rel is not None else None, "call": call}


def _predicted_tone(direction: str | None, call: str | None) -> str:
    """Combines the cascade's own direction (up=short-liq crisis, price spiked up; down=long-liq
    crisis, price dropped) with the sweep-vs-continuation call into one predicted-price-direction
    tone: continuation keeps going the way the cascade already moved (up+continuation=bullish,
    down+continuation=bearish); sweep predicts a reversal (up+sweep=bearish, down+sweep=bullish).
    no_read or a missing direction stays neutral. 2026-08-29 user request: this chip should read
    short/long/no-signal like the dashboard's other directional model-indicators (see
    DIRECTIONAL_MODEL_CHIP_KEYS in app.js) -- replaces the old all-"warn" _CALL_TONE mapping, whose
    "call alone isn't inherently good or bad" reasoning was true for call in isolation but not once
    paired with this same direction field, which was already computed and available."""
    if call == "continuation" and direction in ("up", "down"):
        return "good" if direction == "up" else "bad"
    if call == "sweep" and direction in ("up", "down"):
        return "good" if direction == "down" else "bad"
    return "neutral"


def compute_liquidation_cascade_sweep_trend_signal() -> dict:
    """Returns {"warmed_up", "error", "event_active", "call" ("continuation"|"sweep"|"no_read"|None),
    "direction" ("up"|"down"|None), "wick_body_ratio", "nif_whale_rel", "minutes_ago",
    "cascade_ts_utc", "price", "tone", "history" (oldest-to-newest tone strings, HISTORY_BARS long),
    "times" (matching ms epoch timestamps)}. Never raises -- degrades to warmed_up=False."""
    try:
        kl = _fetch_klines()
        if kl is None or kl.empty:
            return _empty("price_fetch_failed")
        tail_risk = _read_tail_risk_1m()
        if tail_risk is None or tail_risk.empty:
            return _empty("tail_risk_read_failed")
        micro = _read_microstructure()  # None is tolerated -- nif_whale_rel just stays None per-onset

        onsets = _replay_hawkes_onsets(tail_risk)
        evaluated = [r for o in onsets if (r := _evaluate_onset(o["t0"], o["crisis_type"], kl, micro))]

        now = kl["timestamp"].iloc[-1]
        price = float(kl["close"].iloc[-1])

        # sparkline: for each of the last HISTORY_BARS klines bars, was a genuine-breach event
        # active (within ACTIVE_WINDOW_MINUTES of it) at that bar's time, and what was its call?
        hist_bars = kl.tail(HISTORY_BARS).reset_index(drop=True)
        history, times = [], []
        for bar_ts in hist_bars["timestamp"]:
            covering = [e for e in evaluated if e["t0"] <= bar_ts <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
            tone = _predicted_tone(covering[-1]["direction"], covering[-1]["call"]) if covering else "neutral"
            history.append(tone)
            times.append(bar_ts.isoformat())  # ISO string, matching this dashboard's other tone-strip time arrays

        current = [e for e in evaluated if e["t0"] <= now <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
        if not current:
            return {"warmed_up": True, "error": None, "event_active": False, "call": None,
                    "direction": None, "wick_body_ratio": None, "nif_whale_rel": None,
                    "minutes_ago": None, "cascade_ts_utc": None, "price": price,
                    "tone": "neutral", "history": history, "times": times}

        latest = current[-1]
        minutes_ago = int((now - latest["t0"]).total_seconds() // 60)
        return {
            "warmed_up": True, "error": None, "event_active": True, "call": latest["call"],
            "direction": latest["direction"], "wick_body_ratio": latest["wick_body_ratio"],
            "nif_whale_rel": latest["nif_whale_rel"], "minutes_ago": minutes_ago,
            "cascade_ts_utc": latest["t0"].isoformat(), "price": price,
            "tone": _predicted_tone(latest["direction"], latest["call"]), "history": history, "times": times,
        }
    except Exception as e:  # noqa: BLE001 -- never raise, same contract as sibling live_* modules
        return _empty(f"compute_error: {e}")


if __name__ == "__main__":
    import json
    result = compute_liquidation_cascade_sweep_trend_signal()
    history = result.pop("history", [])
    times = result.pop("times", [])
    print(json.dumps(result, indent=2, default=str))
    print(f"history: {len(history)} bars")
