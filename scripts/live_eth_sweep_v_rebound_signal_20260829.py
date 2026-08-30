#!/usr/bin/env python3
"""Read-only live event-triggered signal: when a liquidity_sweep fires (wick pierces the causal
48-bar swing high/low, close reclaims back inside -- the dashboard's own `liquidity_sweep`
evidence-signal definition), what's the probability it forms a clean V자반등 (sustained reversal)
within 60 minutes vs 지지/횡보 (no real reversal attempt)?

Backed by the TabPFN (Tier0 22 features + rsi) model, v7b label (2026-08-30, user-driven redesign
after the v4 binary label's "지지/횡보" bucket was found to visually overlap with genuine V자반등
examples): V자반등(1) = a close within the first 30min reaches 1.5x pre-sweep ATR AND the 60min
window's peak-to-end giveback ratio stays <=0.20 (a real, sustained move, not just a brief touch).
지지/횡보(0) = the first 30min's best CLOSE never even reached 1.0x ATR (genuinely never
attempted a real move) -- everything in between (reached 1.0-1.5x ATR, or reached 1.5x+ but gave
back too much) is EXCLUDED from training entirely as fuzzy/ambiguous, not forced into either
class -- see docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md for the full
v5->v6->v7->v7b design history. Validated: VAL AUC 0.7342+/-0.0008, OOS AUC 0.7621+/-0.0005,
reserved-holdout AUC 0.7788+/-0.0005 -- far above v4's own 0.663/0.667/0.682 on the same 3
chronological periods (this v7b population is a stricter, ~42%-of-events subset by design, so the
two AUCs are not measuring identical tasks, but the jump is large and consistent across all 3
splits and all 4 seeds). TabPFN is in-context inference, not a saved/trained model file: every
call re-fits on the SAME FROZEN TRAIN context
(data/labels/eth_5m_sweep_v_rebound_20260829/tabpfn_train_context_frozen_v7b_20260830.csv,
3,783 rows, ts<2025-09-01 -- exactly the configuration that produced the numbers above, not
re-extended with newer data). Live inference always returns a continuous probability for any new
sweep (the fuzzy middle was only excluded from TRAINING, not from what the model can be asked to
score) -- there is no separate "판단보류" model output, just this one probability.

*** DISCRETIONARY READING AID -- NOT WIRED INTO trading_bot.py, NOT AUTOMATED ENTRY/EXIT. ***
This dashboard's formal pre-registered VAL/OOS/holdout evidence-signal bar, unlike the
liquidation-cascade-gated sibling (live_liquidation_cascade_sweep_trend_signal_20260828.py, n=121
over 41 days) -- this one applies to EVERY liquidity_sweep (14,259 events, 2024-01 onward), not
just cascade-co-occurring ones, so it fires far more often (~14.7/day historically) and rests on
a much larger validated sample. See feedback_dashboard_indicators_ic_bar_not_pnl_bar: exposure
bar is statistical information content, not an economic/cost-gate pass -- this hasn't been tested
as an automated-entry economic strategy, only as a classifier of an already-defined event's outcome.

Feature formulas are reused, not reimplemented, from build_eth_5m_sweep_v_rebound_features_
tier0_20260829.py::build_indicator_frame (same compute_indicators/add_creative_indicators/
add_broad_indicators/add_causal_columns chain), just fed with freshly-fetched klines instead of
the static training CSV. rsi uses Wilder smoothing (verified 2026-08-29 to match the canonical
training_features_*.csv rsi column to ~0.03 mean abs diff, corr 0.997).

TabPFN inference measured at ~3s per fit+predict cycle on the server GPU (2026-08-29) -- cheap
enough to run on every cache refresh (EVIDENCE_SIGNAL_CACHE_SECONDS=60 in dashboard/server.py,
same pattern as every other live_*_signal module here).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators  # noqa: E402
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402

TRAIN_CONTEXT_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/tabpfn_train_context_frozen_v7b_20260830.csv"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
FETCH_LIMIT = 1500          # ~5.2 days at 5m -- clears the 864-bar longest indicator warmup with margin
SWEEP_LOOKBACK_BARS = 48    # 4h -- matches the live liquidity_sweep definition elsewhere on this dashboard
HISTORY_BARS = 48           # 4h sparkline strip, matches this dashboard's HISTORY_BARS convention
ACTIVE_WINDOW_MINUTES = 60  # v7b: the label's own forecast horizon widened 30->60min (fast-window
                             # close-confirm at 30min, giveback held/checked over the full 60min) --
                             # a call is only "live" for the window it actually makes a claim about

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]

_TRAIN_CACHE: pd.DataFrame | None = None
_SWEEP_IMPL = None


def _load_sweep_impl():
    global _SWEEP_IMPL
    if _SWEEP_IMPL is None:
        spec = importlib.util.spec_from_file_location("sweep_impl_live_20260829", SWEEP_IMPL_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SWEEP_IMPL = module
    return _SWEEP_IMPL


def _load_train_context() -> pd.DataFrame:
    global _TRAIN_CACHE
    if _TRAIN_CACHE is None:
        df = pd.read_csv(TRAIN_CONTEXT_CSV)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        _TRAIN_CACHE = df
    return _TRAIN_CACHE


def _empty(error: str) -> dict:
    return {"warmed_up": False, "error": error, "event_active": False, "call": None,
            "direction": None, "proba_rebound": None, "minutes_ago": None,
            "sweep_ts_utc": None, "price": None, "tone": "neutral", "history": [], "times": []}


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
            "taker_buy_base", "tq", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    now_ms = int(time.time() * 1000)
    if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
        df = df.iloc[:-1].reset_index(drop=True)  # drop the still-forming bar
    return df


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _build_features(kl: pd.DataFrame) -> pd.DataFrame:
    """Exact port of build_eth_5m_sweep_v_rebound_features_tier0_20260829.py::build_indicator_frame
    + main()'s per-row feature derivation (lines 119-153 of that script), fed with freshly-fetched
    klines instead of the static training CSV -- same functions, same formulas, same column names."""
    sweep_impl = _load_sweep_impl()
    frame = compute_indicators(kl)
    frame = add_creative_indicators(frame)
    frame = add_broad_indicators(frame)

    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    ret3_mean = ret3.rolling(288, min_periods=288).mean()
    ret3_std = ret3.rolling(288, min_periods=288).std()
    frame["ret3_z"] = (ret3 - ret3_mean) / ret3_std.replace(0.0, np.nan)

    causal = sweep_impl.add_causal_columns(kl[["timestamp", "open", "high", "low", "close"]].copy())
    frame["sweep_level_low"] = causal["sweep_level_low"]
    frame["sweep_level_high"] = causal["sweep_level_high"]
    frame["atr"] = causal["atr"]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    frame["rsi"] = _rsi_wilder(frame["close"])
    return frame


def _sweep_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Same trigger condition as label_events() in build_eth_5m_liquidity_sweep_v_rebound_labels_
    20260829.py: a bar's low/high pierces the causal 48-bar swing level and its close reclaims
    back inside. Computes the same per-event features that script does (is_downside,
    sweep_penetration_atr, flow_aligned_delta_z) for every qualifying row in `frame`."""
    low, high, close = frame["low"], frame["high"], frame["close"]
    level_low, level_high, atr = frame["sweep_level_low"], frame["sweep_level_high"], frame["atr"]

    is_down_sweep = level_low.notna() & (low < level_low) & (close > level_low)
    is_up_sweep = level_high.notna() & (high > level_high) & (close < level_high)
    sweeps = frame.loc[is_down_sweep | is_up_sweep].copy()
    if sweeps.empty:
        return sweeps

    is_down = is_down_sweep.loc[sweeps.index]
    sweeps["is_downside"] = is_down.astype(np.int8)
    level = np.where(is_down, level_low.loc[sweeps.index], level_high.loc[sweeps.index]).astype(float)
    penetration = np.where(
        is_down, level - sweeps["low"].to_numpy(), sweeps["high"].to_numpy() - level
    )
    sweeps["sweep_penetration_atr"] = penetration / sweeps["atr"].to_numpy()
    sweeps["flow_aligned_delta_z"] = np.where(is_down, sweeps["delta_z"], -sweeps["delta_z"])
    return sweeps


def _predicted_tone(direction: str | None, call: str | None) -> str:
    """direction is which side the sweep pierced (down=swept below support, up=swept above
    resistance); call is the model's rebound-vs-continuation read. rebound predicts a genuine,
    sustained bounce back the other way (down+rebound=bullish/good, up+rebound=bearish/bad) --
    this is the only call type the v7b label actually validated as a directional move (giveback
    ratio <=0.20 held over the full 60min). continuation ("미반등") only means the label's fast-
    window never even reached 1.0x ATR -- i.e. no rebound attempt was made -- NOT that price
    decisively kept moving the way the sweep pierced (that stronger claim was never checked by the
    label). 2026-08-31 user request: stop folding continuation into good/bad by direction (it was
    overclaiming a decisive move the label never validated, see dashboard MODEL_INDICATOR_DETAIL.
    v_rebound's "[배지 표시]" paragraph) -- continuation now always maps to its own "flat" tone,
    direction-agnostic, distinct from "neutral" (used only when there's no active sweep at all).
    2026-08-29 user request (superseded by the above for continuation, still applies to rebound):
    read short/long/no-signal like the dashboard's other directional model-indicators (see
    DIRECTIONAL_MODEL_CHIP_KEYS in app.js) instead of a flat warn/neutral tone."""
    if call == "rebound" and direction in ("up", "down"):
        return "good" if direction == "down" else "bad"
    if call == "continuation" and direction in ("up", "down"):
        return "flat"
    return "neutral"


def compute_eth_sweep_v_rebound_signal() -> dict:
    """Returns {"warmed_up", "error", "event_active", "call" ("rebound"|"continuation"|None),
    "direction" ("up"|"down"|None), "proba_rebound" (0-1 or None), "minutes_ago",
    "sweep_ts_utc", "price", "tone" ("good"|"bad"|"flat"|"neutral", direction x call resolved via
    _predicted_tone), "history" (oldest-to-newest tone strings, HISTORY_BARS long), "times"
    (matching ISO timestamps)}. Never raises."""
    try:
        kl = _fetch_klines()
        if kl is None or len(kl) < 900:
            return _empty("price_fetch_failed_or_insufficient_history")

        frame = _build_features(kl)
        sweeps = _sweep_rows(frame).dropna(subset=FEATURES)
        if sweeps.empty and frame[FEATURES + ["atr"]].tail(1).isna().any(axis=1).all():
            return _empty("indicators_not_warmed_up")

        now = frame["timestamp"].iloc[-1]
        price = float(frame["close"].iloc[-1])

        proba_by_ts: dict[pd.Timestamp, float] = {}
        if not sweeps.empty:
            train = _load_train_context()
            from tabpfn import TabPFNClassifier
            clf = TabPFNClassifier(device="cuda", random_state=20260829)
            clf.fit(train[FEATURES], train["label"].to_numpy())
            proba = clf.predict_proba(sweeps[FEATURES])[:, 1]
            for ts, p in zip(sweeps["timestamp"], proba):
                proba_by_ts[ts] = float(p)

        def call_of(p: float) -> str:
            return "rebound" if p >= 0.5 else "continuation"

        # per-event (timestamp, swept side, call) so both the history loop and the current read can
        # resolve a real predicted price direction, not just "was a sweep active" -- see
        # _predicted_tone() below.
        sweep_events = []
        if not sweeps.empty:
            for row in sweeps.itertuples():
                p_row = proba_by_ts.get(row.timestamp)
                if p_row is None:
                    continue
                direction = "down" if int(row.is_downside) == 1 else "up"
                sweep_events.append({"t0": row.timestamp, "direction": direction, "call": call_of(p_row)})
        sweep_events.sort(key=lambda e: e["t0"])

        # sparkline: for each of the last HISTORY_BARS bars, was a sweep active (within
        # ACTIVE_WINDOW_MINUTES of it) at that time, and which way did it predict?
        hist_bars = frame.tail(HISTORY_BARS)["timestamp"]
        history, times = [], []
        for bar_ts in hist_bars:
            covering = [e for e in sweep_events if e["t0"] <= bar_ts <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
            tone = _predicted_tone(covering[-1]["direction"], covering[-1]["call"]) if covering else "neutral"
            history.append(tone)
            times.append(bar_ts.isoformat())

        current = [e for e in sweep_events if e["t0"] <= now <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
        if not current:
            return {"warmed_up": True, "error": None, "event_active": False, "call": None,
                    "direction": None, "proba_rebound": None, "minutes_ago": None,
                    "sweep_ts_utc": None, "price": price, "tone": "neutral",
                    "history": history, "times": times}

        latest = current[-1]
        p = proba_by_ts[latest["t0"]]
        minutes_ago = int((now - latest["t0"]).total_seconds() // 60)
        return {
            "warmed_up": True, "error": None, "event_active": True, "call": latest["call"],
            "direction": latest["direction"],
            "proba_rebound": round(p, 4), "minutes_ago": minutes_ago,
            "sweep_ts_utc": latest["t0"].isoformat(), "price": price,
            "tone": _predicted_tone(latest["direction"], latest["call"]),
            "history": history, "times": times,
        }
    except Exception as e:  # noqa: BLE001 -- never raise, same contract as sibling live_* modules
        return _empty(f"compute_error: {e}")


if __name__ == "__main__":
    import json
    result = compute_eth_sweep_v_rebound_signal()
    history = result.pop("history", [])
    times = result.pop("times", [])
    print(json.dumps(result, indent=2, default=str))
    print(f"history: {len(history)} bars")
