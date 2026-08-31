#!/usr/bin/env python3
"""Read-only live event-triggered signal: this dashboard's "V자 반등락" specialized-detector chip.

**2026-08-31 upgrade -- 9-trigger multitrigger model replaces the sweep-only v7b model.** Full
history: memory eth_v_rebound_sweep_gated_recall_gap_20260831 (5-stage recall-gap diagnostic that
motivated this, then the label/feature/TabPFN/holdout chain that validated it), docs/homer/
README.md's "V자반등/지지횡보" section. Filename/function name kept unchanged from the v1-v7b
lineage per this project's established convention (dated filenames are not renamed through later
version upgrades -- e.g. build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py stayed named
through its own v1->v4 evolution).

**Why this upgrade happened**: a 90-day recall-gap audit found the OLD sweep-only trigger caught
only ~25.5% of an estimated true population of qualifying V-shaped reversals (sweep is just one of
several conditions that precede a real V자반등; most instances happen without a sweep at all, at
equal or better quality). Widening the candidate pool to 9 triggers (all OR'd, not AND'd) closes
that gap while reusing the EXACT SAME v7b outcome/label formula unchanged (fast_move_atr_mult>=1.5x
within 30min AND giveback_ratio<=0.20 within 60min) -- trigger and label are fully decoupled axes;
only which bars get scored changed, not what "V자반등" means.

The 9 triggers (OR'd; downside=candidate for an upward rebound, upside=mirror):
  1. liquidity_sweep, 2. taker_delta_z_climax, 3. short_term_return_z, 4. orthogonal_combo,
  5. smt_divergence, 6. fib_extension_exhaustion, 7. demarker_extreme, 8. kalman_deviation_meanrev
  -- all 8 reused verbatim via compute_signals() (live_evidence_signal_dashboard_20260823.py),
  the same canonical pre-TabPFN boolean-trigger source this dashboard's own evidence-signal chips
  use (demarker_extreme/kalman_deviation_meanrev were themselves 2026-08-31 Homer candidate-pool
  additions to that function, independently validated and deployed by a concurrent session).
  9. local_extreme -- the only genuinely new trigger, no precondition: bar is the highest/lowest
  in a +-30min (+-6 bar) window. Turned out to be both the single largest AND highest-hit-rate
  trigger of the 9 (22.2% vs the others' 12-20%) in the full-history label build.

Validated (data/labels/eth_5m_v_rebound_multitrigger_20260831/): cheap_gate VAL/OOS AUC
0.8296/0.8119 (single seed) -> 4-seed stability (VAL mean=0.8289 std=0.0007, OOS mean=0.8125
std=0.0004, all 4 seeds beat sweep-only v7b) -> HOLDOUT (classification+economics, ONE-TIME spend):
VAL/OOS/HOLDOUT AUC 0.8292/0.8127/0.8465 (HOLDOUT is the HIGHEST of the three, no degradation;
sweep-only v7b was 0.7342/0.7621/0.7788 for comparison) + trailing-stop economics (SL=4.0/ARM=1.5/
Trail=0.1, selected on VAL+OOS only) VAL+11.97bp/OOS+20.96bp/HOLDOUT+9.28bp (win rate 83.8/86.7/
85.7%, stable across splits) -- sweep-only v7b never passed its own economic gate (0/205 SL/ARM/
Trail combos were simultaneously profitable on VAL+OOS). This is the first time this project's
"V자반등" model family has cleared BOTH a classification bar this strong AND an economic gate.

TabPFN is in-context inference, not a saved/trained model file: every call re-fits on the SAME
FROZEN TRAIN context (data/labels/eth_5m_v_rebound_multitrigger_20260831/tabpfn_train_context_
frozen_multitrigger_v1_20260831.csv, 6,000 rows -- a random subsample of the full 17,961-row TRAIN
split preserving its NATURAL ~32.7% label rate, not rebalanced, so live probabilities stay
calibrated the same way the validated VAL/OOS/HOLDOUT numbers above were measured; sized down from
the full TRAIN set specifically for live latency -- measured 2.88s fit+predict(1 row) on the
server GPU with this context size, research_eth_v_rebound_multitrigger_live_latency_check_
20260831.py, comparable to the sweep-only v7b script's own ~3s/cycle at its 3,783-row context).
Single-seed inference (random_state=20260829, matching this script's own pre-upgrade convention --
the 4-seed ensemble was for validation robustness, not live serving). Live inference always returns
a continuous probability for any new candidate (the excluded middle only applies to TRAINING).

*** DISCRETIONARY READING AID -- NOT WIRED INTO trading_bot.py, NOT AUTOMATED ENTRY/EXIT. ***
Feature/BTC-fetch/RSI-Wilder machinery below is otherwise unchanged from the pre-upgrade version
(Tier0 22 + rsi = 23 features, exact same formulas) -- only trigger detection changed, from sweep-
only to 9-way OR. Values are reused, not reimplemented, from build_eth_5m_v_rebound_multitrigger_
labels_20260831.py::main() (candidate/trigger construction) and this script's own pre-existing
_build_features()/_rsi_wilder() (Tier0 feature computation, untouched).
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
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

TRAIN_CONTEXT_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/tabpfn_train_context_frozen_multitrigger_v1_20260831.csv"
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
BTC_SYMBOL = "BTCUSDT"
FETCH_LIMIT = 1500          # ~5.2 days at 5m -- clears the 864-bar longest indicator warmup with margin
SWEEP_LOOKBACK_BARS = 48    # 4h -- matches the live liquidity_sweep definition elsewhere on this dashboard
HISTORY_BARS = 48           # 4h sparkline strip, matches this dashboard's HISTORY_BARS convention
ACTIVE_WINDOW_MINUTES = 60  # v7b outcome window, unchanged by the trigger-side upgrade
LOCAL_EXTREME_W = 6         # +-30min, matches build_eth_5m_v_rebound_multitrigger_labels_20260831.py

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]

NAMED_TRIGGERS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
                  "orthogonal_combo", "smt_divergence", "fib_extension_exhaustion",
                  "demarker_extreme", "kalman_deviation_meanrev"]

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
            "sweep_ts_utc": None, "price": None, "tone": "neutral", "history": [], "times": [],
            "triggers": None}


def _fetch_klines(symbol: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(FUTURES_KLINES_URL,
                             params={"symbol": symbol, "interval": "5m", "limit": FETCH_LIMIT},
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
    """Unchanged by the 2026-08-31 trigger upgrade -- exact port of
    build_eth_5m_sweep_v_rebound_features_tier0_20260829.py::build_indicator_frame + main()'s
    per-row feature derivation, fed with freshly-fetched klines instead of the static training
    CSV. Produces sweep_level_low/high/atr (used generically for ANY trigger's direction-relative
    features now, not sweep-specific) plus all Tier0 columns."""
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


def _multitrigger_rows(frame: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Any of the 9 triggers OR'd, with generic (non-sweep-specific) direction-relative features
    computed the same way build_eth_5m_v_rebound_multitrigger_features_tier0_20260831.py does for
    training -- is_downside/sweep_penetration_atr/flow_aligned_delta_z are well-defined for any
    candidate regardless of which trigger(s) fired (they describe "how extended is this bar vs.
    the recent 48-bar range", not literally "did a sweep happen"). Returns one row per firing bar
    with a `triggers` column listing which of the 9 fired (comma-joined, sorted)."""
    n = len(frame)
    low, high, close = frame["low"].to_numpy(), frame["high"].to_numpy(), frame["close"].to_numpy()

    down = {name: sig[f"bottom_{name}"].fillna(False).to_numpy() for name in NAMED_TRIGGERS}
    up = {name: sig[f"top_{name}"].fillna(False).to_numpy() for name in NAMED_TRIGGERS}

    W = LOCAL_EXTREME_W
    local_low = np.zeros(n, dtype=bool)
    local_high = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        seg_lo, seg_hi = low[i - W:i + W + 1], high[i - W:i + W + 1]
        if low[i] == seg_lo.min():
            local_low[i] = True
        if high[i] == seg_hi.max():
            local_high[i] = True
    down["local_extreme"] = local_low
    up["local_extreme"] = local_high

    level_low = frame["sweep_level_low"].to_numpy()
    level_high = frame["sweep_level_high"].to_numpy()
    atr = frame["atr"].to_numpy()
    delta_z = frame["delta_z"].to_numpy()

    rows = []
    for is_down, triggers in ((True, down), (False, up)):
        any_fire = np.zeros(n, dtype=bool)
        for arr in triggers.values():
            any_fire |= arr
        for i in np.flatnonzero(any_fire):
            fired = sorted(name for name, arr in triggers.items() if arr[i])
            level = level_low[i] if is_down else level_high[i]
            penetration = (level - low[i]) if is_down else (high[i] - level)
            rows.append({
                "pos": i, "timestamp": frame["timestamp"].iloc[i], "is_downside": int(is_down),
                "sweep_penetration_atr": penetration / atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else np.nan,
                "flow_aligned_delta_z": delta_z[i] if is_down else -delta_z[i],
                "triggers": ",".join(fired),
            })
    return pd.DataFrame(rows)


def _predicted_tone(direction: str | None, call: str | None) -> str:
    """direction is which side the candidate leans (down=support-side, expecting an upward
    rebound; up=resistance-side, expecting a downward reversal); call is the model's rebound-vs-
    continuation read. Unchanged by the 2026-08-31 trigger upgrade -- semantics identical, see
    dashboard MODEL_INDICATOR_DETAIL.v_rebound's "[배지 표시]" paragraph for the full history of
    this tone-mapping decision."""
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
    (matching ISO timestamps), "triggers" (comma-joined names of which of the 9 triggers fired for
    the current event, or None). Never raises."""
    try:
        kl = _fetch_klines(SYMBOL)
        if kl is None or len(kl) < 900:
            return _empty("price_fetch_failed_or_insufficient_history")
        btc_kl = _fetch_klines(BTC_SYMBOL)  # None on failure -- smt_divergence just won't fire this cycle

        frame = _build_features(kl)
        sig = compute_signals(kl, btc_df=btc_kl, funding_df=None)
        candidates = _multitrigger_rows(frame, sig)
        candidates = candidates.merge(frame[["timestamp"] + [c for c in FEATURES if c not in
                                       ("is_downside", "sweep_penetration_atr", "flow_aligned_delta_z")]],
                                       on="timestamp", how="left")
        candidates = candidates.dropna(subset=FEATURES)
        if candidates.empty and frame[FEATURES + ["atr"]].tail(1).isna().any(axis=1).all():
            return _empty("indicators_not_warmed_up")

        now = frame["timestamp"].iloc[-1]
        price = float(frame["close"].iloc[-1])

        proba_by_pos: dict[int, float] = {}
        if not candidates.empty:
            train = _load_train_context()
            from tabpfn import TabPFNClassifier
            clf = TabPFNClassifier(device="cuda", random_state=20260829)
            clf.fit(train[FEATURES], train["label"].to_numpy())
            proba = clf.predict_proba(candidates[FEATURES])[:, 1]
            for pos, p in zip(candidates["pos"], proba):
                proba_by_pos[int(pos)] = float(p)

        def call_of(p: float) -> str:
            return "rebound" if p >= 0.5 else "continuation"

        events = []
        for row in candidates.itertuples():
            p_row = proba_by_pos.get(int(row.pos))
            if p_row is None:
                continue
            direction = "down" if int(row.is_downside) == 1 else "up"
            events.append({"t0": frame["timestamp"].iloc[int(row.pos)], "direction": direction,
                            "call": call_of(p_row), "proba": p_row, "triggers": row.triggers})
        events.sort(key=lambda e: e["t0"])

        # 2026-08-31 fix: pick the OLDEST still-unexpired event ([0]), not the newest ([-1]).
        # With 9 triggers firing far more densely than sweep alone did, a newer candidate is
        # almost always available before an older one's ACTIVE_WINDOW_MINUTES elapses -- always
        # showing the newest meant a call got overwritten after a single 5min bar nearly every
        # cycle (user-reported: "V자반등 shows for 5min then flips back to 미반등"), defeating the
        # documented 60-minute persistence this field's own name (minutes_ago) implies. Taking the
        # oldest active event instead lets each call actually hold the display for its intended
        # window before yielding to whatever is next in the queue.
        hist_bars = frame.tail(HISTORY_BARS)["timestamp"]
        history, times = [], []
        for bar_ts in hist_bars:
            covering = [e for e in events if e["t0"] <= bar_ts <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
            tone = _predicted_tone(covering[0]["direction"], covering[0]["call"]) if covering else "neutral"
            history.append(tone)
            times.append(bar_ts.isoformat())

        current = [e for e in events if e["t0"] <= now <= e["t0"] + pd.Timedelta(minutes=ACTIVE_WINDOW_MINUTES)]
        if not current:
            return {"warmed_up": True, "error": None, "event_active": False, "call": None,
                    "direction": None, "proba_rebound": None, "minutes_ago": None,
                    "sweep_ts_utc": None, "price": price, "tone": "neutral",
                    "history": history, "times": times, "triggers": None}

        shown = current[0]
        minutes_ago = int((now - shown["t0"]).total_seconds() // 60)
        return {
            "warmed_up": True, "error": None, "event_active": True, "call": shown["call"],
            "direction": shown["direction"],
            "proba_rebound": round(shown["proba"], 4), "minutes_ago": minutes_ago,
            "sweep_ts_utc": shown["t0"].isoformat(), "price": price,
            "tone": _predicted_tone(shown["direction"], shown["call"]),
            "history": history, "times": times, "triggers": shown["triggers"],
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
