#!/usr/bin/env python3
"""Read-only live estimate of a Coinglass-style "liquidation heatmap" for the Snapshot tab,
2026-08-24 -- the self-hosted alternative discussed that day: a support/resistance overlay
derived from an estimated liquidation-price density, not a proprietary paid feed.

*** ESTIMATE, NOT MEASUREMENT -- SAME CAVEAT AS COINGLASS ITSELF. ***
No exchange publishes real per-position entry price / leverage / size -- that data does not exist
publicly anywhere, Coinglass included (see docs/experiments/
eth_candidate_liquidation_heatmap_magnet_signal_scoping_20260822.md for the full prior scoping of
this territory). What IS exact: the liquidation-price formula given an entry price and leverage
(a maintenance-margin schedule via Binance's /fapi/v1/leverageBracket). What NOBODY outside the
exchange knows: which price levels actually have open positions, at what leverage, at what size.
This module fills that gap the same way public heatmap tools do -- treat each historical candle's
close as a hypothetical entry price, spread it across a handful of common leverage tiers, weight
by that candle's volume and recency, and drop any hypothetical position whose liquidation price
has ALREADY been crossed by subsequent price action (i.e. it would already be closed by now, so
it is not a live "magnet"). The result is a price-binned density -- the densest bins below current
price read as support, above as resistance, same as reading the visual heatmap by eye.

Unlike the OI-delta indicator (scripts/live_oi_delta_signal_20260824.py), this has deliberately NOT
been back-tested for a return-prediction edge -- the 2026-08-22 scoping's whole point was that
proving a systematic edge here needs deep OOS history Binance's own API cannot provide (a 500-point
cap regardless of resolution). This is a discretionary reading aid only (support/resistance levels
for a human to weigh), the same use the user already made of Coinglass's paid chart -- it must
never be fed into trading_bot.py or any promotion/backtest path as a validated signal.

=== Methodology (deliberately simple, documented so the estimate stays honest) ===
- Source: Binance futures klines (fetched by the caller -- this module is pure/no I/O, mirrors
  compute_signals()'s split in live_evidence_signal_dashboard_20260823.py).
- LEVERAGE_TIERS is a fixed set of common retail perpetual leverage choices, not the full
  notional-tiered bracket schedule (that needs a signed API call this dashboard has no key for);
  MAINTENANCE_MARGIN_RATE is a flat approximation of Binance's lowest-notional-tier rate.
- Each candle's CLOSE stands in for a hypothetical entry price (no better proxy is available);
  weight = candle volume x exp(-age / RECENCY_HALFLIFE_HOURS).
- A hypothetical position is dropped once price has already crossed its liquidation level (checked
  against every subsequent candle's high/low) -- otherwise old, already-triggered levels would
  paint phantom "magnets" that already fired weeks ago.
- Surviving liquidation prices are binned at BIN_WIDTH_PCT of current price; the top N bins by
  density on each side of current price are returned as support/resistance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_HOURS = 24 * 7  # ~7 days at 1h bars -- 2026-08-24 changed from 45d after
                         # eth_dashboard_liquidation_map_sr_backtest_20260824 found 7d resistance
                         # is the only Coinglass-preset lookback (1/7/30/90d) with a fresh-forward
                         # edge over a distance-matched random level (support has no edge at any
                         # preset tested; 45d -- this constant's old value -- had no edge either)
LEVERAGE_TIERS = (10, 20, 25, 50, 75, 100)
MAINTENANCE_MARGIN_RATE = 0.005  # flat approximation of Binance's lowest-bracket USDT-M rate
RECENCY_HALFLIFE_HOURS = 240.0  # 10 days -- a 7-day-old bar (the oldest in-window bar now) still
                                # contributes ~62% of a fresh bar's weight (unchanged: the backtest
                                # validated exactly this halflife at 7d lookback, not a shorter one)
BIN_WIDTH_PCT = 0.0025  # 0.25% of current price per bucket
MAX_LEVELS_PER_SIDE = 6
MIN_LEVEL_SHARE = 0.05  # drop bins under 5% of the strongest bin's weight -- noise floor


def _suffix_min_after(arr: np.ndarray) -> np.ndarray:
    """out[i] = min(arr[i+1:]) for i in [0, len(arr)-2]; length len(arr)-1."""
    suffix = np.minimum.accumulate(arr[::-1])[::-1]
    return suffix[1:]


def _suffix_max_after(arr: np.ndarray) -> np.ndarray:
    """out[i] = max(arr[i+1:]) for i in [0, len(arr)-2]; length len(arr)-1."""
    suffix = np.maximum.accumulate(arr[::-1])[::-1]
    return suffix[1:]


def _empty(reason: str, bars_used: int = 0) -> dict:
    return {
        "warmed_up": False, "error": reason, "current_price": None, "bars_used": bars_used,
        "lookback_hours": 0.0, "support_levels": [], "resistance_levels": [],
    }


def compute_raw_bins(df: pd.DataFrame, current_price: float):
    """The weighting/binning half of compute_liquidation_levels(), split out 2026-08-24 so a
    caller can merge bins from multiple lookback windows into one combined density (see
    scripts/research_eth_liquidation_map_1d7d_formula_merge_20260824.py) instead of only being
    able to compare each window's already-filtered top-N output. bin_width depends only on
    current_price (not on the lookback window), so two windows sharing a current_price land their
    bucket indices on the identical absolute price grid and are directly summable.

    Returns (bins, bin_width, n, age_hours) or None on any shape/degeneracy problem (insufficient
    rows, bad current_price, no surviving bins, all-zero weight) -- never raises."""
    if df is None or len(df) < 20 or not (current_price > 0):
        return None

    d = df.reset_index(drop=True)
    close = d["close"].to_numpy(dtype="float64")
    high = d["high"].to_numpy(dtype="float64")
    low = d["low"].to_numpy(dtype="float64")
    volume = d["volume"].to_numpy(dtype="float64")
    ts = pd.to_datetime(d["timestamp"], utc=True)
    now = ts.iloc[-1]
    age_hours = (now - ts).dt.total_seconds().to_numpy() / 3600.0
    recency_weight = np.exp(-age_hours / RECENCY_HALFLIFE_HOURS)
    base_weight = volume * recency_weight

    n = len(d)
    # future_min_low[i]/future_max_high[i]: extremes strictly AFTER i. An empty suffix (i==n-1)
    # must never count as "already triggered", hence the +inf/-inf pad for the last row.
    future_min_low = np.full(n, np.inf)
    future_max_high = np.full(n, -np.inf)
    if n > 1:
        future_min_low[:-1] = _suffix_min_after(low)
        future_max_high[:-1] = _suffix_max_after(high)

    bin_width = max(current_price * BIN_WIDTH_PCT, 1e-9)
    bins: dict[int, float] = {}

    def add(price_level: np.ndarray, weight: np.ndarray, alive: np.ndarray) -> None:
        idx = np.where(alive & (price_level > 0))[0]
        if not len(idx):
            return
        bucket = np.round(price_level[idx] / bin_width).astype("int64")
        for b, wv in zip(bucket.tolist(), weight[idx].tolist()):
            bins[b] = bins.get(b, 0.0) + wv

    per_tier_weight = base_weight / len(LEVERAGE_TIERS)
    for lev in LEVERAGE_TIERS:
        long_liq = close * (1.0 - 1.0 / lev + MAINTENANCE_MARGIN_RATE)
        short_liq = close * (1.0 + 1.0 / lev - MAINTENANCE_MARGIN_RATE)
        add(long_liq, per_tier_weight, future_min_low > long_liq)
        add(short_liq, per_tier_weight, future_max_high < short_liq)

    if not bins:
        return None
    if not (max(bins.values()) > 0):  # all-zero-volume window (degenerate, but must not raise)
        return None
    return bins, bin_width, n, age_hours


def levels_from_bins(bins: dict, bin_width: float, current_price: float) -> dict:
    """Top-N-per-side extraction/formatting shared by compute_liquidation_levels() and any caller
    merging multiple windows' bins first. Assumes bins is non-empty with max(values) > 0 (both
    guaranteed by compute_raw_bins()'s contract -- callers building bins some other way must
    uphold the same guarantee)."""
    max_weight = max(bins.values())
    levels = [
        {"price": b * bin_width, "weight": w, "weight_pct": w / max_weight}
        for b, w in bins.items()
        if w / max_weight >= MIN_LEVEL_SHARE
    ]
    support = sorted((lv for lv in levels if lv["price"] < current_price), key=lambda lv: -lv["weight"])[:MAX_LEVELS_PER_SIDE]
    resistance = sorted((lv for lv in levels if lv["price"] > current_price), key=lambda lv: -lv["weight"])[:MAX_LEVELS_PER_SIDE]
    support.sort(key=lambda lv: -lv["price"])   # nearest-to-price first
    resistance.sort(key=lambda lv: lv["price"])  # nearest-to-price first

    def fmt(lv: dict) -> dict:
        return {
            "price": round(lv["price"], 4),
            "weight_pct": round(lv["weight_pct"], 4),
            "distance_pct": round((lv["price"] - current_price) / current_price * 100, 3),
        }

    return {"support_levels": [fmt(lv) for lv in support], "resistance_levels": [fmt(lv) for lv in resistance]}


def compute_liquidation_levels(df: pd.DataFrame, current_price: float) -> dict:
    """df: ascending-time klines with columns timestamp (tz-aware)/close/high/low/volume, already
    fetched+cleaned by the caller (same contract as compute_signals() in
    live_evidence_signal_dashboard_20260823.py -- forming bar already dropped). Never raises --
    returns warmed_up=False on any shape problem so the caller can render a "warming up" state.
    Returns {"warmed_up", "error", "current_price", "bars_used", "lookback_hours",
    "support_levels": [{"price","weight_pct","distance_pct"}, ...] (nearest-to-price first),
    "resistance_levels": [...] (nearest-to-price first)}."""
    n_fallback = 0 if df is None else int(len(df))
    raw = compute_raw_bins(df, current_price)
    if raw is None:
        reason = "insufficient_data" if (df is None or len(df) < 20 or not (current_price > 0)) else "no_surviving_levels"
        return _empty(reason, n_fallback)
    bins, bin_width, n, age_hours = raw
    return {
        "warmed_up": True,
        "error": None,
        "current_price": float(current_price),
        "bars_used": int(n),
        "lookback_hours": float(age_hours[0]) if n else 0.0,
        **levels_from_bins(bins, bin_width, current_price),
    }


EVENT_DRIVEN_BREAK_TOLERANCE_PCT = 0.005  # close-based break trigger
EVENT_DRIVEN_DRIFT_TOLERANCE_PCT = 0.10   # 2nd, complementary price trigger -- without this, a side
                                           # anchored at an extreme price stops resetting forever once
                                           # price stops revisiting it (confirmed empirically: support
                                           # froze at the 2022-08 low for ~35,000 of 41,000 backtest
                                           # hours with break-only triggering)
EVENT_DRIVEN_MIN_FLOOR_HOURS = 24
EVENT_DRIVEN_MAX_LOOKBACK_HOURS = 24 * 7
EVENT_DRIVEN_BOOTSTRAP_HOURS = 24 * 7


def compute_event_driven_levels(df: pd.DataFrame, current_price: float) -> dict:
    """Event/price-triggered variant of compute_liquidation_levels(), added 2026-08-24 after the
    user pointed at a real Coinglass heatmap showing exactly this pattern: an old level stays put
    until price actually breaks it or drifts far away, then a new band forms near wherever price
    now trades. Each side (support/resistance) keeps its own reset point -- the last bar where its
    active level set was invalidated, either by a close crossing it by EVENT_DRIVEN_BREAK_TOLERANCE_PCT
    or by drifting more than EVENT_DRIVEN_DRIFT_TOLERANCE_PCT away from current price (the 2nd
    trigger exists only to stop a level freezing forever once price stops revisiting it -- see
    EVENT_DRIVEN_DRIFT_TOLERANCE_PCT's comment). Between resets the level set is frozen; recompute
    reuses compute_raw_bins()/levels_from_bins() unmodified on the window since that side's last
    reset, clamped to [EVENT_DRIVEN_MIN_FLOOR_HOURS, EVENT_DRIVEN_MAX_LOOKBACK_HOURS].

    Backtested in scripts/research_eth_liquidation_map_event_driven_reset_20260824.py over 4.7y of
    hourly data: touch/hold win-rate 60-67% vs a distance-matched random level, the highest of every
    lookback variant tried that session (fixed 1d/7d/30d/90d, and formula-merged 1d+7d). The actual
    price reaction after a touch was small (near-zero to slightly negative vs placebo), so, same as
    compute_liquidation_levels(), this stays a discretionary support/resistance reference only --
    never wire it into trading_bot.py or any promotion/backtest path as a validated signal.

    df must carry enough history for the state machine to settle before "now" (median reset gap is
    44-54h, p90 262h) -- the caller should fetch on the order of a month-plus of hourly bars, not
    just EVENT_DRIVEN_BOOTSTRAP_HOURS worth, or the returned levels risk still reflecting the
    artificial bootstrap seed rather than genuine recent price-driven resets.

    Returns the same shape as compute_liquidation_levels() except lookback_hours is replaced by
    independent support_window_hours/resistance_window_hours (the two sides reset at different
    times, so a single shared lookback no longer means anything). distance_pct on each level is
    recomputed against the final current_price (not the price at that level's own reset), so it
    always reads as "how far from right now", even though the level's price itself was set earlier."""
    d = None if df is None else df.reset_index(drop=True)
    n = 0 if d is None else len(d)
    if n < EVENT_DRIVEN_BOOTSTRAP_HOURS + 2 or not (current_price > 0):
        return _empty("insufficient_data", n)

    close = d["close"].to_numpy(dtype="float64")

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - EVENT_DRIVEN_MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - EVENT_DRIVEN_MIN_FLOOR_HOURS))
        raw = compute_raw_bins(d.iloc[start:i + 1], float(close[i]))
        if raw is None:
            return []
        bins, bin_width, _, _ = raw
        return levels_from_bins(bins, bin_width, float(close[i]))[key]

    support_reset_idx = resistance_reset_idx = 0
    support_levels = regenerate(0, EVENT_DRIVEN_BOOTSTRAP_HOURS, "support_levels")
    resistance_levels = regenerate(0, EVENT_DRIVEN_BOOTSTRAP_HOURS, "resistance_levels")

    for i in range(EVENT_DRIVEN_BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - EVENT_DRIVEN_BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + EVENT_DRIVEN_BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > EVENT_DRIVEN_DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > EVENT_DRIVEN_DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels = regenerate(support_reset_idx, i, "support_levels")
            support_reset_idx = i
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i

    def _redistance(levels: list[dict]) -> list[dict]:
        return [{**lv, "distance_pct": round((lv["price"] - current_price) / current_price * 100, 3)} for lv in levels]

    return {
        "warmed_up": True,
        "error": None,
        "current_price": float(current_price),
        "bars_used": int(n),
        "support_window_hours": float((n - 1) - support_reset_idx),
        "resistance_window_hours": float((n - 1) - resistance_reset_idx),
        "support_levels": _redistance(support_levels),
        "resistance_levels": _redistance(resistance_levels),
    }
