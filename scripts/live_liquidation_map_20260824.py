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
- Surviving liquidation prices are binned at BIN_WIDTH_PCT of current price; within
  MAX_LEVEL_DISTANCE_PCT of current price, the top N bins by density on each side are returned as
  support/resistance (2026-08-25: added the distance cap so a single far-but-heavy bin can't take a
  top-N slot and blow out the displayed range -- see MAX_LEVEL_DISTANCE_PCT's own comment).
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
BIN_WIDTH_PCT = 0.001  # 0.1% of current price per bucket -- 2026-08-25 tightened from 0.0025 at
                       # user request for finer heatmap resolution (still well above the ~0.02%
                       # raw-point floor set by 168 hourly bars x 6 leverage tiers x 2 sides)
MAX_LEVELS_PER_SIDE = 6
MIN_LEVEL_SHARE = 0.05  # drop bins under 5% of the strongest bin's weight -- noise floor
MAX_LEVEL_DISTANCE_PCT = 0.05  # 2026-08-25 user request: top-N-per-side was picked by weight alone,
                                # no distance limit -- a single far-but-heavy bin (e.g. a real +10.3%
                                # resistance bin the same day) could take a top-6 slot and blow out
                                # the displayed range. Bins beyond this are excluded from the
                                # candidate pool entirely (not filtered after selection), so a closer,
                                # merely-adequate bin can take the slot instead.


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


def _prepare_common(df: pd.DataFrame, current_price: float) -> dict | None:
    """The weighting/survival-filter arithmetic shared by every compute_raw_bins() call regardless
    of which entry-price array is used -- split out 2026-08-26 so compute_spliced_levels()/
    compute_spliced_heatmap_history() can run their mid-priced and close-priced passes over the
    SAME window without recomputing volume/recency weighting or the future-high/low survival
    filter twice (only the leverage-tier liquidation-price + binning step in _bins_from_common()
    actually differs between the two passes). Returns None on the same shape/degeneracy problems
    compute_raw_bins() always guarded against."""
    if df is None or len(df) < 20 or not (current_price > 0):
        return None
    d = df.reset_index(drop=True)
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

    return {
        "n": n, "age_hours": age_hours, "base_weight": base_weight,
        "future_min_low": future_min_low, "future_max_high": future_max_high,
        "bin_width": max(current_price * BIN_WIDTH_PCT, 1e-9),
        "close": d["close"].to_numpy(dtype="float64"),
    }


def _bins_from_common(common: dict, entry_price: np.ndarray | None) -> dict | None:
    """The entry-price-dependent half of compute_raw_bins(): leverage-tier liquidation prices,
    binned and weighted using _prepare_common()'s shared arrays. entry_price=None reproduces v1's
    original behavior (each candle's own close as its hypothetical entry); an explicit array lets
    a caller price the long/short sides differently (see compute_spliced_levels())."""
    entry_px = common["close"] if entry_price is None else np.asarray(entry_price, dtype="float64")
    bin_width = common["bin_width"]
    future_min_low = common["future_min_low"]
    future_max_high = common["future_max_high"]
    per_tier_weight = common["base_weight"] / len(LEVERAGE_TIERS)
    bins: dict[int, float] = {}

    def add(price_level: np.ndarray, weight: np.ndarray, alive: np.ndarray) -> None:
        # 2026-08-26: tried an np.unique/np.bincount vectorization here (same pattern as
        # live_liquidation_map_v2_20260825.py's _accumulate()) expecting a speedup, but measured
        # -30% (slower) at the actual 24-bar production window -- np.unique's sort overhead beats
        # a plain Python loop only at much larger n than this module ever sees. Reverted; the real
        # win for the spliced two-pass call is _prepare_common() below, not this inner loop.
        idx = np.where(alive & (price_level > 0))[0]
        if not len(idx):
            return
        bucket = np.round(price_level[idx] / bin_width).astype("int64")
        for b, wv in zip(bucket.tolist(), weight[idx].tolist()):
            bins[b] = bins.get(b, 0.0) + wv

    for lev in LEVERAGE_TIERS:
        long_liq = entry_px * (1.0 - 1.0 / lev + MAINTENANCE_MARGIN_RATE)
        short_liq = entry_px * (1.0 + 1.0 / lev - MAINTENANCE_MARGIN_RATE)
        add(long_liq, per_tier_weight, future_min_low > long_liq)
        add(short_liq, per_tier_weight, future_max_high < short_liq)

    if not bins or not (max(bins.values()) > 0):  # all-zero-volume window (degenerate, no raise)
        return None
    return bins


def compute_raw_bins(df: pd.DataFrame, current_price: float, entry_price: np.ndarray | None = None):
    """The weighting/binning half of compute_liquidation_levels(), split out 2026-08-24 so a
    caller can merge bins from multiple lookback windows into one combined density (see
    scripts/research_eth_liquidation_map_1d7d_formula_merge_20260824.py) instead of only being
    able to compare each window's already-filtered top-N output. bin_width depends only on
    current_price (not on the lookback window), so two windows sharing a current_price land their
    bucket indices on the identical absolute price grid and are directly summable.

    entry_price: optional override for the per-candle hypothetical entry price (defaults to that
    candle's own close). 2026-08-26: added for compute_spliced_levels(), which prices the long
    side off (high+low)/2 while the short side still prices off close -- passing None reproduces
    the original close-only behavior exactly (see _bins_from_common()).

    Returns (bins, bin_width, n, age_hours) or None on any shape/degeneracy problem (insufficient
    rows, bad current_price, no surviving bins, all-zero weight) -- never raises."""
    common = _prepare_common(df, current_price)
    if common is None:
        return None
    bins = _bins_from_common(common, entry_price)
    if bins is None:
        return None
    return bins, common["bin_width"], common["n"], common["age_hours"]


def levels_from_bins(bins: dict, bin_width: float, current_price: float) -> dict:
    """Top-N-per-side extraction/formatting shared by compute_liquidation_levels() and any caller
    merging multiple windows' bins first. Assumes bins is non-empty with max(values) > 0 (both
    guaranteed by compute_raw_bins()'s contract -- callers building bins some other way must
    uphold the same guarantee). Candidates beyond MAX_LEVEL_DISTANCE_PCT of current_price are
    excluded before the top-N-by-weight cut, so support_levels/resistance_levels can each return
    fewer than MAX_LEVELS_PER_SIDE entries if not enough surviving bins fall within that band --
    reported honestly (never backfilled with a farther bin just to hit the count)."""
    max_weight = max(bins.values())
    levels = [
        {"price": b * bin_width, "weight": w, "weight_pct": w / max_weight}
        for b, w in bins.items()
        if w / max_weight >= MIN_LEVEL_SHARE
        and abs(b * bin_width - current_price) / current_price <= MAX_LEVEL_DISTANCE_PCT
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
    "resistance_levels": [...] (nearest-to-price first),
    "bin_width", "heatmap_bins": [{"price","weight_pct"}, ...] (every surviving bin both sides,
    not just the top MAX_LEVELS_PER_SIDE filtered into support_levels/resistance_levels -- same
    "full density for rendering" field compute_event_driven_levels() returns, 2026-08-25 added here
    too so a caller can drive the same heatmap-band rendering off either function)}."""
    n_fallback = 0 if df is None else int(len(df))
    raw = compute_raw_bins(df, current_price)
    if raw is None:
        reason = "insufficient_data" if (df is None or len(df) < 20 or not (current_price > 0)) else "no_surviving_levels"
        return _empty(reason, n_fallback)
    bins, bin_width, n, age_hours = raw
    max_weight = max(bins.values())
    heatmap_bins = sorted(
        ({"price": round(b * bin_width, 4), "weight_pct": round(w / max_weight, 4)} for b, w in bins.items()),
        key=lambda x: x["price"],
    )
    return {
        "warmed_up": True,
        "error": None,
        "current_price": float(current_price),
        "bars_used": int(n),
        "lookback_hours": float(age_hours[0]) if n else 0.0,
        "bin_width": round(bin_width, 4),
        "heatmap_bins": heatmap_bins,
        **levels_from_bins(bins, bin_width, current_price),
    }


def compute_spliced_levels(df: pd.DataFrame, current_price: float) -> dict:
    """Support/resistance levels spliced from two independent pricing passes, 2026-08-26 --
    support_levels come entirely from a (high+low)/2-priced pass, resistance_levels entirely from
    the original close-priced pass (identical to compute_liquidation_levels()'s resistance side).
    Backtested in scripts/research_eth_liquidation_map_spliced_hybrid_multifold_20260826.py and
    scripts/research_eth_liquidation_map_final_recommendation_seed_robust_20260826.py (20-seed-
    averaged over 4 chronological folds spanning bear/choppy/bull regimes): support pairWR +0.12
    over the pure-close baseline (0.590 vs 0.469 pooled, far outside the ~0.03-0.04 seed-noise
    floor), resistance statistically unchanged (0.524 vs 0.538 pooled, well within noise -- at or
    above the close-only baseline in 3 of 4 folds).

    The two passes share _prepare_common()'s weighting/survival-filter arrays (computed once) but
    each gets its OWN levels_from_bins() call, so MIN_LEVEL_SHARE normalizes against each side's
    own max bin weight, never a value influenced by the other side's bins. This is deliberate: an
    earlier variant that merged both sides' bins into one dict before a single levels_from_bins()
    call let a taller (high+low)/2-driven support peak silently raise the shared normalization
    floor and drop resistance bins that would otherwise have cleared it -- diluting both sides at
    once (see eth_liquidation_map_hybrid_price_multifold_20260826 memory). Splicing two fully
    independently-filtered passes has no such coupling.

    Same return shape as compute_liquidation_levels() (drop-in replacement for that function's
    caller). heatmap_bins is spliced the same way: the support half from the mid pass, the
    resistance half from the close pass, each normalized against its own side's max weight."""
    n_fallback = 0 if df is None else int(len(df))
    common = _prepare_common(df, current_price)
    if common is None:
        reason = "insufficient_data" if (df is None or len(df) < 20 or not (current_price > 0)) else "no_surviving_levels"
        return _empty(reason, n_fallback)

    d = df.reset_index(drop=True)
    mid_price = (d["high"].to_numpy(dtype="float64") + d["low"].to_numpy(dtype="float64")) / 2.0
    bins_mid = _bins_from_common(common, mid_price)
    bins_close = _bins_from_common(common, None)
    if bins_mid is None or bins_close is None:
        return _empty("no_surviving_levels", n_fallback)

    bin_width = common["bin_width"]
    support = levels_from_bins(bins_mid, bin_width, current_price)["support_levels"]
    resistance = levels_from_bins(bins_close, bin_width, current_price)["resistance_levels"]

    def side_heatmap(bins: dict, below: bool) -> list[dict]:
        side_bins = {b: w for b, w in bins.items() if (b * bin_width < current_price) == below}
        max_w = max(side_bins.values()) if side_bins else 0.0
        if not (max_w > 0):
            return []
        return [{"price": round(b * bin_width, 4), "weight_pct": round(w / max_w, 4)} for b, w in side_bins.items()]

    heatmap_bins = sorted(side_heatmap(bins_mid, True) + side_heatmap(bins_close, False), key=lambda x: x["price"])

    return {
        "warmed_up": True,
        "error": None,
        "current_price": float(current_price),
        "bars_used": int(common["n"]),
        "lookback_hours": float(common["age_hours"][0]) if common["n"] else 0.0,
        "bin_width": round(bin_width, 4),
        "heatmap_bins": heatmap_bins,
        "support_levels": support,
        "resistance_levels": resistance,
    }


HEATMAP_HISTORY_DISPLAY_HOURS = 6  # 2026-08-25 user request: matches the chart's own visible-candle
                                    # window (narrowed from ~8h to 4h at the same request, then
                                    # widened 4h->6h same day -- "4시간은 너무 작다" -- see
                                    # dashboard/live/app.js SNAPSHOT_CHART_MAX_CANDLES), so every
                                    # column the chart can actually show has its own real snapshot.


def compute_heatmap_history(df: pd.DataFrame, current_price: float, lookback_hours: int,
                             display_hours: int = HEATMAP_HISTORY_DISPLAY_HOURS) -> list[dict]:
    """Time series of causal heatmap snapshots for the chart's density-history overlay, 2026-08-25 --
    replaces the old single "now" snapshot + client-side sweep-darkening hack (drawDensitySeg in
    app.js), which could only ever go from a bin's live color to permanently dark the instant price
    first swept it, never back -- see eth_liquidation_map_coinglass_visual_logic_replication_20260825
    memory: the user noticed a real Coinglass screenshot shows a swept price re-lighting later as
    fresh volume re-accumulates there, which a single frozen "now" snapshot cannot represent no
    matter how the frontend post-processes it.

    Returns oldest-to-newest [{"ts_utc", "bins": [{"price","weight_pct"}, ...]}, ...], one entry per
    hourly kline boundary covering roughly the last display_hours (plus a little slack so the
    earliest DISPLAYED candle -- not just the earliest snapshot -- still has a snapshot at or before
    it). Each entry is exactly what compute_liquidation_levels()'s heatmap_bins would have been if
    computed AT that historical moment: compute_raw_bins() over ONLY the trailing lookback_hours of
    data ending at that boundary, so a bin already dead by then has zero weight, and a bin that
    hadn't yet accumulated volume by then is likewise absent -- no lookahead past each snapshot's own
    boundary. current_price is passed unchanged to every snapshot (compute_raw_bins() only uses it
    for bin_width, never for aliveness -- see that function's own contract) so every snapshot shares
    one fixed price grid; using each snapshot's own historical close instead would make bin_width
    drift slightly hour to hour and the same price row would no longer line up across columns.

    weight_pct in every snapshot is normalized against the SAME global max (the single strongest bin
    across the WHOLE returned history, not each snapshot's own max) -- so relative brightness is
    comparable across time too (a genuinely quiet hour reads as dim, not artificially rescaled to
    look as loud as the strongest hour). Returns [] on any shape problem (never raises), same
    "degrade gracefully" contract as the other compute_* functions in this module.

    df must cover at least lookback_hours + display_hours (+ a couple hours of slack) of hourly bars
    so the earliest displayed snapshot still gets a full, untruncated lookback_hours window."""
    if df is None or not (current_price > 0):
        return []
    d = df.reset_index(drop=True)
    n = len(d)
    if n < lookback_hours + 2:
        return []
    start_i = max(lookback_hours - 1, n - display_hours - 3)
    raws = []
    for i in range(start_i, n):
        window = d.iloc[max(0, i - lookback_hours + 1): i + 1]
        raw = compute_raw_bins(window, current_price)
        if raw is not None:
            bins, bin_width, _, _ = raw
            raws.append((d["timestamp"].iloc[i], bins, bin_width))
    if not raws:
        return []
    global_max = max(max(bins.values()) for _, bins, _ in raws)
    if not (global_max > 0):
        return []
    return [
        {
            "ts_utc": ts.isoformat(),
            "bins": [
                {"price": round(b * bin_width, 4), "weight_pct": round(w / global_max, 4)}
                for b, w in bins.items()
            ],
        }
        for ts, bins, bin_width in raws
    ]


def compute_spliced_heatmap_history(df: pd.DataFrame, current_price: float, lookback_hours: int,
                                    display_hours: int = HEATMAP_HISTORY_DISPLAY_HOURS) -> list[dict]:
    """Time-varying density history for compute_spliced_levels(), 2026-08-26 -- same per-boundary
    causal snapshot contract as compute_heatmap_history() (see that function's docstring: one entry
    per hourly kline boundary, no lookahead past each snapshot's own boundary, current_price fixed
    across all snapshots so the price grid stays aligned), but each snapshot's support half comes
    from a (high+low)/2-priced pass and its resistance half from the close-priced pass -- matching
    compute_spliced_levels()'s split so the density-history overlay stays visually consistent with
    the live support/resistance lines drawn on top of it.

    weight_pct is normalized per SIDE against that side's own global max across the whole returned
    history (support bins against the largest support bin seen in any mid-priced snapshot,
    resistance bins against the largest resistance bin seen in any close-priced snapshot) --
    deliberately not a single combined max, for the same cross-side-contamination reason documented
    in compute_spliced_levels(). Returns [] on any shape problem (never raises)."""
    if df is None or not (current_price > 0):
        return []
    d = df.reset_index(drop=True)
    n = len(d)
    if n < lookback_hours + 2:
        return []
    mid_price_full = (d["high"].to_numpy(dtype="float64") + d["low"].to_numpy(dtype="float64")) / 2.0

    start_i = max(lookback_hours - 1, n - display_hours - 3)
    raws = []
    for i in range(start_i, n):
        start = max(0, i - lookback_hours + 1)
        window = d.iloc[start: i + 1]
        common = _prepare_common(window, current_price)
        if common is None:
            continue
        bins_mid = _bins_from_common(common, mid_price_full[start: i + 1])
        bins_close = _bins_from_common(common, None)
        if bins_mid is None and bins_close is None:
            continue
        bin_width = common["bin_width"]
        support_bins = {b: w for b, w in (bins_mid or {}).items() if b * bin_width < current_price}
        resistance_bins = {b: w for b, w in (bins_close or {}).items() if b * bin_width > current_price}
        raws.append((d["timestamp"].iloc[i], support_bins, resistance_bins, bin_width))
    if not raws:
        return []
    global_max_support = max((max(s.values()) for _, s, _, _ in raws if s), default=0.0)
    global_max_resistance = max((max(r.values()) for _, _, r, _ in raws if r), default=0.0)
    if not (global_max_support > 0 or global_max_resistance > 0):
        return []

    def fmt_side(bins: dict, bin_width: float, global_max: float) -> list[dict]:
        if not (global_max > 0):
            return []
        return [{"price": round(b * bin_width, 4), "weight_pct": round(w / global_max, 4)} for b, w in bins.items()]

    return [
        {
            "ts_utc": ts.isoformat(),
            "bins": sorted(fmt_side(support_bins, bin_width, global_max_support) +
                          fmt_side(resistance_bins, bin_width, global_max_resistance),
                          key=lambda x: x["price"]),
        }
        for ts, support_bins, resistance_bins, bin_width in raws
    ]


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
    always reads as "how far from right now", even though the level's price itself was set earlier.

    Also returns heatmap_bins: the FULL surviving density (every bin compute_raw_bins() produced
    for whichever side's window is currently active), not just the top MAX_LEVELS_PER_SIDE filtered
    into support_levels/resistance_levels -- 2026-08-24, added after the user compared a real
    Coinglass screenshot showing continuous shading across many price rows, not a handful of
    discrete lines; support_levels/resistance_levels already throw most of that shape away. Purely
    a rendering aid (a wider/richer picture of the same estimate) -- the backtested win-rate claim
    still refers only to the filtered top-N levels, not to every bin surviving here."""
    d = None if df is None else df.reset_index(drop=True)
    n = 0 if d is None else len(d)
    if n < EVENT_DRIVEN_BOOTSTRAP_HOURS + 2 or not (current_price > 0):
        return _empty("insufficient_data", n)

    close = d["close"].to_numpy(dtype="float64")

    def regenerate(reset_idx: int, i: int, key: str):
        start = max(reset_idx, i - EVENT_DRIVEN_MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - EVENT_DRIVEN_MIN_FLOOR_HOURS))
        raw = compute_raw_bins(d.iloc[start:i + 1], float(close[i]))
        if raw is None:
            return [], None
        bins, bin_width, _, _ = raw
        return levels_from_bins(bins, bin_width, float(close[i]))[key], (bins, bin_width)

    support_reset_idx = resistance_reset_idx = 0
    support_levels, support_raw = regenerate(0, EVENT_DRIVEN_BOOTSTRAP_HOURS, "support_levels")
    resistance_levels, resistance_raw = regenerate(0, EVENT_DRIVEN_BOOTSTRAP_HOURS, "resistance_levels")

    for i in range(EVENT_DRIVEN_BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - EVENT_DRIVEN_BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + EVENT_DRIVEN_BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > EVENT_DRIVEN_DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > EVENT_DRIVEN_DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels, support_raw = regenerate(support_reset_idx, i, "support_levels")
            support_reset_idx = i
        if broke_resistance or drift_resistance:
            resistance_levels, resistance_raw = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i

    def _redistance(levels: list[dict], side: str) -> list[dict]:
        # The break check above only fires past EVENT_DRIVEN_BREAK_TOLERANCE_PCT (0.5%), so a level
        # can sit up to that far on the "wrong" side of the true final current_price without the
        # state machine having reset it yet -- e.g. a resistance level 0.2% below current_price,
        # still technically un-broken by its own 0.5% margin. Internally that lag is intentional
        # (avoids reset thrash right at the boundary); displaying it unchanged is not -- a level
        # labeled "resistance" sitting under the price reads as a bug, not a nuance, so it is
        # dropped here rather than shown with a distance sign that contradicts its own label.
        out = []
        for lv in levels:
            dist = round((lv["price"] - current_price) / current_price * 100, 3)
            if side == "support" and dist >= 0:
                continue
            if side == "resistance" and dist <= 0:
                continue
            out.append({**lv, "distance_pct": dist})
        return out

    def _heatmap_side(raw, side: str) -> list[dict]:
        if raw is None:
            return []
        bins, bin_width = raw
        max_weight = max(bins.values()) if bins else 0.0
        out = []
        for b, w in bins.items():
            price = b * bin_width
            if side == "support" and price >= current_price:
                continue
            if side == "resistance" and price <= current_price:
                continue
            out.append({"price": round(price, 4), "weight_pct": round(w / max_weight, 4) if max_weight > 0 else 0.0})
        return out

    heatmap_bins = sorted(_heatmap_side(support_raw, "support") + _heatmap_side(resistance_raw, "resistance"),
                           key=lambda b: b["price"])

    return {
        "warmed_up": True,
        "error": None,
        "current_price": float(current_price),
        "bars_used": int(n),
        "support_window_hours": float((n - 1) - support_reset_idx),
        "resistance_window_hours": float((n - 1) - resistance_reset_idx),
        "support_levels": _redistance(support_levels, "support"),
        "resistance_levels": _redistance(resistance_levels, "resistance"),
        "bin_width": round(current_price * BIN_WIDTH_PCT, 4),
        "heatmap_bins": heatmap_bins,
    }
