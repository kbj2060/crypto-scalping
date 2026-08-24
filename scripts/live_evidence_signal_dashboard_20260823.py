#!/usr/bin/env python3
"""Read-only, on-demand/live terminal dashboard for a human discretionary trader: shows which of
the 7 validated ETH "reversal evidence" signals (2026-08-14 research lineage + one 2026-08-24
addition) are firing on the most recently CLOSED ETH/USDT 5-minute bar, right now.

2026-08-24 (same day, later still): trimmed from 11 down to 6 "core" signals + 1 "experimental"
signal, on user request, for glanceability -- all 11 were genuinely real (non-noise) lift, this
was never a correctness cull. Dropped: bollinger_pctb_extreme (2.34x, redundant oscillator-family
overlap with orthogonal_combo), cvd_divergence (2.05x) and vwap_extreme (1.98x) (weakest of the
11), sweep_flow_combo (3.42x) and smt_flow_combo (3.45x) (both share their taker-delta leg with
taker_delta_z_climax already kept standalone -- this repo's own "confirmation-stacking dilutes,
doesn't strengthen" finding, replicated 3x independently, argues for keeping the simpler
standalone parent over the compound combo; both were also top-side-disabled by design, i.e. half
the chip was always dead). The 6 kept are deliberately maximally-orthogonal information families
(price-position+orderflow hybrid / price geometry / volume+candle / cross-asset / pure momentum /
pure orderflow), each still bidirectional. Added fib_extension_exhaustion (bottom 3.27x/top 2.32x,
docs/experiments/eth_fibonacci_harmonic_geometric_evidence_20260824.md) as a 7th, EXPERIMENTAL
entry: a genuinely different information family (Fibonacci extension-zone geometry, ~9-11% bar
overlap with liquidity_sweep) but on a materially thinner sample (n=183-193 vs hundreds-to-
thousands for the other 6) and steeper VAL->OOS lift decay -- flagged as such in its own
description/detail text, not held to the same confidence as the 6 core signals.

*** NOT A TRADING ALGORITHM. INFORMATIONAL / PROBABILITY-SHIFT CONTEXT ONLY. ***
See docs/experiments/eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md: four
independent attempts to wire the original top-6 subset directly into automated trading decisions
(a hard forced-exit veto, an exit_head model feature, a position-sizing feature, and a standalone
entry-trigger confluence formula tested across vote thresholds K=1..6) ALL lost to always_long/
always_short benchmarks. These signals are real, reproducible evidence of elevated reversal
probability (rank-stable across an independent out-of-window replication, see
docs/experiments/eth_evidence_signal_ranking_stability_mar_jul_2026_20260814.md), but they must
be treated as context for a human's own discretionary judgment -- never as an automated buy/sell
trigger. Even a true positive historically still saw ~0.5-0.85% further adverse price movement
before the real pivot -- no signal here calls the exact bottom/top.
History: cvd_divergence/vwap_extreme/bollinger_pctb_extreme/sweep_flow_combo/smt_flow_combo were
added 2026-08-24 (from the same 2026-08-14 22-signal scorecard plus the ICT-2022 component study,
docs/experiments/eth_ict2022_ob_smt_po3_component_evidence_20260824.md) and then trimmed the same
day -- see the "trimmed from 11" note at the top of this docstring for why and what survives.

=== Formula provenance (fidelity to the original research; nothing here is re-derived) ===
All formulas, thresholds, and window lengths below are transcribed VERBATIM from the scripts that
actually produced the validated numbers -- not approximated from the prose docs:
  - fast_k / slow_k / p_fast / p_slow (used by orthogonal_combo):
    scripts/backtest_eth_slowk_williamsr_persistence_confluence_20260814.py::compute_indicators
  - delta_z / vol_z / lower_wick_ratio / upper_wick_ratio:
    scripts/analyze_eth_creative_reversal_evidence_signals_20260814.py::add_creative_indicators
  - liquidity_sweep (48-bar swing sweep):
    scripts/analyze_eth_broad_evidence_signal_sweep_20260814.py
  - short_term_return_z (3-bar/15m return z-score):
    scripts/analyze_eth_deep_evidence_signal_sweep_round2_20260814.py
  - smt_divergence: scripts/analyze_eth_ict2022_ob_smt_po3_component_evidence_20260824.py::add_smt
  - fib_extension_exhaustion (experimental): scripts/analyze_eth_fibonacci_harmonic_geometric_
    evidence_20260824.py::add_leg_direction/add_fib_zones
  - The original top-6 set + bottom/top mirror thresholds + net_score = bottom_votes - top_votes
    convention: scripts/backtest_eth_evidence_signal_top6_confluence_20260814.py (the
    pre-registered "top-6" formula, reused here with zero new thresholds).
This script hand-transcribes the formulas rather than importing those modules directly, because
several of them pull in this repo's heavy backtest/training stack (core.causal_futures_backtest,
multi-window gate definitions, GPU-touching modules, etc.) that a lightweight read-only live
dashboard has no business depending on. The math itself is copied 1:1 -- see each block below.

=== Data source ===
Binance USDT-M perpetual FUTURES public REST API (fapi.binance.com/fapi/v1/klines), no API key.
This matches scripts/update_features.py::ensure_klines, which is what builds/maintains
data/eth_5m_1year.csv -- the exact CSV that analyze_eth_creative_reversal_evidence_signals_
20260814.py (DATA_PATH) and this whole 2026-08-14 research lineage reads. Using spot data here
would silently compute these signals on a different (if closely correlated) price/volume/order-
flow series than the one the validated lift numbers were measured on.

=== Read-only / safety ===
GET requests only, to public Binance endpoints, no API key, no order placement. Writes nothing to
disk. Does not import trading_bot.py or anything under trading_bot_modules/.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "ETHUSDT"
BTC_SYMBOL = "BTCUSDT"  # smt_divergence's cross-asset non-confirmation leg only, 2026-08-24
INTERVAL = "5m"
# The largest rolling window used by any of the 6 signals is the 864-bar percentile-rank window
# (p_fast/p_slow, feeding orthogonal_combo). We fetch the futures endpoint's max (1500) so that
# signal is never NaN and there's a healthy buffer of computed history left over for the
# "bars since last fired" context stats below. (Deliberately more than the "~300-500" floor
# mentioned in the task -- 500 bars would leave orthogonal_combo permanently unavailable.)
FETCH_LIMIT = 1500

STOCH_N = 14           # Williams %R / Fast %K lookback
SLOWK_SMOOTH = 3        # Slow %K smoothing
PCTRANK_WINDOW = 864    # p_fast/p_slow rolling percentile-rank window (3 days of 5m bars)
ZSCORE_WINDOW = 288     # delta_z / vol_z / ret3_z rolling window (1 day of 5m bars)
SWEEP_LOOKBACK = 48     # liquidity_sweep prior swing high/low lookback (4 hours)
ATR_N = 14              # _atr_pct window (context line only, not part of any of the 6 signals)
EPS = 1e-12

DEFAULT_WATCH_INTERVAL_S = 60
MIN_WATCH_INTERVAL_S = 5  # floor to avoid hammering Binance's public endpoint if misconfigured

DISCLAIMER = """\
================================================================================
  *** INFORMATIONAL CONTEXT ONLY -- NOT A TRADING SIGNAL, NOT A BUY/SELL CALL ***
================================================================================
  6 core signals are the repo's 2026-08-14 validated "reversal evidence" signals
  (lift vs a random bar's distance to a real zigzag swing pivot), trimmed from an
  original 11 on 2026-08-24 for glanceability (all 11 were real, non-noise lift --
  see module docstring for exactly what was dropped and why). A 7th, EXPERIMENTAL
  signal (fib_extension_exhaustion) was added the same day from a different
  information family (Fibonacci extension geometry) on a thinner sample -- see its
  own description below. All 7 are probability-shift CONTEXT for a human's own
  discretionary judgment -- never an automated trigger:
    - 4/4 independent attempts to wire the original top-6 subset into automated
      trading decisions (a hard forced-exit veto, an exit_head model feature, a
      position-sizing feature, and a standalone entry-trigger confluence rule
      tested across vote thresholds K=1..6) ALL lost to always_long/always_short
      benchmarks.
    - Even a correct signal historically still saw ~0.5-0.85% further adverse
      price movement before the real pivot -- nothing here calls the exact
      bottom/top.
    - fib_extension_exhaustion additionally failed its own market-order economic
      gate 0/16 (docs/experiments/eth_fibonacci_harmonic_geometric_evidence_20260824.md)
      -- same "context only" status as the rest, but with a thinner evidentiary base.
    - "fired?" shows a 15-MIN SUSTAIN WINDOW (2026-08-24, corrected same day after an
      empirical decay check), not just the single bar the condition fired on -- once
      a signal fires it stays shown as active through 3 bars later (15 min), the
      exact bar-offset up to which lift measurably stays above baseline for all 7
      signals (an initial 1h/12-bar window was tried and found to display several
      signals as active well past the point their real-time lift had decayed below
      random -- see scripts/analyze_eth_dashboard7_sustain_window_decay_20260824.py).
      Not a new/looser firing condition. "last fired" always reports the true
      original firing bar regardless of the sustain window.
  Nothing printed below is an instruction to buy or sell.
================================================================================"""

SIGNAL_ORDER = [
    ("orthogonal_combo", "adaptive oscillator extreme (p_fast/p_slow<=.10 or >=.90) AND taker delta z beyond +-2"),
    ("liquidity_sweep", "wick pokes past prior 48-bar swing high/low, closes back inside"),
    ("volume_wick_climax", "volume z-score>=2 AND opposite-direction wick>=50% of bar range"),
    ("short_term_return_z", "3-bar (15m) return z-score beyond +-2.5"),
    ("taker_delta_z_climax", "net aggressive taker buy/sell volume z-score beyond +-2 (standalone)"),
    ("smt_divergence", "ETH breaks its own 48-bar swing low/high while BTC's does NOT (cross-asset non-confirmation)"),
    ("fib_extension_exhaustion", "EXPERIMENTAL: price extends 127.2-161.8% beyond a causally-detected 48-bar swing leg (thinner sample, n~190, than the 6 core signals -- see docs/experiments/eth_fibonacci_harmonic_geometric_evidence_20260824.md)"),
]


def log_err(msg: str) -> None:
    print(f"[live_evidence_signal_dashboard] {msg}", file=sys.stderr, flush=True)


def fetch_klines(limit: int = FETCH_LIMIT, max_retries: int = 3, timeout: float = 15.0,
                 symbol: str = SYMBOL) -> pd.DataFrame:
    """GET-only fetch of recent perpetual-futures 5m klines (ETHUSDT by default) from Binance's
    public REST API. Retries with backoff on any network/HTTP error; raises RuntimeError only
    after max_retries is exhausted (caller decides whether that's fatal or just skip-this-cycle).
    `symbol` override is used for the BTC_SYMBOL fetch that feeds smt_divergence."""
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
            "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                FUTURES_KLINES_URL,
                params={"symbol": symbol, "interval": INTERVAL, "limit": limit},
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                raise ValueError("empty klines response")
            df = pd.DataFrame(raw, columns=cols)
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
                df[c] = df[c].astype(np.float64)
            df["open_time"] = df["open_time"].astype(np.int64)
            df["close_time"] = df["close_time"].astype(np.int64)
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

            # Binance returns the still-FORMING bar too when queried without an explicit endTime.
            # Drop it so "latest row" == "most recent CLOSED bar", matching the task's requirement.
            now_ms = int(time.time() * 1000)
            if len(df) and int(df.iloc[-1]["close_time"]) >= now_ms:
                df = df.iloc[:-1].reset_index(drop=True)

            if len(df) < PCTRANK_WINDOW:
                log_err(f"warn: only {len(df)} closed bars available (< {PCTRANK_WINDOW} needed to "
                        f"warm up orthogonal_combo's percentile-rank window) -- some signals will read N/A.")
            return df
        except Exception as e:  # noqa: BLE001 -- any fetch/parse failure should retry, never crash the loop
            last_err = e
            if attempt < max_retries:
                sleep_s = 2 ** attempt
                log_err(f"klines fetch failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
    raise RuntimeError(f"failed to fetch {symbol} klines after {max_retries} attempts: {last_err}")


def compute_signals(df: pd.DataFrame, btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Computes the 7 signal families (bottom + top mirror; all bidirectional, including the
    experimental 7th) in SIGNAL_ORDER on freshly fetched bars -- 5 of the 6 core signals from
    scripts/backtest_eth_evidence_signal_top6_confluence_20260814.py, smt_divergence from
    analyze_eth_ict2022_ob_smt_po3_component_evidence_20260824.py (2026-08-24), and the
    experimental fib_extension_exhaustion from analyze_eth_fibonacci_harmonic_geometric_evidence_
    20260824.py (2026-08-24, later same day -- causal 48-bar leg-direction detection ported
    verbatim from that script's add_leg_direction()/add_fib_zones()). Every threshold/window
    below is copied verbatim from that lineage -- see module docstring.

    `btc_df` (BTCUSDT klines, same columns/interval as `df`) is optional -- if omitted (or the
    caller's BTC fetch failed this cycle), smt_divergence simply never fires rather than raising;
    ETH-only signals are entirely unaffected."""
    out = df.copy()
    close, open_, high, low, volume = out["close"], out["open"], out["high"], out["low"], out["volume"]
    taker_buy = out["taker_buy_base"]

    # --- compute_indicators (backtest_eth_slowk_williamsr_persistence_confluence_20260814.py) ---
    hh = high.rolling(STOCH_N, min_periods=STOCH_N).max()
    ll = low.rolling(STOCH_N, min_periods=STOCH_N).min()
    rng_stoch = (hh - ll).replace(0.0, np.nan)
    williams_r = -100.0 * (hh - close) / rng_stoch
    fast_k = 100.0 + williams_r  # raw Fast Stochastic %K(14)
    slow_k = fast_k.rolling(SLOWK_SMOOTH, min_periods=SLOWK_SMOOTH).mean()
    p_fast = fast_k.rolling(PCTRANK_WINDOW, min_periods=PCTRANK_WINDOW).rank(pct=True)
    p_slow = slow_k.rolling(PCTRANK_WINDOW, min_periods=PCTRANK_WINDOW).rank(pct=True)

    # --- add_creative_indicators (analyze_eth_creative_reversal_evidence_signals_20260814.py) ---
    delta = 2.0 * taker_buy - volume  # net aggressive buy volume this bar
    delta_z = (delta - delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        delta.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    vol_z = (volume - volume.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        volume.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    rng_body = (high - low).replace(0.0, np.nan)
    lower_wick_ratio = (np.minimum(open_, close) - low) / (rng_body + EPS)
    upper_wick_ratio = (high - np.maximum(open_, close)) / (rng_body + EPS)

    # --- short_term_return_z (analyze_eth_deep_evidence_signal_sweep_round2_20260814.py, as
    #     reused verbatim by backtest_eth_evidence_signal_top6_confluence_20260814.py) ---
    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)

    # --- liquidity_sweep (analyze_eth_broad_evidence_signal_sweep_20260814.py) ---
    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    sweep_low = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_high = (high > swing_high_prior) & (close < swing_high_prior)

    # --- fib_extension_exhaustion (analyze_eth_fibonacci_harmonic_geometric_evidence_20260824.py::
    #     add_leg_direction/add_fib_zones, EXPERIMENTAL 7th signal) -- causal 48-bar leg-direction
    #     via vectorized argmin/argmax over the trailing window ending at bar i-1 (same window
    #     swing_low_prior/swing_high_prior already use), then a zone-touch 127.2-161.8% beyond the
    #     leg's far extreme. leg_up = the low occurred before the high in that window (most recent
    #     extreme is the high); leg_down mirrors it. ---
    leg_window = SWEEP_LOOKBACK
    low_arr, high_arr = low.to_numpy(), high.to_numpy()
    n = len(out)
    low_pos = np.full(n, -1, dtype=np.int64)
    high_pos = np.full(n, -1, dtype=np.int64)
    if n > leg_window:
        lo_windows = np.lib.stride_tricks.sliding_window_view(low_arr, leg_window)
        hi_windows = np.lib.stride_tricks.sliding_window_view(high_arr, leg_window)
        idx = np.arange(leg_window, n)
        j = idx - leg_window
        low_pos[idx] = j + lo_windows[j].argmin(axis=1)
        high_pos[idx] = j + hi_windows[j].argmax(axis=1)
    leg_up = pd.Series(low_pos < high_pos, index=out.index)
    leg_down = pd.Series(high_pos < low_pos, index=out.index)
    fib_rng = (swing_high_prior - swing_low_prior).replace(0.0, np.nan)
    fib_ext_top = leg_up & high.between(swing_high_prior + 0.272 * fib_rng, swing_high_prior + 0.618 * fib_rng)
    fib_ext_bottom = leg_down & low.between(swing_low_prior - 0.618 * fib_rng, swing_low_prior - 0.272 * fib_rng)

    # --- _atr_pct (eval_omega4_1_atr_safety_sltp_20260622.py) -- context line only, not a signal ---
    prev_close = close.shift(1)
    prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(ATR_N, min_periods=1).mean() / close.clip(lower=1e-12)

    out["p_fast"], out["p_slow"], out["delta_z"], out["vol_z"] = p_fast, p_slow, delta_z, vol_z
    out["lower_wick_ratio"], out["upper_wick_ratio"] = lower_wick_ratio, upper_wick_ratio
    out["ret3_z"], out["atr_pct"] = ret3_z, atr_pct

    out["bottom_orthogonal_combo"] = (p_fast <= 0.10) & (p_slow <= 0.10) & (delta_z <= -2.0)
    out["top_orthogonal_combo"] = (p_fast >= 0.90) & (p_slow >= 0.90) & (delta_z >= 2.0)

    out["bottom_liquidity_sweep"] = sweep_low
    out["top_liquidity_sweep"] = sweep_high

    out["bottom_volume_wick_climax"] = (vol_z >= 2.0) & (lower_wick_ratio >= 0.5)
    out["top_volume_wick_climax"] = (vol_z >= 2.0) & (upper_wick_ratio >= 0.5)

    out["bottom_short_term_return_z"] = ret3_z <= -2.5
    out["top_short_term_return_z"] = ret3_z >= 2.5

    out["bottom_taker_delta_z_climax"] = delta_z <= -2.0
    out["top_taker_delta_z_climax"] = delta_z >= 2.0

    # --- smt_divergence (2026-08-24 addition) ---
    # eth_break_low/high reuse swing_low_prior/swing_high_prior (already computed above for
    # liquidity_sweep) -- verbatim match to add_smt()'s "f['low'] < f['swing_low_prior']" in the
    # source script, since add_sweep() there computes that same 48-bar prior-swing series.
    if btc_df is not None and len(btc_df) and "timestamp" in btc_df.columns:
        btc_aligned = out[["timestamp"]].merge(
            btc_df[["timestamp", "high", "low"]].rename(columns={"high": "btc_high", "low": "btc_low"}),
            on="timestamp", how="left",
        )
        btc_swing_low = btc_aligned["btc_low"].rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
        btc_swing_high = btc_aligned["btc_high"].rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
        btc_holds_low = (btc_aligned["btc_low"] > btc_swing_low).to_numpy()
        btc_holds_high = (btc_aligned["btc_high"] < btc_swing_high).to_numpy()
        smt_bottom = (low < swing_low_prior) & btc_holds_low
        smt_top = (high > swing_high_prior) & btc_holds_high
    else:
        smt_bottom = pd.Series(False, index=out.index)
        smt_top = pd.Series(False, index=out.index)

    out["bottom_smt_divergence"] = smt_bottom
    out["top_smt_divergence"] = smt_top

    out["bottom_fib_extension_exhaustion"] = fib_ext_bottom
    out["top_fib_extension_exhaustion"] = fib_ext_top

    bottom_cols = [f"bottom_{name}" for name, _ in SIGNAL_ORDER]
    top_cols = [f"top_{name}" for name, _ in SIGNAL_ORDER]

    # --- sustain window (2026-08-24, user request: signals flash for one bar then vanish) ---
    # Pure DISPLAY-layer change: the raw bottom_{name}/top_{name} firing columns above are
    # UNCHANGED (still the exact, already-measured lift-vs-pivot definitions -- last_fired_ts in
    # server.py still reads these raw columns, so "fired X ago" always reports the true original
    # firing bar, never reset by the sustain window). What's new is `_active`: a bar is "active"
    # if it fired on THIS bar or any of the SUSTAIN_BARS bars before it -- a rolling-max, not a
    # new firing condition, so it needs no new lift validation and adds no multiple-comparison
    # risk (unlike loosening a threshold, which WAS declined the same day for exactly that risk).
    #
    # SUSTAIN_BARS was originally set to 12 (K_HORIZONS["K12_1h"]) on the mistaken assumption
    # that "the headline lift is measured over a 1h forward window" means "the evidence stays
    # equally strong for 1h" -- it does NOT: K12_1h measures the CUMULATIVE probability of a
    # pivot occurring SOMEWHERE within the 12 bars after firing, not a stationary elevated state
    # that holds at every bar within that window. The user directly asked "could a sustained-only
    # bar be a false positive, does accuracy hold" and this was checked empirically the same day
    # (scripts/analyze_eth_dashboard7_sustain_window_decay_20260824.py): lift decays fast and
    # monotonically after the original firing bar. All 14 side x signal series stay clearly above
    # the 1.0x baseline through bar-offset 3 (15 min), but several cross to AT-OR-BELOW baseline
    # starting offset 4 (fib_extension_exhaustion top 0.98x, smt_divergence top 1.03x) and multiple
    # are below 1.0x (worse than random) by offset 8-11 (liquidity_sweep bottom 0.94x,
    # taker_delta_z_climax bottom 0.89x) -- i.e. the original 12-bar sustain window was displaying
    # several signals as "active" for stretches where they were measurably WORSE than a random bar.
    # SUSTAIN_BARS=4 is the corrected value: pandas rolling(N).max() at the current bar covers
    # offsets [0, N-1], so N=4 includes offsets 0-3 (all 14 series still clearly above baseline,
    # weakest fib_extension_exhaustion top at 1.03x) and excludes offset 4 (first crossings to
    # at-or-below baseline: fib_extension_exhaustion top 0.98x, smt_divergence top 1.03x) -- not
    # a re-guessed round number, the exact empirical cutoff. In elapsed time this is 15 minutes
    # (offset 3 = 3 bars * 5min after the firing bar).
    SUSTAIN_BARS = 4
    for col in bottom_cols + top_cols:
        out[f"{col}_active"] = out[col].fillna(False).rolling(SUSTAIN_BARS, min_periods=1).max().astype(bool)
    active_bottom_cols = [f"{c}_active" for c in bottom_cols]
    active_top_cols = [f"{c}_active" for c in top_cols]
    out["bottom_votes"] = out[active_bottom_cols].sum(axis=1).astype(int)
    out["top_votes"] = out[active_top_cols].sum(axis=1).astype(int)
    out["net_score"] = out["bottom_votes"] - out["top_votes"]
    return out


def bars_since_last_true(series: pd.Series) -> int | None:
    """Bars since `series` was last True (0 = fired on the latest/current bar). None means it
    never fired anywhere in the currently loaded lookback (not proof it never fires -- just not
    observed in the ~1500-bar / ~5.2-day window this run happened to load)."""
    true_idx = np.flatnonzero(series.fillna(False).to_numpy())
    if len(true_idx) == 0:
        return None
    return int(len(series) - 1 - true_idx[-1])


def fmt_bars_ago(bars: int | None) -> str:
    if bars is None:
        return "not in lookback"
    if bars == 0:
        return "NOW"
    minutes = bars * 5
    if minutes < 60:
        return f"{bars}bar / {minutes}m ago"
    return f"{bars}bar / {minutes / 60.0:.1f}h ago"


def render(sig: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(DISCLAIMER)
    lines.append("")

    now_utc = datetime.now(timezone.utc)
    latest = sig.iloc[-1]
    latest_ready = bool(pd.notna(latest["p_fast"]) and pd.notna(latest["p_slow"]))

    lines.append(f"Generated:              {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"Latest CLOSED 5m bar:   {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append(f"Current price (close):  {latest['close']:.2f} USDT   (ETHUSDT perpetual futures, fapi.binance.com)")
    lines.append(f"Bars loaded:            {len(sig)} closed 5m bars (still-forming bar excluded)")
    if pd.notna(latest["atr_pct"]):
        lines.append(f"Realized vol (ATR{ATR_N}%): {latest['atr_pct'] * 100:.3f}%  <- rough recent-volatility context, not one of the 6 signals")
    if not latest_ready:
        lines.append("")
        lines.append(f"  [WARNING] fewer than {PCTRANK_WINDOW} closed bars available -- orthogonal_combo (and possibly")
        lines.append("  other signals) cannot be computed reliably on this run; all fired? columns below read N/A.")
    lines.append("")

    header = f"{'Signal':<26}{'Bottom (long-side evidence)':<34}{'Top (short-side evidence)':<34}"
    lines.append("-" * len(header))
    lines.append(header)
    lines.append(f"{'':<24}{'fired?':<10}{'last fired':<24}{'fired?':<10}{'last fired':<24}")
    lines.append("-" * len(header))

    for name, _desc in SIGNAL_ORDER:
        bcol, tcol = f"bottom_{name}", f"top_{name}"
        if latest_ready:
            b_fired, t_fired = bool(latest[f"{bcol}_active"]), bool(latest[f"{tcol}_active"])
            b_txt = "YES" if b_fired else "no"
            t_txt = "YES" if t_fired else "no"
        else:
            b_txt = t_txt = "N/A"
        b_ago = fmt_bars_ago(bars_since_last_true(sig[bcol])) if latest_ready else "N/A"
        t_ago = fmt_bars_ago(bars_since_last_true(sig[tcol])) if latest_ready else "N/A"
        lines.append(f"{name:<26}{b_txt:<10}{b_ago:<24}{t_txt:<10}{t_ago:<24}")

    lines.append("-" * len(header))
    lines.append("")
    lines.append("Signal definitions:")
    for name, desc in SIGNAL_ORDER:
        lines.append(f"  - {name}: {desc}")
    lines.append("")

    if latest_ready:
        bv, tv, net = int(latest["bottom_votes"]), int(latest["top_votes"]), int(latest["net_score"])
        lines.append(f"NET_SCORE (bottom_votes - top_votes) on latest closed bar: {net:+d}   [bottom_votes={bv}  top_votes={tv}]")
        if bv == 0 and tv == 0:
            tail = "No elevated bottom- or top-side reversal evidence on the latest closed bar."
        elif bv > tv:
            tail = f"{bv} bottom-side (long-evidence) signal(s) firing vs {tv} top-side -- net tilt toward bottom-side evidence."
        elif tv > bv:
            tail = f"{tv} top-side (short-evidence) signal(s) firing vs {bv} bottom-side -- net tilt toward top-side evidence."
        else:
            tail = f"{bv} bottom-side and {tv} top-side signal(s) both firing -- mixed/conflicting evidence, no net tilt."
        lines.append(f"  -> {tail}")
        lines.append("     Reminder: this is probability-shift CONTEXT ONLY -- see disclaimer above. Not a trade trigger.")
    else:
        lines.append("NET_SCORE: unavailable this run (insufficient warmup -- see WARNING above).")

    return "\n".join(lines)


def fetch_btc_klines_safe() -> pd.DataFrame | None:
    """BTC fetch for smt_divergence's cross-asset leg -- failure here must never take down the
    ETH-only signals, so it's caught and logged, not raised."""
    try:
        return fetch_klines(symbol=BTC_SYMBOL)
    except RuntimeError as e:
        log_err(f"BTC klines fetch failed ({e}) -- smt_divergence will read as not-fired this "
                f"cycle; ETH-only signals unaffected.")
        return None


def run_once() -> int:
    try:
        raw = fetch_klines()
    except RuntimeError as e:
        log_err(str(e))
        return 1
    btc_raw = fetch_btc_klines_safe()
    sig = compute_signals(raw, btc_df=btc_raw)
    print(render(sig))
    return 0


def run_watch(interval: int) -> int:
    if interval < MIN_WATCH_INTERVAL_S:
        log_err(f"--watch {interval}s is below the floor of {MIN_WATCH_INTERVAL_S}s (avoids hammering "
                f"Binance's public endpoint); clamping to {MIN_WATCH_INTERVAL_S}s.")
        interval = MIN_WATCH_INTERVAL_S
    print(f"[watch mode] refreshing every {interval}s (2 requests/cycle -- ETH + BTC). Press Ctrl+C to stop.\n")
    try:
        while True:
            try:
                raw = fetch_klines()
                btc_raw = fetch_btc_klines_safe()
                sig = compute_signals(raw, btc_df=btc_raw)
                print(render(sig))
            except RuntimeError as e:
                log_err(f"{e} -- skipping this cycle, will retry in {interval}s.")
            print(f"\n(next refresh in {interval}s -- Ctrl+C to stop)\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch mode] stopped by user.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only live ETH reversal-evidence-signal dashboard. INFORMATIONAL ONLY -- not a trading signal.",
    )
    parser.add_argument(
        "--watch", nargs="?", const=DEFAULT_WATCH_INTERVAL_S, default=None, type=int, metavar="SECONDS",
        help=f"Loop and reprint every SECONDS (default {DEFAULT_WATCH_INTERVAL_S}s if flag given with no value). "
             f"Omit this flag entirely for one-shot mode (default).",
    )
    args = parser.parse_args()
    if args.watch is None:
        return run_once()
    return run_watch(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
