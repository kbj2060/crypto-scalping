#!/usr/bin/env python3
"""Read-only, on-demand/live terminal dashboard for a human discretionary trader: shows which of
the 8 validated ETH "reversal evidence" signals (2026-08-14 research lineage + later additions)
are firing on the most recently CLOSED ETH/USDT 5-minute bar, right now.

2026-08-25: added dalton_rule2_balance_edge as an 8th signal, on user request (dashboard exposure
now gated on "real statistical information content", not "passed a full economic PnL gate" -- see
feedback_dashboard_indicators_ic_bar_not_pnl_bar memory). Retrospective lift was real and VAL/OOS
stable (bottom 1.69->1.89x, top 1.66->1.42x) but the signal failed a *different* kind of test than
fib_extension_exhaustion below: not an economic/cost-gate failure, but a translation failure --
0/6 windows beat always_long/always_short even at ZERO transaction cost, because a fixed 1.6xATR
TP can't survive the signal's own measured 4-5 bar lead time before the real pivot. See
docs/experiments/eth_dalton_rule2_balance_edge_costgate_20260815.md.

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

2026-08-25 (same day, later still): added funding_oscillator_combo as a 9th signal, on user
request ("find combo candidates using other info sources: liquidation/funding/cross-asset").
Liquidation was ruled out -- its only data source (tail_risk_1m/oi_lsratio) has zero historical
overlap with this scorecard's VAL+OOS window, and forcing a recent-only-window test would re-peek
data still accumulating toward its own pre-registered 09-15 gate. Funding (ETH oscillator oversold
AND funding_z<=-2, data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv -- the corrected file, see
eth_funding_ethfi_mislabel_20260824) beat orthogonal_combo at 1h in BOTH the original window
(4.00x vs 3.51x, n=110) and an independent out-of-window replication on 2026-03-01..07-20 (4.04x
vs 3.92x @1h, and clearly ahead at 4h/8h too there) -- see research_eth_funding_crossasset_
combo_signal_20260825.py / research_eth_funding_oscillator_combo_oow_20260825.py. BOTTOM-SIDE ONLY:
this data's funding rate never exceeds 0.0001 in the validating window (exchange premium-clamp
behavior, independently confirmed via live API checks the same day, see
eth_f4c_cross_exchange_funding_spread_status_20260825), so funding_z>=2.0 essentially never fires
-- the top_funding_oscillator_combo formula is left in verbatim (bidirectional code, matching every
other signal here) but will rarely-to-never light up; this is a real data characteristic, not a bug.
2026-08-25 (same day, later still): economic cost-gate run (backtest_eth_funding_oscillator_
combo_costgate_20260825.py, same engine/TP:SL/6-window convention as dalton_rule2_balance_edge)
-- FAILED 0/6 windows vs always_long/always_short at the standard 10bp roundtrip cost, breakeven
cost only 0.0-9.6bp per window. Same failure class as dalton_rule2_balance_edge and
fib_extension_exhaustion: real detection, fixed-barrier automated entry doesn't survive it.

2026-08-27: funding_oscillator_combo REMOVED as a standalone signal (rarely visible in practice --
its bottom_last_fired_ts/top_last_fired_ts were both null across the entire live ~5.2-day lookback
the day this was noticed) and folded into orthogonal_combo's BOTTOM leg instead, as an OR condition
(delta_z<=-2 OR funding_z<=-2). research_eth_funding_oscillator_union_combo_20260827.py tested this
union on the same two windows: bottom lift held (3.51x->3.56x original, 3.92x->4.01x OOW) while
trigger count rose ~3x and the median gap between fires fell from 5.8h to 1.75h (original) / 7.8h to
3.9h (OOW) -- funding_oscillator_combo alone had gone as long as 55 days without firing. TOP was
deliberately NOT merged: funding_z>=2.0 fires rarely there too, and in the OOW window its few fires
were BELOW-baseline (lift 0.78x) and measurably dragged orthogonal_combo's own top lift down
(4.14x->3.90x if merged) -- so top stays delta_z-only, exactly its pre-2026-08-27 formula. Now 8
signals total (was 9).

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
  - dalton_rule2_balance_edge (added 2026-08-25, balance_edge_low/balance_edge_high portion only):
    scripts/analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.py::add_amt_features
  - orthogonal_combo's funding_z OR-input (2026-08-27, bottom-side only): funding_z leg from
    scripts/research_eth_funding_crossasset_combo_signal_20260825.py::load_funding_z (rolling
    z-score of data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv's last_funding_rate); originally a
    separate funding_oscillator_combo signal (out-of-window replication in scripts/research_eth_
    funding_oscillator_combo_oow_20260825.py), merged into orthogonal_combo's bottom leg per
    scripts/research_eth_funding_oscillator_union_combo_20260827.py.
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
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"  # orthogonal_combo's bottom-leg funding_z input (2026-08-27; formerly funding_oscillator_combo's own leg)
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
FUNDING_HISTORY_LIMIT = 100    # 100 x 8h ~= 33 days, warms up FUNDING_Z_WINDOW with margin
FUNDING_Z_WINDOW = 90          # ~30 days of 8h funding observations, matches research_eth_funding_crossasset_combo_signal_20260825.py::load_funding_z
FUNDING_Z_MIN_PERIODS = 30     # same
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
  own description below. An 8th (dalton_rule2_balance_edge) was added 2026-08-25 --
  real, VAL/OOS-stable lift, but it failed a DIFFERENT kind of test than
  fib_extension_exhaustion (see its own bullet below). A 9th (funding_oscillator_
  combo), also added 2026-08-25, combined the same oscillator leg as orthogonal_
  combo with a funding-rate extreme; it was REMOVED 2026-08-27 and folded into
  orthogonal_combo's own BOTTOM leg instead (see orthogonal_combo's bullet below)
  after it turned out to rarely display at all (up to 55 days without firing).
  All 8 are probability-shift CONTEXT for a human's own discretionary judgment --
  never an automated trigger:
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
    - dalton_rule2_balance_edge failed a TRANSLATION test, not an economic one: 0/6
      windows beat always_long/always_short even at ZERO transaction cost (breakeven
      cost 0.0-1.6bp) -- a fixed 1.6xATR TP can't survive the signal's own measured
      4-5 bar lead time before the real pivot arrives (docs/experiments/
      eth_dalton_rule2_balance_edge_costgate_20260815.md). The underlying detection
      itself is real; only a fixed-barrier automated exit was shown not to work.
    - orthogonal_combo's BOTTOM leg is an OR of two confirming conditions since
      2026-08-27: taker delta_z<=-2 OR funding_z<=-2 (funding_z beat orthogonal_
      combo's own OLD delta_z-only lift at 1h in two independent windows before
      the merge -- research_eth_funding_crossasset_combo_signal_20260825.py /
      research_eth_funding_oscillator_combo_oow_20260825.py -- but as its own
      standalone chip it rarely displayed, up to 55 days without firing). The
      union was verified in research_eth_funding_oscillator_union_combo_20260827.py
      to hold lift in both windows while ~3x'ing trigger frequency. orthogonal_
      combo's TOP leg deliberately stays delta_z-only -- funding_z's top leg fires
      rarely (exchange premium-clamp keeps ETH funding under 0.0001 most of the
      time) and its few OOW-window fires were BELOW-baseline (0.78x lift), so
      merging it there would have hurt, not helped. The old standalone funding_
      oscillator_combo's economic cost-gate result (backtest_eth_funding_
      oscillator_combo_costgate_20260825.py): FAILED 0/6 windows vs always_long/
      always_short at the standard 10bp roundtrip cost -- same failure CLASS as
      dalton_rule2_balance_edge and fib_extension_exhaustion (real detection,
      fixed-barrier automated entry doesn't survive it). This dashboard's
      inclusion bar is statistical information content, not that economic gate --
      see feedback_dashboard_indicators_ic_bar_not_pnl_bar.
    - "fired?" shows a 15-MIN SUSTAIN WINDOW (2026-08-24, corrected same day after an
      empirical decay check), not just the single bar the condition fired on -- once
      a signal fires it stays shown as active through 3 bars later (15 min), the
      exact bar-offset up to which lift measurably stays above baseline for the
      original 7 signals (an initial 1h/12-bar window was tried and found to display
      several signals as active well past the point their real-time lift had decayed
      below random -- see scripts/analyze_eth_dashboard7_sustain_window_decay_20260824.py).
      dalton_rule2_balance_edge (added 2026-08-25) reuses the same 4-bar window as
      a design default -- its own decay curve was not separately re-measured.
      Not a new/looser firing condition. "last fired" always reports the true
      original firing bar regardless of the sustain window.
  Nothing printed below is an instruction to buy or sell.
================================================================================"""

# Ordered by 1h-horizon precision (accuracy), descending -- computed once across all 9 in
# scripts/research_eth_evidence_signal_scorecard_ci_20260825.py (same live compute_signals(),
# same VAL+OOS window, same event_study lift methodology as everywhere else in this lineage).
# Ranking key is each signal's stronger side (max of bottom/top precision) at 1h; re-run that
# script if a signal's formula changes and this order needs to be re-derived, don't hand-tweak.
SIGNAL_ORDER = [
    # 2026-09-02 (user request): reordered by REAL profit win rate (trade_return>0 after cost, not
    # raw directional accuracy), per docs/homer/README.md ss5.11's cross-signal synthesis (2026-09-01,
    # "best floor" VAL figures -- see that section for the full PF/mean/median/tail-dependence table
    # this ordering summarizes down to a single sort key). Order affects ONLY display (this list also
    # drives bottom_votes/top_votes via an order-independent column sum, and the CLI debug table) --
    # no computed value depends on list order. fib_extension_exhaustion is deliberately placed LAST
    # despite its raw 73.9% figure (which would rank #2) -- ss5.11 found its 91.1% directional accuracy
    # masks a 23.0pp cost-erosion gap (by far the worst of the 8) and the 2026-09-01 direction-flip
    # audit already withdrew its economics-gate claim entirely (classification/AUC only, still valid).
    ("demarker_extreme", "DeMarker(14) high/low-based oscillator >=0.90 (top) or <=0.10 (bottom) -- 2026-08-31: Homer candidate-pool signal (not one of the original 8), TabPFN meta-label deployed, VAL/OOS/HOLDOUT AUC 0.7527/0.7157/0.7464 (this project's best classification result). Permutation importance found bb_pctb, not dem itself, is the actual classification driver -- confirmed not a lookahead bug via a dedicated audit, see docs/homer/README.md"),
    ("orthogonal_combo", "adaptive oscillator extreme (p_fast/p_slow<=.10 or >=.90) AND (BOTTOM: taker delta_z<=-2 OR funding_z<=-2; TOP: taker delta_z>=2 only) -- 2026-08-27: bottom leg OR-merged with the former standalone funding_oscillator_combo signal after research_eth_funding_oscillator_union_combo_20260827.py showed the union beats/matches both originals' lift in two independent windows while ~3x'ing trigger frequency (funding_oscillator_combo alone had gone up to 55 days without firing); TOP deliberately excludes funding_z (its rare OOW-window fires were below-baseline and hurt lift)"),
    ("short_term_return_z", "3-bar (15m) return z-score beyond +-2.5"),
    ("taker_delta_z_climax", "net aggressive taker buy/sell volume z-score beyond +-2 (standalone)"),
    ("smt_divergence", "ETH breaks its own 48-bar swing low/high while BTC's does NOT (cross-asset non-confirmation)"),
    ("liquidity_sweep", "wick pokes past prior 48-bar swing high/low, closes back inside"),
    ("kalman_deviation_meanrev", "(close - Kalman-filtered trend level)/level, rolling-288-bar z-scored, >=2.0 (top) or <=-2.0 (bottom) -- 2026-08-31: Homer candidate-pool signal, TabPFN meta-label deployed, VAL/OOS/HOLDOUT AUC 0.6569/0.6311/0.6284, see docs/homer/README.md"),
    ("fib_extension_exhaustion", "price extends 27.2-61.8% beyond a causally-detected 48-bar swing leg's opposite extreme (\"extension exhaustion\", betting on reversal) -- 2026-08-31: TabPFN meta-label deployed, VAL/OOS/HOLDOUT AUC 0.605/0.620/0.621; the earlier \"thinner sample, n~190\" note was a 5.5-month-window artifact, full-history recount is bottom=1078/top=1072, see docs/homer/README.md. ss2026-09-01: economics-gate claim WITHDRAWN (direction-flip audit + ss5.11's 23.0pp cost-erosion gap, worst of the 8) -- classification/AUC only, placed last in this list despite raw 73.9% profit-win-rate figure for exactly that reason."),
    # 2026-08-31: volume_wick_climax and dalton_rule2_balance_edge REMOVED from the dashboard
    # entirely (user decision) -- both accumulated consistent negative evidence across independent
    # angles after their Homer TabPFN metalabel upgrade: volume_wick_climax had this project's
    # worst HOLDOUT AUC (0.529, barely above random) and a 0/12 trailing-stop cost-gate FAILED
    # (v1 and v2); dalton_rule2_balance_edge had lift<1x (its own low-vol-regime gate added no
    # information over the edge-proximity condition alone) and a 0/96 trailing-stop cost-gate
    # FAILED. Neither's rule-based detection was ever proven wrong, but neither showed any
    # validated informational or economic edge worth continuing to surface here -- see
    # eth_volume_wick_climax_metalabel_v1_weak_signal_20260830.md /
    # eth_dalton_rule2_balance_edge_metalabel_v1_20260830.md. compute_signals() itself still
    # computes bottom_volume_wick_climax/top_volume_wick_climax and bottom_dalton_rule2_balance_edge/
    # top_dalton_rule2_balance_edge unchanged (used by liquidity_sweep's phase1 overlap check and
    # any future re-evaluation) -- only removed from this display/voting list.
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


def fetch_funding_history(limit: int = FUNDING_HISTORY_LIMIT, max_retries: int = 3, timeout: float = 15.0,
                          symbol: str = SYMBOL) -> pd.DataFrame:
    """GET-only fetch of recent funding-rate history (public, no API key) -- feeds
    orthogonal_combo's bottom-leg funding_z input only (2026-08-27; formerly a separate
    funding_oscillator_combo signal's own leg). Returns [calc_time, funding_z], already
    rolling-z-scored (same FUNDING_Z_WINDOW/FUNDING_Z_MIN_PERIODS convention, verbatim, as
    scripts/research_eth_funding_crossasset_combo_signal_20260825.py::load_funding_z)."""
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(FUNDING_URL, params={"symbol": symbol, "limit": limit}, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                raise ValueError("empty funding-rate response")
            df = pd.DataFrame(raw)
            df["calc_time"] = pd.to_datetime(df["fundingTime"].astype(np.int64), unit="ms", utc=True)
            df["fundingRate"] = df["fundingRate"].astype(np.float64)
            df = df.sort_values("calc_time").drop_duplicates("calc_time", keep="last").reset_index(drop=True)
            mean = df["fundingRate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).mean()
            std = df["fundingRate"].rolling(FUNDING_Z_WINDOW, min_periods=FUNDING_Z_MIN_PERIODS).std()
            df["funding_z"] = (df["fundingRate"] - mean) / std.replace(0.0, np.nan)
            return df[["calc_time", "funding_z"]]
        except Exception as e:  # noqa: BLE001 -- any fetch/parse failure should retry, never crash the loop
            last_err = e
            if attempt < max_retries:
                sleep_s = 2 ** attempt
                log_err(f"funding-rate fetch failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_s}s...")
                time.sleep(sleep_s)
    raise RuntimeError(f"failed to fetch {symbol} funding-rate history after {max_retries} attempts: {last_err}")


def fetch_funding_safe() -> pd.DataFrame | None:
    """Funding fetch for orthogonal_combo's bottom-leg funding_z input -- failure here must never
    take down the other signals, so it's caught and logged, not raised (mirrors
    fetch_btc_klines_safe)."""
    try:
        return fetch_funding_history()
    except RuntimeError as e:
        log_err(f"funding-rate fetch failed ({e}) -- orthogonal_combo's bottom leg degrades to "
                f"delta_z-only this cycle (its pre-2026-08-27 formula); other signals unaffected.")
        return None


def compute_signals(df: pd.DataFrame, btc_df: pd.DataFrame | None = None,
                    funding_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Computes the 8 signal families (bottom + top mirror; all bidirectional, including the
    experimental 7th) in SIGNAL_ORDER on freshly fetched bars -- 5 of the 6 core signals from
    scripts/backtest_eth_evidence_signal_top6_confluence_20260814.py, smt_divergence from
    analyze_eth_ict2022_ob_smt_po3_component_evidence_20260824.py (2026-08-24), and the
    experimental fib_extension_exhaustion from analyze_eth_fibonacci_harmonic_geometric_evidence_
    20260824.py (2026-08-24, later same day -- causal 48-bar leg-direction detection ported
    verbatim from that script's add_leg_direction()/add_fib_zones()). Every threshold/window
    below is copied verbatim from that lineage -- see module docstring.

    `btc_df` (BTCUSDT klines, same columns/interval as `df`) is optional -- if omitted (or the
    caller's BTC fetch failed this cycle), smt_divergence simply never fires rather than raising;
    ETH-only signals are entirely unaffected. `funding_df` ([calc_time, funding_z] from
    fetch_funding_history/fetch_funding_safe) is likewise optional -- if omitted, orthogonal_combo's
    bottom leg just degrades to its delta_z-only pre-2026-08-27 formula rather than raising."""
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

    # --- _atr_pct (eval_omega4_1_atr_safety_sltp_20260622.py) -- context line, AND the
    # low-vol-regime gate input for dalton_rule2_balance_edge below (2026-08-25) ---
    prev_close = close.shift(1)
    prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = tr.rolling(ATR_N, min_periods=1).mean() / close.clip(lower=1e-12)

    # --- dalton_rule2_balance_edge (analyze_eth_amt_vsa_footprint_ifvg_component_evidence_
    #     20260815.py::add_amt_features, balance_edge_low/high portion, verbatim -- 2026-08-25) ---
    # range_low/range_high are NOT shifted (unlike swing_low_prior/swing_high_prior above) --
    # Dalton Rule 2 asks "is price CURRENTLY sitting near the edge of its own recent range", so
    # the current bar is deliberately included in the 48-bar window.
    dalton_atr_pctile = atr_pct.rolling(288, min_periods=144).rank(pct=True)
    dalton_low_vol_regime = dalton_atr_pctile <= 0.30
    dalton_range_low = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min()
    dalton_range_high = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max()
    dalton_tol = 0.15 * (dalton_range_high - dalton_range_low)
    balance_edge_low = dalton_low_vol_regime & ((low - dalton_range_low).abs() <= dalton_tol)
    balance_edge_high = dalton_low_vol_regime & ((dalton_range_high - high).abs() <= dalton_tol)

    # --- funding_z, orthogonal_combo's bottom-leg OR-input (2026-08-25, merged into orthogonal_
    #     combo 2026-08-27) -- research_eth_funding_crossasset_combo_signal_20260825.py /
    #     research_eth_funding_oscillator_combo_oow_20260825.py. funding_z merged in via
    #     merge_asof(direction="backward"): only the most recently PUBLISHED funding rate as of
    #     each bar, no lookahead (fetch_funding_history already rolling-z-scored it). Originally a
    #     separate funding_oscillator_combo signal; research_eth_funding_oscillator_union_combo_
    #     20260827.py showed OR-ing it into orthogonal_combo's BOTTOM leg (delta_z<=-2 OR
    #     funding_z<=-2) beats/matches both original signals' lift in two independent windows while
    #     ~3x'ing trigger frequency and cutting the median gap between fires from 5.8h to 1.75h
    #     (original window) / 7.8h to 3.9h (OOW) -- funding_oscillator_combo alone had gone up to
    #     55 days without firing. The TOP leg deliberately does NOT get the same OR: this data's
    #     funding rate rarely exceeds 0.0001 (exchange premium-clamp, see
    #     eth_f4c_cross_exchange_funding_spread_status_20260825), and the union script found the
    #     rare OOW-window top fires funding_z>=2.0 did produce were BELOW-baseline (lift 0.78x) and
    #     measurably dragged orthogonal_combo's own top lift down (4.14x->3.90x) -- so top stays
    #     delta_z-only, unchanged from pre-2026-08-27 behavior.
    if funding_df is not None and len(funding_df):
        out = pd.merge_asof(out.sort_values("timestamp"), funding_df, left_on="timestamp", right_on="calc_time", direction="backward")
    else:
        out["funding_z"] = np.nan
    funding_z = out["funding_z"]

    out["p_fast"], out["p_slow"], out["delta_z"], out["vol_z"] = p_fast, p_slow, delta_z, vol_z
    out["lower_wick_ratio"], out["upper_wick_ratio"] = lower_wick_ratio, upper_wick_ratio
    out["ret3_z"], out["atr_pct"] = ret3_z, atr_pct

    out["bottom_orthogonal_combo"] = (p_fast <= 0.10) & (p_slow <= 0.10) & ((delta_z <= -2.0) | (funding_z <= -2.0))
    out["top_orthogonal_combo"] = (p_fast >= 0.90) & (p_slow >= 0.90) & (delta_z >= 2.0)  # funding_z top leg deliberately excluded, see 2026-08-27 note above

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

    out["bottom_dalton_rule2_balance_edge"] = balance_edge_low
    out["top_dalton_rule2_balance_edge"] = balance_edge_high

    # --- demarker_extreme (2026-08-31 addition, Homer candidate-pool signal, not one of the
    # original 8) -- verbatim formula from research_eth_demarker_evidence_signal_lift_check_
    # 20260831.py::compute_demarker(), inlined here to match this module's own self-contained
    # convention (no cross-script imports elsewhere in this file). ---
    dem_up_move = high.diff()
    dem_down_move = low.shift(1) - low
    dem_de_max = dem_up_move.clip(lower=0.0).fillna(0.0)
    dem_de_min = dem_down_move.clip(lower=0.0).fillna(0.0)
    dem_sma_max = dem_de_max.rolling(14, min_periods=14).mean()
    dem_sma_min = dem_de_min.rolling(14, min_periods=14).mean()
    dem = dem_sma_max / (dem_sma_max + dem_sma_min).replace(0.0, np.nan)

    # --- kalman_deviation_meanrev (2026-08-31 addition, Homer candidate-pool signal) -- same
    # F/H/Q/R as features/engineering.py::_kalman_trend_velocity, extended to also keep the level
    # state x[0] (that live feature only returns velocity x[1]) -- inlined verbatim from
    # research_eth_candidate_pool_raw_lift_check_20260831.py::kalman_level_and_velocity(). ---
    _kf_close = close.to_numpy()
    _kf_F = np.array([[1.0, 1.0], [0.0, 1.0]])
    _kf_H = np.array([[1.0, 0.0]])
    _kf_Q = np.eye(2) * 1e-5
    _kf_R = np.array([[1e-3]])
    _kf_x = np.array([_kf_close[0], 0.0])
    _kf_P = np.eye(2)
    _kf_levels = np.empty(len(_kf_close))
    for _kf_i in range(len(_kf_close)):
        _kf_x = _kf_F @ _kf_x
        _kf_P = _kf_F @ _kf_P @ _kf_F.T + _kf_Q
        _kf_S = (_kf_H @ _kf_P @ _kf_H.T + _kf_R)[0, 0]
        _kf_K = (_kf_P @ _kf_H.T).flatten() / _kf_S
        _kf_inn = _kf_close[_kf_i] - (_kf_H @ _kf_x)[0]
        _kf_x = _kf_x + _kf_K * _kf_inn
        _kf_P = (np.eye(2) - np.outer(_kf_K, _kf_H)) @ _kf_P
        _kf_levels[_kf_i] = _kf_x[0]
    kalman_dev = pd.Series((_kf_close - _kf_levels) / _kf_levels, index=out.index)
    kalman_dev_mean = kalman_dev.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
    kalman_dev_std = kalman_dev.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std()
    kalman_dev_z = (kalman_dev - kalman_dev_mean) / kalman_dev_std.replace(0.0, np.nan)

    out["dem"], out["kalman_dev_z"] = dem, kalman_dev_z
    out["bottom_demarker_extreme"] = dem <= 0.10
    out["top_demarker_extreme"] = dem >= 0.90
    out["bottom_kalman_deviation_meanrev"] = kalman_dev_z <= -2.0
    out["top_kalman_deviation_meanrev"] = kalman_dev_z >= 2.0

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

    # 2026-08-30: taker_delta_z_climax/short_term_return_z are now served by trained TabPFN
    # meta-label models (live_evidence_signal_metalabel_20260829.py) whose proba is a genuine
    # prediction of an outcome up to that model's own trained HORIZON forward (2h/1h respectively)
    # -- NOT a claim that the raw trigger condition itself stays elevated-probability that long
    # (the analysis above, which found taker_delta_z_climax bottom's RAW-trigger lift falls to
    # 0.89x by offset 8-11, is answering a different question and is NOT contradicted by this).
    # Hiding a model's already-computed 2h/1h-ahead prediction after only 4 bars (20min), while
    # the outcome it actually describes is still unresolved for another 1h40m/40min, understated
    # what the model was claiming (user-identified 2026-08-30) -- these two now use their own
    # trained HORIZON as their sustain window instead. Must match each research script's HORIZON
    # constant exactly (research_eth_taker_delta_climax_metalabel_tabpfn_20260829.HORIZON=24,
    # research_eth_short_term_return_z_metalabel_tabpfn_20260829.HORIZON=12) -- live_evidence_
    # signal_metalabel_20260829.py duplicates these same two numbers for its own proba cache TTL,
    # same manual-sync convention already used for SUSTAIN_BARS itself (see also
    # live_liquidation_sweep_combo_signal_20260825.py's own copy of SUSTAIN_BARS=4).
    # dalton_rule2_balance_edge added 2026-08-30 (HORIZON=30, research_eth_dalton_rule2_balance_
    # edge_metalabel_tabpfn_20260830.py) -- same manual-sync requirement.
    # liquidity_sweep added 2026-08-30 (HORIZON=30/150min, standard touch-based-MFE redo --
    # research_eth_liquidity_sweep_topdown_metalabel_final_20260830.py -- replacing the V_REBOUND-
    # model relay bridge, which was never wired into this SUSTAIN_BARS_OVERRIDE dict in the first
    # place since it lived entirely in dashboard/server.py's own separate ACTIVE_WINDOW logic).
    SUSTAIN_BARS_OVERRIDE = {
        "taker_delta_z_climax": 24, "short_term_return_z": 12,
        "liquidity_sweep": 30, "orthogonal_combo": 24, "smt_divergence": 72,
        "fib_extension_exhaustion": 20,
        "demarker_extreme": 8, "kalman_deviation_meanrev": 12,
    }
    for name, _ in SIGNAL_ORDER:
        n_bars = SUSTAIN_BARS_OVERRIDE.get(name, SUSTAIN_BARS)
        out[f"bottom_{name}_active"] = out[f"bottom_{name}"].fillna(False).rolling(n_bars, min_periods=1).max().astype(bool)
        out[f"top_{name}_active"] = out[f"top_{name}"].fillna(False).rolling(n_bars, min_periods=1).max().astype(bool)

    # --- history-strip fill window (2026-09-01, user request) -- a SEPARATE concept from _active
    # above. _active (bottom_fired/top_fired badge, votes, net_score) is UNCHANGED: fixed bar-count
    # per signal's own trained HORIZON. This is only for the Snapshot tab's per-bar strip
    # (dashboard/server.py's bottom_history/top_history): a single raw-fire bar was too subtle to
    # read at a glance, so instead fill every bar from the fire through whichever comes first --
    # this signal's own K*ATR take-profit price (the exact same K each signal's live TabPFN
    # metalabel's tp_price uses, see live_evidence_signal_metalabel_20260829.py::_tp_price/
    # METALABEL_SIGNALS) actually being touched, or its trained HORIZON elapsing. User explicitly
    # confirmed this can fill the ENTIRE visible 48-bar/4h strip when the horizon genuinely runs
    # that long (smt_divergence's 72-bar/6h horizon exceeds it) -- no cap added, unlike the
    # rejected middle-ground design. What must still never happen (this is built directly on top of
    # eth_dashboard_evidence_signal_history_strip_sustain_window_bug_20260831's fix) is a genuine
    # second re-fire silently disappearing into one indistinguishable block -- so the true raw
    # column also rides along separately (bottom_{name}/top_{name} are already exactly that) purely
    # so the frontend can force a visible segment boundary at each actual re-fire even mid-fill
    # (app.js::toneStripSvg's rawFire param).
    #
    # K must match METALABEL_SIGNALS[name]["k"] in live_evidence_signal_metalabel_20260829.py
    # exactly -- manually synced (not imported), same reason SUSTAIN_BARS_OVERRIDE above is: that
    # module pulls in TabPFN/torch, which compute_signals() and its other callers must not be
    # forced to import just for eight float constants.
    K_OVERRIDE = {
        "taker_delta_z_climax": 2.00, "short_term_return_z": 1.75,
        "liquidity_sweep": 4.00, "orthogonal_combo": 3.571, "smt_divergence": 4.20,
        "fib_extension_exhaustion": 2.35,
        "demarker_extreme": 0.70, "kalman_deviation_meanrev": 2.5,
    }

    def _fill_until_tp_or_horizon(raw: pd.Series, k: float, horizon_bars: int, side: str) -> pd.Series:
        n = len(raw)
        filled = np.zeros(n, dtype=bool)
        raw_arr = raw.fillna(False).to_numpy()
        high_a, low_a, close_a, atr_a = high.to_numpy(), low.to_numpy(), close.to_numpy(), atr_pct.to_numpy()
        for i in np.flatnonzero(raw_arr):
            end = min(i + horizon_bars, n - 1)
            if not np.isnan(atr_a[i]):
                target = k * atr_a[i]
                level = close_a[i] * (1 - target) if side == "top" else close_a[i] * (1 + target)
                for b in range(i + 1, end + 1):
                    touched = (low_a[b] <= level) if side == "top" else (high_a[b] >= level)
                    if touched:
                        end = b
                        break
            filled[i:end + 1] = True
        return pd.Series(filled, index=raw.index)

    for name, _ in SIGNAL_ORDER:
        if name in K_OVERRIDE:
            horizon = SUSTAIN_BARS_OVERRIDE.get(name, SUSTAIN_BARS)
            out[f"bottom_{name}_fill"] = _fill_until_tp_or_horizon(out[f"bottom_{name}"], K_OVERRIDE[name], horizon, "bottom")
            out[f"top_{name}_fill"] = _fill_until_tp_or_horizon(out[f"top_{name}"], K_OVERRIDE[name], horizon, "top")
        else:
            # No metalabel K for this signal (shouldn't currently happen -- SIGNAL_ORDER's names are
            # exactly K_OVERRIDE's keys today) -- fall back to the raw column so the strip degrades
            # to single-bar blips instead of raising.
            out[f"bottom_{name}_fill"] = out[f"bottom_{name}"]
            out[f"top_{name}_fill"] = out[f"top_{name}"]

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
    funding_raw = fetch_funding_safe()
    sig = compute_signals(raw, btc_df=btc_raw, funding_df=funding_raw)
    print(render(sig))
    return 0


def run_watch(interval: int) -> int:
    if interval < MIN_WATCH_INTERVAL_S:
        log_err(f"--watch {interval}s is below the floor of {MIN_WATCH_INTERVAL_S}s (avoids hammering "
                f"Binance's public endpoint); clamping to {MIN_WATCH_INTERVAL_S}s.")
        interval = MIN_WATCH_INTERVAL_S
    print(f"[watch mode] refreshing every {interval}s (3 requests/cycle -- ETH + BTC + funding). Press Ctrl+C to stop.\n")
    try:
        while True:
            try:
                raw = fetch_klines()
                btc_raw = fetch_btc_klines_safe()
                funding_raw = fetch_funding_safe()
                sig = compute_signals(raw, btc_df=btc_raw, funding_df=funding_raw)
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
