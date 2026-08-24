#!/usr/bin/env python3
"""Evidence study (NOT a trading algorithm, NOT a promotion claim) for 4 more discretionary
orderflow/auction-theory trader frameworks, follow-up to
analyze_eth_yush_orderflow_component_evidence_20260815.py (same harness, same causal-level
construction pattern, same retrospective zigzag-pivot ground truth). Same methodology note
applies: looking forward from a trigger bar to a real pivot is the retrospective study design
itself, not a fresh-forward/live claim.

Frameworks and why these 4 (chosen to be genuinely DIFFERENT geometric objects from what the
Yush study and the master evidence scorecard already covered, not relabelings of
liquidity-sweep / single-bar delta / single-bar absorption / session-levels / value-area-edge,
all of which were already measured and are cited here instead of re-derived):

  AMT (Jim Dalton, "the godfather of Market Profile" -- this is literally the theory Yush's LAF
  model cites) -- Dalton's 3 rules are about a MULTI-bar balance/rotation regime, not a single
  level touch:
    B1 excess_tail_deep   -- Dalton's "excess" = a single-print tail that pokes meaningfully
                             BEYOND the prior range and fails, magnitude-gated (>=1.0x ATR poke
                             depth) vs the already-tested plain liquidity_sweep (any poke depth,
                             3.01x lift in the master scorecard) -- tests whether poke DEPTH is
                             the active ingredient or whether "any poke that closes back in" is
                             already the whole story.
    B2 balance_edge_reject-- Dalton Rule 2 ("inside balance, edges get rejected") gated on an
                             actual LOW-VOLATILITY REGIME (rolling ATR%-percentile <= 0.30), vs
                             Yush's Y3 value-area-edge test which had NO regime gate and scored
                             0.96/0.81 (no edge). Direct test of whether the regime gate is the
                             missing ingredient.
    B3 balance_breakout_continuation -- Dalton Rule 3 ("accept outside balance -> imbalanced,
                             seeks new value"). This is a CONTINUATION claim, tested with the
                             race-outcome harness (reused, not reimplemented), not a reversal
                             pivot test.

  VSA (Tom Williams, Wyckoff's volume+spread reading formalized) -- the polarity flip: every
  previously-tested winning signal in this repo's scorecard fires on HIGH volume. VSA's core
  claim is that LOW volume (absence of effort) is itself informative:
    C1 no_demand / no_supply -- up-bar with volume <= 50% of its 20-bar average AND spread
                                (high-low) <= 40% of its 20-bar average spread (bearish warning);
                                mirror for no_supply (bullish). Literal textbook VSA thresholds.

  Footprint / stacked-imbalance trading (per-price-level bid/ask ratio, order-flow-platform
  literature) -- the repo has no per-price-level book history, so "3 consecutive price levels at
  4:1 imbalance" cannot be reconstructed. The bar-level analogue actually available and NOT yet
  tested is PERSISTENCE across consecutive whole bars (distinct from the already-scored
  single-bar taker-flow-surge signal, which fires on ONE extreme bar):
    D1 persistent_delta   -- N=3 consecutive 5m bars (15 min) all one-sided taker_buy_ratio
                             (>=0.55 or <=0.45) in the same direction.

  iFVG (Dodgy DD / ICT-derivative, ~41k free community, verified funded-trader profile similar
  to Yush) -- a genuinely different geometric object (a 3-bar price GAP, not a volume/delta
  event or a fixed level):
    E1 fvg_zone_touch     -- price re-enters an unmitigated 3-bar Fair Value Gap zone (classic
                             ICT definition: bar[i-2].high < bar[i].low = bullish FVG price
                             void; mirror for bearish) formed within the last 48 bars.
    E2 fvg_inversion      -- the literal Dodgy DD setup: liquidity sweep (reusing this repo's
                             existing causal 48-bar swing sweep definition, sweep_low/sweep_high,
                             unmodified) followed within 12 bars by an FVG forming in the
                             reversal direction, and CONFIRMED by a later bar's close (not wick)
                             crossing back through the gap -- "a wick piercing the gap is
                             insufficient; it implies rejection, not inversion" per the source
                             material.

Causality: every rolling stat and level uses only bars up to and including the trigger bar
(rolling().shift(1) where a "prior" object is needed); FVG zones use only the 3 bars that form
them, all at or before the touch/inversion bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_broad_evidence_signal_sweep_20260814 import load_raw, race_outcomes  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)

ETH_PATH = ROOT / "data" / "eth_5m_1year.csv"
OUT_DIR = ROOT / "tmp" / "eth_amt_vsa_footprint_ifvg_component_evidence_20260815"


def load_frame() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]
    df = pd.read_csv(ETH_PATH, usecols=cols, parse_dates=["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def add_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Identical definition to the repo's existing A3_liquidity_sweep (48-bar causal swing)."""
    f = frame
    swing_low_prior = f["low"].rolling(48, min_periods=48).min().shift(1)
    swing_high_prior = f["high"].rolling(48, min_periods=48).max().shift(1)
    f["swing_low_prior"], f["swing_high_prior"] = swing_low_prior, swing_high_prior
    f["sweep_low"] = (f["low"] < swing_low_prior) & (f["close"] > swing_low_prior)
    f["sweep_high"] = (f["high"] > swing_high_prior) & (f["close"] < swing_high_prior)
    return f


def add_amt_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame
    poke_depth_low = (f["swing_low_prior"] - f["low"]) / f["atr_price"].replace(0.0, np.nan)
    poke_depth_high = (f["high"] - f["swing_high_prior"]) / f["atr_price"].replace(0.0, np.nan)
    f["excess_tail_bottom"] = f["sweep_low"] & (poke_depth_low >= 1.0)
    f["excess_tail_top"] = f["sweep_high"] & (poke_depth_high >= 1.0)

    atr_pctile = f["atr_pct"].rolling(288, min_periods=144).rank(pct=True)
    f["low_vol_regime"] = atr_pctile <= 0.30
    range_low = f["low"].rolling(48, min_periods=48).min()
    range_high = f["high"].rolling(48, min_periods=48).max()
    tol = 0.15 * (range_high - range_low)
    f["balance_edge_low"] = f["low_vol_regime"] & ((f["low"] - range_low).abs() <= tol)
    f["balance_edge_high"] = f["low_vol_regime"] & ((range_high - f["high"]).abs() <= tol)

    prior_low_vol = f["low_vol_regime"].shift(1).fillna(False)
    f["balance_breakout_up"] = prior_low_vol & (f["close"] > range_high.shift(1))
    f["balance_breakout_down"] = prior_low_vol & (f["close"] < range_low.shift(1))
    return f


def add_vsa_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame
    avg_vol = f["volume"].rolling(20, min_periods=20).mean()
    avg_spread = (f["high"] - f["low"]).rolling(20, min_periods=20).mean()
    spread = f["high"] - f["low"]
    low_vol_bar = (f["volume"] <= 0.5 * avg_vol) & (spread <= 0.4 * avg_spread)
    f["no_demand"] = low_vol_bar & (f["close"] > f["open"])
    f["no_supply"] = low_vol_bar & (f["close"] < f["open"])
    return f


def add_footprint_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame
    ratio = f["taker_buy_quote"] / f["quote_volume"].replace(0.0, np.nan)
    buy_heavy = ratio >= 0.55
    sell_heavy = ratio <= 0.45
    f["persistent_delta_up"] = buy_heavy & buy_heavy.shift(1).fillna(False) & buy_heavy.shift(2).fillna(False)
    f["persistent_delta_down"] = sell_heavy & sell_heavy.shift(1).fillna(False) & sell_heavy.shift(2).fillna(False)
    return f


def add_ifvg_features(frame: pd.DataFrame, lookback: int = 48, confirm_k: int = 12) -> pd.DataFrame:
    """3-bar Fair Value Gap zones (classic ICT def), touch of an unmitigated zone, and the
    Dodgy-DD-style "sweep -> FVG forms -> body-close inversion" sequence."""
    f = frame
    h, l, c = f["high"].to_numpy(), f["low"].to_numpy(), f["close"].to_numpy()
    n = len(f)

    bull_gap_lo = np.full(n, np.nan)   # zone = [bull_gap_lo, bull_gap_hi], formed at i (uses i-2..i)
    bull_gap_hi = np.full(n, np.nan)
    bear_gap_lo = np.full(n, np.nan)
    bear_gap_hi = np.full(n, np.nan)
    for i in range(2, n):
        if h[i - 2] < l[i]:
            bull_gap_lo[i], bull_gap_hi[i] = h[i - 2], l[i]
        if l[i - 2] > h[i]:
            bear_gap_lo[i], bear_gap_hi[i] = h[i], l[i - 2]

    touch_bull = np.zeros(n, dtype=bool)
    touch_bear = np.zeros(n, dtype=bool)
    invert_bull_to_bear = np.zeros(n, dtype=bool)   # bullish gap's low gets body-closed-through -> bearish signal
    invert_bear_to_bull = np.zeros(n, dtype=bool)   # bearish gap's high gets body-closed-through -> bullish signal

    open_bull: list[tuple[int, float, float]] = []   # (formed_at, lo, hi)
    open_bear: list[tuple[int, float, float]] = []
    for i in range(n):
        if not np.isnan(bull_gap_lo[i]):
            open_bull.append((i, float(bull_gap_lo[i]), float(bull_gap_hi[i])))
        if not np.isnan(bear_gap_lo[i]):
            open_bear.append((i, float(bear_gap_lo[i]), float(bear_gap_hi[i])))
        open_bull = [z for z in open_bull if i - z[0] <= lookback]
        open_bear = [z for z in open_bear if i - z[0] <= lookback]

        for formed_at, lo, hi in open_bull:
            if formed_at == i:
                continue
            if lo <= l[i] <= hi or lo <= h[i] <= hi or (l[i] <= lo and h[i] >= hi):
                touch_bull[i] = True
            body_lo, body_hi = min(f["open"].iat[i], c[i]), max(f["open"].iat[i], c[i])
            if i - formed_at <= confirm_k and body_hi < lo:
                invert_bull_to_bear[i] = True

        for formed_at, lo, hi in open_bear:
            if formed_at == i:
                continue
            if lo <= l[i] <= hi or lo <= h[i] <= hi or (l[i] <= lo and h[i] >= hi):
                touch_bear[i] = True
            body_lo, body_hi = min(f["open"].iat[i], c[i]), max(f["open"].iat[i], c[i])
            if i - formed_at <= confirm_k and body_lo > hi:
                invert_bear_to_bull[i] = True

    f["fvg_touch_bull"] = touch_bull   # price re-entering a bullish (support) void -> tested as a bottom signal
    f["fvg_touch_bear"] = touch_bear   # bearish (resistance) void -> top signal
    f["fvg_invert_bearish_signal"] = invert_bull_to_bear
    f["fvg_invert_bullish_signal"] = invert_bear_to_bull

    sweep_low_recent = f["sweep_low"].rolling(12, min_periods=1).max().astype(bool)
    sweep_high_recent = f["sweep_high"].rolling(12, min_periods=1).max().astype(bool)
    f["sweep_then_ifvg_bottom"] = invert_bear_to_bull & sweep_low_recent
    f["sweep_then_ifvg_top"] = invert_bull_to_bear & sweep_high_recent
    return f


def reversal_components(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "B1_excess_tail_deep": frame["excess_tail_bottom"],
            "B2_balance_edge_reject": frame["balance_edge_low"],
            "C1_vsa_no_supply": frame["no_supply"],
            "D1_persistent_delta_buy": frame["persistent_delta_up"],
            "E1_fvg_touch": frame["fvg_touch_bull"],
            "E2_fvg_inversion": frame["fvg_invert_bullish_signal"],
            "E3_sweep_then_ifvg": frame["sweep_then_ifvg_bottom"],
        }
    return {
        "B1_excess_tail_deep": frame["excess_tail_top"],
        "B2_balance_edge_reject": frame["balance_edge_high"],
        "C1_vsa_no_demand": frame["no_demand"],
        "D1_persistent_delta_sell": frame["persistent_delta_down"],
        "E1_fvg_touch": frame["fvg_touch_bear"],
        "E2_fvg_inversion": frame["fvg_invert_bearish_signal"],
        "E3_sweep_then_ifvg": frame["sweep_then_ifvg_top"],
    }


def run_reversal(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    pivot_pos = frame.index[frame["timestamp"].isin(pivots.loc[pivots["pivot_type"] == side, "timestamp"])].to_numpy()
    rows = []
    for name, mask in reversal_components(frame, side).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"side": side, "signal": name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"]})
    return pd.DataFrame(rows)


def run_balance_breakout_continuation(frame: pd.DataFrame, window_mask: np.ndarray) -> pd.DataFrame:
    close, high, low = frame["close"].to_numpy(), frame["high"].to_numpy(), frame["low"].to_numpy()
    atr_pct = frame["atr_pct"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    rows = []
    for name, mask, d in (("B3_balance_breakout_up", frame["balance_breakout_up"], 1),
                          ("B3_balance_breakout_down", frame["balance_breakout_down"], -1)):
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            base_out = race_outcomes(all_pos, d, close, high, low, atr_pct, K)
            base_decided = base_out != 0
            base_rate = float((base_out[base_decided] == 1).mean()) if base_decided.any() else float("nan")
            out = race_outcomes(trigger_pos, d, close, high, low, atr_pct, K)
            decided = out != 0
            hit_rate = float((out[decided] == 1).mean()) if decided.any() else float("nan")
            lift = hit_rate / base_rate if base_rate and np.isfinite(base_rate) and base_rate > 0 else float("nan")
            rows.append({"signal": name, "horizon": k_name, "n_triggers": int(len(trigger_pos)),
                         "precision": hit_rate, "baseline_rate": base_rate, "lift": lift})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    frame = compute_indicators(raw).reset_index(drop=True)
    frame = add_sweep(frame)
    frame = add_amt_features(frame)
    frame = add_vsa_features(frame)
    frame = add_footprint_proxy(frame)
    frame = add_ifvg_features(frame)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    rows = pd.concat([run_reversal(frame, window_mask, pivots, s) for s in ("bottom", "top")], ignore_index=True)
    cont = run_balance_breakout_continuation(frame, window_mask)

    pd.set_option("display.width", 180)
    cols = ["signal", "n_triggers", "precision", "baseline_rate", "lift", "recall", "excess_move_mean_pct"]
    for side in ("bottom", "top"):
        print(f"\n\n########## REVERSAL: {side.upper()} ##########")
        sub = rows[rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- {horizon} --")
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    print("\n\n########## B3: BALANCE-BREAKOUT CONTINUATION (AMT Rule 3, race outcome) ##########")
    print(cont[["signal", "horizon", "n_triggers", "precision", "baseline_rate", "lift"]].to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "reversal_component_evidence_table.csv", index=False)
    cont.to_csv(OUT_DIR / "balance_breakout_continuation_table.csv", index=False)
    print(f"\nWrote tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
