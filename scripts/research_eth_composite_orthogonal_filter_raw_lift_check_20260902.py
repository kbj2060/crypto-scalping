#!/usr/bin/env python3
"""Composite (AND-filter) pre-check: do the orthogonal-but-weak 2026-09-02 survey candidates add
anything ON TOP of a deployed base signal, the way orthogonal_combo's oscillator x order-flow AND
does? -- user question 2026-09-02 ("복합 오실레이터 신호처럼 복합으로 만들 순 없어?").

SCOPE (why only two filters)

  A-1 Lee-Mykland is already excluded: the composite was implicitly tested in that run's
  residual/intersection decomposition (D_strz_confirmed_by_lm == "STRZ AND LM"), which improved the
  bottom side (3.06 vs 2.79) but WORSENED the top (2.39 vs 3.00) -- sign-inconsistent, this repo's
  noise signature. C-1 round-number is excluded because its offset-placebo test showed there is no
  mechanism to AND with. That leaves Corwin-Schultz/Abdi-Ranaldo (overlap 22-43%) and VPIN (36.3%)
  -- both genuinely orthogonal to the deployed signals, which is exactly the precondition that makes
  orthogonal_combo work.

THE TRAP THIS SCRIPT IS BUILT AROUND

  AND-ing ANY filter onto a base signal mechanically raises precision, because it selects a subset
  and shrinks n. This repo has been burned twice by that shape (orthogonal_combo's kept-only AUC
  overestimate; fib_extension_exhaustion's ARM=0.5 exit-structure artifact). So a raw
  "composite lift > base lift" comparison proves nothing. Two controls, both pre-registered:

  CONTROL 1 -- random-subsample null (B=200). Draw n_composite fires uniformly at random from the
    base signal's fires, recompute lift, repeat. The composite must sit in the upper tail of THAT
    distribution, not merely above the full-sample base lift. (Same discipline as
    feedback_grid_pass_count_needs_random_subsample_null_20260902.)

  CONTROL 2 -- threshold-matched base. If the goal is "fewer, better fires", the cheapest way is to
    tighten the base signal's own threshold. So: raise |ret3_z| (or sweep depth) until the base
    alone fires n_composite times, and compare. A filter that cannot beat simply tightening the
    base is not adding information -- it is just a rarity knob.

BASES     short_term_return_z (|ret3_z| >= 2.5, tightened via the z threshold)
          liquidity_sweep (tightened via sweep DEPTH past the prior 48-bar swing level)
FILTERS   cs_spread p99/p95, ar_spread p99, vpin_volclock p99/p95 (rolling-864 percentile, as in
          the A-2/A-3 pre-checks). VPIN was previously scored only against the unsigned magnitude
          target, never against pivots -- here the base supplies the direction.

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the sibling scripts.
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

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    OOS_END,
    event_study,
    load_zigzag_pivots,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
)
from research_eth_lee_mykland_jump_raw_lift_check_20260902 import ZSCORE_WINDOW, wilson_ci  # noqa: E402
from research_eth_corwin_schultz_spread_raw_lift_check_20260902 import (  # noqa: E402
    abdi_ranaldo_spread,
    load_cs_fn,
)
from research_eth_vpin_toxicity_jump_magnitude_raw_lift_check_20260902 import (  # noqa: E402
    VPIN_BUCKETS_PER_DAY,
    load_frame_with_taker,
    vpin_volume_clock,
)

PCT_WINDOW = 864
SWEEP_LOOKBACK = 48
STRZ_THRESHOLD = 2.5
N_NULL_DRAWS = 200
RNG_SEED = 20260902


def rolling_pct(s: pd.Series) -> pd.Series:
    return s.rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True)


def lift_of(pos: np.ndarray, pivot_pos: np.ndarray, all_pos: np.ndarray, K: int) -> float:
    if len(pos) == 0:
        return float("nan")
    return event_study(pos, pivot_pos, all_pos, K)["lift"]


def main() -> None:
    raw = load_frame_with_taker()
    pivots = load_zigzag_pivots()
    high, low, close = raw["high"], raw["low"], raw["close"]
    volume, taker_buy, ts = raw["volume"], raw["taker_buy_base"], raw["timestamp"]

    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    fit_mask = (ts < VAL_START).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: {int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    # ---- bases, with a continuous "strength" knob for the threshold-matched control ----
    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    sweep_lo = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_hi = (high > swing_high_prior) & (close < swing_high_prior)
    # depth = how far the wick poked past the level (the natural strength knob for a sweep)
    depth_lo = ((swing_low_prior - low) / close).where(sweep_lo)
    depth_hi = ((high - swing_high_prior) / close).where(sweep_hi)

    bases = {
        ("short_term_return_z", "bottom"): (ret3_z <= -STRZ_THRESHOLD, -ret3_z),
        ("short_term_return_z", "top"): (ret3_z >= STRZ_THRESHOLD, ret3_z),
        ("liquidity_sweep", "bottom"): (sweep_lo, depth_lo),
        ("liquidity_sweep", "top"): (sweep_hi, depth_hi),
    }

    # ---- filters ----
    cs = pd.Series(load_cs_fn()(high.to_numpy(), low.to_numpy()), index=close.index)
    ar = abdi_ranaldo_spread(high, low, close)
    bucket_volume = float(volume[fit_mask].mean()) * 288.0 / VPIN_BUCKETS_PER_DAY
    vpin = pd.Series(vpin_volume_clock(volume.to_numpy(), taker_buy.to_numpy(), bucket_volume),
                     index=close.index)
    filters = {
        "cs_p99": rolling_pct(cs) >= 0.99, "cs_p95": rolling_pct(cs) >= 0.95,
        "ar_p99": rolling_pct(ar) >= 0.99,
        "vpin_p99": rolling_pct(vpin) >= 0.99, "vpin_p95": rolling_pct(vpin) >= 0.95,
    }

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (bname, side), (bmask, strength) in bases.items():
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        base_pos = np.flatnonzero(bmask.fillna(False).to_numpy() & window_mask)
        strength_at_base = strength.to_numpy()[base_pos]
        for K_name, K in K_HORIZONS.items():
            base_lift = lift_of(base_pos, pivot_pos, all_pos, K)
            for fname, fmask in filters.items():
                comp_pos = np.flatnonzero(bmask.fillna(False).to_numpy()
                                          & fmask.fillna(False).to_numpy() & window_mask)
                n = len(comp_pos)
                if n < 20:
                    rows.append({"base": bname, "side": side, "filter": fname, "horizon": K_name,
                                 "n": n, "base_lift": round(base_lift, 2), "comp_lift": np.nan,
                                 "null_pctile": np.nan, "matched_lift": np.nan, "verdict": "n<20"})
                    continue
                st = event_study(comp_pos, pivot_pos, all_pos, K)
                comp_lift = st["lift"]
                lo, hi = wilson_ci(int(round(st["precision"] * n)), n)
                # CONTROL 1: random subsample of the base, same n
                null = np.array([lift_of(rng.choice(base_pos, size=n, replace=False), pivot_pos, all_pos, K)
                                 for _ in range(N_NULL_DRAWS)])
                pctile = float((null < comp_lift).mean() * 100)
                # CONTROL 2: tighten the base's own threshold to the same n
                order = np.argsort(-strength_at_base)          # strongest first
                matched_pos = np.sort(base_pos[order[:n]])
                matched_lift = lift_of(matched_pos, pivot_pos, all_pos, K)
                rows.append({"base": bname, "side": side, "filter": fname, "horizon": K_name, "n": n,
                             "base_lift": round(base_lift, 2), "comp_lift": round(comp_lift, 2),
                             "ci_lo_pct": round(lo * 100, 1), "ci_hi_pct": round(hi * 100, 1),
                             "null_pctile": round(pctile, 1),
                             "matched_lift": round(matched_lift, 2),
                             "beats_null95": comp_lift > np.percentile(null, 95),
                             "beats_matched": comp_lift > matched_lift})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_composite_orthogonal_filter_raw_lift_check_20260902"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 400)
    for K_name in K_HORIZONS:
        print(f"\n=== horizon {K_name} — composite vs (random-subsample null, threshold-matched base) ===")
        sub = df[df["horizon"] == K_name]
        print(sub[["base", "side", "filter", "n", "base_lift", "comp_lift", "null_pctile",
                   "matched_lift", "beats_null95", "beats_matched"]].to_string(index=False))
    surv = df[(df["beats_null95"] == True) & (df["beats_matched"] == True)]  # noqa: E712
    print(f"\n=== SURVIVORS (beat BOTH the 95th-pct random-subsample null AND the threshold-matched base) ===")
    print(surv.to_string(index=False) if len(surv) else "  NONE — 0 of "
          f"{int(df['comp_lift'].notna().sum())} evaluated cells")
    print(f"\nWrote {out_dir / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
