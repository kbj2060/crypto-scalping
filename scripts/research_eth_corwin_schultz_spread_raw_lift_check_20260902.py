#!/usr/bin/env python3
"""Raw-rule lift pre-check for the Corwin-Schultz spread proxy as a Homer evidence-signal candidate
-- docs/homer/external_literature_signal_candidates_20260902.md's A-3. Same event_study/zigzag-pivot
lift methodology as research_eth_candidate_pool_raw_lift_check_20260831.py and
research_eth_lee_mykland_jump_raw_lift_check_20260902.py (imported verbatim). Retrospective
evidence-gathering diagnostic, not a live-tradeable signal claim, not Fresh-Forward gated.

WHY THIS CANDIDATE

  All 8 deployed evidence signals are price/oscillator/flow extremes. The LIQUIDITY-STATE axis has
  been empty since dalton_rule2_balance_edge was removed (2026-08-31). Literature: Corwin-Schultz
  (2012) and Abdi-Ranaldo (2017) outperform other low-frequency liquidity proxies at explaining
  crypto liquidity's TIME-SERIES variation (irrespective of frequency/venue/HF benchmark), and
  Frontiers-in-Blockchain (2026) ranked CS 3rd of 12 microstructure features by stability selection
  (0.79) -- the top liquidity-family feature, while Amihud (0.51) and Kyle's lambda (0.56) were not
  individually significant.

  The SAME Frontiers paper is also the strongest counter-evidence: as a CONTINUOUS minute-level
  predictor at a 5-min holding horizon nothing survived fees (net Sharpe -52). So this is tested
  ONLY as a rare-event trigger, never as a continuous forecast.

THE CONFOUND THIS SCRIPT IS BUILT AROUND

  CS is computed from high-low ranges, i.e. from the same raw material as ATR / realized volatility.
  A "spread spike" could be nothing but a "volatility spike", which this repo already has everywhere.
  The 2026-08-25 taker-flow-variance-compression rejection died on exactly this failure mode (a
  -0.59..-0.83 collinearity with rolling volume turned a raw IC of -0.44 into nothing), so the
  control arm is pre-registered here rather than added after a positive result:

    hl_range_pct  -- the plain high-low log range, same threshold rule, same directional condition.
                     If CS's lift is really just volatility, this control matches or beats it.

  Plus Spearman diagnostics between cs_spread / ar_spread / hl_range / atr_pct on the eval window.

  The 2026-09-02 Lee-Mykland rejection adds the second pre-registered guard: OVERLAP against the
  deployed signals. That candidate passed on lift (2.80-3.17x) and died because 78-96% of its fires
  were within 3 bars of short_term_return_z's. Overlap vs short_term_return_z AND liquidity_sweep
  (the wick/high-low signal CS is most likely to duplicate) is therefore computed up front.

ARMS

  1. cs_spread          -- Corwin-Schultz, _corwin_schultz_spread from
                           scripts/eth_dc_financial_ml_feature_construction_20260820.py (canonical
                           two-period estimator, already in the 154-feature set), reused verbatim.
  2. ar_spread          -- Abdi-Ranaldo (2017), NOT previously in this repo. S^2 = 4(c-eta_t)(c-eta_t+1);
                           implemented lagged one bar so the estimate at bar i uses bars i-1,i only
                           (the textbook form needs bar t+1 and would be lookahead).
  3. hl_range           -- CONTROL (see above).
  4. short_term_return_z / liquidity_sweep -- deployed benchmarks + overlap references, verbatim
                           from live_evidence_signal_dashboard_20260823.py.

  Trigger rule for 1-3: rolling-864 percentile of the estimator >= PCT_PRIMARY (0.99), matching this
  repo's percentile_window=864 convention (p_fast/p_slow, atr_percentile_864). Percentile rather than
  z-score because a clipped-at-zero, heavily right-skewed series makes a rolling z ill-behaved; a
  z>=2.0 arm and a 0.95-percentile arm are reported as sensitivity (README ss5.6).

  DIRECTIONAL vs UNCONDITIONAL: a spread spike is direction-agnostic, so both framings are run.
  The unconditional arm fires the same bars for both sides (does illiquidity precede pivots at all?);
  the directional arm gates on the sign of the 2-bar return (the estimator's own window): move down
  -> bottom side, move up -> top side ("liquidity evaporated into a sell-off, expect reversion").

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the sibling scripts.
"""
from __future__ import annotations

import importlib.util
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
    load_frame,
)
from research_eth_lee_mykland_jump_raw_lift_check_20260902 import (  # noqa: E402
    OVERLAP_TOL_BARS,
    ZSCORE_WINDOW,
    overlap_stats,
    wilson_ci,
)

PCT_WINDOW = 864          # matches p_fast/p_slow/atr_percentile_864 in the deployed dashboard
PCT_PRIMARY = 0.99        # primary rare-event cutoff
PCT_SECONDARY = 0.95      # sensitivity
Z_SECONDARY = 2.0         # sensitivity (rolling-288 z, the repo's delta_z/vol_z convention)
SWEEP_LOOKBACK = 48       # liquidity_sweep prior swing high/low lookback, verbatim
STRZ_THRESHOLD = 2.5      # deployed short_term_return_z cutoff, verbatim
DIR_RETURN_BARS = 2       # sign window for the directional arm = the CS estimator's own 2-bar window


def load_cs_fn():
    """_corwin_schultz_spread from the 154-feature builder, loaded by path (that module imports
    fine standalone, but this keeps the dependency explicit and avoids executing its __main__)."""
    path = ROOT / "scripts" / "eth_dc_financial_ml_feature_construction_20260820.py"
    spec = importlib.util.spec_from_file_location("finml_features_20260902", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._corwin_schultz_spread


def abdi_ranaldo_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Abdi & Ranaldo (2017) two-day corrected-high-low spread, LAGGED ONE BAR for causality.

    Textbook: S_t^2 = 4 * (c_t - eta_t) * (c_t - eta_{t+1}), eta = (log H + log L)/2. That needs
    bar t+1, so the value at bar i here is the estimate anchored at i-1 (using bars i-1 and i) --
    known at i, no lookahead. Negative estimates clipped to 0 per the paper."""
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    s2 = 4.0 * (c - eta) * (c - eta.shift(-1))
    s2 = s2.shift(1)                       # anchor at i-1 -> value known at bar i
    return np.sqrt(s2.clip(lower=0.0))


def rolling_pct(s: pd.Series, window: int = PCT_WINDOW) -> pd.Series:
    return s.rolling(window, min_periods=window).rank(pct=True)


def rolling_z(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (s - mean) / std


def main() -> None:
    cs_fn = load_cs_fn()
    raw = load_frame()
    pivots = load_zigzag_pivots()
    high, low, close, ts = raw["high"], raw["low"], raw["close"], raw["timestamp"]

    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    cs = pd.Series(cs_fn(high.to_numpy(), low.to_numpy()), index=close.index)
    ar = abdi_ranaldo_spread(high, low, close)
    hl = np.log(high / low)
    atr_pct = (pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
                         axis=1).max(axis=1).rolling(288, min_periods=288).mean() / close)

    print(f"cs_spread: {float(cs[window_mask].gt(0).mean()) * 100:.1f}% of in-window bars > 0, "
          f"median(nonzero) {float(cs[window_mask][cs[window_mask] > 0].median()) * 1e4:.2f}bp")
    print(f"ar_spread: {float(ar[window_mask].gt(0).mean()) * 100:.1f}% of in-window bars > 0, "
          f"median(nonzero) {float(ar[window_mask][ar[window_mask] > 0].median()) * 1e4:.2f}bp")

    # --- confound diagnostic (pre-registered, see docstring) ---
    diag = pd.DataFrame({"cs": cs, "ar": ar, "hl_range": hl, "atr_pct_288": atr_pct})[window_mask].dropna()
    print("\n=== confound diagnostic: Spearman on the eval window ===")
    print(diag.corr(method="spearman").round(3).to_string())

    ret_dir = close / close.shift(DIR_RETURN_BARS) - 1.0
    down, up = (ret_dir < 0).fillna(False), (ret_dir > 0).fillna(False)

    estimators = {"cs_spread": cs, "ar_spread": ar, "hl_range_CONTROL": hl}
    arms: list[tuple[str, str, pd.Series]] = []
    for nm, series in estimators.items():
        pct = rolling_pct(series)
        fire = (pct >= PCT_PRIMARY).fillna(False)
        arms += [(f"{nm}_p99_dir", "bottom", fire & down), (f"{nm}_p99_dir", "top", fire & up)]
        arms += [(f"{nm}_p99_uncond", "bottom", fire), (f"{nm}_p99_uncond", "top", fire)]
    # sensitivity: CS only
    cs_p95 = (rolling_pct(cs) >= PCT_SECONDARY).fillna(False)
    cs_z2 = (rolling_z(cs) >= Z_SECONDARY).fillna(False)
    arms += [("cs_spread_p95_dir", "bottom", cs_p95 & down), ("cs_spread_p95_dir", "top", cs_p95 & up),
             ("cs_spread_z20_dir", "bottom", cs_z2 & down), ("cs_spread_z20_dir", "top", cs_z2 & up)]

    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    arms += [("short_term_return_z_ref", "bottom", ret3_z <= -STRZ_THRESHOLD),
             ("short_term_return_z_ref", "top", ret3_z >= STRZ_THRESHOLD),
             ("liquidity_sweep_ref", "bottom", (low < swing_low_prior) & (close > swing_low_prior)),
             ("liquidity_sweep_ref", "top", (high > swing_high_prior) & (close < swing_high_prior))]

    rows, fires = [], {}
    for name, side, trig in arms:
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        trigger_pos = np.flatnonzero(trig.fillna(False).to_numpy() & window_mask)
        fires[(name, side)] = trigger_pos
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            n, prec = stats["n_triggers"], stats["precision"]
            hits = round(prec * n) if n and np.isfinite(prec) else 0
            lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
            rows.append({"signal": name, "side": side, "horizon": k_name, "n_triggers": n,
                         "precision": prec, "ci_lo": lo, "ci_hi": hi,
                         "baseline_rate": stats["baseline_rate"], "lift": stats["lift"],
                         "recall": stats["recall"]})

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_corwin_schultz_spread_raw_lift_check_20260902"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)
    diag.corr(method="spearman").to_csv(out_dir / "confound_spearman.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for horizon in K_HORIZONS:
        print(f"\n=== horizon {horizon} ===")
        sub = df[df["horizon"] == horizon].copy()
        sub["prec_pct"] = (sub["precision"] * 100).round(1)
        sub["ci_lo_pct"] = (sub["ci_lo"] * 100).round(1)
        sub["ci_hi_pct"] = (sub["ci_hi"] * 100).round(1)
        sub["lift_x"] = sub["lift"].round(2)
        sub["recall_pct"] = (sub["recall"] * 100).round(1)
        print(sub[["signal", "side", "n_triggers", "prec_pct", "ci_lo_pct", "ci_hi_pct",
                   "lift_x", "recall_pct"]].to_string(index=False))

    print(f"\n=== overlap vs deployed signals (+-{OVERLAP_TOL_BARS} bars, same window) ===")
    ov_rows = []
    for arm in ("cs_spread_p99_dir", "ar_spread_p99_dir", "hl_range_CONTROL_p99_dir", "cs_spread_p95_dir"):
        for ref in ("short_term_return_z_ref", "liquidity_sweep_ref"):
            for side in ("bottom", "top"):
                st = overlap_stats(fires[(arm, side)], fires[(ref, side)], OVERLAP_TOL_BARS)
                ov_rows.append({"signal": arm, "vs": ref, "side": side,
                                "n_sig": len(fires[(arm, side)]), "n_ref": len(fires[(ref, side)]),
                                "jaccard_exact_pct": round(st["jaccard_exact_bar"] * 100, 1),
                                "sig_near_ref_pct": round(st["frac_a_near_b"] * 100, 1),
                                "ref_near_sig_pct": round(st["frac_b_near_a"] * 100, 1)})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out_dir / "overlap.csv", index=False)
    print(ov.to_string(index=False))
    print(f"\nWrote {out_dir}/{{scorecard,overlap,confound_spearman}}.csv")


if __name__ == "__main__":
    main()
