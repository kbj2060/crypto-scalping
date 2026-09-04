#!/usr/bin/env python3
"""Preliminary raw-rule lift check for a candidate 9th Homer evidence signal (DeMarker), before any
label design / TabPFN work -- see docs/homer/README.md. Mirrors research_eth_evidence_signal_
scorecard_ci_20260825.py's exact event_study/zigzag-pivot lift methodology (imported verbatim, same
data, same VAL+OOS window) so results are directly comparable to the other 8 signals' already-
recorded raw-rule lift numbers (e.g. orthogonal_combo 3.56x@1h, dalton_rule2_balance_edge
1.6-1.74x@1h). This is a retrospective evidence-gathering diagnostic (see analyze_eth_confluence_
oscillator_bottom_top_evidence_20260814.py's own docstring) -- NOT a live-tradeable signal claim,
not subject to the Fresh-Forward causal walk-forward rule (that governs promotion/model-selection
claims, not this raw-rule pre-check).

3 rules tested, formalized from the user's DeMarker/SMC/Wyckoff/volume-profile strategy description:

  1. dem_smc_divergence_sweep (TOP/short only): price sweeps the prior 48-bar swing high (the exact
     `sweep_high` liquidity_sweep already uses) while DeMarker is still >=0.70 (overbought) but LOWER
     than DeMarker was at the swing-high bar being swept (bearish divergence) -- "가격이 이전 고점을
     돌파하며 휩소를 만들 때 DeMarker가 과매수권역에서 더 낮은 고점을 형성".
  2. dem_wyckoff_spring_rebound (BOTTOM/long only): a sweep of the prior 48-bar swing low
     (`sweep_low`) occurred within the last 6 bars (30min, SPRING_CONFIRM_WINDOW -- a free choice,
     not specified by the user; comparable in scale to this repo's existing SUSTAIN_BARS=4/20min
     "how long counts as recent" convention) with DeMarker<=0.30 at that bar, and DeMarker has now
     crossed back above 0.30 -- "스프링 구간에서 DeMarker가 과매도에서 빠르게 반등".
  3. dem_vp_edge_exhaustion (BOTH sides): price extends beyond the trailing-200-bar volume-profile
     Value Area (VAH/VAL, core/cvp.py's `_compute_volume_profile` algorithm reused verbatim) computed
     from the PRIOR 200 bars only (current bar excluded -- deliberately NOT cvp.py's own current-
     bar-inclusive convention, to avoid the self-inclusion contamination CLAUDE.md documents for
     dalton's ATR gate: a breakout bar's own extreme volume/price must not be allowed to pull VAH/VAL
     toward itself) while DeMarker is at an extreme (top: high>vah & DeMarker>=0.90; bottom:
     low<val & DeMarker<=0.10) -- "VAH/VAL 바깥 이탈 + DeMarker 극한값 -> 평균회귀".

Also reports each rule's DeMarker-alone and structure-alone (sweep-alone / VAH-VAL-breakout-alone)
component baselines (prefixed `_`), so the table shows whether the compound condition actually adds
lift beyond either half alone -- the same question this project's ablations for orthogonal_combo/
dalton already asked of their own compound conditions.

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17 (data/eth_5m_1year.csv's actual
coverage ends exactly at OOS_END) -- identical to the reference scorecard script.
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

Z_95 = 1.959963984540054

SWEEP_LOOKBACK = 48        # matches SWEEP_LOOKBACK in live_evidence_signal_dashboard_20260823.py
DEM_N = 14                 # standard DeMarker period; matches this repo's STOCH_N=14 comparable-oscillator convention
SPRING_CONFIRM_WINDOW = 6  # 30min "quickly rebounds" window -- free choice, see module docstring
VP_LOOKBACK = 200          # matches core/cvp.py's / features/engineering.py's live default
VP_N_BINS = 50             # matches core/cvp.py's default


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI -- copied verbatim from research_eth_evidence_signal_scorecard_ci_20260825.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_cvp_volume_profile_fn():
    """core/cvp.py::_compute_volume_profile, loaded bypassing core/__init__.py (which hard-imports
    the `binance` package, not installed in this env) -- same spec_from_file_location idiom already
    used in this repo by research_eth_sweep_v_rebound_shallow_xlstm_20260829.py::load_tier0_builder()."""
    spec = importlib.util.spec_from_file_location("cvp_standalone_20260831", ROOT / "core" / "cvp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_volume_profile


def compute_demarker(high: pd.Series, low: pd.Series, n: int = DEM_N) -> pd.Series:
    up_move = high.diff()
    down_move = low.shift(1) - low
    de_max = up_move.clip(lower=0.0).fillna(0.0)
    de_min = down_move.clip(lower=0.0).fillna(0.0)
    sma_max = de_max.rolling(n, min_periods=n).mean()
    sma_min = de_min.rolling(n, min_periods=n).mean()
    return sma_max / (sma_max + sma_min).replace(0.0, np.nan)


def compute_prior_extreme_positions(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Position of the bar that set the trailing `window`-bar swing high/low ending at i-1 --
    copied verbatim from live_evidence_signal_dashboard_20260823.py::compute_signals()'s
    fib_extension_exhaustion leg-direction block (high_pos/low_pos)."""
    n = len(high)
    low_pos = np.full(n, -1, dtype=np.int64)
    high_pos = np.full(n, -1, dtype=np.int64)
    if n > window:
        lo_windows = np.lib.stride_tricks.sliding_window_view(low, window)
        hi_windows = np.lib.stride_tricks.sliding_window_view(high, window)
        idx = np.arange(window, n)
        j = idx - window
        low_pos[idx] = j + lo_windows[j].argmin(axis=1)
        high_pos[idx] = j + hi_windows[j].argmax(axis=1)
    return high_pos, low_pos


def compute_rolling_vah_val(hlc3: np.ndarray, volume: np.ndarray, start_pos: int, end_pos: int,
                             lookback: int, n_bins: int, vp_fn) -> tuple[np.ndarray, np.ndarray]:
    """VAH/VAL at each bar in [start_pos, end_pos), computed from the PRIOR `lookback` bars only
    (window i-lookback..i-1, current bar excluded -- see module docstring for why). Restricted to
    [start_pos, end_pos) rather than the full 2023-2026 history purely for runtime -- event_study
    only ever looks at the VAL+OOS window anyway."""
    n = end_pos - start_pos
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    for out_i, i in enumerate(range(start_pos, end_pos)):
        lo = i - lookback
        _, _, _, v_high, v_low = vp_fn(hlc3[lo:i], volume[lo:i], n_bins)
        vah[out_i] = v_high
        val[out_i] = v_low
    return vah, val


def main() -> None:
    vp_fn = load_cvp_volume_profile_fn()
    raw = load_frame()
    pivots = load_zigzag_pivots()

    high, low, close, volume = raw["high"], raw["low"], raw["close"], raw["volume"]
    hlc3 = (high + low + close) / 3.0

    dem = compute_demarker(high, low)

    swing_low_prior = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high_prior = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)
    sweep_low = (low < swing_low_prior) & (close > swing_low_prior)
    sweep_high = (high > swing_high_prior) & (close < swing_high_prior)

    high_pos, _ = compute_prior_extreme_positions(high.to_numpy(), low.to_numpy(), SWEEP_LOOKBACK)
    dem_arr = dem.to_numpy()
    prior_high_dem = np.full(len(dem_arr), np.nan)
    valid_h = high_pos >= 0
    prior_high_dem[valid_h] = dem_arr[high_pos[valid_h]]
    prior_high_dem = pd.Series(prior_high_dem, index=dem.index)

    ts = raw["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    first_pos, last_pos = int(all_pos.min()), int(all_pos.max()) + 1
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    # --- Rule 1: DeMarker bearish divergence at a resistance sweep (TOP only) ---
    rule1_top = sweep_high & (dem >= 0.70) & (dem < prior_high_dem)

    # --- Rule 2: Wyckoff spring -- oversold sweep of support, then a quick rebound (BOTTOM only) ---
    spring_oversold = sweep_low & (dem <= 0.30)
    recent_spring_oversold = spring_oversold.rolling(SPRING_CONFIRM_WINDOW, min_periods=1).max().fillna(0).astype(bool)
    dem_exits_oversold = (dem.shift(1) <= 0.30) & (dem > 0.30)
    rule2_bottom = recent_spring_oversold & dem_exits_oversold

    # --- Rule 3: Volume-profile edge exhaustion (BOTH sides) ---
    vah_arr, val_arr = compute_rolling_vah_val(hlc3.to_numpy(), volume.to_numpy(), first_pos, last_pos,
                                                VP_LOOKBACK, VP_N_BINS, vp_fn)
    vah_level = pd.Series(np.nan, index=dem.index)
    val_level = pd.Series(np.nan, index=dem.index)
    vah_level.iloc[first_pos:last_pos] = vah_arr
    val_level.iloc[first_pos:last_pos] = val_arr
    rule3_top = (high > vah_level) & (dem >= 0.90)
    rule3_bottom = (low < val_level) & (dem <= 0.10)

    # --- component-alone baselines (context only, not separate rule proposals) ---
    triggers = [
        ("dem_smc_divergence_sweep", "top", rule1_top),
        ("dem_wyckoff_spring_rebound", "bottom", rule2_bottom),
        ("dem_vp_edge_exhaustion", "top", rule3_top),
        ("dem_vp_edge_exhaustion", "bottom", rule3_bottom),
        ("_dem_alone_overbought_oversold", "top", dem >= 0.70),
        ("_dem_alone_overbought_oversold", "bottom", dem <= 0.30),
        ("_dem_alone_extreme", "top", dem >= 0.90),
        ("_dem_alone_extreme", "bottom", dem <= 0.10),
        ("_vp_breakout_alone", "top", high > vah_level),
        ("_vp_breakout_alone", "bottom", low < val_level),
        ("_sweep_alone", "top", sweep_high),
        ("_sweep_alone", "bottom", sweep_low),
    ]

    rows = []
    for name, side, trigger_series in triggers:
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        trigger_pos = np.flatnonzero(trigger_series.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            n, prec = stats["n_triggers"], stats["precision"]
            hits = round(prec * n) if n and np.isfinite(prec) else 0
            lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
            rows.append({
                "signal": name, "side": side, "horizon": k_name,
                "n_triggers": n, "precision": prec, "ci_lo": lo, "ci_hi": hi,
                "baseline_rate": stats["baseline_rate"], "lift": stats["lift"],
                "recall": stats["recall"],
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_demarker_evidence_signal_lift_check_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    for horizon in K_HORIZONS:
        print(f"\n=== horizon {horizon} ===")
        sub = df[df["horizon"] == horizon].copy()
        sub["precision_pct"] = (sub["precision"] * 100).round(1)
        sub["ci_lo_pct"] = (sub["ci_lo"] * 100).round(1)
        sub["ci_hi_pct"] = (sub["ci_hi"] * 100).round(1)
        sub["baseline_pct"] = (sub["baseline_rate"] * 100).round(1)
        sub["lift_x"] = sub["lift"].round(2)
        cols = ["signal", "side", "n_triggers", "precision_pct", "ci_lo_pct", "ci_hi_pct", "baseline_pct", "lift_x"]
        print(sub[cols].to_string(index=False))

    print(f"\nWrote {out_dir / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
