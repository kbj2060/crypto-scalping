#!/usr/bin/env python3
"""Raw-rule lift pre-check for 5 candidates in the Homer '후보 풀' (candidate pool) --
docs/homer/README.md's '## 후보 풀' section (added 2026-08-31 alongside DeMarker, see memory
eth_homer_candidate_pool_established_20260831 for the selection/exclusion rationale). Same
event_study/zigzag-pivot lift methodology as research_eth_evidence_signal_scorecard_ci_20260825.py
and research_eth_demarker_evidence_signal_lift_check_20260831.py (imported verbatim) -- this is a
retrospective evidence-gathering diagnostic, not a live-tradeable signal claim, not Fresh-Forward
gated (see either prior script's docstring for why).

5 candidates, each formalized from the user's strategy description:

  1. vpoc_pinball_reject (BOTH sides): price tests the trailing-200-bar (prior bars only, current
     bar excluded -- same self-inclusion-contamination reasoning as the DeMarker script's VAH/VAL)
     volume-profile POC from one side and closes back on the same side without sustaining the
     break -- a POC-anchored mirror of liquidity_sweep's sweep_high/sweep_low touch-and-reject.
  2. renko_reversal_brick (BOTH sides): classic close-only percentage Renko (brick = 0.5% of the
     last brick's close, RENKO_BRICK_PCT -- free choice, not user-specified; 1-brick reversal
     convention, the textbook original, not the "2-brick" variant some platforms default to).
     Trigger fires on the bar whose close completes the FIRST brick in a new direction after an
     established opposite trend. Built from the close-price sequence only (no intrabar path
     assumed -- avoids the optimistic/pessimistic-intrabar-ordering ambiguity this repo has
     flagged elsewhere for barrier-style logic).
  3. btc_eth_ratio_zscore_meanrev (BOTH sides): ETH/BTC close-price ratio, rolling-288-bar
     z-scored -- the exact same recipe this repo already uses for delta_z/vol_z/ret3_z/funding_z
     (rolling mean/std, +-2.0 threshold). Extreme z (ETH rich/cheap vs BTC) is tested against
     ETH's OWN zigzag pivots, not a simultaneous two-leg trade -- this is a directional-lift
     pre-check, not a pairs-trade backtest.
  4. kalman_deviation_meanrev (BOTH sides): features/engineering.py::_kalman_trend_velocity's
     exact state-space model (F/H/Q/R, obs_noise=1e-3/proc_noise=1e-5) extended to also keep the
     level state x[0] (the live feature only returns velocity x[1], discarding the level) --
     (close-level)/level, z-scored with the same rolling-288 recipe as #3.
  5. tpo_single_print_reentry (BOTH sides): 30-min (6-bar) TPO periods over UTC calendar days,
     PRIOR COMPLETED day only (no intraday lookahead -- today's price is checked against
     yesterday's already-finalized profile). A price level touched by exactly one period that day
     is a "single print"; today's price re-entering one of yesterday's single-print levels (top
     half of yesterday's range -> top side, bottom half -> bottom side) is the trigger. Tests
     "does a thin/unstable zone get revisited and resolve nearby" -- a simplification of the
     user's "vacuum fill" mechanic, which describes a continuation-through move that doesn't map
     cleanly onto the pivot-prediction framework every other signal in this lineage uses (same
     issue as VPOC's "magnet" sub-case, which is why vpoc_pinball_reject tests rejection instead).

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the other two
scorecard scripts in this lineage (data/eth_5m_1year.csv's actual coverage ends exactly at OOS_END).
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
ZSCORE_WINDOW = 288  # matches live_evidence_signal_dashboard_20260823.py's ZSCORE_WINDOW
VP_LOOKBACK = 200     # matches core/cvp.py's / features/engineering.py's live default
VP_N_BINS = 50        # matches core/cvp.py's default
RENKO_BRICK_PCT = 0.005    # free choice, not user-specified -- see module docstring
TPO_PERIOD_BARS = 6        # 30min at 5m bars, standard TPO period length
TPO_NUM_LEVELS = 100       # free choice, not user-specified -- see module docstring


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI -- copied verbatim from research_eth_evidence_signal_scorecard_ci_20260825.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def rolling_zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (s - mean) / std


# ---------------------------------------------------------------------------
# 1. VPOC pinball reject
# ---------------------------------------------------------------------------
def load_cvp_volume_profile_fn():
    """core/cvp.py::_compute_volume_profile, loaded bypassing core/__init__.py (which hard-imports
    the `binance` package, not installed in this env) -- same idiom already used in this repo by
    research_eth_sweep_v_rebound_shallow_xlstm_20260829.py::load_tier0_builder() and reused by
    research_eth_demarker_evidence_signal_lift_check_20260831.py."""
    spec = importlib.util.spec_from_file_location("cvp_standalone_20260831b", ROOT / "core" / "cvp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_volume_profile


def compute_rolling_vah_val_poc(hlc3: np.ndarray, volume: np.ndarray, start_pos: int, end_pos: int,
                                 lookback: int, n_bins: int, vp_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VAH/VAL/POC at each bar in [start_pos, end_pos), computed from the PRIOR `lookback` bars
    only (window i-lookback..i-1, current bar excluded) -- avoids a breakout/reject bar's own
    extreme volume/price pulling the level toward itself."""
    n = end_pos - start_pos
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    poc = np.full(n, np.nan)
    for out_i, i in enumerate(range(start_pos, end_pos)):
        lo = i - lookback
        _, _, v_poc, v_high, v_low = vp_fn(hlc3[lo:i], volume[lo:i], n_bins)
        vah[out_i] = v_high
        val[out_i] = v_low
        poc[out_i] = v_poc
    return vah, val, poc


# ---------------------------------------------------------------------------
# 2. Renko reversal brick
# ---------------------------------------------------------------------------
def compute_renko_reversals(close: np.ndarray, brick_pct: float = RENKO_BRICK_PCT) -> tuple[np.ndarray, np.ndarray]:
    """Classic close-only percentage Renko, 1-brick reversal. Brick size is recomputed each step
    as brick_pct * last_brick_close (keeps brick size proportional across ETH's multi-year price
    range instead of a fixed-$ brick that would be inconsistent across price regimes)."""
    n = len(close)
    reversal_up = np.zeros(n, dtype=bool)
    reversal_down = np.zeros(n, dtype=bool)
    last_brick_close = close[0]
    direction = 0
    n_bricks_total = 0
    for i in range(1, n):
        price = close[i]
        brick_size = brick_pct * last_brick_close
        if price >= last_brick_close + brick_size:
            n_bricks = int((price - last_brick_close) // brick_size)
            last_brick_close += n_bricks * brick_size
            if direction == -1:
                reversal_up[i] = True
            direction = 1
            n_bricks_total += n_bricks
        elif price <= last_brick_close - brick_size:
            n_bricks = int((last_brick_close - price) // brick_size)
            last_brick_close -= n_bricks * brick_size
            if direction == 1:
                reversal_down[i] = True
            direction = -1
            n_bricks_total += n_bricks
    print(f"  [renko] {n_bricks_total} total bricks formed over {n} bars, "
          f"{int(reversal_up.sum())} up-reversals / {int(reversal_down.sum())} down-reversals (full history)")
    return reversal_up, reversal_down


# ---------------------------------------------------------------------------
# 4. Kalman deviation
# ---------------------------------------------------------------------------
def kalman_level_and_velocity(close: np.ndarray, obs_noise: float = 1e-3,
                               proc_noise: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Extends features/engineering.py::_kalman_trend_velocity (identical F/H/Q/R, identical
    obs_noise/proc_noise defaults) to also return the level state x[0] -- the live feature only
    keeps velocity (x[1]) and discards the level every step."""
    n = len(close)
    F = np.array([[1., 1.], [0., 1.]])
    H = np.array([[1., 0.]])
    Q = np.eye(2) * proc_noise
    R = np.array([[obs_noise]])
    x = np.array([close[0], 0.0])
    P = np.eye(2)
    levels = np.empty(n)
    velocities = np.empty(n)
    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T).flatten() / S
        inn = close[i] - (H @ x)[0]
        x = x + K * inn
        P = (np.eye(2) - np.outer(K, H)) @ P
        levels[i] = x[0]
        velocities[i] = x[1]
    return levels, velocities


# ---------------------------------------------------------------------------
# 5. TPO single print
# ---------------------------------------------------------------------------
def compute_daily_single_print_zones(day_low: float, day_high: float, lows: np.ndarray, highs: np.ndarray,
                                      num_levels: int = TPO_NUM_LEVELS,
                                      period_bars: int = TPO_PERIOD_BARS) -> tuple[np.ndarray, np.ndarray]:
    if day_high - day_low < 1e-9 or len(lows) < period_bars:
        return np.array([]), np.array([])
    levels = np.linspace(day_low, day_high, num_levels)
    period_id = np.arange(len(lows)) // period_bars
    touch_count = np.zeros(num_levels, dtype=int)
    for p in range(period_id.max() + 1):
        mask = period_id == p
        p_low, p_high = lows[mask].min(), highs[mask].max()
        touch_count += ((levels >= p_low) & (levels <= p_high)).astype(int)
    single = touch_count == 1
    mid = (day_low + day_high) / 2.0
    return levels[single & (levels > mid)], levels[single & (levels <= mid)]


def compute_tpo_reentry_triggers(raw: pd.DataFrame, first_pos: int, last_pos: int) -> tuple[np.ndarray, np.ndarray]:
    date = raw["timestamp"].dt.floor("D")
    zone_start = date.iloc[first_pos] - pd.Timedelta(days=1)
    zone_end = date.iloc[last_pos - 1]
    relevant_dates = date[(date >= zone_start) & (date <= zone_end)].unique()

    zones_by_date: dict = {}
    n_zero_zone_days = 0
    total_top, total_bottom = 0, 0
    for d in relevant_dates:
        day_mask = (date == d).to_numpy()
        lows, highs = raw.loc[day_mask, "low"].to_numpy(), raw.loc[day_mask, "high"].to_numpy()
        top_lv, bot_lv = compute_daily_single_print_zones(lows.min(), highs.max(), lows, highs)
        zones_by_date[d] = (top_lv, bot_lv)
        if len(top_lv) == 0 and len(bot_lv) == 0:
            n_zero_zone_days += 1
        total_top += len(top_lv)
        total_bottom += len(bot_lv)
    n_days = len(relevant_dates)
    print(f"  [tpo] {n_days} days profiled, {n_zero_zone_days} with zero single-print levels, "
          f"avg {total_top / max(n_days, 1):.1f} top-side / {total_bottom / max(n_days, 1):.1f} bottom-side levels/day")

    n = last_pos - first_pos
    tpo_top = np.zeros(n, dtype=bool)
    tpo_bottom = np.zeros(n, dtype=bool)
    prev_date = (date - pd.Timedelta(days=1)).to_numpy()
    high_arr, low_arr = raw["high"].to_numpy(), raw["low"].to_numpy()
    for out_i, i in enumerate(range(first_pos, last_pos)):
        zones = zones_by_date.get(prev_date[i])
        if zones is None:
            continue
        top_lv, bot_lv = zones
        if len(top_lv):
            tpo_top[out_i] = bool(((top_lv >= low_arr[i]) & (top_lv <= high_arr[i])).any())
        if len(bot_lv):
            tpo_bottom[out_i] = bool(((bot_lv >= low_arr[i]) & (bot_lv <= high_arr[i])).any())
    return tpo_top, tpo_bottom


def main() -> None:
    vp_fn = load_cvp_volume_profile_fn()
    raw = load_frame()
    pivots = load_zigzag_pivots()

    high, low, close, volume = raw["high"], raw["low"], raw["close"], raw["volume"]
    hlc3 = (high + low + close) / 3.0

    ts = raw["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    first_pos, last_pos = int(all_pos.min()), int(all_pos.max()) + 1
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    # --- 1. VPOC pinball reject ---
    vah_arr, val_arr, poc_arr = compute_rolling_vah_val_poc(hlc3.to_numpy(), volume.to_numpy(),
                                                             first_pos, last_pos, VP_LOOKBACK, VP_N_BINS, vp_fn)
    poc_level = pd.Series(np.nan, index=close.index)
    poc_level.iloc[first_pos:last_pos] = poc_arr
    vpoc_bottom = (close.shift(1) > poc_level) & (low <= poc_level) & (close > poc_level)
    vpoc_top = (close.shift(1) < poc_level) & (high >= poc_level) & (close < poc_level)
    print(f"  [vpoc] {int(vpoc_bottom[window_mask].sum())} bottom-side / {int(vpoc_top[window_mask].sum())} top-side reject triggers in-window")

    # --- 2. Renko reversal brick (needs full history for correct sequential brick state) ---
    renko_up, renko_down = compute_renko_reversals(close.to_numpy())

    # --- 3. BTC/ETH ratio z-score ---
    btc_raw = pd.read_csv(ROOT / "data" / "btc_5m_1year.csv", usecols=["timestamp", "close"],
                          parse_dates=["timestamp"]).rename(columns={"close": "btc_close"})
    btc_aligned = raw[["timestamp"]].merge(btc_raw, on="timestamp", how="left")
    ratio = close / btc_aligned["btc_close"]
    ratio_z = rolling_zscore(ratio)
    ratio_bottom = ratio_z <= -2.0   # ETH cheap vs BTC -> predicts ETH reversal up
    ratio_top = ratio_z >= 2.0       # ETH rich vs BTC -> predicts ETH reversal down

    # --- 4. Kalman deviation mean-reversion (needs full history for filter convergence) ---
    levels, _velocities = kalman_level_and_velocity(close.to_numpy())
    kalman_dev = pd.Series((close.to_numpy() - levels) / levels, index=close.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    kalman_bottom = kalman_dev_z <= -2.0
    kalman_top = kalman_dev_z >= 2.0

    # --- 5. TPO single print re-entry ---
    tpo_top_arr, tpo_bottom_arr = compute_tpo_reentry_triggers(raw, first_pos, last_pos)
    tpo_top = pd.Series(False, index=close.index)
    tpo_bottom = pd.Series(False, index=close.index)
    tpo_top.iloc[first_pos:last_pos] = tpo_top_arr
    tpo_bottom.iloc[first_pos:last_pos] = tpo_bottom_arr

    triggers = [
        ("vpoc_pinball_reject", "top", vpoc_top),
        ("vpoc_pinball_reject", "bottom", vpoc_bottom),
        ("renko_reversal_brick", "top", pd.Series(renko_down, index=close.index)),
        ("renko_reversal_brick", "bottom", pd.Series(renko_up, index=close.index)),
        ("btc_eth_ratio_zscore_meanrev", "top", ratio_top),
        ("btc_eth_ratio_zscore_meanrev", "bottom", ratio_bottom),
        ("kalman_deviation_meanrev", "top", kalman_top),
        ("kalman_deviation_meanrev", "bottom", kalman_bottom),
        ("tpo_single_print_reentry", "top", tpo_top),
        ("tpo_single_print_reentry", "bottom", tpo_bottom),
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
    out_dir = ROOT / "tmp" / "eth_candidate_pool_raw_lift_check_20260831"
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
