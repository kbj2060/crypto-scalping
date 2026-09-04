#!/usr/bin/env python3
"""Autocorrelation-regime gating check for the 2 Homer candidate-pool signals the user chose to
carry forward: kalman_deviation_meanrev and DeMarker-extreme (2026-08-31 narrowing decision, see
memory eth_homer_candidate_pool_raw_lift_check_20260831). Both are mean-reversion theses, and
features/engineering.py already has an untested, causal, already-implemented "is this actually a
mean-reverting regime right now" feature: _return_autocorrelation (lag-1, rolling-48 Pearson
autocorrelation of returns -- per that function's own docstring, negative = mean-reverting regime,
positive = momentum regime, ~0 = random walk). This checks whether gating either signal by that
regime improves its already-measured raw lift.

Also settles how DeMarker is carried forward: this script uses "DeMarker extreme" (dem>=0.90 top /
dem<=0.10 bottom) as the definition, NOT the original 3-rule SMC-divergence/Wyckoff-spring
construction -- research_eth_demarker_evidence_signal_lift_check_20260831.py's component-alone
ablation already showed DeMarker-extreme-alone explains most of those rules' lift, with the
SMC/Wyckoff-specific conditioning adding little to nothing (Rule2 was actively worse). Same
event_study/zigzag-pivot methodology as every other script in this lineage.

For each of the 2 signals x 2 sides, reports 3 variants at each horizon:
  - ungated: the raw trigger, no regime filter (recomputed here for a clean side-by-side against
    the already-recorded numbers from the prior 2 scripts)
  - gated_meanrev: trigger AND autocorr_48 < 0 (mean-reverting regime -- the theoretically-motivated gate)
  - gated_momentum: trigger AND autocorr_48 >= 0 (the complement -- lets a reader see whether the
    gate is actually discriminating or just cutting the sample in half at random)

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the rest of this lineage.
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
    load_frame,
)
from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402

Z_95 = 1.959963984540054
AUTOCORR_WINDOW = 48   # matches features/engineering.py::_return_autocorrelation's default
AUTOCORR_LAG = 1       # matches features/engineering.py::_return_autocorrelation's default


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI -- copied verbatim from research_eth_evidence_signal_scorecard_ci_20260825.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def compute_return_autocorrelation(close: pd.Series, window: int = AUTOCORR_WINDOW,
                                    lag: int = AUTOCORR_LAG) -> pd.Series:
    """Copied verbatim from features/engineering.py::FeatureEngineer._return_autocorrelation
    (lag=1 rolling-48 Pearson autocorrelation of returns) -- minus the `self` parameter."""
    returns = close.pct_change().fillna(0)

    def _autocorr(x):
        if len(x) < lag + 4:
            return 0.0
        r_t = x[lag:]
        r_tm = x[:-lag]
        denom = r_t.std() * r_tm.std()
        if denom < 1e-10:
            return 0.0
        return np.corrcoef(r_t, r_tm)[0, 1]

    return (
        returns
        .rolling(window, min_periods=window // 2)
        .apply(_autocorr, raw=True)
        .fillna(0)
    )


def main() -> None:
    raw = load_frame()
    pivots = load_zigzag_pivots()
    high, low, close = raw["high"], raw["low"], raw["close"]

    ts = raw["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    autocorr = compute_return_autocorrelation(close)
    meanrev_regime = (autocorr < 0).to_numpy()
    print(f"  [autocorr] mean-reverting bars (autocorr<0): {meanrev_regime[window_mask].mean() * 100:.1f}% of in-window bars")

    dem = compute_demarker(high, low)
    dem_top = (dem >= 0.90).fillna(False)
    dem_bottom = (dem <= 0.10).fillna(False)

    levels, _velocities = kalman_level_and_velocity(close.to_numpy())
    kalman_dev = pd.Series((close.to_numpy() - levels) / levels, index=close.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    kalman_top = (kalman_dev_z >= 2.0).fillna(False)
    kalman_bottom = (kalman_dev_z <= -2.0).fillna(False)

    base_signals = [
        ("demarker_extreme", "top", dem_top),
        ("demarker_extreme", "bottom", dem_bottom),
        ("kalman_deviation_meanrev", "top", kalman_top),
        ("kalman_deviation_meanrev", "bottom", kalman_bottom),
    ]

    triggers = []
    for name, side, base in base_signals:
        base_arr = base.to_numpy()
        triggers.append((name, side, "ungated", base_arr))
        triggers.append((name, side, "gated_meanrev", base_arr & meanrev_regime))
        triggers.append((name, side, "gated_momentum", base_arr & ~meanrev_regime))

    rows = []
    for name, side, variant, trigger_arr in triggers:
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        trigger_pos = np.flatnonzero(trigger_arr & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            n, prec = stats["n_triggers"], stats["precision"]
            hits = round(prec * n) if n and np.isfinite(prec) else 0
            lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
            rows.append({
                "signal": name, "side": side, "variant": variant, "horizon": k_name,
                "n_triggers": n, "precision": prec, "ci_lo": lo, "ci_hi": hi,
                "baseline_rate": stats["baseline_rate"], "lift": stats["lift"],
                "recall": stats["recall"],
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_autocorr_regime_gate_kalman_demarker_20260831"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    for horizon in K_HORIZONS:
        print(f"\n=== horizon {horizon} ===")
        sub = df[df["horizon"] == horizon].copy()
        sub["precision_pct"] = (sub["precision"] * 100).round(1)
        sub["baseline_pct"] = (sub["baseline_rate"] * 100).round(1)
        sub["lift_x"] = sub["lift"].round(2)
        cols = ["signal", "side", "variant", "n_triggers", "precision_pct", "baseline_pct", "lift_x"]
        print(sub[cols].to_string(index=False))

    print(f"\nWrote {out_dir / 'scorecard.csv'}")


if __name__ == "__main__":
    main()
