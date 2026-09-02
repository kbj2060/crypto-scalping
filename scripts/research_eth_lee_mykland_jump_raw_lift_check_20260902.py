#!/usr/bin/env python3
"""Raw-rule lift pre-check for the Lee-Mykland jump test as a Homer evidence-signal candidate --
docs/homer/external_literature_signal_candidates_20260902.md's A-1 (the #1 recommendation of the
2026-09-02 external-literature survey). Same event_study/zigzag-pivot lift methodology as
research_eth_candidate_pool_raw_lift_check_20260831.py (VPOC/Renko/BTC-ETH/Kalman/TPO) and
research_eth_demarker_evidence_signal_lift_check_20260831.py -- imported verbatim. Retrospective
evidence-gathering diagnostic, not a live-tradeable signal claim, not Fresh-Forward gated (see
either prior script's docstring for why).

WHAT IS TESTED

  Lee & Mykland (2008), "Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics",
  RFS 21(6). Statistic:

      L(t) = r(t) / sqrt(V(t)),   V(t) = 1/(K-2) * sum_{j=t-K+2}^{t-1} |r(j)| * |r(j-1)|

  V is the *bipower variation* over the K bars ENDING AT t-1 -- the current bar is excluded, so a
  jump cannot inflate its own denominator. This is the structural difference from the deployed
  `short_term_return_z` signal (3-bar return / rolling-288 stdev), whose denominator IS contaminated
  by the move it is trying to flag. K=270 is Lee-Mykland's own Table-1 recommendation for 5-minute
  data (carried over verbatim; their 78-bar equity day vs this 288-bar crypto day is noted as a
  deliberate, unadjusted borrowing).

  Threshold is NOT a free parameter. Under the no-jump null, max|L| over n observations is
  asymptotically Gumbel:

      (max|L| - C_n) / S_n  ->  Gumbel,
      C_n = (2 log n)^0.5 / c - [log(pi) + log(log n)] / (2 c (2 log n)^0.5),
      S_n = 1 / (c (2 log n)^0.5),   c = sqrt(2/pi)

  so a chosen alpha fixes the FALSE-POSITIVE RATE (here: per n=288 bars = per calendar day),
  rather than the trigger count being an artifact of a hand-picked z cutoff. Primary alpha=0.01
  (pre-registered in the survey doc); 0.05/0.10 reported as sensitivity, per README ss5.6's
  "never trust a grid boundary".

  Boudt, Croux & Laurent (2011) intraday periodicity filter: 24/7 crypto still has a strong
  time-of-day volatility shape, which inflates |L| in active hours and suppresses it in quiet ones.
  f_j is the MAD-based periodicity factor for each of the 288 five-minute slots, normalised so
  mean(f^2)=1, and L*(t) = L(t)/f_slot(t). CAUSALITY: f_j is estimated ONLY on bars strictly before
  VAL_START (2023-12-31..2025-08-31, ~1.7yr), never on the evaluation window.

ARMS (all BOTH sides; down-jump -> "bottom" side, up-jump -> "top" side, matching the
short_term_return_z sign convention)

  1. lm_jump_adj  -- periodicity-adjusted L*, the actual candidate.
  2. lm_jump_raw  -- unadjusted L, an ablation that isolates what the Boudt filter contributes.
  3. short_term_return_z -- the deployed signal (ret3_z beyond +-2.5, verbatim from
     live_evidence_signal_dashboard_20260823.py), recomputed in this same window as a benchmark
     AND as the overlap reference: the survey doc requires measuring how independent the new
     trigger is (repo's independence yardstick: smt_divergence <-> liquidity_sweep at 6.0-9.5%).

Window: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17, identical to the other scorecard
scripts in this lineage (data/eth_5m_1year.csv's coverage ends exactly at OOS_END).
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

Z_95 = 1.959963984540054
ZSCORE_WINDOW = 288          # matches live_evidence_signal_dashboard_20260823.py's ZSCORE_WINDOW
LM_WINDOW = 270              # Lee-Mykland (2008) Table 1 recommendation for 5-minute data
BARS_PER_DAY = 288           # 5m bars in 24h -- also the Gumbel testing-region size n
ALPHAS = (0.01, 0.05, 0.10)  # 0.01 primary (pre-registered), rest = sensitivity
STRZ_THRESHOLD = 2.5         # deployed short_term_return_z cutoff, verbatim
OVERLAP_TOL_BARS = 3         # proximity tolerance for the "near-duplicate firing" overlap measure


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score 95% CI -- copied verbatim from research_eth_evidence_signal_scorecard_ci_20260825.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def gumbel_threshold(alpha: float, n: int = BARS_PER_DAY) -> float:
    """Lee-Mykland critical value for |L| at significance `alpha` over a testing region of n bars."""
    c = np.sqrt(2.0 / np.pi)
    sqrt_2logn = np.sqrt(2.0 * np.log(n))
    c_n = sqrt_2logn / c - (np.log(np.pi) + np.log(np.log(n))) / (2.0 * c * sqrt_2logn)
    s_n = 1.0 / (c * sqrt_2logn)
    beta_star = -np.log(-np.log(1.0 - alpha))
    return float(c_n + s_n * beta_star)


def lee_mykland_stat(close: pd.Series, window: int = LM_WINDOW) -> pd.Series:
    """L(t) = r(t)/sqrt(V(t)) with V(t) the bipower variation over bars t-window+2 .. t-1.

    The rolling sum is taken through t-1 and then shifted, so the current bar's own return never
    enters its own denominator (the whole point of the statistic)."""
    r = np.log(close).diff()
    prod = r.abs() * r.abs().shift(1)                      # |r_j| * |r_{j-1}|
    bv = prod.rolling(window - 2, min_periods=window - 2).sum().shift(1) / (window - 2)
    return r / np.sqrt(bv.replace(0.0, np.nan))


def periodicity_factors(stat: pd.Series, timestamps: pd.Series, fit_mask: np.ndarray) -> np.ndarray:
    """Boudt-Croux-Laurent MAD-based intraday periodicity factor for each of the 288 5m slots.

    MAD_j = 1.486 * median(|L| in slot j), normalised so mean(f^2) = 1. Estimated on `fit_mask`
    bars only (pre-VAL history) -- never on the evaluation window."""
    slot = (timestamps.dt.hour * 12 + timestamps.dt.minute // 5).to_numpy()
    vals = stat.to_numpy()
    usable = fit_mask & np.isfinite(vals)
    mad = np.full(BARS_PER_DAY, np.nan)
    for j in range(BARS_PER_DAY):
        sel = vals[usable & (slot == j)]
        if len(sel) >= 30:
            mad[j] = 1.486 * np.median(np.abs(sel))
    if not np.isfinite(mad).any():
        raise RuntimeError("periodicity estimation found no usable slots")
    mad = np.where(np.isfinite(mad), mad, np.nanmedian(mad))
    f = mad / np.sqrt(np.mean(mad ** 2))
    return f[slot]


def overlap_stats(a_pos: np.ndarray, b_pos: np.ndarray, tol: int) -> dict:
    """Exact-bar Jaccard plus the fraction of each side's fires that sit within `tol` bars of the
    other's -- the repo's independence yardstick (smt<->liquidity_sweep 6.0-9.5%)."""
    set_a, set_b = set(a_pos.tolist()), set(b_pos.tolist())
    union = len(set_a | set_b)
    jaccard = len(set_a & set_b) / union if union else float("nan")
    b_sorted = np.sort(b_pos)

    def near_frac(src: np.ndarray, dst: np.ndarray) -> float:
        if len(src) == 0 or len(dst) == 0:
            return float("nan")
        idx = np.searchsorted(dst, src)
        left = np.where(idx > 0, dst[np.clip(idx - 1, 0, len(dst) - 1)], -10 ** 9)
        right = np.where(idx < len(dst), dst[np.clip(idx, 0, len(dst) - 1)], 10 ** 9)
        dist = np.minimum(np.abs(src - left), np.abs(src - right))
        return float((dist <= tol).mean())

    return {
        "jaccard_exact_bar": jaccard,
        "frac_a_near_b": near_frac(np.sort(a_pos), b_sorted),
        "frac_b_near_a": near_frac(b_sorted, np.sort(a_pos)),
    }


def main() -> None:
    raw = load_frame()
    pivots = load_zigzag_pivots()
    close, ts = raw["close"], raw["timestamp"]

    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    all_pos = np.flatnonzero(window_mask)
    fit_mask = (ts < VAL_START).to_numpy()
    print(f"Window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")
    print(f"Periodicity fit on {int(fit_mask.sum())} pre-VAL bars "
          f"({ts[fit_mask].min().date()}..{ts[fit_mask].max().date()})")

    lm_raw = lee_mykland_stat(close)
    f_slot = periodicity_factors(lm_raw, ts, fit_mask)
    lm_adj = lm_raw / f_slot
    print(f"Periodicity factor f: min {np.nanmin(f_slot):.3f} / max {np.nanmax(f_slot):.3f} "
          f"(ratio {np.nanmax(f_slot) / np.nanmin(f_slot):.2f}x)")

    ret3 = close / close.shift(3) - 1.0
    ret3_z = (ret3 - ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()) / \
        ret3.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)

    arms: list[tuple[str, str, pd.Series]] = []
    for alpha in ALPHAS:
        thr = gumbel_threshold(alpha)
        tag = f"a{int(alpha * 100):02d}"
        print(f"Gumbel threshold alpha={alpha:.2f} (n={BARS_PER_DAY}): |L| > {thr:.4f}")
        arms += [
            (f"lm_jump_adj_{tag}", "bottom", lm_adj <= -thr),
            (f"lm_jump_adj_{tag}", "top", lm_adj >= thr),
            (f"lm_jump_raw_{tag}", "bottom", lm_raw <= -thr),
            (f"lm_jump_raw_{tag}", "top", lm_raw >= thr),
        ]
    arms += [
        ("short_term_return_z_ref", "bottom", ret3_z <= -STRZ_THRESHOLD),
        ("short_term_return_z_ref", "top", ret3_z >= STRZ_THRESHOLD),
    ]

    rows = []
    fires: dict[tuple[str, str], np.ndarray] = {}
    for name, side, trigger_series in arms:
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = raw.index[raw["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        trigger_pos = np.flatnonzero(trigger_series.fillna(False).to_numpy() & window_mask)
        fires[(name, side)] = trigger_pos
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            n, prec = stats["n_triggers"], stats["precision"]
            hits = round(prec * n) if n and np.isfinite(prec) else 0
            lo, hi = wilson_ci(hits, n) if n else (float("nan"), float("nan"))
            rows.append({
                "signal": name, "side": side, "horizon": k_name,
                "n_triggers": n, "precision": prec, "ci_lo": lo, "ci_hi": hi,
                "baseline_rate": stats["baseline_rate"], "lift": stats["lift"],
                "recall": stats["recall"], "median_lead_bars": stats["median_lead_bars"],
            })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "tmp" / "eth_lee_mykland_jump_raw_lift_check_20260902"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "scorecard.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    for horizon in K_HORIZONS:
        print(f"\n=== horizon {horizon} ===")
        sub = df[df["horizon"] == horizon].copy()
        sub["precision_pct"] = (sub["precision"] * 100).round(1)
        sub["ci_lo_pct"] = (sub["ci_lo"] * 100).round(1)
        sub["ci_hi_pct"] = (sub["ci_hi"] * 100).round(1)
        sub["baseline_pct"] = (sub["baseline_rate"] * 100).round(1)
        sub["lift_x"] = sub["lift"].round(2)
        sub["recall_pct"] = (sub["recall"] * 100).round(1)
        cols = ["signal", "side", "n_triggers", "precision_pct", "ci_lo_pct", "ci_hi_pct",
                "baseline_pct", "lift_x", "recall_pct"]
        print(sub[cols].to_string(index=False))

    print("\n=== overlap vs deployed short_term_return_z (same window) ===")
    ov_rows = []
    for alpha in ALPHAS:
        tag = f"a{int(alpha * 100):02d}"
        for arm in (f"lm_jump_adj_{tag}", f"lm_jump_raw_{tag}"):
            for side in ("bottom", "top"):
                st = overlap_stats(fires[(arm, side)], fires[("short_term_return_z_ref", side)],
                                   OVERLAP_TOL_BARS)
                ov_rows.append({"signal": arm, "side": side,
                                "n_lm": len(fires[(arm, side)]),
                                "n_strz": len(fires[("short_term_return_z_ref", side)]),
                                "jaccard_exact_pct": round(st["jaccard_exact_bar"] * 100, 1),
                                f"lm_within_{OVERLAP_TOL_BARS}b_of_strz_pct":
                                    round(st["frac_a_near_b"] * 100, 1),
                                f"strz_within_{OVERLAP_TOL_BARS}b_of_lm_pct":
                                    round(st["frac_b_near_a"] * 100, 1)})
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(out_dir / "overlap.csv", index=False)
    print(ov.to_string(index=False))

    print(f"\nWrote {out_dir / 'scorecard.csv'} and {out_dir / 'overlap.csv'}")


if __name__ == "__main__":
    main()
