#!/usr/bin/env python3
"""Selection-bias-aware performance statistics.

Every promotion attempt in this repo has followed one arc: search a parameter
space, find a spectacular in-sample number, watch it die out of sample. The
missing instrument is a way to ask, before getting excited, "is this number
bigger than what a search of this size pulls out of pure noise?"

  expected_max_sharpe        E[max SR] over N trials under the null of no skill.
  probabilistic_sharpe_ratio P(true SR > benchmark), skew/kurtosis adjusted.
                             Bailey & Lopez de Prado (2012).
  deflated_sharpe_ratio      PSR with the benchmark set to the expected max of
                             the search itself. Bailey & Lopez de Prado (2014).
                             DSR < 0.95 => the winner is inside the noise floor.
  pbo_cscv                   Probability of Backtest Overfitting via
                             Combinatorially Symmetric Cross-Validation.
                             Bailey, Borwein, Lopez de Prado & Zhu (2017).
  falsification_audit        Runs the exact same best-of-N selection against synthetic
                             zero-predictability and microstructure-placebo panels built
                             from this search's own shape and volatility. A search that
                             clears its own bar as easily on those as on the real panel
                             is not measuring skill -- it is an adaptive-specification-
                             search artifact. Nikolopoulos, "Spurious Predictability in
                             Financial Machine Learning" (arXiv:2604.15531, 2026). Run
                             this BEFORE spending VAL/OOS budget on a search's winner,
                             not after.

Sharpes here are per-period and must match the period of the return series
passed alongside them (daily returns -> daily Sharpe, n_obs = number of days).
Annualising before calling these functions inflates every statistic.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329
# Equity at or below this is a wiped-out account; the path after it is float
# arithmetic, not a tradeable return series.
RUIN_FLOOR = 1e-6


def sharpe(returns: np.ndarray) -> float:
    """Per-period Sharpe of a return series. 0.0 for degenerate/too-short input."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(r) / sd)


def _sharpe_columns(block: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of every column of a (periods x configs) block."""
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(block, axis=0)
        sd = np.nanstd(block, axis=0, ddof=1)
        out = np.where(sd > 1e-15, mu / sd, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def periodic_returns(equity: np.ndarray, bars_per_period: int) -> np.ndarray:
    """Per-bar equity curve -> returns sampled every bars_per_period bars.

    Once equity reaches the ruin floor the account is gone; the rest of the
    series is reported flat rather than propagating float noise into the stats.
    """
    eq = np.asarray(equity, dtype=np.float64)
    idx = np.arange(0, eq.size, max(1, int(bars_per_period)))
    pts = np.maximum(eq[idx], 0.0)
    if pts.size < 3:
        return np.zeros(0, dtype=np.float64)
    prev, nxt = pts[:-1], pts[1:]
    alive = prev > RUIN_FLOOR
    return np.where(alive, (nxt - prev) / np.where(alive, prev, 1.0), 0.0)


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """E[max SR] across n_trials independent trials whose Sharpes have spread sr_std.

    This is the bar a search of this size must clear before its winner means
    anything: run enough configurations against noise and one of them will look
    skilled. Assumes independent trials -- guided samplers (TPE, CMA-ES) violate
    that, so prefer random search when this number has to be exact.
    """
    if n_trials < 2 or not np.isfinite(sr_std) or sr_std <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_std * ((1.0 - g) * z1 + g * z2))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    skewness: float,
    kurt: float,
    benchmark_sr: float = 0.0,
) -> float:
    """P(true SR > benchmark_sr) given observed_sr estimated over n_obs periods.

    kurt is raw kurtosis (3.0 for a normal), not excess. Negative skew and fat
    tails both make a given Sharpe less trustworthy, which is why they enter.
    """
    if n_obs < 3:
        return float("nan")
    denom_sq = 1.0 - skewness * observed_sr + 0.25 * (kurt - 1.0) * observed_sr ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(best_returns: np.ndarray, trial_sharpes: np.ndarray) -> dict:
    """DSR for the winner of a search, given every trial's Sharpe.

    best_returns  per-period returns of the selected configuration.
    trial_sharpes per-period Sharpe of every trial in the search, winner included.

    Returns the intermediate quantities too -- the noise floor is usually the
    number that settles the argument.
    """
    r = np.asarray(best_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    srs = np.asarray(trial_sharpes, dtype=np.float64)
    srs = srs[np.isfinite(srs)]

    observed = sharpe(r)
    n_trials = int(srs.size)
    sr_std = float(np.std(srs, ddof=1)) if n_trials > 1 else 0.0
    floor = expected_max_sharpe(n_trials, sr_std)
    dsr = (
        probabilistic_sharpe_ratio(
            observed, int(r.size), float(skew(r)), float(kurtosis(r, fisher=False)), floor
        )
        if r.size >= 3
        else float("nan")
    )
    return {
        "observed_sharpe": observed,
        "n_obs": int(r.size),
        "n_trials": n_trials,
        "trial_sharpe_std": sr_std,
        "noise_floor_sharpe": floor,
        "deflated_sharpe_ratio": dsr,
        "passes_95": bool(np.isfinite(dsr) and dsr >= 0.95),
    }


def pbo_cscv(returns_matrix: np.ndarray, n_splits: int = 10) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    returns_matrix  (periods x configurations) per-period returns.

    Rows are cut into n_splits contiguous chunks. For every way of holding out
    half the chunks, the configuration with the best in-sample Sharpe is picked
    and its rank among all configurations is measured out-of-sample. PBO is the
    share of splits where that in-sample winner lands below the out-of-sample
    median -- i.e. how often this selection procedure picks a future loser.
    PBO near 0.5 means the search carries no information at all.
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (periods x configurations)")
    n_periods, n_cfg = m.shape
    if n_cfg < 2:
        raise ValueError("PBO needs at least 2 configurations")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even so the chunks can be halved")
    if n_periods < n_splits * 3:
        raise ValueError(
            f"need >= {n_splits * 3} periods for {n_splits} splits, got {n_periods}"
        )

    chunks = np.array_split(np.arange(n_periods), n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        held = set(combo)
        is_rows = np.concatenate([chunks[c] for c in combo])
        oos_rows = np.concatenate([chunks[c] for c in range(n_splits) if c not in held])
        is_sr = _sharpe_columns(m[is_rows])
        oos_sr = _sharpe_columns(m[oos_rows])
        best = int(np.argmax(is_sr))
        # Relative rank of the in-sample winner within the OOS Sharpe spread.
        rank = float(np.sum(oos_sr <= oos_sr[best])) / (n_cfg + 1.0)
        rank = min(max(rank, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(rank / (1.0 - rank)))

    lg = np.asarray(logits, dtype=np.float64)
    return {
        "pbo": float(np.mean(lg <= 0.0)),
        "n_combinations": int(lg.size),
        "median_logit": float(np.median(lg)),
    }


def _demeaned_block_bootstrap_panel(
    panel: np.ndarray, rng: np.random.Generator, block_size: int
) -> np.ndarray:
    """Demean every column, then rebuild it from randomly-chosen, circularly-wrapped
    blocks of its own (demeaned) history (Politis & Romano 1994 circular block
    bootstrap), independently per column.

    Demeaning first is load-bearing: a plain shuffle or shift of the raw column
    reorders the same values, so its mean/std -- and therefore its Sharpe -- comes
    out identical to the original every time, which would make a genuine edge
    indistinguishable from itself instead of from noise. Forcing the true mean to
    zero before resampling gives a null where no configuration has real skill, while
    each column keeps its OWN empirical autocorrelation, volatility-clustering and
    fat-tail fingerprint (unlike the i.i.d. Gaussian zero_predictability_null) and
    each column is resampled independently, destroying genuine cross-sectional timing
    alignment between configurations. This is the "microstructure placebo".
    """
    n_periods, n_cfg = panel.shape
    demeaned = panel - np.nanmean(panel, axis=0, keepdims=True)
    bs = max(1, min(int(block_size), n_periods))
    n_blocks = -(-n_periods // bs)  # ceil
    starts = rng.integers(0, n_periods, size=(n_blocks, n_cfg))
    offsets = np.arange(bs)
    idx = (starts[:, None, :] + offsets[None, :, None]) % n_periods
    idx = idx.reshape(n_blocks * bs, n_cfg)[:n_periods]
    return np.take_along_axis(demeaned, idx, axis=0)


def falsification_audit(
    returns_matrix: np.ndarray,
    *,
    n_null_draws: int = 500,
    block_size: int = 20,
    min_percentile: float = 0.95,
    seed: int = 20260809,
) -> dict:
    """Could this exact search have produced its winner out of noise alone?

    returns_matrix  (periods x configurations) per-period returns, the same shape
                    pbo_cscv takes -- every configuration the search actually tried,
                    winner included.
    block_size      bootstrap block length (in periods) for the microstructure
                    placebo. Should span whatever horizon this data's serial
                    correlation/vol-clustering actually decays over; too small
                    degenerates toward i.i.d. resampling, too large toward a single
                    whole-series draw.

    The real best-of-N Sharpe is compared against two synthetic reference classes,
    each drawn `n_null_draws` times at the search's own shape (same n_trials, same
    n_periods -- so the null carries the exact same multiple-testing multiplicity):

      zero_predictability_null   every configuration replaced by i.i.d. Gaussian
                                 noise at zero mean, matched to its own volatility.
                                 A market with no edge anywhere.
      microstructure_placebo_null  every configuration's own return series, demeaned
                                 and circular-block-bootstrapped (see
                                 _demeaned_block_bootstrap_panel). Real volatility
                                 clustering, autocorrelation and fat tails, forced to
                                 zero true mean, no genuine timing.

    A pipeline whose real winner is unremarkable against either null (falls below
    `min_percentile` of null draws) cannot tell real predictability from a search
    artifact, independent of what the headline Sharpe/PnL number says. This is meant
    to gate a search's winner BEFORE it is allowed to consume VAL/OOS budget, mirroring
    the "falsification audit" workflow in Nikolopoulos (arXiv:2604.15531, 2026).
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (periods x configurations)")
    n_periods, n_cfg = m.shape
    if n_cfg < 2:
        raise ValueError("falsification audit needs at least 2 configurations")
    if n_periods < 10:
        raise ValueError("falsification audit needs at least 10 periods per configuration")

    real_best = float(np.max(_sharpe_columns(m)))

    rng = np.random.default_rng(seed)
    col_std = np.nanstd(m, axis=0, ddof=1)
    col_std = np.where(col_std > 1e-15, col_std, 1e-15)

    zero_pred_null = np.empty(int(n_null_draws), dtype=np.float64)
    placebo_null = np.empty(int(n_null_draws), dtype=np.float64)
    for i in range(int(n_null_draws)):
        synthetic = rng.normal(0.0, col_std, size=(n_periods, n_cfg))
        zero_pred_null[i] = np.max(_sharpe_columns(synthetic))
        resampled = _demeaned_block_bootstrap_panel(m, rng, block_size)
        placebo_null[i] = np.max(_sharpe_columns(resampled))

    zero_pred_percentile = float(np.mean(zero_pred_null < real_best))
    placebo_percentile = float(np.mean(placebo_null < real_best))
    passed = zero_pred_percentile >= min_percentile and placebo_percentile >= min_percentile

    return {
        "real_best_sharpe": real_best,
        "n_trials": int(n_cfg),
        "n_periods": int(n_periods),
        "n_null_draws": int(n_null_draws),
        "zero_predictability_null_mean": float(zero_pred_null.mean()),
        "zero_predictability_null_p95": float(np.percentile(zero_pred_null, 95)),
        "zero_predictability_percentile": zero_pred_percentile,
        "microstructure_placebo_null_mean": float(placebo_null.mean()),
        "microstructure_placebo_null_p95": float(np.percentile(placebo_null, 95)),
        "microstructure_placebo_percentile": placebo_percentile,
        "min_percentile_required": float(min_percentile),
        "passes_falsification_audit": bool(passed),
    }

