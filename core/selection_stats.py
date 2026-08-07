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

