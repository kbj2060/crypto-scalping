#!/usr/bin/env python3
"""Liquidation-map v2 estimate: OI-cohort survival + direction split. Pure/no-I/O module, same
role and caller contract as scripts/live_liquidation_map_20260824.py (v1) -- design and rationale
in docs/experiments/eth_liquidation_map_v2_oi_cohort_direction_design_20260825.md.

What changes vs v1 (see the design doc's W-table):
- Entry mass: v1 treats EVERY candle close as a hypothetical entry weighted by volume; v2 births a
  cohort only when open interest actually rose that bar (dOI+ in ETH contracts) -- volume can't
  tell position-opening from churn, dOI can.
- Survival: v1 decays by an arbitrary 240h recency halflife; v2 decays all cohorts pro-rata by the
  measured OI decline (dOI-) -- positions leave when they actually close, not on a clock. This also
  removes v1's hard [24h,168h] window + reset state machine entirely: old cohorts persist exactly
  as long as the exchange-reported OI says positions persist.
- Direction: v1 gives both sides equal weight per candle (structurally symmetric map); v2 splits
  each birth into long/short by taker share (v2b) optionally blended with the global long/short
  account fraction (v2c). All splits are parameterless or fixed constants -- no tuned coefficients.
- Unchanged from v1 (imported, not copied): liquidation-price formula per leverage tier, flat
  maintenance-margin approximation, bin math and top-N level extraction (levels_from_bins), and
  the already-crossed drop rule (a level price has passed through is dead).

Same caveat stack as v1 applies verbatim: this is an ESTIMATE (nobody outside the exchange knows
real per-position entries/leverage), a discretionary reading aid only -- never wire into
trading_bot.py or any promotion path. Gate for replacing v1 on the dashboard: the A/B backtest in
scripts/research_eth_liquidation_map_v2_cohort_ab_backtest_20260825.py must show v2 >= v1 OOS.

Data-quality guards (from the Phase 0 audit, data/research/
eth_liquidation_map_v2_phase0_data_audit_20260825.json):
- OI <= 0 rows are exchange-side dropouts published as literal zeros (75 found in 2.6y) -- the
  CALLER must clean them (ffill) before prepare_cohort_arrays(); asserted here.
- |dOI| <= volume is an identity in contracts (positions can't change hands faster than they
  trade); violations after zero-cleaning are 0.05% of bars, worst 1.84x -- clamped here, and the
  clamped delta drives BOTH births and decay so the two stay consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as v1

VARIANTS = ("v2a", "v2b", "v2c")
LONG_SHARE_CLIP = (0.1, 0.9)  # keep splits away from degenerate all-one-side maps


def prepare_cohort_arrays(df: pd.DataFrame) -> dict:
    """df: ascending 1h bars with columns close/high/low/volume/oi and (for v2b/v2c) taker_buy_base
    / long_account_frac. oi must be pre-cleaned (no NaN/<=0). Returns the precomputed arrays every
    compute_cohort_levels() call shares -- O(n) once, O(i) per snapshot after."""
    oi = df["oi"].to_numpy(dtype="float64")
    if not np.all(np.isfinite(oi)) or not np.all(oi > 0):
        raise ValueError("oi must be positive and finite -- clean zeros/NaN (ffill) before calling")
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    volume = df["volume"].to_numpy(dtype="float64")

    doi = np.diff(oi, prepend=oi[0])          # bar 0: no prior -> delta 0, cohort never born there
    doi = np.clip(doi, -volume, volume)       # identity clamp (see module docstring)
    births = np.maximum(doi, 0.0)
    prev_oi = np.concatenate([[oi[0]], oi[:-1]])
    decay_ratio = 1.0 + np.minimum(doi, 0.0) / prev_oi
    S = np.cumsum(np.log(decay_ratio))        # non-increasing -> exp(S_i - S_j) <= 1, no overflow

    if "taker_buy_base" in df.columns:
        with np.errstate(invalid="ignore", divide="ignore"):
            taker_share = df["taker_buy_base"].to_numpy(dtype="float64") / volume
        taker_share = np.clip(np.nan_to_num(taker_share, nan=0.5), *LONG_SHARE_CLIP)
    else:
        taker_share = np.full(len(df), 0.5)
    if "long_account_frac" in df.columns:
        laf = np.nan_to_num(df["long_account_frac"].to_numpy(dtype="float64"), nan=0.5)
    else:
        laf = np.full(len(df), 0.5)
    long_share = {
        "v2a": np.full(len(df), 0.5),
        "v2b": taker_share,
        "v2c": np.clip(0.5 * taker_share + 0.5 * laf, *LONG_SHARE_CLIP),
    }
    return {"close": close, "high": high, "low": low, "births": births, "S": S,
            "long_share": long_share, "n": len(df)}


def _accumulate(bins: dict, bin_width: float, prices: np.ndarray, weights: np.ndarray,
                alive: np.ndarray) -> float:
    idx = alive & (prices > 0) & (weights > 0)
    if not idx.any():
        return 0.0
    b = np.round(prices[idx] / bin_width).astype("int64")
    u, inv = np.unique(b, return_inverse=True)
    w = np.bincount(inv, weights=weights[idx])
    for k, v in zip(u.tolist(), w.tolist()):
        bins[k] = bins.get(k, 0.0) + v
    return float(w.sum())


def compute_cohort_levels(arrs: dict, i: int, variant: str, current_price: float | None = None,
                          max_age_hours: int | None = None) -> dict:
    """Levels as of bar i, using bars <= i only (causal). Returns the v1 payload shape
    (warmed_up/current_price/support_levels/resistance_levels/bin_width/heatmap_bins) plus
    long_usd_total/short_usd_total (the new asymmetry reading) and truncated_mass_pct (surviving
    mass dropped by max_age_hours, 0.0 when uncapped)."""
    cp = float(arrs["close"][i]) if current_price is None else float(current_price)
    if not (cp > 0) or i < 1:
        return {"warmed_up": False, "error": "insufficient_data"}
    c = arrs["close"][: i + 1]
    S = arrs["S"]
    mass = arrs["births"][: i + 1] * np.exp(S[i] - S[: i + 1])
    truncated_pct = 0.0
    if max_age_hours is not None:
        ages = i - np.arange(i + 1)
        old = ages > max_age_hours
        total = mass.sum()
        if total > 0 and old.any():
            truncated_pct = float(mass[old].sum() / total * 100.0)
            mass = np.where(old, 0.0, mass)

    # min low / max high strictly AFTER each entry bar j (empty suffix for j == i -> never crossed)
    lo_after = np.full(i + 1, np.inf)
    hi_after = np.full(i + 1, -np.inf)
    if i >= 1:
        lo_after[:-1] = np.minimum.accumulate(arrs["low"][1: i + 1][::-1])[::-1]
        hi_after[:-1] = np.maximum.accumulate(arrs["high"][1: i + 1][::-1])[::-1]

    ls = arrs["long_share"][variant][: i + 1]
    usd_long = mass * ls * c / len(v1.LEVERAGE_TIERS)
    usd_short = mass * (1.0 - ls) * c / len(v1.LEVERAGE_TIERS)

    bin_width = max(cp * v1.BIN_WIDTH_PCT, 1e-9)
    bins: dict[int, float] = {}
    long_total = short_total = 0.0
    for lev in v1.LEVERAGE_TIERS:
        long_liq = c * (1.0 - 1.0 / lev + v1.MAINTENANCE_MARGIN_RATE)
        short_liq = c * (1.0 + 1.0 / lev - v1.MAINTENANCE_MARGIN_RATE)
        long_total += _accumulate(bins, bin_width, long_liq, usd_long, lo_after > long_liq)
        short_total += _accumulate(bins, bin_width, short_liq, usd_short, hi_after < short_liq)
    if not bins or not (max(bins.values()) > 0):
        return {"warmed_up": False, "error": "no_surviving_levels"}

    max_w = max(bins.values())
    heatmap = sorted(({"price": round(b * bin_width, 4), "weight_pct": round(w / max_w, 4)}
                      for b, w in bins.items() if b * bin_width != cp), key=lambda x: x["price"])
    return {
        "warmed_up": True, "error": None, "current_price": cp, "bars_used": int(i + 1),
        **v1.levels_from_bins(bins, bin_width, cp),
        "bin_width": round(bin_width, 4), "heatmap_bins": heatmap,
        "long_usd_total": round(long_total, 2), "short_usd_total": round(short_total, 2),
        "truncated_mass_pct": round(truncated_pct, 3),
    }
