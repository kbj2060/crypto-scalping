#!/usr/bin/env python3
"""Literature-motivated re-test of the liquidation-map levels, 2026-08-25.

Paper lookup (arXiv/OpenAlex/Crossref, 2026-08-25) for academic work on liquidation clustering and
support/resistance validity turned up the closest real analogue to this repo's estimator: Osler
(2000, "Support for Resistance: Technical Analysis and Intraday Exchange Rates") and Osler (2003,
Journal of Finance, "Currency Orders and Exchange Rate Dynamics: An Explanation for the Predictive
Success of Technical Analysis", DOI 10.1111/1540-6261.00588). Osler (2003) uses real FX dealer
stop-loss/take-profit order data and finds TWO distinct clustering patterns with opposite price
effects: take-profit orders cluster AT round numbers (price tends to REVERSE there -- the classic
"support/resistance holds" story), while stop-loss orders cluster JUST BEYOND round numbers (price
tends to ACCELERATE once it crosses there -- a breakout-continuation story, not a reversal one).

A liquidation is, mechanically, a forced stop-loss, not a take-profit -- so the Osler (2003)
mechanism predicts a liquidation-price cluster should behave like her stop-loss clusters
(acceleration through the level) rather than like her take-profit clusters (reversal at the level).
Every backtest this repo has run on this data so far (eth_dashboard_liquidation_map_sr_backtest_
20260824, eth_liquidation_map_event_driven_reset_20260824, eth_liquidation_map_event_driven_
dwell_filter_20260825, eth_liquidation_map_event_driven_min_floor_sweep_20260825) has scored the
REVERSAL hypothesis only (does price hold/close back on the favorable side). This script tests the
complementary, literature-motivated ACCELERATION hypothesis instead: conditional on a level actually
being broken, does price move FURTHER past it (over 24h/72h) than a distance-matched placebo level's
break does? This does not change or re-tune the event-driven state machine at all -- reuses
ed.simulate()'s snapshots verbatim, same distance-matched placebo draw as ed.evaluate(), same
BREAK_TOLERANCE_PCT break definition already used by the live reset trigger.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_osler_breakout_continuation_20260825.json"
SEED = ed.SEED
CONTINUATION_HOURS = (24, 72)  # matches the magnitude horizons every prior round in this line used
N_BOOT = 2000


def find_break(df: pd.DataFrame, t0: int, level_price: float, side: str, forward_hours: int, break_tol: float):
    """First bar index in (t0, t0+forward_hours] whose CLOSE breaks the level by break_tol -- the
    same close-based criterion the live event-driven reset trigger uses (ed.BREAK_TOLERANCE_PCT),
    not evaluate_forward()'s touch-then-followthrough-close definition (that one requires a prior
    wick touch; this one is the direct break condition, since a break's own bar is the event of
    interest here, not what happens after a separate touch)."""
    closes = df["close"].to_numpy()
    n = len(df)
    fwd_end = min(n, t0 + 1 + forward_hours)
    for i in range(t0 + 1, fwd_end):
        if side == "support" and closes[i] < level_price * (1 - break_tol):
            return i
        if side == "resistance" and closes[i] > level_price * (1 + break_tol):
            return i
    return None


def continuation_return(df: pd.DataFrame, break_i: int, level_price: float, side: str, k: int) -> float:
    """% price move from level_price to close[break_i+k], measured in the BREAKOUT direction
    (positive = price kept moving away from the broken level, i.e. acceleration/continuation)."""
    closes = df["close"].to_numpy()
    n = len(df)
    j = min(n - 1, break_i + k)
    p = closes[j]
    return (level_price - p) / level_price if side == "support" else (p - level_price) / level_price


def bootstrap_ci(real: list, placebo: list, n_boot: int, rng: np.random.Generator):
    if not real or not placebo:
        return None
    real_a, placebo_a = np.array(real), np.array(placebo)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        r = rng.choice(real_a, size=len(real_a), replace=True)
        p = rng.choice(placebo_a, size=len(placebo_a), replace=True)
        diffs[b] = r.mean() - p.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"ci_lo_pct": float(lo * 100), "ci_hi_pct": float(hi * 100),
            "frac_positive": float(np.mean(diffs > 0))}


def evaluate_breakout_continuation(df: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        pool = np.array([lv["distance_pct"] for s in snapshots for lv in s[key]])
        if not len(pool):
            pool = np.array([2.0, -2.0])
        real_cont = {h: [] for h in CONTINUATION_HOURS}
        placebo_cont = {h: [] for h in CONTINUATION_HOURS}
        n_real_levels = n_real_breaks = n_placebo_breaks = 0
        for s in snapshots:
            cp = s["current_price"]
            for lv in s[key]:
                n_real_levels += 1
                bi = find_break(df, s["t0"], lv["price"], side, base.FORWARD_HOURS, ed.BREAK_TOLERANCE_PCT)
                if bi is not None:
                    n_real_breaks += 1
                    for h in CONTINUATION_HOURS:
                        real_cont[h].append(continuation_return(df, bi, lv["price"], side, h))
                pd_ = rng.choice(pool)
                pp = cp * (1 + pd_ / 100.0)
                bi2 = find_break(df, s["t0"], pp, side, base.FORWARD_HOURS, ed.BREAK_TOLERANCE_PCT)
                if bi2 is not None:
                    n_placebo_breaks += 1
                    for h in CONTINUATION_HOURS:
                        placebo_cont[h].append(continuation_return(df, bi2, pp, side, h))
        out[side] = {
            "n_real_levels": n_real_levels, "n_real_breaks": n_real_breaks, "n_placebo_breaks": n_placebo_breaks,
            "break_rate_real": n_real_breaks / n_real_levels if n_real_levels else None,
            "continuation": {},
        }
        for h in CONTINUATION_HOURS:
            boot = bootstrap_ci(real_cont[h], placebo_cont[h], N_BOOT, rng)
            out[side]["continuation"][str(h)] = {
                "n_real": len(real_cont[h]), "n_placebo": len(placebo_cont[h]),
                "mean_real_pct": float(np.mean(real_cont[h]) * 100) if real_cont[h] else None,
                "mean_placebo_pct": float(np.mean(placebo_cont[h]) * 100) if placebo_cont[h] else None,
                "mean_diff_pct": float((np.mean(real_cont[h]) - np.mean(placebo_cont[h])) * 100) if real_cont[h] and placebo_cont[h] else None,
                "bootstrap_95ci": boot,
            }
    return out


def main() -> None:
    df = base.load_hourly()
    n = len(df)
    print(f"hourly bars: {n}", flush=True)
    snapshots = ed.simulate(df)  # unmodified event-driven state machine, same as every prior round
    print(f"eval snapshots: {len(snapshots)}", flush=True)

    def report(label: str, snaps: list[dict]) -> dict:
        result = evaluate_breakout_continuation(df, snaps, np.random.default_rng(SEED))
        print(f"\n=== {label} (n_snapshots={len(snaps)}) ===")
        for side, data in result.items():
            print(f"  {side:11s} break_rate_real={data['break_rate_real']}"
                  f" (n_real_breaks={data['n_real_breaks']}/{data['n_real_levels']},"
                  f" n_placebo_breaks={data['n_placebo_breaks']})")
            for h, c in data["continuation"].items():
                print(f"    {h}h  real={c['mean_real_pct']} placebo={c['mean_placebo_pct']} "
                      f"diff={c['mean_diff_pct']} 95%CI={c['bootstrap_95ci']}")
        return result

    full = report("FULL SAMPLE (primary test)", snapshots)

    split_i = n // 2
    first_half = report("FIRST HALF (consistency check)", [s for s in snapshots if s["t0"] < split_i])
    second_half = report("SECOND HALF (consistency check)", [s for s in snapshots if s["t0"] >= split_i])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"full": full, "first_half": first_half, "second_half": second_half},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
