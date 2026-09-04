#!/usr/bin/env python3
"""BREAK_TOLERANCE_PCT sweep for the event-driven liquidation-map state machine, 2026-08-25 --
this is the ACTUAL lever for the staleness the user reacted to ("85시간이 너무 길어"), not
MAX_LOOKBACK_HOURS (already tested and confirmed NOT to move the staleness number -- see
research_eth_liquidation_map_max_lookback_sweep_20260825.py's finding: median support/resistance
window stayed 44-54h across every MAX_LOOKBACK_HOURS candidate).

Mechanism: support_window_hours/resistance_window_hours ("staleness", how long since a side's
level set was last regenerated) is driven by how OFTEN the break/drift triggers fire, not by how
much data feeds a regeneration. BREAK_TOLERANCE_PCT (currently 0.5%, close crossing an active
level by this much triggers a reset) is the PRIMARY trigger in normal price action -- DRIFT_
TOLERANCE_PCT (10%) is a secondary/rare abandonment safeguard per its own docstring (only fires
when price wanders far from every active level, an edge case). Tightening BREAK_TOLERANCE_PCT
should directly shrink the reset gap; DRIFT_TOLERANCE_PCT and MIN_FLOOR/MAX_LOOKBACK_HOURS are
left at their current live values so only this one lever changes (today's established discipline
-- single-lever, theory-motivated sweeps only, per the MIN_FLOOR_HOURS and MAX_LOOKBACK_HOURS
sweeps' own docstrings).

Candidates: 0.001 (0.1%, tightest already used anywhere in this research line, as one of the
evaluation BUFFER_PCTS), 0.002, 0.003 vs the current live value 0.005 (0.5%) baseline. Note:
BREAK_TOLERANCE_PCT (the reset TRIGGER, this script's lever) and BUFFER_PCTS (the EVALUATION's
own hold/break scoring tolerance, still swept at {0.001,0.005} as always) are separate parameters
-- changing one does not change the other.

Same methodology as every sweep today: intrabar break check (dwell_intrabar), dwell duration,
TRAIN(80%)/OOS(20%)-by-t0 split, placebo comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_dwell_intrabar_break_test_20260825 as intrabar

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_break_tolerance_sweep_20260825.json"
SEED = ed.SEED
TRAIN_FRACTION = 0.8
CANDIDATES = [0.001, 0.002, 0.003]  # vs current live value 0.005 (baseline, always included)
BASELINE = 0.005


def simulate_param(df: pd.DataFrame, break_tolerance_pct: float) -> list[dict]:
    """Copy of ed.simulate() with BREAK_TOLERANCE_PCT parameterized; DRIFT_TOLERANCE_PCT,
    MIN_FLOOR_HOURS, MAX_LOOKBACK_HOURS, bootstrap, and eval spacing all stay at current live
    values (ed.* constants) -- single-lever sweep."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = 0
    resistance_reset_idx = 0

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - ed.MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - ed.MIN_FLOOR_HOURS))
        window = df.iloc[start:i + 1]
        cp = float(close[i])
        raw = liqmap.compute_raw_bins(window, cp)
        if raw is None:
            return []
        bins, bin_width, _, _ = raw
        return liqmap.levels_from_bins(bins, bin_width, cp)[key]

    support_levels = regenerate(0, ed.BOOTSTRAP_HOURS, "support_levels")
    resistance_levels = regenerate(0, ed.BOOTSTRAP_HOURS, "resistance_levels")

    eval_idxs = set(base.asof_indices(n, ed.BOOTSTRAP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS))
    snapshots = []
    n_support_resets = n_resistance_resets = 0
    for i in range(ed.BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - break_tolerance_pct) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + break_tolerance_pct) for lv in resistance_levels)
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > ed.DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > ed.DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels = regenerate(support_reset_idx, i, "support_levels")
            support_reset_idx = i
            n_support_resets += 1
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i
            n_resistance_resets += 1

        if i in eval_idxs:
            snapshots.append({
                "t0": i, "current_price": float(price),
                "support_levels": support_levels, "resistance_levels": resistance_levels,
                "support_window_hours": i - support_reset_idx, "resistance_window_hours": i - resistance_reset_idx,
            })
    print(f"    total resets over {n} bars: support={n_support_resets} resistance={n_resistance_resets}", flush=True)
    return snapshots


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": intrabar.evaluate_dwell_intrabar(df, snaps, rng)}


def print_row(btp: float, r: dict) -> None:
    for side in ("support", "resistance"):
        d = r["eval"][side]["0.005"]
        rr, pw = d["real"], d["paired_outdwell"]
        if not rr.get("n"):
            continue
        print(f"  BREAK_TOL={btp:.3f} [{r['split']:5s}] {side:11s} n={rr['n']:4d} "
              f"mean_dwell={rr['mean_dwell']:5.2f}h pairWR={str(pw['winrate'])[:6]:8s} "
              f"({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")


def main() -> None:
    df = base.load_hourly()
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    print(f"hourly bars: {n}, split at bar {split_i} ({df['timestamp'].iloc[split_i]})", flush=True)

    all_candidates = CANDIDATES + [BASELINE]
    result = {}
    for btp in all_candidates:
        print(f"\n=== BREAK_TOLERANCE_PCT={btp} ===", flush=True)
        snapshots = simulate_param(df, btp)
        sw = [s["support_window_hours"] for s in snapshots]
        rw = [s["resistance_window_hours"] for s in snapshots]
        window_stats = {"support_median_h": float(np.median(sw)), "resistance_median_h": float(np.median(rw))}
        print(f"    median staleness: support={window_stats['support_median_h']:.0f}h "
              f"resistance={window_stats['resistance_median_h']:.0f}h", flush=True)
        train_snaps = [s for s in snapshots if s["t0"] < split_i]
        oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
        train_r = summarize_split("TRAIN", train_snaps, df, 0)
        oos_r = summarize_split("OOS", oos_snaps, df, 1)
        result[str(btp)] = {"train": train_r, "oos": oos_r, "window_stats": window_stats}
        print_row(btp, train_r)
        print_row(btp, oos_r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
