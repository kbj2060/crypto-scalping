#!/usr/bin/env python3
"""Event-driven liquidation-map sweep at MIN_FLOOR_HOURS=MAX_LOOKBACK_HOURS=N for N in
{12, 24, 48, 96}, 2026-08-25 user request: "이벤트 드리븐 12h, 24h, 48h, 96h 테스트 진행해줘."

Setting floor=ceiling=N collapses ed.simulate_param()'s clamp (see research_eth_liquidation_map_
max_lookback_sweep_20260825.py) to a pure fixed trailing N-hour window at every regeneration --
worked through algebraically: start = max(reset_idx, i-N); start = min(start, max(0, i-N)) always
resolves to start = i-N once i>=N, independent of reset_idx. This isolates exactly one axis: does
EVENT-DRIVEN TIMING (levels frozen until a break/drift reset fires, vs. always-fresh) help, when the
regeneration WINDOW WIDTH itself is held construction-equivalent to a fixed-window variant? Today's
already-tested stateless fixed48h/fixed168h (research_eth_liquidation_map_fixed48h_dwell_intrabar_
test_20260825.py, ..._fixed7d_dwell_intrabar_break_test_20260825.py) are the "always-fresh" anchors
this compares against at the two N where both exist (48h, matching close enough to compare shape).

N=12 sits BELOW the current live floor (ed.MIN_FLOOR_HOURS=24) -- without also lowering the floor
for that run, a 12h ceiling would just get overridden back up to 24h by the fixed floor (confirmed
by the same clamp algebra above: floor > ceiling always wins). Hence floor is swept together with
ceiling here, not held fixed like the two earlier same-day sweeps (MAX_LOOKBACK_HOURS-only and
BREAK_TOLERANCE_PCT-only, both eth_liquidation_map_staleness_tuning_rejected_20260825) which only
varied one bound at a time within the then-always-valid floor=24 regime.

BREAK_TOLERANCE_PCT/DRIFT_TOLERANCE_PCT/BOOTSTRAP_HOURS/SEED held at ed's production values across
all 4 candidates -- that reset-trigger axis was already swept and rejected today
(eth_liquidation_map_staleness_tuning_rejected_20260825), so holding it fixed here keeps this a
single-variable (window width) sweep.

Uses today's established rigor: intrabar break check, dwell duration, TRAIN(80%)/OOS(20%)-by-t0
split, distance-matched placebo -- identical methodology to every other liquidation-map test today.
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
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_event_driven_window_sweep_20260825.json"
SEED = ed.SEED
TRAIN_FRACTION = 0.8
CANDIDATES = [12, 24, 48, 96]


def simulate_param(df: pd.DataFrame, window_hours: int) -> list[dict]:
    """Copy of ed.simulate() with MIN_FLOOR_HOURS=MAX_LOOKBACK_HOURS=window_hours (both bounds tied
    to the same value -- see module docstring for why this collapses to a pure fixed-N-hour window
    at every regeneration while preserving event-driven freeze-until-reset timing). Break/drift
    triggers, bootstrap, eval spacing unchanged from ed's production values."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = 0
    resistance_reset_idx = 0

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - window_hours)
        start = min(start, max(0, i - window_hours))
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
    n_resets_support = n_resets_resistance = 0
    for i in range(ed.BOOTSTRAP_HOURS + 1, n):
        price = close[i]
        broke_support = any(price < lv["price"] * (1 - ed.BREAK_TOLERANCE_PCT) for lv in support_levels)
        broke_resistance = any(price > lv["price"] * (1 + ed.BREAK_TOLERANCE_PCT) for lv in resistance_levels)
        drift_support = bool(support_levels) and \
            (price - max(lv["price"] for lv in support_levels)) / price > ed.DRIFT_TOLERANCE_PCT
        drift_resistance = bool(resistance_levels) and \
            (min(lv["price"] for lv in resistance_levels) - price) / price > ed.DRIFT_TOLERANCE_PCT

        if broke_support or drift_support:
            support_levels = regenerate(support_reset_idx, i, "support_levels")
            support_reset_idx = i
            n_resets_support += 1
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i
            n_resets_resistance += 1

        if i in eval_idxs:
            snapshots.append({
                "t0": i, "current_price": float(price),
                "support_levels": support_levels, "resistance_levels": resistance_levels,
                "support_window_hours": i - support_reset_idx, "resistance_window_hours": i - resistance_reset_idx,
            })
    return snapshots, n_resets_support, n_resets_resistance


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": intrabar.evaluate_dwell_intrabar(df, snaps, rng)}


def print_row(w: int, r: dict) -> None:
    for side in ("support", "resistance"):
        d = r["eval"][side]["0.005"]
        rr, pw = d["real"], d["paired_outdwell"]
        if not rr.get("n"):
            continue
        print(f"  N={w:3d}h [{r['split']:5s}] {side:11s} n={rr['n']:4d} "
              f"mean_dwell={rr['mean_dwell']:5.2f}h pairWR={str(pw['winrate'])[:6]:8s} "
              f"({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")


def main() -> None:
    df = base.load_hourly()
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    print(f"hourly bars: {n}, split at bar {split_i} ({df['timestamp'].iloc[split_i]})", flush=True)
    print(f"CANDIDATES (floor=ceiling=N, pure fixed-N-hour regeneration window): {CANDIDATES}", flush=True)

    result = {}
    for w in CANDIDATES:
        snapshots, n_resets_support, n_resets_resistance = simulate_param(df, w)
        sw = [s["support_window_hours"] for s in snapshots]
        rw = [s["resistance_window_hours"] for s in snapshots]
        train_snaps = [s for s in snapshots if s["t0"] < split_i]
        oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
        train_r = summarize_split("TRAIN", train_snaps, df, 0)
        oos_r = summarize_split("OOS", oos_snaps, df, 1)
        result[str(w)] = {
            "train": train_r, "oos": oos_r,
            "n_resets_support": n_resets_support, "n_resets_resistance": n_resets_resistance,
            "staleness_median_h": {"support": float(np.median(sw)), "resistance": float(np.median(rw))},
        }
        print(f"\n=== N={w}h (resets: support={n_resets_support} resistance={n_resets_resistance}, "
              f"median staleness: support={np.median(sw):.0f}h resistance={np.median(rw):.0f}h) ===")
        print_row(w, train_r)
        print_row(w, oos_r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
