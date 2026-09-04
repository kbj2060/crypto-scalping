#!/usr/bin/env python3
"""MAX_LOOKBACK_HOURS sweep for the event-driven liquidation-map state machine, 2026-08-25 --
user: "이 윈도우룰 바꿔보자. 이게 너무 길어" after seeing the live map's current window sit at 85h
(support_window_hours/resistance_window_hours in the just-shown chart).

Distinct lever from today's earlier MIN_FLOOR_HOURS sweep (research_eth_liquidation_map_event_
driven_min_floor_sweep_20260825.py, which only tested RAISING the floor 24->48->72 and was REJECTED
-- OOS favored the current 24h floor). MIN_FLOOR_HOURS is a LOWER bound (window >= floor) and also
the data-sufficiency safety margin (compute_raw_bins needs >=20 bars) -- lowering it further risks
starving the estimate, so it's left at ed.MIN_FLOOR_HOURS(24) here. MAX_LOOKBACK_HOURS is the
UPPER bound (window <= cap) and is what actually let today's live snapshot stretch to 85h despite
its own reset having happened only that recently -- it was never swept before (always fixed at 168h
since 2026-08-24). Candidates matched to something closer to the user's actual 1h holding period
without violating the 20-bar minimum: 24h (== MIN_FLOOR_HOURS, forces an effectively FIXED 24h
window with no event-driven variation left), 48h, 72h vs the current 168h baseline.

Uses TODAY's most rigorous, settled methodology: intrabar break check (research_eth_liquidation_
map_dwell_intrabar_break_test_20260825.evaluate_dwell_intrabar, not the older close-based ed.
evaluate()) and dwell duration (not a single fixed-window hold/break verdict), same TRAIN(80%)/OOS
(20%)-by-t0 split and placebo machinery as every other test today.
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
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_max_lookback_sweep_20260825.json"
SEED = ed.SEED
TRAIN_FRACTION = 0.8
CANDIDATES = [24, 48, 72]  # vs current live value 168 (baseline, always included for comparison)
BASELINE = 168


def simulate_param(df: pd.DataFrame, max_lookback_hours: int) -> list[dict]:
    """Copy of ed.simulate() with MAX_LOOKBACK_HOURS parameterized (MIN_FLOOR_HOURS fixed at
    ed.MIN_FLOOR_HOURS); same break/drift triggers, bootstrap, eval spacing as the 20260824 round."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = 0
    resistance_reset_idx = 0

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - max_lookback_hours)
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
        if broke_resistance or drift_resistance:
            resistance_levels = regenerate(resistance_reset_idx, i, "resistance_levels")
            resistance_reset_idx = i

        if i in eval_idxs:
            snapshots.append({
                "t0": i, "current_price": float(price),
                "support_levels": support_levels, "resistance_levels": resistance_levels,
                "support_window_hours": i - support_reset_idx, "resistance_window_hours": i - resistance_reset_idx,
            })
    return snapshots


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": intrabar.evaluate_dwell_intrabar(df, snaps, rng)}


def print_row(mlh: int, r: dict) -> None:
    for side in ("support", "resistance"):
        d = r["eval"][side]["0.005"]
        rr, pw = d["real"], d["paired_outdwell"]
        if not rr.get("n"):
            continue
        print(f"  MAX_LOOKBACK={mlh:4d}h [{r['split']:5s}] {side:11s} n={rr['n']:4d} "
              f"mean_dwell={rr['mean_dwell']:5.2f}h pairWR={str(pw['winrate'])[:6]:8s} "
              f"({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")


def main() -> None:
    df = base.load_hourly()
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    print(f"hourly bars: {n}, split at bar {split_i} ({df['timestamp'].iloc[split_i]})", flush=True)

    all_candidates = CANDIDATES + [BASELINE]
    result = {}
    window_stats = {}
    for mlh in all_candidates:
        snapshots = simulate_param(df, mlh)
        sw = [s["support_window_hours"] for s in snapshots]
        rw = [s["resistance_window_hours"] for s in snapshots]
        window_stats[mlh] = {"support_median_h": float(np.median(sw)), "resistance_median_h": float(np.median(rw))}
        train_snaps = [s for s in snapshots if s["t0"] < split_i]
        oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
        train_r = summarize_split("TRAIN", train_snaps, df, 0)
        oos_r = summarize_split("OOS", oos_snaps, df, 1)
        result[str(mlh)] = {"train": train_r, "oos": oos_r, "window_stats": window_stats[mlh]}
        print(f"\n=== MAX_LOOKBACK_HOURS={mlh} (median actual window: "
              f"support={window_stats[mlh]['support_median_h']:.0f}h resistance={window_stats[mlh]['resistance_median_h']:.0f}h) ===")
        print_row(mlh, train_r)
        print_row(mlh, oos_r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
