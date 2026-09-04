#!/usr/bin/env python3
"""MIN_FLOOR_HOURS sweep for the event-driven liquidation-map state machine, 2026-08-25 follow-up
to eth_liquidation_map_event_driven_dwell_filter_20260825 (which found resistance win-rate AND
magnitude both improve in the 48-168h hours-since-reset bucket, while fresh<48h and stale>168h are
weaker; support shows no such rescue at any dwell time). This script tests whether actually raising
MIN_FLOOR_HOURS (currently 24h -- widens the trailing data window a reset's new levels are computed
from) shifts the deployed behavior toward that better zone, with a proper train/OOS split so the
result isn't just refit to the same data that produced the original diagnosis.

Split: continuous single simulation per candidate (matches how the live state machine actually runs
-- no artificial re-bootstrap at a split date), snapshots then partitioned by t0 into TRAIN (first
80% of bars) used to pick a winner, and OOS (last ~20%, ~11 months) used only to confirm it -- OOS
snapshots are never looked at before the TRAIN-based pick is locked in.

Reuses simulate()'s exact break/drift-trigger logic (only MIN_FLOOR_HOURS is parameterized here;
copied rather than editing research_eth_liquidation_map_event_driven_reset_20260824.py so that
script stays an unmodified historical record) and evaluate() verbatim (unmodified import).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_event_driven_min_floor_sweep_20260825.json"
SEED = ed.SEED
TRAIN_FRACTION = 0.8
CANDIDATES = [24, 48, 72]  # 24h = current live value (baseline); 48h = the dwell_filter finding's
                            # own bucket edge; 72h probed to see whether the trend keeps improving
                            # or 48h is closer to a peak. Kept small (3) -- this is a single-lever,
                            # theory-motivated sweep, not an unconstrained grid.


def simulate_param(df: pd.DataFrame, min_floor_hours: int) -> list[dict]:
    """Copy of ed.simulate() with MIN_FLOOR_HOURS parameterized; everything else (break/drift
    tolerance, max lookback, bootstrap, eval spacing) is unchanged from the 20260824 round."""
    n = len(df)
    close = df["close"].to_numpy()
    support_reset_idx = 0
    resistance_reset_idx = 0
    n_support_resets = 0
    n_resistance_resets = 0

    def regenerate(reset_idx: int, i: int, key: str) -> list[dict]:
        start = max(reset_idx, i - ed.MAX_LOOKBACK_HOURS)
        start = min(start, max(0, i - min_floor_hours))
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
    return snapshots


def summarize(df: pd.DataFrame, snapshots: list[dict]) -> dict:
    if not snapshots:
        return {"n_snapshots": 0}
    ev = ed.evaluate(df, snapshots, np.random.default_rng(SEED))
    out = {"n_snapshots": len(snapshots)}
    for side, data in ev.items():
        out[side] = {
            "winrate_01": data["by_buffer"]["0.001"]["paired"]["winrate"],
            "winrate_05": data["by_buffer"]["0.005"]["paired"]["winrate"],
            "mag24h": data["magnitude"]["24"]["mean_diff_pct"],
            "mag72h": data["magnitude"]["72"]["mean_diff_pct"],
        }
    return out


def main() -> None:
    df = base.load_hourly()
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    print(f"hourly bars: {n}, train/OOS split at bar {split_i} "
          f"({df['timestamp'].iloc[split_i]} -- OOS runs to {df['timestamp'].iloc[-1]})", flush=True)

    result = {}
    for mfh in CANDIDATES:
        print(f"\n=== MIN_FLOOR_HOURS={mfh} ===", flush=True)
        snapshots = simulate_param(df, mfh)
        train_snaps = [s for s in snapshots if s["t0"] < split_i]
        oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
        train_summary = summarize(df, train_snaps)
        oos_summary = summarize(df, oos_snaps)
        result[str(mfh)] = {"train": train_summary, "oos": oos_summary}
        for label, summ in (("TRAIN", train_summary), ("OOS", oos_summary)):
            print(f"  {label} n_snapshots={summ['n_snapshots']}")
            for side in ("support", "resistance"):
                if side not in summ:
                    continue
                r = summ[side]
                print(f"    {side:11s} winrate(0.1%/0.5%)={r['winrate_01']}/{r['winrate_05']} "
                      f"mag24h={r['mag24h']} mag72h={r['mag72h']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
