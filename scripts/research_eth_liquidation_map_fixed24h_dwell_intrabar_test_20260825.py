#!/usr/bin/env python3
"""TRAIN/OOS validation of the STATELESS (always-fresh) mechanism at a 24h fixed window --
compute_liquidation_levels() recomputed fresh at every snapshot from a plain trailing 24h tail, no
reset state machine. Direct sibling of research_eth_liquidation_map_fixed48h_dwell_intrabar_test_
20260825.py (48h) and ..._fixed7d_dwell_intrabar_break_test_20260825.py (168h) -- only LOOKBACK_
HOURS changes.

2026-08-25 user request: after seeing the event-driven floor=ceiling=24h sweep result (research_
eth_liquidation_map_event_driven_window_sweep_20260825.py: support OOS pairWR 0.375, resistance
0.660), user asked "이 2가지 중 어느 것이 더 정확도가 좋지?" (which of these two [stateless-24h vs
event-driven-24h] has better accuracy) -- this script produces the missing stateless-24h side of
that comparison, which had not been run before (only stateless 48h/168h existed).

Uses today's established rigor: intrabar break check, dwell duration, TRAIN(80%)/OOS(20%)-by-t0
split, distance-matched placebo -- identical methodology to every other liquidation-map test today.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell
import scripts.research_eth_liquidation_map_dwell_intrabar_break_test_20260825 as intrabar

LOOKBACK_HOURS = 24
TRAIN_FRACTION = 0.8
SEED = 20260825


def simulate_fixed_24h(df: pd.DataFrame) -> list[dict]:
    """Same shape as the 48h/168h siblings -- compute_liquidation_levels() recomputed fresh from a
    plain LOOKBACK_HOURS(24) tail at every as-of point."""
    n = len(df)
    close = df["close"].to_numpy()
    eval_idxs = base.asof_indices(n, LOOKBACK_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    snapshots = []
    for i in eval_idxs:
        cp = float(close[i])
        start = max(0, i - LOOKBACK_HOURS + 1)
        window = df.iloc[start:i + 1]
        payload = liqmap.compute_liquidation_levels(window, cp)
        if not payload["warmed_up"]:
            continue
        snapshots.append({"t0": i, "current_price": cp,
                          "support_levels": payload["support_levels"],
                          "resistance_levels": payload["resistance_levels"]})
    return snapshots


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": intrabar.evaluate_dwell_intrabar(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"LOOKBACK_HOURS={LOOKBACK_HOURS} (fixed, no state machine), break check: INTRABAR", flush=True)

    snapshots = simulate_fixed_24h(df)
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"snapshots: {len(snapshots)}  split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
          f"TRAIN={len(train_snaps)}, OOS={len(oos_snaps)}", flush=True)

    results = [summarize_split("TRAIN", train_snaps, df, 0), summarize_split("OOS", oos_snaps, df, 1)]

    for r in results:
        print(f"\n{'='*100}\n{r['split']} (n_snapshots={r['n_snapshots']})\n{'='*100}")
        for side in ("support", "resistance"):
            for buf in ("0.005", "0.001"):
                d = r["eval"][side][buf]
                rr, pp, pw = d["real"], d["placebo"], d["paired_outdwell"]
                if not rr.get("n"):
                    print(f"  [{side} buf={float(buf)*100:.1f}%] n=0, skipped")
                    continue
                print(f"\n[{side} buf={float(buf)*100:.1f}%]")
                print(f"  real:    n={rr['n']:4d} mean_dwell={rr['mean_dwell']:5.2f}h "
                      f"median={rr['median_dwell']:4.1f}h censored={rr['censored_pct']:5.1f}%")
                print(f"  placebo: n={pp['n']:4d} mean_dwell={pp['mean_dwell']:5.2f}h "
                      f"median={pp['median_dwell']:4.1f}h censored={pp['censored_pct']:5.1f}%")
                surv_r = "  ".join(f"{k}h:{rr['survival_pct'][k]:.0f}%" for k in map(str, dwell.SURVIVAL_CHECKPOINTS))
                surv_p = "  ".join(f"{k}h:{pp['survival_pct'][k]:.0f}%" for k in map(str, dwell.SURVIVAL_CHECKPOINTS))
                print(f"  survival% real:    {surv_r}")
                print(f"  survival% placebo: {surv_p}")
                print(f"  paired out-dwell winrate: {pw['winrate']} ({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")


if __name__ == "__main__":
    main()
