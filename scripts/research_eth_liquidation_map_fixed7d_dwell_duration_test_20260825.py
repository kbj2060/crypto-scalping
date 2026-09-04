#!/usr/bin/env python3
"""Does the FIXED-lookback liquidation map (compute_liquidation_levels(), recomputed fresh every
as-of point from a trailing 168h/7d window -- the "매봉마다 계산" style other traders' tools use,
per the user's 2026-08-25 question) also fail the same way the LIVE event-driven variant did in
today's dwell-duration test (eth_liquidation_map_dwell_duration_metric_rejected_20260825)?

This is the SAME judgment logic (dwell duration, user's own definition) applied to a DIFFERENT
level-generation mechanism -- not a new metric. Event-driven (compute_event_driven_levels) freezes
its level set between break/drift-triggered resets; this fixed variant recomputes from scratch at
every as-of point from whatever the trailing LOOKBACK_HOURS(168h) window looks like right then --
exactly mirroring dashboard/server.py::load_liquidation_map()'s fixed7d_* computation (currently
computed server-side but not rendered; the live chart/list both read the event-driven fields only,
per this session's earlier confirmation).

Reused unmodified: research_eth_liquidation_map_dwell_duration_test_20260825.evaluate_dwell() (the
exact same touch/placebo/dwell/paired machinery already validated today) and dwell_bars(). Only the
snapshot GENERATOR changes -- simulate_fixed() below replaces ed.simulate() with a call to
liqmap.compute_liquidation_levels() at each as-of point instead of the event-driven state machine.
Same eval grid (base.asof_indices with lookback=LOOKBACK_HOURS=168=ed.BOOTSTRAP_HOURS, same
FORWARD_HOURS/FOLLOWTHROUGH_HOURS defaults, base.FOLLOWTHROUGH_HOURS left unpatched at 24) so both
variants are scored on the IDENTICAL 572 as-of points / TRAIN-OOS split as today's event-driven
dwell test -- a clean, direct comparison, not a differently-scoped one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell

TRAIN_FRACTION = 0.8
SEED = 20260825


def simulate_fixed(df: pd.DataFrame) -> list[dict]:
    """Same snapshot shape as ed.simulate() ({"t0","current_price","support_levels",
    "resistance_levels"}), but each snapshot is computed fresh from liqmap.compute_liquidation_
    levels() on the trailing LOOKBACK_HOURS(168h) window ending at t0 -- exactly
    df.tail(LIQUIDATION_MAP_FIXED_LOOKBACK_HOURS) semantics from load_liquidation_map(), just
    re-run at every as-of point instead of once per live cache refresh."""
    n = len(df)
    close = df["close"].to_numpy()
    eval_idxs = base.asof_indices(n, liqmap.LOOKBACK_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    snapshots = []
    for i in eval_idxs:
        cp = float(close[i])
        start = max(0, i - liqmap.LOOKBACK_HOURS + 1)
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
    return {"split": split, "n_snapshots": len(snaps), "eval": dwell.evaluate_dwell(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"LOOKBACK_HOURS(fixed window)={liqmap.LOOKBACK_HOURS}, "
          f"DWELL_CAP_HOURS={dwell.DWELL_CAP_HOURS}", flush=True)

    snapshots = simulate_fixed(df)
    print(f"snapshots: {len(snapshots)} (event-driven dwell test had 572, for reference)", flush=True)

    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
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
