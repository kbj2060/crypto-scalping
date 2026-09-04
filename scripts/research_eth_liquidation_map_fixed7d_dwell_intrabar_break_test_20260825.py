#!/usr/bin/env python3
"""FIXED-7d liquidation map (compute_liquidation_levels(), recomputed fresh every as-of point --
see research_eth_liquidation_map_fixed7d_dwell_duration_test_20260825.py) scored with the
INTRABAR break check (see research_eth_liquidation_map_dwell_intrabar_break_test_20260825.py and
memory feedback_liquidation_barrier_intrabar_close_consistency) instead of close-based.

2026-08-25, user: "레벨 생성 단계과 판정을 청산맵 로직에 잘 기억하고 7일선도 테스트해줘" --
apply the same generation/judgment intrabar-consistency fix to the fixed-7d variant, mirroring
what was just done for event-driven. Pure glue: fixed7d.simulate_fixed() for level generation
(unmodified), intrabar.evaluate_dwell_intrabar() for scoring (unmodified) -- same pattern already
used for fixed7d's close-based test (which reused dwell.evaluate_dwell()).
"""
from __future__ import annotations

import numpy as np

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_fixed7d_dwell_duration_test_20260825 as fixed7d
import scripts.research_eth_liquidation_map_dwell_intrabar_break_test_20260825 as intrabar
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell

TRAIN_FRACTION = 0.8
SEED = 20260825


def summarize_split(split, snaps, df, seed_off):
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": intrabar.evaluate_dwell_intrabar(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print("break check: INTRABAR, level generation: FIXED 168h rolling (compute_liquidation_levels)", flush=True)

    snapshots = fixed7d.simulate_fixed(df)
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
