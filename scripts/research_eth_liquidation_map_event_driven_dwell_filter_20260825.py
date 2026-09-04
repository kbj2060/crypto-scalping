#!/usr/bin/env python3
"""Dwell-time-filtered re-evaluation of the event-driven liquidation-map levels, 2026-08-25.

Diagnoses the magnitude weakness flagged in eth_liquidation_map_event_driven_reset_20260824: that
round's headline win-rate (60.1-67.5%) coexists with near-zero/negative reaction magnitude. Working
hypothesis there: a freshly-reset level sits close to current price by construction (it was just
regenerated near "wherever price is now"), so an immediate re-touch is close to guaranteed and
inflates touch-based win-rate without reflecting real structural relevance -- a level that survives
untouched for a while and THEN gets touched should be the more meaningful case.

Does NOT change the event-driven state machine at all -- reuses simulate()/evaluate() verbatim from
research_eth_liquidation_map_event_driven_reset_20260824 (same causal walk-forward, same touch/hold/
magnitude/placebo machinery). Each snapshot already carries support_window_hours/
resistance_window_hours (hours since that side's last reset at the moment of evaluation, uncapped --
MIN_FLOOR_HOURS/MAX_LOOKBACK_HOURS bound the RECOMPUTE window, not this counter). This script just
buckets the existing snapshots by that dwell time before calling the existing evaluate() on each
bucket, and separately reproduces the full-sample numbers as a sanity check that bucketing hasn't
silently changed anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_event_driven_dwell_filter_20260825.json"
SEED = ed.SEED  # match the 20260824 round's seed exactly so the full-sample numbers reproduce bit-for-bit

# 3 buckets, not finer -- the round this reuses had only ~150-190 paired episodes per side total
# (eth_liquidation_map_event_driven_reset_20260824), so finer buckets would mostly be sample-size
# noise. "fresh" is the zone the inflation hypothesis targets (near MIN_FLOOR_HOURS=24h); "stale" is
# past MAX_LOOKBACK_HOURS=168h/7d (the recompute window cap -- window_hours itself isn't clamped
# there, it just keeps counting until the next break/drift reset).
BUCKET_EDGES = [0, 48, 168, np.inf]
BUCKET_LABELS = ["fresh(<48h)", "established(48-168h)", "stale(168h+/7d+)"]


def bucket_snapshots(snapshots: list[dict], side: str) -> dict[str, list[dict]]:
    key = f"{side}_window_hours"
    out: dict[str, list[dict]] = {label: [] for label in BUCKET_LABELS}
    for s in snapshots:
        wh = s[key]
        for lo, hi, label in zip(BUCKET_EDGES, BUCKET_EDGES[1:], BUCKET_LABELS):
            if lo <= wh < hi:
                out[label].append(s)
                break
    return out


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}", flush=True)
    snapshots = ed.simulate(df)
    print(f"eval snapshots: {len(snapshots)}", flush=True)

    # Sanity check: full unfiltered snapshots through the unmodified evaluate() should reproduce
    # the 20260824 round's headline (support 60.1%/64.0%, resistance 67.5%/64.0% win-rate).
    full = ed.evaluate(df, snapshots, np.random.default_rng(SEED))
    print("\n=== full (unfiltered) sanity check vs 20260824 headline ===")
    for side, data in full.items():
        for buf, row in data["by_buffer"].items():
            print(f"{side:11s} buf={float(buf) * 100:5.1f}% winrate={row['paired']['winrate']}")
        print(f"{side:11s} magnitude: " + ", ".join(
            f"{h}h diff={data['magnitude'][h]['mean_diff_pct']:.3f}%" for h in data["magnitude"]))

    result: dict = {"full": full, "by_dwell_bucket": {}}
    for side in ("support", "resistance"):
        buckets = bucket_snapshots(snapshots, side)
        print(f"\n=== {side}: win-rate/magnitude by hours-since-reset bucket ===")
        result["by_dwell_bucket"][side] = {}
        for label in BUCKET_LABELS:
            sub = buckets[label]
            n_levels = sum(len(s[f"{side}_levels"]) for s in sub)
            if not sub or not n_levels:
                print(f"{label:24s} n_snapshots=0")
                continue
            ev = ed.evaluate(df, sub, np.random.default_rng(SEED))[side]
            row_01 = ev["by_buffer"].get("0.001", {}).get("paired", {})
            row_05 = ev["by_buffer"].get("0.005", {}).get("paired", {})
            mag24 = ev["magnitude"]["24"]["mean_diff_pct"]
            mag72 = ev["magnitude"]["72"]["mean_diff_pct"]
            print(f"{label:24s} n_snapshots={len(sub):4d} n_levels={n_levels:4d} "
                  f"winrate(0.1%)={row_01.get('winrate')} winrate(0.5%)={row_05.get('winrate')} "
                  f"mag24h={mag24} mag72h={mag72}")
            result["by_dwell_bucket"][side][label] = {
                "n_snapshots": len(sub), "n_levels": n_levels,
                "by_buffer": ev["by_buffer"], "magnitude": ev["magnitude"],
            }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
