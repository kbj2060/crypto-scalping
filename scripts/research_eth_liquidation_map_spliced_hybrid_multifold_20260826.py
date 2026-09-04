#!/usr/bin/env python3
"""2026-08-26 user follow-up: "계산을 2개로 나눠서 mid로 지지를 쓰고 종가로 저항을 쓰는건?" --
different from compute_raw_bins_hybrid() (research_eth_liquidation_map_hybrid_price_multifold_
20260826.py), which merged mid-based long_liq and close-based short_liq into ONE shared bins dict
before levels_from_bins() -- that's exactly what caused the cross-side contamination (MIN_LEVEL_
SHARE normalized against a shared max, so a taller mid-driven support peak silently dropped
resistance bins that would otherwise have cleared the threshold).

This variant runs TWO FULLY INDEPENDENT computations end-to-end -- v1_mid's own complete
compute_raw_bins_entry_price()+levels_from_bins() pipeline (entry_price=mid throughout, exactly as
already validated in the 4-fold/regime tests) and v1_live's own complete liqmap.compute_raw_bins()+
levels_from_bins() pipeline (entry_price=close, production-identical) -- then splices: final
support_levels = the mid run's own support_levels, final resistance_levels = the close run's own
resistance_levels. Neither run's bins/max_weight is ever shared with the other, so there is no
mechanism for cross-contamination. By construction this should reproduce v1_mid's already-reported
support numbers and v1_live's already-reported resistance numbers essentially exactly (modulo the
placebo-RNG-seed sensitivity found in the prior hybrid_price_per_side_norm test) -- this run exists
to confirm that construction is bug-free and to have both halves' numbers side by side in one place.

Uses a FIXED per-fold seed shared across ALL variants (not offset by variant index k) so this run
also fixes the placebo-seed confound flagged in that prior test -- a fair apples-to-apples
comparison this time, not just a repeat of the same methodology.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.live_liquidation_map_20260824 as liqmap
import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir
import scripts.research_eth_liquidation_map_entry_price_isolated_ab_20260826 as epdir
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_spliced_hybrid_multifold_20260826.json"

N_FOLDS = 4
VARIANTS = ("v1_live", "v1_mid", "v1_spliced")


def snapshots_spliced(df: pd.DataFrame, eval_idxs: list[int], mid_full: np.ndarray) -> list[dict]:
    close = df["close"].to_numpy()
    out = []
    for i in eval_idxs:
        start = max(0, i - v1dir.LOOKBACK_HOURS_LIVE + 1)
        window = df.iloc[start:i + 1]
        cp = float(close[i])

        raw_mid = epdir.compute_raw_bins_entry_price(window, cp, mid_full[start:i + 1])
        raw_close = liqmap.compute_raw_bins(window, cp)
        if raw_mid is None or raw_close is None:
            continue
        bins_mid, bw_mid, _, _ = raw_mid
        bins_close, bw_close, _, _ = raw_close
        lv_mid = liqmap.levels_from_bins(bins_mid, bw_mid, cp)
        lv_close = liqmap.levels_from_bins(bins_close, bw_close, cp)
        out.append({"t0": i, "current_price": cp,
                    "support_levels": lv_mid["support_levels"], "resistance_levels": lv_close["resistance_levels"]})
    return out


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    mid = (high + low) / 2.0

    n = len(df)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} eval_points={len(eval_idxs)}", flush=True)

    all_snaps = {
        "v1_live": epdir.snapshots_v1_entry_price(df, eval_idxs, close),
        "v1_mid": epdir.snapshots_v1_entry_price(df, eval_idxs, mid),
        "v1_spliced": snapshots_spliced(df, eval_idxs, mid),
    }
    for var in VARIANTS:
        print(f"{var} snapshots: {len(all_snaps[var])}", flush=True)

    fold_bounds = np.linspace(0, len(eval_idxs), N_FOLDS + 1).astype(int)
    folds = []
    for f in range(N_FOLDS):
        fold_eval_idxs = eval_idxs[fold_bounds[f]: fold_bounds[f + 1]]
        idx_lo, idx_hi = fold_eval_idxs[0], fold_eval_idxs[-1]
        folds.append({"fold": f, "t0_lo": idx_lo, "t0_hi": idx_hi,
                      "ts_lo": str(df["timestamp"].iloc[idx_lo]), "ts_hi": str(df["timestamp"].iloc[idx_hi])})

    results = []
    for var in VARIANTS:
        snaps = all_snaps[var]
        for fold in folds:
            lo, hi = fold["t0_lo"], fold["t0_hi"]
            sel = [s for s in snaps if lo <= s["t0"] <= hi]
            rng = np.random.default_rng(20260826 + fold["fold"])  # FIXED per-fold seed, shared across variants
            ev = ed.evaluate(df, sel, rng)
            results.append({"variant": var, "fold": fold["fold"], "n_snapshots": len(sel), "eval": ev})
            print(f"evaluated {var} fold{fold['fold']} (n={len(sel)})", flush=True)

    print("\n=== fold-average pairWR (buf 0.5%%, FIXED per-fold seed across variants) ===", flush=True)
    for var in VARIANTS:
        for side in ("support", "resistance"):
            wrs = [r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] for r in results
                  if r["variant"] == var and r["eval"][side]["by_buffer"]["0.005"]["paired"]["winrate"] is not None]
            print(f"{var:12s} {side:11s} mean={np.mean(wrs):.3f}  per-fold={['%.3f' % w for w in wrs]}", flush=True)

    print("\n=== per-fold detail ===", flush=True)
    for r_ in results:
        for side in ("support", "resistance"):
            row = r_["eval"][side]["by_buffer"]["0.005"]
            print(f"{r_['variant']:12s} fold{r_['fold']} {side:11s} pairWR={row['paired']['winrate']} "
                  f"holdR={row['real']['hold_rate']} holdP={row['placebo']['hold_rate']} "
                  f"nTouch={row['real']['n_touched']}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "n_folds": N_FOLDS, "folds": folds, "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
