#!/usr/bin/env python3
"""Multi-fold re-verification of v1_mid (entry price = (high+low)/2, see
research_eth_liquidation_map_entry_price_isolated_ab_20260826.py) across several DIFFERENT time
periods, 2026-08-26 user follow-up ("다른 기간으로 재검증해줘") after the survivor-pool-asymmetry
diagnostic found no structural explanation for resistance's cross-variant fragility, leaving open
whether v1_mid's own single 80/20 TRAIN/OOS split (support improved, resistance mixed) reflects a
real, stable effect or that one particular split's luck.

No new data: reuses the identical 290-point eval grid (same WARMUP_HOURS/FORWARD_HOURS/
FOLLOWTHROUGH_HOURS as every prior variant this thread) so results stay directly comparable, but
instead of one 80/20 TRAIN/OOS cut, breaks the eval points into N_FOLDS contiguous chronological
blocks and evaluates v1_live vs v1_mid INDEPENDENTLY within each block. If v1_mid's support edge is
real it should show up as a consistent direction across most/all folds, not just the single window
already tested; if it was that window's luck, it should look inconsistent/sign-flipping fold to
fold -- same logic as the project's established TRAIN-promising/OOS-rejected precedent, applied at
finer granularity.

Per-fold n is deliberately kept close to the original OOS n=62 (not thinner) so this isn't trading
one noisy small sample for four even noisier ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_v2_phase0_data_audit_20260825 as audit
import scripts.research_eth_liquidation_map_v1_direction_isolated_ab_20260826 as v1dir
import scripts.research_eth_liquidation_map_entry_price_isolated_ab_20260826 as epdir
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_v1_mid_walkforward_multifold_20260826.json"

N_FOLDS = 4
VARIANTS = ("v1_live", "v1_mid")


def main() -> None:
    px1h = v2ab.load_hourly_with_taker()
    m, clean = audit.load_metrics()
    df, join_stats = audit.hourly_join(m, px1h)
    print(f"join: {join_stats}", flush=True)

    epdir._identity_check(df)

    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    mid = (high + low) / 2.0
    entry_price = {"v1_live": close, "v1_mid": mid}

    n = len(df)
    eval_idxs = base.asof_indices(n, v1dir.WARMUP_HOURS, base.FORWARD_HOURS, base.FOLLOWTHROUGH_HOURS)
    print(f"bars={n} eval_points={len(eval_idxs)}", flush=True)

    all_snaps = {}
    for var in VARIANTS:
        all_snaps[var] = epdir.snapshots_v1_entry_price(df, eval_idxs, entry_price[var])
        print(f"{var} snapshots: {len(all_snaps[var])}", flush=True)

    fold_bounds = np.linspace(0, len(eval_idxs), N_FOLDS + 1).astype(int)
    folds = []
    for f in range(N_FOLDS):
        fold_eval_idxs = eval_idxs[fold_bounds[f]: fold_bounds[f + 1]]
        idx_lo, idx_hi = fold_eval_idxs[0], fold_eval_idxs[-1]
        folds.append({"fold": f, "t0_lo": idx_lo, "t0_hi": idx_hi,
                      "ts_lo": str(df["timestamp"].iloc[idx_lo]), "ts_hi": str(df["timestamp"].iloc[idx_hi])})
        print(f"fold {f}: {folds[-1]['ts_lo']} .. {folds[-1]['ts_hi']}  "
              f"n_eval_points={len(fold_eval_idxs)}", flush=True)

    results = []
    for k, (name, snaps) in enumerate(all_snaps.items()):
        for fold in folds:
            lo, hi = fold["t0_lo"], fold["t0_hi"]
            sel = [s for s in snaps if lo <= s["t0"] <= hi]
            rng = np.random.default_rng(20260826 + k * 100 + fold["fold"])
            ev = ed.evaluate(df, sel, rng)
            n_lv = [len(s["support_levels"]) + len(s["resistance_levels"]) for s in sel]
            results.append({
                "variant": name, "fold": fold["fold"], "ts_lo": fold["ts_lo"], "ts_hi": fold["ts_hi"],
                "n_snapshots": len(sel),
                "avg_levels_per_snapshot": round(float(np.mean(n_lv)), 2) if n_lv else 0.0,
                "eval": ev,
            })
            print(f"evaluated {name} fold{fold['fold']} (n={len(sel)})", flush=True)

    print(f"\n{'variant':8s} {'fold':5s} {'period':23s} {'side':11s} {'buf%':5s} {'pairWR':7s} "
          f"{'holdR':7s} {'holdP':7s} {'mag24d':8s} {'mag72d':8s} {'nTouch':6s}")
    for r_ in results:
        period = f"{r_['ts_lo'][:10]}~{r_['ts_hi'][:10]}"
        for side in ("support", "resistance"):
            d = r_["eval"][side]
            for buf in ("0.005",):
                row = d["by_buffer"][buf]
                mag24 = d["magnitude"]["24"]["mean_diff_pct"]
                mag72 = d["magnitude"]["72"]["mean_diff_pct"]
                print(f"{r_['variant']:8s} {r_['fold']:<5d} {period:23s} {side:11s} {float(buf)*100:4.1f} "
                      f"{str(row['paired']['winrate'])[:6]:7s} {str(row['real']['hold_rate'])[:6]:7s} "
                      f"{str(row['placebo']['hold_rate'])[:6]:7s} "
                      f"{('None' if mag24 is None else f'{mag24:+.3f}'):8s} "
                      f"{('None' if mag72 is None else f'{mag72:+.3f}'):8s} "
                      f"{row['real']['n_touched']:6d}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "n_folds": N_FOLDS, "folds": folds,
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
