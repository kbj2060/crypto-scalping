#!/usr/bin/env python3
"""2026-08-26 user follow-up: "시드 검증까지 마치고 나서 최종적으로 어떤 걸 쓰는 걸 추천하니?" --
closes out the placebo-RNG-seed sensitivity found in research_eth_liquidation_map_hybrid_price_
per_side_norm_multifold_20260826.py (same variant, different script runs, different seeds ->
noticeably different pairWR) by averaging ed.evaluate()'s placebo draw over N_SEEDS independent
seeds per cell instead of trusting any single draw. This is the final comparison meant to settle
v1_live vs v1_mid vs v1_spliced (support from v1_mid's own pipeline, resistance from v1_live's own
pipeline, research_eth_liquidation_map_spliced_hybrid_multifold_20260826.py) -- reuses those
scripts' snapshot-generation functions unmodified, only the evaluation/averaging is new.

Reports both POOLED (all 290 eval points as one group -- the single most relevant "how would this
look over the whole available history" read) and the same 4 chronological folds as before (for
regime-robustness context), each averaged over N_SEEDS placebo draws with mean AND std reported so
the residual noise floor is visible rather than hidden behind one more single-seed number.
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
import scripts.research_eth_liquidation_map_spliced_hybrid_multifold_20260826 as spliced
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "data" / "research" / "eth_liquidation_map_final_recommendation_seed_robust_20260826.json"

N_FOLDS = 4
N_SEEDS = 20
SEED_BASE = 900000
VARIANTS = ("v1_live", "v1_mid", "v1_spliced")
VARIANT_OFFSET = {"v1_live": 0, "v1_mid": 1, "v1_spliced": 2}  # deterministic, NOT hash() (randomized per-process)


def evaluate_multiseed(df: pd.DataFrame, snaps: list[dict], n_seeds: int, seed_base: int) -> dict:
    per_side = {"support": {"pairWR": [], "holdR": [], "holdP": [], "mag24": [], "mag72": []},
                "resistance": {"pairWR": [], "holdR": [], "holdP": [], "mag24": [], "mag72": []}}
    for s in range(n_seeds):
        rng = np.random.default_rng(seed_base + s)
        ev = ed.evaluate(df, snaps, rng)
        for side in ("support", "resistance"):
            row = ev[side]["by_buffer"]["0.005"]
            wr = row["paired"]["winrate"]
            if wr is not None:
                per_side[side]["pairWR"].append(wr)
            per_side[side]["holdR"].append(row["real"]["hold_rate"])
            per_side[side]["holdP"].append(row["placebo"]["hold_rate"])
            m24 = ev[side]["magnitude"]["24"]["mean_diff_pct"]
            m72 = ev[side]["magnitude"]["72"]["mean_diff_pct"]
            if m24 is not None:
                per_side[side]["mag24"].append(m24)
            if m72 is not None:
                per_side[side]["mag72"].append(m72)
    out = {}
    for side, d in per_side.items():
        out[side] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) if v else (None, None)
                     for k, v in d.items()}
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
    print(f"bars={n} eval_points={len(eval_idxs)}, {N_SEEDS} seeds/cell", flush=True)

    all_snaps = {
        "v1_live": epdir.snapshots_v1_entry_price(df, eval_idxs, close),
        "v1_mid": epdir.snapshots_v1_entry_price(df, eval_idxs, mid),
        "v1_spliced": spliced.snapshots_spliced(df, eval_idxs, mid),
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

    results = {"pooled": {}, "folds": {f["fold"]: {} for f in folds}}
    for var in VARIANTS:
        snaps = all_snaps[var]
        seed_base_pooled = SEED_BASE + VARIANT_OFFSET[var] * 100
        results["pooled"][var] = evaluate_multiseed(df, snaps, N_SEEDS, seed_base_pooled)
        print(f"pooled {var} done", flush=True)
        for fold in folds:
            lo, hi = fold["t0_lo"], fold["t0_hi"]
            sel = [s for s in snaps if lo <= s["t0"] <= hi]
            seed_base_fold = SEED_BASE + VARIANT_OFFSET[var] * 100 + (fold["fold"] + 1) * 10000
            results["folds"][fold["fold"]][var] = evaluate_multiseed(df, sel, N_SEEDS, seed_base_fold)
            print(f"fold{fold['fold']} {var} done (n={len(sel)})", flush=True)

    print(f"\n=== POOLED (all {len(eval_idxs)} eval points, {N_SEEDS}-seed avg) pairWR mean(std) ===", flush=True)
    for var in VARIANTS:
        r = results["pooled"][var]
        print(f"{var:12s} support={r['support']['pairWR'][0]}({r['support']['pairWR'][1]})  "
              f"resistance={r['resistance']['pairWR'][0]}({r['resistance']['pairWR'][1]})", flush=True)

    print(f"\n=== per-fold ({N_SEEDS}-seed avg) pairWR mean(std) ===", flush=True)
    for f in range(N_FOLDS):
        period = f"{folds[f]['ts_lo'][:10]}~{folds[f]['ts_hi'][:10]}"
        for var in VARIANTS:
            r = results["folds"][f][var]
            print(f"fold{f} {period:23s} {var:12s} support={r['support']['pairWR'][0]}({r['support']['pairWR'][1]})  "
                  f"resistance={r['resistance']['pairWR'][0]}({r['resistance']['pairWR'][1]})", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "join_stats": join_stats, "n_bars": n, "n_seeds": N_SEEDS, "n_folds": N_FOLDS, "folds": folds,
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
