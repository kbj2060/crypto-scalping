"""Per-asset fine optimisation of the redesigned JM regime3 detector at a fixed feature count.

Asset-generalised form of scripts/optimize_btc_jm_m12_20260810.py (which was written for BTC while
its run was already in flight, so it stays as-is rather than being edited underneath a live job).
Same four gaps closed, same protocol; only the asset and the m grid differ:

  BTC  m in {8, 12}   ranked search peaked at m=12 (OOS .9371) with m=8 a hair behind (.9322) but
                      neutral rather than negative on economic separation
  ETH  m in {6, 8, 12} ranked search peaked at m=6 (OOS .9310, seed-stable to 45 sigma) but that is
                      also its WORST point on economic separation (OOS t -2.08 vs -0.59 at m=8),
                      so the peak is not automatically the pick

What the ranked sweep left un-optimised, and this closes:
  1. temperature sat on the grid's lower edge -> extended down to 0.02
  2. K was only {3, 4}, winner took the lower end -> extended to {2, 3, 4, 5, 6}
  3. lambda was a coarse 7-point log grid -> refined to 10 points over 0.25-8
  4. "which m features" was never searched, only two ranking prefixes -> adds greedy forward
     selection, a wrapper that scores candidates through the real JM pipeline so feature
     interactions are visible (on BTC it picked up ofi_acceleration / smart_money_flow, which
     univariate ANOVA-F ranks far too low to ever reach a top-m prefix)

Every configuration is fitted with 5 randomly-drawn seeds; gates are applied to the WORST seed and
selection uses the MEAN VAL balanced accuracy, with the across-seed std reported, so a lead
smaller than its own initialisation noise cannot win.

Protocol: fit and select features on 2024, tune on 2025-09..12, and 2026-01..03 is scored but
never consulted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    EVAL_WINDOWS, FIT_YEAR, LABEL_BASES, MIN_CLASS_COVERAGE, MIN_MEDIAN_RUN_BARS,
    SELECTION_WINDOW, _class_proba, _state_class_matrix, causal_decode_V, fit_jm, softmax_states,
    window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.ranked_jm_feature_selection_20260810 import load_pool, rankings_for  # noqa: E402

DEFAULT_M_GRID = {"btc": (8, 12), "eth": (6, 8, 12)}
K_GRID = (2, 3, 4, 5, 6)
LAMBDA_PER_DIM_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SCALERS = ("standard", "robust")
TEMPERATURE_RATIO_GRID = (0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0)
# Drawn at random, not as a fixed-increment ladder: a ladder off one base value produces clustered
# draws that look diverse and are not, which this project was burned by in the 2026-08-01 audit.
SEEDS = (7529, 481003, 26611, 903174, 155827)

GREEDY_K, GREEDY_LPD, GREEDY_SCALER = 3, 2.0, "standard"
GREEDY_TEMP_RATIO, GREEDY_BASIS = 1.0, "qmatched"
GREEDY_N_INIT, GREEDY_N_ITER = 1, 6
FULL_N_INIT, FULL_N_ITER = 3, 10

# set from argv before any pool is created; fork-based workers inherit it
ASSET = "btc"


def _pipeline_balanced_accuracy(idx, k, lpd, scaler, seed, temp_ratio, basis, n_init, n_iter):
    pool = load_pool(ASSET, scaler)
    x = pool[f"x_{FIT_YEAR}"][:, list(idx)]
    y = pool[f"y_{basis}_{FIT_YEAR}"]
    lam = lpd * len(idx)
    mu, _ = fit_jm(x, k=k, lam=lam, seed=seed, n_init=n_init, n_iter=n_iter)
    V = causal_decode_V(x, mu, lam)
    spread = max(float(np.median(V.max(axis=1) - V.min(axis=1))), 1e-9)
    sp = softmax_states(V, temp_ratio * spread)
    proba = _class_proba(sp, _state_class_matrix(sp, y))
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, np.argmax(proba, axis=1)))


def _greedy_step(args):
    chosen, cand = args
    return cand, _pipeline_balanced_accuracy(tuple(chosen) + (cand,), GREEDY_K, GREEDY_LPD,
                                             GREEDY_SCALER, SEEDS[0], GREEDY_TEMP_RATIO,
                                             GREEDY_BASIS, GREEDY_N_INIT, GREEDY_N_ITER)


def greedy_forward(workers: int, m_max: int) -> list[int]:
    """Nested by construction, so one run to the largest m supplies every smaller m as a prefix."""
    pool = load_pool(ASSET, GREEDY_SCALER)
    chosen: list[int] = []
    remaining = list(range(pool["dim"]))
    for step in range(m_max):
        jobs = [(list(chosen), c) for c in remaining]
        best_c, best_s = None, -np.inf
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c, s in ex.map(_greedy_step, jobs, chunksize=4):
                if s > best_s:
                    best_c, best_s = c, s
        chosen.append(int(best_c))
        remaining.remove(int(best_c))
        print(f"  greedy step {step + 1:>2}/{m_max}: +{pool['cols'][best_c]:<32} "
              f"in-sample bal_acc={best_s:.4f}", flush=True)
    return chosen


def evaluate_config(args) -> dict:
    fset_name, m, idx, scaler, k, lpd, seed = args
    pool = load_pool(ASSET, scaler)
    cols_idx = list(idx)
    lam = lpd * len(cols_idx)
    t0 = time.time()
    mu, obj = fit_jm(pool[f"x_{FIT_YEAR}"][:, cols_idx], k=k, lam=lam, seed=seed,
                     n_init=FULL_N_INIT, n_iter=FULL_N_ITER)
    fit_secs = time.time() - t0
    V = {y: causal_decode_V(pool[f"x_{y}"][:, cols_idx], mu, lam) for y in ("2024", "2025", "2026")}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    cells = []
    for ratio in TEMPERATURE_RATIO_GRID:
        sp = {y: softmax_states(Vv, ratio * spread) for y, Vv in V.items()}
        for basis in LABEL_BASES:
            state_class = _state_class_matrix(sp[FIT_YEAR], pool[f"y_{basis}_{FIT_YEAR}"])
            windows = {}
            for name, (yr, _s, _e) in EVAL_WINDOWS.items():
                mask = pool["window_masks"][name]
                proba = _class_proba(sp[yr][mask], state_class)
                windows[name] = window_metrics(np.argmax(proba, axis=1).astype(np.int64),
                                               pool[f"y_{basis}_{yr}"][mask],
                                               pool[f"close_{yr}"][mask])
            cells.append({"temperature_ratio": ratio, "label_basis": basis, "windows": windows})
    return {"feature_set": fset_name, "m": m, "scaler": scaler, "k": k, "lambda_per_dim": lpd,
            "lambda": lam, "seed": seed, "fit_objective": obj,
            "live_states_on_fit": int(len(np.unique(np.argmin(V[FIT_YEAR], axis=1)))),
            "v_spread_fit_median": spread, "fit_seconds": round(fit_secs, 1), "cells": cells}


def aggregate(results: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in results:
        for c in r["cells"]:
            key = (r["feature_set"], r["m"], r["scaler"], r["k"], r["lambda_per_dim"],
                   c["temperature_ratio"], c["label_basis"])
            buckets.setdefault(key, []).append({"seed": r["seed"], "windows": c["windows"],
                                                "live_states": r["live_states_on_fit"]})
    rows = []
    sel = SELECTION_WINDOW
    for key, entries in buckets.items():
        fset, m, scaler, k, lpd, ratio, basis = key
        row = {"feature_set": fset, "m": m, "scaler": scaler, "k": k, "lambda_per_dim": lpd,
               "temperature_ratio": ratio, "label_basis": basis, "n_seeds": len(entries),
               "live_states_min": min(e["live_states"] for e in entries)}
        for wname in EVAL_WINDOWS:
            for metric in ("balanced_accuracy", "median_run_bars", "min_class_coverage",
                           "economic_separation_tstat", "flip_rate"):
                vals = np.array([e["windows"][wname][metric] for e in entries], dtype=float)
                row[f"{wname}_{metric}"] = float(vals.mean())
                if metric in ("balanced_accuracy", "economic_separation_tstat"):
                    row[f"{wname}_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    row[f"{wname}_{metric}_min"] = float(vals.min())
        row["gate_median_run_min"] = float(min(e["windows"][sel]["median_run_bars"] for e in entries))
        row["gate_coverage_min"] = float(min(e["windows"][sel]["min_class_coverage"] for e in entries))
        row["passes_gates"] = bool(row["gate_median_run_min"] >= MIN_MEDIAN_RUN_BARS
                                   and row["gate_coverage_min"] >= MIN_CLASS_COVERAGE)
        rows.append(row)
    return rows


def main() -> None:
    global ASSET
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["btc", "eth"])
    ap.add_argument("--m", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--lambda-per-dim", type=float, nargs="+", default=None,
                    help="override the lambda grid (used to extend below the default floor)")
    ap.add_argument("--k", type=int, nargs="+", default=None, help="override the K grid")
    ap.add_argument("--reuse-feature-sets-from", type=Path, default=None,
                    help="take greedy feature sets from a finished report instead of re-running "
                         "the (expensive) forward selection")
    ap.add_argument("--tag", default=None, help="suffix for the output report filename")
    args = ap.parse_args()
    ASSET = args.asset
    m_grid = tuple(args.m) if args.m else DEFAULT_M_GRID[ASSET]
    lam_grid = tuple(args.lambda_per_dim) if args.lambda_per_dim else LAMBDA_PER_DIM_GRID
    k_grid = tuple(args.k) if args.k else K_GRID

    pool = load_pool(ASSET, "standard")
    cols = pool["cols"]
    rk = rankings_for(ASSET, "standard")
    m_max = max(m_grid)
    greedy_full = None
    if args.reuse_feature_sets_from is not None:
        prev = json.loads(args.reuse_feature_sets_from.read_text())["feature_sets"]
        key = f"greedy_fwd{m_max}"
        if key in prev:
            greedy_full = [cols.index(f) for f in prev[key]]
            print(f"=== {ASSET}: reusing greedy feature set from "
                  f"{args.reuse_feature_sets_from.name}", flush=True)
    elif not args.skip_greedy:
        print(f"=== {ASSET}: greedy forward selection to m={m_max} "
              f"(2024 fit window only, K={GREEDY_K} lpd={GREEDY_LPD} {GREEDY_SCALER})", flush=True)
        t0 = time.time()
        greedy_full = greedy_forward(args.workers, m_max)
        print(f"  greedy done in {time.time() - t0:.0f}s", flush=True)

    fsets: dict[str, tuple[int, list[int]]] = {}
    for m in m_grid:
        fsets[f"f_rank_top{m}"] = (m, [int(i) for i in rk["f_rank"][:m]])
        fsets[f"mrmr_top{m}"] = (m, [int(i) for i in rk["mrmr"][:m]])
        if greedy_full is not None:
            fsets[f"greedy_fwd{m}"] = (m, [int(i) for i in greedy_full[:m]])
    for name, (m, idx) in fsets.items():
        print(f"  {name}: {[cols[i] for i in idx]}", flush=True)

    jobs = [(name, m, tuple(idx), sc, k, lpd, seed)
            for name, (m, idx) in fsets.items()
            for sc in SCALERS for k in k_grid for lpd in lam_grid for seed in SEEDS]
    n_cells = len(jobs) // len(SEEDS) * len(TEMPERATURE_RATIO_GRID) * len(LABEL_BASES)
    print(f"\n=== {ASSET}: {len(jobs)} JM fits ({len(SEEDS)} seeds x "
          f"{len(jobs)//len(SEEDS)} configs) -> {n_cells} seed-averaged cells "
          f"({args.workers} workers)", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(evaluate_config, jobs, chunksize=5), 1):
            results.append(res)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)", flush=True)

    report = {
        "asset": ASSET, "m_grid": list(m_grid),
        "feature_sets": {n: [cols[i] for i in idx] for n, (m, idx) in fsets.items()},
        "grid": {"m": list(m_grid), "k": list(k_grid),
                 "lambda_per_dim": list(lam_grid), "scalers": list(SCALERS),
                 "temperature_ratio": list(TEMPERATURE_RATIO_GRID),
                 "label_bases": list(LABEL_BASES), "seeds": list(SEEDS)},
        "protocol": {"fit_year": FIT_YEAR, "selection_window": SELECTION_WINDOW,
                     "eval_windows": {k: list(v) for k, v in EVAL_WINDOWS.items()},
                     "n_init": FULL_N_INIT, "n_iter": FULL_N_ITER,
                     "gates_use_worst_seed": True, "greedy_selection_window": FIT_YEAR,
                     "fresh_forward_bar_by_bar": True,
                     "trade_ledgers_used_as_input": False,
                     "saved_parent_exit_timestamps_used": False,
                     "future_rows_used_for_entry": False},
        "cells": aggregate(results),
    }
    suffix = f"_{args.tag}" if args.tag else ""
    path = OUT_DIR / f"optimize_{ASSET}_m{'_'.join(str(m) for m in m_grid)}{suffix}_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  -> {path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
