"""BTC regime3 JM: fine optimisation with the feature count fixed at 12.

The ranked search established m=12 as the peak of the feature-count curve (OOS balanced accuracy
.937 at m=12 vs .904 at m=24 and .790 at m=130), and it swept K/lambda/scaler/temperature jointly
rather than in isolation. But four things were left un-optimised, and this script closes each:

  1. TEMPERATURE SAT ON THE GRID EDGE. The winner picked the lowest ratio offered (0.25), so the
     optimum may lie below it and was simply not reachable. Extended down to 0.02.
  2. K WAS ONLY {3, 4}. The winner chose K=3, the lower end. Extended to {2, 3, 4, 5, 6}.
  3. LAMBDA WAS A COARSE 7-POINT LOG GRID. The winner sat at lambda_per_dim=2 with the neighbouring
     points a factor of 2-4 away. Refined to 10 points across 0.25-8.
  4. "WHICH 12" WAS NEVER SEARCHED -- only two ranking PREFIXES (ANOVA-F top-12, mRMR top-12) were
     tried. This adds a greedy forward-selected 12, a wrapper method that scores each candidate
     through the actual JM pipeline instead of a univariate proxy, so interactions between
     features can be seen. Selection runs on the 2024 fit window only.

It also fixes a power problem the sweep had: every JM fit there used a single seed, and the fit is
seed-dependent through its k-means++ initialisation, so a config could win on an initialisation
draw. Here every configuration is fitted with 5 randomly-drawn (not fixed-increment) seeds and
selected on the MEAN VAL balanced accuracy, with the across-seed standard deviation reported --
a config whose lead is smaller than its own seed noise is not a lead.

Protocol unchanged: fit and select features on 2024, choose hyperparameters on 2025-09..12, and
2026-01..03 is scored but never consulted.
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

ASSET = "btc"
# Both peak candidates are carried through the whole optimisation rather than committing to one:
# on the ranked search m=12 won on agreement (OOS .9371) while m=8 was a hair behind (.9322) but
# neutral instead of negative on economic separation, and that trade-off is not resolvable from
# the coarse grid alone.
M_GRID = (8, 12)
K_GRID = (2, 3, 4, 5, 6)
LAMBDA_PER_DIM_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SCALERS = ("standard", "robust")
TEMPERATURE_RATIO_GRID = (0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0)
# 5 seeds drawn at random rather than as a fixed-increment ladder: a ladder off one base value
# produces clustered draws that look diverse and are not, which this project has already been
# burned by once (the 2026-08-01 Sigma3-1h audit).
SEEDS = (7529, 481003, 26611, 903174, 155827)

# greedy forward selection reference config -- held fixed so the search ranks FEATURES, not
# hyperparameters; the full grid re-tunes afterwards on whatever set it returns
GREEDY_K = 3
GREEDY_LPD = 2.0
GREEDY_SCALER = "standard"
GREEDY_TEMP_RATIO = 1.0
GREEDY_BASIS = "qmatched"
GREEDY_N_INIT, GREEDY_N_ITER = 1, 6
FULL_N_INIT, FULL_N_ITER = 3, 10


def _score_in_sample(idx: tuple[int, ...], k: int, lpd: float, scaler: str, seed: int,
                     temp_ratio: float, basis: str, n_init: int, n_iter: int) -> float:
    """Balanced accuracy of the full JM -> state_class -> argmax pipeline on the FIT window."""
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


def _greedy_step(args: tuple) -> tuple[int, float]:
    chosen, cand = args
    return cand, _score_in_sample(tuple(chosen) + (cand,), GREEDY_K, GREEDY_LPD, GREEDY_SCALER,
                                  SEEDS[0], GREEDY_TEMP_RATIO, GREEDY_BASIS,
                                  GREEDY_N_INIT, GREEDY_N_ITER)


def greedy_forward(workers: int, m_max: int) -> list[int]:
    """Forward selection is nested by construction, so one run to the largest m also supplies
    every smaller m as a prefix -- greedy_fwd12[:8] IS the greedy 8-feature set."""
    pool = load_pool(ASSET, GREEDY_SCALER)
    d = pool["dim"]
    chosen: list[int] = []
    remaining = list(range(d))
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
              f"in-sample bal_acc={best_s:.4f}")
    return chosen


def evaluate_config(args: tuple) -> dict:
    fset_name, m, idx, scaler, k, lpd, seed = args
    pool = load_pool(ASSET, scaler)
    cols_idx = list(idx)
    lam = lpd * len(cols_idx)
    t0 = time.time()
    mu, obj = fit_jm(pool[f"x_{FIT_YEAR}"][:, cols_idx], k=k, lam=lam, seed=seed,
                     n_init=FULL_N_INIT, n_iter=FULL_N_ITER)
    fit_secs = time.time() - t0
    V = {y: causal_decode_V(pool[f"x_{y}"][:, cols_idx], mu, lam)
         for y in ("2024", "2025", "2026")}
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
    """Collapse the seed dimension: one row per (feature_set, scaler, K, lambda, T, basis)."""
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
                if metric == "balanced_accuracy":
                    row[f"{wname}_{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    row[f"{wname}_{metric}_min"] = float(vals.min())
        # gates use the WORST seed, so a config only passes if it passes robustly
        row["gate_median_run_min"] = float(min(e["windows"][sel]["median_run_bars"] for e in entries))
        row["gate_coverage_min"] = float(min(e["windows"][sel]["min_class_coverage"] for e in entries))
        row["passes_gates"] = bool(row["gate_median_run_min"] >= MIN_MEDIAN_RUN_BARS
                                   and row["gate_coverage_min"] >= MIN_CLASS_COVERAGE)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()

    pool = load_pool(ASSET, "standard")
    cols = pool["cols"]
    rk = rankings_for(ASSET, "standard")
    m_max = max(M_GRID)
    greedy_full: list[int] | None = None
    if not args.skip_greedy:
        print(f"=== greedy forward selection to m={m_max} "
              f"(2024 fit window only, K={GREEDY_K} lpd={GREEDY_LPD} {GREEDY_SCALER})")
        t0 = time.time()
        greedy_full = greedy_forward(args.workers, m_max)
        print(f"  greedy done in {time.time() - t0:.0f}s")

    fsets: dict[str, tuple[int, list[int]]] = {}
    for m in M_GRID:
        fsets[f"f_rank_top{m}"] = (m, [int(i) for i in rk["f_rank"][:m]])
        fsets[f"mrmr_top{m}"] = (m, [int(i) for i in rk["mrmr"][:m]])
        if greedy_full is not None:
            fsets[f"greedy_fwd{m}"] = (m, [int(i) for i in greedy_full[:m]])
    for name, (m, idx) in fsets.items():
        print(f"  {name}: {[cols[i] for i in idx]}")

    jobs = [(name, m, tuple(idx), sc, k, lpd, seed)
            for name, (m, idx) in fsets.items()
            for sc in SCALERS for k in K_GRID for lpd in LAMBDA_PER_DIM_GRID for seed in SEEDS]
    n_cells = len(jobs) // len(SEEDS) * len(TEMPERATURE_RATIO_GRID) * len(LABEL_BASES)
    print(f"\n=== {len(jobs)} JM fits ({len(SEEDS)} seeds x {len(jobs)//len(SEEDS)} configs) "
          f"-> {n_cells} seed-averaged cells ({args.workers} workers)")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(evaluate_config, jobs, chunksize=5), 1):
            results.append(res)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")

    rows = aggregate(results)
    report = {
        "asset": ASSET, "m_grid": list(M_GRID),
        "feature_sets": {n: [cols[i] for i in idx] for n, (m, idx) in fsets.items()},
        "grid": {"m": list(M_GRID), "k": list(K_GRID), "lambda_per_dim": list(LAMBDA_PER_DIM_GRID),
                 "scalers": list(SCALERS), "temperature_ratio": list(TEMPERATURE_RATIO_GRID),
                 "label_bases": list(LABEL_BASES), "seeds": list(SEEDS)},
        "protocol": {"fit_year": FIT_YEAR, "selection_window": SELECTION_WINDOW,
                     "eval_windows": {k: list(v) for k, v in EVAL_WINDOWS.items()},
                     "n_init": FULL_N_INIT, "n_iter": FULL_N_ITER,
                     "gates_use_worst_seed": True,
                     "greedy_selection_window": FIT_YEAR,
                     "fresh_forward_bar_by_bar": True,
                     "trade_ledgers_used_as_input": False,
                     "saved_parent_exit_timestamps_used": False,
                     "future_rows_used_for_entry": False},
        "cells": rows,
    }
    path = OUT_DIR / f"optimize_{ASSET}_m{'_'.join(str(m) for m in M_GRID)}_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  -> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
