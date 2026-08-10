"""Supervised counterpart to the sparse-JM feature search: ranked nested-panel selection.

The sparse jump model selects features by BETWEEN-CLUSTER sum of squares -- how well a feature
separates the states the model itself discovered. That is the method-native criterion, but it is
unsupervised, and it is not the criterion the artifact is graded on: downstream, the state->class
matrix is fit against the ADX/slope/BB rule label, so what matters is whether a feature separates
THOSE classes. A feature can carve the JM's clusters beautifully and be irrelevant to the label,
and vice versa. Running only the sparse search would leave that half of the question unanswered.

So this script ranks the same 130-candidate pool by supervised class separation on the 2024 fit
window and evaluates nested top-m panels through the identical JM pipeline. Two rankings:

  f_rank  ANOVA F between the three label classes -- deliberately the same statistic the sparse
          model uses (between-group over within-group spread), just computed against the label's
          classes instead of the model's clusters, so the two searches are directly comparable.
  mrmr    the same F, greedily discounted by mean |correlation| with what is already selected.
          For a Euclidean model this matters more than usual: a top-m list that is twelve
          restatements of "trend" silently triples the weight of trend in the distance metric,
          which is the original complaint against wide24.

Ranking uses the qmatched label basis for both assets. Ranking against the frozen label on BTC
would be ranking against a target that is 82% one class; both bases are still scored afterwards.
Ranking is computed on the 2024 fit window only, never on VAL or OOS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    EVAL_WINDOWS, FIT_YEAR, LABEL_BASES, SELECTION_WINDOW, _class_proba, _state_class_matrix,
    causal_decode_V, fit_jm, slice_window, softmax_states, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.sparse_jm_feature_selection_20260810 import (  # noqa: E402
    CACHE_DIR, SEED, TEMPERATURE_RATIO_GRID, build_cache,
)

RANKINGS = ("f_rank", "mrmr")
TOP_M_GRID = (4, 6, 8, 12, 16, 24, 32, 48, 64, 130)
LAMBDA_PER_DIM_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
K_GRID = (3, 4)
SCALERS = ("standard", "robust")
RANK_BASIS = "qmatched"
N_INIT, N_ITER = 3, 10


@lru_cache(maxsize=2)
def load_pool(asset: str, scaler_kind: str) -> dict:
    z = np.load(CACHE_DIR / f"{asset}__{scaler_kind}.npz", allow_pickle=True)
    out = {"cols": [str(c) for c in z["cols"]], "dim": int(z["dim"])}
    for y in ("2024", "2025", "2026"):
        out[f"x_{y}"] = z[f"x_{y}"]
        out[f"close_{y}"] = z[f"close_{y}"]
        for b in LABEL_BASES:
            out[f"y_{b}_{y}"] = z[f"y_{b}_{y}"].astype(np.int64)
    out["window_masks"] = {
        name: slice_window(pd.Series(z[f"ts_{yr}"]), start, end)
        for name, (yr, start, end) in EVAL_WINDOWS.items()
    }
    return out


@lru_cache(maxsize=4)
def rankings_for(asset: str, scaler_kind: str) -> dict:
    pool = load_pool(asset, scaler_kind)
    x = pool[f"x_{FIT_YEAR}"]
    y = pool[f"y_{RANK_BASIS}_{FIT_YEAR}"]
    f, _ = f_classif(x, y)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f_order = list(np.argsort(-f))

    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    picked = [int(np.argmax(f))]
    remaining = set(range(x.shape[1])) - set(picked)
    while remaining:
        best_j, best_s = None, -np.inf
        for j in remaining:
            redundancy = float(np.mean([corr[j, p] for p in picked]))
            s = f[j] / (redundancy + 1e-6)
            if s > best_s:
                best_j, best_s = j, s
        picked.append(int(best_j))
        remaining.discard(best_j)
    return {"f_rank": [int(i) for i in f_order], "mrmr": picked,
            "f_values": {pool["cols"][i]: float(f[i]) for i in f_order[:40]}}


def evaluate_cell(args: tuple) -> dict:
    asset, scaler_kind, ranking, m, k, lpd = args
    pool = load_pool(asset, scaler_kind)
    idx = np.asarray(rankings_for(asset, scaler_kind)[ranking][:m])
    cols = [pool["cols"][i] for i in idx]
    d = len(idx)
    lam = lpd * d
    t0 = time.time()
    mu, obj = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=k, lam=lam, seed=SEED,
                     n_init=N_INIT, n_iter=N_ITER)
    fit_secs = time.time() - t0

    V = {y: causal_decode_V(pool[f"x_{y}"][:, idx], mu, lam) for y in ("2024", "2025", "2026")}
    v_spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    cells = []
    for ratio in TEMPERATURE_RATIO_GRID:
        sp = {y: softmax_states(Vv, ratio * v_spread) for y, Vv in V.items()}
        for basis in LABEL_BASES:
            state_class = _state_class_matrix(sp[FIT_YEAR], pool[f"y_{basis}_{FIT_YEAR}"])
            windows = {}
            for name, (yr, _s, _e) in EVAL_WINDOWS.items():
                msk = pool["window_masks"][name]
                proba = _class_proba(sp[yr][msk], state_class)
                windows[name] = window_metrics(np.argmax(proba, axis=1).astype(np.int64),
                                               pool[f"y_{basis}_{yr}"][msk],
                                               pool[f"close_{yr}"][msk])
            cells.append({"temperature_ratio": ratio, "label_basis": basis, "windows": windows})
    return {"asset": asset, "scaler": scaler_kind, "ranking": ranking, "top_m": m, "k": k,
            "lambda_per_dim": lpd, "lambda": lam, "dim": d, "cols": cols,
            "live_states_on_fit": int(len(np.unique(np.argmin(V[FIT_YEAR], axis=1)))),
            "v_spread_fit_median": v_spread, "fit_objective": obj,
            "fit_seconds": round(fit_secs, 1), "cells": cells}


def flatten(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        for c in r["cells"]:
            rows.append({
                "ranking": r["ranking"], "top_m": r["top_m"], "scaler": r["scaler"], "k": r["k"],
                "lambda_per_dim": r["lambda_per_dim"], "lambda": r["lambda"],
                "temperature_ratio": c["temperature_ratio"], "label_basis": c["label_basis"],
                "live_states_on_fit": r["live_states_on_fit"], "cols_head": r["cols"][:8],
                **{f"{wn}_{key}": c["windows"][wn][key]
                   for wn in EVAL_WINDOWS
                   for key in ("balanced_accuracy", "accuracy", "median_run_bars", "flip_rate",
                               "min_class_coverage", "economic_separation_fwd1h",
                               "economic_separation_tstat")},
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    for asset in args.assets:
        for sk in SCALERS:
            build_cache(asset, sk)
        rk = rankings_for(asset, "standard")
        print(f"\n=== {asset}: top-15 by ANOVA F vs the {RANK_BASIS} label (2024 fit window)")
        for i, (name, val) in enumerate(list(rk["f_values"].items())[:15], 1):
            print(f"   {i:>2}. {name:<34} F={val:,.0f}")
        pool = load_pool(asset, "standard")
        print(f"    mrmr first 15: {[pool['cols'][i] for i in rk['mrmr'][:15]]}")

        jobs = [(asset, sk, rank, m, k, lpd)
                for sk in SCALERS for rank in RANKINGS for m in TOP_M_GRID
                for k in K_GRID for lpd in LAMBDA_PER_DIM_GRID]
        jobs.sort(key=lambda j: j[1])
        print(f"=== {asset}: {len(jobs)} JM fits ({args.workers} workers)")
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(evaluate_cell, jobs, chunksize=8), 1):
                results.append(res)
                if i % 40 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
        report = {
            "asset": asset,
            "method": "supervised ranked nested-panel selection (ANOVA F / mRMR) + jump model",
            "rank_label_basis": RANK_BASIS,
            "grid": {"rankings": list(RANKINGS), "top_m": list(TOP_M_GRID),
                     "lambda_per_dim": list(LAMBDA_PER_DIM_GRID), "k": list(K_GRID),
                     "scalers": list(SCALERS),
                     "temperature_ratio": list(TEMPERATURE_RATIO_GRID),
                     "label_bases": list(LABEL_BASES)},
            "protocol": {"fit_year": FIT_YEAR, "selection_window": SELECTION_WINDOW,
                         "eval_windows": {k: list(v) for k, v in EVAL_WINDOWS.items()},
                         "n_init": N_INIT, "n_iter": N_ITER, "seed": SEED,
                         "fresh_forward_bar_by_bar": True,
                         "trade_ledgers_used_as_input": False,
                         "saved_parent_exit_timestamps_used": False,
                         "future_rows_used_for_entry": False},
            "rankings": {r: [load_pool(asset, "standard")["cols"][i]
                             for i in rankings_for(asset, "standard")[r]] for r in RANKINGS},
            "f_values_top40": rk["f_values"],
            "cells": flatten(results),
        }
        path = OUT_DIR / f"ranked_{asset}_report.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"  -> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
