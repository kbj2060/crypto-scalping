"""Stage 2 of the JM-only regime3 redesign: the per-asset search.

Grid, run INDEPENDENTLY for BTC and ETH (that independence is the point -- nothing here is ported
from one asset to the other):

    panel          jm6 | jm9 | jm9_perp | wide24_decorr | state12 | wide24
    scaler         robust(5,95) | standard
    K              2 | 3 | 4                (K states -> 3 classes via the state->class matrix)
    lambda_per_dim 0.05 | 0.1 | 0.25 | 0.5 | 1 | 2 | 4     (actual lambda = lambda_per_dim * d)
    temperature    0.25 | 0.5 | 1 | 2 | 4  x lambda
    label basis    frozen | qmatched       (scored, not fitted -- see the lib's label note)

lambda is parameterised PER DIMENSION because the per-bar cost ||x - mu||^2 scales with d; a fixed
absolute lambda would mean something different for a 6-dim panel than a 24-dim one and the panel
comparison would be meaningless. This is exactly the bug in the inherited lambda=4: it was carried
over from a different panel without rescaling.

Only the JM fit is expensive, and it depends on neither temperature nor the label basis, so each
(panel, scaler, K, lambda) is fitted once and then read out across all 10 (temperature x basis)
combinations. 252 fits per asset, parallel across cores.

Split discipline: fit on 2024, SELECT on 2025-09-01..2025-12-31, and 2026-01-01..2026-03-31 is
scored but never read by selection. The report also carries the VAL->OOS Spearman rank
correlation across every gated config, which is the honest read on whether tuning on this axis
transfers at all -- this project has repeatedly found VAL-selected regime configs that invert OOS.

Output: data/ensemble/reports/jm_redesign_20260810/sweep_{asset}_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_BASES, SELECTION_WINDOW, _class_proba,
    _state_class_matrix, causal_decode_V, fit_jm, passes_gates, slice_window, softmax_states,
    window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import CACHE_DIR, OUT_DIR, PANELS, SCALERS  # noqa: E402

K_GRID = (2, 3, 4)
LAMBDA_PER_DIM_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
TEMPERATURE_RATIO_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
SEED = 7529
SWEEP_N_INIT = 3
SWEEP_N_ITER = 10


@lru_cache(maxsize=4)
def load_cache(asset: str, panel: str, scaler: str) -> dict:
    z = np.load(CACHE_DIR / f"{asset}__{panel}__{scaler}.npz", allow_pickle=True)
    out = {"dim": int(z["dim"]), "cols": [str(c) for c in z["cols"]]}
    for y in ("2024", "2025", "2026"):
        out[f"x_{y}"] = z[f"x_{y}"]
        out[f"close_{y}"] = z[f"close_{y}"]
        out[f"ts_{y}"] = z[f"ts_{y}"]
        for b in LABEL_BASES:
            out[f"y_{b}_{y}"] = z[f"y_{b}_{y}"].astype(np.int64)
    out["window_masks"] = {
        name: slice_window(pd.Series(z[f"ts_{yr}"]), start, end)
        for name, (yr, start, end) in EVAL_WINDOWS.items()
    }
    return out


def evaluate_config(args: tuple) -> dict:
    asset, panel, scaler, k, lpd = args
    cache = load_cache(asset, panel, scaler)
    d = cache["dim"]
    lam = float(lpd) * d
    t0 = time.time()
    mu, obj = fit_jm(cache[f"x_{FIT_YEAR}"], k=k, lam=lam, seed=SEED,
                     n_init=SWEEP_N_INIT, n_iter=SWEEP_N_ITER)
    fit_secs = time.time() - t0

    V = {y: causal_decode_V(cache[f"x_{y}"], mu, lam) for y in ("2024", "2025", "2026")}
    # How much of K actually survives the fit: a config that collapses to fewer live states is
    # not really the K it claims to be, and that is worth recording rather than hiding.
    hard_fit = np.argmin(V[FIT_YEAR], axis=1)
    live_states = int(len(np.unique(hard_fit)))

    # Temperature is expressed as a ratio of the config's OWN typical cross-state V spread,
    # measured on the fit window. Anchoring it to lambda (as the inherited build did) does not
    # work: the switch clamp bounds only the carried-forward V, and the current bar's cost is
    # added on top, so the realised spread is far wider than lambda and the same ratio meant
    # wildly different confidence on different panels. Ratio-of-spread makes one temperature
    # setting comparable across every cell in the grid.
    v_spread = float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1)))
    v_spread = max(v_spread, 1e-9)

    cells = []
    for ratio in TEMPERATURE_RATIO_GRID:
        temperature = ratio * v_spread
        sp = {y: softmax_states(V[y], temperature) for y in V}
        conf_fit = np.sort(sp[FIT_YEAR], axis=1)[:, -1]
        for basis in LABEL_BASES:
            state_class = _state_class_matrix(sp[FIT_YEAR], cache[f"y_{basis}_{FIT_YEAR}"])
            windows = {}
            for name, (yr, _s, _e) in EVAL_WINDOWS.items():
                m = cache["window_masks"][name]
                proba = _class_proba(sp[yr][m], state_class)
                windows[name] = window_metrics(
                    np.argmax(proba, axis=1).astype(np.int64),
                    cache[f"y_{basis}_{yr}"][m],
                    cache[f"close_{yr}"][m],
                )
            cells.append({
                "temperature_ratio": ratio,
                "temperature": temperature,
                "label_basis": basis,
                "fit_confidence_mean": float(conf_fit.mean()),
                "fit_confidence_std": float(conf_fit.std()),
                "state_class_matrix": state_class.tolist(),
                "windows": windows,
            })
    return {
        "asset": asset, "panel": panel, "scaler": scaler, "dim": d, "k": k,
        "lambda_per_dim": float(lpd), "lambda": lam, "seed": SEED,
        "fit_objective": obj, "fit_seconds": round(fit_secs, 2),
        "live_states_on_fit": live_states, "v_spread_fit_median": v_spread,
        "mu": mu.tolist(), "cells": cells,
    }


def flatten(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        for c in r["cells"]:
            rows.append({
                "panel": r["panel"], "scaler": r["scaler"], "dim": r["dim"], "k": r["k"],
                "lambda_per_dim": r["lambda_per_dim"], "lambda": r["lambda"],
                "temperature_ratio": c["temperature_ratio"],
                "temperature": c["temperature"], "label_basis": c["label_basis"],
                "live_states_on_fit": r["live_states_on_fit"],
                "fit_confidence_mean": c["fit_confidence_mean"],
                "fit_confidence_std": c["fit_confidence_std"],
                **{f"{w}_{key}": c["windows"][w][key]
                   for w in EVAL_WINDOWS
                   for key in ("balanced_accuracy", "accuracy", "median_run_bars", "flip_rate",
                               "min_class_coverage", "economic_separation_fwd1h",
                               "economic_separation_tstat")},
            })
    return rows


def gated_winner(rows: list[dict], label_basis: str | None = None,
                 criterion: str = "balanced_accuracy") -> dict:
    """The single selection rule, defined once so the sweep summary and the final build cannot
    disagree: among cells passing the persistence and coverage gates on the SELECTION window only,
    take the best `criterion` on that same window. The OOS window is never consulted.

    `label_basis=None` compares the two bases against each other and takes the better VAL cell.
    """
    sel = SELECTION_WINDOW
    pool = [r for r in rows
            if (label_basis is None or r["label_basis"] == label_basis)
            and r[f"{sel}_median_run_bars"] >= 12.0
            and r[f"{sel}_min_class_coverage"] >= 0.05]
    if not pool:
        raise RuntimeError(f"no cell passes the gates (label_basis={label_basis})")
    return max(pool, key=lambda r: r[f"{sel}_{criterion}"])


def summarise(asset: str, rows: list[dict]) -> dict:
    sel = SELECTION_WINDOW
    out: dict = {"n_cells": len(rows), "by_label_basis": {}}
    for basis in LABEL_BASES:
        sub = [r for r in rows if r["label_basis"] == basis]
        gated = [r for r in sub
                 if r[f"{sel}_median_run_bars"] >= 12.0 and r[f"{sel}_min_class_coverage"] >= 0.05]
        entry: dict = {"n_cells": len(sub), "n_passing_gates": len(gated)}
        if gated:
            v = np.array([r[f"{sel}_balanced_accuracy"] for r in gated])
            o = np.array([r["oos_balanced_accuracy"] for r in gated])
            rho, p = spearmanr(v, o)
            entry["val_vs_oos_rank_spearman"] = {"rho": float(rho), "p_value": float(p),
                                                 "n": int(len(gated))}
            entry["winner"] = gated_winner(sub, label_basis=basis)
            entry["top10"] = sorted(gated, key=lambda r: -r[f"{sel}_balanced_accuracy"])[:10]
            # best per panel, so a panel is not judged by one lucky hyperparameter cell
            entry["best_per_panel"] = {
                p_: max([r for r in gated if r["panel"] == p_],
                        key=lambda r: r[f"{sel}_balanced_accuracy"])
                for p_ in PANELS if any(r["panel"] == p_ for r in gated)
            }
        out["by_label_basis"][basis] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in args.assets:
        jobs = [(asset, p, s, k, l)
                for p in PANELS for s in SCALERS for k in K_GRID for l in LAMBDA_PER_DIM_GRID]
        print(f"\n=== {asset}: {len(jobs)} JM fits x "
              f"{len(TEMPERATURE_RATIO_GRID) * len(LABEL_BASES)} readouts "
              f"= {len(jobs) * len(TEMPERATURE_RATIO_GRID) * len(LABEL_BASES)} scored cells "
              f"({args.workers} workers)")
        t0 = time.time()
        results = []
        # chunked by (panel, scaler) so each worker reuses its cached npz via lru_cache
        jobs.sort(key=lambda j: (j[1], j[2]))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(evaluate_config, jobs, chunksize=7), 1):
                results.append(res)
                if i % 21 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} fits  ({time.time() - t0:.0f}s elapsed)")
        rows = flatten(results)
        report = {
            "asset": asset,
            "grid": {"panels": list(PANELS), "scalers": list(SCALERS), "k": list(K_GRID),
                     "lambda_per_dim": list(LAMBDA_PER_DIM_GRID),
                     "temperature_ratio": list(TEMPERATURE_RATIO_GRID),
                     "label_bases": list(LABEL_BASES)},
            "protocol": {
                "fit_year": FIT_YEAR,
                "selection_window": SELECTION_WINDOW,
                "eval_windows": {k: list(v) for k, v in EVAL_WINDOWS.items()},
                "sweep_n_init": SWEEP_N_INIT, "sweep_n_iter": SWEEP_N_ITER, "seed": SEED,
                "fresh_forward_bar_by_bar": True,
                "trade_ledgers_used_as_input": False,
                "saved_parent_exit_timestamps_used": False,
                "future_rows_used_for_entry": False,
            },
            "summary": summarise(asset, rows),
            "cells": rows,
            "fits": [{kk: vv for kk, vv in r.items() if kk != "cells"} for r in results],
        }
        path = OUT_DIR / f"sweep_{asset}_report.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"  -> {path}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
