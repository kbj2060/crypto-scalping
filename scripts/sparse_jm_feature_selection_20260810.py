"""JM-native INPUT FEATURE redesign: sparse jump model feature selection over the full column pool.

The panel sweep (sweep_jm_regime_redesign_20260810.py) compared six hand-authored panels. That is
not a feature redesign -- it never searched the feature space, it only ranked six guesses, three of
which were pure-price literature panels. This script does the actual search, with the method built
for exactly this problem:

  Sparse Jump Model -- Nystrup, Kolm & Lindstrom, "Feature selection in jump models"
  (Expert Systems with Applications 184, 2021), which lifts Witten & Tibshirani's (2010) sparse
  k-means into the jump-model coordinate descent.

Instead of choosing a panel, it learns a non-negative weight vector w over ALL candidate features
under ||w||_2 = 1 and ||w||_1 <= kappa, alternating:

  (a) given w, fit the jump model on the w-weighted metric. Weighted squared distance is plain
      squared distance on columns scaled by sqrt(w), so this reuses the same DP unchanged;
  (b) given the state path, set a_j = between-cluster sum of squares of feature j (the jump
      penalty is feature-independent and drops out of this update), then
      w = S(a, delta)_+ / ||S(a, delta)_+||_2 with delta binary-searched to meet the L1 budget.

kappa is the single sparsity knob: for m equally-weighted features ||w||_1 = sqrt(m), so kappa
sweeps roughly 2 to 130 effective features. Features whose between-cluster separation does not pay
for their L1 cost get exactly zero weight and drop out. This answers the question the panel sweep
could not: which inputs does a jump model actually want, per asset, and how much does each matter.

Candidate pool: every numeric column in the asset's own feature file except
  * raw non-stationary levels (open/high/low/close/volume/quote_volume/trades/OI value/BTC levels)
    -- a 2024-fit scaler on a price level does not survive into 2026;
  * calendar and session encodings (hour/minute sin-cos, session flags, is_hour_open) -- these
    would let the model split "regimes" by clock time, which is not a regime;
plus the EWM return / downside-deviation / Sortino ladder from the literature panels, so the
search can pick those up if they earn their place. 121 raw columns + 9 ladder = 130 candidates,
and the pool is identical across assets, so any per-asset difference in the selected set is a real
difference in the assets and not an artifact of what data each one has.

Protocol is unchanged from the panel sweep: fit and select features on 2024 only, choose
kappa/lambda/K/temperature on 2025-09..12, and 2026-01..03 is scored but never consulted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_BASES, LABEL_CONFIGS, LABEL_MODE, SELECTION_WINDOW,
    SOURCES, _class_proba, _cost_matrix, _num, _read, _state_class_matrix, apply_scale,
    causal_decode_V, fit_scale, labels_for, offline_dp, quantile_matched_label_config,
    reference_label_quantiles, slice_window, softmax_states, window_metrics,
)
from scripts.jm_regime_redesign_lib_20260810 import _ewm_panel  # noqa: E402
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402

CACHE_DIR = OUT_DIR / "sparse_cache"
SEED = 7529

EXCLUDE_COLS = {
    # non-stationary raw levels: a scaler fit on 2024 does not transfer to 2026
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value",
    "close_btc", "volume_btc", "quote_volume_btc",
    # calendar / session: would let the model call a time of day a "regime"
    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "is_hour_open",
}
EXCLUDE_PREFIXES = ("session_",)

# kappa = ||w||_1 under ||w||_2 = 1, so m equally-weighted features give kappa = sqrt(m): this
# grid spans ~4 features up to sqrt(130) = 11.4, i.e. no sparsity at all (the dense control).
KAPPA_GRID = (2.0, 3.0, 4.0, 5.5, 7.5, 9.0, 11.4)
# The ||w||_2 = 1 constraint normalises the weighted per-bar cost to roughly ||w||_1 ~ kappa, so
# lambda lives on a completely different scale than in the unweighted panel sweep (where it had to
# absorb the raw dimension d). Measured on BTC: the persistence transition is sharp and late --
# lambda 8 -> median run 5 bars, 16 -> 7, 32 -> 111 -- so the grid straddles that band rather than
# the sub-1 values the unnormalised intuition suggests.
LAMBDA_GRID = (4.0, 8.0, 16.0, 22.0, 32.0, 48.0)
K_GRID = (3, 4)
SCALERS = ("standard", "robust")
TEMPERATURE_RATIO_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

OUTER_ITERS = 15
W_TOL = 1e-4
# The first outer pass runs on all ~130 candidates purely to bootstrap a state path for the first
# weight update, so it is deliberately cheap; once the L1 budget bites, the active set is small
# and the remaining passes are fast, which is where the fit quality actually comes from.
FIRST_N_INIT, FIRST_N_ITER = 1, 5
WARM_N_ITER = 5


# --------------------------------------------------------------------------------------------
# candidate pool
# --------------------------------------------------------------------------------------------
def candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in frame.columns
            if c not in EXCLUDE_COLS and not c.startswith(EXCLUDE_PREFIXES)]
    out = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name, values in _ewm_panel(frame).items():
        out[name] = values
    return out


# --------------------------------------------------------------------------------------------
# sparse jump model
# --------------------------------------------------------------------------------------------
def update_weights(a: np.ndarray, kappa: float) -> np.ndarray:
    """w = S(a, delta)_+ / ||S(a, delta)_+||_2, delta >= 0 smallest meeting ||w||_1 <= kappa."""
    a = np.maximum(a, 0.0)
    if not np.any(a > 0):
        w = np.ones(len(a))
        return w / np.linalg.norm(w)

    def w_at(delta: float) -> np.ndarray:
        s = np.maximum(a - delta, 0.0)
        n = np.linalg.norm(s)
        return s / n if n > 0 else np.zeros_like(s)

    w = w_at(0.0)
    if w.sum() <= kappa:
        return w
    lo, hi = 0.0, float(a.max())
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        wm = w_at(mid)
        if wm.sum() > kappa:
            lo = mid
        else:
            hi = mid
    w = w_at(hi)
    if not np.any(w > 0):
        # kappa below the floor of 1 (a single feature): keep the single best-separated feature
        w = np.zeros_like(a)
        w[int(np.argmax(a))] = 1.0
    return w


def between_cluster_ss(x: np.ndarray, states: np.ndarray, k: int) -> np.ndarray:
    """a_j = sum_s n_s (mu_sj - xbar_j)^2 -- the share of feature j's total spread that the state
    partition explains. Features the partition does not separate contribute nothing and are the
    first to be thresholded to zero weight."""
    xbar = x.mean(axis=0)
    a = np.zeros(x.shape[1])
    for s in range(k):
        m = states == s
        n_s = int(m.sum())
        if n_s > 0:
            a += n_s * (x[m].mean(axis=0) - xbar) ** 2
    return a


def _kmeanspp(x: np.ndarray, k: int, rng) -> np.ndarray:
    mu = [x[rng.integers(len(x))]]
    while len(mu) < k:
        d2 = np.min(_cost_matrix(x, np.asarray(mu)), axis=1)
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.full(len(x), 1.0 / len(x))
        mu.append(x[rng.choice(len(x), p=p)])
    return np.asarray(mu, dtype=np.float64)


def _jm_descent(x: np.ndarray, mu: np.ndarray, lam: float, n_iter: int):
    prev = None
    states = None
    for _ in range(n_iter):
        states = offline_dp(_cost_matrix(x, mu), lam)
        for s in range(len(mu)):
            m = states == s
            if m.sum() > 10:
                mu[s] = x[m].mean(axis=0)
        if prev is not None and (states == prev).all():
            break
        prev = states
    return mu, states


def sparse_jm_fit(x: np.ndarray, k: int, lam: float, kappa: float, seed: int):
    """Alternate (weighted JM fit) <-> (weight update) until w stops moving.

    Returns (w over ALL columns, mu in ORIGINAL feature space, final states, n_outer, converged).
    Keeping mu in the original space means the artifact stores weights and centroids separately,
    so the selected feature set is directly readable rather than baked into a rotation.
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    # Witten-Tibshirani initialisation: start from UNIFORM weights and let the first weight update
    # apply the L1 budget using real between-cluster separations. Soft-thresholding a constant
    # vector cannot rank anything -- it drives every weight to zero at once.
    w = np.full(d, 1.0 / np.sqrt(d))
    mu_w = None
    states = None
    converged = False
    n_outer = 0
    for it in range(OUTER_ITERS):
        n_outer = it + 1
        active = np.flatnonzero(w > 0)
        sw = np.sqrt(w[active])
        xw = x[:, active] * sw
        if mu_w is None:
            best_obj, best = np.inf, None
            for _ in range(FIRST_N_INIT):
                mu0 = _kmeanspp(xw, k, rng)
                mu1, st = _jm_descent(xw, mu0, lam, FIRST_N_ITER)
                obj = float(((xw - mu1[st]) ** 2).sum() + lam * (np.diff(st) != 0).sum())
                if obj < best_obj:
                    best_obj, best = obj, (mu1.copy(), st.copy())
            mu_a, states = best
        else:
            # warm start: carry the previous centroids into the re-weighted metric
            mu_a, states = _jm_descent(xw, mu_w[:, active] * sw, lam, WARM_N_ITER)
        # back to original space
        mu = np.zeros((k, d))
        for s in range(k):
            m = states == s
            mu[s] = x[m].mean(axis=0) if m.any() else x.mean(axis=0)
        mu_w = mu
        w_new = update_weights(between_cluster_ss(x, states, k), kappa)
        denom = np.abs(w).sum()
        if denom > 0 and np.abs(w_new - w).sum() / denom < W_TOL:
            w = w_new
            converged = True
            break
        w = w_new
    return w, mu_w, states, n_outer, converged


def weighted_decode_V(x: np.ndarray, mu: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    active = np.flatnonzero(w > 0)
    sw = np.sqrt(w[active])
    return causal_decode_V(x[:, active] * sw, mu[:, active] * sw, lam)


# --------------------------------------------------------------------------------------------
# cache + evaluation
# --------------------------------------------------------------------------------------------
def build_cache(asset: str, scaler_kind: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{asset}__{scaler_kind}.npz"
    if out_path.exists():
        return
    frames = {y: _read(p) for y, p in SOURCES[asset].items()}
    fit_frame = frames[FIT_YEAR]
    ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
    cfgs = {"frozen": dict(LABEL_CONFIGS[LABEL_MODE]),
            "qmatched": quantile_matched_label_config(fit_frame, ref_q)}

    cand = {y: candidate_frame(f) for y, f in frames.items()}
    fit_cand = cand[FIT_YEAR]
    keep = [c for c in fit_cand.columns
            if fit_cand[c].notna().any() and float(fit_cand[c].std(skipna=True) or 0.0) > 0]
    cand = {y: v[keep] for y, v in cand.items()}
    x_fit, scaler, medians, clip_bounds = fit_scale(cand[FIT_YEAR], scaler_kind)

    payload = {"cols": np.asarray(keep, dtype=object), "dim": np.int64(len(keep))}
    for y, f in frames.items():
        payload[f"x_{y}"] = (x_fit if y == FIT_YEAR
                             else apply_scale(cand[y], scaler, medians, clip_bounds))
        payload[f"close_{y}"] = _num(f, "close").ffill().bfill().to_numpy(dtype=np.float64)
        payload[f"ts_{y}"] = f["timestamp"].to_numpy(dtype="datetime64[ns]")
        for b in LABEL_BASES:
            payload[f"y_{b}_{y}"] = labels_for(f, cfgs[b]).astype(np.int8)
    np.savez_compressed(out_path, **payload)
    joblib.dump({"scaler": scaler, "medians": medians, "cols": keep,
                 "clip_bounds": clip_bounds, "label_configs": cfgs},
                CACHE_DIR / f"{asset}__{scaler_kind}__scaler.joblib")
    print(f"  cached {asset}/{scaler_kind}: {len(keep)} candidate features")


@lru_cache(maxsize=2)
def load_cache(asset: str, scaler_kind: str) -> dict:
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


def evaluate_cell(args: tuple) -> dict:
    asset, scaler_kind, k, lam, kappa = args
    cache = load_cache(asset, scaler_kind)
    cols = cache["cols"]
    t0 = time.time()
    w, mu, states_fit, n_outer, converged = sparse_jm_fit(
        cache[f"x_{FIT_YEAR}"], k=k, lam=lam, kappa=kappa, seed=SEED)
    fit_secs = time.time() - t0

    sel = np.flatnonzero(w > 0)
    order = sel[np.argsort(-w[sel])]
    V = {y: weighted_decode_V(cache[f"x_{y}"], mu, w, lam) for y in ("2024", "2025", "2026")}
    v_spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)

    cells = []
    for ratio in TEMPERATURE_RATIO_GRID:
        sp = {y: softmax_states(Vv, ratio * v_spread) for y, Vv in V.items()}
        for basis in LABEL_BASES:
            state_class = _state_class_matrix(sp[FIT_YEAR], cache[f"y_{basis}_{FIT_YEAR}"])
            windows = {}
            for name, (yr, _s, _e) in EVAL_WINDOWS.items():
                m = cache["window_masks"][name]
                proba = _class_proba(sp[yr][m], state_class)
                windows[name] = window_metrics(np.argmax(proba, axis=1).astype(np.int64),
                                               cache[f"y_{basis}_{yr}"][m],
                                               cache[f"close_{yr}"][m])
            cells.append({"temperature_ratio": ratio, "label_basis": basis, "windows": windows})
    return {
        "asset": asset, "scaler": scaler_kind, "k": k, "lambda": lam, "kappa": kappa,
        "n_selected": int(len(sel)), "n_candidates": len(cols),
        "n_outer_iters": n_outer, "w_converged": converged,
        "selected_features": [{"feature": cols[i], "weight": float(w[i])} for i in order],
        "v_spread_fit_median": v_spread,
        "live_states_on_fit": int(len(np.unique(states_fit))),
        "fit_seconds": round(fit_secs, 1), "cells": cells,
    }


def flatten(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        for c in r["cells"]:
            rows.append({
                "scaler": r["scaler"], "k": r["k"], "lambda": r["lambda"], "kappa": r["kappa"],
                "n_selected": r["n_selected"], "w_converged": r["w_converged"],
                "live_states_on_fit": r["live_states_on_fit"],
                "temperature_ratio": c["temperature_ratio"], "label_basis": c["label_basis"],
                "top_features": [f["feature"] for f in r["selected_features"][:8]],
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in args.assets:
        print(f"\n=== {asset}: building candidate cache")
        for sk in SCALERS:
            build_cache(asset, sk)
        jobs = [(asset, sk, k, lam, kap)
                for sk in SCALERS for k in K_GRID for lam in LAMBDA_GRID for kap in KAPPA_GRID]
        jobs.sort(key=lambda j: j[1])
        print(f"=== {asset}: {len(jobs)} sparse-JM fits ({args.workers} workers)")
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(evaluate_cell, jobs, chunksize=4), 1):
                results.append(res)
                if i % 20 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
        report = {
            "asset": asset,
            "method": "sparse jump model (Nystrup/Kolm/Lindstrom 2021; Witten-Tibshirani soft-threshold)",
            "grid": {"kappa": list(KAPPA_GRID), "lambda": list(LAMBDA_GRID), "k": list(K_GRID),
                     "scalers": list(SCALERS), "temperature_ratio": list(TEMPERATURE_RATIO_GRID),
                     "label_bases": list(LABEL_BASES)},
            "protocol": {"fit_year": FIT_YEAR, "selection_window": SELECTION_WINDOW,
                         "eval_windows": {k: list(v) for k, v in EVAL_WINDOWS.items()},
                         "outer_iters": OUTER_ITERS, "seed": SEED,
                         "fresh_forward_bar_by_bar": True,
                         "trade_ledgers_used_as_input": False,
                         "saved_parent_exit_timestamps_used": False,
                         "future_rows_used_for_entry": False},
            "cells": flatten(results),
            "fits": [{kk: vv for kk, vv in r.items() if kk != "cells"} for r in results],
        }
        path = OUT_DIR / f"sparse_{asset}_report.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"  -> {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
