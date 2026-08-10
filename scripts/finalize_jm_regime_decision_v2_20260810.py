"""Final per-asset selection, exhaustive over the timeliness gate. Supersedes v1.

v1 shortlisted by VAL balanced accuracy before running the expensive oracle lag analysis. That was
wrong in a way that mattered: the binding gate on BTC is detection lag, and accuracy and timeliness
run in OPPOSITE directions here (the accuracy-best greedy_fwd8 cells are the slowest to detect),
so pre-filtering on accuracy discarded exactly the cells the gate was meant to find. Verified
directly: mrmr_top8 / robust / K=3 / lambda_per_dim=0.1 / T=0.25 scores detection lag 10.0 on all
five seeds and wave-Q1 0.661, i.e. it PASSES -- and v1's shortlist never tested it.

The fix is to stop shortlisting. Temperature and label basis are free read-outs off a finished
fit, so the expensive axis is only (feature set x scaler x K x lambda) -- 144 fits for the BTC
low-lambda grid, not thousands. Every cell that clears the cheap gates is therefore scored on the
oracle instruments, with no accuracy-based pre-selection anywhere.

Rule (pre-registered, unchanged):
  G1 consistency  seed std of VAL balanced accuracy <= 0.01 AND VAL->OOS drift >= -0.02
  G2 persistence  worst-seed median run >= anchor's AND whipsaw share <= anchor's
  G3 timeliness   detection lag <= anchor's - 2 bars AND wave-Q1 >= anchor's
                  AND lag-profile peak <= 12 bars
  tie-break       OOS balanced accuracy
  reported only   economic separation (VAL->OOS rank transfer ~0 to negative, cannot select on it)

Anchor = the model actually being replaced (BTC: the live 12-state HMM; ETH: the lambda=4 shadow).
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
    EVAL_WINDOWS, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, SOURCES, _class_proba, _num, _read,
    _state_class_matrix, causal_decode_V, fit_jm, labels_for, quantile_matched_label_config,
    reference_label_quantiles, run_lengths, slice_window, softmax_states, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.ranked_jm_feature_selection_20260810 import load_pool  # noqa: E402
from scripts.scorecard_jm_regime_decision_20260810 import (  # noqa: E402
    CANDIDATES, ORACLE_THETA, SEEDS, WHIPSAW_MAX_BARS, detection_lag, lag_profile,
    predictions_from_csv, to_direction, wave_position, wave_quintiles,
)
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

ANCHOR = {"btc": "INCUMBENT live HMM wide24", "eth": "INCUMBENT JM lam4 wide24"}
REPORTS = {"btc": ["optimize_btc_m8_12_lowlam_report.json"],
           "eth": ["optimize_eth_m6_8_12_full_report.json"]}
SEED_STD_MAX, DRIFT_MIN, LAG_PEAK_MAX, DETLAG_MARGIN = 0.01, -0.02, 12, 2

_CTX: dict = {}


def _init(asset: str, fsets: dict, n_seeds: int) -> None:
    frames = {y: _read(p) for y, p in SOURCES[asset].items()}
    ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
    cfgs = {"frozen": dict(LABEL_CONFIGS[LABEL_MODE]),
            "qmatched": quantile_matched_label_config(frames[FIT_YEAR], ref_q)}
    labels = {b: {y: labels_for(f, cfgs[b]) for y, f in frames.items()} for b in cfgs}
    oracle, pivots, pos, closes, masks = {}, {}, {}, {}, {}
    for y, f in frames.items():
        c = _num(f, "close").ffill().bfill().to_numpy()
        d, pv = zigzag_oracle(c, ORACLE_THETA)
        oracle[y], pivots[y], pos[y], closes[y] = d, pv, wave_position(pv, len(c)), c
    for w, (yr, s, e) in EVAL_WINDOWS.items():
        masks[w] = slice_window(frames[yr]["timestamp"], s, e)
    _CTX.update(asset=asset, fsets=fsets, seeds=SEEDS[:n_seeds], labels=labels, cfgs=cfgs,
                oracle=oracle, pivots=pivots, pos=pos, closes=closes, masks=masks)


def _metrics(pred: np.ndarray, yr: str, wname: str, basis: str) -> dict:
    mask = _CTX["masks"][wname]
    valid = mask & (pred >= 0)
    wm = window_metrics(pred[valid], _CTX["labels"][basis][yr][valid], _CTX["closes"][yr][valid])
    rl = run_lengths(pred[valid])
    wm["whipsaw_share"] = float((rl < WHIPSAW_MAX_BARS).mean()) if len(rl) else 1.0
    d = to_direction(pred)
    d[~mask] = 0
    idxs = np.flatnonzero(mask)
    wm["lag_peak_bars"] = lag_profile(d, _CTX["oracle"][yr], mask)["peak_lag_bars"]
    dl = detection_lag(d, _CTX["oracle"][yr], _CTX["pivots"][yr], int(idxs[0]), int(idxs[-1]))
    wm["detection_lag_median"] = dl["median_bars"] if dl["median_bars"] is not None else 999.0
    wm["wave_Q1"] = wave_quintiles(d, _CTX["oracle"][yr], _CTX["pos"][yr], mask)["Q1"] or 0.0
    return wm


def evaluate_fit(spec: tuple) -> list[dict]:
    """One (feature set, scaler, K, lambda) fit; every requested (temperature, basis) read off it."""
    fset, scaler, k, lpd, readouts = spec
    asset = _CTX["asset"]
    pool = load_pool(asset, scaler)
    idx = [pool["cols"].index(f) for f in _CTX["fsets"][fset]]
    lam = lpd * len(idx)
    per_seed_V = []
    for seed in _CTX["seeds"]:
        mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=k, lam=lam, seed=seed, n_init=3, n_iter=10)
        V = {y: causal_decode_V(pool[f"x_{y}"][:, idx], mu, lam) for y in ("2024", "2025", "2026")}
        spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
        per_seed_V.append((V, spread))

    out = []
    for ratio, basis in readouts:
        acc = {"val": [], "oos": []}
        for V, spread in per_seed_V:
            sp = {y: softmax_states(v, ratio * spread) for y, v in V.items()}
            sc = _state_class_matrix(sp[FIT_YEAR], _CTX["labels"][basis][FIT_YEAR])
            for wname in ("val", "oos"):
                yr = EVAL_WINDOWS[wname][0]
                pred = np.argmax(_class_proba(sp[yr], sc), axis=1).astype(np.int64)
                acc[wname].append(_metrics(pred, yr, wname, basis))
        row = {"feature_set": fset, "scaler": scaler, "k": k, "lambda_per_dim": lpd,
               "temperature_ratio": ratio, "label_basis": basis, "n_seeds": len(per_seed_V)}
        for wname in ("val", "oos"):
            for key in acc[wname][0]:
                vals = [s[key] for s in acc[wname] if isinstance(s[key], (int, float))]
                if vals:
                    row[f"{wname}_{key}"] = float(np.mean(vals))
                    if key == "balanced_accuracy":
                        row[f"{wname}_{key}_worst"] = float(np.min(vals))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    final = {}

    for asset in args.assets:
        cells, fsets = [], {}
        for name in REPORTS[asset]:
            p = OUT_DIR / name
            if not p.exists():
                print(f"[skip] {asset}: {name} not found")
                continue
            rep = json.loads(p.read_text())
            cells.extend(rep["cells"])
            fsets.update(rep["feature_sets"])
        if not cells:
            continue

        keep = [c for c in cells
                if c["passes_gates"] and c["val_balanced_accuracy_std"] <= SEED_STD_MAX
                and (c["oos_balanced_accuracy"] - c["val_balanced_accuracy"]) >= DRIFT_MIN]
        groups: dict[tuple, list[tuple]] = {}
        for c in keep:
            groups.setdefault((c["feature_set"], c["scaler"], c["k"], c["lambda_per_dim"]),
                              []).append((c["temperature_ratio"], c["label_basis"]))
        print(f"\n=== {asset.upper()}  {len(keep)} cells clear G1+G2-cheap, "
              f"spanning {len(groups)} distinct JM fits -- ALL scored, no shortlist")

        _init(asset, fsets, args.seeds)
        anchor_spec = next(s for n, _k, s in CANDIDATES[asset] if n == ANCHOR[asset])
        a_basis = "qmatched" if asset == "btc" else "frozen"
        a_pred = predictions_from_csv(asset, anchor_spec,
                                      {y: _read(p) for y, p in SOURCES[asset].items()})
        a = {w: _metrics(a_pred[EVAL_WINDOWS[w][0]], EVAL_WINDOWS[w][0], w, a_basis)
             for w in ("val", "oos")}["oos"]
        print(f"  ANCHOR {ANCHOR[asset]}: OOS bal={a['balanced_accuracy']:.4f} "
              f"run={a['median_run_bars']:.0f} whip={a['whipsaw_share']:.2f} "
              f"detlag={a['detection_lag_median']:.1f} Q1={a['wave_Q1']:.3f} "
              f"peak={a['lag_peak_bars']:.0f} sep={a['economic_separation_tstat']:+.2f}")
        print(f"  gates -> run>={a['median_run_bars']:.0f}, whip<={a['whipsaw_share']:.2f}, "
              f"detlag<={a['detection_lag_median'] - DETLAG_MARGIN:.1f}, "
              f"Q1>={a['wave_Q1']:.3f}, peak<={LAG_PEAK_MAX}")

        specs = [(f, s, k, l, sorted(set(r))) for (f, s, k, l), r in groups.items()]
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init, initargs=(asset, fsets, args.seeds)) as ex:
            for i, res in enumerate(ex.map(evaluate_fit, specs, chunksize=1), 1):
                rows.extend(res)
                if i % 20 == 0 or i == len(specs):
                    print(f"  {i}/{len(specs)} fits scored ({time.time() - t0:.0f}s)", flush=True)

        for r in rows:
            r["passes_rule"] = bool(
                r["oos_median_run_bars"] >= a["median_run_bars"]
                and r["oos_whipsaw_share"] <= a["whipsaw_share"]
                and r["oos_detection_lag_median"] <= a["detection_lag_median"] - DETLAG_MARGIN
                and r["oos_wave_Q1"] >= a["wave_Q1"]
                and r["oos_lag_peak_bars"] <= LAG_PEAK_MAX)
        winners = [r for r in rows if r["passes_rule"]]
        print(f"\n  {len(winners)}/{len(rows)} cells PASS the full rule")
        for r in sorted(winners, key=lambda r: -r["oos_balanced_accuracy"])[:10]:
            print(f"    {r['feature_set']:<14} {r['scaler']:<8} k={r['k']} "
                  f"lpd={r['lambda_per_dim']:<6g} T={r['temperature_ratio']:<5g} "
                  f"[{r['label_basis'][:4]}] | OOS bal={r['oos_balanced_accuracy']:.4f} "
                  f"run={r['oos_median_run_bars']:>4.0f} whip={r['oos_whipsaw_share']:.2f} "
                  f"detlag={r['oos_detection_lag_median']:>5.1f} Q1={r['oos_wave_Q1']:.3f} "
                  f"sep={r['oos_economic_separation_tstat']:+.2f}")
        w = max(winners, key=lambda r: r["oos_balanced_accuracy"]) if winners else None
        if w:
            print(f"\n  WINNER: {w['feature_set']} {w['scaler']} k={w['k']} "
                  f"lpd={w['lambda_per_dim']} T={w['temperature_ratio']} basis={w['label_basis']}")
            print(f"          features: {fsets[w['feature_set']]}")
        else:
            print("\n  NO CANDIDATE PASSES THE PRE-REGISTERED RULE")
        final[asset] = {"anchor": a, "winner": w, "n_pass": len(winners),
                        "cells": rows, "feature_sets": fsets}

    out = OUT_DIR / "final_decision_v2.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(final)
    out.write_text(json.dumps(prev, indent=2, default=str))
    print(f"\nfinal decision -> {out}")


if __name__ == "__main__":
    main()
