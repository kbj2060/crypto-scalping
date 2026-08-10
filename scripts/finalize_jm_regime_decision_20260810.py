"""Final per-asset selection: apply the pre-registered three-criterion rule to the optimisation.

The optimisation reports 10,800 seed-averaged cells per asset, which is far too many to run the
oracle lag analysis over, so the rule is applied in the order its gates can be evaluated:

  stage 1 (cheap, from the optimisation report)
      G1 consistency  seed std of VAL balanced accuracy <= 0.01 AND VAL->OOS drift >= -0.02
      G2 persistence  worst-seed median run and coverage already gated in the report
  stage 2 (expensive, needs the zigzag oracle -- run only on the shortlist)
      G2b whipsaw share <= the replaced model's
      G3 timeliness   detection lag <= replaced model's - 2 bars
                      AND wave-Q1 agreement >= replaced model's
                      AND lag-profile peak <= 12 bars
  tie-break           OOS balanced accuracy
  reported, not gated economic separation (its VAL->OOS rank transfer is ~0 to negative, so it
                      cannot be selected on -- but a large negative is recorded as a caveat)

The anchor is the model that would actually be REPLACED: BTC's live 12-state HMM, and ETH's
lambda=4 JM shadow. Anchoring to an unpromoted candidate instead would let a config win by beating
something that is not in production.

Shortlisting takes the best cells per feature-set family rather than the global top-N, so a family
that wins on timeliness is not crowded out of stage 2 by a family that only wins on accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from scripts.ranked_jm_feature_selection_20260810 import load_pool, rankings_for  # noqa: E402
from scripts.scorecard_jm_regime_decision_20260810 import (  # noqa: E402
    CANDIDATES, ORACLE_THETA, SEEDS, WHIPSAW_MAX_BARS, detection_lag, lag_profile,
    predictions_from_csv, to_direction, wave_position, wave_quintiles,
)
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

# the model each asset's winner would replace, and therefore the gate anchor
ANCHOR = {"btc": "INCUMBENT live HMM wide24", "eth": "INCUMBENT JM lam4 wide24"}
# BTC is split across two optimisation runs: the original lambda_per_dim 0.25-8 grid, and a later
# extension below 0.25. The extension exists because the first grid's floor was raised from the
# coarse search's 0.05 to 0.25, which silently excluded the region where timeliness lives --
# lambda trades persistence against detection lag directly, and every candidate in the 0.25-8 grid
# failed the lag gate. Both are read so the decision covers the full lambda range.
REPORT = {"btc": ["optimize_btc_m8_12_lowlam_report.json"],
          "eth": ["optimize_eth_m6_8_12_report.json"]}
SHORTLIST_PER_FAMILY = 3
SEED_STD_MAX = 0.01
DRIFT_MIN = -0.02
LAG_PEAK_MAX = 12
DETLAG_MARGIN = 2


def shortlist(rows: list[dict]) -> list[dict]:
    out = []
    for basis in ("frozen", "qmatched"):
        pool = [r for r in rows
                if r["label_basis"] == basis and r["passes_gates"]
                and r["val_balanced_accuracy_std"] <= SEED_STD_MAX
                and (r["oos_balanced_accuracy"] - r["val_balanced_accuracy"]) >= DRIFT_MIN]
        for fam in sorted({r["feature_set"] for r in pool}):
            fam_rows = [r for r in pool if r["feature_set"] == fam]
            # best by accuracy AND best by separation, so neither axis is crowded out
            picks = sorted(fam_rows, key=lambda r: -r["val_balanced_accuracy"])[:SHORTLIST_PER_FAMILY]
            picks += sorted(fam_rows, key=lambda r: -r["oos_economic_separation_tstat"])[:1]
            seen = set()
            for p in picks:
                key = (p["feature_set"], p["m"], p["scaler"], p["k"], p["lambda_per_dim"],
                       p["temperature_ratio"], p["label_basis"])
                if key not in seen:
                    seen.add(key)
                    out.append(p)
    return out


def predict(asset: str, cols_idx: list[int], scaler: str, k: int, lpd: float, temp_ratio: float,
            labels_fit: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    pool = load_pool(asset, scaler)
    lam = lpd * len(cols_idx)
    mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, cols_idx], k=k, lam=lam, seed=seed, n_init=3, n_iter=10)
    V = {y: causal_decode_V(pool[f"x_{y}"][:, cols_idx], mu, lam) for y in ("2024", "2025", "2026")}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    sp = {y: softmax_states(v, temp_ratio * spread) for y, v in V.items()}
    sc = _state_class_matrix(sp[FIT_YEAR], labels_fit)
    return {y: np.argmax(_class_proba(sp[y], sc), axis=1).astype(np.int64) for y in sp}


def full_metrics(preds_by_seed, frames, labels, oracle, pivots, pos) -> dict:
    out = {}
    for wname in ("val", "oos"):
        yr, start, end = EVAL_WINDOWS[wname]
        mask = slice_window(frames[yr]["timestamp"], start, end)
        close = _num(frames[yr], "close").ffill().bfill().to_numpy()
        per = []
        for preds in preds_by_seed:
            pred = preds[yr]
            valid = mask & (pred >= 0)
            wm = window_metrics(pred[valid], labels[yr][valid], close[valid])
            rl = run_lengths(pred[valid])
            wm["whipsaw_share"] = float((rl < WHIPSAW_MAX_BARS).mean()) if len(rl) else 1.0
            d = to_direction(pred)
            d[~mask] = 0
            idxs = np.flatnonzero(mask)
            wm["lag_peak_bars"] = lag_profile(d, oracle[yr], mask)["peak_lag_bars"]
            dl = detection_lag(d, oracle[yr], pivots[yr], int(idxs[0]), int(idxs[-1]))
            wm["detection_lag_median"] = dl["median_bars"] if dl["median_bars"] is not None else 999.0
            wm["wave_Q1"] = wave_quintiles(d, oracle[yr], pos[yr], mask)["Q1"] or 0.0
            per.append(wm)
        agg = {}
        for key in per[0]:
            vals = [p[key] for p in per if isinstance(p[key], (int, float))]
            if vals:
                agg[key] = float(np.mean(vals))
        out[wname] = agg
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc"])
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds]
    final = {}

    for asset in args.assets:
        cells, fsets = [], {}
        for name in REPORT[asset]:
            path = OUT_DIR / name
            if not path.exists():
                print(f"[skip] {asset}: {name} not found")
                continue
            rep = json.loads(path.read_text())
            cells.extend(rep["cells"])
            fsets.update(rep["feature_sets"])
        if not cells:
            continue
        cand = shortlist(cells)
        print(f"\n=== {asset.upper()}  shortlist {len(cand)} cells "
              f"(from {len(cells)} after G1 + persistence/coverage)")

        frames = {y: _read(p) for y, p in SOURCES[asset].items()}
        ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
        cfgs = {"frozen": dict(LABEL_CONFIGS[LABEL_MODE]),
                "qmatched": quantile_matched_label_config(frames[FIT_YEAR], ref_q)}
        oracle, pivots, pos = {}, {}, {}
        for y, f in frames.items():
            close = _num(f, "close").ffill().bfill().to_numpy()
            dirn, pv = zigzag_oracle(close, ORACLE_THETA)
            oracle[y], pivots[y] = dirn, pv
            pos[y] = wave_position(pv, len(close))

        # anchor, measured identically
        anchor_spec = next(s for n, k_, s in CANDIDATES[asset] if n == ANCHOR[asset])
        a_labels = {y: labels_for(f, cfgs["qmatched" if asset == "btc" else "frozen"])
                    for y, f in frames.items()}
        a_pred = predictions_from_csv(asset, anchor_spec, frames)
        anchor = full_metrics([a_pred], frames, a_labels, oracle, pivots, pos)
        a = anchor["oos"]
        print(f"  ANCHOR {ANCHOR[asset]}: OOS bal={a['balanced_accuracy']:.4f} "
              f"run={a['median_run_bars']:.0f} whip={a['whipsaw_share']:.2f} "
              f"detlag={a['detection_lag_median']:.1f} Q1={a['wave_Q1']:.3f} "
              f"lagpeak={a['lag_peak_bars']:.0f} sep={a['economic_separation_tstat']:+.2f}")
        print(f"  gates -> run>={a['median_run_bars']:.0f}, whip<={a['whipsaw_share']:.2f}, "
              f"detlag<={a['detection_lag_median'] - DETLAG_MARGIN:.1f}, "
              f"Q1>={a['wave_Q1']:.3f}, lagpeak<={LAG_PEAK_MAX}")

        scored = []
        for c in cand:
            idx = [load_pool(asset, c["scaler"])["cols"].index(f)
                   for f in fsets[c["feature_set"]]]
            labels = {y: labels_for(f, cfgs[c["label_basis"]]) for y, f in frames.items()}
            preds = [predict(asset, idx, c["scaler"], c["k"], c["lambda_per_dim"],
                             c["temperature_ratio"], labels[FIT_YEAR], s) for s in seeds]
            mt = full_metrics(preds, frames, labels, oracle, pivots, pos)
            o = mt["oos"]
            passes = (o["median_run_bars"] >= a["median_run_bars"]
                      and o["whipsaw_share"] <= a["whipsaw_share"]
                      and o["detection_lag_median"] <= a["detection_lag_median"] - DETLAG_MARGIN
                      and o["wave_Q1"] >= a["wave_Q1"]
                      and o["lag_peak_bars"] <= LAG_PEAK_MAX)
            scored.append({**c, "oos_full": o, "val_full": mt["val"], "passes_rule": bool(passes)})
            print(f"  {'PASS' if passes else 'fail'} {c['feature_set']:<14} m={c['m']:<2} "
                  f"{c['scaler']:<8} k={c['k']} lpd={c['lambda_per_dim']:<5g} "
                  f"T={c['temperature_ratio']:<4g} [{c['label_basis'][:4]}] | "
                  f"OOS bal={o['balanced_accuracy']:.4f} run={o['median_run_bars']:>4.0f} "
                  f"whip={o['whipsaw_share']:.2f} detlag={o['detection_lag_median']:>5.1f} "
                  f"Q1={o['wave_Q1']:.3f} peak={o['lag_peak_bars']:>3.0f} "
                  f"sep={o['economic_separation_tstat']:+.2f}")

        winners = [s for s in scored if s["passes_rule"]]
        if winners:
            w = max(winners, key=lambda s: s["oos_full"]["balanced_accuracy"])
            print(f"\n  WINNER: {w['feature_set']} m={w['m']} {w['scaler']} k={w['k']} "
                  f"lpd={w['lambda_per_dim']} T={w['temperature_ratio']} "
                  f"basis={w['label_basis']}")
            print(f"          features: {fsets[w['feature_set']]}")
        else:
            w = None
            print("\n  NO CANDIDATE PASSES THE PRE-REGISTERED RULE")
        final[asset] = {"anchor": anchor, "scored": scored, "winner": w,
                        "feature_sets": fsets}

    out = OUT_DIR / "final_decision.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(final)
    out.write_text(json.dumps(prev, indent=2, default=str))
    print(f"\nfinal decision -> {out}")


if __name__ == "__main__":
    main()
