"""Build the shippable artifacts for the rule-selected JM regime3 detectors, per asset.

Final configurations, selected by the pre-registered three-criterion rule (consistency /
persistence / timeliness, each gated against the model being replaced) with the registered
tie-break (OOS balanced accuracy) applied inside the chosen feature family:

  BTC  mrmr_top8      robust   K=3  lambda_per_dim=0.1  T_ratio=1.0   label basis frozen
  ETH  f_rank_top12   standard K=3  lambda_per_dim=4.0  T_ratio=0.25  label basis frozen

Both use the frozen ADX/slope/BB label thresholds, i.e. the label contract already live, so the
state->class semantics downstream are unchanged.

ETH's temperature is 0.25, NOT the 4.0 the registered tie-break selected, and that is a deliberate
documented override. Temperature does not change which class wins, only how peaked the emitted
probabilities are -- and every selection metric used here reads the argmax, so the rule was blind
to it. Measured on 2026, T=4.0 emits confidence 0.336 (a 3-class uniform is 0.333), entropy 1.000
and margin 0.003: `confidence`, `entropy` and `margin` become constants, so three of the six
contract columns carry no information into the parent and sidecar. T=0.25 also passes the full
rule, and against the model it replaces it is BETTER on confidence spread (0.656 vs the lambda=4
shadow's 0.450) and on whipsaw (0.13 vs 0.16); the cost is 2.8pp of balanced accuracy, which is
agreement with a rule label and was never one of the three stated criteria. BTC needs no such
override: T=1.0 is already the sharpest of its rule-passing temperatures.

Emits, per asset:
  data/ensemble/supervised/{asset}_regime3_current_jm_redesign_{TAG}_2024.joblib
      self-contained: panel column list, per-column winsorisation bounds, fill medians, the fitted
      scaler, JM centroids, lambda, temperature, and the state->class matrix -- everything needed
      to rebuild the transform from a raw frame at live time, with no reference to any cache.
  data/ensemble/supervised/{asset}_regime3_current_hmm_{TAG}_{year}_maskedname.csv

The CSV column prefix stays `regime3_current_sensitive_wide24_` even though neither winner uses
the wide24 panel. That is deliberate and load-bearing: every downstream consumer (parent trainer,
risk sidecar, live runtime) keys on those literal names, so keeping them makes the swap drop-in.
The `wide24` token is a historical name, not a claim about the inputs -- the joblib's `panel_cols`
field is the authority on what actually feeds the model.

Two verifications run before anything is written, and both are reported rather than absorbed:
  * the scaler refit on the selected columns must reproduce the cached pool values (all the
    transforms are per-column, so restricting the column set must be a no-op numerically);
  * the higher-budget refit must reproduce the decision run's OOS balanced accuracy, otherwise the
    coordinate descent landed elsewhere and the selection does not describe this artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM, SOURCES,
    _class_proba, _num, _read, _state_class_matrix, apply_scale, causal_decode_V, fit_jm,
    fit_scale, labels_for, order_states_by_return, quantile_matched_label_config,
    reference_label_quantiles, run_lengths, slice_window, softmax_states, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.ranked_jm_feature_selection_20260810 import load_pool  # noqa: E402
from scripts.scorecard_jm_regime_decision_20260810 import (  # noqa: E402
    ORACLE_THETA, WHIPSAW_MAX_BARS, detection_lag, lag_profile, to_direction, wave_position,
    wave_quintiles,
)
from scripts.sparse_jm_feature_selection_20260810 import candidate_frame  # noqa: E402
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

SUPERVISED_DIR = ROOT / "data/ensemble/supervised"
BUILD_N_INIT, BUILD_N_ITER, SEED = 5, 15, 7529
REFIT_TOLERANCE = 0.01

FINAL = {
    "btc": {"feature_set": "mrmr_top8", "scaler": "robust", "k": 3,
            "lambda_per_dim": 0.1, "temperature_ratio": 1.0, "label_basis": "frozen"},
    "eth": {"feature_set": "f_rank_top12", "scaler": "standard", "k": 3,
            "lambda_per_dim": 4.0, "temperature_ratio": 0.25, "label_basis": "frozen"},
}


def build(asset: str, tag: str) -> dict:
    cfg = FINAL[asset]
    dec = json.loads((OUT_DIR / "final_decision_v2.json").read_text())[asset]
    cols = dec["feature_sets"][cfg["feature_set"]]
    ref = next(c for c in dec["cells"]
               if c["feature_set"] == cfg["feature_set"] and c["scaler"] == cfg["scaler"]
               and c["k"] == cfg["k"] and c["lambda_per_dim"] == cfg["lambda_per_dim"]
               and c["temperature_ratio"] == cfg["temperature_ratio"]
               and c["label_basis"] == cfg["label_basis"])
    print(f"\n=== {asset.upper()}  {cfg['feature_set']} ({len(cols)} features) "
          f"{cfg['scaler']} K={cfg['k']} lpd={cfg['lambda_per_dim']} "
          f"T={cfg['temperature_ratio']} [{cfg['label_basis']}]")
    print(f"    decision-run OOS bal={ref['oos_balanced_accuracy']:.4f} "
          f"detlag={ref['oos_detection_lag_median']:.1f} Q1={ref['oos_wave_Q1']:.3f}")

    frames = {y: _read(p) for y, p in SOURCES[asset].items()}
    ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
    label_cfg = (dict(LABEL_CONFIGS[LABEL_MODE]) if cfg["label_basis"] == "frozen"
                 else quantile_matched_label_config(frames[FIT_YEAR], ref_q))
    labels = {y: labels_for(f, label_cfg) for y, f in frames.items()}

    panels = {y: candidate_frame(f)[cols] for y, f in frames.items()}
    x_fit, scaler, medians, clip = fit_scale(panels[FIT_YEAR], cfg["scaler"])

    # verification 1: restricting the column set must be numerically a no-op
    pool = load_pool(asset, cfg["scaler"])
    pool_idx = [pool["cols"].index(c) for c in cols]
    max_dev = float(np.abs(x_fit - pool[f"x_{FIT_YEAR}"][:, pool_idx]).max())
    print(f"    scaler-restriction check: max |refit - cached| = {max_dev:.3e} "
          f"({'ok' if max_dev < 1e-9 else 'MISMATCH'})")

    lam = cfg["lambda_per_dim"] * len(cols)
    mu, obj = fit_jm(x_fit, k=cfg["k"], lam=lam, seed=SEED,
                     n_init=BUILD_N_INIT, n_iter=BUILD_N_ITER)
    x = {y: (x_fit if y == FIT_YEAR else apply_scale(panels[y], scaler, medians, clip))
         for y in frames}
    V = {y: causal_decode_V(v, mu, lam) for y, v in x.items()}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    temperature = cfg["temperature_ratio"] * spread
    sp = {y: softmax_states(v, temperature) for y, v in V.items()}
    state_class = _state_class_matrix(sp[FIT_YEAR], labels[FIT_YEAR])
    order, mean_ret = order_states_by_return(
        np.argmin(V[FIT_YEAR], axis=1),
        _num(frames[FIT_YEAR], "close").ffill().bfill().to_numpy(), cfg["k"])

    windows = {}
    for wname, (yr, start, end) in EVAL_WINDOWS.items():
        mask = slice_window(frames[yr]["timestamp"], start, end)
        close = _num(frames[yr], "close").ffill().bfill().to_numpy()
        pred = np.argmax(_class_proba(sp[yr], state_class), axis=1).astype(np.int64)
        wm = window_metrics(pred[mask], labels[yr][mask], close[mask])
        rl = run_lengths(pred[mask])
        wm["whipsaw_share"] = float((rl < WHIPSAW_MAX_BARS).mean()) if len(rl) else 1.0
        if wname in ("val", "oos"):
            oracle, pivots = zigzag_oracle(close, ORACLE_THETA)
            pos = wave_position(pivots, len(close))
            d = to_direction(pred)
            d[~mask] = 0
            idxs = np.flatnonzero(mask)
            wm["lag_peak_bars"] = lag_profile(d, oracle, mask)["peak_lag_bars"]
            wm["detection_lag_median"] = detection_lag(
                d, oracle, pivots, int(idxs[0]), int(idxs[-1]))["median_bars"]
            wm["wave_Q1"] = wave_quintiles(d, oracle, pos, mask)["Q1"]
        windows[wname] = wm

    drift = windows["oos"]["balanced_accuracy"] - ref["oos_balanced_accuracy"]
    verdict = "matches decision run" if abs(drift) <= REFIT_TOLERANCE else "DIVERGED"
    o = windows["oos"]
    print(f"    refit OOS bal={o['balanced_accuracy']:.4f} (drift {drift:+.4f}, {verdict})  "
          f"run={o['median_run_bars']:.0f} whip={o['whipsaw_share']:.2f} "
          f"detlag={o['detection_lag_median']:.1f} Q1={o['wave_Q1']:.3f} "
          f"sep={o['economic_separation_tstat']:+.2f}")

    payload = {
        "model_id": f"{asset}_regime3_current_jm_redesign_{tag}",
        "classes": CLASSES3,
        "prefix_stem": PREFIX_STEM,
        "output_column_prefix": f"{PREFIX_STEM}_wide24_",
        "feature_set_name": cfg["feature_set"],
        "panel_cols": cols,
        "scaler_kind": cfg["scaler"],
        "scaler": scaler,
        "feature_medians": medians.to_dict(),
        "winsor_lower": clip[0].to_dict(),
        "winsor_upper": clip[1].to_dict(),
        "label_basis": cfg["label_basis"],
        "label_config": label_cfg,
        "jm_mu": mu,
        "jm_k": cfg["k"],
        "jm_lambda": lam,
        "jm_lambda_per_dim": cfg["lambda_per_dim"],
        "jm_temperature": temperature,
        "jm_temperature_ratio": cfg["temperature_ratio"],
        "jm_v_spread_fit_median": spread,
        "jm_seed": SEED,
        "jm_fit_objective": obj,
        "jm_n_init": BUILD_N_INIT,
        "jm_n_iter": BUILD_N_ITER,
        "state_class_matrix": state_class,
        "state_order_bear_chop_bull": order,
        "state_mean_fwd12bar_logret": mean_ret,
        "fit_year": FIT_YEAR,
        "selection_rule": "consistency/persistence/timeliness gated vs replaced model; "
                          "tie-break OOS balanced accuracy",
    }
    SUPERVISED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SUPERVISED_DIR / f"{asset}_regime3_current_jm_redesign_{tag}_2024.joblib"
    joblib.dump(payload, model_path)
    print(f"    -> {model_path.name}")

    outputs = {}
    prefix = f"{PREFIX_STEM}_wide24_"
    for year, frame in frames.items():
        proba = _class_proba(sp[year], state_class)
        out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
        for i, name in enumerate(CLASSES3):
            out[f"{prefix}{name}_prob"] = proba[:, i]
        s = np.sort(proba, axis=1)
        out[f"{prefix}confidence"] = s[:, -1]
        out[f"{prefix}entropy"] = (-np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1)
                                   / np.log(len(CLASSES3)))
        out[f"{prefix}margin"] = s[:, -1] - s[:, -2]
        p = SUPERVISED_DIR / f"{asset}_regime3_current_hmm_{tag}_{year}_maskedname.csv"
        out.to_csv(p, index=False)
        outputs[year] = str(p.relative_to(ROOT))
        print(f"    -> {p.name}  ({len(out):,} rows, "
              f"conf mean {out[f'{prefix}confidence'].mean():.3f})")

    return {
        "asset": asset, "config": cfg, "features": cols, "model_path": str(model_path.relative_to(ROOT)),
        "outputs": outputs, "windows": windows,
        "decision_run_oos_balanced_accuracy": ref["oos_balanced_accuracy"],
        "refit_drift": drift, "refit_verdict": verdict,
        "scaler_restriction_max_dev": max_dev,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    ap.add_argument("--tag", default="jmredesign_20260810")
    args = ap.parse_args()
    report = {"tag": args.tag, "assets": {}}
    for asset in args.assets:
        report["assets"][asset] = build(asset, args.tag)
    p = OUT_DIR / f"build_final_{args.tag}_report.json"
    p.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nbuild report -> {p}")


if __name__ == "__main__":
    main()
