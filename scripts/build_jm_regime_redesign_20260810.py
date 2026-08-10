"""Stage 3b of the JM-only regime3 redesign: refit the per-asset winner and emit the artifacts.

Takes the winning (panel, scaler, K, lambda_per_dim, temperature_ratio, label_basis) cell from the
sweep report, refits at the full restart budget (the sweep runs a reduced budget for throughput),
and writes:

  * data/ensemble/supervised/{asset}_regime3_current_jm_redesign_{TAG}_2024.joblib
      -- self-contained: panel name, frozen column list, scaler, fill medians, winsorization
         bounds, centroids, lambda, temperature, and the state->class matrix. Enough to rebuild
         the whole transform from a raw frame at live time, with no reference to the cache.
  * data/ensemble/supervised/{asset}_regime3_current_hmm_{TAG}_{year}_maskedname.csv

The CSV column prefix stays `regime3_current_sensitive_wide24_` even when the winning panel is NOT
wide24. That is deliberate and load-bearing: every downstream consumer (parent trainer, risk
sidecar, live runtime) keys on those literal column names, so keeping them makes a winner a drop-in
swap. The `wide24` token in the prefix is a historical name, not a claim about the input panel --
the joblib's `panel` field is the authority on what actually feeds the model.

The refit is verified against the sweep cell before anything is written: a materially different
VAL score means the coordinate descent landed on different centroids and the sweep's ranking does
not describe this artifact, so the difference is reported rather than silently absorbed.
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
    CLASSES3, EVAL_WINDOWS, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM, SELECTION_WINDOW,
    SOURCES, _class_proba, _read, _state_class_matrix, apply_scale, build_panel, causal_decode_V,
    choose_decorr_cols, fit_scale, fit_jm, labels_for, order_states_by_return, quantile_matched_label_config,
    reference_label_quantiles, slice_window, softmax_states, window_metrics,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.sweep_jm_regime_redesign_20260810 import SEED, gated_winner  # noqa: E402

SUPERVISED_DIR = ROOT / "data/ensemble/supervised"
BUILD_N_INIT = 5
BUILD_N_ITER = 15
REFIT_TOLERANCE = 0.01  # VAL balanced-accuracy drift that still counts as "the same model"


def build_asset(asset: str, tag: str, label_basis: str | None, criterion: str) -> dict:
    rep = json.loads((OUT_DIR / f"sweep_{asset}_report.json").read_text())
    cell = gated_winner(rep["cells"], label_basis=label_basis, criterion=criterion)
    basis = cell["label_basis"]
    panel, scaler_kind = cell["panel"], cell["scaler"]
    k, lpd, tratio = int(cell["k"]), float(cell["lambda_per_dim"]), float(cell["temperature_ratio"])
    print(f"[{asset}] winner: panel={panel} scaler={scaler_kind} k={k} lambda_per_dim={lpd} "
          f"T_ratio={tratio} label_basis={basis}")
    print(f"          sweep VAL bal_acc={cell[f'{SELECTION_WINDOW}_balanced_accuracy']:.4f} "
          f"OOS bal_acc={cell['oos_balanced_accuracy']:.4f}")

    frames = {year: _read(path) for year, path in SOURCES[asset].items()}
    fit_frame = frames[FIT_YEAR]

    if basis == "frozen":
        label_cfg = dict(LABEL_CONFIGS[LABEL_MODE])
    else:
        ref_q = reference_label_quantiles(_read(SOURCES["eth"][FIT_YEAR]))
        label_cfg = quantile_matched_label_config(fit_frame, ref_q)
    labels = {y: labels_for(f, label_cfg) for y, f in frames.items()}

    decorr_cols = choose_decorr_cols(fit_frame) if panel == "wide24_decorr" else None
    panels_by_year = {y: build_panel(f, panel, decorr_cols=decorr_cols) for y, f in frames.items()}
    cols = panels_by_year[FIT_YEAR][1]
    d = len(cols)
    lam = lpd * d

    x_fit, scaler, medians, clip_bounds = fit_scale(panels_by_year[FIT_YEAR][0], scaler_kind)
    print(f"[{asset}] refitting JM k={k} lambda={lam:.3f} (d={d}) "
          f"n_init={BUILD_N_INIT} n_iter={BUILD_N_ITER}")
    mu, obj = fit_jm(x_fit, k=k, lam=lam, seed=SEED, n_init=BUILD_N_INIT, n_iter=BUILD_N_ITER)

    x = {y: (x_fit if y == FIT_YEAR else apply_scale(panels_by_year[y][0], scaler, medians, clip_bounds))
         for y in frames}
    V = {y: causal_decode_V(xv, mu, lam) for y, xv in x.items()}
    v_spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    temperature = tratio * v_spread
    sp = {y: softmax_states(Vv, temperature) for y, Vv in V.items()}
    state_class = _state_class_matrix(sp[FIT_YEAR], labels[FIT_YEAR])

    order, mean_ret = order_states_by_return(np.argmin(V[FIT_YEAR], axis=1),
                                             pd.to_numeric(fit_frame["close"], errors="coerce").to_numpy(), k)

    windows = {}
    for name, (yr, start, end) in EVAL_WINDOWS.items():
        m = slice_window(frames[yr]["timestamp"], start, end)
        proba = _class_proba(sp[yr][m], state_class)
        windows[name] = window_metrics(np.argmax(proba, axis=1).astype(np.int64),
                                       labels[yr][m],
                                       pd.to_numeric(frames[yr]["close"], errors="coerce").to_numpy()[m])
    drift = windows[SELECTION_WINDOW]["balanced_accuracy"] - cell[f"{SELECTION_WINDOW}_balanced_accuracy"]
    verdict = "matches sweep" if abs(drift) <= REFIT_TOLERANCE else "DIVERGED FROM SWEEP"
    print(f"[{asset}] refit VAL bal_acc={windows[SELECTION_WINDOW]['balanced_accuracy']:.4f} "
          f"(drift {drift:+.4f}, {verdict})  OOS bal_acc={windows['oos']['balanced_accuracy']:.4f}")

    payload = {
        "model_id": f"{asset}_regime3_current_jm_redesign_{tag}",
        "classes": CLASSES3,
        "prefix_stem": PREFIX_STEM,
        "output_column_prefix": f"{PREFIX_STEM}_wide24_",
        "panel": panel,
        "panel_cols": cols,
        "panel_decorr_cols": decorr_cols,
        "scaler_kind": scaler_kind,
        "scaler": scaler,
        "feature_medians": medians.to_dict(),
        "winsor_lower": clip_bounds[0].to_dict(),
        "winsor_upper": clip_bounds[1].to_dict(),
        "label_basis": basis,
        "label_config": label_cfg,
        "jm_mu": mu,
        "jm_k": k,
        "jm_lambda": lam,
        "jm_lambda_per_dim": lpd,
        "jm_temperature": temperature,
        "jm_temperature_ratio": tratio,
        "jm_v_spread_fit_median": v_spread,
        "jm_seed": SEED,
        "jm_fit_objective": obj,
        "state_class_matrix": state_class,
        "state_order_bear_chop_bull": order,
        "state_mean_fwd12bar_logret": mean_ret,
        "fit_year": FIT_YEAR,
        "selection_window": SELECTION_WINDOW,
    }
    SUPERVISED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SUPERVISED_DIR / f"{asset}_regime3_current_jm_redesign_{tag}_2024.joblib"
    joblib.dump(payload, model_path)
    print(f"[{asset}] -> {model_path}")

    outputs = {}
    prefix = f"{PREFIX_STEM}_wide24_"
    for year, frame in frames.items():
        proba = _class_proba(sp[year], state_class)
        out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
        for i, name in enumerate(CLASSES3):
            out[f"{prefix}{name}_prob"] = proba[:, i]
        s = np.sort(proba, axis=1)
        out[f"{prefix}confidence"] = s[:, -1]
        out[f"{prefix}entropy"] = -np.sum(proba * np.log(np.clip(proba, 1e-12, None)), axis=1) / np.log(len(CLASSES3))
        out[f"{prefix}margin"] = s[:, -1] - s[:, -2]
        out_path = SUPERVISED_DIR / f"{asset}_regime3_current_hmm_{tag}_{year}_maskedname.csv"
        out.to_csv(out_path, index=False)
        outputs[year] = str(out_path.relative_to(ROOT))
        print(f"[{asset}]   {year} -> {out_path.name}")

    return {
        "asset": asset,
        "winner_cell": cell,
        "refit": {"lambda": lam, "dim": d, "temperature": temperature,
                  "v_spread_fit_median": v_spread, "fit_objective": obj,
                  "n_init": BUILD_N_INIT, "n_iter": BUILD_N_ITER},
        "refit_vs_sweep_val_drift": drift,
        "refit_verdict": verdict,
        "label_config": label_cfg,
        "windows": windows,
        "model_path": str(model_path.relative_to(ROOT)),
        "outputs": outputs,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    ap.add_argument("--tag", default="jmredesign_20260810")
    ap.add_argument("--label-basis", default=None,
                    help="frozen | qmatched; default picks the better of the two on VAL")
    ap.add_argument("--criterion", default="balanced_accuracy",
                    choices=["balanced_accuracy", "economic_separation_tstat"])
    args = ap.parse_args()

    report = {"tag": args.tag, "criterion": args.criterion, "assets": {}}
    for asset in args.assets:
        report["assets"][asset] = build_asset(asset, args.tag, args.label_basis, args.criterion)
    path = OUT_DIR / f"build_{args.tag}_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nbuild report -> {path}")


if __name__ == "__main__":
    main()
