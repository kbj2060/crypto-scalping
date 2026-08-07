#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as base8b  # noqa: E402
import train_eval_omega1_2_8b_regime_threshold_cash_sleeve_20260618 as threshold_exp  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8b_regime_expert_cash_sleeve_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
EV_GRID = (0.001, 0.002, 0.003, 0.004, 0.005, 0.006)
REGIMES = threshold_exp.REGIMES
MIN_EXPERT_LABEL_ROWS = 500


def _make_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=9,
        l2_regularization=2.0,
        random_state=int(seed),
    )


def _fit_predict_lower_bound_by_regime(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    val_regime: np.ndarray,
    oos_regime: np.ndarray,
    long_col: str,
    short_col: str,
    *,
    seed: int,
    cal_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    val_long = np.zeros(len(x_val), dtype=np.float64)
    val_short = np.zeros(len(x_val), dtype=np.float64)
    oos_long = np.zeros(len(x_oos), dtype=np.float64)
    oos_short = np.zeros(len(x_oos), dtype=np.float64)
    diag: dict[str, Any] = {"target_cols": [long_col, short_col], "cal_q": float(cal_q), "experts": {}}

    for expert_id, regime in enumerate(REGIMES):
        regime_labels = labels[val_regime[labels["i"].to_numpy(dtype=np.int64)] == regime].sort_values("i").reset_index(drop=True)
        if len(regime_labels) < MIN_EXPERT_LABEL_ROWS:
            raise RuntimeError(f"{regime} expert has too few label rows: {len(regime_labels)} < {MIN_EXPERT_LABEL_ROWS}")
        idx = regime_labels["i"].to_numpy(dtype=np.int64)
        y_long = regime_labels[long_col].to_numpy(dtype=np.float64)
        y_short = regime_labels[short_col].to_numpy(dtype=np.float64)
        folds_meta: list[dict[str, Any]] = []
        n = len(idx)

        for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
            train_end = int(n * train_frac)
            val_end = int(n * end_frac)
            if train_end < 100 or val_end <= train_end:
                continue
            train_pos = np.arange(train_end)
            val_pos = np.arange(train_end, val_end)
            train_idx = idx[train_pos]
            val_idx = idx[val_pos]
            ml = _make_regressor(seed + expert_id * 1000 + fold_id * 10 + 1)
            ms = _make_regressor(seed + expert_id * 1000 + fold_id * 10 + 2)
            x_train = x_val.iloc[train_idx].to_numpy(dtype=np.float64)
            ml.fit(x_train, y_long[train_pos])
            ms.fit(x_train, y_short[train_pos])
            ql = float(np.quantile(np.abs(y_long[train_pos] - ml.predict(x_train)), cal_q))
            qs = float(np.quantile(np.abs(y_short[train_pos] - ms.predict(x_train)), cal_q))
            val_long[val_idx] = ml.predict(x_val.iloc[val_idx].to_numpy(dtype=np.float64)).astype(np.float64) - ql
            val_short[val_idx] = ms.predict(x_val.iloc[val_idx].to_numpy(dtype=np.float64)).astype(np.float64) - qs
            folds_meta.append({"fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx)), "long_abs_resid_q": ql, "short_abs_resid_q": qs})

        final_long = _make_regressor(seed + expert_id * 1000 + 901)
        final_short = _make_regressor(seed + expert_id * 1000 + 902)
        x_train_all = x_val.iloc[idx].to_numpy(dtype=np.float64)
        final_long.fit(x_train_all, y_long)
        final_short.fit(x_train_all, y_short)
        ql = float(np.quantile(np.abs(y_long - final_long.predict(x_train_all)), cal_q))
        qs = float(np.quantile(np.abs(y_short - final_short.predict(x_train_all)), cal_q))
        oos_mask = oos_regime == regime
        if bool(oos_mask.any()):
            oos_long[oos_mask] = final_long.predict(x_oos.loc[oos_mask].to_numpy(dtype=np.float64)).astype(np.float64) - ql
            oos_short[oos_mask] = final_short.predict(x_oos.loc[oos_mask].to_numpy(dtype=np.float64)).astype(np.float64) - qs
        diag["experts"][regime] = {
            "label_rows": int(len(idx)),
            "oos_rows": int(oos_mask.sum()),
            "folds": folds_meta,
            "final_long_abs_resid_q": ql,
            "final_short_abs_resid_q": qs,
            "long_positive_labels": int((y_long > 0.0).sum()),
            "short_positive_labels": int((y_short > 0.0).sum()),
        }

    return val_long, val_short, oos_long, oos_short, diag


def _actions(
    scores: dict[str, np.ndarray],
    regimes: np.ndarray,
    thresholds: dict[str, float],
    *,
    utility_min: float,
    margin_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    return threshold_exp._actions(scores, regimes, thresholds, utility_min=utility_min, margin_min=margin_min)


def _risk_from_bundle(bundle: dict[str, Any]) -> sleeve.FallbackRisk:
    risk_payload = dict(bundle["risk"])
    return sleeve.FallbackRisk(
        str(risk_payload["name"]),
        float(risk_payload["take_profit"]),
        float(risk_payload["stop_loss"]),
        float(risk_payload["notional"]),
        float(risk_payload["leverage"]),
        int(risk_payload["max_hold_bars"]),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = threshold_exp._load_bundle()
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = base8b._build_payloads()
    feature_cols = list(bundle["feature_cols"])
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[feature_cols]
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[feature_cols]
    val_regime = threshold_exp._route_regime(x_val)
    oos_regime = threshold_exp._route_regime(x_oos)
    risk = _risk_from_bundle(bundle)
    fee = float(meta["fee"])
    slip = float(meta["slip"])

    print(json.dumps({"stage": "labels"}, ensure_ascii=True), flush=True)
    path_labels, path_diag = base8b._path_label_table(val_payload, risk)
    ev_labels, ev_diag = base8b._utility_from_path_labels(
        path_labels,
        risk,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )
    utility_cfg = dict(base8b.UTILITY_CFGS[int(bundle.get("utility_cfg_id", 0) or 0)])
    utility_labels, utility_diag = base8b._utility_from_path_labels(path_labels, risk, utility_cfg)

    print(json.dumps({"stage": "fit_regime_experts", "target": "ev"}, ensure_ascii=True), flush=True)
    ev_vl, ev_vs, ev_ol, ev_os, ev_fit_diag = _fit_predict_lower_bound_by_regime(
        x_val,
        x_oos,
        ev_labels,
        val_regime,
        oos_regime,
        "long_net",
        "short_net",
        seed=618000,
        cal_q=float(bundle["calibration"]["ev_quantile"]),
    )
    print(json.dumps({"stage": "fit_regime_experts", "target": "utility"}, ensure_ascii=True), flush=True)
    u_vl, u_vs, u_ol, u_os, utility_fit_diag = _fit_predict_lower_bound_by_regime(
        x_val,
        x_oos,
        utility_labels,
        val_regime,
        oos_regime,
        "long_utility",
        "short_utility",
        seed=619000,
        cal_q=float(bundle["calibration"].get("utility_quantile", 0.50)),
    )

    val_scores = {
        "long_ev": ev_vl,
        "short_ev": ev_vs,
        "long_utility": u_vl,
        "short_utility": u_vs,
        "support_pass": threshold_exp._support_pass(x_val, bundle),
    }
    oos_scores = {
        "long_ev": ev_ol,
        "short_ev": ev_os,
        "long_utility": u_ol,
        "short_utility": u_os,
        "support_pass": threshold_exp._support_pass(x_oos, bundle),
    }
    live_val_scores = threshold_exp._score_bundle(x_val, bundle)
    live_oos_scores = threshold_exp._score_bundle(x_oos, bundle)

    base_val_parent = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos_parent = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val = {**base_val_parent, "primary_entries": base_val_parent["long_entries"] + base_val_parent["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    base_oos = {**base_oos_parent, "primary_entries": base_oos_parent["long_entries"] + base_oos_parent["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    utility_min = float(bundle["utility_min"])
    margin_min = float(bundle["margin_min"])

    rows: list[dict[str, Any]] = []
    for combo in itertools.product(EV_GRID, repeat=3):
        thresholds = {regime: float(value) for regime, value in zip(REGIMES, combo)}
        val_a, val_c = _actions(val_scores, val_regime, thresholds, utility_min=utility_min, margin_min=margin_min)
        oos_a, oos_c = _actions(oos_scores, oos_regime, thresholds, utility_min=utility_min, margin_min=margin_min)
        val_m = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], risk, val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        oos_m = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], risk, oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
        name = "regime_expert_b{bull:.3f}_r{bear:.3f}_c{chop:.3f}".format(**thresholds)
        rows.append(threshold_exp._metric_row(name, thresholds, val_m, oos_m, base_val, base_oos))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "regime_expert_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    selected_thresholds = {r: float(selected[f"{r}_ev_min"]) for r in REGIMES}
    val_selected_a, _ = _actions(val_scores, val_regime, selected_thresholds, utility_min=utility_min, margin_min=margin_min)
    oos_selected_a, _ = _actions(oos_scores, oos_regime, selected_thresholds, utility_min=utility_min, margin_min=margin_min)

    global_thresholds = {r: float(bundle["ev_min"]) for r in REGIMES}
    val_global_a, val_global_c = _actions(live_val_scores, val_regime, global_thresholds, utility_min=utility_min, margin_min=margin_min)
    oos_global_a, oos_global_c = _actions(live_oos_scores, oos_regime, global_thresholds, utility_min=utility_min, margin_min=margin_min)
    global_val = sleeve._metrics_with_fallback(val_payload["frame"], val_payload["dec"], risk, val_global_a, val_global_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
    global_oos = sleeve._metrics_with_fallback(oos_payload["frame"], oos_payload["dec"], risk, oos_global_a, oos_global_c, 0.0, fee=fee, slip=slip, cost_mult=3.0)
    global_control = threshold_exp._metric_row("live_8b_global_model_control", global_thresholds, global_val, global_oos, base_val, base_oos)

    telemetry = {
        "validation_regime_rows": {r: int((val_regime == r).sum()) for r in REGIMES},
        "oos_regime_rows": {r: int((oos_regime == r).sum()) for r in REGIMES},
        "selected_policy_counts": {
            "validation": threshold_exp._policy_counts(val_regime, val_selected_a),
            "oos": threshold_exp._policy_counts(oos_regime, oos_selected_a),
        },
    }
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_regime_expert_eval",
        "method": "Train separate bull/bear/chop EV lower-bound and utility lower-bound sleeve experts on validation cash-label rows. Route by parent Regime3 current router. Select only thresholds on validation; OOS is diagnostic only.",
        "bundle_path": str(threshold_exp.BUNDLE_PATH),
        "risk": dict(bundle["risk"]),
        "utility_cfg": utility_cfg,
        "utility_min": utility_min,
        "margin_min": margin_min,
        "ev_grid": list(EV_GRID),
        "baseline_parent_only": {"validation": base_val, "oos": base_oos},
        "live_global_model_control": global_control,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "diagnostics": {
            "path_labels": path_diag,
            "ev_labels": ev_diag,
            "utility_labels": utility_diag,
            "ev_fit": ev_fit_diag,
            "utility_fit": utility_fit_diag,
        },
        "telemetry": telemetry,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "regime_expert_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=threshold_exp._json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "global_control": global_control}, indent=2, ensure_ascii=True, default=threshold_exp._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
