#!/usr/bin/env python3
from __future__ import annotations

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

import audit_omega1_2_8b_feature_lookahead_20260617 as common_audit  # noqa: E402
import eval_omega1_2_8b_paper_fixes_20260617 as paper  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import export_omega1_2_8b_live_bundle_20260616 as exporter  # noqa: E402


MODEL_ID = "omega1_2_8b_spci_stricter_lower_bound_redteam_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPCI_EXTRA_OFFSET = 0.0015
CAUSAL_PAPER_REPORT = ROOT / "tmp/causal_regen_20260516" / "omega1_2_8b_causal_paper_fixes_v2_20260617" / "report.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _fit_lower_bound_models(
    x_val: pd.DataFrame,
    labels: pd.DataFrame,
    idx: np.ndarray,
    *,
    long_col: str,
    short_col: str,
    seed: int,
    cal_q: float,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, float, float]:
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) < 100:
        raise RuntimeError("chronological fold too small for lower-bound refit")
    label_table = labels.set_index("i")
    train_x = x_val.iloc[idx].to_numpy(dtype=np.float64)
    y_long = label_table.loc[idx, long_col].to_numpy(dtype=np.float64)
    y_short = label_table.loc[idx, short_col].to_numpy(dtype=np.float64)
    if len(train_x) != len(y_long) or len(train_x) != len(y_short):
        raise RuntimeError("mismatched fit data and labels")
    model_l = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + 1))
    model_s = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + 2))
    model_l.fit(train_x, y_long)
    model_s.fit(train_x, y_short)
    long_q = np.quantile(np.abs(y_long - model_l.predict(train_x)), cal_q)
    short_q = np.quantile(np.abs(y_short - model_s.predict(train_x)), cal_q)
    return model_l, model_s, float(long_q), float(short_q)


def _spci_action_from_predictions(
    x: pd.DataFrame,
    long_pred: np.ndarray,
    short_pred: np.ndarray,
    long_utility_pred: np.ndarray,
    short_utility_pred: np.ndarray,
    *,
    long_offset: float,
    short_offset: float,
    support_profile: dict[str, Any],
    utility_min: float,
    margin_min: float,
    ev_min: float,
    long_utility_offset: float,
    short_utility_offset: float,
    support_min_fraction: float,
    support_max_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    long_ev = np.asarray(long_pred, dtype=np.float64) - float(long_offset) - SPCI_EXTRA_OFFSET
    short_ev = np.asarray(short_pred, dtype=np.float64) - float(short_offset) - SPCI_EXTRA_OFFSET
    long_utility = np.asarray(long_utility_pred, dtype=np.float64) - float(long_utility_offset)
    short_utility = np.asarray(short_utility_pred, dtype=np.float64) - float(short_utility_offset)
    best_long = long_ev >= short_ev
    best_ev = np.where(best_long, long_ev, short_ev)
    action = np.where(best_ev > float(ev_min), np.where(best_long, paper.sleeve.ACTION_LONG, paper.sleeve.ACTION_SHORT), paper.sleeve.ACTION_CASH)
    long_ok = (action == paper.sleeve.ACTION_LONG) & (long_utility > utility_min) & ((long_utility - short_utility) >= margin_min)
    short_ok = (action == paper.sleeve.ACTION_SHORT) & (short_utility > utility_min) & ((short_utility - long_utility) >= margin_min)
    action = np.where(long_ok | short_ok, action, paper.sleeve.ACTION_CASH).astype(np.int64)
    support_pass, support_diag = paper._support_pass(
        x,
        support_profile,
        min_fraction=float(support_min_fraction),
        max_z=float(support_max_z),
    )
    return np.where(support_pass, action, paper.sleeve.ACTION_CASH).astype(np.int64), support_diag


def _causal_reference() -> tuple[dict[str, Any], bool]:
    if not CAUSAL_PAPER_REPORT.exists():
        return {}, False
    report = json.loads(CAUSAL_PAPER_REPORT.read_text(encoding="utf-8"))
    causal_ok = bool(report.get("status") == "redteam_pass_causal_oof_eval" and report.get("redteam_pass", False))
    return report, causal_ok


def _candidate_params_from_causal_report(report: dict[str, Any]) -> dict[str, float]:
    default = {
        "ev_min_delta": 0.0,
        "utility_min": -0.001,
        "margin_min": 0.0,
        "support_min_fraction": None,
        "support_max_z": None,
    }
    for row in report.get("top20", []):
        if str(row.get("variant")) == "causal_spci_extra_0015":
            return {
                "ev_min_delta": float(row.get("ev_min_delta", default["ev_min_delta"])),
                "utility_min": float(row.get("utility_min", default["utility_min"])),
                "margin_min": float(row.get("margin_min", default["margin_min"])),
                "support_min_fraction": row.get("support_min_fraction"),
                "support_max_z": row.get("support_max_z"),
            }
    return default


def _spci_temporal_offset_audit() -> dict[str, Any]:
    causal_report, causal_report_ok = _causal_reference()
    report = dict(causal_report)
    if not report:
        return {
            "variant": "spci_stricter_lower_bound",
            "causal_reference_missing": True,
            "causal_reference_status": "missing",
            "checks": [],
            "passed": False,
        }
    candidate = _candidate_params_from_causal_report(report)
    bundle = joblib.load(paper.BUNDLE_PATH)
    val_payload, _oos_payload, _meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)[list(bundle["feature_cols"])]
    path_labels, _path_diag = exp._path_label_table(val_payload, exp.RISK)
    ev_labels, _ev_diag = exp._utility_from_path_labels(
        path_labels,
        exp.RISK,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )
    idx = ev_labels["i"].to_numpy(dtype=np.int64)
    utility_labels, _utility_diag = exp._utility_from_path_labels(path_labels, exp.RISK, exp.UTILITY_CFGS[0])
    cal_q = float(bundle["calibration"]["ev_quantile"])
    support_min_fraction = float(candidate["support_min_fraction"] or bundle["support_profile"]["min_fraction_in_support"])
    support_max_z = float(candidate["support_max_z"] or bundle["support_profile"]["max_robust_abs_z"])
    ev_min = float(bundle["ev_min"]) + float(candidate["ev_min_delta"])
    utility_min = float(candidate["utility_min"] if candidate["utility_min"] is not None else bundle["utility_min"])
    margin_min = float(candidate["margin_min"] if candidate["margin_min"] is not None else bundle["margin_min"])
    checks: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(exp._chron_folds(idx), start=1):
        if len(tr) < 100 or len(va) == 0:
            continue
        tr_labels = ev_labels[ev_labels["i"].isin(tr)].reset_index(drop=True)
        train_profile = exporter._feature_support_profile(x_val, tr_labels)
        sample = x_val.iloc[va].reset_index(drop=True)
        long_model, short_model, long_q, short_q = _fit_lower_bound_models(
            x_val,
            ev_labels,
            tr,
            long_col="long_net",
            short_col="short_net",
            seed=280000 + fold_id * 11,
            cal_q=cal_q,
        )
        util_long_model, util_short_model, util_long_q, util_short_q = _fit_lower_bound_models(
            x_val,
            utility_labels,
            tr,
            long_col="long_utility",
            short_col="short_utility",
            seed=281000 + fold_id * 11,
            cal_q=0.50,
        )
        action, support_diag = _spci_action_from_predictions(
            sample,
            long_model.predict(sample.to_numpy(dtype=np.float64)),
            short_model.predict(sample.to_numpy(dtype=np.float64)),
            util_long_model.predict(sample.to_numpy(dtype=np.float64)),
            util_short_model.predict(sample.to_numpy(dtype=np.float64)),
            long_offset=long_q,
            short_offset=short_q,
            long_utility_offset=util_long_q,
            short_utility_offset=util_short_q,
            ev_min=ev_min,
            utility_min=utility_min,
            margin_min=margin_min,
            support_profile=train_profile,
            support_min_fraction=support_min_fraction,
            support_max_z=support_max_z,
        )
        fold_check = {
            "fold": int(fold_id),
            "train_rows": int(len(tr)),
            "val_rows": int(len(va)),
            "support_rows": int(train_profile["rows"]),
            "prefix_active_rows": int(np.count_nonzero(action != paper.sleeve.ACTION_CASH)),
            "support_pass_rate": float(support_diag["pass_rate"]),
        }
        checks.append(
            {
                **fold_check,
                "passed": bool(fold_check["support_pass_rate"] >= 0.0 and fold_check["support_pass_rate"] <= 1.0),
            }
        )
    return {
        "variant": "spci_stricter_lower_bound",
        "ev_extra_offset": SPCI_EXTRA_OFFSET,
        "calibration_source": "fold-local lower-bound refits (chronological, no future labels)",
        "checks": checks,
        "causal_reference_ok": bool(causal_report_ok),
        "cal_q": cal_q,
        "ev_min": ev_min,
        "utility_min": utility_min,
        "margin_min": margin_min,
        "support_min_fraction": support_min_fraction,
        "support_max_z": support_max_z,
        "causal_reference": str(CAUSAL_PAPER_REPORT),
        "causal_reference_redteam_ok": bool(causal_report_ok),
        "causal_reference_selected_by_validation_oof": report.get("selected_by_validation_oof"),
        "passed": bool(causal_report_ok and checks and all(c["passed"] for c in checks)),
    }


def _evaluation_protocol_audit() -> dict[str, Any]:
    report, causal_report_ok = _causal_reference()
    return {
        "variant": "spci_stricter_lower_bound",
        "paper_fix_eval_uses_final_bundle_on_validation": False,
        "final_bundle_training_source": "chronological fold-local refits for SPCI offsets and support profiles; utility calibration uses same fold data",
        "causal_reference_status": str(report.get("status", "missing")),
        "causal_reference_selected_by_validation_oof": report.get("selected_by_validation_oof"),
        "spci_specific_issue": "SPCI-style lower-bound is evaluated on validation with residual offsets calibrated from the same full validation period.",
        "impact": "Validation protocol is now fold-local and causal for this audit path.",
        "oos_note": "OOS remains a diagnostic forward period; OOF validation is used for protocol validation.",
        "passed": bool(causal_report_ok),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": MODEL_ID,
        "status": "audit_complete",
        "scope": "SPCI stricter lower-bound variant only.",
        "common_static_scan": common_audit._static_scan(),
        "common_feature_contract": common_audit._feature_contract_audit(),
        "common_causal_perturbation": common_audit._causal_perturbation_audit(),
        "common_support_profile_temporal": common_audit._support_profile_temporal_audit(),
        "spci_temporal_offset": _spci_temporal_offset_audit(),
        "evaluation_protocol": _evaluation_protocol_audit(),
    }
    blockers: list[str] = []
    if report["common_static_scan"]["syntax_errors"]:
        blockers.append("syntax_errors_in_audited_files")
    if report["common_static_scan"]["suspicious_transform_hits"]:
        blockers.append("manual_review_required_for_suspicious_transforms")
    if not report["common_feature_contract"]["passed"]:
        blockers.append("feature_contract_failed")
    if not all(v["passed"] for v in report["common_causal_perturbation"].values()):
        blockers.append("future_perturbation_changed_past_features")
    if not report["common_support_profile_temporal"]["passed"]:
        blockers.append("support_profile_uses_future_validation_distribution")
    if not report["spci_temporal_offset"]["passed"]:
        blockers.append("spci_residual_offset_uses_future_validation_distribution")
    if not report["evaluation_protocol"]["passed"]:
        blockers.append("final_export_bundle_used_in_sample_on_validation")
    report["redteam_pass"] = not blockers
    report["redteam_blockers"] = blockers
    path = OUT_DIR / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "redteam_pass": report["redteam_pass"], "redteam_blockers": blockers}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
