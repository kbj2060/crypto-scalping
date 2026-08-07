#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402


MODEL_ID = "omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260618"
OUT_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID
BUNDLE_PATH = OUT_DIR / "numeric_cash_sleeve.joblib"
MANIFEST_PATH = OUT_DIR / "candidate_manifest.json"
REPORT_PATH = ROOT / "tmp/causal_regen_20260516" / "omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616" / "report.json"
PARENT_MODEL_ID = "omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608"
TARGET_CANDIDATE = os.environ.get(
    "OMEGA1_2_8B_TARGET_CANDIDATE",
    "full_retrain_ev_cal0.80_ev0.004_numcfg0_u0.002_m0.000",
)


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


def _fit_final_lower_bound(
    x_val: pd.DataFrame,
    labels: pd.DataFrame,
    long_col: str,
    short_col: str,
    *,
    seed: int,
    cal_q: float,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, float, float]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_long = np.zeros(len(x_val), dtype=np.float64)
    y_short = np.zeros(len(x_val), dtype=np.float64)
    y_long[idx] = labels[long_col].to_numpy(dtype=np.float64)
    y_short[idx] = labels[short_col].to_numpy(dtype=np.float64)
    x_train = x_val.iloc[idx].to_numpy(dtype=np.float64)
    long_model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=9,
        l2_regularization=2.0,
        random_state=int(seed + 101),
    )
    short_model = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=9,
        l2_regularization=2.0,
        random_state=int(seed + 102),
    )
    long_model.fit(x_train, y_long[idx])
    short_model.fit(x_train, y_short[idx])
    long_offset = float(np.quantile(np.abs(y_long[idx] - long_model.predict(x_train)), cal_q))
    short_offset = float(np.quantile(np.abs(y_short[idx] - short_model.predict(x_train)), cal_q))
    return long_model, short_model, long_offset, short_offset


def _feature_support_profile(x_val: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    if len(idx) < 100:
        raise RuntimeError(f"Omega1.2.8b support profile requires at least 100 cash rows, got {len(idx)}")
    support = x_val.iloc[idx].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q_low = support.quantile(0.005, numeric_only=True).fillna(0.0)
    q_high = support.quantile(0.995, numeric_only=True).fillna(0.0)
    med = support.median(numeric_only=True).fillna(0.0)
    iqr = (support.quantile(0.75, numeric_only=True) - support.quantile(0.25, numeric_only=True)).replace(0.0, np.nan).fillna(1.0)
    return {
        "source": "validation_cash_rows",
        "rows": int(len(idx)),
        "quantile_low": 0.005,
        "quantile_high": 0.995,
        "min_fraction_in_support": 0.92,
        "max_robust_abs_z": 8.0,
        "median": {str(k): float(v) for k, v in med.to_dict().items()},
        "iqr": {str(k): float(v) for k, v in iqr.to_dict().items()},
        "low": {str(k): float(v) for k, v in q_low.to_dict().items()},
        "high": {str(k): float(v) for k, v in q_high.to_dict().items()},
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)) and isinstance(value, bool) is False:
        return float(value)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, (str, bytes)):
        s = str(value).strip().lower()
        if s in {"", "none", "null", "nan", "na", "n/a", "none"}:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


def _safe_int(value: Any) -> int:
    return int(round(_safe_float(value)))


def _load_selected_candidate(report: dict[str, Any], ranking_csv: Path, target_candidate: str) -> dict[str, Any]:
    selected = report.get("selected_by_validation")
    if isinstance(selected, dict) and str(selected.get("candidate", "")).strip() == target_candidate:
        return dict(selected)
    if not ranking_csv.exists():
        raise RuntimeError(f"target candidate {target_candidate} is not selected and ranking csv is missing: {ranking_csv}")
    with ranking_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("candidate", "")).strip() != target_candidate:
                continue
            return {
                "candidate": target_candidate,
                "family": str(row.get("family", "")),
                "utility_cfg_id": _safe_int(row.get("utility_cfg_id")),
                "cal_q": _safe_float(row.get("cal_q")),
                "ev_min": _safe_float(row.get("ev_min")),
                "utility_min": _safe_float(row.get("utility_min")),
                "margin_min": _safe_float(row.get("margin_min")),
            }
    raise RuntimeError(f"target candidate {target_candidate} not found in selected_by_validation nor ranking: {ranking_csv}")


def main() -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    ranking_path = REPORT_PATH.parent / "full_retrain_numeric_cash_sleeve_ranking.csv"
    selected = _load_selected_candidate(report, ranking_path, str(TARGET_CANDIDATE))
    if str(selected.get("candidate", "")) != str(TARGET_CANDIDATE):
        raise RuntimeError(
            f"unexpected Omega1.2.8b selected candidate: {selected.get('candidate')} != {TARGET_CANDIDATE}"
        )
    if str(selected["family"]) != "ev_lower_bound_numeric_agreement_veto":
        raise RuntimeError(f"unexpected Omega1.2.8b selected family: {selected['family']}")

    val_payload, _oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    path_labels, _path_diag = exp._path_label_table(val_payload, exp.RISK)
    ev_labels, _ev_diag = exp._utility_from_path_labels(
        path_labels,
        exp.RISK,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )
    support_profile = _feature_support_profile(x_val, ev_labels)
    utility_cfg_id = int(selected["utility_cfg_id"])
    utility_cfg = dict(exp.UTILITY_CFGS[utility_cfg_id])
    utility_labels, _utility_diag = exp._utility_from_path_labels(path_labels, exp.RISK, utility_cfg)

    ev_long_model, ev_short_model, ev_long_offset, ev_short_offset = _fit_final_lower_bound(
        x_val,
        ev_labels,
        "long_net",
        "short_net",
        seed=280000,
        cal_q=float(selected["cal_q"]),
    )
    utility_long_model, utility_short_model, utility_long_offset, utility_short_offset = _fit_final_lower_bound(
        x_val,
        utility_labels,
        "long_utility",
        "short_utility",
        seed=281000 + utility_cfg_id * 100,
        cal_q=0.50,
    )

    leverage = float(exp.RISK.leverage)
    risk = {
        **asdict(exp.RISK),
        "notional_exposure": float(exp.RISK.notional),
        "position_fraction": float(exp.RISK.notional) / max(leverage, 1e-12),
    }
    bundle = {
        "model_id": MODEL_ID,
        "base_model_id": PARENT_MODEL_ID,
        "parent_artifact": str(meta["parent_dir"]),
        "long_model": ev_long_model,
        "short_model": ev_short_model,
        "utility_long_model": utility_long_model,
        "utility_short_model": utility_short_model,
        "feature_cols": list(x_val.columns),
        "risk": risk,
        "ev_min": float(selected["ev_min"]),
        "utility_min": float(selected["utility_min"]),
        "margin_min": float(selected["margin_min"]),
        "support_profile": support_profile,
        "conservative_gate": {
            "type": "support_profile_and_conformal_lower_bound",
            "require_support_profile": True,
            "require_utility_agreement": True,
            "block_if_out_of_support": True,
            "block_if_nonpositive_lower_bound": True,
        },
        "utility_cfg_id": utility_cfg_id,
        "utility_cfg": utility_cfg,
        "calibration": {
            "type": "absolute_residual_lower_bound",
            "ev_quantile": float(selected["cal_q"]),
            "utility_quantile": 0.50,
            "long_abs_residual_offset": ev_long_offset,
            "short_abs_residual_offset": ev_short_offset,
            "long_utility_abs_residual_offset": utility_long_offset,
            "short_utility_abs_residual_offset": utility_short_offset,
            "selection_policy": str(report["selection_policy"]),
            "source_report": str(REPORT_PATH),
            "hf_paper_motivation": [
                "SPCI-style sequential/non-exchangeable time-series calibration",
                "CQL-style conservative/OOD action blocking",
            ],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    manifest = {
        "model_id": MODEL_ID,
        "alias": "omega1.2.8b_full_retrain_numeric_cash_sleeve_leverage_only",
        "status": "live_candidate_wired",
        "bundle": str(BUNDLE_PATH),
        "entrypoint": "scripts/train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616.py",
        "exporter": "scripts/export_omega1_2_8b_live_bundle_20260616.py",
        "report": str(REPORT_PATH),
        "parent_model_id": PARENT_MODEL_ID,
        "parent_artifact": str(meta["parent_dir"]),
        "feature_count": int(x_val.shape[1]),
        "feature_cols": list(x_val.columns),
        "risk": risk,
        "support_profile": {
            "source": support_profile["source"],
            "rows": int(support_profile["rows"]),
            "quantile_low": float(support_profile["quantile_low"]),
            "quantile_high": float(support_profile["quantile_high"]),
            "min_fraction_in_support": float(support_profile["min_fraction_in_support"]),
            "max_robust_abs_z": float(support_profile["max_robust_abs_z"]),
        },
        "conservative_gate": dict(bundle["conservative_gate"]),
        "selected_by_validation": selected,
        "redteam_pass": bool(report["redteam_pass"]),
        "redteam_blockers": list(report["redteam_blockers"]),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"bundle": str(BUNDLE_PATH), "manifest": str(MANIFEST_PATH)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
