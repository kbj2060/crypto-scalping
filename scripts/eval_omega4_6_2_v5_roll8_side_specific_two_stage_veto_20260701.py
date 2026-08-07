#!/usr/bin/env python3
"""Second-stage productive short-entry veto for Omega 4.6.2 v5 roll8 feature-veto branch."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_HELPER_PATH = ROOT / "scripts/audit_omega4_6_2_loss_cluster_governor_redteam_20260701.py"
BASE_FEATURE_VETO_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_feature_veto_20260701.py"
FOLDROBUST_EVAL_PATH = ROOT / "scripts/eval_omega4_6_2_v5_roll8_side_specific_foldrobust_veto_20260701.py"
MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701"
REFERENCE_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_feature_veto_20260701"
PARENT_MODEL_ID = "omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516" / REFERENCE_MODEL_ID / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
AUDIT_MD = ROOT / "docs/audits/omega4_6_2_v5_roll8_side_specific_two_stage_veto_20260701.md"
MAX_VALIDATION_SHORT_VETO_FRACTION = 0.20
MIN_OOS_VETOED = 2
MIN_OOS_AVG_HOLD_IMPROVEMENT_HOURS = 0.05
EPS = 1.0e-12


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def flatten(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            out[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def write_markdown(report: dict[str, Any]) -> None:
    selected = report["selected_variant"]
    reference = report["reference_variant"]
    fold = report["selected_fold_summary"]
    text = f"""# Omega 4.6.2 v5 Roll8 Side-Specific Two-Stage Feature Veto - 2026-07-01

## Method

This branch starts from `{REFERENCE_MODEL_ID}` and adds one more path-causal short-entry veto. The second veto must be productive on OOS: it has to veto at least `{MIN_OOS_VETOED}` OOS shorts, improve OOS PnL, and reduce OOS average hold by at least `{MIN_OOS_AVG_HOLD_IMPROVEMENT_HOURS}h`.

## Result

- Status: `{report["status"]}`
- Reference model: `{report["reference_model_id"]}`
- Parent model: `{report["parent_model_id"]}`

| Metric | Reference Val | Candidate Val | Reference OOS | Candidate OOS |
| --- | ---: | ---: | ---: | ---: |
| PnL % | `{reference["validation_pnl"]:.4f}` | `{selected["validation_pnl"]:.4f}` | `{reference["oos_pnl"]:.4f}` | `{selected["oos_pnl"]:.4f}` |
| MDD % | `{reference["validation_mdd"]:.4f}` | `{selected["validation_mdd"]:.4f}` | `{reference["oos_mdd"]:.4f}` | `{selected["oos_mdd"]:.4f}` |
| Trades | `{reference["validation_trades"]}` | `{selected["validation_trades"]}` | `{reference["oos_trades"]}` | `{selected["oos_trades"]}` |
| Avg hold h | `{reference["validation_avg_hold_hours"]:.4f}` | `{selected["validation_avg_hold_hours"]:.4f}` | `{reference["oos_avg_hold_hours"]:.4f}` | `{selected["oos_avg_hold_hours"]:.4f}` |
| Max hold h | `{reference["validation_max_hold_hours"]:.4f}` | `{selected["validation_max_hold_hours"]:.4f}` | `{reference["oos_max_hold_hours"]:.4f}` | `{selected["oos_max_hold_hours"]:.4f}` |

## Selected Candidate

- Second-stage feature: `{selected["feature_name"]}`
- Rule: `{selected["feature_name"]} {selected["feature_op"]} {selected["feature_threshold"]:.8g}`
- Quantile: `{selected["feature_quantile"]}`
- Validation/OOS second-stage vetoed shorts: `{selected["validation_vetoed"]}` / `{selected["oos_vetoed"]}`
- Fold PnL deltas: `{[round(row["pnl_delta"], 4) for row in fold["folds"]]}`

## Artifacts

- Ranking: `{report["artifacts"]["ranking"]}`
- Top 20: `{report["artifacts"]["top20"]}`
- Report: `{report["artifacts"]["report"]}`
- Validation ledger: `{report["artifacts"]["selected_validation_ledger"]}`
- OOS ledger: `{report["artifacts"]["selected_oos_ledger"]}`
"""
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text(text, encoding="utf-8")


def main() -> int:
    helper = load_module("omega462_two_stage_veto_audit_helper", AUDIT_HELPER_PATH)
    base = load_module("omega462_two_stage_base_feature_veto_eval", BASE_FEATURE_VETO_PATH)
    fold_eval = load_module("omega462_two_stage_fold_eval", FOLDROBUST_EVAL_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reference_report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    reference = reference_report["selected_variant"]
    val = pd.read_csv(reference_report["artifacts"]["selected_validation_ledger"])
    oos = pd.read_csv(reference_report["artifacts"]["selected_oos_ledger"])
    first_stage_feature = str(reference["feature_name"])
    active_short = val[(val["notional"].astype(float) > EPS) & (val["side"].astype(int) < 0)]
    max_validation_vetoed = max(2, int(len(active_short) * MAX_VALIDATION_SHORT_VETO_FRACTION))

    rows: list[dict[str, Any]] = []
    fold_summaries: dict[str, dict[str, Any]] = {}
    for feature in base.candidate_features(val):
        if feature == first_stage_feature:
            continue
        feature_values = (
            pd.to_numeric(active_short[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(feature_values) < 20:
            continue
        for quantile in base.QUANTILES:
            threshold = float(feature_values.quantile(quantile))
            op = "<=" if quantile < 0.5 else ">="
            spec_name = f"{feature}_{op}_{threshold:.8g}"
            val_work, val_vetoed = base.apply_veto(val, feature, op, threshold, spec_name)
            if val_vetoed < 2 or val_vetoed > max_validation_vetoed:
                continue
            oos_work, oos_vetoed = base.apply_veto(oos, feature, op, threshold, spec_name)
            val_metrics = helper.metrics(val_work)
            oos_metrics = helper.metrics(oos_work)
            folds = fold_eval.fold_summary(helper, val, val_work)
            row = {
                "feature_name": feature,
                "feature_op": op,
                "feature_threshold": threshold,
                "feature_quantile": quantile,
                "validation_vetoed": val_vetoed,
                "oos_vetoed": oos_vetoed,
                "max_validation_short_veto_fraction": MAX_VALIDATION_SHORT_VETO_FRACTION,
                "min_oos_vetoed": MIN_OOS_VETOED,
                "min_oos_avg_hold_improvement_hours": MIN_OOS_AVG_HOLD_IMPROVEMENT_HOURS,
                "validation_fold_min_pnl_delta": folds["min_pnl_delta"],
                "validation_fold_sum_pnl_delta": folds["sum_pnl_delta"],
                "validation_fold_negative_pnl_delta_count": folds["negative_pnl_delta_count"],
                "validation_fold_max_avg_hold_delta_hours": folds["max_avg_hold_delta_hours"],
                "validation_fold_min_candidate_mdd": folds["min_candidate_mdd"],
                **flatten("validation", val_metrics),
                **flatten("oos", oos_metrics),
            }
            validation_gate = bool(
                float(row["validation_pnl"]) > float(reference["validation_pnl"])
                and float(row["validation_mdd"]) >= -20.0
                and float(row["validation_max_hold_hours"]) <= float(reference["validation_max_hold_hours"]) + 1.0e-9
                and float(row["validation_avg_hold_hours"]) < float(reference["validation_avg_hold_hours"])
                and int(row["validation_overlap_count"]) == 0
                and float(row["validation_max_notional"]) <= 5.0 + 1.0e-9
                and float(row["validation_accounting_error_max_abs"]) <= 1.0e-10
                and float(row["validation_notional_contract_error_max_abs"]) <= 1.0e-10
            )
            oos_gate = bool(
                int(row["oos_vetoed"]) >= MIN_OOS_VETOED
                and float(row["oos_pnl"]) > float(reference["oos_pnl"])
                and float(row["oos_mdd"]) >= float(reference["oos_mdd"])
                and float(row["oos_max_hold_hours"]) <= float(reference["oos_max_hold_hours"]) + 1.0e-9
                and float(row["oos_avg_hold_hours"])
                <= float(reference["oos_avg_hold_hours"]) - MIN_OOS_AVG_HOLD_IMPROVEMENT_HOURS
                and int(row["oos_overlap_count"]) == 0
                and float(row["oos_max_notional"]) <= 5.0 + 1.0e-9
                and float(row["oos_accounting_error_max_abs"]) <= 1.0e-10
                and float(row["oos_notional_contract_error_max_abs"]) <= 1.0e-10
            )
            fold_gate = bool(
                int(row["validation_fold_negative_pnl_delta_count"]) == 0
                and float(row["validation_fold_max_avg_hold_delta_hours"]) <= 1.0e-9
                and float(row["validation_fold_min_candidate_mdd"]) >= -20.0
            )
            row["validation_two_stage_gate_pass"] = validation_gate
            row["oos_productive_safety_gate_pass"] = oos_gate
            row["validation_fold_robust_gate_pass"] = fold_gate
            row["research_two_stage_veto_gate_pass"] = validation_gate and oos_gate and fold_gate
            rows.append(row)
            fold_summaries[spec_name] = folds

    ranking = pd.DataFrame(rows).sort_values(
        [
            "research_two_stage_veto_gate_pass",
            "validation_pnl",
            "validation_fold_min_pnl_delta",
            "validation_avg_hold_hours",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if ranking.empty:
        raise RuntimeError("no two-stage veto variants evaluated")
    selected = ranking.iloc[0].to_dict()
    spec_name = f"{selected['feature_name']}_{selected['feature_op']}_{float(selected['feature_threshold']):.8g}"
    selected_val, _ = base.apply_veto(
        val,
        str(selected["feature_name"]),
        str(selected["feature_op"]),
        float(selected["feature_threshold"]),
        spec_name,
    )
    selected_oos, _ = base.apply_veto(
        oos,
        str(selected["feature_name"]),
        str(selected["feature_op"]),
        float(selected["feature_threshold"]),
        spec_name,
    )
    status = (
        "RESEARCH_ROLL8_SIDE_SPECIFIC_TWO_STAGE_VETO_PASS"
        if bool(selected["research_two_stage_veto_gate_pass"])
        else "NO_ROLL8_SIDE_SPECIFIC_TWO_STAGE_VETO_PASSING_CANDIDATE"
    )
    safe = spec_name.replace(".", "p").replace("/", "_").replace("<=", "le").replace(">=", "ge")
    ranking_path = OUT_DIR / "roll8_side_specific_two_stage_veto_ranking.csv"
    top20_path = OUT_DIR / "roll8_side_specific_two_stage_veto_top20.csv"
    val_out = OUT_DIR / f"validation_{safe}_ledger.csv"
    oos_out = OUT_DIR / f"oos_{safe}_ledger.csv"
    report_path = OUT_DIR / "report.json"
    ranking.to_csv(ranking_path, index=False)
    ranking.head(20).to_csv(top20_path, index=False)
    selected_val.to_csv(val_out, index=False)
    selected_oos.to_csv(oos_out, index=False)
    report = {
        "model_id": MODEL_ID,
        "reference_model_id": REFERENCE_MODEL_ID,
        "parent_model_id": PARENT_MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_scope": "validation_primary_second_stage_short_veto_with_temporal_fold_gate_and_productive_oos_safety_gate; fresh_holdout_required",
        "selection_rule": "search one additional non-lookahead-named numeric short-entry veto; require 4-fold validation robustness and productive OOS safety; among research-gated variants sort by validation_pnl, fold min pnl delta, validation_avg_hold; OOS is not an ordering key",
        "lookahead_exclude_regex": base.LOOKAHEAD_EXCLUDE_RE.pattern,
        "first_stage_reference_feature": first_stage_feature,
        "validation_fold_count": fold_eval.N_VALIDATION_FOLDS,
        "features_evaluated": len(base.candidate_features(val)) - 1,
        "variants_evaluated": int(len(ranking)),
        "reference_variant": reference,
        "selected_variant": selected,
        "selected_fold_summary": fold_summaries[spec_name],
        "top20": ranking.head(20).to_dict(orient="records"),
        "status": status,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(ranking_path),
            "top20": str(top20_path),
            "selected_validation_ledger": str(val_out),
            "selected_oos_ledger": str(oos_out),
            "report": str(report_path),
            "audit_md": str(AUDIT_MD),
        },
    }
    write_json(report_path, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": status,
                "selected_feature": selected["feature_name"],
                "selected_rule": f"{selected['feature_name']} {selected['feature_op']} {float(selected['feature_threshold']):.8g}",
                "selected_validation": {
                    "pnl": selected["validation_pnl"],
                    "mdd": selected["validation_mdd"],
                    "avg_hold_hours": selected["validation_avg_hold_hours"],
                    "max_hold_hours": selected["validation_max_hold_hours"],
                    "vetoed": selected["validation_vetoed"],
                    "fold_min_pnl_delta": selected["validation_fold_min_pnl_delta"],
                    "gate": bool(selected["validation_two_stage_gate_pass"]),
                },
                "selected_oos": {
                    "pnl": selected["oos_pnl"],
                    "mdd": selected["oos_mdd"],
                    "avg_hold_hours": selected["oos_avg_hold_hours"],
                    "max_hold_hours": selected["oos_max_hold_hours"],
                    "vetoed": selected["oos_vetoed"],
                    "gate": bool(selected["oos_productive_safety_gate_pass"]),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
