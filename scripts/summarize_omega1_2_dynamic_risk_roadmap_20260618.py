#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
OUT_DIR = BASE / "omega1_2_dynamic_risk_roadmap_summary_20260618"


REPORTS = [
    {
        "stage": "baseline",
        "technique": "Omega 1.2.8b fixed/current live-like baseline",
        "status": "executed",
        "path": BASE / "omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616/report.json",
        "selector": "selected_by_validation",
    },
    {
        "stage": "control",
        "technique": "notional=1, leverage=2 fixed retrain",
        "status": "executed",
        "path": BASE / "omega1_2_8v_notional1_leverage2_retrain_20260617/report.json",
        "selector": "selected_by_validation",
    },
    {
        "stage": "level1",
        "technique": "ATR SL/TP + vol-scaled notional + inverse-vol leverage",
        "status": "executed_new",
        "path": BASE / "omega1_2_8w_report_level1_atr_dynamic_risk_20260618/report.json",
        "selector": "selected_by_validation",
    },
    {
        "stage": "level1",
        "technique": "Kelly notional",
        "status": "not_executed_missing_trade_distribution_contract",
        "path": None,
        "selector": None,
        "note": "Report suggests Kelly from historical win/loss stats, but this active path lacks a locked trade-distribution contract for Kelly sizing. Not added as fallback because project rules require fail-fast contract changes.",
    },
    {
        "stage": "level2",
        "technique": "MFE/MAE-like dynamic TP/SL/notional/leverage regressors",
        "status": "executed_existing",
        "path": BASE / "omega1_2_8b_full_retrain_numeric_cash_sleeve_hf7head_20260617_dynamic_risk/report.json",
        "selector": "selected_by_validation",
    },
    {
        "stage": "level2",
        "technique": "continuous risk heads",
        "status": "executed_existing",
        "path": BASE / "omega1_2_learned_continuous_risk_20260605_extratrees_delta_anchor_blend060_edge0035_r5000_c32_e800_seed260634/report.json",
        "selector": "results",
    },
    {
        "stage": "level2",
        "technique": "survival/hazard proxy through lifecycle exit controller",
        "status": "executed_proxy",
        "path": BASE / "omega1_2_mamba_sac_lifecycle_controller_20260604_dsac_risk_selector_retest_q075_seed260621/report.json",
        "selector": "results",
        "note": "lifelines is not installed, so CoxPH was not run. This row uses the existing lifecycle exit-controller backtest as the closest available hold-time hazard proxy.",
    },
    {
        "stage": "level2",
        "technique": "contextual bandit/profile risk selector",
        "status": "executed_existing",
        "path": BASE / "omega1_2_risk_scale_selector_20260605_hgb_profiles_edge0035_r6000_e800_seed260640/report.json",
        "selector": "results",
    },
    {
        "stage": "level2",
        "technique": "factorized bucket risk selector",
        "status": "executed_existing",
        "path": BASE / "omega1_2_bucket_risk_selector_position_20260605_hgb_posbucket_r2500_c48_s260650/report.json",
        "selector": "results",
    },
    {
        "stage": "level3",
        "technique": "factored autoregressive soft-bucket risk policy",
        "status": "executed_existing",
        "path": BASE / "omega1_2_autoreg_softbucket_risk_20260606_failfast_cvar_rescale_cap080_thr014_s260802/report.json",
        "selector": "results",
    },
    {
        "stage": "level3",
        "technique": "DSAC discrete risk-template selector",
        "status": "executed_existing",
        "path": BASE / "omega1_2_lifecycle_with_dsac_risk_selector_20260605_full_q075_s600_e800_noveto_actor_seed260623/report.json",
        "selector": "results",
    },
    {
        "stage": "level3",
        "technique": "Omega2.1 DSAC overlay",
        "status": "executed_existing",
        "path": BASE / "omega2_1_dsac_overlay_20260609/report.json",
        "selector": "top_first_non_baseline",
    },
]


def _clean(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _reason(row: dict[str, Any], prefix: str, key: str) -> int | None:
    reasons = row.get(f"{prefix}_reasons")
    if isinstance(reasons, dict):
        value = reasons.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _metrics_from_selected(data: dict[str, Any], key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = data.get(key)
    if not isinstance(selected, dict):
        raise KeyError(key)
    return selected, selected


def _metrics_from_results(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results = data.get("results")
    if not isinstance(results, dict):
        raise KeyError("results")
    val = results.get("validation")
    oos = results.get("oos")
    if not isinstance(val, dict) or not isinstance(oos, dict):
        raise KeyError("results.validation/oos")
    selected = {
        "candidate": data.get("model_id", "results"),
        "val_pnl": val.get("pnl"),
        "val_mdd": val.get("mdd"),
        "val_wr": val.get("wr"),
        "val_trades": val.get("trades"),
        "val_reasons": val.get("reasons"),
        "oos_pnl": oos.get("pnl"),
        "oos_mdd": oos.get("mdd"),
        "oos_wr": oos.get("wr"),
        "oos_trades": oos.get("trades"),
        "oos_reasons": oos.get("reasons"),
    }
    return selected, selected


def _metrics_from_top0(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    top = data.get("top")
    if not isinstance(top, list) or not top or not isinstance(top[0], dict):
        raise KeyError("top[0]")
    return top[0], top[0]


def _metrics_from_top_first_non_baseline(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    top = data.get("top")
    if not isinstance(top, list):
        raise KeyError("top")
    for row in top:
        if not isinstance(row, dict):
            continue
        if row.get("source") != "baseline":
            return row, row
    raise KeyError("top non-baseline")


def _extract(spec: dict[str, Any]) -> dict[str, Any]:
    path = spec.get("path")
    base_row = {
        "stage": spec["stage"],
        "technique": spec["technique"],
        "status": spec["status"],
        "path": str(path) if path else "",
        "note": spec.get("note", ""),
    }
    if path is None:
        return base_row
    if not Path(path).exists():
        return {**base_row, "status": "missing_report", "note": "report.json not found"}

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    selector = spec.get("selector")
    if selector == "results":
        selected, raw = _metrics_from_results(data)
    elif selector == "top0":
        selected, raw = _metrics_from_top0(data)
    elif selector == "top_first_non_baseline":
        selected, raw = _metrics_from_top_first_non_baseline(data)
    elif isinstance(selector, str):
        selected, raw = _metrics_from_selected(data, selector)
    else:
        raise ValueError(f"Unknown selector: {selector!r}")

    val_pnl = _num(selected, "val_pnl")
    oos_pnl = _num(selected, "oos_pnl")
    val_mdd = _num(selected, "val_mdd")
    oos_mdd = _num(selected, "oos_mdd")
    val_score = None
    if val_pnl is not None and val_mdd is not None:
        val_score = val_pnl + 10.0 * val_mdd
    oos_score = None
    if oos_pnl is not None and oos_mdd is not None:
        oos_score = oos_pnl + 10.0 * oos_mdd

    return {
        **base_row,
        "model_id": data.get("model_id", ""),
        "candidate": selected.get("candidate", ""),
        "val_pnl": val_pnl,
        "val_mdd": val_mdd,
        "val_wr": _num(selected, "val_wr"),
        "val_trades": _num(selected, "val_trades"),
        "oos_pnl": oos_pnl,
        "oos_mdd": oos_mdd,
        "oos_wr": _num(selected, "oos_wr"),
        "oos_trades": _num(selected, "oos_trades"),
        "val_score_pnl_plus_10mdd": val_score,
        "oos_score_pnl_plus_10mdd": oos_score,
        "val_stop_loss": _reason(selected, "val", "stop_loss") or _reason(selected, "val", "primary_stop_loss"),
        "val_take_profit": _reason(selected, "val", "take_profit") or _reason(selected, "val", "primary_take_profit"),
        "val_max_hold": _reason(selected, "val", "max_hold") or _reason(selected, "val", "fallback_max_hold"),
        "oos_stop_loss": _reason(selected, "oos", "stop_loss") or _reason(selected, "oos", "primary_stop_loss"),
        "oos_take_profit": _reason(selected, "oos", "take_profit") or _reason(selected, "oos", "primary_take_profit"),
        "oos_max_hold": _reason(selected, "oos", "max_hold") or _reason(selected, "oos", "fallback_max_hold"),
        "raw_selected": _clean(raw),
    }


def _promotion(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    if row.get("status", "").startswith("not_executed") or row.get("status") == "missing_report":
        return "no_test"
    val_pnl = _num(row, "val_pnl")
    oos_pnl = _num(row, "oos_pnl")
    oos_mdd = _num(row, "oos_mdd")
    base_oos_pnl = _num(baseline, "oos_pnl")
    base_oos_mdd = _num(baseline, "oos_mdd")
    if None in (val_pnl, oos_pnl, oos_mdd, base_oos_pnl, base_oos_mdd):
        return "insufficient_metrics"
    if val_pnl <= 0:
        return "reject_val_loss"
    if oos_pnl >= base_oos_pnl and oos_mdd >= base_oos_mdd:
        return "candidate_promotable"
    if oos_pnl >= base_oos_pnl:
        return "watch_mdd_regression"
    if oos_mdd >= base_oos_mdd:
        return "watch_pnl_regression"
    return "reject_oos_regression"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_extract(spec) for spec in REPORTS]
    baseline = next(row for row in rows if row["stage"] == "baseline")
    for row in rows:
        row["decision"] = "baseline" if row is baseline else _promotion(row, baseline)

    fieldnames = [
        "stage",
        "technique",
        "status",
        "decision",
        "model_id",
        "candidate",
        "val_pnl",
        "val_mdd",
        "val_wr",
        "val_trades",
        "oos_pnl",
        "oos_mdd",
        "oos_wr",
        "oos_trades",
        "val_stop_loss",
        "val_take_profit",
        "val_max_hold",
        "oos_stop_loss",
        "oos_take_profit",
        "oos_max_hold",
        "path",
        "note",
    ]
    csv_path = OUT_DIR / "roadmap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "status": "roadmap_stage_summary_complete",
        "selection_policy": "Use validation-selected candidates where available; OOS remains diagnostic. Rows with prior result-only reports are not reselected by OOS.",
        "baseline": baseline,
        "rows": rows,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "csv": str(csv_path),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(_clean(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print("stage,technique,status,decision,val_pnl,val_mdd,oos_pnl,oos_mdd,oos_wr,oos_trades")
    for row in rows:
        print(
            ",".join(
                str(row.get(key, ""))
                for key in [
                    "stage",
                    "technique",
                    "status",
                    "decision",
                    "val_pnl",
                    "val_mdd",
                    "oos_pnl",
                    "oos_mdd",
                    "oos_wr",
                    "oos_trades",
                ]
            )
        )
    print(json.dumps(summary["artifacts"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
