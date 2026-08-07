#!/usr/bin/env python3
from __future__ import annotations

import importlib
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_ID = "clean_base_causal_sleeve_conformal_veto_v1_5"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_2026.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_ledger.csv"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/clean_base_causal_sleeve_conformal_veto_v1_5/sleeve_conformal_veto.pkl"
DEFAULT_OUT = ROOT / "data/ensemble/reports/parent_v15_redteam_full_audit_20260511.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _try_import(module: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(module)
        return {"ok": True, "module": module, "model_id": getattr(mod, "MODEL_ID", None)}
    except Exception as exc:
        return {"ok": False, "module": module, "error": repr(exc)}


def _py_compile(paths: list[Path]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "py_compile", *[str(p) for p in paths]]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr[-4000:]}


def _ledger_audit(report: dict[str, Any], ledger: pd.DataFrame) -> dict[str, Any]:
    numeric = ledger.select_dtypes(include=[np.number])
    nonfinite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()) if not numeric.empty else 0
    required = {
        "trade_id",
        "entry_idx",
        "core_exit_idx",
        "timestamp",
        "action",
        "core_side",
        "core_notional",
        "gross_notional",
        "net_notional",
        "leverage",
        "total_fee_cash",
        "trade_pnl_pct",
        "cash_before",
        "cash_after",
    }
    missing = sorted(required - set(ledger.columns))
    transition_error = None
    if not missing:
        expected_after = pd.to_numeric(ledger["cash_before"], errors="coerce") * (
            1.0 + pd.to_numeric(ledger["trade_pnl_pct"], errors="coerce") / 100.0
        )
        actual_after = pd.to_numeric(ledger["cash_after"], errors="coerce")
        transition_error = float(np.nanmax(np.abs(expected_after.to_numpy(dtype=float) - actual_after.to_numpy(dtype=float))))
    fee_error = None
    fee_cols = ["core_entry_fee_cash", "core_exit_fee_cash", "sleeve_entry_fee_cash", "sleeve_exit_fee_cash"]
    if all(c in ledger.columns for c in fee_cols + ["total_fee_cash"]):
        fee_sum = sum(pd.to_numeric(ledger[c], errors="coerce").fillna(0.0) for c in fee_cols)
        fee_error = float(np.nanmax(np.abs(fee_sum.to_numpy(dtype=float) - pd.to_numeric(ledger["total_fee_cash"], errors="coerce").to_numpy(dtype=float))))
    final_pnl = None
    report_pnl = float(report["cost_1x"]["pnl"])
    if "cash_after" in ledger.columns and len(ledger):
        final_pnl = float((float(ledger.iloc[-1]["cash_after"]) - 1.0) * 100.0)
    trade_count_match = int(report["cost_1x"]["trades"]) == int(len(ledger))
    ts_ok = True
    if "timestamp" in ledger.columns:
        ts = pd.to_datetime(ledger["timestamp"], errors="coerce")
        ts_ok = bool(ts.notna().all() and ts.is_monotonic_increasing)
    idx_ok = True
    if {"entry_idx", "core_exit_idx"}.issubset(ledger.columns):
        entry = pd.to_numeric(ledger["entry_idx"], errors="coerce")
        exit_idx = pd.to_numeric(ledger["core_exit_idx"], errors="coerce")
        idx_ok = bool((exit_idx >= entry).all())
    passed = (
        not missing
        and nonfinite == 0
        and transition_error is not None
        and transition_error <= 1e-9
        and fee_error is not None
        and fee_error <= 1e-9
        and final_pnl is not None
        and abs(final_pnl - report_pnl) <= 1e-9
        and trade_count_match
        and ts_ok
        and idx_ok
    )
    return {
        "passed": bool(passed),
        "rows": int(len(ledger)),
        "missing_columns": missing,
        "nonfinite_numeric_values": nonfinite,
        "cash_transition_max_abs_error": transition_error,
        "fee_sum_max_abs_error": fee_error,
        "report_pnl": report_pnl,
        "final_pnl_from_ledger": final_pnl,
        "trade_count_match": trade_count_match,
        "timestamp_monotonic": ts_ok,
        "exit_idx_not_before_entry_idx": idx_ok,
    }


def _artifact_audit(report: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    sleeve_features = list(artifact.get("sleeve_features") or [])
    conformal_features = list(artifact.get("conformal_features") or [])
    all_features = sleeve_features + conformal_features
    forbidden = [
        c
        for c in all_features
        if "future" in c.lower()
        or "target" in c.lower()
        or "label" in c.lower()
        or "realized" in c.lower()
        or "cash_after" in c.lower()
        or ("regime" in c.lower() and not c.startswith("clean_regime_2024_unsup_v4_"))
        or "hdb" in c.lower()
        or c.lower().startswith("hmm_")
    ]
    selected_match = artifact.get("selected_config") == report.get("selected_config")
    residual_match = math.isclose(float(artifact.get("selected_residual_q")), float(report.get("selected_residual_q")), rel_tol=0.0, abs_tol=1e-15)
    return {
        "passed": bool(artifact.get("model_id") == MODEL_ID and selected_match and residual_match and not forbidden),
        "model_id": artifact.get("model_id"),
        "selected_config_match_report": selected_match,
        "selected_residual_q_match_report": residual_match,
        "sleeve_features": sleeve_features,
        "conformal_features": conformal_features,
        "forbidden_runtime_features": forbidden,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exists = {str(p): p.exists() for p in (args.report, args.ledger, args.model)}
    blocking: list[str] = []
    warnings: list[str] = []
    if not all(exists.values()):
        blocking.append("missing_required_v15_artifact")
    report = _load_json(args.report)
    ledger = pd.read_csv(args.ledger)
    artifact = joblib.load(args.model)
    source_files = [
        ROOT / "scripts/train_eval_clean_base_causal_sleeve_conformal_veto_v1_5.py",
        ROOT / "scripts/train_eval_clean_base_plus_causal_conviction_sleeve_v1_1.py",
        ROOT / "scripts/train_eval_clean_base_causal_trade_editor_v1_3.py",
        ROOT / "scripts/train_eval_clean_base_lifecycle_editor_v1.py",
        ROOT / "scripts/train_eval_clean_base_exit_hazard_recalibrator_v1.py",
        ROOT / "scripts/train_eval_lifecycle_v1_drawdown_governor_v1.py",
    ]
    compile_audit = _py_compile(source_files)
    import_audit = _try_import("scripts.train_eval_clean_base_causal_sleeve_conformal_veto_v1_5")
    ledger_check = _ledger_audit(report, ledger)
    artifact_check = _artifact_audit(report, artifact)
    source_inputs = {
        "default_policy_exists": (ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl").exists(),
        "default_exit_exists": (ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl").exists(),
        "train_csv_exists": (ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv").exists(),
        "eval_csv_exists": (ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv").exists(),
    }
    if not compile_audit["ok"]:
        blocking.append("source_py_compile_failed")
    if not import_audit["ok"]:
        blocking.append("source_import_failed")
    if not ledger_check["passed"]:
        blocking.append("ledger_accounting_or_integrity_failed")
    if not artifact_check["passed"]:
        blocking.append("artifact_report_or_feature_contract_failed")
    if not source_inputs["default_policy_exists"] or not source_inputs["default_exit_exists"]:
        blocking.append("source_reproduction_missing_parent_policy_or_exit_artifact")
    if not source_inputs["train_csv_exists"] or not source_inputs["eval_csv_exists"]:
        blocking.append("source_reproduction_missing_train_or_eval_csv")
    if float(report["cost_3x"]["pnl"]) <= 0.0:
        warnings.append("cost3_stress_negative")
    if report.get("verdict") != "promote":
        warnings.append(f"report_verdict_is_{report.get('verdict')}")
    checks = {
        "exists": exists,
        "source_compile": compile_audit,
        "source_import": import_audit,
        "source_inputs": source_inputs,
        "artifact": artifact_check,
        "ledger": ledger_check,
        "report_metrics": {
            "verdict": report.get("verdict"),
            "cost_1x": report.get("cost_1x"),
            "cost_2x": report.get("cost_2x"),
            "cost_3x": report.get("cost_3x"),
            "promotion_gate": report.get("promotion_gate"),
            "preservation_audit_passed": bool(report.get("preservation_audit", {}).get("passed")),
            "accounting_audit_passed": bool(report.get("sleeve_accounting_audit", {}).get("passed")),
            "causality_audit_passed": bool(report.get("causality_audit", {}).get("passed")),
        },
    }
    out = {
        "model_id": MODEL_ID,
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "checks": checks,
        "redteam_verdict": (
            "Existing artifact/ledger is internally consistent, but full source reproduction is blocked by missing parent policy/exit artifacts."
            if blocking == ["source_reproduction_missing_parent_policy_or_exit_artifact"]
            else "See blocking list."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"status": out["status"], "blocking": blocking, "warnings": warnings, "audit": str(args.out)}, indent=2, ensure_ascii=False))
    return 0 if not blocking else 2


if __name__ == "__main__":
    raise SystemExit(main())
