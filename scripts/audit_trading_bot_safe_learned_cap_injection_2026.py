#!/usr/bin/env python3
from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRADING_BOT = ROOT / "trading_bot.py"
MODEL_PATH = ROOT / "data/ensemble/supervised/clean_base_deep_gated_gross_v2_safe_cap_buckets/deep_gated_gross_safe_cap_buckets.pkl"
REPORT_PATH = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_buckets_2026.json"
AUDIT_PATH = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_buckets_audit.json"
LEDGER_PATH = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_safe_cap_buckets_ledger.csv"
OUT_PATH = ROOT / "data/ensemble/reports/trading_bot_safe_learned_cap_injection_audit_2026.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return data


def _metric(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        x = float(obj.get(key, default))
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _bin(value: float, cuts: list[float]) -> int:
    return int(np.searchsorted(np.asarray(cuts, dtype=np.float64), float(value), side="right"))


def _cost_pass(edge: float, notional: float, *, fee: float, slip: float, buffer: float) -> tuple[bool, float, float]:
    expected = float(edge * max(float(notional), 0.0))
    hurdle = float(2.0 * (float(fee) + float(slip)) * max(float(notional), 0.0) + float(buffer))
    return bool(expected > hurdle), expected, hurdle


def _audit() -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    for path in (TRADING_BOT, MODEL_PATH, REPORT_PATH, AUDIT_PATH, LEDGER_PATH):
        if not path.exists():
            blocking.append(f"missing required path: {path}")
    if blocking:
        return {"status": "fail", "blocking": blocking, "warnings": warnings, "checks": checks}

    try:
        py_compile.compile(str(TRADING_BOT), doraise=True)
        for mod in (ROOT / "trading_bot_modules").glob("*.py"):
            py_compile.compile(str(mod), doraise=True)
        checks["py_compile"] = True
    except Exception as e:
        blocking.append(f"py_compile failed: {e}")
        checks["py_compile"] = False

    source = TRADING_BOT.read_text(encoding="utf-8")
    source_checks = {
        "default_model_points_to_safe_cap": "deep_gated_gross_safe_cap_buckets.pkl" in source,
        "default_report_points_to_safe_cap": "clean_base_deep_gated_gross_v2_safe_cap_buckets_2026.json" in source,
        "safe_cap_layer_present": "_lifecycle_v1_apply_safe_learned_cap" in source,
        "safe_cap_enabled_default_true": re.search(r"FINAL_GOVERNOR_SAFE_LEARNED_CAP_ENABLE.*True", source) is not None,
        "dsac_overlay_default_false": re.search(r"FINAL_GOVERNOR_DSAC_OVERLAY_ENABLE.*False", source) is not None,
        "risk_cap_override_present": "max_notional_override=safe_cap_risk_cap" in source,
        "safe_cap_block_source_present": "lifecycle_v1|safe_learned_cap_block" in source,
        "position_accounting_audit_enabled": "POSITION_ACCOUNTING_AUDIT_PATH" in source,
    }
    checks["source"] = source_checks
    for name, ok in source_checks.items():
        if not ok:
            blocking.append(f"source check failed: {name}")

    model_audit = _load_json(AUDIT_PATH)
    checks["model_audit"] = model_audit
    if str(model_audit.get("status", "")).lower() != "pass":
        blocking.append(f"safe cap model audit is not pass: {model_audit.get('blocking', [])}")

    report = _load_json(REPORT_PATH)
    selected = dict(report.get("selected", {}) or {})
    candidate = dict(selected.get("candidate", {}) or {})
    checks["report_model_id"] = report.get("model_id")
    checks["report_verdict"] = report.get("verdict")
    checks["selected_candidate"] = {
        k: candidate.get(k)
        for k in ("name", "scheme", "cost_buffer", "gate_notional_mode", "fallback_cap", "max_notional")
    }
    if report.get("model_id") != "clean_base_deep_gated_gross_v2_safe_cap_buckets":
        blocking.append("report model_id is not safe cap buckets")
    if report.get("verdict") != "promote_candidate":
        blocking.append("safe cap report verdict is not promote_candidate")
    if "governor" in selected:
        blocking.append("selected report unexpectedly contains an MDD/governor layer")
    if candidate.get("name") != "learned_action_edge3_min10_buf0p0035_gatefinal":
        warnings.append(f"selected candidate differs from expected no-MDD safe cap: {candidate.get('name')}")
    if float(candidate.get("max_notional", 0.0) or 0.0) > 5.0:
        blocking.append("candidate max_notional exceeds 5.0")
    if str(candidate.get("gate_notional_mode", "")).lower() != "final":
        blocking.append("candidate cost gate is not final-notional based")

    baseline = dict(report.get("baseline_static_cost_firewall", {}) or {})
    for scope in ("validation_cost1", "validation_cost2", "validation_cost3", "oos_cost1", "oos_cost2", "oos_cost3"):
        metrics = dict(selected.get(scope, {}) or {})
        checks[scope] = {
            k: metrics.get(k)
            for k in ("pnl", "mdd", "trades", "blocked", "boosted", "avg_notional", "max_notional", "max_margin_fraction", "liquidations", "ruin_events")
        }
        if _metric(metrics, "liquidations") != 0.0 or _metric(metrics, "ruin_events") != 0.0:
            blocking.append(f"{scope} liquidation/ruin invariant failed")
        if _metric(metrics, "max_margin_fraction") > 1.0 + 1e-12:
            blocking.append(f"{scope} max_margin_fraction exceeds 1.0")
        if scope.endswith("cost2") or scope.endswith("cost3"):
            if _metric(metrics, "pnl") <= 0.0:
                blocking.append(f"{scope} cost stress does not survive")
    if _metric(dict(selected.get("oos_cost1", {}) or {}), "pnl") <= _metric(dict(baseline.get("oos_cost1", {}) or {}), "pnl"):
        blocking.append("OOS PnL is not above static cost-firewall baseline")

    payload = joblib.load(MODEL_PATH)
    payload_candidate = dict(payload.get("selected_cap_candidate", {}) or {})
    checks["payload_keys"] = sorted(payload.keys())
    checks["payload_candidate_name"] = payload_candidate.get("name")
    if payload_candidate.get("name") != candidate.get("name"):
        blocking.append("payload selected_cap_candidate does not match report candidate")
    torch_model = Path(str(payload.get("torch_model", "")))
    if not torch_model.exists():
        blocking.append(f"payload torch_model missing: {torch_model}")
    for key in ("sequence_features", "sequence_scaler", "state_model", "head_model", "selected_parent_config", "selected_cap_candidate"):
        if key not in payload:
            blocking.append(f"payload missing key: {key}")

    thresholds = dict(candidate.get("thresholds", {}) or {})
    cap_map = dict(candidate.get("cap_map", {}) or {})
    high_edge = float(list(thresholds.get("edge3", [0.0, 0.0]))[-1]) + 0.001
    high_bucket = f"HIGH|e{_bin(high_edge, list(thresholds.get('edge3', [0.0, 0.0])))}"
    high_cap = float(cap_map.get(high_bucket, candidate.get("fallback_cap", 3.6)))
    planned_high = float(np.clip(3.6 * high_cap / 3.6, 0.0, high_cap))
    high_pass, high_expected, high_hurdle = _cost_pass(
        high_edge,
        planned_high,
        fee=0.0005,
        slip=0.0002,
        buffer=float(candidate.get("cost_buffer", 0.0035) or 0.0035),
    )
    low_bucket = "HIGH|e0"
    low_cap = float(cap_map.get(low_bucket, candidate.get("fallback_cap", 3.6)))
    planned_low = float(np.clip(3.6 * low_cap / 3.6, 0.0, low_cap))
    low_pass, low_expected, low_hurdle = _cost_pass(
        0.0,
        planned_low,
        fee=0.0005,
        slip=0.0002,
        buffer=float(candidate.get("cost_buffer", 0.0035) or 0.0035),
    )
    checks["live_formula_smoke"] = {
        "high_bucket": high_bucket,
        "high_cap": high_cap,
        "planned_high": planned_high,
        "high_cost_pass": high_pass,
        "high_expected": high_expected,
        "high_hurdle": high_hurdle,
        "low_bucket": low_bucket,
        "low_cap": low_cap,
        "planned_low": planned_low,
        "low_cost_pass": low_pass,
        "low_expected": low_expected,
        "low_hurdle": low_hurdle,
    }
    if high_cap != 5.0 or planned_high != 5.0 or not high_pass:
        blocking.append("live formula smoke failed to boost HIGH e2 to 5.0")
    if low_pass:
        blocking.append("live formula smoke failed to block zero-edge trade")

    ledger = pd.read_csv(LEDGER_PATH)
    checks["ledger"] = {
        "rows": int(len(ledger)),
        "blocked": int(pd.to_numeric(ledger.get("blocked", False), errors="coerce").fillna(0).astype(bool).sum()),
        "max_experiment_notional": float(pd.to_numeric(ledger.get("experiment_notional", 0.0), errors="coerce").fillna(0.0).max()),
        "max_margin_fraction": float(pd.to_numeric(ledger.get("margin_fraction", 0.0), errors="coerce").fillna(0.0).max()),
        "liquidated_count": int(pd.to_numeric(ledger.get("liquidated", False), errors="coerce").fillna(0).astype(bool).sum()),
    }
    if checks["ledger"]["max_experiment_notional"] > 5.0 + 1e-12:
        blocking.append("ledger experiment notional exceeds 5.0")
    if checks["ledger"]["max_margin_fraction"] > 1.0 + 1e-12:
        blocking.append("ledger margin fraction exceeds 1.0")
    if checks["ledger"]["liquidated_count"] != 0:
        blocking.append("ledger contains liquidations")

    return {
        "status": "pass" if not blocking else "fail",
        "model": "safe learned cap, no MDD governor",
        "blocking": blocking,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> int:
    report = _audit()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "blocking": report["blocking"], "warnings": report["warnings"], "out": str(OUT_PATH)}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
