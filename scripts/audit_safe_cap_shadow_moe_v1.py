#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_2026.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_ledger.csv"
DEFAULT_BASE_LEDGER = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_base_safe_cap_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/safe_cap_shadow_moe_v1_redteam_audit.json"


def _finite_numeric(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return bool(np.isfinite(s.fillna(0.0).to_numpy(dtype=float)).all())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-team audit for safe_cap_shadow_moe_v1.")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--base-ledger", type=Path, default=DEFAULT_BASE_LEDGER)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    ledger = pd.read_csv(args.ledger)
    base = pd.read_csv(args.base_ledger)
    blocking: list[str] = []
    warnings: list[str] = []

    selected = report.get("selected", {})
    oos1 = selected.get("oos_cost1", {})
    oos2 = selected.get("oos_cost2", {})
    oos3 = selected.get("oos_cost3", {})
    data = report.get("data", {})

    if len(ledger) != int(data.get("oos_parent_trades", -1)):
        blocking.append("ledger row count does not match OOS parent trade count")
    if len(base) != len(ledger):
        blocking.append("base and shadow ledger row counts differ")

    required = {
        "trade_id",
        "entry_idx",
        "core_exit_idx",
        "core_side",
        "base_experiment_notional",
        "experiment_notional",
        "cash_before",
        "cash_after",
        "entry_fee_cash",
        "exit_fee_cash",
        "trade_pnl_pct",
        "shadow_action",
        "shadow_support_score",
        "shadow_conflict_score",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        blocking.append(f"ledger missing columns: {missing}")

    for col in ("base_experiment_notional", "experiment_notional", "cash_before", "cash_after", "entry_fee_cash", "exit_fee_cash", "trade_pnl_pct"):
        if not _finite_numeric(ledger, col):
            blocking.append(f"ledger column is missing or non-finite: {col}")

    if len(base) == len(ledger):
        base_n = pd.to_numeric(base.get("experiment_notional", 0.0), errors="coerce").fillna(0.0)
        shadow_n = pd.to_numeric(ledger.get("experiment_notional", 0.0), errors="coerce").fillna(0.0)
        created = (base_n <= 1e-12) & (shadow_n > 1e-12)
        if bool(created.any()):
            blocking.append("shadow overlay created entries blocked by safe-cap parent")
        if "core_side" in base.columns and "core_side" in ledger.columns:
            if not bool((pd.to_numeric(base["core_side"], errors="coerce") == pd.to_numeric(ledger["core_side"], errors="coerce")).all()):
                blocking.append("shadow overlay changed trade side")
        if "core_exit_idx" in base.columns and "core_exit_idx" in ledger.columns:
            if not bool((pd.to_numeric(base["core_exit_idx"], errors="coerce") == pd.to_numeric(ledger["core_exit_idx"], errors="coerce")).all()):
                blocking.append("shadow overlay changed exit index")

    if float(oos1.get("max_margin_fraction", 999.0)) > 1.0 + 1e-12:
        blocking.append("OOS max margin fraction exceeds 1.0")
    if int(oos1.get("liquidations", 1) or 0) > 0:
        blocking.append("OOS liquidation detected")
    if int(oos1.get("ruin_events", 1) or 0) > 0:
        blocking.append("OOS account ruin detected")
    if float(oos2.get("pnl", -1.0)) <= 0.0:
        blocking.append("OOS 2x cost survival failed")
    if float(oos3.get("pnl", -1.0)) <= 0.0:
        blocking.append("OOS 3x cost survival failed")
    if float(oos1.get("max_notional", 999.0)) > 5.0 + 1e-12:
        blocking.append("OOS notional exceeds 5.0 cap")

    fee_cols = ["entry_fee_cash", "exit_fee_cash"]
    for col in fee_cols:
        if col in ledger.columns and bool((pd.to_numeric(ledger[col], errors="coerce").fillna(0.0) < -1e-12).any()):
            blocking.append(f"negative fee column detected: {col}")
    if "cash_after" in ledger.columns and bool((pd.to_numeric(ledger["cash_after"], errors="coerce").fillna(0.0) < -1e-12).any()):
        blocking.append("negative cash_after detected")

    selected_shadow = selected.get("candidate", {}).get("name", "")
    if selected_shadow == "shadow_noop":
        warnings.append("selection fell back to shadow_noop; this is audit-safe but not an alpha improvement")

    audit = {
        "model_id": report.get("model_id"),
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "checked_report": str(args.report),
        "checked_ledger": str(args.ledger),
        "checked_base_ledger": str(args.base_ledger),
        "metrics": {
            "oos_cost1": {k: oos1.get(k) for k in ("pnl", "mdd", "trades", "blocked", "avg_notional", "max_notional", "max_margin_fraction")},
            "oos_cost2": {k: oos2.get(k) for k in ("pnl", "mdd", "trades", "blocked", "avg_notional", "max_notional", "max_margin_fraction")},
            "oos_cost3": {k: oos3.get(k) for k in ("pnl", "mdd", "trades", "blocked", "avg_notional", "max_notional", "max_margin_fraction")},
        },
        "invariants": {
            "no_new_entries": not any("created entries" in b for b in blocking),
            "no_side_changes": not any("changed trade side" in b for b in blocking),
            "no_exit_changes": not any("changed exit index" in b for b in blocking),
            "accounting_columns_present": not missing,
            "cost2_survives": float(oos2.get("pnl", -1.0)) > 0.0,
            "cost3_survives": float(oos3.get("pnl", -1.0)) > 0.0,
        },
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
