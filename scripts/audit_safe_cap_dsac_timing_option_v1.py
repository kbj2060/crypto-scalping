#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_2026.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_ledger.csv"
DEFAULT_BASE_LEDGER = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_base_ledger.csv"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/safe_cap_dsac_timing_option_v1_redteam_audit.json"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


def _finite(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce")
    return bool(np.isfinite(s.fillna(0.0).to_numpy(dtype=float)).all())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Red-team audit for safe_cap_dsac_timing_option_v1.")
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
    selected = report.get("selected", {})
    oos1 = selected.get("oos_cost1", {})
    oos2 = selected.get("oos_cost2", {})
    oos3 = selected.get("oos_cost3", {})
    data = report.get("data", {})
    blocking: list[str] = []
    warnings: list[str] = []

    if len(ledger) != int(data.get("oos_parent_trades", -1)):
        blocking.append("ledger row count does not match OOS parent trades")
    if len(base) != len(ledger):
        blocking.append("base and selected ledger row counts differ")

    required = {
        "trade_id",
        "original_entry_idx",
        "entry_idx",
        "original_exit_idx",
        "exit_idx",
        "core_side",
        "base_experiment_notional",
        "experiment_notional",
        "cash_before",
        "cash_after",
        "entry_fee_cash",
        "exit_fee_cash",
        "trade_pnl_pct",
        "timing_action",
        "blocked",
    }
    missing = sorted(required - set(ledger.columns))
    if missing:
        blocking.append(f"ledger missing required columns: {missing}")
    for col in ("base_experiment_notional", "experiment_notional", "cash_before", "cash_after", "entry_fee_cash", "exit_fee_cash", "trade_pnl_pct"):
        if not _finite(ledger, col):
            blocking.append(f"ledger column missing or non-finite: {col}")

    if len(base) == len(ledger):
        base_n = pd.to_numeric(base.get("experiment_notional", 0.0), errors="coerce").fillna(0.0)
        selected_n = pd.to_numeric(ledger.get("experiment_notional", 0.0), errors="coerce").fillna(0.0)
        safe_base_n = pd.to_numeric(ledger.get("base_experiment_notional", 0.0), errors="coerce").fillna(0.0)
        if bool(((base_n <= 1e-12) & (selected_n > 1e-12)).any()):
            blocking.append("timing layer created trades blocked by safe-cap parent")
        if bool((selected_n > safe_base_n + 1e-12).any()):
            blocking.append("timing layer increased notional above safe base")
        if "core_side" in base.columns and "core_side" in ledger.columns:
            if not bool((pd.to_numeric(base["core_side"], errors="coerce") == pd.to_numeric(ledger["core_side"], errors="coerce")).all()):
                blocking.append("timing layer changed side")
        if "exit_idx" in base.columns and "exit_idx" in ledger.columns:
            if not bool((pd.to_numeric(base["exit_idx"], errors="coerce") == pd.to_numeric(ledger["exit_idx"], errors="coerce")).all()):
                blocking.append("timing layer changed exit index")

    if "entry_idx" in ledger.columns and "original_entry_idx" in ledger.columns:
        if bool((pd.to_numeric(ledger["entry_idx"], errors="coerce") < pd.to_numeric(ledger["original_entry_idx"], errors="coerce")).any()):
            blocking.append("timing layer entered before original entry")
    if {"entry_idx", "exit_idx", "blocked"}.issubset(ledger.columns):
        active = ~ledger["blocked"].astype(bool)
        if bool((pd.to_numeric(ledger.loc[active, "entry_idx"], errors="coerce") >= pd.to_numeric(ledger.loc[active, "exit_idx"], errors="coerce")).any()):
            blocking.append("active selected trade has entry_idx >= exit_idx")

    if float(oos1.get("max_margin_fraction", 999.0)) > 1.0 + 1e-12:
        blocking.append("OOS max margin fraction exceeds 1")
    if float(oos1.get("max_notional", 999.0)) > 5.0 + 1e-12:
        blocking.append("OOS notional exceeds 5")
    if int(oos1.get("liquidations", 1) or 0) > 0:
        blocking.append("OOS liquidation detected")
    if int(oos1.get("ruin_events", 1) or 0) > 0:
        blocking.append("OOS account ruin detected")
    if float(oos2.get("pnl", -1.0)) <= 0.0:
        blocking.append("OOS 2x cost survival failed")
    if float(oos3.get("pnl", -1.0)) <= 0.0:
        blocking.append("OOS 3x cost survival failed")
    for col in ("entry_fee_cash", "exit_fee_cash", "liquidation_fee_cash"):
        if col in ledger.columns and bool((pd.to_numeric(ledger[col], errors="coerce").fillna(0.0) < -1e-12).any()):
            blocking.append(f"negative fee detected: {col}")
    if selected.get("candidate", {}).get("name") == "noop_safe_cap_replay":
        warnings.append("selection fell back to noop; audit-safe but no DSAC timing alpha promoted")

    audit = {
        "model_id": report.get("model_id"),
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "checked_report": str(args.report),
        "checked_ledger": str(args.ledger),
        "checked_base_ledger": str(args.base_ledger),
        "metrics": {
            "oos_cost1": {k: oos1.get(k) for k in ("pnl", "mdd", "trades", "blocked", "skipped", "reduced", "delayed", "avg_notional", "max_notional", "max_margin_fraction")},
            "oos_cost2": {k: oos2.get(k) for k in ("pnl", "mdd", "trades", "blocked", "skipped", "reduced", "delayed", "avg_notional", "max_notional", "max_margin_fraction")},
            "oos_cost3": {k: oos3.get(k) for k in ("pnl", "mdd", "trades", "blocked", "skipped", "reduced", "delayed", "avg_notional", "max_notional", "max_margin_fraction")},
        },
        "invariants": {
            "no_new_entries": not any("created trades" in b for b in blocking),
            "side_never_changes": not any("changed side" in b for b in blocking),
            "exit_index_unchanged": not any("changed exit" in b for b in blocking),
            "entry_never_before_original": not any("before original" in b for b in blocking),
            "notional_never_above_safe_base": not any("increased notional" in b for b in blocking),
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
