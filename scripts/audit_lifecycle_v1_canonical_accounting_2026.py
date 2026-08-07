#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.train_eval_clean_base_lifecycle_editor_v1 import (  # noqa: E402
    BASE_REFERENCE,
    LifecycleRuntimeConfig,
    _base_frame,
    _base_trade_plan,
    _compact,
    _days,
    _fill_price,
    _range,
    _read,
    _sha256,
    _split_train_validation,
    backtest_lifecycle_editor,
)
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_LIFECYCLE_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"
DEFAULT_LIFECYCLE_MODEL = ROOT / "data/ensemble/supervised/clean_base_lifecycle_editor_v1/lifecycle_editor.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026.json"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026_ledger.csv"
DEFAULT_FIXED_COST = ROOT / "data/ensemble/reports/lifecycle_v1_canonical_accounting_audit_2026_fixed_ledger_cost.csv"
DEFAULT_DOC = ROOT / "docs/experiments/lifecycle_v1_canonical_accounting_audit_2026.md"


def _load_lifecycle_model(model_path: Path, report_path: Path) -> tuple[dict[str, Any], LifecycleRuntimeConfig, dict[str, Any]]:
    payload = joblib.load(model_path)
    recalibrator = dict(payload["recalibrator"])
    cfg = LifecycleRuntimeConfig(**dict(payload["selected_runtime_config"]))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        cfg = _load_lifecycle_cfg(report)
    except Exception:
        pass
    return recalibrator, cfg, report


def _raw(side: int, entry_price: float, exit_price: float) -> float:
    if side > 0:
        return float((exit_price - entry_price) / max(entry_price, 1e-12))
    return float((entry_price - exit_price) / max(entry_price, 1e-12))


def _mark_exit_price(close: np.ndarray, idx: int, side: int, slip: float) -> float:
    px = float(close[int(np.clip(idx, 0, len(close) - 1))])
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _slip_paid_estimate(cash_before: float, cash_after_entry_fee: float, notional: float, slip: float) -> float:
    return float((cash_before + cash_after_entry_fee) * float(notional) * abs(float(slip)))


def _action(row: dict[str, Any]) -> str:
    edit = str(row.get("edit", "noop")).upper()
    exit_reason = str(row.get("exit_reason", "base_exit")).upper()
    return f"{edit}__{exit_reason}"


def _simulate_fixed_plan(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    base_trades: list[dict[str, Any]],
    lifecycle_plan: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    base_by_entry = {int(t["entry_idx"]): t for t in base_trades}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    prior_trade_giveback = 0.0
    action_distribution: dict[str, int] = {}
    edit_counts: dict[str, int] = {}
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []

    for trade_id, life in enumerate(lifecycle_plan):
        entry_idx = int(life["entry_idx"])
        base = base_by_entry[entry_idx]
        side = int(life["side"])
        base_side = int(base["side"])
        candidate_side = side
        base_exit_idx = int(base["exit_idx"])
        candidate_exit_idx = int(life["effective_exit_idx"])
        effective_notional = float(life["effective_notional"])
        candidate_notional = effective_notional
        base_notional = float(base["base_notional"])
        base_leverage = float(base["leverage"])
        candidate_leverage = float(life["leverage"])
        action = _action(life)

        cash_before = cash
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        entry_fill_idx = min(entry_idx + 1, len(df) - 1)
        exit_fill_idx = min(candidate_exit_idx + 1, len(df) - 1)
        entry_price = _fill_price(fill_px, entry_fill_idx, side, slip, entry=True)
        entry_fee = cash * float(fee) * effective_notional
        cash -= entry_fee
        entry_equity = cash
        peak_unrealized = 0.0
        max_giveback = 0.0

        for j in range(entry_idx, candidate_exit_idx + 1):
            mark_exit = _mark_exit_price(close, j, side, slip)
            unrealized = _raw(side, entry_price, mark_exit) * effective_notional
            peak_unrealized = max(peak_unrealized, unrealized)
            max_giveback = max(max_giveback, peak_unrealized - unrealized)
            equity = cash * (1.0 + unrealized)
            peak = max(peak, equity)
            mdd = min(mdd, equity / max(peak, 1e-12) - 1.0)

        exit_price = _fill_price(fill_px, exit_fill_idx, side, slip, entry=False)
        raw = _raw(side, entry_price, exit_price)
        cash_after_entry_fee = cash
        cash = cash * (1.0 + raw * effective_notional)
        exit_fee = cash_after_entry_fee * float(fee) * effective_notional
        cash -= exit_fee
        fee_paid = float(entry_fee + exit_fee)
        slip_paid = _slip_paid_estimate(cash_before, cash_after_entry_fee, effective_notional, slip)
        trade_pnl_pct = float((cash / max(cash_before, 1e-12) - 1.0) * 100.0)
        wins += int(cash > entry_equity)
        notional_sum += effective_notional
        leverage_sum += candidate_leverage
        action_distribution[action] = action_distribution.get(action, 0) + 1
        edit = str(life.get("edit", "noop"))
        exit_reason = str(life.get("exit_reason", "base_exit"))
        edit_counts[edit] = edit_counts.get(edit, 0) + 1
        exits[exit_reason] = exits.get(exit_reason, 0) + 1

        ledger.append(
            {
                "trade_id": int(trade_id),
                "timestamp": str(df["timestamp"].iloc[entry_idx]) if "timestamp" in df.columns else str(entry_idx),
                "entry_idx": int(entry_idx),
                "base_exit_idx": int(base_exit_idx),
                "candidate_exit_idx": int(candidate_exit_idx),
                "side": int(side),
                "base_side": int(base_side),
                "candidate_side": int(candidate_side),
                "base_notional": float(base_notional),
                "candidate_notional": float(candidate_notional),
                "effective_notional": float(effective_notional),
                "base_leverage": float(base_leverage),
                "candidate_leverage": float(candidate_leverage),
                "action": action,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "fee_paid": fee_paid,
                "slip_paid": slip_paid,
                "prior_trade_giveback_pre_decision": float(prior_trade_giveback),
                "current_trade_giveback_after_close": float(max_giveback),
                "trade_pnl_pct": trade_pnl_pct,
                "cash_before": float(cash_before),
                "cash_after": float(cash),
            }
        )
        prior_trade_giveback = max_giveback

    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(ledger[0].keys()) if ledger else ["trade_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ledger)

    trades = len(lifecycle_plan)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "avg_notional": float(notional_sum / max(trades, 1)),
        "avg_leverage": float(leverage_sum / max(trades, 1)),
        "edit_counts": edit_counts,
        "exits": exits,
        "action_distribution": action_distribution,
        "ledger": ledger,
    }


def _compact_fixed(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "pnl",
        "mdd",
        "trades",
        "trades_per_day",
        "wr",
        "avg_notional",
        "avg_leverage",
        "edit_counts",
        "exits",
        "action_distribution",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _preservation_audit(
    base_trades: list[dict[str, Any]],
    lifecycle_plan: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    *,
    max_notional: float,
) -> dict[str, Any]:
    base_by_entry = {int(t["entry_idx"]): t for t in base_trades}
    life_by_entry = {int(t["entry_idx"]): t for t in lifecycle_plan}
    ledger_entries = {int(t["entry_idx"]) for t in ledger}
    violations = {
        "trade_count_changed": int(len(base_trades) != len(lifecycle_plan) or len(lifecycle_plan) != len(ledger)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "candidate_side_changed": 0,
        "entry_deleted": int(len(set(base_by_entry) - ledger_entries)),
        "new_entries": int(len(ledger_entries - set(base_by_entry))),
        "exit_after_base": 0,
        "negative_effective_notional": 0,
        "effective_notional_cap": 0,
        "leverage_increased": 0,
        "leverage_changed": 0,
        "cooldown_changed": 0,
    }
    measured = {
        "notional_increased_over_base": 0,
        "notional_decreased_vs_base": 0,
        "notional_equal_to_base": 0,
    }
    for row in ledger:
        entry = int(row["entry_idx"])
        base = base_by_entry.get(entry)
        life = life_by_entry.get(entry)
        if base is None or life is None:
            continue
        violations["entry_idx_changed"] += int(entry != int(base["entry_idx"]) or entry != int(life["entry_idx"]))
        violations["side_changed"] += int(int(row["side"]) != int(base["side"]))
        violations["candidate_side_changed"] += int(int(row["candidate_side"]) != int(base["side"]) or int(row["candidate_side"]) != int(life["side"]))
        violations["exit_after_base"] += int(int(row["candidate_exit_idx"]) > int(base["exit_idx"]))
        violations["negative_effective_notional"] += int(float(row["effective_notional"]) < -1e-12)
        violations["effective_notional_cap"] += int(float(row["effective_notional"]) > float(max_notional) + 1e-12)
        violations["leverage_increased"] += int(float(row["candidate_leverage"]) > float(base["leverage"]) + 1e-12)
        violations["leverage_changed"] += int(abs(float(row["candidate_leverage"]) - float(base["leverage"])) > 1e-12)
        violations["cooldown_changed"] += int(int(base.get("cooldown_bars", 0)) != int(life.get("cooldown_bars", 0)))
        measured["notional_increased_over_base"] += int(float(row["candidate_notional"]) > float(base["base_notional"]) + 1e-12)
        measured["notional_decreased_vs_base"] += int(float(row["candidate_notional"]) < float(base["base_notional"]) - 1e-12)
        measured["notional_equal_to_base"] += int(abs(float(row["candidate_notional"]) - float(base["base_notional"])) <= 1e-12)

    allowed_invariant_keys = [
        "trade_count_changed",
        "entry_idx_changed",
        "side_changed",
        "candidate_side_changed",
        "entry_deleted",
        "new_entries",
        "exit_after_base",
        "negative_effective_notional",
        "effective_notional_cap",
        "leverage_increased",
        "leverage_changed",
        "cooldown_changed",
    ]
    return {
        "passed": bool(sum(violations[k] for k in allowed_invariant_keys) == 0),
        "base_trades": int(len(base_trades)),
        "candidate_trades": int(len(lifecycle_plan)),
        "ledger_trades": int(len(ledger)),
        "violations": violations,
        "measured_policy_changes": measured,
        "strict_base_notional_preservation_passed": bool(measured["notional_increased_over_base"] == 0 and measured["notional_decreased_vs_base"] == 0),
        "note": "Lifecycle V1 intentionally edits notional; measured notional increases are reported separately from side/timing/leverage/cap invariants.",
    }


def _giveback_carry_audit(ledger: list[dict[str, Any]], *, tol: float = 1e-12) -> dict[str, Any]:
    violations = 0
    max_abs_diff = 0.0
    first_prior = None
    for i, row in enumerate(ledger):
        prior = float(row["prior_trade_giveback_pre_decision"])
        current = float(row["current_trade_giveback_after_close"])
        if i == 0:
            first_prior = prior
            diff = abs(prior)
        else:
            prev_current = float(ledger[i - 1]["current_trade_giveback_after_close"])
            diff = abs(prior - prev_current)
        max_abs_diff = max(max_abs_diff, diff)
        violations += int(diff > tol)
        if not np.isfinite(current):
            violations += 1
    return {
        "passed": bool(violations == 0),
        "rows_checked": int(len(ledger)),
        "violations": int(violations),
        "first_prior_trade_giveback": first_prior,
        "max_abs_carry_diff": float(max_abs_diff),
        "tolerance": float(tol),
    }


def _cost_rows(cost: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for label, metrics in cost.items():
        out.append(
            {
                "cost_mode": label,
                "pnl": metrics.get("pnl"),
                "mdd": metrics.get("mdd"),
                "trades": metrics.get("trades"),
                "trades_per_day": metrics.get("trades_per_day"),
                "avg_notional": metrics.get("avg_notional"),
                "avg_leverage": metrics.get("avg_leverage"),
            }
        )
    return out


def _write_fixed_cost_csv(path: Path, path_changing: dict[str, dict[str, Any]], fixed_ledger: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["cost_type", "cost_mode", "pnl", "mdd", "trades", "trades_per_day", "avg_notional", "avg_leverage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cost_type, block in (("path_changing", path_changing), ("fixed_ledger", fixed_ledger)):
            for row in _cost_rows(block):
                row["cost_type"] = cost_type
                writer.writerow(row)


def _write_doc(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = report["candidate_oos"]
    fixed = report["cost_fixed_ledger_3x"]
    path_3x = report["cost_path_changing_3x"]
    path.write_text(
        "\n".join(
            [
                "# Lifecycle V1 Canonical Accounting Audit 2026",
                "",
                f"Verdict: `{report['top_level_verdict']}`",
                "",
                "## Purpose",
                "",
                "Freeze Lifecycle V1 as the best current research substrate and audit its accounting package without adding a new alpha layer.",
                "",
                "## Key Results",
                "",
                f"- Lifecycle V1 OOS PnL 1x: `{candidate['pnl']:.6f}%`",
                f"- Lifecycle V1 OOS MDD 1x: `{candidate['mdd']:.6f}%`",
                f"- Trades/day: `{candidate['trades_per_day']:.6f}`",
                f"- Fixed-ledger 3x PnL: `{fixed['pnl']:.6f}%`",
                f"- Path-changing 3x PnL: `{path_3x['pnl']:.6f}%`",
                f"- Preservation audit passed: `{report['preservation_audit']['passed']}`",
                f"- Giveback carry audit passed: `{report['accounting_audit']['giveback_carry_audit']['passed']}`",
                "",
                "## Cost Summary",
                "",
                "| View | 1x PnL | 2x PnL | 3x PnL | 3x Trades |",
                "|---|---:|---:|---:|---:|",
                f"| Path-changing | `{report['cost_path_changing_1x']['pnl']:.6f}%` | `{report['cost_path_changing_2x']['pnl']:.6f}%` | `{report['cost_path_changing_3x']['pnl']:.6f}%` | `{report['cost_path_changing_3x']['trades']}` |",
                f"| Fixed ledger | `{report['cost_fixed_ledger_1x']['pnl']:.6f}%` | `{report['cost_fixed_ledger_2x']['pnl']:.6f}%` | `{report['cost_fixed_ledger_3x']['pnl']:.6f}%` | `{report['cost_fixed_ledger_3x']['trades']}` |",
                "",
                "## Cost Separation",
                "",
                "This package reports two different cost views:",
                "",
                "- `cost_path_changing_*`: rebuilds the clean-base trade path under each fee/slippage multiplier.",
                "- `cost_fixed_ledger_*`: keeps the 1x Lifecycle V1 ledger fixed and changes only fee/slippage.",
                "",
                "## Promotion Decision",
                "",
                "Lifecycle V1 remains a research candidate, not a promoted production model. It improves PnL over the clean base but misses the clean-base MDD gate, fails path-changing 3x survival, and still lacks realistic funding/impact/partial-fill replay.",
                "",
                "## Artifacts",
                "",
                f"- Report: `{report['artifacts']['report']}`",
                f"- Ledger: `{report['artifacts']['ledger_csv']}`",
                f"- Cost CSV: `{report['artifacts']['fixed_ledger_cost_csv']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical accounting audit package for Lifecycle V1.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--lifecycle-report", type=Path, default=DEFAULT_LIFECYCLE_REPORT)
    parser.add_argument("--lifecycle-model", type=Path, default=DEFAULT_LIFECYCLE_MODEL)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--split-date", default="2025-11-01")
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slip", type=float, default=0.0002)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--fixed-cost-csv-out", type=Path, default=DEFAULT_FIXED_COST)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    recalibrator, lifecycle_cfg, lifecycle_report = _load_lifecycle_model(args.lifecycle_model, args.lifecycle_report)

    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)

    oos_pre_1x = _base_frame(oos_df, policy, entry_cfg)
    _feat, oos_decisions, _close, _fill = oos_pre_1x
    clean_oos = backtest_no_limit_exit(
        oos_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=oos_pre_1x,
    )

    path_changing: dict[str, dict[str, Any]] = {}
    path_changing_full: dict[str, dict[str, Any]] = {}
    path_trade_counts: dict[str, int] = {}
    fixed_ledger: dict[str, dict[str, Any]] = {}
    fixed_ledger_full: dict[str, dict[str, Any]] = {}
    base_trades_1x: list[dict[str, Any]] | None = None
    lifecycle_plan_1x: list[dict[str, Any]] | None = None

    for mult in (1.0, 2.0, 3.0):
        label = f"{int(mult)}x"
        pre = _base_frame(oos_df, policy, entry_cfg)
        base_trades = _base_trade_plan(
            oos_df,
            exit_model,
            risk_cfg,
            exit_cfg,
            pre,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
        )
        full = backtest_lifecycle_editor(
            oos_df,
            exit_model,
            recalibrator,
            lifecycle_cfg,
            base_trades,
            exit_cfg,
            pre,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
        )
        path_changing_full[label] = full
        path_changing[label] = _compact(full)
        path_trade_counts[label] = int(full["trades"])
        if mult == 1.0:
            base_trades_1x = base_trades
            lifecycle_plan_1x = list(full["lifecycle_plan"])

    if base_trades_1x is None or lifecycle_plan_1x is None:
        raise RuntimeError("failed to build 1x lifecycle plan")

    for mult in (1.0, 2.0, 3.0):
        label = f"{int(mult)}x"
        full = _simulate_fixed_plan(
            oos_df,
            oos_pre_1x,
            base_trades_1x,
            lifecycle_plan_1x,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
            ledger_out=args.ledger_csv_out if mult == 1.0 else None,
        )
        fixed_ledger_full[label] = full
        fixed_ledger[label] = _compact_fixed(full)

    preservation = _preservation_audit(
        base_trades_1x,
        lifecycle_plan_1x,
        fixed_ledger_full["1x"]["ledger"],
        max_notional=float(risk_cfg.get("max_notional", 3.6)),
    )
    decision_audit = _decision_audit(oos_decisions, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0)
    giveback_carry_audit = _giveback_carry_audit(fixed_ledger_full["1x"]["ledger"])
    accounting_audit = {
        "passed": bool(
            preservation["passed"]
            and decision_audit.get("passed", False)
            and fixed_ledger["1x"]["trades"] == path_changing["1x"]["trades"]
            and giveback_carry_audit["passed"]
        ),
        "ledger_prior_state_fix": "prior_trade_giveback_pre_decision is written before the current trade executes; current_trade_giveback_after_close is written after close and carried to the next row.",
        "path_changing_and_fixed_ledger_costs_separated": True,
        "giveback_carry_audit": giveback_carry_audit,
    }
    mdd_gap = float(path_changing["1x"]["mdd"]) - float(BASE_REFERENCE["mdd"])
    reject_reasons = [
        "audit_artifact_only_not_promoted",
        f"Lifecycle V1 MDD gate miss vs clean base: {mdd_gap:.6f} percentage points",
        "realistic funding/impact/partial-fill replay missing",
    ]
    if float(path_changing["3x"]["pnl"]) <= 0.0:
        reject_reasons.append("path-changing 3x cost stress does not survive")
    if preservation["measured_policy_changes"]["notional_increased_over_base"] > 0:
        reject_reasons.append("Lifecycle V1 intentionally boosts notional over clean base; requires risk approval before promotion")

    report = {
        "type": "lifecycle_v1_canonical_accounting_audit_2026",
        "top_level_verdict": "audit_artifact_only_not_promoted",
        "best_research_candidate": "clean_base_lifecycle_editor_v1",
        "clean_base_reference": BASE_REFERENCE,
        "clean_base_oos_recomputed": _compact(clean_oos),
        "lifecycle_v1_reference": {
            "source_report": str(args.lifecycle_report),
            "source_verdict": lifecycle_report.get("verdict"),
            "selected_for_report": lifecycle_report.get("selected_for_report"),
            "selected_config": lifecycle_cfg.__dict__,
        },
        "candidate_oos": fixed_ledger["1x"],
        "cost_path_changing_1x": path_changing["1x"],
        "cost_path_changing_2x": path_changing["2x"],
        "cost_path_changing_3x": path_changing["3x"],
        "cost_fixed_ledger_1x": fixed_ledger["1x"],
        "cost_fixed_ledger_2x": fixed_ledger["2x"],
        "cost_fixed_ledger_3x": fixed_ledger["3x"],
        "trade_count_path_changing": path_trade_counts,
        "trade_count_fixed_ledger": {k: int(v["trades"]) for k, v in fixed_ledger.items()},
        "action_distribution": fixed_ledger["1x"].get("action_distribution", {}),
        "edit_counts": fixed_ledger["1x"].get("edit_counts", {}),
        "exits": fixed_ledger["1x"].get("exits", {}),
        "decision_frame_audit": decision_audit,
        "preservation_audit": preservation,
        "accounting_audit": accounting_audit,
        "reject_reasons": reject_reasons,
        "promotion_gate": {
            "pnl_beats_clean_base": bool(float(fixed_ledger["1x"]["pnl"]) >= float(BASE_REFERENCE["pnl"])),
            "mdd_beats_clean_base": bool(float(fixed_ledger["1x"]["mdd"]) >= float(BASE_REFERENCE["mdd"])),
            "trades_per_day_at_least_clean_base": bool(float(fixed_ledger["1x"]["trades_per_day"]) >= float(BASE_REFERENCE["trades_per_day"])),
            "fixed_ledger_cost_1x_positive": bool(float(fixed_ledger["1x"]["pnl"]) > 0.0),
            "fixed_ledger_cost_2x_positive": bool(float(fixed_ledger["2x"]["pnl"]) > 0.0),
            "fixed_ledger_cost_3x_positive": bool(float(fixed_ledger["3x"]["pnl"]) > 0.0),
            "path_changing_cost_1x_positive": bool(float(path_changing["1x"]["pnl"]) > 0.0),
            "path_changing_cost_2x_positive": bool(float(path_changing["2x"]["pnl"]) > 0.0),
            "path_changing_cost_3x_positive": bool(float(path_changing["3x"]["pnl"]) > 0.0),
            "cost_3x_survives_both_views": bool(float(fixed_ledger["3x"]["pnl"]) > 0.0 and float(path_changing["3x"]["pnl"]) > 0.0),
            "invariant_audit_passed": bool(preservation["passed"] and accounting_audit["passed"]),
        },
        "data_contract": {
            "train_range": _range(train_df),
            "validation_range": _range(val_df),
            "oos_range": _range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
            "runtime_inputs": [
                "frozen clean base decisions",
                "frozen clean base exit governor state",
                "train-only lifecycle hazard buckets",
                "validation-selected LifecycleRuntimeConfig",
            ],
        },
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "report": str(args.report_out),
            "ledger_csv": str(args.ledger_csv_out),
            "fixed_ledger_cost_csv": str(args.fixed_cost_csv_out),
            "doc": str(args.doc_out),
        },
        "frozen_artifacts": {
            "base_policy": str(args.policy),
            "base_policy_sha256": _sha256(args.policy),
            "base_exit_governor": str(args.exit_model),
            "base_exit_governor_sha256": _sha256(args.exit_model),
            "lifecycle_v1_model": str(args.lifecycle_model),
            "lifecycle_v1_model_sha256": _sha256(args.lifecycle_model),
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_fixed_cost_csv(args.fixed_cost_csv_out, path_changing, fixed_ledger)
    _write_doc(args.doc_out, report)
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "ledger": str(args.ledger_csv_out),
                "fixed_cost": str(args.fixed_cost_csv_out),
                "doc": str(args.doc_out),
                "verdict": report["top_level_verdict"],
                "candidate_oos": report["candidate_oos"],
                "promotion_gate": report["promotion_gate"],
                "preservation_passed": report["preservation_audit"]["passed"],
                "accounting_passed": report["accounting_audit"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
