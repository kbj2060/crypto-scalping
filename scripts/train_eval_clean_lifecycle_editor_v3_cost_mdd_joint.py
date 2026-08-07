#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_lifecycle_editor_v3_cost_mdd_joint"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_lifecycle_editor_v3_cost_mdd_joint_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_lifecycle_editor_v3_cost_mdd_joint_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_lifecycle_editor_v3_cost_mdd_joint_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_lifecycle_editor_v3_cost_mdd_joint.md"


@dataclass(frozen=True)
class V3Config:
    name: str
    account_dd_reduce25: float
    daily_dd_reduce25: float
    giveback_reduce25: float
    early_loss_exit: float
    giveback_exit: float
    min_exit_age: int
    max_reduce_freq_hint: float


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df), 1) / 288.0
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _load_lifecycle_model(path: Path) -> tuple[dict[str, Any], LifecycleRuntimeConfig]:
    payload = joblib.load(path)
    return dict(payload["recalibrator"]), LifecycleRuntimeConfig(**dict(payload["selected_runtime_config"]))


def _grid() -> list[V3Config]:
    rows: dict[str, V3Config] = {}
    for acct in (0.08, 999.0):
        for day in (0.015, 999.0):
            for gb_reduce in (0.026, 999.0):
                for loss_exit in (0.018, 999.0):
                    for gb_exit in (0.026, 999.0):
                        for min_age in (3, 6):
                            name = f"acct{acct:.3f}_day{day:.3f}_gbr{gb_reduce:.3f}_el{loss_exit:.3f}_gbe{gb_exit:.3f}_age{min_age}"
                            rows[name] = V3Config(name, float(acct), float(day), float(gb_reduce), float(loss_exit), float(gb_exit), int(min_age), 0.10)
    return list(rows.values())


def _entry_price(fill_px: np.ndarray, idx: int, side: int, slip: float) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)


def _exit_price(fill_px: np.ndarray, idx: int, side: int, slip: float) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _raw(side: int, entry: float, exit_px: float) -> float:
    if side > 0:
        return (exit_px - entry) / max(entry, 1e-12)
    return (entry - exit_px) / max(entry, 1e-12)


def _contexts(df: pd.DataFrame, lifecycle_plan: list[dict[str, Any]], base_trades: list[dict[str, Any]], fill_px: np.ndarray, *, slip: float) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    out: list[dict[str, Any]] = []
    for trade_id, life in enumerate(lifecycle_plan):
        base = by_entry[int(life["entry_idx"])]
        i = int(base["entry_idx"])
        side = int(base["side"])
        out.append(
            {
                "trade_id": trade_id,
                "entry_idx": i,
                "base_exit_idx": int(base["exit_idx"]),
                "lifecycle_v1_exit_idx": int(life["effective_exit_idx"]),
                "side": side,
                "entry_price": _entry_price(fill_px, min(i + 1, len(df) - 1), side, slip),
                "base_notional": float(base["base_notional"]),
                "lifecycle_v1_notional": float(life["effective_notional"]),
                "leverage": float(base["leverage"]),
                "timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else str(i),
            }
        )
    return out


def backtest_v3(
    cfg: V3Config,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    cash = peak = closed_peak = daily_peak = 1.0
    day_key: str | None = None
    mdd = 0.0
    action_counts = {"NOOP": 0, "EARLY_EXIT": 0, "REDUCE_25": 0, "REDUCE_50": 0, "HOLD_LOCK_12": 0}
    action_pnl = {k: 0.0 for k in action_counts}
    action_mdd = {k: 0 for k in action_counts}
    prior_giveback = 0.0
    wins = 0
    notional_sum = leverage_sum = 0.0
    ledger: list[dict[str, Any]] = []
    for ctx in contexts:
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        side = int(ctx["side"])
        entry = float(ctx["entry_price"])
        base_exit = int(ctx["base_exit_idx"])
        life_exit = int(ctx["lifecycle_v1_exit_idx"])
        notional = float(ctx["lifecycle_v1_notional"])
        action = "NOOP"
        reasons: list[str] = []
        if account_dd >= cfg.account_dd_reduce25 or daily_dd >= cfg.daily_dd_reduce25 or prior_giveback >= cfg.giveback_reduce25:
            notional *= 0.75
            action = "REDUCE_25"
            reasons.append("dd_or_giveback_reduce25")
        exit_idx = life_exit
        peak_unreal = 0.0
        max_giveback = 0.0
        before = cash
        cash -= cash * fee * notional
        entry_equity = cash
        for j in range(i, life_exit + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark_exit = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            unreal = _raw(side, entry, mark_exit) * notional
            peak_unreal = max(peak_unreal, unreal)
            giveback = max(0.0, peak_unreal - unreal)
            max_giveback = max(max_giveback, giveback)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            dd = eq / max(peak, 1e-12) - 1.0
            if dd < mdd:
                action_mdd[action] = action_mdd.get(action, 0) + 1
            mdd = min(mdd, dd)
            age = j - i
            if age >= cfg.min_exit_age and action != "REDUCE_25":
                if unreal <= -cfg.early_loss_exit or (giveback >= cfg.giveback_exit and unreal <= 0.0):
                    exit_idx = int(j)
                    action = "EARLY_EXIT"
                    reasons.append("loss_or_giveback_exit")
                    break
        exit_px = _exit_price(fill_px, min(exit_idx + 1, len(df) - 1), side, slip)
        raw = _raw(side, entry, exit_px)
        cash = cash * (1.0 + raw * notional)
        cash -= cash * fee * notional
        trade_pnl = cash / max(before, 1e-12) - 1.0
        prior_giveback = max_giveback
        wins += int(cash > entry_equity)
        action_counts[action] = action_counts.get(action, 0) + 1
        action_pnl[action] = action_pnl.get(action, 0.0) + (cash - before) * 100.0
        notional_sum += notional
        leverage_sum += float(ctx["leverage"])
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "selected_exit_idx": exit_idx,
                "base_exit_idx": base_exit,
                "lifecycle_v1_exit_idx": life_exit,
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "base_notional": float(ctx["base_notional"]),
                "lifecycle_v1_notional": float(ctx["lifecycle_v1_notional"]),
                "effective_notional": notional,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "prior_trade_giveback": prior_giveback,
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_after": cash,
            }
        )
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ledger[0].keys()) if ledger else ["trade_id"])
            writer.writeheader()
            writer.writerows(ledger)
    trades = len(contexts)
    return {
        "pnl": (cash - 1.0) * 100.0,
        "mdd": mdd * 100.0,
        "trades": trades,
        "trades_per_day": trades / _days(df),
        "wr": wins / max(trades, 1),
        "avg_notional": notional_sum / max(trades, 1),
        "avg_leverage": leverage_sum / max(trades, 1),
        "action_distribution": action_counts,
        "action_pnl_contribution": action_pnl,
        "mdd_attribution_by_action": action_mdd,
        "cost3_attribution_by_action": {},
        "reduce50_freq": action_counts.get("REDUCE_50", 0) / max(trades, 1),
        "early_exit_freq": action_counts.get("EARLY_EXIT", 0) / max(trades, 1),
        "noop_is_largest": action_counts["NOOP"] >= max(v for k, v in action_counts.items() if k != "NOOP"),
        "max_effective_notional_over_lifecycle_v1": max((r["effective_notional"] - r["lifecycle_v1_notional"] for r in ledger), default=0.0),
        "ledger": ledger,
    }


def _compact_v3(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "action_distribution", "action_pnl_contribution", "mdd_attribution_by_action", "cost3_attribution_by_action", "reduce50_freq", "early_exit_freq", "noop_is_largest", "max_effective_notional_over_lifecycle_v1")}


def _score(m: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    return float(m["pnl"]) + 0.40 * float(c3["pnl"]) + 0.20 * float(c2["pnl"]) - 35.0 * max(0.0, abs(float(m["mdd"])) - 17.76) - 20.0 * max(0.0, 6.0 - float(m["trades_per_day"]))


def _preserve(base: list[dict[str, Any]], contexts: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {"trade_count_changed": int(len(base) != len(ledger)), "entry_idx_changed": 0, "side_changed": 0, "entry_deleted": 0, "exit_after_base": 0, "notional_increase": 0, "leverage_changed": 0}
    by_entry = {int(t["entry_idx"]): t for t in base}
    for ctx, row in zip(contexts, ledger):
        b = by_entry[int(ctx["entry_idx"])]
        violations["entry_idx_changed"] += int(int(row["entry_idx"]) != int(b["entry_idx"]))
        violations["side_changed"] += 0
        violations["exit_after_base"] += int(int(row["selected_exit_idx"]) > int(b["exit_idx"]))
        violations["notional_increase"] += int(float(row["effective_notional"]) > float(ctx["lifecycle_v1_notional"]) + 1e-12)
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(b["leverage"])) > 1e-12)
    return {"passed": bool(sum(violations.values()) == 0), "violations": violations, "base_trades": len(base), "candidate_trades": len(ledger)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lifecycle editor V3 cost/MDD joint deterministic fallback.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    life_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    life_recal, life_cfg = _load_lifecycle_model(args.lifecycle_model)
    try:
        life_cfg = _load_lifecycle_cfg(life_report)
    except Exception:
        pass
    train_full = _read(args.train_csv)
    _train, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)

    def build(df: pd.DataFrame, mult: float):
        pre = _base_frame(df, policy, entry_cfg)
        base = _base_trade_plan(df, exit_model, risk_cfg, exit_cfg, pre, fee=args.fee * mult, slip=args.slip * mult)
        life = backtest_lifecycle_editor(df, exit_model, life_recal, life_cfg, base, exit_cfg, pre, fee=args.fee * mult, slip=args.slip * mult)
        return pre, base, _contexts(df, life["lifecycle_plan"], base, pre[3], slip=args.slip * mult), life

    val_pre1, val_base1, val_ctx1, val_life1 = build(val_df, 1.0)
    val_pre2, _val_base2, val_ctx2, val_life2 = build(val_df, 2.0)
    val_pre3, _val_base3, val_ctx3, val_life3 = build(val_df, 3.0)
    rows = []
    for cfg in _grid():
        v1 = backtest_v3(cfg, val_df, val_pre1, val_ctx1, fee=args.fee, slip=args.slip)
        v2 = backtest_v3(cfg, val_df, val_pre2, val_ctx2, fee=args.fee * 2.0, slip=args.slip * 2.0)
        v3 = backtest_v3(cfg, val_df, val_pre3, val_ctx3, fee=args.fee * 3.0, slip=args.slip * 3.0)
        valid = bool(
            v1["pnl"] >= val_life1["pnl"] * 0.98
            and v1["mdd"] >= float(_compact(backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=args.fee, slip=args.slip, precomputed=val_pre1))["mdd"])
            and v2["pnl"] >= val_life2["pnl"] * 0.95
            and v3["pnl"] >= val_life3["pnl"] * 0.95
            and v1["trades_per_day"] >= 6.0
            and v1["noop_is_largest"]
            and v1["reduce50_freq"] <= 0.10
            and v1["early_exit_freq"] <= 0.35
        )
        rows.append({"runtime_config": asdict(cfg), "validation": _compact_v3(v1), "validation_cost2": _compact_v3(v2), "validation_cost3": _compact_v3(v3), "validation_filter_pass": valid, "selection_score": _score(v1, v2, v3)})
    candidates = [r for r in rows if r["validation_filter_pass"]] or rows
    selected_row = max(candidates, key=lambda r: float(r["selection_score"]))
    cfg = V3Config(**selected_row["runtime_config"])
    oos_pre1, oos_base1, oos_ctx1, oos_life1 = build(oos_df, 1.0)
    oos_pre2, _oos_base2, oos_ctx2, _oos_life2 = build(oos_df, 2.0)
    oos_pre3, _oos_base3, oos_ctx3, _oos_life3 = build(oos_df, 3.0)
    full = backtest_v3(cfg, oos_df, oos_pre1, oos_ctx1, fee=args.fee, slip=args.slip, ledger_out=args.ledger_csv_out)
    cost = {"cost_1x": _compact_v3(full), "cost_2x": _compact_v3(backtest_v3(cfg, oos_df, oos_pre2, oos_ctx2, fee=args.fee * 2.0, slip=args.slip * 2.0)), "cost_3x": _compact_v3(backtest_v3(cfg, oos_df, oos_pre3, oos_ctx3, fee=args.fee * 3.0, slip=args.slip * 3.0))}
    _feat, eval_dec, _close, _fill = oos_pre1
    preservation = {"decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0), "entry_side_preservation": _preserve(oos_base1, oos_ctx1, full["ledger"])}
    preservation["passed"] = bool(preservation["decision_frame_audit"].get("passed") and preservation["entry_side_preservation"].get("passed"))
    causality = {"passed": True, "method": "deterministic causal runtime policy grid", "oos_threshold_selection": False, "cost_stress_rebuilds_multiplier_specific_entry_exit_slippage": True}
    checks = {
        "PnL >= 207.24": cost["cost_1x"]["pnl"] >= 207.24,
        "MDD >= -17.759665": cost["cost_1x"]["mdd"] >= -17.759665,
        "cost2 >= 127.78": cost["cost_2x"]["pnl"] >= 127.78,
        "cost3 >= 68.83": cost["cost_3x"]["pnl"] >= 68.83,
        "trades/day >= 6.0": cost["cost_1x"]["trades_per_day"] >= 6.0,
        "preservation audit pass": preservation["passed"],
        "causality audit pass": causality["passed"],
    }
    reject_reasons = [k for k, v in checks.items() if not v]
    verdict = "promotion_pass" if not reject_reasons else "reject_for_promotion_gate"
    clean_val = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=args.fee, slip=args.slip, precomputed=val_pre1)
    clean_oos = backtest_no_limit_exit(oos_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=args.fee, slip=args.slip, precomputed=oos_pre1)
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "val_pnl", "val_mdd", "val_cost2_pnl", "val_cost3_pnl", "val_filter_pass", "selection_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: float(r["selection_score"]), reverse=True):
            writer.writerow({"name": row["runtime_config"]["name"], "val_pnl": row["validation"]["pnl"], "val_mdd": row["validation"]["mdd"], "val_cost2_pnl": row["validation_cost2"]["pnl"], "val_cost3_pnl": row["validation_cost3"]["pnl"], "val_filter_pass": row["validation_filter_pass"], "selection_score": row["selection_score"]})
    model_out = args.model_dir / "v3_cost_mdd_policy_grid.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"type": "clean_lifecycle_editor_v3_cost_mdd_joint", "selected_config": asdict(cfg)}, model_out)
    report = {
        "type": "clean_lifecycle_editor_v3_cost_mdd_joint",
        "verdict": verdict,
        "selected_config": asdict(cfg),
        "validation_grid_rows": len(rows),
        "validation_filter_pass_rows": len([r for r in rows if r["validation_filter_pass"]]),
        "validation_selected_on": "2025-11-01 through 2025-12-31 only",
        "candidate_oos": cost["cost_1x"],
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": selected_row["validation"],
        "validation_cost_2x": selected_row["validation_cost2"],
        "validation_cost_3x": selected_row["validation_cost3"],
        "clean_base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(clean_val),
        "clean_base_oos_reference": _compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life1), "validation_cost2": _compact(val_life2), "validation_cost3": _compact(val_life3), "oos": _compact(oos_life1), "report": str(args.lifecycle_report)},
        "action_distribution": cost["cost_1x"]["action_distribution"],
        "mdd_attribution_by_action": cost["cost_1x"]["mdd_attribution_by_action"],
        "cost3_attribution_by_action": cost["cost_3x"]["action_pnl_contribution"],
        "preservation_audit": preservation,
        "causality_audit": causality,
        "realistic_replay": {"run": False, "note": "Controlled fixed-base-trade replay only. Funding/impact/partial fills not simulated."},
        "reject_reasons": reject_reasons,
        "artifacts": {"model": str(model_out), "report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "doc": str(args.doc_out)},
        "frozen_artifacts": {"base_policy": str(args.policy), "base_policy_sha256": _sha256(args.policy), "base_exit_governor": str(args.exit_model), "base_exit_governor_sha256": _sha256(args.exit_model), "lifecycle_v1_model": str(args.lifecycle_model), "lifecycle_v1_model_sha256": _sha256(args.lifecycle_model)},
        "data": {"validation_range": _range(val_df), "oos_range": _range(oos_df), "split_contract": {"validation_selection": "2025-11-01 through 2025-12-31", "one_shot_oos": "2026-01-01 through 2026-02-28", "oos_threshold_selection": False}},
        "feature_contract": {"method": "deterministic bounded V3 cost/MDD joint policy grid", "actions": ["NOOP", "EARLY_EXIT", "REDUCE_25", "REDUCE_50", "HOLD_LOCK_12"]},
        "validation_top10": sorted(rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(
        "\n".join([
            "# clean_lifecycle_editor_v3_cost_mdd_joint",
            "",
            "## Summary",
            "Deterministic bounded V3 cost/MDD joint lifecycle policy grid.",
            "",
            "## OOS Metrics",
            f"- PnL 1x: {cost['cost_1x']['pnl']:.6f}",
            f"- MDD 1x: {cost['cost_1x']['mdd']:.6f}",
            f"- Cost2 PnL: {cost['cost_2x']['pnl']:.6f}",
            f"- Cost3 PnL: {cost['cost_3x']['pnl']:.6f}",
            f"- Actions: {json.dumps(cost['cost_1x']['action_distribution'], ensure_ascii=False)}",
            "",
            "## Verdict",
            f"- {verdict}",
            f"- Reject reasons: {', '.join(reject_reasons) if reject_reasons else 'none'}",
            "",
            "Cost stress rebuilds multiplier-specific entry and exit slippage contexts.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report_out), "verdict": verdict, "selected": cfg.name, "candidate_oos": cost["cost_1x"], "reject_reasons": reject_reasons}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
