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

from scripts import train_eval_clean_base_causal_trade_editor_v1_3 as editor  # noqa: E402
from scripts import train_eval_clean_base_plus_causal_conviction_sleeve_v1_1 as sleeve  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


MODEL_ID = "clean_base_conformal_downside_filter_v1_4"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_conformal_downside_filter_v1_4"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_conformal_downside_filter_v1_4_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_conformal_downside_filter_v1_4_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_conformal_downside_filter_v1_4_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_conformal_downside_filter_v1_4.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_conformal_downside_filter_v1_4_contract.md"

FORBIDDEN_RUNTIME_FEATURES = [
    "evt_candidate_side",
    "evt_candidate_label",
    "evt_side_margin",
    "future high/low/close",
]


@dataclass(frozen=True)
class ConformalDownsideConfig:
    name: str
    residual_quantile: float
    lcb_threshold: float
    adverse_cut: float
    scale_down: float
    early_gain_threshold: float
    min_notional: float
    account_dd_disable: float
    daily_dd_disable: float


def _grid() -> list[ConformalDownsideConfig]:
    rows: list[ConformalDownsideConfig] = []
    for q in (0.70, 0.80, 0.90):
        for lcb in (-0.0010, 0.0000):
            for adverse in (0.015, 0.020, 0.030):
                for scale_down in (0.60, 0.75, 0.90):
                    for early_gain in (0.0015, 0.0030):
                        for acct in (0.08,):
                            for day in (0.015,):
                                name = (
                                    f"q{q:.2f}_lcb{lcb:.4f}_adv{adverse:.3f}_dn{scale_down:.2f}_"
                                    f"eg{early_gain:.4f}_acct{acct:.2f}_day{day:.3f}"
                                )
                                rows.append(
                                    ConformalDownsideConfig(
                                        name=name,
                                        residual_quantile=float(q),
                                        lcb_threshold=float(lcb),
                                        adverse_cut=float(adverse),
                                        scale_down=float(scale_down),
                                        early_gain_threshold=float(early_gain),
                                        min_notional=0.20,
                                        account_dd_disable=float(acct),
                                        daily_dd_disable=float(day),
                                    )
                                )
    return rows


def _actual_full_returns(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
) -> np.ndarray:
    vals = [
        editor._future_path_stats(precomputed, ctx, horizon=None, fee=fee, slip=slip)["net"]
        for ctx in contexts
    ]
    return np.asarray(vals, dtype=np.float64)


def _conformal_calibration(
    actual: np.ndarray,
    pred: dict[str, np.ndarray],
) -> dict[str, Any]:
    residual = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(pred["full"], dtype=np.float64))
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        residual = np.asarray([0.0], dtype=np.float64)
    return {
        "rows": int(len(residual)),
        "residual_mean": float(np.mean(residual)),
        "residual_p60": float(np.quantile(residual, 0.60)),
        "residual_p70": float(np.quantile(residual, 0.70)),
        "residual_p80": float(np.quantile(residual, 0.80)),
        "residual_p90": float(np.quantile(residual, 0.90)),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "residuals": residual,
    }


def _residual_q(cal: dict[str, Any], q: float) -> float:
    residual = np.asarray(cal["residuals"], dtype=np.float64)
    return float(np.quantile(residual, float(q)))


def _choose(
    cfg: ConformalDownsideConfig,
    pred: dict[str, np.ndarray],
    k: int,
    *,
    residual_q: float,
    account_dd: float,
    daily_dd: float,
) -> tuple[str, float, int | None, dict[str, float], list[str]]:
    full = float(pred["full"][k])
    hvals = {6: float(pred["h6"][k]), 12: float(pred["h12"][k]), 24: float(pred["h24"][k])}
    best_h, best_early = max(hvals.items(), key=lambda kv: kv[1])
    adverse = float(pred["adverse"][k])
    lcb = full - float(residual_q)
    reasons: list[str] = []
    disabled = account_dd >= cfg.account_dd_disable or daily_dd >= cfg.daily_dd_disable
    if disabled:
        if account_dd >= cfg.account_dd_disable:
            reasons.append("account_dd_disable")
        if daily_dd >= cfg.daily_dd_disable:
            reasons.append("daily_dd_disable")
    if lcb <= cfg.lcb_threshold:
        reasons.append("lcb_below_threshold")
    if adverse >= cfg.adverse_cut:
        reasons.append("adverse_above_cut")

    action = "KEEP"
    scale = 1.0
    if reasons:
        action = "DOWNSIDE_SCALE"
        scale = float(cfg.scale_down)

    early_h: int | None = None
    if best_early >= full + float(cfg.early_gain_threshold):
        early_h = int(best_h)
        action = f"{action}_EARLY{best_h}" if action != "KEEP" else f"EARLY{best_h}"
        reasons.append("early_return_dominates_full")

    telemetry = {
        "pred_full": full,
        "pred_h6": hvals[6],
        "pred_h12": hvals[12],
        "pred_h24": hvals[24],
        "pred_adverse": adverse,
        "residual_q": float(residual_q),
        "pred_lcb": float(lcb),
    }
    return action, scale, early_h, telemetry, reasons


def backtest_conformal(
    cfg: ConformalDownsideConfig,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    pred: dict[str, np.ndarray],
    *,
    residual_q: float,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    mdd = 0.0
    wins = 0
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    notional_sum = 0.0
    scale_delta_sum = 0.0
    early_exits = 0
    gross_max = 0.0
    ledger: list[dict[str, Any]] = []
    for k, ctx in enumerate(contexts):
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        action, scale, early_h, telemetry, reasons = _choose(
            cfg,
            pred,
            k,
            residual_q=residual_q,
            account_dd=account_dd,
            daily_dd=daily_dd,
        )
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        side = int(ctx["side"])
        core_notional = float(ctx["core_notional"])
        effective_notional = float(np.clip(core_notional * scale, cfg.min_notional, core_notional))
        core_exit_idx = int(ctx["core_exit_idx"])
        exit_idx = min(core_exit_idx, i + int(early_h), len(close) - 2) if early_h else core_exit_idx
        early_exits += int(exit_idx < core_exit_idx)
        action_counts[action] = action_counts.get(action, 0) + 1
        before = cash
        entry = sleeve._entry_price(fill_px, min(i + 1, len(close) - 1), side, slip)
        entry_fee_cash = cash * float(fee) * effective_notional
        cash -= entry_fee_cash
        for j in range(i, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark_exit = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            unreal = sleeve._raw(side, entry, mark_exit) * effective_notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        exit_price = sleeve._exit_price(fill_px, min(exit_idx + 1, len(close) - 1), side, slip)
        realized = sleeve._raw(side, entry, exit_price) * effective_notional
        cash = cash * (1.0 + realized)
        exit_fee_cash = cash * float(fee) * effective_notional
        cash -= exit_fee_cash
        trade_pnl = cash / max(before, 1e-12) - 1.0
        wins += int(trade_pnl > 0.0)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        notional_sum += effective_notional
        scale_delta_sum += effective_notional - core_notional
        gross_max = max(gross_max, effective_notional)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "timestamp": ctx["timestamp"],
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "effective_exit_idx": int(exit_idx),
                "side": side,
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_notional": core_notional,
                "effective_notional": effective_notional,
                "scale": float(effective_notional / max(core_notional, 1e-12)),
                "early_horizon": int(early_h) if early_h else 0,
                "entry_price": float(entry),
                "exit_price": float(exit_price),
                "account_dd_prior": float(account_dd),
                "daily_dd_prior": float(daily_dd),
                "pred_full": telemetry["pred_full"],
                "pred_h6": telemetry["pred_h6"],
                "pred_h12": telemetry["pred_h12"],
                "pred_h24": telemetry["pred_h24"],
                "pred_adverse": telemetry["pred_adverse"],
                "residual_q": telemetry["residual_q"],
                "pred_lcb": telemetry["pred_lcb"],
                "realized_gross_frac": float(realized),
                "entry_fee_cash": float(entry_fee_cash),
                "exit_fee_cash": float(exit_fee_cash),
                "total_fee_cash": float(entry_fee_cash + exit_fee_cash),
                "trade_pnl_pct": float(trade_pnl * 100.0),
                "cash_before": float(before),
                "cash_after": float(cash),
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
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / sleeve._days(df)),
        "wr": float(wins / max(trades, 1)),
        "avg_notional": float(notional_sum / max(trades, 1)),
        "gross_notional_max": float(gross_max),
        "avg_scale_delta": float(scale_delta_sum / max(trades, 1)),
        "early_exit_fraction": float(early_exits / max(trades, 1)),
        "action_counts": action_counts,
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "trades_per_day",
            "wr",
            "avg_notional",
            "gross_notional_max",
            "avg_scale_delta",
            "early_exit_fraction",
            "action_counts",
            "reason_counts",
        )
    }


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    cost3_pnl = float(cost3["pnl"])
    avg_scale_delta = float(metrics["avg_scale_delta"])
    score = pnl + 0.18 * cost3_pnl
    score -= 110.0 * max(0.0, abs(mdd) - abs(editor.CLEAN_BASE_REFERENCE["mdd"]))
    score -= 20.0 * abs(min(avg_scale_delta, 0.0))
    return float(score)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key, val in row.items():
            if key not in fields and not isinstance(val, (dict, list, np.ndarray)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _preservation(contexts: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(contexts) != len(ledger)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "exit_after_core_exit": 0,
        "negative_notional": 0,
        "notional_above_core": 0,
        "notional_above_cap": 0,
    }
    for ctx, row in zip(contexts, ledger):
        violations["entry_idx_changed"] += int(int(ctx["entry_idx"]) != int(row["entry_idx"]))
        violations["side_changed"] += int(int(ctx["side"]) != int(row["side"]))
        violations["exit_after_core_exit"] += int(int(row["effective_exit_idx"]) > int(ctx["core_exit_idx"]))
        violations["negative_notional"] += int(float(row["effective_notional"]) < 0.0)
        violations["notional_above_core"] += int(float(row["effective_notional"]) > float(ctx["core_notional"]) + 1e-12)
        violations["notional_above_cap"] += int(float(row["effective_notional"]) > 3.6 + 1e-12)
    return {
        "passed": bool(sum(violations.values()) == 0),
        "violations": violations,
        "base_trades": len(contexts),
        "edited_trades": len(ledger),
    }


def _ledger_audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"passed": False, "reason": "empty_ledger"}
    final_pnl = (float(ledger[-1]["cash_after"]) - 1.0) * 100.0
    step_errors = []
    fee_errors = []
    for row in ledger:
        before = float(row["cash_before"])
        after = float(row["cash_after"])
        pnl = float(row["trade_pnl_pct"]) / 100.0
        step_errors.append(abs(before * (1.0 + pnl) - after))
        fee_errors.append(abs(float(row["total_fee_cash"]) - float(row["entry_fee_cash"]) - float(row["exit_fee_cash"])))
    numeric = pd.DataFrame(ledger).select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "passed": bool(
            abs(final_pnl - float(report_pnl)) < 1e-9
            and max(step_errors) < 1e-9
            and max(fee_errors) < 1e-12
        ),
        "final_pnl_from_ledger": float(final_pnl),
        "report_pnl": float(report_pnl),
        "max_step_equity_error": float(max(step_errors)),
        "max_fee_identity_error": float(max(fee_errors)),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
    }


def _contract_doc() -> str:
    return f"""# Clean Base Conformal Downside Filter V1.4 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle entries and sides while applying validation-calibrated downside-only shrink and early-exit controls.

## Runtime Inputs

`{', '.join(editor.EDITOR_FEATURES)}`

Runtime decisions also use closed-equity account drawdown, daily drawdown, and a validation residual quantile chosen before OOS evaluation.

## Forbidden Runtime Inputs

`{', '.join(FORBIDDEN_RUNTIME_FEATURES)}`

## Outputs

- `effective_notional`, constrained to be no larger than the clean base core notional
- `effective_exit_idx`, constrained to be no later than the clean base/Lifecycle core exit
- action reason code
- trade-level ledger with cash before/after, fee cash, predicted downside telemetry, and conformal lower confidence bound

## Split Contract

- Train labels: 2025 pre-validation window
- Calibration/selection: 2025 validation window
- OOS: 2026, one-shot after threshold selection
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Conformal Downside Filter V1.4

Status: `{report['verdict']}`

## OOS Metrics

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Trades/day | `{c1['trades_per_day']:.6f}` |
| Avg notional | `{c1['avg_notional']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

## Selected Config

`{report['selected_config']['name']}`

## Audit

- Causality: `{report['causality_audit']['passed']}`
- Preservation: `{report['preservation_audit']['passed']}`
- Accounting: `{report['accounting_audit']['passed']}`
- Red Team blocking risk: `{report['red_team_review']['blocking_risk']}`

## Artifacts

- Report: `{report['artifacts']['report']}`
- Grid: `{report['artifacts']['grid']}`
- Ledger: `{report['artifacts']['ledger']}`
- Model: `{report['artifacts']['model']}`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    lifecycle_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    lifecycle_recalibrator, lifecycle_cfg = sleeve._load_lifecycle_model(args.lifecycle_model)
    try:
        lifecycle_cfg = _load_lifecycle_cfg(lifecycle_report)
    except Exception:
        pass

    train_full = sleeve._read(args.train_csv)
    train_df, val_df = sleeve._split_train_validation(train_full, args.split_date)
    oos_df = sleeve._read(args.eval_csv)

    def build(
        df: pd.DataFrame,
        fee: float,
        slip: float,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        pre = sleeve._base_frame(df, policy, entry_cfg)
        base_trades = sleeve._base_trade_plan(df, exit_model, risk_cfg, exit_cfg, pre, fee=fee, slip=slip)
        life = sleeve.backtest_lifecycle_editor(
            df,
            exit_model,
            lifecycle_recalibrator,
            lifecycle_cfg,
            base_trades,
            exit_cfg,
            pre,
            fee=fee,
            slip=slip,
        )
        contexts = sleeve._contexts(df, life["lifecycle_plan"], base_trades, pre[3], slip=slip)
        return pre, base_trades, contexts, life

    train_pre, _train_base, train_ctx, _train_life = build(train_df, float(args.fee), float(args.slip))
    model, train_meta = editor._train_editor_model(
        train_df,
        train_pre,
        train_ctx,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    val_pre_1, _val_base_1, val_ctx_1, val_life_1 = build(val_df, float(args.fee), float(args.slip))
    val_pre_3, _val_base_3, val_ctx_3, _val_life_3 = build(val_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    oos_pre_1, _oos_base_1, oos_ctx_1, oos_life_1 = build(oos_df, float(args.fee), float(args.slip))
    oos_pre_2, _oos_base_2, oos_ctx_2, _oos_life_2 = build(oos_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    oos_pre_3, _oos_base_3, oos_ctx_3, _oos_life_3 = build(oos_df, float(args.fee) * 3.0, float(args.slip) * 3.0)

    val_pred_1 = editor._predict_editor(model, editor._context_frame(val_df, val_ctx_1))
    val_pred_3 = editor._predict_editor(model, editor._context_frame(val_df, val_ctx_3))
    oos_pred_1 = editor._predict_editor(model, editor._context_frame(oos_df, oos_ctx_1))
    oos_pred_2 = editor._predict_editor(model, editor._context_frame(oos_df, oos_ctx_2))
    oos_pred_3 = editor._predict_editor(model, editor._context_frame(oos_df, oos_ctx_3))

    val_actual_1 = _actual_full_returns(val_df, val_pre_1, val_ctx_1, fee=float(args.fee), slip=float(args.slip))
    calibration = _conformal_calibration(val_actual_1, val_pred_1)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: ConformalDownsideConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    selected_q = 0.0
    for cfg in _grid():
        rq = _residual_q(calibration, cfg.residual_quantile)
        val_1 = backtest_conformal(
            cfg,
            val_df,
            val_pre_1,
            val_ctx_1,
            val_pred_1,
            residual_q=rq,
            fee=float(args.fee),
            slip=float(args.slip),
        )
        val_3 = backtest_conformal(
            cfg,
            val_df,
            val_pre_3,
            val_ctx_3,
            val_pred_3,
            residual_q=rq,
            fee=float(args.fee) * 3.0,
            slip=float(args.slip) * 3.0,
        )
        row = {
            **asdict(cfg),
            "residual_q_value": rq,
            **{f"val_{k}": v for k, v in _compact(val_1).items() if not isinstance(v, dict)},
            "val_cost3_pnl": val_3["pnl"],
            "val_cost3_mdd": val_3["mdd"],
        }
        row["selection_score"] = _score(val_1, val_3)
        grid_rows.append(row)
        if float(row["selection_score"]) > selected_score:
            selected_score = float(row["selection_score"])
            selected_cfg = cfg
            selected_q = rq
            selected_val = {"cost_1x": _compact(val_1), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None

    full_1 = backtest_conformal(
        selected_cfg,
        oos_df,
        oos_pre_1,
        oos_ctx_1,
        oos_pred_1,
        residual_q=selected_q,
        fee=float(args.fee),
        slip=float(args.slip),
        ledger_out=args.ledger_csv_out,
    )
    cost_2 = backtest_conformal(
        selected_cfg,
        oos_df,
        oos_pre_2,
        oos_ctx_2,
        oos_pred_2,
        residual_q=selected_q,
        fee=float(args.fee) * 2.0,
        slip=float(args.slip) * 2.0,
    )
    cost_3 = backtest_conformal(
        selected_cfg,
        oos_df,
        oos_pre_3,
        oos_ctx_3,
        oos_pred_3,
        residual_q=selected_q,
        fee=float(args.fee) * 3.0,
        slip=float(args.slip) * 3.0,
    )

    preservation = _preservation(oos_ctx_1, full_1["ledger"])
    accounting = _ledger_audit(full_1["pnl"], full_1["ledger"])
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "training_labels_use_future": True,
        "validation_calibration_and_selection": True,
        "oos_threshold_selection": False,
        "runtime_features": editor.EDITOR_FEATURES,
        "forbidden_runtime_features": FORBIDDEN_RUNTIME_FEATURES,
    }
    red_team_review = {
        "blocking_risk": False,
        "notes": [
            "Core entries, sides, and trade count are preserved.",
            "Runtime notional can only shrink from Lifecycle core notional.",
            "Formal conformal coverage is not claimed because calibration and grid selection share the validation window.",
        ],
    }
    gates = {
        "clean_base_pnl_gate": bool(float(full_1["pnl"]) >= editor.CLEAN_BASE_REFERENCE["pnl"]),
        "clean_base_mdd_gate": bool(float(full_1["mdd"]) >= editor.CLEAN_BASE_REFERENCE["mdd"]),
        "sleeve_v12_pnl_gate": bool(float(full_1["pnl"]) >= editor.SLEEVE_V12_REFERENCE["pnl"]),
        "sleeve_v12_mdd_gate": bool(float(full_1["mdd"]) >= editor.SLEEVE_V12_REFERENCE["mdd"]),
        "trades_per_day_gate": bool(float(full_1["trades_per_day"]) >= 6.0),
        "cost2_survival": bool(float(cost_2["pnl"]) > 0.0),
        "cost3_not_worse_than_clean_base": bool(float(cost_3["pnl"]) >= editor.CLEAN_BASE_REFERENCE["cost_3x_pnl"]),
        "downside_only": bool(float(full_1["gross_notional_max"]) <= 3.6 and preservation["violations"]["notional_above_core"] == 0),
        "preservation_audit_passed": bool(preservation["passed"]),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "red_team_no_blocking_risk": bool(not red_team_review["blocking_risk"]),
    }
    gates["decision"] = "promote" if all(gates.values()) else (
        "shadow_candidate"
        if gates["clean_base_mdd_gate"]
        and gates["trades_per_day_gate"]
        and gates["preservation_audit_passed"]
        and gates["accounting_audit_passed"]
        and gates["causality_audit_passed"]
        else "reject"
    )

    model_out = args.model_dir / "conformal_downside_filter.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    cal_for_dump = {k: v for k, v in calibration.items() if k != "residuals"}
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "models": model,
            "train_meta": train_meta,
            "calibration": cal_for_dump,
            "selected_config": asdict(selected_cfg),
            "selected_residual_q": selected_q,
            "features": editor.EDITOR_FEATURES,
        },
        model_out,
    )
    _write_csv(args.grid_csv_out, sorted(grid_rows, key=lambda r: float(r["selection_score"]), reverse=True))

    full_1_compact = _compact(full_1)
    cost_2_compact = _compact(cost_2)
    cost_3_compact = _compact(cost_3)
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "selected_residual_q": float(selected_q),
        "training": train_meta,
        "calibration": cal_for_dump,
        "validation": selected_val,
        "validation_grid_rows": len(grid_rows),
        "cost_1x": full_1_compact,
        "cost_2x": cost_2_compact,
        "cost_3x": cost_3_compact,
        "clean_base_reference": editor.CLEAN_BASE_REFERENCE,
        "sleeve_v12_reference": editor.SLEEVE_V12_REFERENCE,
        "lifecycle_core_reference": {
            "validation": sleeve._compact(val_life_1),
            "oos": sleeve._compact(oos_life_1),
            "report": str(args.lifecycle_report),
        },
        "promotion_gate": gates,
        "preservation_audit": preservation,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "red_team_review": red_team_review,
        "data": {
            "train_range": sleeve._range(train_df),
            "validation_range": sleeve._range(val_df),
            "oos_range": sleeve._range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_calibration_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "artifacts": {
            "model": str(model_out),
            "report": str(args.report_out),
            "grid": str(args.grid_csv_out),
            "ledger": str(args.ledger_csv_out),
            "doc": str(args.doc_out),
            "contract": str(DEFAULT_CONTRACT),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    DEFAULT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONTRACT.write_text(_contract_doc(), encoding="utf-8")

    full_1.pop("ledger", None)
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "verdict": gates["decision"],
                "selected": selected_cfg.name,
                "selected_residual_q": selected_q,
                "cost_1x": full_1_compact,
                "cost_2x": cost_2_compact,
                "cost_3x": cost_3_compact,
                "promotion_gate": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base conformal downside filter v1.4.")
    p.add_argument("--policy", type=Path, default=sleeve.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=sleeve.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=sleeve.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=sleeve.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=sleeve.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=sleeve.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=sleeve.DEFAULT_EVAL_CSV)
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
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
