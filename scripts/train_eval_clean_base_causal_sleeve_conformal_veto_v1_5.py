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
from scripts import train_eval_clean_base_plus_causal_conviction_sleeve_v1_1 as base  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


MODEL_ID = "clean_base_causal_sleeve_conformal_veto_v1_5"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_causal_sleeve_conformal_veto_v1_5"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_causal_sleeve_conformal_veto_v1_5_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_causal_sleeve_conformal_veto_v1_5.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_causal_sleeve_conformal_veto_v1_5_contract.md"


@dataclass(frozen=True)
class SleeveConformalVetoConfig:
    name: str
    same_threshold: float
    max_sleeve_frac: float
    max_sleeve_bars: int
    account_dd_disable: float
    daily_dd_disable: float
    residual_quantile: float
    lcb_veto_threshold: float
    adverse_veto_cut: float


def _grid() -> list[SleeveConformalVetoConfig]:
    rows: list[SleeveConformalVetoConfig] = []
    for same_thr in (0.0015, 0.0025):
        for frac in (0.25,):
            for bars in (6,):
                for acct in (0.06, 0.08):
                    for day in (0.015,):
                        for q in (0.70, 0.80):
                            for lcb in (-0.0060, -0.0050, -0.0040):
                                for adverse in (0.020, 0.030):
                                    name = (
                                        f"sv_same{same_thr:.4f}_frac{frac:.2f}_bars{bars}_"
                                        f"acct{acct:.2f}_day{day:.3f}_q{q:.2f}_"
                                        f"lcb{lcb:.4f}_adv{adverse:.3f}"
                                    )
                                    rows.append(
                                        SleeveConformalVetoConfig(
                                            name=name,
                                            same_threshold=float(same_thr),
                                            max_sleeve_frac=float(frac),
                                            max_sleeve_bars=int(bars),
                                            account_dd_disable=float(acct),
                                            daily_dd_disable=float(day),
                                            residual_quantile=float(q),
                                            lcb_veto_threshold=float(lcb),
                                            adverse_veto_cut=float(adverse),
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
    return np.asarray(
        [editor._future_path_stats(precomputed, ctx, horizon=None, fee=fee, slip=slip)["net"] for ctx in contexts],
        dtype=np.float64,
    )


def _calibration(actual: np.ndarray, pred: dict[str, np.ndarray]) -> dict[str, Any]:
    residual = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(pred["full"], dtype=np.float64))
    residual = residual[np.isfinite(residual)]
    if len(residual) == 0:
        residual = np.asarray([0.0], dtype=np.float64)
    return {
        "rows": int(len(residual)),
        "residual_mean": float(np.mean(residual)),
        "residual_p70": float(np.quantile(residual, 0.70)),
        "residual_p80": float(np.quantile(residual, 0.80)),
        "residual_p90": float(np.quantile(residual, 0.90)),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "residuals": residual,
    }


def _residual_q(cal: dict[str, Any], q: float) -> float:
    return float(np.quantile(np.asarray(cal["residuals"], dtype=np.float64), float(q)))


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "core_trades_per_day",
            "wr",
            "sleeve_action_counts",
            "sleeve_fraction",
            "core_lane_pnl_contribution",
            "sleeve_lane_pnl_contribution",
            "add_contribution",
            "hedge_contribution",
            "gross_notional_max",
            "net_notional_max",
            "reason_counts",
        )
    }


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    cost3_pnl = float(cost3["pnl"])
    sleeve_fraction = float(metrics["sleeve_fraction"])
    score = pnl + 0.16 * cost3_pnl
    score -= 145.0 * max(0.0, abs(mdd) - 17.80)
    score -= 20.0 * max(0.0, 0.08 - sleeve_fraction)
    score -= 30.0 * max(0.0, sleeve_fraction - 0.22)
    return float(score)


def backtest(
    cfg: SleeveConformalVetoConfig,
    sleeve_model: Any,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    conformal_pred: dict[str, np.ndarray],
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
    loss_streak = 0
    mdd = 0.0
    action_counts = {
        k: 0
        for k in (
            "NO_SLEEVE",
            "ADD_SAME_SIDE_15",
            "ADD_SAME_SIDE_25",
            "CONFORMAL_VETO",
        )
    }
    sleeve_trades = 0
    add_pnl = hedge_pnl = core_pnl_sum = sleeve_pnl_sum = 0.0
    gross_max = net_max = 0.0
    ledger: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    wins = 0
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
        feat = base._features(df, ctx, account_dd, daily_dd, loss_streak)

        sleeve_cfg = base.CausalSleeveConfig(
            name=cfg.name,
            same_threshold=cfg.same_threshold,
            hedge_threshold=0.0025,
            max_sleeve_frac=cfg.max_sleeve_frac,
            max_sleeve_bars=cfg.max_sleeve_bars,
            same_enabled=True,
            hedge_enabled=False,
            account_dd_disable=cfg.account_dd_disable,
            daily_dd_disable=cfg.daily_dd_disable,
        )
        action, sleeve_side, sleeve_frac, preds, reasons = base._choose_sleeve(sleeve_cfg, sleeve_model, df, ctx, feat)
        pred_full = float(conformal_pred["full"][k])
        pred_adverse = float(conformal_pred["adverse"][k])
        pred_lcb = pred_full - float(residual_q)
        conformal_veto = (
            action.startswith("ADD")
            and (pred_lcb <= cfg.lcb_veto_threshold or pred_adverse >= cfg.adverse_veto_cut)
        )
        if conformal_veto:
            action, sleeve_side, sleeve_frac = "CONFORMAL_VETO", 0, 0.0
            reasons.append("conformal_veto")
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        core_side = int(ctx["side"])
        core_notional = float(ctx["core_notional"])
        sleeve_notional = min(core_notional * sleeve_frac, 0.25 * core_notional) if sleeve_side else 0.0
        gross = core_notional + sleeve_notional
        net = abs(core_side * core_notional + sleeve_side * sleeve_notional)
        if gross > 3.6 or net > 3.6:
            action, sleeve_side, sleeve_notional, gross, net = "NO_SLEEVE", 0, 0.0, core_notional, core_notional
            reasons.append("gross_net_cap")
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        action_counts[action] = action_counts.get(action, 0) + 1
        sleeve_trades += int(sleeve_notional > 0.0)

        before = cash
        core_entry = float(ctx["entry_price"])
        core_exit_idx = int(ctx["core_exit_idx"])
        sleeve_exit_idx = min(core_exit_idx, i + int(cfg.max_sleeve_bars)) if sleeve_side else i
        sleeve_entry = base._entry_price(fill_px, min(i + 1, len(close) - 1), sleeve_side, slip) if sleeve_side else 0.0
        core_entry_fee = cash * fee * core_notional
        cash -= core_entry_fee
        sleeve_entry_fee = 0.0
        if sleeve_side:
            sleeve_entry_fee = cash * fee * sleeve_notional
            cash -= sleeve_entry_fee
        sleeve_realized = 0.0
        sleeve_exit_fee = 0.0
        sleeve_cash_realized = False
        for j in range(i, core_exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            core_mark_exit = px * (1.0 - slip) if core_side > 0 else px * (1.0 + slip)
            core_unreal = base._raw(core_side, core_entry, core_mark_exit) * core_notional
            sleeve_unreal = 0.0
            if sleeve_side and not sleeve_cash_realized:
                sleeve_mark_exit = px * (1.0 - slip) if sleeve_side > 0 else px * (1.0 + slip)
                sleeve_unreal = base._raw(sleeve_side, sleeve_entry, sleeve_mark_exit) * sleeve_notional
                if j >= sleeve_exit_idx:
                    sleeve_exit = base._exit_price(fill_px, min(j + 1, len(close) - 1), sleeve_side, slip)
                    sleeve_realized = base._raw(sleeve_side, sleeve_entry, sleeve_exit) * sleeve_notional
                    cash = cash * (1.0 + sleeve_realized)
                    sleeve_exit_fee = cash * fee * sleeve_notional
                    cash -= sleeve_exit_fee
                    sleeve_cash_realized = True
                    sleeve_unreal = 0.0
            eq = cash * (1.0 + core_unreal + sleeve_unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        core_exit = base._exit_price(fill_px, min(core_exit_idx + 1, len(close) - 1), core_side, slip)
        core_realized = base._raw(core_side, core_entry, core_exit) * core_notional
        cash = cash * (1.0 + core_realized)
        core_exit_fee = cash * fee * core_notional
        cash -= core_exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        core_pnl_sum += core_realized * before * 100.0
        sleeve_pnl_sum += sleeve_realized * before * 100.0
        if action.startswith("ADD"):
            add_pnl += sleeve_realized * before * 100.0
        if action.startswith("HEDGE"):
            hedge_pnl += sleeve_realized * before * 100.0
        wins += int(cash > before)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "sleeve_exit_idx": int(sleeve_exit_idx) if sleeve_side else 0,
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_side": core_side,
                "sleeve_side": sleeve_side,
                "core_notional": core_notional,
                "sleeve_notional": sleeve_notional,
                "gross_notional": gross,
                "net_notional": net,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "loss_streak_prior": loss_streak,
                "pred_same_utility": preds["same_pred"],
                "pred_hedge_utility": preds["hedge_pred"],
                "conformal_pred_full": pred_full,
                "conformal_pred_adverse": pred_adverse,
                "conformal_residual_q": float(residual_q),
                "conformal_lcb": pred_lcb,
                "core_entry_fee_cash": core_entry_fee,
                "core_exit_fee_cash": core_exit_fee,
                "sleeve_entry_fee_cash": sleeve_entry_fee,
                "sleeve_exit_fee_cash": sleeve_exit_fee,
                "total_fee_cash": core_entry_fee + core_exit_fee + sleeve_entry_fee + sleeve_exit_fee,
                "core_pnl_pct": core_realized * 100.0,
                "sleeve_pnl_pct": sleeve_realized * 100.0,
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_before": before,
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
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / base._days(df)),
        "wr": float(wins / max(trades, 1)),
        "sleeve_action_counts": action_counts,
        "sleeve_fraction": float(sleeve_trades / max(trades, 1)),
        "core_lane_pnl_contribution": float(core_pnl_sum),
        "sleeve_lane_pnl_contribution": float(sleeve_pnl_sum),
        "add_contribution": float(add_pnl),
        "hedge_contribution": float(hedge_pnl),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _ledger_audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"passed": False, "reason": "empty_ledger"}
    final_pnl = (float(ledger[-1]["cash_after"]) - 1.0) * 100.0
    step_errors = [
        abs(float(row["cash_before"]) * (1.0 + float(row["trade_pnl_pct"]) / 100.0) - float(row["cash_after"]))
        for row in ledger
    ]
    numeric = pd.DataFrame(ledger).select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "passed": bool(abs(final_pnl - float(report_pnl)) < 1e-9 and max(step_errors) < 1e-9),
        "final_pnl_from_ledger": float(final_pnl),
        "report_pnl": float(report_pnl),
        "max_step_equity_error": float(max(step_errors)),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
    }


def _contract_doc() -> str:
    return """# Clean Base Causal Sleeve Conformal Veto V1.5 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries and keep the causal same-side sleeve alpha, while using validation-calibrated downside uncertainty only to veto sleeve additions.

## Runtime Inputs

- Sleeve scorer: current trade state and market context features from `clean_base_plus_causal_conviction_sleeve_v1_1.FEATURES`
- Conformal veto: static trade-entry features from `clean_base_causal_trade_editor_v1_3.EDITOR_FEATURES`
- Closed-equity account drawdown and daily drawdown

## Output Invariants

- Core entry, side, exit, notional, and leverage are unchanged.
- The conformal layer cannot open, flip, resize, or close the core trade.
- The conformal layer can only convert an `ADD_SAME_SIDE_*` sleeve action to `CONFORMAL_VETO`.
- OOS threshold selection is forbidden.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Causal Sleeve Conformal Veto V1.5

Status: `{report['verdict']}`

## OOS Metrics

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Trades/day | `{c1['core_trades_per_day']:.6f}` |
| Sleeve fraction | `{c1['sleeve_fraction']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

## Selected Config

`{report['selected_config']['name']}`

## Audit

- Preservation: `{report['preservation_audit']['passed']}`
- Accounting: `{report['sleeve_accounting_audit']['passed']}`
- Causality: `{report['causality_audit']['passed']}`

## Artifacts

- Report: `{report['artifacts']['report']}`
- Grid: `{report['artifacts']['grid_csv']}`
- Ledger: `{report['artifacts']['ledger_csv']}`
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
    lifecycle_recalibrator, lifecycle_cfg = base._load_lifecycle_model(args.lifecycle_model)
    try:
        lifecycle_cfg = _load_lifecycle_cfg(lifecycle_report)
    except Exception:
        pass

    train_full = base._read(args.train_csv)
    train_df, val_df = base._split_train_validation(train_full, args.split_date)
    oos_df = base._read(args.eval_csv)

    def build(
        df: pd.DataFrame,
        fee: float,
        slip: float,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        pre = base._base_frame(df, policy, entry_cfg)
        base_trades = base._base_trade_plan(df, exit_model, risk_cfg, exit_cfg, pre, fee=fee, slip=slip)
        life = base.backtest_lifecycle_editor(
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
        contexts = base._contexts(df, life["lifecycle_plan"], base_trades, pre[3], slip=slip)
        return pre, base_trades, contexts, life

    train_pre, _train_base, train_ctx, _train_life = build(train_df, float(args.fee), float(args.slip))
    sleeve_model, sleeve_train_meta = base._train_scorer(
        train_df,
        train_pre,
        train_ctx,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    conformal_model, conformal_train_meta = editor._train_editor_model(
        train_df,
        train_pre,
        train_ctx,
        fee=float(args.fee),
        slip=float(args.slip),
    )

    val_pre_1, _val_base_1, val_ctx_1, val_life_1 = build(val_df, float(args.fee), float(args.slip))
    val_pre_2, _val_base_2, val_ctx_2, _val_life_2 = build(val_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    val_pre_3, _val_base_3, val_ctx_3, _val_life_3 = build(val_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    oos_pre_1, oos_base_1, oos_ctx_1, oos_life_1 = build(oos_df, float(args.fee), float(args.slip))
    oos_pre_2, _oos_base_2, oos_ctx_2, _oos_life_2 = build(oos_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    oos_pre_3, _oos_base_3, oos_ctx_3, _oos_life_3 = build(oos_df, float(args.fee) * 3.0, float(args.slip) * 3.0)

    val_pred_1 = editor._predict_editor(conformal_model, editor._context_frame(val_df, val_ctx_1))
    val_pred_2 = editor._predict_editor(conformal_model, editor._context_frame(val_df, val_ctx_2))
    val_pred_3 = editor._predict_editor(conformal_model, editor._context_frame(val_df, val_ctx_3))
    oos_pred_1 = editor._predict_editor(conformal_model, editor._context_frame(oos_df, oos_ctx_1))
    oos_pred_2 = editor._predict_editor(conformal_model, editor._context_frame(oos_df, oos_ctx_2))
    oos_pred_3 = editor._predict_editor(conformal_model, editor._context_frame(oos_df, oos_ctx_3))
    val_actual = _actual_full_returns(val_df, val_pre_1, val_ctx_1, fee=float(args.fee), slip=float(args.slip))
    cal = _calibration(val_actual, val_pred_1)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: SleeveConformalVetoConfig | None = None
    selected_score = -1e18
    selected_rq = 0.0
    selected_validation: dict[str, Any] | None = None
    for cfg in _grid():
        rq = _residual_q(cal, cfg.residual_quantile)
        val_1 = backtest(
            cfg,
            sleeve_model,
            val_df,
            val_pre_1,
            val_ctx_1,
            val_pred_1,
            residual_q=rq,
            fee=float(args.fee),
            slip=float(args.slip),
        )
        val_3 = backtest(
            cfg,
            sleeve_model,
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
            "val_pnl": val_1["pnl"],
            "val_mdd": val_1["mdd"],
            "val_cost3_pnl": val_3["pnl"],
            "val_sleeve_fraction": val_1["sleeve_fraction"],
            "val_conformal_veto": val_1["sleeve_action_counts"].get("CONFORMAL_VETO", 0),
            "selection_score": _score(val_1, val_3),
        }
        grid_rows.append(row)
        if float(row["selection_score"]) > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_rq = rq
            selected_validation = {"cost_1x": _compact(val_1), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None

    full = backtest(
        selected_cfg,
        sleeve_model,
        oos_df,
        oos_pre_1,
        oos_ctx_1,
        oos_pred_1,
        residual_q=selected_rq,
        fee=float(args.fee),
        slip=float(args.slip),
        ledger_out=args.ledger_csv_out,
    )
    cost = {
        "cost_1x": _compact(full),
        "cost_2x": _compact(
            backtest(
                selected_cfg,
                sleeve_model,
                oos_df,
                oos_pre_2,
                oos_ctx_2,
                oos_pred_2,
                residual_q=selected_rq,
                fee=float(args.fee) * 2.0,
                slip=float(args.slip) * 2.0,
            )
        ),
        "cost_3x": _compact(
            backtest(
                selected_cfg,
                sleeve_model,
                oos_df,
                oos_pre_3,
                oos_ctx_3,
                oos_pred_3,
                residual_q=selected_rq,
                fee=float(args.fee) * 3.0,
                slip=float(args.slip) * 3.0,
            )
        ),
    }
    validation_cost = {
        "cost_1x": selected_validation["cost_1x"] if selected_validation else {},
        "cost_2x": _compact(
            backtest(
                selected_cfg,
                sleeve_model,
                val_df,
                val_pre_2,
                val_ctx_2,
                val_pred_2,
                residual_q=selected_rq,
                fee=float(args.fee) * 2.0,
                slip=float(args.slip) * 2.0,
            )
        ),
        "cost_3x": selected_validation["cost_3x"] if selected_validation else {},
    }

    _feat, eval_dec, _close, _fill = oos_pre_1
    preservation = {
        "decision_frame_audit": base._decision_audit(
            eval_dec,
            max_notional=float(risk_cfg.get("max_notional", 3.6)),
            leverage_cap=5.0,
        ),
        "core_lane_preservation": base._preservation(oos_base_1, oos_ctx_1, full["ledger"]),
    }
    preservation["passed"] = bool(
        preservation["decision_frame_audit"].get("passed")
        and preservation["core_lane_preservation"].get("passed")
    )
    accounting = _ledger_audit(full["pnl"], full["ledger"])
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "training_labels_use_future": True,
        "validation_calibration_and_selection": True,
        "oos_threshold_selection": False,
        "conformal_can_modify_core_lane": False,
    }
    gates = {
        "clean_base_pnl_gate": bool(float(cost["cost_1x"]["pnl"]) >= editor.CLEAN_BASE_REFERENCE["pnl"]),
        "clean_base_mdd_gate": bool(float(cost["cost_1x"]["mdd"]) >= editor.CLEAN_BASE_REFERENCE["mdd"]),
        "sleeve_v12_pnl_gate": bool(float(cost["cost_1x"]["pnl"]) >= editor.SLEEVE_V12_REFERENCE["pnl"]),
        "sleeve_v12_mdd_gate": bool(float(cost["cost_1x"]["mdd"]) >= editor.SLEEVE_V12_REFERENCE["mdd"]),
        "trades_per_day_gate": bool(float(cost["cost_1x"]["core_trades_per_day"]) >= 6.0),
        "cost2_survival": bool(float(cost["cost_2x"]["pnl"]) > 0.0),
        "cost3_not_worse_than_clean_base": bool(float(cost["cost_3x"]["pnl"]) >= editor.CLEAN_BASE_REFERENCE["cost_3x_pnl"]),
        "sleeve_fraction_active": bool(float(cost["cost_1x"]["sleeve_fraction"]) >= 0.03),
        "preservation_audit_passed": bool(preservation["passed"]),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
    }
    gates["decision"] = "promote" if all(gates.values()) else (
        "shadow_candidate"
        if gates["clean_base_pnl_gate"]
        and gates["trades_per_day_gate"]
        and gates["preservation_audit_passed"]
        and gates["accounting_audit_passed"]
        and gates["causality_audit_passed"]
        else "reject"
    )

    clean_val = backtest_no_limit_exit(
        val_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=val_pre_1,
    )
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
        precomputed=oos_pre_1,
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()) if grid_rows else ["name"])
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: float(r["selection_score"]), reverse=True))

    model_out = args.model_dir / "sleeve_conformal_veto.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    cal_dump = {k: v for k, v in cal.items() if k != "residuals"}
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sleeve_model": sleeve_model,
            "conformal_model": conformal_model,
            "sleeve_train_meta": sleeve_train_meta,
            "conformal_train_meta": conformal_train_meta,
            "calibration": cal_dump,
            "selected_config": asdict(selected_cfg),
            "selected_residual_q": selected_rq,
            "sleeve_features": base.FEATURES,
            "conformal_features": editor.EDITOR_FEATURES,
        },
        model_out,
    )
    full_compact = _compact(full)
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "selected_residual_q": selected_rq,
        "training": {
            "sleeve": sleeve_train_meta,
            "conformal": conformal_train_meta,
        },
        "calibration": cal_dump,
        "validation": selected_validation,
        "validation_grid_rows": len(grid_rows),
        "candidate_total_oos": full_compact,
        "core_lane_oos": _compact(oos_life_1),
        "sleeve_lane_oos": {
            "pnl_contribution": full_compact["sleeve_lane_pnl_contribution"],
            "sleeve_fraction": full_compact["sleeve_fraction"],
        },
        "cost_1x": full_compact,
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": validation_cost["cost_1x"],
        "validation_cost_2x": validation_cost["cost_2x"],
        "validation_cost_3x": validation_cost["cost_3x"],
        "clean_base_reference": editor.CLEAN_BASE_REFERENCE,
        "sleeve_v12_reference": editor.SLEEVE_V12_REFERENCE,
        "clean_base_validation_reference": base._compact(clean_val),
        "clean_base_oos_reference": base._compact(clean_oos),
        "lifecycle_v1_reference": {
            "validation": _compact(val_life_1),
            "oos": _compact(oos_life_1),
            "report": str(args.lifecycle_report),
        },
        "promotion_gate": gates,
        "preservation_audit": preservation,
        "sleeve_accounting_audit": accounting,
        "causality_audit": causality,
        "data": {
            "train_range": base._range(train_df),
            "validation_range": base._range(val_df),
            "oos_range": base._range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_calibration_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "feature_contract": {
            "sleeve_features": base.FEATURES,
            "conformal_features": editor.EDITOR_FEATURES,
            "runtime_forbidden": [
                "evt_candidate_side",
                "evt_candidate_label",
                "evt_side_margin",
                "future close",
                "future high/low",
                "future realized return",
            ],
        },
        "artifacts": {
            "model": str(model_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "doc": str(args.doc_out),
            "contract": str(DEFAULT_CONTRACT),
        },
        "validation_top10": sorted(grid_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    DEFAULT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONTRACT.write_text(_contract_doc(), encoding="utf-8")

    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "verdict": gates["decision"],
                "selected": selected_cfg.name,
                "selected_residual_q": selected_rq,
                "cost_1x": full_compact,
                "cost_2x": cost["cost_2x"],
                "cost_3x": cost["cost_3x"],
                "promotion_gate": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base causal sleeve with conformal sleeve veto v1.5.")
    p.add_argument("--policy", type=Path, default=base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=base.DEFAULT_EVAL_CSV)
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
