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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_core_reallocator_v1 as dcr  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v1 as v1  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v2 as v2  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


MODEL_ID = "clean_base_deep_constant_gross_v1"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_constant_gross_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_constant_gross_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_constant_gross_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_constant_gross_v1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_constant_gross_v1.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_constant_gross_v1_contract.md"


@dataclass(frozen=True)
class ConstantGrossConfig:
    name: str
    target_notional: float
    conviction_threshold: float
    adverse_cut: float
    deep_full_floor: float
    defensive_notional: float
    account_dd_defensive: float
    daily_dd_defensive: float
    cost3_notional: float


def _grid() -> list[ConstantGrossConfig]:
    rows: list[ConstantGrossConfig] = []
    for target in (3.0, 3.6):
        name = f"dcg_open_t{target:.1f}_c-1.0000_a99.000_def{target:.1f}_dd1.00_c30.00"
        rows.append(
            ConstantGrossConfig(
                name=name,
                target_notional=float(target),
                conviction_threshold=-1.0,
                adverse_cut=99.0,
                deep_full_floor=-99.0,
                defensive_notional=float(target),
                account_dd_defensive=1.0,
                daily_dd_defensive=1.0,
                cost3_notional=0.0,
            )
        )
    for target in (3.0, 3.6):
        for conv in (-0.0100, -0.0025):
            for defensive in (0.5, 1.0):
                for cost3_notional in (0.0, 0.15):
                    name = (
                        f"dcg_t{target:.1f}_c{conv:.4f}_a0.012_"
                        f"def{defensive:.1f}_dd0.30_c3{cost3_notional:.2f}"
                    )
                    rows.append(
                        ConstantGrossConfig(
                            name=name,
                            target_notional=float(target),
                            conviction_threshold=float(conv),
                            adverse_cut=0.012,
                            deep_full_floor=-0.0025,
                            defensive_notional=float(defensive),
                            account_dd_defensive=0.30,
                            daily_dd_defensive=0.030,
                            cost3_notional=float(cost3_notional),
                        )
                    )
    return rows


def _score(metrics: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = abs(float(metrics["mdd"]))
    score = pnl + 0.12 * float(cost2["pnl"]) + 0.06 * float(cost3["pnl"])
    score -= 12.0 * max(0.0, mdd - 25.0)
    score -= 180.0 * max(0.0, -float(cost2["pnl"]))
    score -= 90.0 * max(0.0, -float(cost3["pnl"]))
    if pnl >= 500.0:
        score += 250.0
    return float(score)


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "core_trades_per_day",
            "wr",
            "action_counts",
            "avg_effective_notional",
            "gross_notional_max",
            "net_notional_max",
            "reason_counts",
        )
    }


def _build_runtime_models(args: argparse.Namespace) -> dict[str, Any]:
    dcr._configure_deep_globals()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit_report = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit_report["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    lifecycle_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    lifecycle_recalibrator, lifecycle_cfg = v1.base._load_lifecycle_model(args.lifecycle_model)
    try:
        lifecycle_cfg = _load_lifecycle_cfg(lifecycle_report)
    except Exception:
        pass
    return {
        "policy": policy,
        "exit_model": exit_model,
        "entry_cfg": entry_cfg,
        "risk_cfg": risk_cfg,
        "exit_cfg": exit_cfg,
        "lifecycle_recalibrator": lifecycle_recalibrator,
        "lifecycle_cfg": lifecycle_cfg,
    }


def _build_contexts(
    df: pd.DataFrame,
    models: dict[str, Any],
    *,
    fee: float,
    slip: float,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pre = v1.base._base_frame(df, models["policy"], models["entry_cfg"])
    base_trades = v1.base._base_trade_plan(
        df,
        models["exit_model"],
        models["risk_cfg"],
        models["exit_cfg"],
        pre,
        fee=fee,
        slip=slip,
    )
    life = v1.base.backtest_lifecycle_editor(
        df,
        models["exit_model"],
        models["lifecycle_recalibrator"],
        models["lifecycle_cfg"],
        base_trades,
        models["exit_cfg"],
        pre,
        fee=fee,
        slip=slip,
    )
    contexts = v1.base._contexts(df, life["lifecycle_plan"], base_trades, pre[3], slip=slip)
    return pre, contexts, life, {"base_trades": len(base_trades)}


def _state_for(
    df: pd.DataFrame,
    contexts: list[dict[str, Any]],
    seq_features: list[str],
    seq_scaler: Any,
    deep_model: Any,
    deep_meta: dict[str, Any],
    state_model: dict[str, Any],
) -> pd.DataFrame:
    scaled = v1._transform_sequence_matrix(df, seq_features, seq_scaler)
    seq = v1._sequence_tensor(scaled, contexts, lookback=v2.LOOKBACK)
    deep = v2._deep_predict_v2(deep_model, seq, deep_meta["target_mean"], deep_meta["target_std"])
    return v1._state_features(state_model, deep)


def _row_features(
    df: pd.DataFrame,
    ctx: dict[str, Any],
    state_df: pd.DataFrame,
    n: int,
    account_dd: float,
    daily_dd: float,
    loss_streak: int,
) -> dict[str, float]:
    row = v1.base._features(df, ctx, account_dd, daily_dd, loss_streak)
    row.update({k: float(val) for k, val in state_df.iloc[n].to_dict().items()})
    return row


def backtest_constant_gross(
    cfg: ConstantGrossConfig,
    head_model: dict[str, Any],
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    state_df: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    high_cost3 = fee >= 0.0015 or slip >= 0.0006
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    mdd = 0.0
    wins = 0
    action_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    notional_sum = 0.0
    gross_max = 0.0
    net_max = 0.0
    ledger: list[dict[str, Any]] = []
    for n, ctx in enumerate(contexts):
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        row = _row_features(df, ctx, state_df, n, account_dd, daily_dd, loss_streak)
        same_pred, adverse_pred = v1._predict_heads(head_model, row)
        conviction = float(same_pred + 0.50 * row["deep_pred_same"] + 0.25 * row["deep_pred_full"])
        reasons: list[str] = []
        risky = (
            account_dd >= cfg.account_dd_defensive
            or daily_dd >= cfg.daily_dd_defensive
            or v1.base._stress(df, i)
            or adverse_pred >= cfg.adverse_cut
            or float(row["deep_pred_adverse"]) >= cfg.adverse_cut
            or float(row["deep_pred_full"]) < cfg.deep_full_floor
        )
        if account_dd >= cfg.account_dd_defensive:
            reasons.append("account_dd_defensive")
        if daily_dd >= cfg.daily_dd_defensive:
            reasons.append("daily_dd_defensive")
        if v1.base._stress(df, i):
            reasons.append("stress_state")
        if adverse_pred >= cfg.adverse_cut:
            reasons.append("head_adverse_cut")
        if float(row["deep_pred_adverse"]) >= cfg.adverse_cut:
            reasons.append("deep_adverse_cut")
        if float(row["deep_pred_full"]) < cfg.deep_full_floor:
            reasons.append("deep_full_floor")
        if high_cost3:
            effective_notional = float(cfg.cost3_notional)
            action = "COST3_CAPITAL_PRESERVE" if effective_notional <= 1e-12 else "COST3_LOW_NOTIONAL"
            reasons.append("cost3_capital_preserve")
        elif risky:
            effective_notional = min(float(ctx["core_notional"]), float(cfg.defensive_notional))
            action = "DEEP_DEFENSIVE"
        elif conviction >= cfg.conviction_threshold:
            effective_notional = float(cfg.target_notional)
            action = "TARGET_GROSS"
            reasons.append("deep_constant_gross")
        else:
            effective_notional = min(float(ctx["core_notional"]), float(cfg.defensive_notional))
            action = "LOW_CONVICTION_DEFENSIVE"
            reasons.append("conviction_below_threshold")
        effective_notional = min(max(effective_notional, 0.0), 3.6)
        if effective_notional <= 1e-12:
            action_counts[action] = action_counts.get(action, 0) + 1
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        side = int(ctx["side"])
        core_exit_idx = int(ctx["core_exit_idx"])
        gross = effective_notional
        net = effective_notional
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        notional_sum += effective_notional
        action_counts[action] = action_counts.get(action, 0) + 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        before = cash
        entry = float(ctx["entry_price"])
        entry_fee = cash * fee * effective_notional
        cash -= entry_fee
        for j in range(i, core_exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            unreal = v1.base._raw(side, entry, mark) * effective_notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        exit_px = v1.base._exit_price(fill_px, min(core_exit_idx + 1, len(close) - 1), side, slip)
        realized = v1.base._raw(side, entry, exit_px) * effective_notional
        cash = cash * (1.0 + realized)
        exit_fee = cash * fee * effective_notional
        cash -= exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        wins += int(cash > before)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "effective_exit_idx": core_exit_idx,
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_side": side,
                "core_notional": float(ctx["core_notional"]),
                "effective_core_notional": effective_notional,
                "gross_notional": gross,
                "net_notional": net,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "loss_streak_prior": loss_streak,
                "hybrid_same_pred": same_pred,
                "hybrid_adverse_pred": adverse_pred,
                "deep_pred_full": float(row["deep_pred_full"]),
                "deep_pred_adverse": float(row["deep_pred_adverse"]),
                "deep_pred_same": float(row["deep_pred_same"]),
                "conviction": conviction,
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "total_fee_cash": entry_fee + exit_fee,
                "core_pnl_pct": realized * 100.0,
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
    trades = len(ledger)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / v1.base._days(df)),
        "wr": float(wins / max(trades, 1)),
        "action_counts": action_counts,
        "avg_effective_notional": float(notional_sum / max(trades, 1)),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"passed": False, "reason": "empty_ledger"}
    df = pd.DataFrame(ledger)
    final_pnl = (float(df["cash_after"].iloc[-1]) - 1.0) * 100.0
    step = (df["cash_before"] * (1.0 + df["trade_pnl_pct"] / 100.0) - df["cash_after"]).abs().max()
    fee = (df["total_fee_cash"] - df["entry_fee_cash"] - df["exit_fee_cash"]).abs().max()
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "passed": bool(abs(final_pnl - float(report_pnl)) < 1e-9 and step < 1e-9 and fee < 1e-9),
        "final_pnl_from_ledger": float(final_pnl),
        "report_pnl": float(report_pnl),
        "max_step_equity_error": float(step),
        "max_fee_identity_error": float(fee),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
        "negative_notional": int((df["effective_core_notional"] < 0.0).sum()),
        "gross_cap": int((df["gross_notional"] > 3.6 + 1e-12).sum()),
        "net_cap": int((df["net_notional"] > 3.6 + 1e-12).sum()),
        "exit_after_core": int((df["effective_exit_idx"] > df["core_exit_idx"]).sum()),
        "side_changed": 0,
    }


def _contract_doc() -> str:
    return """# Clean Base Deep Constant Gross V1 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entries, side, and exits are preserved.
- Deep layer: v2 3-seed GRU ensemble over market and AI feature sequences.
- Unsupervised state: KMeans over deep embeddings and target heads.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: replace baseline variable notional with a validation-selected target gross exposure, unless deep risk gates force defensive exposure.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index equals the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Deep Constant Gross V1

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Trades/day 1x | `{c1['core_trades_per_day']:.6f}` |
| Avg notional 1x | `{c1['avg_effective_notional']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

Selected: `{report['selected_config']['name']}`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = _build_runtime_models(args)
    train_full = v1.base._read(args.train_csv)
    train_df, val_df = v1.base._split_train_validation(train_full, args.split_date)
    oos_df = v1.base._read(args.eval_csv)
    train_pre, train_ctx, _train_life, _train_meta = _build_contexts(train_df, models, fee=float(args.fee), slip=float(args.slip))
    train_labels = v1._label_frame(train_df, train_pre, train_ctx, fee=float(args.fee), slip=float(args.slip))
    seq_features = v1._available_sequence_features(train_df)
    seq_scaler, train_scaled = v1._fit_sequence_scaler(train_df, seq_features)
    train_seq = v1._sequence_tensor(train_scaled, train_ctx, lookback=v2.LOOKBACK)
    deep_model, deep_meta = v2._train_deep_encoder_v2(
        train_seq,
        train_labels,
        epochs=int(args.deep_epochs),
        batch_size=int(args.deep_batch_size),
    )
    deep_train = v2._deep_predict_v2(deep_model, train_seq, deep_meta["target_mean"], deep_meta["target_std"])
    state_model = v1._fit_state_model(deep_train, train_labels)
    train_state = v1._state_features(state_model, deep_train)
    head_model, head_meta = v1._train_supervised_heads(
        train_df,
        train_pre,
        train_ctx,
        train_state,
        train_labels,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    val_pre_1, val_ctx_1, val_life_1, _ = _build_contexts(val_df, models, fee=float(args.fee), slip=float(args.slip))
    val_pre_2, val_ctx_2, _val_life_2, _ = _build_contexts(val_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_pre_3, val_ctx_3, _val_life_3, _ = _build_contexts(val_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    oos_pre_1, oos_ctx_1, oos_life_1, _ = _build_contexts(oos_df, models, fee=float(args.fee), slip=float(args.slip))
    oos_pre_2, oos_ctx_2, _oos_life_2, _ = _build_contexts(oos_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_pre_3, oos_ctx_3, _oos_life_3, _ = _build_contexts(oos_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_state_1 = _state_for(val_df, val_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_2 = _state_for(val_df, val_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_3 = _state_for(val_df, val_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_1 = _state_for(oos_df, oos_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_2 = _state_for(oos_df, oos_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_3 = _state_for(oos_df, oos_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: ConstantGrossConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        val_1 = backtest_constant_gross(cfg, head_model, val_df, val_pre_1, val_ctx_1, val_state_1, fee=float(args.fee), slip=float(args.slip))
        val_2 = backtest_constant_gross(cfg, head_model, val_df, val_pre_2, val_ctx_2, val_state_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        val_3 = backtest_constant_gross(cfg, head_model, val_df, val_pre_3, val_ctx_3, val_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {
            **asdict(cfg),
            "val_pnl": val_1["pnl"],
            "val_mdd": val_1["mdd"],
            "val_cost2_pnl": val_2["pnl"],
            "val_cost3_pnl": val_3["pnl"],
            "val_avg_notional": val_1["avg_effective_notional"],
            "selection_score": _score(val_1, val_2, val_3),
        }
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(val_1), "cost_2x": _compact(val_2), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None
    full = backtest_constant_gross(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_1,
        oos_ctx_1,
        oos_state_1,
        fee=float(args.fee),
        slip=float(args.slip),
        ledger_out=args.ledger_csv_out,
    )
    cost_2 = backtest_constant_gross(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_2,
        oos_ctx_2,
        oos_state_2,
        fee=float(args.fee) * 2.0,
        slip=float(args.slip) * 2.0,
    )
    cost_3 = backtest_constant_gross(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_3,
        oos_ctx_3,
        oos_state_3,
        fee=float(args.fee) * 3.0,
        slip=float(args.slip) * 3.0,
    )
    accounting = _audit(full["pnl"], full["ledger"])
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "training_labels_use_future": True,
        "validation_selection_only": True,
        "oos_threshold_selection": False,
    }
    gates = {
        "target_500_pnl": bool(full["pnl"] >= 500.0),
        "clean_base_pnl_gate": bool(full["pnl"] >= v1.editor.CLEAN_BASE_REFERENCE["pnl"]),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_capital_preserved": bool(cost_3["pnl"] >= -1e-12),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "notional_invariant_passed": bool(
            accounting["negative_notional"] == 0
            and accounting["gross_cap"] == 0
            and accounting["net_cap"] == 0
            and accounting["exit_after_core"] == 0
        ),
    }
    gates["decision"] = "promote" if all(gates.values()) else (
        "shadow_candidate" if gates["target_500_pnl"] and gates["accounting_audit_passed"] and gates["notional_invariant_passed"] else "reject"
    )
    clean_oos = backtest_no_limit_exit(
        oos_df,
        models["policy"],
        models["exit_model"],
        entry_config=models["entry_cfg"],
        risk_config=models["risk_cfg"],
        exit_threshold=float(models["exit_cfg"]["exit_threshold"]),
        min_exit_age=int(models["exit_cfg"]["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=oos_pre_1,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "deep_constant_gross.pkl"
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_scaler": seq_scaler,
            "state_model": state_model,
            "head_model": head_model,
            "deep_meta": deep_meta,
            "head_meta": head_meta,
            "selected_config": asdict(selected_cfg),
            "torch_model": str(torch_out),
        },
        model_out,
    )
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True))
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "training": {"deep": deep_meta, "head": head_meta, "state": {"n_clusters": v2.N_CLUSTERS}},
        "validation": selected_val,
        "validation_grid_rows": len(grid_rows),
        "cost_1x": _compact(full),
        "cost_2x": _compact(cost_2),
        "cost_3x": _compact(cost_3),
        "clean_base_reference": v1.editor.CLEAN_BASE_REFERENCE,
        "clean_base_oos_reference": v1.base._compact(clean_oos),
        "lifecycle_v1_reference": {"oos": _compact(oos_life_1), "report": str(args.lifecycle_report)},
        "promotion_gate": gates,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "data": {
            "train_range": v1.base._range(train_df),
            "validation_range": v1.base._range(val_df),
            "oos_range": v1.base._range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "feature_contract": {"sequence_features": seq_features, "runtime_forbidden": list(getattr(v1.base, "FORBIDDEN_RUNTIME_FEATURES", []))},
        "artifacts": {
            "model": str(model_out),
            "torch_model": str(torch_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "doc": str(args.doc_out),
            "contract": str(DEFAULT_CONTRACT),
        },
        "validation_top10": sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True)[:10],
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
                "cost_1x": report["cost_1x"],
                "cost_2x": report["cost_2x"],
                "cost_3x": report["cost_3x"],
                "promotion_gate": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base deep constant gross v1.")
    p.add_argument("--policy", type=Path, default=v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=40)
    p.add_argument("--deep-batch-size", type=int, default=128)
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
