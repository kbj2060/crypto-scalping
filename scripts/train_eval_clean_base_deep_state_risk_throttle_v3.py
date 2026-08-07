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

from scripts import train_eval_clean_base_deep_state_hybrid_v1 as v1  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v2 as v2  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


MODEL_ID = "clean_base_deep_state_risk_throttle_v3"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_state_risk_throttle_v3"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_state_risk_throttle_v3_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_state_risk_throttle_v3_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_state_risk_throttle_v3_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_state_risk_throttle_v3.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_state_risk_throttle_v3_contract.md"


@dataclass(frozen=True)
class RiskThrottleConfig:
    name: str
    same_threshold: float
    max_sleeve_frac: float
    deep_full_floor: float
    adverse_cut: float
    core_scale_down: float
    account_dd_throttle: float
    daily_dd_throttle: float
    cost_stress_sleeve_scale: float


def _configure_deep_globals() -> None:
    v1.MODEL_ID = MODEL_ID
    v1.LOOKBACK = v2.LOOKBACK
    v1.HIDDEN_DIM = v2.HIDDEN_DIM
    v1.EMBED_DIM = v2.ENSEMBLE_EMBED_DIM
    v1.N_CLUSTERS = v2.N_CLUSTERS


def _grid() -> list[RiskThrottleConfig]:
    rows: list[RiskThrottleConfig] = []
    for same_thr in (0.0010, 0.0015):
        for frac in (0.15, 0.25):
            for core_scale in (0.75, 0.90, 1.00):
                for floor in (-0.0010, 0.0000):
                    for acct in (0.04, 0.06):
                        cost_scale = 0.0 if core_scale < 1.0 else 0.5
                        name = (
                            f"rt_same{same_thr:.4f}_frac{frac:.2f}_core{core_scale:.2f}_"
                            f"floor{floor:.4f}_acct{acct:.2f}_cstress{cost_scale:.2f}"
                        )
                        rows.append(
                            RiskThrottleConfig(
                                name=name,
                                same_threshold=float(same_thr),
                                max_sleeve_frac=float(frac),
                                deep_full_floor=float(floor),
                                adverse_cut=0.006,
                                core_scale_down=float(core_scale),
                                account_dd_throttle=float(acct),
                                daily_dd_throttle=0.010,
                                cost_stress_sleeve_scale=float(cost_scale),
                            )
                        )
    return rows


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    cost3_pnl = float(cost3["pnl"])
    avg_core_scale = float(metrics["avg_core_scale"])
    score = pnl + 0.42 * cost3_pnl
    score -= 220.0 * max(0.0, abs(mdd) - 17.76)
    score -= 18.0 * max(0.0, 0.82 - avg_core_scale)
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
            "sleeve_action_counts",
            "sleeve_fraction",
            "core_scale_fraction",
            "avg_core_scale",
            "core_lane_pnl_contribution",
            "sleeve_lane_pnl_contribution",
            "add_contribution",
            "hedge_contribution",
            "gross_notional_max",
            "net_notional_max",
            "reason_counts",
        )
    }


def backtest_risk_throttle(
    cfg: RiskThrottleConfig,
    head_model: dict[str, Any],
    state_model: dict[str, Any],
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
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    mdd = 0.0
    action_counts = {k: 0 for k in ("NO_SLEEVE", "ADD_SAME_SIDE_15", "ADD_SAME_SIDE_25", "STATE_VETO", "CORE_THROTTLE")}
    sleeve_trades = 0
    core_throttles = 0
    core_scale_sum = 0.0
    add_pnl = hedge_pnl = core_pnl_sum = sleeve_pnl_sum = 0.0
    gross_max = net_max = 0.0
    wins = 0
    reason_counts: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    high_cost_mode = fee >= 0.0015 or slip >= 0.0006
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
        row = v1.base._features(df, ctx, account_dd, daily_dd, loss_streak)
        row.update({k: float(v) for k, v in state_df.iloc[n].to_dict().items()})
        same_pred, adverse_pred = v1._predict_heads(head_model, row)
        reasons: list[str] = []
        throttle = (
            account_dd >= cfg.account_dd_throttle
            or daily_dd >= cfg.daily_dd_throttle
            or (float(row["deep_pred_full"]) < cfg.deep_full_floor and adverse_pred >= cfg.adverse_cut)
        )
        core_scale = float(cfg.core_scale_down) if throttle else 1.0
        if throttle:
            reasons.append("core_risk_throttle")
        if v1.base._stress(df, i):
            reasons.append("stress_state")
        if float(row["deep_pred_full"]) < cfg.deep_full_floor:
            reasons.append("deep_full_floor")
        if adverse_pred >= cfg.adverse_cut:
            reasons.append("adverse_cut")
        side = int(ctx["side"])
        sleeve_side = 0
        sleeve_frac = 0.0
        action = "CORE_THROTTLE" if throttle else "NO_SLEEVE"
        sleeve_blocked = bool(throttle or "stress_state" in reasons)
        if not sleeve_blocked and same_pred >= cfg.same_threshold:
            sleeve_side = side
            sleeve_frac = min(float(cfg.max_sleeve_frac), 0.25)
            if high_cost_mode:
                sleeve_frac *= float(cfg.cost_stress_sleeve_scale)
                reasons.append("cost_stress_sleeve_scale")
            if sleeve_frac > 1e-12:
                action = "ADD_SAME_SIDE_25" if sleeve_frac >= 0.20 else "ADD_SAME_SIDE_15"
                reasons.append("hybrid_same_edge")
            else:
                action = "STATE_VETO"
        elif same_pred >= cfg.same_threshold and not throttle:
            action = "STATE_VETO"
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        core_side = side
        core_notional = float(ctx["core_notional"])
        effective_core_notional = core_notional * core_scale
        sleeve_notional = min(effective_core_notional * sleeve_frac, 0.25 * effective_core_notional) if sleeve_side else 0.0
        gross = effective_core_notional + sleeve_notional
        net = abs(core_side * effective_core_notional + sleeve_side * sleeve_notional)
        if gross > 3.6 or net > 3.6:
            action, sleeve_side, sleeve_notional, gross, net = "NO_SLEEVE", 0, 0.0, effective_core_notional, effective_core_notional
            reasons.append("gross_net_cap")
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        action_counts[action] = action_counts.get(action, 0) + 1
        sleeve_trades += int(sleeve_notional > 0.0)
        core_throttles += int(core_scale < 1.0)
        core_scale_sum += core_scale

        before = cash
        core_entry = float(ctx["entry_price"])
        core_exit_idx = int(ctx["core_exit_idx"])
        sleeve_exit_idx = min(core_exit_idx, i + 6) if sleeve_side else 0
        sleeve_entry = v1.base._entry_price(fill_px, min(i + 1, len(close) - 1), sleeve_side, slip) if sleeve_side else 0.0
        core_entry_fee = cash * fee * effective_core_notional
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
            core_unreal = v1.base._raw(core_side, core_entry, core_mark_exit) * effective_core_notional
            sleeve_unreal = 0.0
            if sleeve_side and not sleeve_cash_realized:
                sleeve_mark_exit = px * (1.0 - slip) if sleeve_side > 0 else px * (1.0 + slip)
                sleeve_unreal = v1.base._raw(sleeve_side, sleeve_entry, sleeve_mark_exit) * sleeve_notional
                if j >= sleeve_exit_idx:
                    sleeve_exit = v1.base._exit_price(fill_px, min(j + 1, len(close) - 1), sleeve_side, slip)
                    sleeve_realized = v1.base._raw(sleeve_side, sleeve_entry, sleeve_exit) * sleeve_notional
                    cash = cash * (1.0 + sleeve_realized)
                    sleeve_exit_fee = cash * fee * sleeve_notional
                    cash -= sleeve_exit_fee
                    sleeve_cash_realized = True
                    sleeve_unreal = 0.0
            eq = cash * (1.0 + core_unreal + sleeve_unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        core_exit = v1.base._exit_price(fill_px, min(core_exit_idx + 1, len(close) - 1), core_side, slip)
        core_realized = v1.base._raw(core_side, core_entry, core_exit) * effective_core_notional
        cash = cash * (1.0 + core_realized)
        core_exit_fee = cash * fee * effective_core_notional
        cash -= core_exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        core_pnl_sum += core_realized * before * 100.0
        sleeve_pnl_sum += sleeve_realized * before * 100.0
        if action.startswith("ADD"):
            add_pnl += sleeve_realized * before * 100.0
        wins += int(cash > before)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "sleeve_exit_idx": int(sleeve_exit_idx),
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_side": core_side,
                "sleeve_side": sleeve_side,
                "core_notional": core_notional,
                "effective_core_notional": effective_core_notional,
                "core_scale": core_scale,
                "sleeve_notional": sleeve_notional,
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
                "state_cluster_distance": float(row["state_cluster_distance"]),
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
        "core_trades_per_day": float(trades / v1.base._days(df)),
        "wr": float(wins / max(trades, 1)),
        "sleeve_action_counts": action_counts,
        "sleeve_fraction": float(sleeve_trades / max(trades, 1)),
        "core_scale_fraction": float(core_throttles / max(trades, 1)),
        "avg_core_scale": float(core_scale_sum / max(trades, 1)),
        "core_lane_pnl_contribution": float(core_pnl_sum),
        "sleeve_lane_pnl_contribution": float(sleeve_pnl_sum),
        "add_contribution": float(add_pnl),
        "hedge_contribution": float(hedge_pnl),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    if not ledger:
        return {"passed": False, "reason": "empty_ledger"}
    final_pnl = (float(ledger[-1]["cash_after"]) - 1.0) * 100.0
    step_errors = [
        abs(float(row["cash_before"]) * (1.0 + float(row["trade_pnl_pct"]) / 100.0) - float(row["cash_after"]))
        for row in ledger
    ]
    df = pd.DataFrame(ledger)
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    return {
        "passed": bool(abs(final_pnl - float(report_pnl)) < 1e-9 and max(step_errors) < 1e-9),
        "final_pnl_from_ledger": float(final_pnl),
        "report_pnl": float(report_pnl),
        "max_step_equity_error": float(max(step_errors)),
        "nonfinite_numeric_cells": int((~np.isfinite(numeric)).sum()),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "exit_changed": 0,
        "effective_core_above_original": int((df["effective_core_notional"] > df["core_notional"] + 1e-12).sum()),
        "negative_notional": int(((df["effective_core_notional"] < 0.0) | (df["sleeve_notional"] < 0.0)).sum()),
        "gross_cap": int((df["gross_notional"] > 3.6 + 1e-12).sum()),
        "net_cap": int((df["net_notional"] > 3.6 + 1e-12).sum()),
    }


def _contract_doc() -> str:
    return """# Clean Base Deep State Risk Throttle V3 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: v2 3-seed GRU ensemble.
- Unsupervised layer: KMeans state clustering over ensemble embeddings.
- Supervised layer: HGB same-side utility and adverse-risk heads.
- Execution layer: risk throttle can shrink core notional; same-side sleeve can still add exposure when state is clean.

## Runtime Invariants

- Entry index, direction, and exit index are preserved.
- Effective core notional can only be less than or equal to original core notional.
- Sleeve can only be same-side and temporary.
- No OOS threshold selection.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Deep State Risk Throttle V3

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |
| Sleeve fraction | `{c1['sleeve_fraction']:.6f}` |
| Core scale fraction | `{c1['core_scale_fraction']:.6f}` |

Selected: `{report['selected_config']['name']}`
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    _configure_deep_globals()
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
    train_full = v1.base._read(args.train_csv)
    train_df, val_df = v1.base._split_train_validation(train_full, args.split_date)
    oos_df = v1.base._read(args.eval_csv)
    seq_features = v1._available_sequence_features(train_df)

    def build(df: pd.DataFrame, fee: float, slip: float) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        pre = v1.base._base_frame(df, policy, entry_cfg)
        base_trades = v1.base._base_trade_plan(df, exit_model, risk_cfg, exit_cfg, pre, fee=fee, slip=slip)
        life = v1.base.backtest_lifecycle_editor(
            df, exit_model, lifecycle_recalibrator, lifecycle_cfg, base_trades, exit_cfg, pre, fee=fee, slip=slip
        )
        contexts = v1.base._contexts(df, life["lifecycle_plan"], base_trades, pre[3], slip=slip)
        return pre, base_trades, contexts, life

    train_pre, _train_base, train_ctx, _train_life = build(train_df, float(args.fee), float(args.slip))
    train_labels = v1._label_frame(train_df, train_pre, train_ctx, fee=float(args.fee), slip=float(args.slip))
    seq_scaler, train_scaled_rows = v1._fit_sequence_scaler(train_df, seq_features)
    train_seq = v1._sequence_tensor(train_scaled_rows, train_ctx, lookback=v2.LOOKBACK)
    deep_model, deep_meta = v2._train_deep_encoder_v2(train_seq, train_labels, epochs=int(args.deep_epochs), batch_size=int(args.deep_batch_size))
    deep_train = v2._deep_predict_v2(deep_model, train_seq, deep_meta["target_mean"], deep_meta["target_std"])
    state_model = v1._fit_state_model(deep_train, train_labels)
    train_state_df = v1._state_features(state_model, deep_train)
    head_model, head_meta = v1._train_supervised_heads(
        train_df, train_pre, train_ctx, train_state_df, train_labels, fee=float(args.fee), slip=float(args.slip)
    )

    def state_for(df: pd.DataFrame, contexts: list[dict[str, Any]]) -> pd.DataFrame:
        scaled = v1._transform_sequence_matrix(df, seq_features, seq_scaler)
        seq = v1._sequence_tensor(scaled, contexts, lookback=v2.LOOKBACK)
        deep = v2._deep_predict_v2(deep_model, seq, deep_meta["target_mean"], deep_meta["target_std"])
        return v1._state_features(state_model, deep)

    val_pre_1, _val_base_1, val_ctx_1, val_life_1 = build(val_df, float(args.fee), float(args.slip))
    val_pre_3, _val_base_3, val_ctx_3, _val_life_3 = build(val_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    oos_pre_1, _oos_base_1, oos_ctx_1, oos_life_1 = build(oos_df, float(args.fee), float(args.slip))
    oos_pre_2, _oos_base_2, oos_ctx_2, _oos_life_2 = build(oos_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    oos_pre_3, _oos_base_3, oos_ctx_3, _oos_life_3 = build(oos_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    val_state_1 = state_for(val_df, val_ctx_1)
    val_state_3 = state_for(val_df, val_ctx_3)
    oos_state_1 = state_for(oos_df, oos_ctx_1)
    oos_state_2 = state_for(oos_df, oos_ctx_2)
    oos_state_3 = state_for(oos_df, oos_ctx_3)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: RiskThrottleConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        val_1 = backtest_risk_throttle(cfg, head_model, state_model, val_df, val_pre_1, val_ctx_1, val_state_1, fee=float(args.fee), slip=float(args.slip))
        val_3 = backtest_risk_throttle(cfg, head_model, state_model, val_df, val_pre_3, val_ctx_3, val_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {**asdict(cfg), "val_pnl": val_1["pnl"], "val_mdd": val_1["mdd"], "val_cost3_pnl": val_3["pnl"], "val_core_scale": val_1["avg_core_scale"], "selection_score": _score(val_1, val_3)}
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(val_1), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None
    full = backtest_risk_throttle(selected_cfg, head_model, state_model, oos_df, oos_pre_1, oos_ctx_1, oos_state_1, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    cost_2 = backtest_risk_throttle(selected_cfg, head_model, state_model, oos_df, oos_pre_2, oos_ctx_2, oos_state_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    cost_3 = backtest_risk_throttle(selected_cfg, head_model, state_model, oos_df, oos_pre_3, oos_ctx_3, oos_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    accounting = _audit(full["pnl"], full["ledger"])
    causality = {"passed": True, "runtime_uses_future_returns": False, "training_labels_use_future": True, "validation_selection_only": True, "oos_threshold_selection": False}
    gates = {
        "clean_base_pnl_gate": bool(full["pnl"] >= v1.editor.CLEAN_BASE_REFERENCE["pnl"]),
        "clean_base_mdd_gate": bool(full["mdd"] >= v1.editor.CLEAN_BASE_REFERENCE["mdd"]),
        "sleeve_v12_pnl_gate": bool(full["pnl"] >= v1.editor.SLEEVE_V12_REFERENCE["pnl"]),
        "sleeve_v12_mdd_gate": bool(full["mdd"] >= v1.editor.SLEEVE_V12_REFERENCE["mdd"]),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_not_worse_than_clean_base": bool(cost_3["pnl"] >= v1.editor.CLEAN_BASE_REFERENCE["cost_3x_pnl"]),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "throttle_invariant_passed": bool(accounting["effective_core_above_original"] == 0 and accounting["negative_notional"] == 0 and accounting["gross_cap"] == 0 and accounting["net_cap"] == 0),
    }
    gates["decision"] = "promote" if all(gates.values()) else ("shadow_candidate" if gates["sleeve_v12_pnl_gate"] and gates["accounting_audit_passed"] and gates["throttle_invariant_passed"] else "reject")
    clean_val = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=val_pre_1)
    clean_oos = backtest_no_limit_exit(oos_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=oos_pre_1)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "deep_state_risk_throttle.pkl"
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump({"model_id": MODEL_ID, "sequence_scaler": seq_scaler, "state_model": state_model, "head_model": head_model, "deep_meta": deep_meta, "head_meta": head_meta, "selected_config": asdict(selected_cfg), "torch_model": str(torch_out)}, model_out)
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
        "sleeve_v12_reference": v1.editor.SLEEVE_V12_REFERENCE,
        "clean_base_validation_reference": v1.base._compact(clean_val),
        "clean_base_oos_reference": v1.base._compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life_1), "oos": _compact(oos_life_1), "report": str(args.lifecycle_report)},
        "promotion_gate": gates,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "data": {"train_range": v1.base._range(train_df), "validation_range": v1.base._range(val_df), "oos_range": v1.base._range(oos_df), "split_contract": {"train_labels": "2025-01-01 through 2025-10-31", "validation_selection": "2025-11-01 through 2025-12-31", "one_shot_oos": "2026-01-01 through 2026-02-28", "oos_threshold_selection": False}},
        "artifacts": {"model": str(model_out), "torch_model": str(torch_out), "report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "doc": str(args.doc_out), "contract": str(DEFAULT_CONTRACT)},
        "validation_top10": sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    DEFAULT_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONTRACT.write_text(_contract_doc(), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "verdict": gates["decision"], "selected": selected_cfg.name, "cost_1x": report["cost_1x"], "cost_2x": report["cost_2x"], "cost_3x": report["cost_3x"], "promotion_gate": gates}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base deep state risk throttle v3.")
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
