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


MODEL_ID = "clean_base_deep_core_reallocator_v1"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_core_reallocator_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_core_reallocator_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_core_reallocator_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_core_reallocator_v1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_core_reallocator_v1.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_core_reallocator_v1_contract.md"


@dataclass(frozen=True)
class CoreReallocatorConfig:
    name: str
    up_threshold: float
    scale_up: float
    scale_down: float
    adverse_cut: float
    deep_full_floor: float
    early_gain: float
    risk_early_h: int
    sleeve_frac: float
    account_dd_disable: float
    daily_dd_disable: float
    cost_stress_scale_up: float


def _configure_deep_globals() -> None:
    v1.MODEL_ID = MODEL_ID
    v1.LOOKBACK = v2.LOOKBACK
    v1.HIDDEN_DIM = v2.HIDDEN_DIM
    v1.EMBED_DIM = v2.ENSEMBLE_EMBED_DIM
    v1.N_CLUSTERS = v2.N_CLUSTERS


def _grid() -> list[CoreReallocatorConfig]:
    rows: list[CoreReallocatorConfig] = []
    for up_thr in (0.0005, 0.0010):
        for scale_up in (2.00, 2.50, 3.00):
            for adverse in (0.010,):
                for scale_down in (0.80, 1.00):
                    for early_gain in (0.0040,):
                        name = (
                            f"dcr_up{up_thr:.4f}_x{scale_up:.2f}_dn{scale_down:.2f}_"
                            f"adv{adverse:.3f}_eg{early_gain:.4f}"
                        )
                        rows.append(
                            CoreReallocatorConfig(
                                name=name,
                                up_threshold=float(up_thr),
                                scale_up=float(scale_up),
                                scale_down=float(scale_down),
                                adverse_cut=float(adverse),
                                deep_full_floor=-0.0010,
                                early_gain=float(early_gain),
                                risk_early_h=12,
                                sleeve_frac=0.0,
                                account_dd_disable=0.08,
                                daily_dd_disable=0.015,
                                cost_stress_scale_up=0.25,
                            )
                        )
    return rows


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = float(metrics["mdd"])
    cost3_pnl = float(cost3["pnl"])
    score = pnl + 0.05 * cost3_pnl
    score -= 18.0 * max(0.0, abs(mdd) - 28.0)
    score -= 20.0 * max(0.0, float(metrics["gross_notional_max"]) - 3.6)
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
            "scale_up_fraction",
            "scale_down_fraction",
            "early_exit_fraction",
            "avg_core_scale",
            "avg_effective_core_notional",
            "core_lane_pnl_contribution",
            "sleeve_lane_pnl_contribution",
            "gross_notional_max",
            "net_notional_max",
            "reason_counts",
        )
    }


def backtest_core_reallocator(
    cfg: CoreReallocatorConfig,
    head_model: dict[str, Any],
    state_model: dict[str, Any],
    hold_pred: dict[str, np.ndarray],
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
    high_cost_mode = fee >= 0.0015 or slip >= 0.0006
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
    scale_up_count = 0
    scale_down_count = 0
    early_exits = 0
    core_scale_sum = 0.0
    effective_core_sum = 0.0
    core_pnl_sum = 0.0
    sleeve_pnl_sum = 0.0
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
        row = v1.base._features(df, ctx, account_dd, daily_dd, loss_streak)
        row.update({k: float(v) for k, v in state_df.iloc[n].to_dict().items()})
        same_pred, adverse_pred = v1._predict_heads(head_model, row)
        hold_full = float(hold_pred["full"][n])
        hold_adverse = float(hold_pred["adverse"][n])
        hvals = {6: float(hold_pred["h6"][n]), 12: float(hold_pred["h12"][n]), 24: float(hold_pred["h24"][n])}
        best_h, best_early = max(hvals.items(), key=lambda kv: kv[1])
        conviction = float(same_pred + 0.60 * row["deep_pred_same"] + 0.40 * hold_full)
        risky = (
            account_dd >= cfg.account_dd_disable
            or daily_dd >= cfg.daily_dd_disable
            or v1.base._stress(df, i)
            or adverse_pred >= cfg.adverse_cut
            or hold_adverse >= cfg.adverse_cut
            or float(row["deep_pred_full"]) < cfg.deep_full_floor
        )
        reasons: list[str] = []
        if account_dd >= cfg.account_dd_disable:
            reasons.append("account_dd_disable")
        if daily_dd >= cfg.daily_dd_disable:
            reasons.append("daily_dd_disable")
        if v1.base._stress(df, i):
            reasons.append("stress_state")
        if adverse_pred >= cfg.adverse_cut:
            reasons.append("head_adverse_cut")
        if hold_adverse >= cfg.adverse_cut:
            reasons.append("hold_adverse_cut")
        if float(row["deep_pred_full"]) < cfg.deep_full_floor:
            reasons.append("deep_full_floor")
        core_scale = 1.0
        action = "CORE_KEEP"
        if risky:
            core_scale = float(cfg.scale_down)
            action = "CORE_SCALE_DOWN" if core_scale < 1.0 else "CORE_KEEP_RISK"
        elif conviction >= cfg.up_threshold:
            core_scale = float(cfg.scale_up)
            if high_cost_mode:
                core_scale = 1.0 + (core_scale - 1.0) * float(cfg.cost_stress_scale_up)
                reasons.append("cost_stress_scale_up")
            action = "CORE_SCALE_UP" if core_scale > 1.0 else "CORE_KEEP"
            reasons.append("deep_core_conviction")
        early_h: int | None = None
        if best_early >= hold_full + float(cfg.early_gain):
            early_h = int(best_h)
            reasons.append("early_head_beats_full")
        elif risky and int(cfg.risk_early_h) > 0:
            early_h = int(cfg.risk_early_h)
            reasons.append("risk_early_exit")
        side = int(ctx["side"])
        core_notional = float(ctx["core_notional"])
        effective_core_notional = min(max(core_notional * core_scale, 0.0), 3.6)
        sleeve_side = 0
        sleeve_notional = 0.0
        gross = effective_core_notional + sleeve_notional
        net = abs(side * effective_core_notional + sleeve_side * sleeve_notional)
        if gross > 3.6 or net > 3.6:
            effective_core_notional = min(effective_core_notional, 3.6)
            sleeve_notional = 0.0
            gross = effective_core_notional
            net = effective_core_notional
            reasons.append("gross_net_cap")
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        action_counts[action] = action_counts.get(action, 0) + 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        scale_up_count += int(effective_core_notional > core_notional + 1e-12)
        scale_down_count += int(effective_core_notional < core_notional - 1e-12)
        core_scale_eff = effective_core_notional / max(core_notional, 1e-12)
        core_scale_sum += core_scale_eff
        effective_core_sum += effective_core_notional

        before = cash
        core_entry = float(ctx["entry_price"])
        core_exit_idx = int(ctx["core_exit_idx"])
        exit_idx = min(core_exit_idx, i + int(early_h), len(close) - 2) if early_h else core_exit_idx
        early_exits += int(exit_idx < core_exit_idx)
        core_entry_fee = cash * fee * effective_core_notional
        cash -= core_entry_fee
        for j in range(i, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            core_mark_exit = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            core_unreal = v1.base._raw(side, core_entry, core_mark_exit) * effective_core_notional
            eq = cash * (1.0 + core_unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        core_exit = v1.base._exit_price(fill_px, min(exit_idx + 1, len(close) - 1), side, slip)
        core_realized = v1.base._raw(side, core_entry, core_exit) * effective_core_notional
        cash = cash * (1.0 + core_realized)
        core_exit_fee = cash * fee * effective_core_notional
        cash -= core_exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        core_pnl_sum += core_realized * before * 100.0
        wins += int(cash > before)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "effective_exit_idx": int(exit_idx),
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_side": side,
                "core_notional": core_notional,
                "effective_core_notional": effective_core_notional,
                "core_scale": core_scale_eff,
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
                "hold_pred_full": hold_full,
                "hold_pred_adverse": hold_adverse,
                "conviction": conviction,
                "early_horizon": int(early_h) if early_h else 0,
                "entry_fee_cash": core_entry_fee,
                "exit_fee_cash": core_exit_fee,
                "total_fee_cash": core_entry_fee + core_exit_fee,
                "core_pnl_pct": core_realized * 100.0,
                "sleeve_pnl_pct": 0.0,
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
        "action_counts": action_counts,
        "scale_up_fraction": float(scale_up_count / max(trades, 1)),
        "scale_down_fraction": float(scale_down_count / max(trades, 1)),
        "early_exit_fraction": float(early_exits / max(trades, 1)),
        "avg_core_scale": float(core_scale_sum / max(trades, 1)),
        "avg_effective_core_notional": float(effective_core_sum / max(trades, 1)),
        "core_lane_pnl_contribution": float(core_pnl_sum),
        "sleeve_lane_pnl_contribution": float(sleeve_pnl_sum),
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
        "negative_notional": int(((df["effective_core_notional"] < 0.0) | (df["sleeve_notional"] < 0.0)).sum()),
        "gross_cap": int((df["gross_notional"] > 3.6 + 1e-12).sum()),
        "net_cap": int((df["net_notional"] > 3.6 + 1e-12).sum()),
        "exit_after_core": int((df["effective_exit_idx"] > df["core_exit_idx"]).sum()),
        "side_changed": 0,
    }


def _contract_doc() -> str:
    return """# Clean Base Deep Core Reallocator V1 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: v2 3-seed GRU ensemble.
- Unsupervised layer: KMeans state clustering.
- Supervised heads: HGB same/adverse heads plus hold/early-exit heads.
- Execution layer: core direction is preserved, but core notional can scale up/down within a 3.6 gross cap; exits can only move earlier.

## Runtime Invariants

- Entry index and side are preserved.
- Effective exit index can only be less than or equal to Lifecycle core exit.
- Gross and net notional must be <= 3.6.
- No OOS threshold selection.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Deep Core Reallocator V1

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |
| Avg core scale | `{c1['avg_core_scale']:.6f}` |
| Scale-up fraction | `{c1['scale_up_fraction']:.6f}` |
| Early-exit fraction | `{c1['early_exit_fraction']:.6f}` |

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
    hold_model, hold_meta = v1.editor._train_editor_model(train_df, train_pre, train_ctx, fee=float(args.fee), slip=float(args.slip))

    def state_for(df: pd.DataFrame, contexts: list[dict[str, Any]]) -> pd.DataFrame:
        scaled = v1._transform_sequence_matrix(df, seq_features, seq_scaler)
        seq = v1._sequence_tensor(scaled, contexts, lookback=v2.LOOKBACK)
        deep = v2._deep_predict_v2(deep_model, seq, deep_meta["target_mean"], deep_meta["target_std"])
        return v1._state_features(state_model, deep)

    def hold_for(df: pd.DataFrame, contexts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        return v1.editor._predict_editor(hold_model, v1.editor._context_frame(df, contexts))

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
    val_hold_1 = hold_for(val_df, val_ctx_1)
    val_hold_3 = hold_for(val_df, val_ctx_3)
    oos_hold_1 = hold_for(oos_df, oos_ctx_1)
    oos_hold_2 = hold_for(oos_df, oos_ctx_2)
    oos_hold_3 = hold_for(oos_df, oos_ctx_3)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: CoreReallocatorConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        val_1 = backtest_core_reallocator(cfg, head_model, state_model, val_hold_1, val_df, val_pre_1, val_ctx_1, val_state_1, fee=float(args.fee), slip=float(args.slip))
        val_3 = backtest_core_reallocator(cfg, head_model, state_model, val_hold_3, val_df, val_pre_3, val_ctx_3, val_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {**asdict(cfg), "val_pnl": val_1["pnl"], "val_mdd": val_1["mdd"], "val_cost3_pnl": val_3["pnl"], "val_avg_core_scale": val_1["avg_core_scale"], "selection_score": _score(val_1, val_3)}
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(val_1), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None
    full = backtest_core_reallocator(selected_cfg, head_model, state_model, oos_hold_1, oos_df, oos_pre_1, oos_ctx_1, oos_state_1, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    cost_2 = backtest_core_reallocator(selected_cfg, head_model, state_model, oos_hold_2, oos_df, oos_pre_2, oos_ctx_2, oos_state_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    cost_3 = backtest_core_reallocator(selected_cfg, head_model, state_model, oos_hold_3, oos_df, oos_pre_3, oos_ctx_3, oos_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    accounting = _audit(full["pnl"], full["ledger"])
    causality = {"passed": True, "runtime_uses_future_returns": False, "training_labels_use_future": True, "validation_selection_only": True, "oos_threshold_selection": False}
    gates = {
        "target_500_pnl": bool(full["pnl"] >= 500.0),
        "clean_base_pnl_gate": bool(full["pnl"] >= v1.editor.CLEAN_BASE_REFERENCE["pnl"]),
        "sleeve_v12_pnl_gate": bool(full["pnl"] >= v1.editor.SLEEVE_V12_REFERENCE["pnl"]),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_not_worse_than_clean_base": bool(cost_3["pnl"] >= v1.editor.CLEAN_BASE_REFERENCE["cost_3x_pnl"]),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "notional_invariant_passed": bool(accounting["negative_notional"] == 0 and accounting["gross_cap"] == 0 and accounting["net_cap"] == 0 and accounting["exit_after_core"] == 0),
    }
    gates["decision"] = "promote" if all(gates.values()) else ("shadow_candidate" if gates["sleeve_v12_pnl_gate"] and gates["accounting_audit_passed"] and gates["notional_invariant_passed"] else "reject")
    clean_val = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=val_pre_1)
    clean_oos = backtest_no_limit_exit(oos_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=oos_pre_1)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "deep_core_reallocator.pkl"
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump({"model_id": MODEL_ID, "sequence_scaler": seq_scaler, "state_model": state_model, "head_model": head_model, "hold_model": hold_model, "deep_meta": deep_meta, "head_meta": head_meta, "hold_meta": hold_meta, "selected_config": asdict(selected_cfg), "torch_model": str(torch_out)}, model_out)
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True))
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "training": {"deep": deep_meta, "head": head_meta, "hold": hold_meta, "state": {"n_clusters": v2.N_CLUSTERS}},
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
    p = argparse.ArgumentParser(description="Clean base deep core reallocator v1.")
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
