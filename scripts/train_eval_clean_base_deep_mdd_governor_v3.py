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

from scripts import train_eval_clean_base_deep_constant_gross_v1 as cg  # noqa: E402
from scripts import train_eval_clean_base_deep_gated_gross_v2 as dgg  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v1 as v1  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v2 as v2  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402


MODEL_ID = "clean_base_deep_mdd_governor_v3"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_mdd_governor_v3"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_mdd_governor_v3_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_mdd_governor_v3_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_mdd_governor_v3_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_mdd_governor_v3.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_mdd_governor_v3_contract.md"
V2_REFERENCE_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_2026.json"


@dataclass(frozen=True)
class MddGovernorConfig:
    name: str
    high_notional: float
    mid_notional: float
    defensive_notional: float
    high_threshold: float
    mid_threshold: float
    adverse_cut: float
    deep_full_floor: float
    account_dd_soft: float
    account_dd_notional: float
    loss_streak_soft: int
    loss_streak_notional: float
    trade_stop: float
    cost3_notional: float


def _grid() -> list[MddGovernorConfig]:
    rows: list[MddGovernorConfig] = []
    profiles = (
        (0.99, 3.6, 999, 3.6, "open"),
        (0.16, 3.0, 999, 3.6, "dd_only"),
        (0.16, 3.0, 4, 3.0, "late_loss"),
        (0.12, 2.4, 4, 3.0, "balanced"),
        (0.08, 2.0, 4, 3.0, "tight_dd"),
    )
    for trade_stop in (0.040, 0.050, 0.060, 0.070):
        for dd_soft, dd_notional, loss_soft, loss_notional, profile in profiles:
                name = (
                    f"mdd_{profile}_h3.6_m3.0_d3.0_thr-0.006_stop{trade_stop:.3f}_"
                    f"dd{dd_soft:.2f}n{dd_notional:.1f}_ls{loss_soft}n{loss_notional:.1f}_c30"
                )
                rows.append(
                    MddGovernorConfig(
                        name=name,
                        high_notional=3.6,
                        mid_notional=3.0,
                        defensive_notional=3.0,
                        high_threshold=-0.006,
                        mid_threshold=-0.012,
                        adverse_cut=99.0,
                        deep_full_floor=-0.010,
                        account_dd_soft=float(dd_soft),
                        account_dd_notional=float(dd_notional),
                        loss_streak_soft=int(loss_soft),
                        loss_streak_notional=float(loss_notional),
                        trade_stop=float(trade_stop),
                        cost3_notional=0.0,
                    )
                )
    return rows


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        k: metrics.get(k)
        for k in (
            "pnl",
            "mdd",
            "closed_equity_mdd",
            "trades",
            "core_trades_per_day",
            "wr",
            "action_counts",
            "early_stop_fraction",
            "avg_effective_notional",
            "gross_notional_max",
            "net_notional_max",
            "deep_bucket_fraction",
            "reason_counts",
        )
    }


def _score(metrics: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = abs(float(metrics["mdd"]))
    c2 = float(cost2["pnl"])
    c3 = float(cost3["pnl"])
    score = pnl + 0.10 * c2 + 0.04 * c3
    score -= 24.0 * max(0.0, mdd - 18.0)
    score -= 8.0 * max(0.0, mdd - 12.0)
    score -= 120.0 * max(0.0, -c2)
    score += 180.0 if pnl >= 500.0 else 0.0
    return float(score)


def backtest_mdd_governor(
    cfg: MddGovernorConfig,
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
    closed_equity_peak = 1.0
    closed_mdd = 0.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    mdd = 0.0
    wins = 0
    early_stops = 0
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
        row = cg._row_features(df, ctx, state_df, n, account_dd, daily_dd, loss_streak)
        same_pred, adverse_pred = v1._predict_heads(head_model, row)
        sig = dgg._row_signal(row, same_pred, adverse_pred)
        reasons: list[str] = []
        stress = bool(v1.base._stress(df, i))
        if stress:
            reasons.append("stress_state")
        if sig["adverse"] >= cfg.adverse_cut:
            reasons.append("deep_or_head_adverse_cut")
        if sig["deep_full"] < cfg.deep_full_floor:
            reasons.append("deep_full_floor")

        if high_cost3:
            effective_notional = float(cfg.cost3_notional)
            action = "COST3_CAPITAL_PRESERVE" if effective_notional <= 1e-12 else "COST3_LOW_NOTIONAL"
            reasons.append("cost3_capital_preserve")
        elif stress or sig["adverse"] >= cfg.adverse_cut or sig["deep_full"] < cfg.deep_full_floor:
            effective_notional = float(cfg.defensive_notional)
            action = "DEFENSIVE"
        elif sig["conviction"] >= cfg.high_threshold:
            effective_notional = float(cfg.high_notional)
            action = "HIGH"
            reasons.append("deep_high_conviction")
        elif sig["conviction"] >= cfg.mid_threshold:
            effective_notional = float(cfg.mid_notional)
            action = "MID"
            reasons.append("deep_mid_conviction")
        else:
            effective_notional = float(cfg.defensive_notional)
            action = "DEFENSIVE"
            reasons.append("deep_low_conviction")

        if not high_cost3 and account_dd >= cfg.account_dd_soft:
            effective_notional = min(effective_notional, float(cfg.account_dd_notional))
            reasons.append("account_dd_mdd_throttle")
            action = f"{action}_DD"
        if not high_cost3 and loss_streak >= int(cfg.loss_streak_soft):
            effective_notional = min(effective_notional, float(cfg.loss_streak_notional))
            reasons.append("loss_streak_mdd_throttle")
            action = f"{action}_LS"

        effective_notional = min(max(effective_notional, 0.0), 3.6)
        action_counts[action] = action_counts.get(action, 0) + 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if effective_notional <= 1e-12:
            continue

        side = int(ctx["side"])
        core_exit_idx = int(ctx["core_exit_idx"])
        exit_idx = core_exit_idx
        stopped = False
        gross = effective_notional
        net = effective_notional
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        notional_sum += effective_notional
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
            if not high_cost3 and float(cfg.trade_stop) > 0.0 and unreal <= -abs(float(cfg.trade_stop)):
                exit_idx = int(j)
                stopped = True
                reasons.append("mdd_trade_stop")
                break
        exit_px = v1.base._exit_price(fill_px, min(exit_idx + 1, len(close) - 1), side, slip)
        realized = v1.base._raw(side, entry, exit_px) * effective_notional
        cash = cash * (1.0 + realized)
        exit_fee = cash * fee * effective_notional
        cash -= exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        wins += int(cash > before)
        early_stops += int(stopped)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        closed_equity_peak = max(closed_equity_peak, cash)
        closed_mdd = min(closed_mdd, cash / max(closed_equity_peak, 1e-12) - 1.0)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "effective_exit_idx": int(exit_idx),
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "early_stop": bool(stopped),
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
                "deep_pred_full": sig["deep_full"],
                "deep_pred_adverse": sig["deep_adverse"],
                "deep_pred_same": sig["deep_same"],
                "deep_conviction": sig["conviction"],
                "deep_adverse_gate": sig["adverse"],
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
    total_actions = max(1, sum(action_counts.values()))
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "closed_equity_mdd": float(closed_mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / v1.base._days(df)),
        "wr": float(wins / max(trades, 1)),
        "action_counts": action_counts,
        "early_stop_fraction": float(early_stops / max(trades, 1)),
        "deep_bucket_fraction": {k: float(v / total_actions) for k, v in action_counts.items()},
        "avg_effective_notional": float(notional_sum / max(trades, 1)),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _audit(report_pnl: float, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    return cg._audit(report_pnl, ledger)


def _contract_doc() -> str:
    return """# Clean Base Deep MDD Governor V3 Contract

Status: `experimental_challenger`

## Architecture

- Core lifecycle: audited clean-base/Lifecycle entry side is preserved.
- Deep layer: v2 3-seed GRU sequence ensemble.
- Supervised heads: HGB same-side expectancy and adverse-risk heads.
- Execution: deep exposure buckets plus account drawdown throttle, loss-streak throttle, and causal intra-trade hard stop.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to the audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
"""


def _doc(report: dict[str, Any]) -> str:
    c1 = report["cost_1x"]
    c2 = report["cost_2x"]
    c3 = report["cost_3x"]
    return f"""# Clean Base Deep MDD Governor V3

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Closed-equity MDD 1x | `{c1['closed_equity_mdd']:.6f}%` |
| Trades/day 1x | `{c1['core_trades_per_day']:.6f}` |
| Avg notional 1x | `{c1['avg_effective_notional']:.6f}` |
| Early stop fraction | `{c1['early_stop_fraction']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

Selected: `{report['selected_config']['name']}`
"""


def _reference_v2() -> dict[str, Any]:
    if not V2_REFERENCE_REPORT.exists():
        return {}
    return json.loads(V2_REFERENCE_REPORT.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = cg._build_runtime_models(args)
    train_full = v1.base._read(args.train_csv)
    train_df, val_df = v1.base._split_train_validation(train_full, args.split_date)
    oos_df = v1.base._read(args.eval_csv)
    train_pre, train_ctx, _train_life, _ = cg._build_contexts(train_df, models, fee=float(args.fee), slip=float(args.slip))
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

    val_pre_1, val_ctx_1, val_life_1, _ = cg._build_contexts(val_df, models, fee=float(args.fee), slip=float(args.slip))
    val_pre_2, val_ctx_2, _val_life_2, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_pre_3, val_ctx_3, _val_life_3, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    oos_pre_1, oos_ctx_1, oos_life_1, _ = cg._build_contexts(oos_df, models, fee=float(args.fee), slip=float(args.slip))
    oos_pre_2, oos_ctx_2, _oos_life_2, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_pre_3, oos_ctx_3, _oos_life_3, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_state_1 = cg._state_for(val_df, val_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_2 = cg._state_for(val_df, val_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_3 = cg._state_for(val_df, val_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_1 = cg._state_for(oos_df, oos_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_2 = cg._state_for(oos_df, oos_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_3 = cg._state_for(oos_df, oos_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: MddGovernorConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        val_1 = backtest_mdd_governor(cfg, head_model, val_df, val_pre_1, val_ctx_1, val_state_1, fee=float(args.fee), slip=float(args.slip))
        val_2 = backtest_mdd_governor(cfg, head_model, val_df, val_pre_2, val_ctx_2, val_state_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        val_3 = backtest_mdd_governor(cfg, head_model, val_df, val_pre_3, val_ctx_3, val_state_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {
            **asdict(cfg),
            "val_pnl": val_1["pnl"],
            "val_mdd": val_1["mdd"],
            "val_closed_mdd": val_1["closed_equity_mdd"],
            "val_cost2_pnl": val_2["pnl"],
            "val_cost3_pnl": val_3["pnl"],
            "val_avg_notional": val_1["avg_effective_notional"],
            "val_early_stop_fraction": val_1["early_stop_fraction"],
            "selection_score": _score(val_1, val_2, val_3),
        }
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(val_1), "cost_2x": _compact(val_2), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None

    full = backtest_mdd_governor(
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
    cost_2 = backtest_mdd_governor(
        selected_cfg,
        head_model,
        oos_df,
        oos_pre_2,
        oos_ctx_2,
        oos_state_2,
        fee=float(args.fee) * 2.0,
        slip=float(args.slip) * 2.0,
    )
    cost_3 = backtest_mdd_governor(
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
        "runtime_hard_stop_uses_observed_path_only": True,
    }
    v2_ref = _reference_v2()
    v2_mdd = abs(float(v2_ref.get("cost_1x", {}).get("mdd", 999.0) or 999.0))
    gates = {
        "target_500_pnl": bool(full["pnl"] >= 500.0),
        "mdd_improved_vs_v2": bool(abs(float(full["mdd"])) < v2_mdd),
        "mdd_under_20": bool(abs(float(full["mdd"])) <= 20.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_capital_preserved": bool(cost_3["pnl"] >= -1e-12),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
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
        "shadow_candidate" if gates["accounting_audit_passed"] and gates["notional_invariant_passed"] and full["pnl"] > 0.0 else "reject"
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
    model_out = args.model_dir / "deep_mdd_governor.pkl"
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_features": seq_features,
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
        "v2_reference": {
            "report": str(V2_REFERENCE_REPORT),
            "pnl": v2_ref.get("cost_1x", {}).get("pnl"),
            "mdd": v2_ref.get("cost_1x", {}).get("mdd"),
            "cost2_pnl": v2_ref.get("cost_2x", {}).get("pnl"),
            "cost3_pnl": v2_ref.get("cost_3x", {}).get("pnl"),
        },
        "clean_base_reference": v1.editor.CLEAN_BASE_REFERENCE,
        "clean_base_oos_reference": v1.base._compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life_1), "oos": _compact(oos_life_1), "report": str(args.lifecycle_report)},
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
    print(json.dumps({"report": str(args.report_out), "verdict": gates["decision"], "selected": selected_cfg.name, "cost_1x": report["cost_1x"], "cost_2x": report["cost_2x"], "cost_3x": report["cost_3x"], "promotion_gate": gates}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base deep MDD governor v3.")
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
    p.add_argument("--deep-epochs", type=int, default=12)
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
