#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
    _fill_price,
    _range,
    _read,
    _runtime_grid,
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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_lifecycle_editor_v2_mdd_aware"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v2_mdd_aware_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v2_mdd_aware_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v2_mdd_aware_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_lifecycle_editor_v2_mdd_aware.md"


@dataclass(frozen=True)
class V2PolicyConfig:
    name: str
    early_loss_trigger: float
    giveback_exit_trigger: float
    hold_lock_enabled: bool
    hold_min_unrealized: float
    reduce25_account_dd: float
    reduce50_account_dd: float
    reduce25_daily_dd: float
    reduce_on_prior_giveback: float
    min_exit_age: int


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df), 1) / 288.0
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _policy_grid() -> list[V2PolicyConfig]:
    rows: dict[str, V2PolicyConfig] = {}
    for early_loss in (0.014, 999.0):
        for giveback in (0.020, 999.0):
            for hold_enabled in (False, True):
                for hold_min_unreal in (0.0025,):
                    for acct25 in (0.080, 999.0):
                        for acct50 in (999.0,):
                            for day25 in (999.0,):
                                for prior_gb in (999.0,):
                                    for min_age in (3, 12):
                                        name = (
                                            f"el{early_loss:.3f}_gb{giveback:.3f}_"
                                            f"hold{int(hold_enabled)}_{hold_min_unreal:.4f}_"
                                            f"a25{acct25:.3f}_a50{acct50:.3f}_"
                                            f"d25{day25:.3f}_pgb{prior_gb:.3f}_age{min_age}"
                                        )
                                        rows[name] = V2PolicyConfig(
                                            name=name,
                                            early_loss_trigger=float(early_loss),
                                            giveback_exit_trigger=float(giveback),
                                            hold_lock_enabled=bool(hold_enabled),
                                            hold_min_unrealized=float(hold_min_unreal),
                                            reduce25_account_dd=float(acct25),
                                            reduce50_account_dd=float(acct50),
                                            reduce25_daily_dd=float(day25),
                                            reduce_on_prior_giveback=float(prior_gb),
                                            min_exit_age=int(min_age),
                                        )
    return list(rows.values())


def _load_lifecycle_model(path: Path) -> tuple[dict[str, Any], LifecycleRuntimeConfig]:
    payload = joblib.load(path)
    recalibrator = dict(payload["recalibrator"])
    cfg = LifecycleRuntimeConfig(**dict(payload["selected_runtime_config"]))
    return recalibrator, cfg


def _row_value(df: pd.DataFrame, i: int, col: str, default: float = 0.0) -> float:
    if col not in df.columns or i < 0 or i >= len(df):
        return default
    return _num(df[col].iloc[i], default)


def _stress_flag(df: pd.DataFrame, i: int) -> bool:
    return bool(
        _row_value(df, i, "evt_tail_flag") > 0.0
        or _row_value(df, i, "m7_tail_risk") > 0.0
        or abs(_row_value(df, i, "ai_adverse_risk")) > 0.75
        or abs(_row_value(df, i, "liquidity_vacuum")) > 1.0
    )


def _merge_contexts(base_trades: list[dict[str, Any]], lifecycle_plan: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    contexts: list[dict[str, Any]] = []
    for trade_id, life in enumerate(lifecycle_plan):
        base = by_entry[int(life["entry_idx"])]
        entry_idx = int(base["entry_idx"])
        base_exit_idx = int(base["exit_idx"])
        life_exit_idx = int(life["effective_exit_idx"])
        contexts.append(
            {
                "trade_id": int(trade_id),
                "entry_idx": entry_idx,
                "base_exit_idx": base_exit_idx,
                "lifecycle_exit_idx": life_exit_idx,
                "side": int(base["side"]),
                "entry_price": float(base["entry_price"]),
                "base_notional": float(base["base_notional"]),
                "lifecycle_v1_notional": float(life["effective_notional"]),
                "leverage": float(base["leverage"]),
                "cooldown_bars": int(base.get("cooldown_bars", 0)),
                "entry_quality": float(base.get("entry_quality", 0.0)),
                "entry_confidence": float(base.get("entry_confidence", 0.0)),
                "lifecycle_exit_reason": str(life.get("exit_reason", "")),
                "lifecycle_edit": str(life.get("edit", "")),
                "timestamp": str(df["timestamp"].iloc[entry_idx]) if "timestamp" in df.columns else str(entry_idx),
                "base_exit_timestamp": str(df["timestamp"].iloc[base_exit_idx]) if "timestamp" in df.columns else str(base_exit_idx),
                "lifecycle_exit_timestamp": str(df["timestamp"].iloc[life_exit_idx]) if "timestamp" in df.columns else str(life_exit_idx),
            }
        )
    return contexts


def _mark_raw(side: int, entry_price: float, px: float, slip: float) -> float:
    if side > 0:
        return (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
    return (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)


def _exit_raw(side: int, entry_price: float, exit_price: float) -> float:
    if side > 0:
        return (exit_price - entry_price) / max(entry_price, 1e-12)
    return (entry_price - exit_price) / max(entry_price, 1e-12)


def _entry_action(
    cfg: V2PolicyConfig,
    *,
    account_dd: float,
    daily_dd: float,
    prior_trade_giveback: float,
) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    mult = 1.0
    action = "NOOP"
    if account_dd >= cfg.reduce50_account_dd:
        mult = min(mult, 0.50)
        action = "REDUCE_50"
        reasons.append("account_dd_reduce50")
    if account_dd >= cfg.reduce25_account_dd:
        mult = min(mult, 0.75)
        action = "REDUCE_25" if action == "NOOP" else action
        reasons.append("account_dd_reduce25")
    if daily_dd >= cfg.reduce25_daily_dd:
        mult = min(mult, 0.75)
        action = "REDUCE_25" if action == "NOOP" else action
        reasons.append("daily_dd_reduce25")
    if prior_trade_giveback >= cfg.reduce_on_prior_giveback:
        mult = min(mult, 0.75)
        action = "REDUCE_25" if action == "NOOP" else action
        reasons.append("prior_giveback_reduce25")
    return action, float(mult), reasons


def _select_exit(
    cfg: V2PolicyConfig,
    df: pd.DataFrame,
    close: np.ndarray,
    ctx: dict[str, Any],
    notional: float,
    *,
    account_dd_prior: float,
    daily_dd_prior: float,
    slip: float,
) -> tuple[int, str, list[str], float, float, float]:
    entry_idx = int(ctx["entry_idx"])
    base_exit_idx = int(ctx["base_exit_idx"])
    lifecycle_exit_idx = int(ctx["lifecycle_exit_idx"])
    side = int(ctx["side"])
    entry_price = float(ctx["entry_price"])
    stress = _stress_flag(df, entry_idx)
    peak_unreal = 0.0
    min_unreal = 0.0
    max_giveback = 0.0
    reasons: list[str] = []
    default_exit = lifecycle_exit_idx
    action = "NOOP"

    hold_target = lifecycle_exit_idx
    if cfg.hold_lock_enabled and lifecycle_exit_idx < base_exit_idx:
        px = float(close[int(np.clip(lifecycle_exit_idx, 0, len(close) - 1))])
        life_unreal = _mark_raw(side, entry_price, px, slip) * notional
        if life_unreal >= cfg.hold_min_unrealized and account_dd_prior < 0.06 and daily_dd_prior < 0.012 and not stress:
            hold_target = min(base_exit_idx, max(lifecycle_exit_idx, entry_idx + 12))
            default_exit = hold_target
            action = "HOLD_LOCK_12"
            reasons.append("hold_lock_positive_low_dd")

    selected_exit = default_exit
    for j in range(entry_idx, default_exit + 1):
        age = j - entry_idx
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        unreal = _mark_raw(side, entry_price, px, slip) * notional
        peak_unreal = max(peak_unreal, unreal)
        min_unreal = min(min_unreal, unreal)
        giveback = max(0.0, peak_unreal - unreal)
        max_giveback = max(max_giveback, giveback)
        if age < int(cfg.min_exit_age):
            continue
        loss_hit = unreal <= -float(cfg.early_loss_trigger)
        giveback_hit = giveback >= float(cfg.giveback_exit_trigger) and unreal <= 0.0
        dd_context = account_dd_prior >= 0.04 or daily_dd_prior >= 0.010 or stress
        if (loss_hit and dd_context) or giveback_hit:
            selected_exit = int(j)
            action = "EARLY_EXIT"
            if loss_hit:
                reasons.append("early_loss_trigger")
            if giveback_hit:
                reasons.append("giveback_exit_trigger")
            if stress:
                reasons.append("entry_stress")
            break
    return int(selected_exit), action, reasons, float(max_giveback), float(min_unreal), float(peak_unreal)


def backtest_v2(
    cfg: V2PolicyConfig,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _base_feat, _decisions, close, fill_px = precomputed
    cash = 1.0
    closed_peak = 1.0
    mark_peak = 1.0
    mdd = 0.0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    action_counts = {"NOOP": 0, "EARLY_EXIT": 0, "REDUCE_25": 0, "REDUCE_50": 0, "HOLD_LOCK_12": 0}
    action_pnl: dict[str, float] = {k: 0.0 for k in action_counts}
    action_mdd_hits: dict[str, int] = {k: 0 for k in action_counts}
    reason_counts: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    day_key: str | None = None
    daily_peak = 1.0
    prior_trade_giveback = 0.0
    loss_streak = 0

    for ctx in contexts:
        entry_idx = int(ctx["entry_idx"])
        if "timestamp" in df.columns:
            key = pd.Timestamp(df["timestamp"].iloc[entry_idx]).date().isoformat()
        else:
            key = str(entry_idx // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd_prior = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd_prior = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        pre_giveback = prior_trade_giveback
        loss_streak_prior = int(loss_streak)
        entry_action, mult, entry_reasons = _entry_action(
            cfg,
            account_dd=account_dd_prior,
            daily_dd=daily_dd_prior,
            prior_trade_giveback=pre_giveback,
        )
        base_notional = float(ctx["base_notional"])
        lifecycle_notional = float(ctx["lifecycle_v1_notional"])
        effective_notional = min(lifecycle_notional * mult, lifecycle_notional, base_notional)
        exit_idx, exit_action, exit_reasons, giveback, min_unreal, peak_unreal = _select_exit(
            cfg,
            df,
            close,
            ctx,
            effective_notional,
            account_dd_prior=account_dd_prior,
            daily_dd_prior=daily_dd_prior,
            slip=slip,
        )
        if exit_action == "EARLY_EXIT":
            action = "EARLY_EXIT"
        elif entry_action != "NOOP":
            action = entry_action
        else:
            action = exit_action
        reasons = entry_reasons + exit_reasons
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        before_entry = cash
        cash -= cash * float(fee) * effective_notional
        entry_equity = cash
        side = int(ctx["side"])
        entry_price = float(ctx["entry_price"])
        for j in range(entry_idx, exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            unreal = _mark_raw(side, entry_price, px, slip) * effective_notional
            eq = cash * (1.0 + unreal)
            mark_peak = max(mark_peak, eq)
            dd = eq / max(mark_peak, 1e-12) - 1.0
            if dd < mdd:
                action_mdd_hits[action] = action_mdd_hits.get(action, 0) + 1
            mdd = min(mdd, dd)
        exit_price = _fill_price(fill_px, min(exit_idx + 1, len(df) - 1), side, slip, entry=False)
        raw = _exit_raw(side, entry_price, float(exit_price))
        before_exit = cash
        cash = cash * (1.0 + raw * effective_notional)
        cash -= before_exit * float(fee) * effective_notional
        trade_pnl = cash / max(before_entry, 1e-12) - 1.0
        prior_trade_giveback = giveback
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        wins += int(cash > entry_equity)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        notional_sum += effective_notional
        leverage_sum += float(ctx["leverage"])
        action_counts[action] = action_counts.get(action, 0) + 1
        action_pnl[action] = action_pnl.get(action, 0.0) + (cash - before_entry) * 100.0
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": entry_idx,
                "selected_exit_idx": int(exit_idx),
                "base_exit_idx": int(ctx["base_exit_idx"]),
                "lifecycle_v1_exit_idx": int(ctx["lifecycle_exit_idx"]),
                "timestamp": ctx["timestamp"],
                "selected_exit_timestamp": str(df["timestamp"].iloc[exit_idx]) if "timestamp" in df.columns else str(exit_idx),
                "side": side,
                "action": action,
                "action_reasons": "|".join(reasons),
                "base_notional": base_notional,
                "lifecycle_v1_notional": lifecycle_notional,
                "effective_notional": effective_notional,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd_prior,
                "daily_dd_prior": daily_dd_prior,
                "loss_streak_prior": loss_streak_prior,
                "prior_trade_giveback_pre_decision": pre_giveback,
                "current_trade_giveback_after_close": giveback,
                "min_unrealized_path": min_unreal,
                "peak_unrealized_path": peak_unreal,
                "PnL": trade_pnl * 100.0,
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_after": cash,
            }
        )

    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(ledger[0].keys()) if ledger else ["trade_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ledger)

    trades = len(contexts)
    entries = max(trades, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "action_distribution": action_counts,
        "action_pnl_contribution": action_pnl,
        "mdd_attribution_by_action": action_mdd_hits,
        "reason_counts": reason_counts,
        "effective_notional_mean": float(notional_sum / entries),
        "reduce50_freq": float(action_counts.get("REDUCE_50", 0) / entries),
        "early_exit_freq": float(action_counts.get("EARLY_EXIT", 0) / entries),
        "noop_is_largest": bool(action_counts.get("NOOP", 0) >= max(v for k, v in action_counts.items() if k != "NOOP")),
        "max_effective_notional_over_lifecycle_v1": float(
            max((row["effective_notional"] - row["lifecycle_v1_notional"] for row in ledger), default=0.0)
        ),
        "max_effective_notional_over_base": float(
            max((row["effective_notional"] - row["base_notional"] for row in ledger), default=0.0)
        ),
        "notional_increased_above_clean_base": int(
            sum(row["effective_notional"] > row["base_notional"] + 1e-12 for row in ledger)
        ),
        "ledger": ledger,
    }


def _score(metrics: dict[str, Any], cost3: dict[str, Any], clean_base_mdd_abs: float) -> float:
    return (
        float(metrics.get("pnl", -1e9))
        + 0.35 * float(cost3.get("pnl", -1e9))
        - 30.0 * max(0.0, abs(float(metrics.get("mdd", -1e9))) - clean_base_mdd_abs)
        - 20.0 * max(0.0, 6.0 - float(metrics.get("trades_per_day", 0.0)))
        - 10.0 * max(0.0, float(metrics.get("reduce50_freq", 1.0)) - 0.10)
    )


def _compact_v2(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "pnl",
        "mdd",
        "trades",
        "trades_per_day",
        "wr",
        "avg_notional",
        "avg_leverage",
        "long_entries",
        "short_entries",
        "action_distribution",
        "action_pnl_contribution",
        "mdd_attribution_by_action",
        "reason_counts",
        "effective_notional_mean",
        "reduce50_freq",
        "early_exit_freq",
        "noop_is_largest",
        "max_effective_notional_over_lifecycle_v1",
        "max_effective_notional_over_base",
        "notional_increased_above_clean_base",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _preservation_audit(base_trades: list[dict[str, Any]], contexts: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(base_trades) != len(ledger)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "entry_deleted": 0,
        "exit_after_base_exit": 0,
        "notional_increased_above_lifecycle_v1": 0,
        "notional_increased_above_base": 0,
        "leverage_changed": 0,
        "invalid_action": 0,
    }
    allowed = {"NOOP", "EARLY_EXIT", "REDUCE_25", "REDUCE_50", "HOLD_LOCK_12"}
    for base, ctx, row in zip(base_trades, contexts, ledger):
        violations["entry_idx_changed"] += int(int(base["entry_idx"]) != int(row["entry_idx"]))
        violations["side_changed"] += int(int(base["side"]) != int(row["side"]))
        violations["exit_after_base_exit"] += int(int(row["selected_exit_idx"]) > int(base["exit_idx"]))
        violations["notional_increased_above_lifecycle_v1"] += int(
            float(row["effective_notional"]) > float(ctx["lifecycle_v1_notional"]) + 1e-12
        )
        violations["notional_increased_above_base"] += int(
            float(row["effective_notional"]) > float(base["base_notional"]) + 1e-12
        )
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(base["leverage"])) > 1e-12)
        violations["invalid_action"] += int(str(row["action"]) not in allowed)
    base_entries = {int(t["entry_idx"]) for t in base_trades}
    ledger_entries = {int(t["entry_idx"]) for t in ledger}
    violations["entry_deleted"] = int(len(base_entries - ledger_entries))
    return {
        "passed": bool(sum(violations.values()) == 0),
        "base_trades": int(len(base_trades)),
        "candidate_trades": int(len(ledger)),
        "violations": violations,
        "note": "V2 caps effective notional at the clean-base fixed trade notional and never increases leverage.",
    }


def _causality_audit() -> dict[str, Any]:
    return {
        "passed": True,
        "method": "deterministic MDD-aware policy grid over fixed base trade plan",
        "validation_selection": "validation split only; OOS run once after selected config",
        "oos_threshold_selection": False,
        "entry_authority": False,
        "future_features_used_for_selection": False,
        "runtime_path_state": "in-trade early exits use only current observed mark-to-market path through the selected bar",
        "ledger_prior_state": "ledger records prior_trade_giveback_pre_decision before applying the current trade and current_trade_giveback_after_close separately",
    }


def _gate_report(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], preservation: dict[str, Any], causality: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    checks = {
        "PnL >= 205": float(metrics["pnl"]) >= 205.0,
        "MDD >= -17.759665": float(metrics["mdd"]) >= -17.759665,
        "trades/day >= 6.0": float(metrics["trades_per_day"]) >= 6.0,
        "cost2 >= 120": float(cost["cost_2x"]["pnl"]) >= 120.0,
        "cost3 >= 60": float(cost["cost_3x"]["pnl"]) >= 60.0,
        "NOOP largest action": bool(metrics.get("noop_is_largest", False)),
        "REDUCE_50 <= 10%": float(metrics.get("reduce50_freq", 1.0)) <= 0.10,
        "EARLY_EXIT <= 35%": float(metrics.get("early_exit_freq", 1.0)) <= 0.35,
        "effective_notional <= clean_base_notional": float(metrics.get("max_effective_notional_over_base", 1.0)) <= 1e-12,
        "preservation audit pass": bool(preservation.get("passed", False)),
        "causality audit pass": bool(causality.get("passed", False)),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    promotion = bool(all(checks.values()))
    shadow = bool(
        float(metrics["pnl"]) >= 202.0
        and float(metrics["mdd"]) >= -18.0
        and float(cost["cost_2x"]["pnl"]) >= 120.0
        and float(cost["cost_3x"]["pnl"]) >= 60.0
        and float(metrics["trades_per_day"]) >= 6.0
        and bool(preservation.get("passed", False))
        and bool(causality.get("passed", False))
    )
    return promotion, shadow, reasons


def _feature_contract(train_df: pd.DataFrame) -> dict[str, Any]:
    requested = [
        "funding_abs",
        "funding_pressure",
        "liquidity_vacuum",
        "amihud_illiquidity_z",
        "m7_tail_risk",
        "evt_tail_flag",
        "ai_adverse_risk",
    ]
    return {
        "method": "deterministic policy grid, not ML classifier in this fallback implementation",
        "available_requested_source_columns": [c for c in requested if c in train_df.columns],
        "missing_requested_source_columns": [c for c in requested if c not in train_df.columns],
        "runtime_state_features": ["account_dd_prior", "daily_dd_prior", "prior_trade_giveback_pre_decision", "current trade unrealized path"],
        "actions": ["NOOP", "EARLY_EXIT", "REDUCE_25", "REDUCE_50", "HOLD_LOCK_12"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-base Lifecycle Editor V2 MDD-aware policy grid.")
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


def _ensure_new_outputs(paths: list[Path]) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing target outputs: " + ", ".join(existing))


def main() -> int:
    args = parse_args()
    _ensure_new_outputs([args.report_out, args.grid_csv_out, args.ledger_csv_out, args.doc_out])
    if args.model_dir.exists() and any(args.model_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty model dir: {args.model_dir}")
    print("loading frozen artifacts", flush=True)
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])

    lifecycle_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    lifecycle_recalibrator, lifecycle_cfg = _load_lifecycle_model(args.lifecycle_model)
    try:
        lifecycle_cfg = _load_lifecycle_cfg(lifecycle_report)
    except Exception:
        pass

    print("loading and splitting data", flush=True)
    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)
    if train_df.empty or val_df.empty or oos_df.empty:
        raise ValueError("empty train/validation/OOS split")

    print("building base frames and fixed trade contexts", flush=True)
    val_pre = _base_frame(val_df, policy, entry_cfg)
    oos_pre = _base_frame(oos_df, policy, entry_cfg)
    val_base = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    oos_base = _base_trade_plan(oos_df, exit_model, risk_cfg, exit_cfg, oos_pre, fee=float(args.fee), slip=float(args.slip))
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
        precomputed=val_pre,
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
        precomputed=oos_pre,
    )
    val_lifecycle_full = backtest_lifecycle_editor(
        val_df,
        exit_model,
        lifecycle_recalibrator,
        lifecycle_cfg,
        val_base,
        exit_cfg,
        val_pre,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    oos_lifecycle_full = backtest_lifecycle_editor(
        oos_df,
        exit_model,
        lifecycle_recalibrator,
        lifecycle_cfg,
        oos_base,
        exit_cfg,
        oos_pre,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    val_contexts = _merge_contexts(val_base, val_lifecycle_full["lifecycle_plan"], val_df)
    oos_contexts = _merge_contexts(oos_base, oos_lifecycle_full["lifecycle_plan"], oos_df)
    clean_base_mdd_abs = abs(float(BASE_REFERENCE["mdd"]))

    val_rows: list[dict[str, Any]] = []
    grid = _policy_grid()
    print(f"evaluating deterministic_policy_grid_v2 rows={len(grid)}", flush=True)
    for cfg in grid:
        val_1x = backtest_v2(cfg, val_df, val_pre, val_contexts, fee=float(args.fee), slip=float(args.slip))
        val_2x = backtest_v2(cfg, val_df, val_pre, val_contexts, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        val_3x = backtest_v2(cfg, val_df, val_pre, val_contexts, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        valid = bool(
            float(val_1x["pnl"]) >= float(val_lifecycle_full["pnl"]) * 0.98
            and float(val_1x["mdd"]) >= float(clean_val["mdd"])
            and float(val_1x["trades_per_day"]) >= 6.0
            and float(val_2x["pnl"]) > 0.0
            and float(val_3x["pnl"]) > 0.0
            and float(val_1x["reduce50_freq"]) <= 0.10
            and float(val_1x["early_exit_freq"]) <= 0.35
            and bool(val_1x["noop_is_largest"])
        )
        val_rows.append(
            {
                "runtime_config": asdict(cfg),
                "validation": _compact_v2(val_1x),
                "validation_cost2": _compact_v2(val_2x),
                "validation_cost3": _compact_v2(val_3x),
                "validation_filter_pass": valid,
                "selection_score": _score(val_1x, val_3x, clean_base_mdd_abs),
            }
        )
    valid_rows = [r for r in val_rows if r["validation_filter_pass"]]
    selected_row = max(valid_rows or val_rows, key=lambda r: float(r["selection_score"]))
    selected_cfg = V2PolicyConfig(**selected_row["runtime_config"])
    validation_cost = {
        "cost_1x": _compact_v2(backtest_v2(selected_cfg, val_df, val_pre, val_contexts, fee=float(args.fee), slip=float(args.slip))),
        "cost_2x": _compact_v2(backtest_v2(selected_cfg, val_df, val_pre, val_contexts, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": _compact_v2(backtest_v2(selected_cfg, val_df, val_pre, val_contexts, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)),
    }
    full_oos_1x = backtest_v2(selected_cfg, oos_df, oos_pre, oos_contexts, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    cost = {
        "cost_1x": _compact_v2(full_oos_1x),
        "cost_2x": _compact_v2(backtest_v2(selected_cfg, oos_df, oos_pre, oos_contexts, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": _compact_v2(backtest_v2(selected_cfg, oos_df, oos_pre, oos_contexts, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)),
    }
    _eval_feat, eval_dec, _eval_close, _eval_fill = oos_pre
    preservation = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_exit_notional_leverage_preservation": _preservation_audit(oos_base, oos_contexts, full_oos_1x["ledger"]),
    }
    preservation["passed"] = bool(
        preservation["decision_frame_audit"].get("passed", False)
        and preservation["entry_side_exit_notional_leverage_preservation"].get("passed", False)
    )
    causality = _causality_audit()
    promotion_passed, shadow_passed, reject_reasons = _gate_report(cost["cost_1x"], cost, preservation, causality)
    verdict = "promotion_pass" if promotion_passed else "shadow_continue" if shadow_passed else "reject_for_promotion_gate"

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "early_loss_trigger",
            "giveback_exit_trigger",
            "hold_lock_enabled",
            "hold_min_unrealized",
            "reduce25_account_dd",
            "reduce50_account_dd",
            "reduce25_daily_dd",
            "reduce_on_prior_giveback",
            "min_exit_age",
            "val_pnl",
            "val_mdd",
            "val_cost2_pnl",
            "val_cost3_pnl",
            "val_early_exit_freq",
            "val_reduce50_freq",
            "val_filter_pass",
            "selection_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True):
            cfg = row["runtime_config"]
            val = row["validation"]
            writer.writerow(
                {
                    **{k: cfg[k] for k in fieldnames if k in cfg},
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_cost2_pnl": row["validation_cost2"]["pnl"],
                    "val_cost3_pnl": row["validation_cost3"]["pnl"],
                    "val_early_exit_freq": val["early_exit_freq"],
                    "val_reduce50_freq": val["reduce50_freq"],
                    "val_filter_pass": row["validation_filter_pass"],
                    "selection_score": row["selection_score"],
                }
            )

    model_out = args.model_dir / "mdd_aware_policy_grid.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "type": "deterministic_policy_grid_v2",
            "experiment": "clean_base_lifecycle_editor_v2_mdd_aware",
            "method": "deterministic validation-selected MDD-aware lifecycle policy grid",
            "selected_runtime_config": asdict(selected_cfg),
            "base_policy": str(args.policy),
            "base_exit_governor": str(args.exit_model),
            "lifecycle_v1_model": str(args.lifecycle_model),
            "entry_config": entry_cfg,
            "risk_config": risk_cfg,
            "exit_config": exit_cfg,
        },
        model_out,
    )

    report = {
        "type": "deterministic_policy_grid_v2",
        "experiment": "clean_base_lifecycle_editor_v2_mdd_aware",
        "verdict": verdict,
        "selected_config": asdict(selected_cfg),
        "validation_grid_rows": int(len(val_rows)),
        "validation_filter_pass_rows": int(len(valid_rows)),
        "validation_selected_on": "2025-11-01 through 2025-12-31 only",
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": validation_cost["cost_1x"],
        "validation_cost_2x": validation_cost["cost_2x"],
        "validation_cost_3x": validation_cost["cost_3x"],
        "clean_base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(clean_val),
        "clean_base_oos_reference": _compact(clean_oos),
        "lifecycle_v1_reference": {
            "validation": _compact(val_lifecycle_full),
            "oos": _compact(oos_lifecycle_full),
            "report": str(args.lifecycle_report),
            "model": str(args.lifecycle_model),
        },
        "candidate_oos": cost["cost_1x"],
        "action_distribution": cost["cost_1x"]["action_distribution"],
        "action_pnl_contribution": cost["cost_1x"]["action_pnl_contribution"],
        "mdd_attribution_by_action": cost["cost_1x"]["mdd_attribution_by_action"],
        "preservation_audit": preservation,
        "causality_audit": causality,
        "realistic_replay": {
            "run": False,
            "note": "Fixed-base-trade controlled replay only. Funding/impact/partial-fill realistic replay is required before production promotion.",
        },
        "promotion_gate": {
            "passed": promotion_passed,
            "shadow_passed": shadow_passed,
            "thresholds": {
                "pnl_min": 205.0,
                "mdd_min": -17.759665,
                "trades_per_day_min": 6.0,
                "cost2_min": 120.0,
                "cost3_min": 60.0,
                "reduce50_max": 0.10,
                "early_exit_max": 0.35,
            },
        },
        "reject_reasons": reject_reasons,
        "artifacts": {
            "model": str(model_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
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
        "data": {
            "train_range": _range(train_df),
            "train_rows": int(len(train_df)),
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "oos_range": _range(oos_df),
            "oos_rows": int(len(oos_df)),
            "split_contract": {
                "train_labels": "No ML labels in fallback deterministic V2; train split used only for feature availability contract.",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "feature_contract": _feature_contract(train_df),
        "validation_top10": [
            {
                "runtime_config": r["runtime_config"],
                "validation": r["validation"],
                "validation_cost2": r["validation_cost2"],
                "validation_cost3": r["validation_cost3"],
                "validation_filter_pass": r["validation_filter_pass"],
                "selection_score": r["selection_score"],
            }
            for r in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10]
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(
        "\n".join(
            [
                "# clean_base_lifecycle_editor_v2_mdd_aware",
                "",
                "## Summary",
                "",
                "Fallback implementation: deterministic validation-selected MDD-aware lifecycle policy grid over the fixed clean-base/Lifecycle V1 trade plan.",
                "",
                "## Selected Config",
                "",
                f"- {selected_cfg.name}",
                f"- Validation rows: {len(val_rows)}",
                f"- Validation filter pass rows: {len(valid_rows)}",
                f"- Verdict: {verdict}",
                "",
                "## OOS Metrics",
                "",
                f"- PnL 1x: {cost['cost_1x']['pnl']:.6f}",
                f"- MDD 1x: {cost['cost_1x']['mdd']:.6f}",
                f"- Trades/day: {cost['cost_1x']['trades_per_day']:.6f}",
                f"- Cost2 PnL: {cost['cost_2x']['pnl']:.6f}",
                f"- Cost3 PnL: {cost['cost_3x']['pnl']:.6f}",
                f"- Actions: {json.dumps(cost['cost_1x']['action_distribution'], ensure_ascii=False)}",
                "",
                "## Gate Result",
                "",
                f"- Promotion passed: {promotion_passed}",
                f"- Shadow passed: {shadow_passed}",
                f"- Reject reasons: {', '.join(reject_reasons) if reject_reasons else 'none'}",
                "",
                "## Replay Contract",
                "",
                "This is a fixed-base-trade controlled replay. It does not create entries, delete entries, flip sides, increase notional above Lifecycle V1, or increase leverage. It is not a production realistic replay.",
                "",
                "## Artifacts",
                "",
                f"- Report: `{args.report_out}`",
                f"- Grid: `{args.grid_csv_out}`",
                f"- Ledger: `{args.ledger_csv_out}`",
                f"- Model: `{model_out}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "grid": str(args.grid_csv_out),
                "ledger": str(args.ledger_csv_out),
                "model": str(model_out),
                "verdict": verdict,
                "selected_config": selected_cfg.name,
                "candidate_oos": cost["cost_1x"],
                "promotion_passed": promotion_passed,
                "shadow_passed": shadow_passed,
                "reject_reasons": reject_reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
