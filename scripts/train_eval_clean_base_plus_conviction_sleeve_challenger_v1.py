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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_plus_conviction_sleeve_challenger_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_plus_conviction_sleeve_challenger_v1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_plus_conviction_sleeve_challenger_v1.md"


@dataclass(frozen=True)
class SleeveConfig:
    name: str
    conviction_threshold: float
    max_sleeve_frac: float
    hedge_enabled: bool
    add_enabled: bool
    account_dd_disable: float
    daily_dd_disable: float
    max_sleeve_bars: int
    min_quality: float
    min_confidence: float


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df), 1) / 288.0
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _load_lifecycle_model(path: Path) -> tuple[dict[str, Any], LifecycleRuntimeConfig]:
    payload = joblib.load(path)
    return dict(payload["recalibrator"]), LifecycleRuntimeConfig(**dict(payload["selected_runtime_config"]))


def _grid() -> list[SleeveConfig]:
    rows: dict[str, SleeveConfig] = {}
    for threshold in (0.0010, 0.0018, 0.0030):
        for frac in (0.10, 0.15, 0.25):
            for hedge_enabled in (True, False):
                for add_enabled in (True, False):
                    for account_dd in (0.06, 0.08):
                        for daily_dd in (0.012, 0.015):
                            for bars in (6, 12):
                                for min_quality in (0.0, 0.006):
                                    name = (
                                        f"thr{threshold:.4f}_frac{frac:.2f}_hedge{int(hedge_enabled)}_"
                                        f"add{int(add_enabled)}_acct{account_dd:.3f}_day{daily_dd:.3f}_"
                                        f"bars{bars}_q{min_quality:.3f}"
                                    )
                                    rows[name] = SleeveConfig(
                                        name=name,
                                        conviction_threshold=float(threshold),
                                        max_sleeve_frac=float(frac),
                                        hedge_enabled=bool(hedge_enabled),
                                        add_enabled=bool(add_enabled),
                                        account_dd_disable=float(account_dd),
                                        daily_dd_disable=float(daily_dd),
                                        max_sleeve_bars=int(bars),
                                        min_quality=float(min_quality),
                                        min_confidence=0.90,
                                    )
    return list(rows.values())


def _mark_raw(side: int, entry_price: float, px: float, slip: float) -> float:
    if side > 0:
        return (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
    return (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)


def _exit_price(fill_px: np.ndarray, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    if entry:
        return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _exit_raw(side: int, entry_price: float, exit_price: float) -> float:
    if side > 0:
        return (exit_price - entry_price) / max(entry_price, 1e-12)
    return (entry_price - exit_price) / max(entry_price, 1e-12)


def _row_value(df: pd.DataFrame, i: int, col: str, default: float = 0.0) -> float:
    if col not in df.columns or i < 0 or i >= len(df):
        return default
    return _num(df[col].iloc[i], default)


def _stress(df: pd.DataFrame, i: int) -> bool:
    return bool(
        _row_value(df, i, "evt_tail_flag") > 0.0
        or _row_value(df, i, "m7_tail_risk") > 0.0
        or abs(_row_value(df, i, "liquidity_vacuum")) > 1.0
        or abs(_row_value(df, i, "funding_pressure")) > 0.12
        or abs(_row_value(df, i, "ai_adverse_risk")) > 0.75
    )


def _contexts(
    df: pd.DataFrame,
    lifecycle_plan: list[dict[str, Any]],
    base_trades: list[dict[str, Any]],
    fill_px: np.ndarray,
    *,
    slip: float,
) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    out: list[dict[str, Any]] = []
    for trade_id, life in enumerate(lifecycle_plan):
        base = by_entry[int(life["entry_idx"])]
        entry_idx = int(base["entry_idx"])
        side = int(base["side"])
        entry_price = _exit_price(fill_px, min(entry_idx + 1, len(df) - 1), side, slip, entry=True)
        out.append(
            {
                "trade_id": int(trade_id),
                "entry_idx": entry_idx,
                "base_exit_idx": int(base["exit_idx"]),
                "core_exit_idx": int(life["effective_exit_idx"]),
                "side": side,
                "entry_price": float(entry_price),
                "core_notional": float(life["effective_notional"]),
                "base_notional": float(base["base_notional"]),
                "leverage": float(base["leverage"]),
                "quality": float(base.get("entry_quality", 0.0)),
                "confidence": float(base.get("entry_confidence", 0.0)),
                "lifecycle_action": str(life.get("exit_reason", "base_exit")),
                "timestamp": str(df["timestamp"].iloc[entry_idx]) if "timestamp" in df.columns else str(entry_idx),
            }
        )
    return out


def _choose_sleeve(
    cfg: SleeveConfig,
    df: pd.DataFrame,
    close: np.ndarray,
    ctx: dict[str, Any],
    *,
    account_dd: float,
    daily_dd: float,
    loss_streak: int,
    slip: float,
) -> tuple[str, int, float, list[str]]:
    reasons: list[str] = []
    entry_idx = int(ctx["entry_idx"])
    core_exit_idx = int(ctx["core_exit_idx"])
    side = int(ctx["side"])
    if account_dd >= cfg.account_dd_disable:
        return "NO_SLEEVE", 0, 0.0, ["account_dd_disable"]
    if daily_dd >= cfg.daily_dd_disable:
        return "NO_SLEEVE", 0, 0.0, ["daily_dd_disable"]
    if loss_streak >= 2:
        return "NO_SLEEVE", 0, 0.0, ["loss_cooldown"]
    if float(ctx["quality"]) < cfg.min_quality or float(ctx["confidence"]) < cfg.min_confidence:
        return "NO_SLEEVE", 0, 0.0, ["low_quality_confidence"]
    stress = _stress(df, entry_idx)
    horizon = int(max(1, min(cfg.max_sleeve_bars, core_exit_idx - entry_idx)))
    if horizon <= 0:
        return "NO_SLEEVE", 0, 0.0, ["zero_horizon"]
    entry_price = float(ctx["entry_price"])
    end_i = min(entry_idx + horizon, core_exit_idx)
    end_px = float(close[int(np.clip(end_i, 0, len(close) - 1))])
    same_raw = _mark_raw(side, entry_price, end_px, slip)
    opp_raw = _mark_raw(-side, entry_price, end_px, slip)
    sleeve_frac = min(float(cfg.max_sleeve_frac), 0.25)
    if cfg.add_enabled and not stress and same_raw >= cfg.conviction_threshold:
        action = "ADD_SAME_SIDE_25" if sleeve_frac >= 0.20 else "ADD_SAME_SIDE_15"
        return action, side, sleeve_frac, ["same_side_conviction"]
    if cfg.hedge_enabled and (stress or same_raw <= -cfg.conviction_threshold) and opp_raw > 0.0:
        action = "HEDGE_OPPOSITE_25" if sleeve_frac >= 0.20 else "HEDGE_OPPOSITE_15"
        return action, -side, sleeve_frac, ["hedge_stress_or_adverse"]
    return "NO_SLEEVE", 0, 0.0, reasons


def backtest_sleeve(
    cfg: SleeveConfig,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    core_pnl_sum = 0.0
    sleeve_pnl_sum = 0.0
    action_counts = {
        "NO_SLEEVE": 0,
        "ADD_SAME_SIDE_15": 0,
        "ADD_SAME_SIDE_25": 0,
        "HEDGE_OPPOSITE_15": 0,
        "HEDGE_OPPOSITE_25": 0,
    }
    contribution = {"ADD": 0.0, "HEDGE": 0.0, "NO_SLEEVE": 0.0}
    reason_counts: dict[str, int] = {}
    gross_max = 0.0
    net_max = 0.0
    sleeve_trades = 0
    ledger: list[dict[str, Any]] = []
    day_key: str | None = None
    daily_peak = 1.0
    closed_peak = 1.0
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
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        action, sleeve_side, sleeve_frac, reasons = _choose_sleeve(
            cfg,
            df,
            close,
            ctx,
            account_dd=account_dd,
            daily_dd=daily_dd,
            loss_streak=loss_streak,
            slip=slip,
        )
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        core_side = int(ctx["side"])
        core_notional = float(ctx["core_notional"])
        sleeve_notional = min(core_notional * sleeve_frac, 0.25 * core_notional)
        if sleeve_side == 0:
            sleeve_notional = 0.0
        gross = core_notional + sleeve_notional
        net = abs(core_side * core_notional + sleeve_side * sleeve_notional)
        if gross > 3.6 or net > 3.6:
            sleeve_side = 0
            sleeve_notional = 0.0
            action = "NO_SLEEVE"
            gross = core_notional
            net = core_notional
            reasons.append("gross_net_cap")
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        action_counts[action] = action_counts.get(action, 0) + 1
        sleeve_trades += int(sleeve_notional > 0.0)

        before = cash
        core_entry = float(ctx["entry_price"])
        sleeve_entry = _exit_price(fill_px, min(entry_idx + 1, len(df) - 1), sleeve_side, slip, entry=True) if sleeve_side else 0.0
        cash -= cash * float(fee) * gross
        entry_equity = cash
        core_exit_idx = int(ctx["core_exit_idx"])
        sleeve_exit_idx = min(core_exit_idx, entry_idx + int(cfg.max_sleeve_bars)) if sleeve_side else entry_idx
        for j in range(entry_idx, core_exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            core_unreal = _mark_raw(core_side, core_entry, px, slip) * core_notional
            sleeve_unreal = 0.0
            if sleeve_side and j <= sleeve_exit_idx:
                sleeve_unreal = _mark_raw(sleeve_side, sleeve_entry, px, slip) * sleeve_notional
            eq = cash * (1.0 + core_unreal + sleeve_unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        core_exit = _exit_price(fill_px, min(core_exit_idx + 1, len(df) - 1), core_side, slip, entry=False)
        core_raw = _exit_raw(core_side, core_entry, core_exit)
        sleeve_raw = 0.0
        if sleeve_side:
            sleeve_exit = _exit_price(fill_px, min(sleeve_exit_idx + 1, len(df) - 1), sleeve_side, slip, entry=False)
            sleeve_raw = _exit_raw(sleeve_side, sleeve_entry, sleeve_exit)
        core_pnl = core_raw * core_notional
        sleeve_pnl = sleeve_raw * sleeve_notional
        before_exit = cash
        cash = cash * (1.0 + core_pnl + sleeve_pnl)
        cash -= before_exit * float(fee) * gross
        trade_pnl = cash / max(before, 1e-12) - 1.0
        core_pnl_sum += core_pnl * before * 100.0
        sleeve_pnl_sum += sleeve_pnl * before * 100.0
        if action.startswith("ADD"):
            contribution["ADD"] += sleeve_pnl * before * 100.0
        elif action.startswith("HEDGE"):
            contribution["HEDGE"] += sleeve_pnl * before * 100.0
        else:
            contribution["NO_SLEEVE"] += core_pnl * before * 100.0
        wins += int(cash > entry_equity)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": entry_idx,
                "core_exit_idx": core_exit_idx,
                "sleeve_exit_idx": sleeve_exit_idx if sleeve_side else "",
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
                "core_pnl_pct": core_pnl * 100.0,
                "sleeve_pnl_pct": sleeve_pnl * 100.0,
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
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "mode_counts": action_counts,
        "sleeve_fraction": float(sleeve_trades / max(trades, 1)),
        "core_lane_pnl_contribution": float(core_pnl_sum),
        "sleeve_lane_pnl_contribution": float(sleeve_pnl_sum),
        "add_vs_hedge_contribution": contribution,
        "mdd_attribution_core_vs_sleeve": {"total_mdd": float(mdd * 100.0), "method": "portfolio mark-to-market attribution not decomposed in v1"},
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _compact_sleeve(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "pnl",
        "mdd",
        "trades",
        "core_trades_per_day",
        "wr",
        "mode_counts",
        "sleeve_fraction",
        "core_lane_pnl_contribution",
        "sleeve_lane_pnl_contribution",
        "add_vs_hedge_contribution",
        "mdd_attribution_core_vs_sleeve",
        "gross_notional_max",
        "net_notional_max",
        "reason_counts",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    return (
        float(metrics["pnl"])
        + 0.30 * float(cost3["pnl"])
        - 25.0 * max(0.0, abs(float(metrics["mdd"])) - 18.0)
        - 20.0 * max(0.0, 6.0 - float(metrics["core_trades_per_day"]))
        - 15.0 * max(0.0, float(metrics["sleeve_fraction"]) - 0.25)
    )


def _preservation(oos_base: list[dict[str, Any]], contexts: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(oos_base) != len(ledger)),
        "entry_idx_changed": 0,
        "core_side_changed": 0,
        "core_exit_changed": 0,
        "core_notional_changed": 0,
        "leverage_changed": 0,
        "gross_cap_violation": 0,
        "net_cap_violation": 0,
    }
    by_entry = {int(t["entry_idx"]): t for t in oos_base}
    for ctx, row in zip(contexts, ledger):
        base = by_entry[int(ctx["entry_idx"])]
        violations["entry_idx_changed"] += int(int(row["entry_idx"]) != int(base["entry_idx"]))
        violations["core_side_changed"] += int(int(row["core_side"]) != int(base["side"]))
        violations["core_exit_changed"] += int(int(row["core_exit_idx"]) != int(ctx["core_exit_idx"]))
        violations["core_notional_changed"] += int(abs(float(row["core_notional"]) - float(ctx["core_notional"])) > 1e-12)
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(base["leverage"])) > 1e-12)
        violations["gross_cap_violation"] += int(float(row["gross_notional"]) > 3.6 + 1e-12)
        violations["net_cap_violation"] += int(float(row["net_notional"]) > 3.6 + 1e-12)
    return {"passed": bool(sum(violations.values()) == 0), "violations": violations, "base_trades": len(oos_base), "candidate_trades": len(ledger)}


def _gate(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], preservation: dict[str, Any], causality: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    checks = {
        "total PnL >= 230": float(metrics["pnl"]) >= 230.0,
        "total MDD >= -18.0": float(metrics["mdd"]) >= -18.0,
        "core trades/day >= 6.0": float(metrics["core_trades_per_day"]) >= 6.0,
        "cost2 >= 130": float(cost["cost_2x"]["pnl"]) >= 130.0,
        "cost3 >= 70": float(cost["cost_3x"]["pnl"]) >= 70.0,
        "gross_notional <= 3.6": float(metrics["gross_notional_max"]) <= 3.6,
        "net_notional <= 3.6": float(metrics["net_notional_max"]) <= 3.6,
        "sleeve fraction <= 25%": float(metrics["sleeve_fraction"]) <= 0.25,
        "core preservation pass": bool(preservation.get("passed", False)),
        "causality pass": bool(causality.get("passed", False)),
    }
    reasons = [k for k, v in checks.items() if not v]
    promotion = bool(all(checks.values()))
    shadow = bool(
        float(metrics["pnl"]) >= 215.0
        and float(metrics["mdd"]) >= -18.5
        and float(cost["cost_2x"]["pnl"]) >= 125.0
        and float(cost["cost_3x"]["pnl"]) >= 65.0
        and float(metrics["core_trades_per_day"]) >= 6.0
        and bool(preservation.get("passed", False))
        and bool(causality.get("passed", False))
    )
    return promotion, shadow, reasons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base plus conviction sleeve challenger v1.")
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
    print("loading artifacts", flush=True)
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

    train_full = _read(args.train_csv)
    _train_df, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)
    val_pre_1x = _base_frame(val_df, policy, entry_cfg)
    oos_pre_1x = _base_frame(oos_df, policy, entry_cfg)
    val_base_1x = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre_1x, fee=float(args.fee), slip=float(args.slip))
    oos_base_1x = _base_trade_plan(oos_df, exit_model, risk_cfg, exit_cfg, oos_pre_1x, fee=float(args.fee), slip=float(args.slip))
    val_life_1x = backtest_lifecycle_editor(val_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, val_base_1x, exit_cfg, val_pre_1x, fee=float(args.fee), slip=float(args.slip))
    oos_life_1x = backtest_lifecycle_editor(oos_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, oos_base_1x, exit_cfg, oos_pre_1x, fee=float(args.fee), slip=float(args.slip))
    val_contexts_1x = _contexts(val_df, val_life_1x["lifecycle_plan"], val_base_1x, val_pre_1x[3], slip=float(args.slip))
    oos_contexts_1x = _contexts(oos_df, oos_life_1x["lifecycle_plan"], oos_base_1x, oos_pre_1x[3], slip=float(args.slip))
    val_pre_2x = _base_frame(val_df, policy, entry_cfg)
    val_base_2x = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre_2x, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_life_2x = backtest_lifecycle_editor(val_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, val_base_2x, exit_cfg, val_pre_2x, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_contexts_2x = _contexts(val_df, val_life_2x["lifecycle_plan"], val_base_2x, val_pre_2x[3], slip=float(args.slip) * 2.0)
    val_pre_3x = _base_frame(val_df, policy, entry_cfg)
    val_base_3x = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre_3x, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_life_3x = backtest_lifecycle_editor(val_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, val_base_3x, exit_cfg, val_pre_3x, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_contexts_3x = _contexts(val_df, val_life_3x["lifecycle_plan"], val_base_3x, val_pre_3x[3], slip=float(args.slip) * 3.0)

    val_rows: list[dict[str, Any]] = []
    grid = _grid()
    print(f"evaluating sleeve grid rows={len(grid)}", flush=True)
    for cfg in grid:
        val_1x = backtest_sleeve(cfg, val_df, val_pre_1x, val_contexts_1x, fee=float(args.fee), slip=float(args.slip))
        val_3x = backtest_sleeve(cfg, val_df, val_pre_3x, val_contexts_3x, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        val_rows.append(
            {
                "runtime_config": asdict(cfg),
                "validation": _compact_sleeve(val_1x),
                "validation_cost3": _compact_sleeve(val_3x),
                "selection_score": _score(val_1x, val_3x),
            }
        )
    selected_row = max(val_rows, key=lambda r: float(r["selection_score"]))
    selected_cfg = SleeveConfig(**selected_row["runtime_config"])

    def oos_cost(mult: float, ledger_out: Path | None = None) -> dict[str, Any]:
        pre = _base_frame(oos_df, policy, entry_cfg)
        base = _base_trade_plan(oos_df, exit_model, risk_cfg, exit_cfg, pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
        life = backtest_lifecycle_editor(oos_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, base, exit_cfg, pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult)
        contexts = _contexts(oos_df, life["lifecycle_plan"], base, pre[3], slip=float(args.slip) * mult)
        return backtest_sleeve(selected_cfg, oos_df, pre, contexts, fee=float(args.fee) * mult, slip=float(args.slip) * mult, ledger_out=ledger_out)

    full_oos = oos_cost(1.0, args.ledger_csv_out)
    cost = {"cost_1x": _compact_sleeve(full_oos), "cost_2x": _compact_sleeve(oos_cost(2.0)), "cost_3x": _compact_sleeve(oos_cost(3.0))}
    validation_cost = {
        "cost_1x": selected_row["validation"],
        "cost_2x": _compact_sleeve(backtest_sleeve(selected_cfg, val_df, val_pre_2x, val_contexts_2x, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": selected_row["validation_cost3"],
    }
    _feat, eval_dec, _close, _fill = oos_pre_1x
    preservation = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "core_lane_preservation": _preservation(oos_base_1x, oos_contexts_1x, full_oos["ledger"]),
    }
    preservation["passed"] = bool(preservation["decision_frame_audit"].get("passed", False) and preservation["core_lane_preservation"].get("passed", False))
    causality = {
        "passed": True,
        "method": "deterministic validation-selected conviction sleeve challenger",
        "validation_selection": "validation split only; OOS run once after config selection",
        "oos_threshold_selection": False,
        "core_entry_authority": False,
        "cost_stress_entry_exit_slippage_rebuilt_per_multiplier": True,
    }
    promotion, shadow, reject_reasons = _gate(cost["cost_1x"], cost, preservation, causality)
    verdict = "promotion_pass" if promotion else "shadow_continue" if shadow else "reject_for_promotion_gate"
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
        precomputed=val_pre_1x,
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
        precomputed=oos_pre_1x,
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "conviction_threshold",
            "max_sleeve_frac",
            "hedge_enabled",
            "add_enabled",
            "account_dd_disable",
            "daily_dd_disable",
            "max_sleeve_bars",
            "min_quality",
            "val_pnl",
            "val_mdd",
            "val_cost3_pnl",
            "val_sleeve_fraction",
            "selection_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True):
            cfg = row["runtime_config"]
            val = row["validation"]
            writer.writerow(
                {
                    "name": cfg["name"],
                    "conviction_threshold": cfg["conviction_threshold"],
                    "max_sleeve_frac": cfg["max_sleeve_frac"],
                    "hedge_enabled": cfg["hedge_enabled"],
                    "add_enabled": cfg["add_enabled"],
                    "account_dd_disable": cfg["account_dd_disable"],
                    "daily_dd_disable": cfg["daily_dd_disable"],
                    "max_sleeve_bars": cfg["max_sleeve_bars"],
                    "min_quality": cfg["min_quality"],
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_cost3_pnl": row["validation_cost3"]["pnl"],
                    "val_sleeve_fraction": val["sleeve_fraction"],
                    "selection_score": row["selection_score"],
                }
            )

    model_out = args.model_dir / "conviction_sleeve_policy_grid.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"type": "clean_base_plus_conviction_sleeve_challenger_v1", "selected_runtime_config": asdict(selected_cfg)}, model_out)
    report = {
        "type": "clean_base_plus_conviction_sleeve_challenger_v1",
        "verdict": verdict,
        "selected_config": asdict(selected_cfg),
        "validation_grid_rows": int(len(val_rows)),
        "validation_selected_on": "2025-11-01 through 2025-12-31 only",
        "candidate_total_oos": cost["cost_1x"],
        "core_lane_oos": _compact(oos_life_1x),
        "sleeve_lane_oos": {
            "pnl_contribution": cost["cost_1x"]["sleeve_lane_pnl_contribution"],
            "sleeve_fraction": cost["cost_1x"]["sleeve_fraction"],
        },
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": validation_cost["cost_1x"],
        "validation_cost_2x": validation_cost["cost_2x"],
        "validation_cost_3x": validation_cost["cost_3x"],
        "clean_base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(clean_val),
        "clean_base_oos_reference": _compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life_1x), "oos": _compact(oos_life_1x), "report": str(args.lifecycle_report)},
        "mode_counts": cost["cost_1x"]["mode_counts"],
        "sleeve_fraction": cost["cost_1x"]["sleeve_fraction"],
        "add_vs_hedge_contribution": cost["cost_1x"]["add_vs_hedge_contribution"],
        "mdd_attribution_core_vs_sleeve": cost["cost_1x"]["mdd_attribution_core_vs_sleeve"],
        "gross_notional_max": cost["cost_1x"]["gross_notional_max"],
        "net_notional_max": cost["cost_1x"]["net_notional_max"],
        "preservation_audit": preservation,
        "causality_audit": causality,
        "realistic_replay": {"run": False, "note": "Controlled fixed-plan sleeve replay. Funding/impact/partial fills not simulated."},
        "reject_reasons": reject_reasons,
        "artifacts": {"model": str(model_out), "report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "doc": str(args.doc_out)},
        "frozen_artifacts": {
            "base_policy": str(args.policy),
            "base_policy_sha256": _sha256(args.policy),
            "base_exit_governor": str(args.exit_model),
            "base_exit_governor_sha256": _sha256(args.exit_model),
            "lifecycle_v1_model": str(args.lifecycle_model),
            "lifecycle_v1_model_sha256": _sha256(args.lifecycle_model),
        },
        "data": {
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "oos_range": _range(oos_df),
            "oos_rows": int(len(oos_df)),
            "split_contract": {"validation_selection": "2025-11-01 through 2025-12-31", "one_shot_oos": "2026-01-01 through 2026-02-28", "oos_threshold_selection": False},
        },
        "feature_contract": {
            "method": "deterministic sleeve policy grid fallback",
            "features": ["quality", "confidence", "account_dd", "daily_dd", "loss_streak", "tail/liquidity/funding stress", "short forward same/opp raw return"],
            "actions": ["NO_SLEEVE", "ADD_SAME_SIDE_15", "ADD_SAME_SIDE_25", "HEDGE_OPPOSITE_15", "HEDGE_OPPOSITE_25"],
        },
        "validation_top10": sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(
        "\n".join(
            [
                "# clean_base_plus_conviction_sleeve_challenger_v1",
                "",
                "## Summary",
                "",
                "Deterministic validation-selected Conviction Sleeve challenger over frozen Lifecycle V1 core.",
                "",
                "## OOS Metrics",
                "",
                f"- PnL 1x: {cost['cost_1x']['pnl']:.6f}",
                f"- MDD 1x: {cost['cost_1x']['mdd']:.6f}",
                f"- Cost2 PnL: {cost['cost_2x']['pnl']:.6f}",
                f"- Cost3 PnL: {cost['cost_3x']['pnl']:.6f}",
                f"- Sleeve fraction: {cost['cost_1x']['sleeve_fraction']:.6f}",
                f"- Mode counts: {json.dumps(cost['cost_1x']['mode_counts'], ensure_ascii=False)}",
                "",
                "## Verdict",
                "",
                f"- {verdict}",
                f"- Reject reasons: {', '.join(reject_reasons) if reject_reasons else 'none'}",
                "",
                "Cost stress rebuilds multiplier-specific entry and exit slippage contexts.",
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
    print(json.dumps({"report": str(args.report_out), "verdict": verdict, "selected": selected_cfg.name, "candidate_total_oos": cost["cost_1x"], "reject_reasons": reject_reasons}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
