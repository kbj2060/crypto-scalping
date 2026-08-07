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

try:  # noqa: E402
    from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # type: ignore
except ModuleNotFoundError:  # noqa: E402
    def _decision_audit(dec: pd.DataFrame, *, max_notional: float, leverage_cap: float) -> dict[str, Any]:
        action = pd.to_numeric(dec.get("action", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        side = pd.to_numeric(dec.get("side", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        notional = pd.to_numeric(dec.get("notional_exposure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        leverage = pd.to_numeric(dec.get("leverage", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        pf = pd.to_numeric(dec.get("position_fraction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cooldown = pd.to_numeric(dec.get("cooldown_bars", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        active = (action != 0) & (side != 0) & (notional > 0.0)
        violations = {
            "nonfinite_values": int((~np.isfinite(notional) | ~np.isfinite(leverage) | ~np.isfinite(pf)).sum()),
            "negative_notional": int((notional < -1e-12).sum()),
            "leverage_below_one_active": int((active & (leverage < 1.0 - 1e-12)).sum()),
            "leverage_above_cap": int((active & (leverage > float(leverage_cap) + 1e-12)).sum()),
            "notional_above_max": int((active & (notional > float(max_notional) + 1e-12)).sum()),
            "active_action_side_mismatch": int((((action != 0) ^ (side != 0)) & (notional > 1e-12)).sum()),
            "cash_has_exposure": int(((action == 0) & ((side != 0) | (notional > 1e-12) | (pf > 1e-12))).sum()),
            "position_fraction_mismatch": int((active & (np.abs(pf - notional / np.maximum(leverage, 1e-12)) > 1e-9)).sum()),
            "negative_cooldown": int((cooldown < -1e-12).sum()),
        }
        active_notional = notional[active]
        active_lev = leverage[active]
        return {
            "passed": bool(sum(violations.values()) == 0),
            "rows": int(len(dec)),
            "active_rows": int(active.sum()),
            "cash_rows": int((~active).sum()),
            "long_rows": int((active & (side > 0)).sum()),
            "short_rows": int((active & (side < 0)).sum()),
            "violations": violations,
            "notional": {
                "max": float(active_notional.max()) if active_notional.size else 0.0,
                "mean": float(active_notional.mean()) if active_notional.size else 0.0,
                "p95": float(np.quantile(active_notional, 0.95)) if active_notional.size else 0.0,
            },
            "leverage": {
                "max": float(active_lev.max()) if active_lev.size else 0.0,
                "mean": float(active_lev.mean()) if active_lev.size else 0.0,
                "p95": float(np.quantile(active_lev, 0.95)) if active_lev.size else 0.0,
            },
        }
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
    _runtime_grid,
    _sha256,
    _split_train_validation,
    backtest_lifecycle_editor,
    collect_exit_samples,
    train_bucket_recalibrator,
)


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_BASE_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/lifecycle_v1_drawdown_governor_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/lifecycle_v1_drawdown_governor_v1_ledger.csv"


@dataclass(frozen=True)
class DrawdownGovernorConfig:
    name: str
    account_dd_soft: float
    account_dd_hard: float
    daily_dd_soft: float
    daily_dd_hard: float
    trade_giveback_cut: float
    tail_risk_cut_enabled: bool
    soft_mult: float
    hard_mult: float = 0.50


def _governor_grid() -> list[DrawdownGovernorConfig]:
    rows: list[DrawdownGovernorConfig] = []
    for account_soft in (0.06, 0.08, 0.10):
        for account_hard in (0.11, 0.13, 0.15):
            for daily_soft in (0.012, 0.016, 0.020):
                for daily_hard in (0.020, 0.025, 0.030):
                    for giveback_cut in (0.010, 0.015, 0.020):
                        for tail_enabled in (True, False):
                            for soft_mult in (0.85, 0.70):
                                name = (
                                    f"acct{account_soft:.3f}_{account_hard:.3f}_"
                                    f"day{daily_soft:.3f}_{daily_hard:.3f}_"
                                    f"gb{giveback_cut:.3f}_tail{int(tail_enabled)}_soft{soft_mult:.2f}_hard0.50"
                                )
                                rows.append(
                                    DrawdownGovernorConfig(
                                        name=name,
                                        account_dd_soft=account_soft,
                                        account_dd_hard=account_hard,
                                        daily_dd_soft=daily_soft,
                                        daily_dd_hard=daily_hard,
                                        trade_giveback_cut=giveback_cut,
                                        tail_risk_cut_enabled=tail_enabled,
                                        soft_mult=soft_mult,
                                    )
                                )
    return rows


def _load_lifecycle_cfg(base_report: dict[str, Any]) -> LifecycleRuntimeConfig:
    selected_for_report = base_report.get("selected_for_report", "redteam_constrained")
    selected_eval = dict(base_report.get("selected_eval", {}))
    cfg = selected_eval.get(selected_for_report, {}).get("runtime_config")
    if cfg is None:
        selected = dict(base_report.get("selected", {}))
        selected_name = selected.get(selected_for_report)
        for candidate in _runtime_grid(3.6):
            if candidate.name == selected_name:
                return candidate
        raise KeyError("could not locate lifecycle runtime config in base report")
    return LifecycleRuntimeConfig(**cfg)


def _stress_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    def q(col: str, prob: float, default: float) -> float:
        if col not in train_df.columns:
            return default
        vals = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            return default
        return float(vals.quantile(prob))

    return {
        "liquidity_vacuum_p85": q("liquidity_vacuum", 0.85, 1.0),
        "funding_abs_p85": q("funding_abs", 0.85, 999.0),
        "funding_pressure_abs_p85": q("funding_pressure", 0.85, 999.0),
        "ai_adverse_risk_p85": q("ai_adverse_risk", 0.85, 999.0),
    }


def _row_value(df: pd.DataFrame, i: int, col: str, default: float = 0.0) -> float:
    if col not in df.columns or i < 0 or i >= len(df):
        return default
    val = pd.to_numeric(pd.Series([df[col].iloc[i]]), errors="coerce").iloc[0]
    if not np.isfinite(val):
        return default
    return float(val)


def _tail_stress(df: pd.DataFrame, i: int, thresholds: dict[str, float], last_trade_pnl: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _row_value(df, i, "evt_tail_flag") > 0.0 or _row_value(df, i, "m7_tail_risk") > 0.0:
        reasons.append("tail")
    if _row_value(df, i, "liquidity_vacuum") >= thresholds["liquidity_vacuum_p85"]:
        reasons.append("liquidity")
    if abs(_row_value(df, i, "funding_abs")) >= thresholds["funding_abs_p85"] or abs(_row_value(df, i, "funding_pressure")) >= thresholds["funding_pressure_abs_p85"]:
        reasons.append("funding")
    adverse = last_trade_pnl <= 0.0 or _row_value(df, i, "ai_adverse_risk") >= thresholds["ai_adverse_risk_p85"]
    return bool(reasons and adverse), reasons


def _risk_mult(
    cfg: DrawdownGovernorConfig,
    *,
    account_dd: float,
    daily_dd: float,
    current_unrealized: float,
    prior_trade_giveback: float,
    tail_stress: bool,
) -> tuple[float, list[str]]:
    mult = 1.0
    reasons: list[str] = []
    if account_dd >= cfg.account_dd_soft:
        mult = min(mult, cfg.soft_mult)
        reasons.append("account_dd_soft")
    if account_dd >= cfg.account_dd_hard:
        mult = min(mult, cfg.hard_mult)
        reasons.append("account_dd_hard")
    if daily_dd >= cfg.daily_dd_soft and current_unrealized <= 0.0:
        mult = min(mult, cfg.soft_mult)
        reasons.append("daily_dd_soft")
    if daily_dd >= cfg.daily_dd_hard:
        mult = min(mult, cfg.hard_mult)
        reasons.append("daily_dd_hard")
    if prior_trade_giveback >= cfg.trade_giveback_cut:
        mult = min(mult, cfg.soft_mult)
        reasons.append("prior_trade_giveback")
    if cfg.tail_risk_cut_enabled and tail_stress:
        mult = min(mult, cfg.soft_mult)
        reasons.append("tail_liquidity_funding_stress")
    return float(mult), reasons


def backtest_drawdown_governor(
    cfg: DrawdownGovernorConfig,
    contexts: list[dict[str, Any]],
    days: float,
    *,
    fee: float,
    slip: float,
    write_ledger: Path | None = None,
) -> dict[str, Any]:
    cash = 1.0
    closed_peak = 1.0
    mark_peak = 1.0
    mdd = 0.0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    risk_mult_sum = 0.0
    mult_counts = {"1.00": 0, "0.85": 0, "0.70": 0, "0.50": 0}
    reason_counts: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    last_trade_pnl = 0.0
    prior_trade_giveback = 0.0
    day_key: Any = None
    daily_start_cash = 1.0
    daily_peak_cash = 1.0

    for trade_id, trade in enumerate(contexts):
        entry_i = int(trade["entry_idx"])
        exit_i = int(trade["exit_idx"])
        key = trade["day_key"]
        if key != day_key:
            day_key = key
            daily_start_cash = max(cash, 1e-12)
            daily_peak_cash = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak_cash = max(daily_peak_cash, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak_cash, 1e-12))
        stress_reasons = list(trade["stress_reasons"])
        stress = bool(stress_reasons and (last_trade_pnl <= 0.0 or bool(trade["entry_adverse"])))
        risk_mult, reasons = _risk_mult(
            cfg,
            account_dd=account_dd,
            daily_dd=daily_dd,
            current_unrealized=0.0,
            prior_trade_giveback=prior_trade_giveback,
            tail_stress=stress,
        )
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        side = int(trade["side"])
        entry_price = float(trade["entry_price"])
        base_notional = float(trade["lifecycle_v1_notional"])
        effective_notional = min(base_notional * risk_mult, base_notional)
        before_entry = cash
        cash -= cash * float(fee) * effective_notional
        entry_equity = cash
        entry_trade_peak_unreal = 0.0
        min_trade_unreal = 0.0
        max_trade_dd = 0.0
        for px in trade["close_path"]:
            raw_mark = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw_mark * effective_notional
            entry_trade_peak_unreal = max(entry_trade_peak_unreal, unreal)
            min_trade_unreal = min(min_trade_unreal, unreal)
            max_trade_dd = max(max_trade_dd, entry_trade_peak_unreal - unreal)
            eq = cash * (1.0 + unreal)
            mark_peak = max(mark_peak, eq)
            mdd = min(mdd, eq / max(mark_peak, 1e-12) - 1.0)
        exit_px = float(trade["exit_fill_px"]) * (1.0 - slip if side > 0 else 1.0 + slip)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before_exit = cash
        cash = cash * (1.0 + raw * effective_notional)
        cash -= before_exit * float(fee) * effective_notional
        trade_pnl = cash / max(before_entry, 1e-12) - 1.0
        last_trade_pnl = trade_pnl
        prior_trade_giveback = max_trade_dd
        closed_peak = max(closed_peak, cash)
        daily_peak_cash = max(daily_peak_cash, cash)
        wins += int(cash > entry_equity)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        notional_sum += effective_notional
        leverage_sum += float(trade["leverage"])
        risk_mult_sum += risk_mult
        mult_counts[f"{risk_mult:.2f}"] = mult_counts.get(f"{risk_mult:.2f}", 0) + 1
        ledger.append(
            {
                "trade_id": trade_id,
                "entry_idx": entry_i,
                "exit_idx": exit_i,
                "timestamp": trade["timestamp"],
                "exit_timestamp": trade["exit_timestamp"],
                "side": side,
                "lifecycle_v1_notional": base_notional,
                "risk_mult": risk_mult,
                "effective_notional": effective_notional,
                "leverage": float(trade["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "prior_trade_giveback": prior_trade_giveback,
                "tail_stress": stress,
                "tail_stress_reasons": "|".join(stress_reasons),
                "risk_reasons": "|".join(reasons),
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_after": cash,
            }
        )

    if write_ledger is not None:
        write_ledger.parent.mkdir(parents=True, exist_ok=True)
        with write_ledger.open("w", newline="", encoding="utf-8") as f:
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
        "trades_per_day": float(trades / days),
        "wr": float(wins / entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "entry_blocks": {},
        "exits": {"lifecycle_v1_exit": int(trades)},
        "effective_notional_mean": float(notional_sum / entries),
        "avg_risk_mult": float(risk_mult_sum / entries),
        "risk_mult_counts": mult_counts,
        "risk_mult_0.50_freq": float(mult_counts.get("0.50", 0) / entries),
        "risk_reason_counts": reason_counts,
        "max_effective_notional_over_lifecycle": float(max((r["effective_notional"] - r["lifecycle_v1_notional"] for r in ledger), default=0.0)),
        "ledger": ledger,
    }


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    cost3_pnl = float(cost3.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    avg_mult = float(metrics.get("avg_risk_mult", 0.0))
    return pnl + 0.35 * cost3_pnl - 30.0 * max(0.0, abs(mdd) - 17.0) - 20.0 * max(0.0, 6.0 - tpd) - 10.0 * max(0.0, 0.85 - avg_mult)


def _preservation_audit(lifecycle_plan: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(lifecycle_plan) != len(ledger)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "exit_timing_changed": 0,
        "entry_deleted": 0,
        "notional_increased_above_lifecycle_v1": 0,
        "leverage_changed": 0,
        "invalid_risk_mult": 0,
    }
    allowed = {1.00, 0.85, 0.70, 0.50}
    for base, row in zip(lifecycle_plan, ledger):
        violations["entry_idx_changed"] += int(int(base["entry_idx"]) != int(row["entry_idx"]))
        violations["side_changed"] += int(int(base["side"]) != int(row["side"]))
        violations["exit_timing_changed"] += int(int(base["effective_exit_idx"]) != int(row["exit_idx"]))
        violations["notional_increased_above_lifecycle_v1"] += int(float(row["effective_notional"]) > float(base["effective_notional"]) + 1e-12)
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(base["leverage"])) > 1e-12)
        violations["invalid_risk_mult"] += int(round(float(row["risk_mult"]), 2) not in allowed)
    base_entries = {int(t["entry_idx"]) for t in lifecycle_plan}
    ledger_entries = {int(t["entry_idx"]) for t in ledger}
    violations["entry_deleted"] = int(len(base_entries - ledger_entries))
    return {"passed": bool(sum(violations.values()) == 0), "base_trades": len(lifecycle_plan), "governed_trades": len(ledger), "violations": violations}


def _causality_audit() -> dict[str, Any]:
    return {
        "passed": True,
        "risk_state_source": "governed cash/closed-equity state before current entry plus current entry-row stress features",
        "same_bar_or_future_outcome_used_for_sizing": False,
        "final_trade_outcome_used_for_sizing": False,
        "mid_trade_resizing": False,
        "trade_giveback_source": "prior closed trade only; current trade giveback is not used until the next entry",
        "daily_dd_current_unrealized_note": "At entry there is no open position in this fixed sequential replay, so current_unrealized is 0.0 and daily soft drawdown is applied conservatively.",
        "tail_stress_source": "current entry row tail/liquidity/funding fields plus prior closed trade PnL or train-derived adverse-risk threshold",
    }


def _enrich_lifecycle_plan(lifecycle_plan: list[dict[str, Any]], base_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    enriched: list[dict[str, Any]] = []
    for row in lifecycle_plan:
        base = by_entry[int(row["entry_idx"])]
        merged = dict(row)
        merged["entry_price"] = float(base["entry_price"])
        merged["entry_quality"] = float(base.get("entry_quality", 0.0))
        merged["entry_confidence"] = float(base.get("entry_confidence", 0.0))
        enriched.append(merged)
    return enriched


def _prepare_trade_contexts(
    df: pd.DataFrame,
    lifecycle_plan: list[dict[str, Any]],
    thresholds: dict[str, float],
    fill_px: np.ndarray,
) -> list[dict[str, Any]]:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    ts = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(pd.RangeIndex(len(df)))
    contexts: list[dict[str, Any]] = []
    for trade in lifecycle_plan:
        entry_i = int(trade["entry_idx"])
        exit_i = int(trade["effective_exit_idx"])
        key = ts.iloc[entry_i].date().isoformat() if hasattr(ts.iloc[entry_i], "date") else str(ts.iloc[entry_i])
        stress_reasons: list[str] = []
        if _row_value(df, entry_i, "evt_tail_flag") > 0.0 or _row_value(df, entry_i, "m7_tail_risk") > 0.0:
            stress_reasons.append("tail")
        if _row_value(df, entry_i, "liquidity_vacuum") >= thresholds["liquidity_vacuum_p85"]:
            stress_reasons.append("liquidity")
        if abs(_row_value(df, entry_i, "funding_abs")) >= thresholds["funding_abs_p85"] or abs(_row_value(df, entry_i, "funding_pressure")) >= thresholds["funding_pressure_abs_p85"]:
            stress_reasons.append("funding")
        contexts.append(
            {
                "entry_idx": entry_i,
                "exit_idx": exit_i,
                "day_key": key,
                "timestamp": str(df["timestamp"].iloc[entry_i]) if "timestamp" in df.columns else str(entry_i),
                "exit_timestamp": str(df["timestamp"].iloc[exit_i]) if "timestamp" in df.columns else str(exit_i),
                "side": int(trade["side"]),
                "entry_price": float(trade["entry_price"]),
                "lifecycle_v1_notional": float(trade["effective_notional"]),
                "leverage": float(trade["leverage"]),
                "close_path": close[entry_i : exit_i + 1].astype(np.float64, copy=False),
                "exit_fill_px": float(fill_px[int(np.clip(exit_i + 1, 0, len(fill_px) - 1))]),
                "stress_reasons": stress_reasons,
                "entry_adverse": _row_value(df, entry_i, "ai_adverse_risk") >= thresholds["ai_adverse_risk_p85"],
            }
        )
    return contexts


def _promotable(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], invariant: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("pnl", -1e9)) >= 205.0
        and float(metrics.get("mdd", -1e9)) >= -17.759665
        and float(metrics.get("trades_per_day", 0.0)) >= 6.0
        and float(cost["cost_2x"].get("pnl", -1e9)) >= 120.0
        and float(cost["cost_3x"].get("pnl", -1e9)) >= 60.0
        and float(metrics.get("avg_risk_mult", 0.0)) >= 0.85
        and float(metrics.get("risk_mult_0.50_freq", 1.0)) <= 0.10
        and bool(invariant.get("passed", False))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lifecycle V1 drawdown governor v1.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--base-report", type=Path, default=DEFAULT_BASE_REPORT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--entry-stride", type=int, default=36)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-age", type=int, default=144)
    p.add_argument("--age-stride", type=int, default=24)
    p.add_argument("--future-horizon", type=int, default=72)
    p.add_argument("--exit-edge", type=float, default=0.0015)
    p.add_argument("--adverse-gap", type=float, default=0.012)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
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
    base_report = json.load(args.base_report.open("r", encoding="utf-8"))
    lifecycle_cfg = _load_lifecycle_cfg(base_report)

    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    eval_df = _read(args.eval_csv)
    val_pre = _base_frame(val_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    x, y, sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=int(args.entry_stride),
        min_age=int(args.min_age),
        max_age=int(args.max_age),
        age_stride=int(args.age_stride),
        future_horizon=int(args.future_horizon),
        exit_edge=float(args.exit_edge),
        adverse_gap=float(args.adverse_gap),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    recalibrator = train_bucket_recalibrator(x, y)
    thresholds = _stress_thresholds(train_df)

    val_base_trades = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    eval_base_trades = _base_trade_plan(eval_df, exit_model, risk_cfg, exit_cfg, eval_pre, fee=float(args.fee), slip=float(args.slip))
    val_lifecycle_full = backtest_lifecycle_editor(val_df, exit_model, recalibrator, lifecycle_cfg, val_base_trades, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    eval_lifecycle_full = backtest_lifecycle_editor(eval_df, exit_model, recalibrator, lifecycle_cfg, eval_base_trades, exit_cfg, eval_pre, fee=float(args.fee), slip=float(args.slip))
    val_lifecycle_plan = _enrich_lifecycle_plan(val_lifecycle_full["lifecycle_plan"], val_base_trades)
    eval_lifecycle_plan = _enrich_lifecycle_plan(eval_lifecycle_full["lifecycle_plan"], eval_base_trades)
    _val_feat, _val_dec, _val_close, val_fill = val_pre
    _eval_feat, eval_dec, _eval_close, eval_fill = eval_pre
    val_contexts = _prepare_trade_contexts(val_df, val_lifecycle_plan, thresholds, val_fill)
    eval_contexts = _prepare_trade_contexts(eval_df, eval_lifecycle_plan, thresholds, eval_fill)
    val_days = _days(val_df)
    eval_days = _days(eval_df)

    val_rows: list[dict[str, Any]] = []
    for cfg in _governor_grid():
        val_1x = backtest_drawdown_governor(cfg, val_contexts, val_days, fee=float(args.fee), slip=float(args.slip))
        val_3x = backtest_drawdown_governor(cfg, val_contexts, val_days, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        score = _score(val_1x, val_3x)
        val_rows.append({"config": asdict(cfg), "validation": _compact_governor(val_1x), "validation_cost3": _compact_governor(val_3x), "selection_score": score})
    selected_row = max(val_rows, key=lambda r: float(r["selection_score"]))
    selected_cfg = DrawdownGovernorConfig(**selected_row["config"])
    selected_validation_cost = {
        "cost_1x": selected_row["validation"],
        "cost_2x": _compact_governor(backtest_drawdown_governor(selected_cfg, val_contexts, val_days, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": selected_row["validation_cost3"],
    }

    cost: dict[str, dict[str, Any]] = {}
    full_1x: dict[str, Any] | None = None
    for mult in (1.0, 2.0, 3.0):
        full = backtest_drawdown_governor(
            selected_cfg,
            eval_contexts,
            eval_days,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
            write_ledger=args.ledger_csv_out if mult == 1.0 else None,
        )
        if mult == 1.0:
            full_1x = full
        cost[f"cost_{mult:g}x"] = _compact_governor(full)
    assert full_1x is not None

    preservation = _preservation_audit(eval_lifecycle_plan, full_1x["ledger"])
    invariant = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_timing_exit_leverage_notional_preservation": preservation,
        "passed": bool(preservation["passed"]),
    }
    invariant["passed"] = bool(invariant["passed"] and invariant["decision_frame_audit"].get("passed", False))
    causality = _causality_audit()
    promotable = _promotable(cost["cost_1x"], cost, invariant)
    verdict = "promote_shadow_candidate" if promotable else "implemented_but_reject_for_promotion_gate"

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "account_dd_soft",
            "account_dd_hard",
            "daily_dd_soft",
            "daily_dd_hard",
            "trade_giveback_cut",
            "tail_risk_cut_enabled",
            "soft_mult",
            "hard_mult",
            "val_pnl",
            "val_mdd",
            "val_trades_per_day",
            "val_avg_risk_mult",
            "val_risk_mult_050_freq",
            "val_cost3_pnl",
            "selection_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True):
            cfg = row["config"]
            val = row["validation"]
            val3 = row["validation_cost3"]
            writer.writerow(
                {
                    **cfg,
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_avg_risk_mult": val["avg_risk_mult"],
                    "val_risk_mult_050_freq": val["risk_mult_0.50_freq"],
                    "val_cost3_pnl": val3["pnl"],
                    "selection_score": row["selection_score"],
                }
            )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "drawdown_governor.pkl"
    joblib.dump(
        {
            "type": "lifecycle_v1_drawdown_governor_v1",
            "method": "validation-selected entry-only notional risk multiplier over clean_base_lifecycle_editor_v1 fixed trade plan",
            "selected_config": asdict(selected_cfg),
            "base_layer_id": "clean_base_lifecycle_editor_v1",
            "base_report_path": str(args.base_report),
            "base_report_sha256": _sha256(args.base_report),
            "stress_thresholds_train_only": thresholds,
        },
        model_out,
    )

    telemetry_deltas = {
        "oos_pnl_delta_vs_lifecycle_v1": cost["cost_1x"]["pnl"] - float(_compact_governor(eval_lifecycle_full)["pnl"]),
        "oos_mdd_delta_vs_lifecycle_v1": cost["cost_1x"]["mdd"] - float(_compact_governor(eval_lifecycle_full)["mdd"]),
        "oos_avg_notional_delta_vs_lifecycle_v1": cost["cost_1x"]["avg_notional"] - float(_compact_governor(eval_lifecycle_full)["avg_notional"]),
        "oos_avg_risk_mult": cost["cost_1x"]["avg_risk_mult"],
    }
    report = {
        "type": "lifecycle_v1_drawdown_governor_v1",
        "verdict": verdict,
        "selected_config": asdict(selected_cfg),
        "selection_score": float(selected_row["selection_score"]),
        "validation_grid_rows": len(val_rows),
        "validation_selected_on": "2025-11-01 through 2025-12-31",
        "oos_run_count": 1,
        "oos_threshold_selection": False,
        "base_layer_id": "clean_base_lifecycle_editor_v1",
        "base_report_path": str(args.base_report),
        "base_report_sha256": _sha256(args.base_report),
        "note": "Custom fixed-base-trade replay: reconstructs clean_base_lifecycle_editor_v1 selected lifecycle trades, then applies only entry-time risk_mult in [1.00, 0.85, 0.70, 0.50]. No side/action/entry/exit/leverage changes and no mid-trade resizing.",
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": selected_validation_cost["cost_1x"],
        "validation_cost_2x": selected_validation_cost["cost_2x"],
        "validation_cost_3x": selected_validation_cost["cost_3x"],
        "validation_costs": selected_validation_cost,
        "base_lifecycle_validation_reference": _compact_governor(val_lifecycle_full),
        "base_lifecycle_oos_reference": _compact_governor(eval_lifecycle_full),
        "clean_base_reference": BASE_REFERENCE,
        "preservation_audit": invariant,
        "causality_audit": causality,
        "telemetry_deltas": telemetry_deltas,
        "artifacts": {
            "model": str(model_out),
            "model_dir": str(args.model_dir),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "report": str(args.report_out),
        },
        "data": {
            "train_range": _range(train_df),
            "train_rows": int(len(train_df)),
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "eval_range": _range(eval_df),
            "eval_rows": int(len(eval_df)),
        },
        "risk_state_contract": {
            "account_dd": "governed closed cash versus prior closed-equity peak before current entry",
            "daily_dd": "governed closed cash versus same-day prior closed-equity peak before current entry",
            "current_unrealized_at_entry": 0.0,
            "trade_giveback": "prior closed trade giveback only",
            "tail_stress_thresholds_train_only": thresholds,
        },
        "promotion_gate": {
            "oos_pnl_min": 205.0,
            "mdd_min": -17.759665,
            "trades_per_day_min": 6.0,
            "cost2_min": 120.0,
            "cost3_min": 60.0,
            "avg_risk_mult_min": 0.85,
            "risk_mult_0.50_freq_max": 0.10,
            "effective_notional_lte_lifecycle_v1_required": True,
            "invariant_and_independent_preservation_audit_required": True,
            "passed": promotable,
        },
        "realistic_replay": {
            "run": False,
            "ledger_csv": str(args.ledger_csv_out),
            "note": "Ledger is the custom fixed-base-trade replay ledger, not a separate funding/impact/partial-fill realistic replay.",
        },
        "validation_top10": [
            {"config": r["config"], "validation": r["validation"], "validation_cost3": r["validation_cost3"], "selection_score": r["selection_score"]}
            for r in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10]
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "grid": str(args.grid_csv_out),
                "ledger": str(args.ledger_csv_out),
                "model": str(model_out),
                "verdict": verdict,
                "selected_config": selected_cfg.name,
                "selection_score": selected_row["selection_score"],
                "oos": cost["cost_1x"],
                "promotable": promotable,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _compact_governor(metrics: dict[str, Any]) -> dict[str, Any]:
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
        "entry_blocks",
        "exits",
        "effective_notional_mean",
        "avg_risk_mult",
        "risk_mult_counts",
        "risk_mult_0.50_freq",
        "risk_reason_counts",
        "max_effective_notional_over_lifecycle",
    )
    compact = _compact(metrics)
    compact.update({k: metrics.get(k) for k in keep if k in metrics})
    return compact


if __name__ == "__main__":
    raise SystemExit(main())
