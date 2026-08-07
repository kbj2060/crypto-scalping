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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.train_eval_clean_base_exit_hazard_recalibrator_v1 import (  # noqa: E402
    _bucket_from_vec,
    train_bucket_recalibrator,
)
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    _base_frame,
    _date_codes,
    _days,
    _exit_probability_vec,
    _feature_vec_fast,
    _fill_price,
    backtest_no_limit_exit,
    collect_exit_samples,
)


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_hybrid_lifecycle_exit_v2"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_hybrid_lifecycle_exit_v2_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_hybrid_lifecycle_exit_v2_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_hybrid_lifecycle_exit_v2_ledger.csv"

BASE_REFERENCE = {
    "pnl": 177.3298088749005,
    "mdd": -17.75966486035323,
    "trades": 363,
    "trades_per_day": 6.1875,
    "cost_1x_pnl": 177.3298088749005,
    "cost_2x_pnl": 92.25487780535948,
    "cost_3x_pnl": -7.969394502459748,
}


@dataclass(frozen=True)
class LifecycleRuntimeConfig:
    name: str
    early_exit_prob: float
    early_hazard_margin: float
    reduce25_margin: float
    reduce50_margin: float
    hold_lock_margin: float
    min_early_age: int
    max_notional: float


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    return [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])]


def _split_train_validation(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    split = pd.Timestamp(split_date)
    return df.loc[ts < split].reset_index(drop=True), df.loc[ts >= split].reset_index(drop=True)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
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
        "edit_counts",
        "action_distribution",
        "exit_prob_mean",
        "exit_prob_p95",
        "hazard_rate_mean",
        "threshold_delta_mean",
        "effective_notional_mean",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _hazard_info(recalibrator: dict[str, Any], vec: np.ndarray) -> tuple[str, float, int]:
    bucket = _bucket_from_vec(vec, dict(recalibrator["thresholds"]))
    info = dict(recalibrator["buckets"].get(bucket, {}))
    return bucket, float(info.get("hazard_rate", recalibrator["global_hazard_rate"])), int(info.get("support", 0))


def _runtime_grid(max_notional: float) -> list[LifecycleRuntimeConfig]:
    rows: dict[str, LifecycleRuntimeConfig] = {}
    for early_exit_prob in (0.50, 0.56, 0.62):
        for early_hazard_margin in (0.04, 0.08):
            for reduce25_margin, reduce50_margin in ((999.0, 999.0), (0.07, 0.14), (0.10, 0.18)):
                for hold_lock_margin in (999.0, 0.08):
                    for min_early_age in (6, 12):
                        name = (
                            f"eep{early_exit_prob:.2f}_eh{early_hazard_margin:.2f}_"
                            f"r25{reduce25_margin:.2f}_r50{reduce50_margin:.2f}_"
                            f"hold{hold_lock_margin:.2f}_age{min_early_age}"
                        )
                        rows[name] = LifecycleRuntimeConfig(
                            name=name,
                            early_exit_prob=float(early_exit_prob),
                            early_hazard_margin=float(early_hazard_margin),
                            reduce25_margin=float(reduce25_margin),
                            reduce50_margin=float(reduce50_margin),
                            hold_lock_margin=float(hold_lock_margin),
                            min_early_age=int(min_early_age),
                            max_notional=float(max_notional),
                        )
    return list(rows.values())


def _base_trade_plan(
    df: pd.DataFrame,
    exit_model: Any,
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    base_feat, decisions, close, fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    day_codes = _date_codes(df)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    cooldowns = pd.to_numeric(decisions["cooldown_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    cash = 1.0
    peak = 1.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    model_cooldown = 0
    cooldown_left = 0
    loss_cooldown_left = 0
    loss_streak = 0
    peak_unrealized = 0.0
    entry_quality = 0.0
    entry_confidence = 0.0
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0
    trades: list[dict[str, Any]] = []

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, cooldown_left, model_cooldown
        nonlocal loss_streak, loss_cooldown_left, daily_trades, peak_unrealized
        exit_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * float(fee) * notional
        daily_trades += 1
        is_win = cash > entry_equity
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_cfg.get("loss_cooldown_bars", 0)))
        trades[-1].update({"exit_idx": int(i), "exit_reason": reason, "base_exit_price": float(exit_price)})
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        cooldown_left = int(model_cooldown)
        model_cooldown = 0
        peak_unrealized = 0.0

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            age = i - entry_idx
            if age >= int(exit_cfg["min_exit_age"]):
                row_vec = _feature_vec_fast(
                    base_values,
                    sides,
                    qualities,
                    confs,
                    i=i,
                    side=pos,
                    age=age,
                    unrealized=unreal,
                    peak_unrealized=peak_unrealized,
                    notional=notional,
                    leverage=leverage,
                    entry_quality=entry_quality,
                    entry_confidence=entry_confidence,
                )
                if _exit_probability_vec(exit_model, row_vec) >= float(exit_cfg["exit_threshold"]):
                    close_position(i, "exit_governor")
                    continue
            continue

        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            continue
        if daily_trades >= int(risk_cfg.get("max_daily_trades", 999999)):
            continue
        if daily_realized <= -abs(float(risk_cfg.get("daily_loss_limit", 0.0))):
            continue
        if daily_dd >= abs(float(risk_cfg.get("daily_dd_limit", 0.0))):
            continue
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            continue

        n = float(notionals[i])
        if account_dd >= float(risk_cfg.get("global_dd_cut", 999.0)):
            n *= float(risk_cfg.get("global_dd_mult", 1.0))
        if loss_streak >= int(risk_cfg.get("loss_streak_soft", 999999)):
            steps = loss_streak - int(risk_cfg.get("loss_streak_soft", 999999)) + 1
            n *= float(risk_cfg.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(risk_cfg.get("daily_profit_boost_start", 999.0)):
            n *= float(risk_cfg.get("daily_profit_boost_mult", 1.0))
        n = float(np.clip(n, 0.0, float(risk_cfg.get("max_notional", 3.6))))
        if n <= 1e-8:
            continue

        pos = int(sides[i])
        entry_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = float(leverages[i])
        model_cooldown = int(cooldowns[i])
        cash -= cash * float(fee) * notional
        peak_unrealized = 0.0
        entry_quality = float(qualities[i])
        entry_confidence = float(confs[i])
        trades.append(
            {
                "entry_idx": int(i),
                "side": int(pos),
                "base_notional": float(n),
                "leverage": float(leverage),
                "cooldown_bars": int(model_cooldown),
                "entry_quality": float(entry_quality),
                "entry_confidence": float(entry_confidence),
                "entry_price": float(entry_price),
                "exit_idx": None,
                "exit_reason": None,
            }
        )

    if pos != 0:
        close_position(len(df) - 2, "forced_end")
    return [t for t in trades if t.get("exit_idx") is not None]


def _entry_edit(
    recalibrator: dict[str, Any],
    cfg: LifecycleRuntimeConfig,
    base_values: np.ndarray,
    side_values: np.ndarray,
    quality_values: np.ndarray,
    confidence_values: np.ndarray,
    trade: dict[str, Any],
) -> tuple[float, str, bool, dict[str, Any]]:
    i = int(trade["entry_idx"])
    vec = _feature_vec_fast(
        base_values,
        side_values,
        quality_values,
        confidence_values,
        i=i,
        side=int(trade["side"]),
        age=0,
        unrealized=0.0,
        peak_unrealized=0.0,
        notional=float(trade["base_notional"]),
        leverage=float(trade["leverage"]),
        entry_quality=float(trade["entry_quality"]),
        entry_confidence=float(trade["entry_confidence"]),
    )
    bucket, hazard, support = _hazard_info(recalibrator, vec)
    global_rate = float(recalibrator["global_hazard_rate"])
    mult = 1.0
    action = "NOOP"
    if hazard >= global_rate + float(cfg.reduce50_margin):
        mult = 0.50
        action = "REDUCE_50"
    elif hazard >= global_rate + float(cfg.reduce25_margin):
        mult = 0.75
        action = "REDUCE_25"
    hold_lock = bool(hazard <= global_rate - float(cfg.hold_lock_margin))
    if hold_lock and action == "NOOP":
        action = "HOLD_LOCK_12"
    effective = float(np.clip(float(trade["base_notional"]) * mult, 0.0, min(float(cfg.max_notional), float(trade["base_notional"]))))
    return effective, action, hold_lock, {"entry_bucket": bucket, "entry_hazard": hazard, "entry_support": support}


def backtest_lifecycle_editor(
    df: pd.DataFrame,
    exit_model: Any,
    recalibrator: dict[str, Any],
    cfg: LifecycleRuntimeConfig,
    base_trades: list[dict[str, Any]],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    base_feat, decisions, close, fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    sides = decisions["side"].astype(int).to_numpy()
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    edit_counts: dict[str, int] = {}
    exits: dict[str, int] = {}
    exit_probs: list[float] = []
    hazard_rates: list[float] = []
    threshold_deltas: list[float] = []
    lifecycle_plan: list[dict[str, Any]] = []
    active_idx = 0

    for i in range(0, len(df) - 1):
        while active_idx < len(base_trades) and int(base_trades[active_idx]["entry_idx"]) < i:
            active_idx += 1
        if active_idx >= len(base_trades):
            eq = cash
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            continue
        trade = base_trades[active_idx]
        entry_i = int(trade["entry_idx"])
        if i < entry_i:
            eq = cash
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            continue

        side = int(trade["side"])
        entry_price = float(trade["entry_price"])
        base_exit_i = int(trade["exit_idx"])
        effective_notional, entry_action, hold_lock, edit_meta = _entry_edit(recalibrator, cfg, base_values, sides, qualities, confs, trade)
        edit_counts[entry_action] = edit_counts.get(entry_action, 0) + 1
        cash -= cash * float(fee) * effective_notional
        entry_equity = cash
        peak_unrealized = 0.0
        exit_i = base_exit_i
        exit_reason = "base_exit"
        for j in range(entry_i, base_exit_i + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            raw_mark = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw_mark * effective_notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            peak_unrealized = max(peak_unrealized, unreal)
            age = j - entry_i
            min_age = max(int(exit_cfg["min_exit_age"]), int(cfg.min_early_age))
            if hold_lock:
                min_age = max(min_age, 12)
            if age < max(1, min_age):
                continue
            vec = _feature_vec_fast(
                base_values,
                sides,
                qualities,
                confs,
                i=j,
                side=side,
                age=age,
                unrealized=unreal,
                peak_unrealized=peak_unrealized,
                notional=effective_notional,
                leverage=float(trade["leverage"]),
                entry_quality=float(trade["entry_quality"]),
                entry_confidence=float(trade["entry_confidence"]),
            )
            bucket, hazard, support = _hazard_info(recalibrator, vec)
            delta = float(hazard - float(recalibrator["global_hazard_rate"]))
            p_exit = _exit_probability_vec(exit_model, vec)
            exit_probs.append(float(p_exit))
            hazard_rates.append(float(hazard))
            threshold_deltas.append(float(delta))
            if support >= 25 and hazard >= float(recalibrator["global_hazard_rate"]) + float(cfg.early_hazard_margin) and p_exit >= float(cfg.early_exit_prob):
                exit_i = j
                exit_reason = "hybrid_early_exit"
                edit_meta.update({"exit_bucket": bucket, "exit_hazard": float(hazard), "exit_support": int(support), "exit_governor_probability": float(p_exit), "threshold_delta": float(delta)})
                break

        exit_price = _fill_price(fill_px, min(exit_i + 1, len(df) - 1), side, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * effective_notional)
        cash -= before * float(fee) * effective_notional
        wins += int(cash > entry_equity)
        long_entries += int(side > 0)
        short_entries += int(side < 0)
        notional_sum += effective_notional
        leverage_sum += float(trade["leverage"])
        exits[exit_reason] = exits.get(exit_reason, 0) + 1
        lifecycle_plan.append(
            {
                "entry_idx": entry_i,
                "base_exit_idx": base_exit_i,
                "effective_exit_idx": int(exit_i),
                "side": side,
                "base_notional": float(trade["base_notional"]),
                "effective_notional": float(effective_notional),
                "leverage": float(trade["leverage"]),
                "cooldown_bars": int(trade["cooldown_bars"]),
                "edit": entry_action,
                "action": "EARLY_EXIT" if exit_reason == "hybrid_early_exit" else entry_action,
                "exit_reason": exit_reason,
                **edit_meta,
            }
        )
        active_idx += 1

    trades = len(lifecycle_plan)
    entries = max(trades, 1)
    action_counts = {"NOOP": 0, "EARLY_EXIT": 0, "HOLD_LOCK_12": 0, "REDUCE_25": 0, "REDUCE_50": 0}
    for row in lifecycle_plan:
        action = str(row.get("action", "NOOP"))
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "entry_blocks": {},
        "exits": exits,
        "edit_counts": edit_counts,
        "action_distribution": action_counts,
        "exit_prob_mean": float(np.mean(exit_probs)) if exit_probs else 0.0,
        "exit_prob_p95": float(np.quantile(exit_probs, 0.95)) if exit_probs else 0.0,
        "hazard_rate_mean": float(np.mean(hazard_rates)) if hazard_rates else 0.0,
        "threshold_delta_mean": float(np.mean(threshold_deltas)) if threshold_deltas else 0.0,
        "effective_notional_mean": float(notional_sum / entries),
        "lifecycle_plan": lifecycle_plan,
    }


def _score(cost_1x: dict[str, Any], cost_3x: dict[str, Any]) -> float:
    pnl = float(cost_1x.get("pnl", -1e9))
    cost3_pnl = float(cost_3x.get("pnl", -1e9))
    mdd = float(cost_1x.get("mdd", -1e9))
    tpd = float(cost_1x.get("trades_per_day", 0.0))
    return pnl + 0.35 * cost3_pnl - 10.0 * max(0.0, abs(mdd) - 17.76) - 15.0 * max(0.0, 5.8 - tpd)


def _preservation_audit(base_trades: list[dict[str, Any]], lifecycle_plan: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(base_trades) != len(lifecycle_plan)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "cooldown_changed": 0,
        "entry_deleted": 0,
        "side_flip": 0,
        "effective_exit_after_base_exit": 0,
        "negative_effective_notional": 0,
        "effective_notional_above_base": 0,
        "leverage_changed": 0,
    }
    for base, edited in zip(base_trades, lifecycle_plan):
        violations["entry_idx_changed"] += int(int(base["entry_idx"]) != int(edited["entry_idx"]))
        violations["side_changed"] += int(int(base["side"]) != int(edited["side"]))
        violations["side_flip"] += int(int(base["side"]) * int(edited["side"]) < 0)
        violations["cooldown_changed"] += int(int(base["cooldown_bars"]) != int(edited["cooldown_bars"]))
        violations["effective_exit_after_base_exit"] += int(int(edited["effective_exit_idx"]) > int(base["exit_idx"]))
        violations["negative_effective_notional"] += int(float(edited["effective_notional"]) < -1e-12)
        violations["effective_notional_above_base"] += int(float(edited["effective_notional"]) > float(base["base_notional"]) + 1e-12)
        violations["leverage_changed"] += int(abs(float(edited["leverage"]) - float(base["leverage"])) > 1e-12)
    base_entries = {int(t["entry_idx"]) for t in base_trades}
    edited_entries = {int(t["entry_idx"]) for t in lifecycle_plan}
    violations["entry_deleted"] = int(len(base_entries - edited_entries))
    return {"passed": bool(sum(violations.values()) == 0), "base_trades": int(len(base_trades)), "edited_trades": int(len(lifecycle_plan)), "violations": violations}


def _action_reject_reasons(action_distribution: dict[str, int]) -> list[str]:
    total = max(int(sum(action_distribution.values())), 1)
    noop = int(action_distribution.get("NOOP", 0))
    reasons: list[str] = []
    if noop < max(action_distribution.values() or [0]):
        reasons.append("noop_not_largest_action_bucket")
    if float(action_distribution.get("EARLY_EXIT", 0)) / total > 0.35:
        reasons.append("early_exit_above_35pct")
    if float(action_distribution.get("REDUCE_50", 0)) / total > 0.10:
        reasons.append("reduce50_above_10pct")
    return reasons


def _reject_reasons(cost: dict[str, dict[str, Any]], invariant: dict[str, Any], independent: dict[str, Any], action_distribution: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    if float(cost["cost_1x"].get("pnl", -1e9)) < 220.0:
        reasons.append("oos_pnl_below_220")
    if float(cost["cost_1x"].get("mdd", -1e9)) < -17.759665:
        reasons.append("oos_mdd_below_clean_base")
    if float(cost["cost_1x"].get("trades_per_day", 0.0)) < 5.8:
        reasons.append("trades_per_day_below_5_8")
    if float(cost["cost_2x"].get("pnl", -1e9)) < 120.0:
        reasons.append("cost2_pnl_below_120")
    if float(cost["cost_3x"].get("pnl", -1e9)) < 60.0:
        reasons.append("cost3_pnl_below_60")
    if not bool(independent.get("passed", False)):
        reasons.append("independent_preservation_audit_failed")
    if not bool(invariant.get("passed", False)):
        reasons.append("decision_invariant_audit_failed")
    reasons.extend(_action_reject_reasons(action_distribution))
    return reasons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-base hybrid lifecycle + exit-hazard editor v2.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
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

    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    eval_df = _read(args.eval_csv)
    if train_df.empty or val_df.empty or eval_df.empty:
        raise ValueError("empty train/validation/eval split")

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

    base_val = backtest_no_limit_exit(
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
    base_oos = backtest_no_limit_exit(
        eval_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=eval_pre,
    )
    val_base_trades = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    eval_base_trades = _base_trade_plan(eval_df, exit_model, risk_cfg, exit_cfg, eval_pre, fee=float(args.fee), slip=float(args.slip))

    val_rows: list[dict[str, Any]] = []
    for cfg in _runtime_grid(float(risk_cfg.get("max_notional", 3.6))):
        val_cost: dict[str, dict[str, Any]] = {}
        for mult in (1.0, 2.0, 3.0):
            metrics = backtest_lifecycle_editor(
                val_df,
                exit_model,
                recalibrator,
                cfg,
                val_base_trades,
                exit_cfg,
                val_pre,
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
            )
            val_cost[f"cost_{mult:g}x"] = _compact(metrics)
        score = _score(val_cost["cost_1x"], val_cost["cost_3x"])
        action_reasons = _action_reject_reasons(dict(val_cost["cost_1x"].get("action_distribution", {})))
        validation_reasons = list(action_reasons)
        if float(val_cost["cost_1x"].get("trades_per_day", 0.0)) < 5.8:
            validation_reasons.append("validation_trades_per_day_below_5_8")
        selectable = not validation_reasons
        val_rows.append(
            {
                "runtime_config": asdict(cfg),
                "validation_cost_1x": val_cost["cost_1x"],
                "validation_cost_2x": val_cost["cost_2x"],
                "validation_cost_3x": val_cost["cost_3x"],
                "validation_score": score if selectable else -1e18,
                "raw_validation_score": score,
                "validation_reject_reasons": validation_reasons,
                "selectable": selectable,
            }
        )
    selected_row = max(val_rows, key=lambda r: float(r["validation_score"]))
    if float(selected_row["validation_score"]) <= -1e17:
        selected_row = max(val_rows, key=lambda r: float(r["raw_validation_score"]))

    _eval_feat, eval_dec, _eval_close, _eval_fill = eval_pre
    cfg = LifecycleRuntimeConfig(**selected_row["runtime_config"])
    cost: dict[str, dict[str, Any]] = {}
    full_1x: dict[str, Any] | None = None
    for mult in (1.0, 2.0, 3.0):
        full = backtest_lifecycle_editor(
            eval_df,
            exit_model,
            recalibrator,
            cfg,
            eval_base_trades,
            exit_cfg,
            eval_pre,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
        )
        if mult == 1.0:
            full_1x = full
        cost[f"cost_{mult:g}x"] = _compact(full)
    assert full_1x is not None
    independent = _preservation_audit(eval_base_trades, full_1x["lifecycle_plan"])
    effective_cap_audit = {
        "passed": bool(independent["violations"].get("effective_notional_above_base", 1) == 0 and independent["violations"].get("negative_effective_notional", 1) == 0),
        "violations": {
            "effective_notional_above_base": int(independent["violations"].get("effective_notional_above_base", 0)),
            "negative_effective_notional": int(independent["violations"].get("negative_effective_notional", 0)),
        },
    }
    invariant = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_timing_cooldown_preservation": independent,
        "effective_notional_cap_audit": effective_cap_audit,
    }
    invariant["passed"] = bool(
        invariant["decision_frame_audit"].get("passed", False)
        and independent.get("passed", False)
        and effective_cap_audit.get("passed", False)
    )
    action_distribution = dict(cost["cost_1x"].get("action_distribution", {}))
    reject_reasons = _reject_reasons(cost, invariant, independent, action_distribution)
    verdict = "promote" if not reject_reasons else "reject"
    candidate_oos = cost["cost_1x"]
    realistic_replay = {
        "run": False,
        "ledger": None,
        "note": "Not run because canonical promotion gates did not pass." if reject_reasons else "Canonical gates passed, but no compatible realistic replay adapter exists for fixed-trade lifecycle-plan edits in this lightweight V2 smoke.",
    }

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "early_exit_prob",
            "early_hazard_margin",
            "reduce25_margin",
            "reduce50_margin",
            "hold_lock_margin",
            "min_early_age",
            "val_pnl",
            "val_mdd",
            "val_trades",
            "val_trades_per_day",
            "val_cost2_pnl",
            "val_cost3_pnl",
            "val_avg_notional",
            "raw_validation_score",
            "validation_score",
            "selectable",
            "validation_reject_reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True):
            cfg_dict = row["runtime_config"]
            val = row["validation_cost_1x"]
            writer.writerow(
                {
                    "name": cfg_dict["name"],
                    "early_exit_prob": cfg_dict["early_exit_prob"],
                    "early_hazard_margin": cfg_dict["early_hazard_margin"],
                    "reduce25_margin": cfg_dict["reduce25_margin"],
                    "reduce50_margin": cfg_dict["reduce50_margin"],
                    "hold_lock_margin": cfg_dict["hold_lock_margin"],
                    "min_early_age": cfg_dict["min_early_age"],
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades": val["trades"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_cost2_pnl": row["validation_cost_2x"].get("pnl"),
                    "val_cost3_pnl": row["validation_cost_3x"].get("pnl"),
                    "val_avg_notional": val["avg_notional"],
                    "raw_validation_score": row["raw_validation_score"],
                    "validation_score": row["validation_score"],
                    "selectable": row["selectable"],
                    "validation_reject_reasons": ";".join(row["validation_reject_reasons"]),
                }
            )

    model_out = args.model_dir / "hybrid_lifecycle_exit_v2.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "type": "clean_base_hybrid_lifecycle_exit_v2",
            "method": "train-only hazard bucket features plus validation-selected fixed-trade lifecycle actions",
            "recalibrator": recalibrator,
            "selected_runtime_config": asdict(cfg),
            "sample_meta": sample_meta,
            "base_policy": str(args.policy),
            "base_exit_governor": str(args.exit_model),
            "entry_config": entry_cfg,
            "risk_config": risk_cfg,
            "exit_config": exit_cfg,
        },
        model_out,
    )

    report = {
        "type": "clean_base_hybrid_lifecycle_exit_v2",
        "verdict": verdict,
        "clean_base_reference": BASE_REFERENCE,
        "candidate_oos": candidate_oos,
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": selected_row["validation_cost_1x"],
        "validation_cost_2x": selected_row["validation_cost_2x"],
        "validation_cost_3x": selected_row["validation_cost_3x"],
        "independent_preservation_audit": independent,
        "decision_invariant_audit": invariant,
        "action_distribution": action_distribution,
        "reject_reasons": reject_reasons,
        "promotion_gate": {
            "passed": not reject_reasons,
            "requirements": {
                "oos_pnl_min": 220.0,
                "oos_mdd_min": -17.759665,
                "trades_per_day_min": 5.8,
                "cost2_pnl_min": 120.0,
                "cost3_pnl_min": 60.0,
                "noop_largest_action_bucket": True,
                "early_exit_max_fraction": 0.35,
                "reduce50_max_fraction": 0.10,
                "independent_preservation_audit": True,
                "effective_notional_cap_audit": True,
            },
            "reject_reasons": reject_reasons,
        },
        "note": "Frozen clean base policy and frozen clean base exit governor. Exit hazard recalibrator V1 is used only as feature generation: hazard rate, threshold delta versus global hazard, support/bucket, and exit governor probability. V2 edits only fixed base lifecycle exposure and earlier exits.",
        "artifacts": {
            "model": str(model_out),
            "grid_csv": str(args.grid_csv_out),
            "report": str(args.report_out),
            "ledger_csv": str(args.ledger_csv_out) if realistic_replay["run"] else None,
        },
        "frozen_artifacts": {
            "base_policy": str(args.policy),
            "base_policy_sha256": _sha256(args.policy),
            "base_exit_governor": str(args.exit_model),
            "base_exit_governor_sha256": _sha256(args.exit_model),
        },
        "data": {
            "train_range": _range(train_df),
            "train_rows": int(len(train_df)),
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "eval_range": _range(eval_df),
            "eval_rows": int(len(eval_df)),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "clean_base_validation_reference": _compact(base_val),
        "clean_base_oos_reference": _compact(base_oos),
        "base_trade_plan": {"validation_trades": int(len(val_base_trades)), "eval_trades": int(len(eval_base_trades))},
        "training": {
            "sample_meta": sample_meta,
            "global_hazard_rate": recalibrator["global_hazard_rate"],
            "bucket_count": len(recalibrator["buckets"]),
            "thresholds": recalibrator["thresholds"],
        },
        "validation_top10": [
            {
                "runtime_config": r["runtime_config"],
                "validation_cost_1x": r["validation_cost_1x"],
                "validation_cost_2x": r["validation_cost_2x"],
                "validation_cost_3x": r["validation_cost_3x"],
                "validation_score": r["validation_score"],
                "raw_validation_score": r["raw_validation_score"],
                "validation_reject_reasons": r["validation_reject_reasons"],
            }
            for r in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True)[:10]
        ],
        "selected_config": asdict(cfg),
        "selected_validation_score": selected_row["validation_score"],
        "selected_raw_validation_score": selected_row["raw_validation_score"],
        "selected_validation_reject_reasons": selected_row["validation_reject_reasons"],
        "realistic_replay": realistic_replay,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "grid": str(args.grid_csv_out),
                "model": str(model_out),
                "verdict": verdict,
                "selected": report["selected_config"]["name"],
                "candidate_oos": candidate_oos,
                "reject_reasons": reject_reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
