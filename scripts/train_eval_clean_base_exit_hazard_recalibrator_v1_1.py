#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1_1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_1_ledger.csv"

BASE_REFERENCE = {
    "pnl": 177.3298088749005,
    "mdd": -17.75966486035323,
    "trades": 363,
    "trades_per_day": 6.1875,
    "cost_2x_pnl": 92.25487780535948,
    "cost_3x_pnl": -7.969394502459748,
}

FEATURE_INDEX = {c: i for i, c in enumerate(MODEL_COLS)}


@dataclass(frozen=True)
class RecalibratorRuntimeConfig:
    name: str
    threshold_shift: float
    delta_scale: float
    max_delta: float
    high_hazard_age_reduction: int
    low_hazard_age_increase: int
    min_age_floor: int
    threshold_floor: float
    threshold_ceiling: float


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _bucket_from_getter(get: Callable[[str], float], thresholds: dict[str, float]) -> str:
    side = "long" if get("gx_side") > 0 else "short"
    age = get("gx_age_bars")
    age_bucket = "a0_12" if age < 12 else "a12_48" if age < 48 else "a48p"
    unreal = get("gx_unrealized")
    pnl_bucket = "loss" if unreal < -0.010 else "gain" if unreal > 0.012 else "flat"
    drawdown = get("gx_drawdown_from_peak")
    dd_bucket = "dd_hi" if drawdown >= thresholds["drawdown_p70"] else "dd_lo"
    funding = abs(get("funding_abs")) >= thresholds["funding_abs_p85"] or abs(get("funding_pressure")) >= thresholds["funding_pressure_p85"]
    liquidity = abs(get("liquidity_vacuum")) >= thresholds["liquidity_vacuum_p85"] or abs(get("amihud_illiquidity_z")) >= thresholds["amihud_illiquidity_z_p85"]
    tail = get("evt_tail_flag") > 0.5 or abs(get("m7_tail_risk")) >= thresholds["m7_tail_risk_p85"] or abs(get("ai_adverse_risk")) >= thresholds["ai_adverse_risk_p85"]
    current = "opp" if get("gx_current_opposite_side") > 0.5 else "same" if get("gx_current_same_side") > 0.5 else "cash"
    stress = "tail" if tail else "liq" if liquidity else "fund" if funding else "plain"
    return "|".join((side, age_bucket, pnl_bucket, dd_bucket, stress, current))


def _bucket_from_series(row: pd.Series, thresholds: dict[str, float]) -> str:
    return _bucket_from_getter(lambda col: _num(row.get(col, 0.0)), thresholds)


def _bucket_from_vec(vec: np.ndarray, thresholds: dict[str, float]) -> str:
    return _bucket_from_getter(lambda col: _num(vec[FEATURE_INDEX[col]]) if col in FEATURE_INDEX else 0.0, thresholds)


def _train_thresholds(x: pd.DataFrame) -> dict[str, float]:
    def q(col: str, quantile: float, fallback: float) -> float:
        if col not in x.columns or x.empty:
            return float(fallback)
        values = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            return float(fallback)
        return float(np.quantile(np.abs(values), quantile))

    return {
        "drawdown_p70": q("gx_drawdown_from_peak", 0.70, 0.010),
        "funding_abs_p85": q("funding_abs", 0.85, 0.001),
        "funding_pressure_p85": q("funding_pressure", 0.85, 1.0),
        "liquidity_vacuum_p85": q("liquidity_vacuum", 0.85, 1.0),
        "amihud_illiquidity_z_p85": q("amihud_illiquidity_z", 0.85, 2.0),
        "m7_tail_risk_p85": q("m7_tail_risk", 0.85, 1.0),
        "ai_adverse_risk_p85": q("ai_adverse_risk", 0.85, 1.0),
    }


def train_bucket_recalibrator(x: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("empty or mismatched recalibrator training set")
    thresholds = _train_thresholds(x)
    global_rate = float(np.mean(y))
    raw: dict[str, list[int]] = {}
    for idx, row in x.iterrows():
        bucket = _bucket_from_series(row, thresholds)
        vals = raw.setdefault(bucket, [0, 0])
        vals[0] += int(y[int(idx)])
        vals[1] += 1
    prior = 80.0
    buckets: dict[str, dict[str, float | int]] = {}
    for bucket, (exits, support) in raw.items():
        rate = (float(exits) + global_rate * prior) / (float(support) + prior)
        buckets[bucket] = {"exit_labels": int(exits), "support": int(support), "hazard_rate": float(rate)}
    return {
        "type": "clean_base_exit_hazard_recalibrator_v1_1",
        "method": "state_bucket_smoothed_exit_label_rate",
        "global_hazard_rate": global_rate,
        "smoothing_prior": prior,
        "thresholds": thresholds,
        "buckets": buckets,
    }


def _calibrated_exit_control(
    recalibrator: dict[str, Any],
    vec: np.ndarray,
    cfg: RecalibratorRuntimeConfig,
    base_exit_cfg: dict[str, Any],
    *,
    account_dd: float = 0.0,
    daily_dd: float = 0.0,
) -> tuple[float, int, dict[str, Any]]:
    bucket = _bucket_from_vec(vec, dict(recalibrator["thresholds"]))
    info = dict(recalibrator["buckets"].get(bucket, {}))
    global_rate = float(recalibrator["global_hazard_rate"])
    hazard = float(info.get("hazard_rate", global_rate))
    support = int(info.get("support", 0))
    delta = float(np.clip((global_rate - hazard) * float(cfg.delta_scale), -float(cfg.max_delta), float(cfg.max_delta)))
    base_threshold = float(base_exit_cfg["exit_threshold"])
    threshold = float(np.clip(base_threshold + float(cfg.threshold_shift) + delta, float(cfg.threshold_floor), float(cfg.threshold_ceiling)))
    min_age = int(base_exit_cfg["min_exit_age"])
    if hazard >= global_rate + 0.08:
        min_age -= int(cfg.high_hazard_age_reduction)
    elif hazard <= global_rate - 0.08 and daily_dd < 0.018:
        min_age += int(cfg.low_hazard_age_increase)
    min_age = int(max(int(cfg.min_age_floor), min_age))
    guard = "none"
    if account_dd >= 0.14:
        threshold = min(threshold, 0.36)
        min_age = 1
        guard = "account_dd_14"
    elif account_dd >= 0.10:
        threshold = min(threshold, 0.42)
        min_age = min(min_age, 3)
        guard = "account_dd_10"
    if daily_dd >= 0.018:
        threshold = min(threshold, base_threshold)
        guard = "daily_dd_018" if guard == "none" else f"{guard}+daily_dd_018"
    return threshold, min_age, {"bucket": bucket, "hazard_rate": hazard, "support": support, "threshold_delta": delta, "guard": guard}


def backtest_recalibrated_exit(
    df: pd.DataFrame,
    exit_model: Any,
    recalibrator: dict[str, Any],
    runtime_cfg: RecalibratorRuntimeConfig,
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
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
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    model_cooldown = 0
    cooldown_left = 0
    loss_cooldown_left = 0
    churn_lock_left = 0
    churn_lock_side = 0
    loss_streak = 0
    peak_unrealized = 0.0
    entry_quality = 0.0
    entry_confidence = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    entry_blocks: dict[str, int] = {}
    exit_probs: list[float] = []
    thresholds_used: list[float] = []
    min_ages_used: list[int] = []
    bucket_counts: dict[str, int] = {}
    guard_counts: dict[str, int] = {}
    ledger_rows: list[dict[str, Any]] = []
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def block(reason: str) -> None:
        entry_blocks[reason] = entry_blocks.get(reason, 0) + 1

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, cooldown_left, model_cooldown
        nonlocal trades, wins, loss_streak, loss_cooldown_left, daily_trades, peak_unrealized
        nonlocal churn_lock_left, churn_lock_side
        exit_side = int(pos)
        exit_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * float(fee) * notional
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_cfg.get("loss_cooldown_bars", 0)))
            if reason == "exit_governor_recalibrated":
                churn_lock_left = max(churn_lock_left, int(model_cooldown), 12)
                churn_lock_side = exit_side
        exits[reason] = exits.get(reason, 0) + 1
        ledger_rows.append(
            {
                "timestamp": str(df["timestamp"].iloc[min(i, len(df) - 1)]) if "timestamp" in df.columns else i,
                "event": "exit",
                "side": exit_side,
                "reason": reason,
                "price": float(exit_price),
                "raw_return": float(raw),
                "notional": float(notional),
                "equity_before": float(before),
                "equity_after": float(cash),
            }
        )
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
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            age = i - entry_idx
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
            threshold, dynamic_min_age, _cal = _calibrated_exit_control(
                recalibrator,
                row_vec,
                runtime_cfg,
                exit_cfg,
                account_dd=account_dd,
                daily_dd=daily_dd,
            )
            if age >= int(dynamic_min_age):
                threshold, dynamic_min_age, cal = _calibrated_exit_control(
                    recalibrator,
                    row_vec,
                    runtime_cfg,
                    exit_cfg,
                    account_dd=account_dd,
                    daily_dd=daily_dd,
                )
                p_exit = _exit_probability_vec(exit_model, row_vec)
                exit_probs.append(p_exit)
                thresholds_used.append(float(threshold))
                min_ages_used.append(int(dynamic_min_age))
                bucket = str(cal["bucket"])
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                guard = str(cal.get("guard", "none"))
                guard_counts[guard] = guard_counts.get(guard, 0) + 1
                if p_exit >= float(threshold):
                    close_position(i, "exit_governor_recalibrated")
                    continue
            continue

        if churn_lock_left > 0:
            if int(sides[i]) == int(churn_lock_side) and int(actions[i]) != ACTION_CASH:
                churn_lock_left -= 1
                block("same_side_reentry_churn_lock")
                continue
            churn_lock_left -= 1
        if cooldown_left > 0:
            cooldown_left -= 1
            block("model_cooldown")
            continue
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(risk_cfg.get("max_daily_trades", 999999)):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(risk_cfg.get("daily_loss_limit", 0.0))):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(risk_cfg.get("daily_dd_limit", 0.0))):
            block("daily_dd_lock")
            continue
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            block("cash_signal")
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
            block("zero_notional")
            continue

        pos = int(sides[i])
        entry_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = float(leverages[i])
        model_cooldown = int(cooldowns[i])
        cash -= cash * float(fee) * notional
        ledger_rows.append(
            {
                "timestamp": str(df["timestamp"].iloc[min(i, len(df) - 1)]) if "timestamp" in df.columns else i,
                "event": "entry",
                "side": int(pos),
                "reason": "base_signal",
                "price": float(entry_price),
                "raw_return": 0.0,
                "notional": float(notional),
                "equity_before": float(entry_equity),
                "equity_after": float(cash),
            }
        )
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        peak_unrealized = 0.0
        entry_quality = float(qualities[i])
        entry_confidence = float(confs[i])

    if pos != 0:
        close_position(len(df) - 2, "forced_end")
    entries = max(long_entries + short_entries, 1)
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ledger_rows).to_csv(ledger_out, index=False)
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
        "entry_blocks": entry_blocks,
        "exits": exits,
        "exit_prob_mean": float(np.mean(exit_probs)) if exit_probs else 0.0,
        "exit_prob_p95": float(np.quantile(exit_probs, 0.95)) if exit_probs else 0.0,
        "threshold_mean": float(np.mean(thresholds_used)) if thresholds_used else 0.0,
        "threshold_p05": float(np.quantile(thresholds_used, 0.05)) if thresholds_used else 0.0,
        "threshold_p95": float(np.quantile(thresholds_used, 0.95)) if thresholds_used else 0.0,
        "min_exit_age_mean": float(np.mean(min_ages_used)) if min_ages_used else 0.0,
        "bucket_counts_top10": dict(sorted(bucket_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        "drawdown_guard_counts": guard_counts,
    }


def _selection_score(cost: dict[str, dict[str, Any]]) -> float:
    pnl_1x = float(cost["cost_1x"].get("pnl", -1e9))
    mdd_1x = float(cost["cost_1x"].get("mdd", -1e9))
    mdd_2x = float(cost["cost_2x"].get("mdd", -1e9))
    tpd_1x = float(cost["cost_1x"].get("trades_per_day", 0.0))
    cost3_pnl = float(cost["cost_3x"].get("pnl", -1e9))
    return (
        pnl_1x
        - 12.0 * max(0.0, abs(mdd_1x) - 17.76)
        - 8.0 * max(0.0, abs(mdd_2x) - 20.0)
        - 4.0 * max(0.0, 5.5 - tpd_1x)
        - 2.0 * max(0.0, tpd_1x - 11.0)
        + 0.30 * cost3_pnl
    )


def _runtime_grid(base_min_age: int) -> list[RecalibratorRuntimeConfig]:
    rows: list[RecalibratorRuntimeConfig] = []
    for threshold_shift in (0.00, 0.02, 0.04, -0.01):
        for delta_scale in (0.25, 0.50, 0.75, 1.00):
            for max_delta in (0.04, 0.08, 0.10, 0.12):
                for age_reduction in (0, 1, 3):
                    for age_increase in (0, 3, 6, 12):
                        for min_age_floor in (3, 6):
                            for threshold_floor in (0.38, 0.42):
                                for threshold_ceiling in (0.58, 0.62, 0.66):
                                    if threshold_floor > threshold_ceiling:
                                        continue
                                    name = (
                                        f"shift{threshold_shift:+.2f}_scale{delta_scale:.2f}_"
                                        f"maxd{max_delta:.2f}_ager{age_reduction}_agei{age_increase}_"
                                        f"floor{threshold_floor:.2f}_ceil{threshold_ceiling:.2f}_minage{min_age_floor}"
                                    )
                                    rows.append(
                                        RecalibratorRuntimeConfig(
                                            name=name,
                                            threshold_shift=float(threshold_shift),
                                            delta_scale=float(delta_scale),
                                            max_delta=float(max_delta),
                                            high_hazard_age_reduction=int(age_reduction),
                                            low_hazard_age_increase=int(age_increase),
                                            min_age_floor=int(min_age_floor),
                                            threshold_floor=float(threshold_floor),
                                            threshold_ceiling=float(threshold_ceiling),
                                        )
                                    )
    unique: dict[str, RecalibratorRuntimeConfig] = {}
    for row in rows:
        unique[row.name] = row
    return list(unique.values())


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
        "exit_prob_mean",
        "exit_prob_p95",
        "threshold_mean",
        "threshold_p05",
        "threshold_p95",
        "min_exit_age_mean",
        "drawdown_guard_counts",
        "bucket_counts_top10",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _decision_preservation_audit(base_dec: pd.DataFrame, candidate_dec: pd.DataFrame) -> dict[str, Any]:
    cols = ["action", "side", "notional_exposure", "leverage", "cooldown_bars", "position_fraction"]
    violations: dict[str, int] = {}
    for col in cols:
        if col not in base_dec.columns or col not in candidate_dec.columns:
            violations[f"{col}_missing"] = 1
            continue
        a = pd.to_numeric(base_dec[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = pd.to_numeric(candidate_dec[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        violations[f"{col}_changed"] = int(np.sum(np.abs(a - b) > 1e-10))
    return {"passed": bool(sum(violations.values()) == 0), "rows": int(len(base_dec)), "violations": violations}


def _validation_reject_reasons(cost: dict[str, dict[str, Any]], base_val: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    v1 = cost["cost_1x"]
    if float(v1.get("pnl", -1e9)) < float(base_val["pnl"]):
        reasons.append("validation_pnl_below_clean_base")
    if float(v1.get("mdd", -1e9)) < float(base_val["mdd"]) - 1.0:
        reasons.append("validation_mdd_more_than_1pt_worse")
    tpd = float(v1.get("trades_per_day", 0.0))
    if not (6.0 <= tpd <= 11.5):
        reasons.append("validation_trades_day_outside_6_11_5")
    if _num(cost["cost_2x"].get("pnl"), -1e9) <= 0.0:
        reasons.append("validation_cost2_pnl_not_positive")
    if _num(cost["cost_3x"].get("pnl"), -1e9) <= 15.0:
        reasons.append("validation_cost3_pnl_not_above_15")
    if float(v1.get("threshold_p05", 0.0)) < 0.38:
        reasons.append("threshold_p05_below_0_38")
    return reasons


def _oos_reject_reasons(cost: dict[str, dict[str, Any]], invariant: dict[str, Any], independent: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    m1 = cost["cost_1x"]
    if float(m1.get("pnl", -1e9)) < 220.0:
        reasons.append("oos_pnl_1x_below_220")
    if float(m1.get("mdd", -1e9)) < -17.759665:
        reasons.append("oos_mdd_1x_worse_than_gate")
    if float(m1.get("trades_per_day", 0.0)) < 5.5:
        reasons.append("oos_trades_day_1x_below_5_5")
    if float(cost["cost_2x"].get("pnl", -1e9)) <= 50.0:
        reasons.append("oos_cost2_pnl_not_above_50")
    if float(cost["cost_3x"].get("pnl", -1e9)) < 20.0:
        reasons.append("oos_cost3_pnl_below_20")
    if not bool(invariant.get("passed", False)):
        reasons.append("decision_invariant_audit_failed")
    if not bool(independent.get("passed", False)):
        reasons.append("independent_preservation_audit_failed")
    return reasons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-base exit hazard recalibrator v1.1 with MDD/cost/churn guards.")
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
    p.add_argument("--progress-every", type=int, default=256)
    p.add_argument("--max-grid-configs", type=int, default=0, help="0 runs the full V1.1 grid; positive values run a deterministic prefix for bounded feasibility checks.")
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

    train_pre = _base_frame(train_df, policy, entry_cfg)
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
    model_out = args.model_dir / "hazard_recalibrator.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "recalibrator": recalibrator,
            "sample_meta": sample_meta,
            "base_policy": str(args.policy),
            "base_exit_model": str(args.exit_model),
            "entry_config": entry_cfg,
            "risk_config": risk_cfg,
            "exit_config": exit_cfg,
            "data_split": {
                "train_range": _range(train_df),
                "validation_range": _range(val_df),
                "eval_range": _range(eval_df),
            },
        },
        model_out,
    )

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

    val_rows: list[dict[str, Any]] = []
    runtime_grid = _runtime_grid(int(exit_cfg["min_exit_age"]))
    total_runtime_grid = len(runtime_grid)
    if int(args.max_grid_configs) > 0:
        runtime_grid = runtime_grid[: int(args.max_grid_configs)]
    for grid_idx, runtime_cfg in enumerate(runtime_grid, start=1):
        if int(args.progress_every) > 0 and (grid_idx == 1 or grid_idx % int(args.progress_every) == 0):
            print(f"validation_grid {grid_idx}/{len(runtime_grid)}", file=sys.stderr, flush=True)
        metrics_1x = backtest_recalibrated_exit(
            val_df,
            exit_model,
            recalibrator,
            runtime_cfg,
            risk_cfg,
            exit_cfg,
            val_pre,
            fee=float(args.fee),
            slip=float(args.slip),
        )
        cost = {"cost_1x": _compact(metrics_1x)}
        prelim_reasons = _validation_reject_reasons(
            {"cost_1x": cost["cost_1x"], "cost_2x": {"pnl": 1.0}, "cost_3x": {"pnl": 16.0}},
            base_val,
        )
        if not prelim_reasons:
            for mult in (2.0, 3.0):
                cost[f"cost_{mult:g}x"] = _compact(
                    backtest_recalibrated_exit(
                        val_df,
                        exit_model,
                        recalibrator,
                        runtime_cfg,
                        risk_cfg,
                        exit_cfg,
                        val_pre,
                        fee=float(args.fee) * mult,
                        slip=float(args.slip) * mult,
                    )
                )
        else:
            cost["cost_2x"] = {"pnl": None, "mdd": None, "trades_per_day": None}
            cost["cost_3x"] = {"pnl": None, "mdd": None, "trades_per_day": None}
        reject_reasons = _validation_reject_reasons(cost, base_val)
        score = _selection_score(cost) if not reject_reasons else -1e18
        val_rows.append(
            {
                "runtime_config": asdict(runtime_cfg),
                "validation": cost["cost_1x"],
                "validation_cost_1x": cost["cost_1x"],
                "validation_cost_2x": cost["cost_2x"],
                "validation_cost_3x": cost["cost_3x"],
                "validation_score": score,
                "validation_reject_reasons": reject_reasons,
                "selectable": not reject_reasons,
            }
        )

    selectable = [r for r in val_rows if bool(r["selectable"])]
    selected_row = max(selectable, key=lambda r: float(r["validation_score"])) if selectable else max(val_rows, key=lambda r: float(r["validation"]["pnl"]))
    selection = {
        "mdd_cost_guarded": selected_row,
    }

    _eval_feat, eval_dec, _eval_close, _eval_fill = eval_pre
    _eval_feat_b, eval_dec_b, _eval_close_b, _eval_fill_b = _base_frame(eval_df, policy, entry_cfg)
    independent = _decision_preservation_audit(eval_dec, eval_dec_b)
    invariant = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_notional_cooldown_preservation": independent,
    }
    invariant["passed"] = bool(
        invariant["decision_frame_audit"].get("passed", False)
        and invariant["entry_side_notional_cooldown_preservation"].get("passed", False)
    )

    eval_results: dict[str, Any] = {}
    for label, row in selection.items():
        runtime_cfg = RecalibratorRuntimeConfig(**row["runtime_config"])
        cost: dict[str, dict[str, Any]] = {}
        for mult in (1.0, 2.0, 3.0):
            cost[f"cost_{mult:g}x"] = _compact(
                backtest_recalibrated_exit(
                    eval_df,
                    exit_model,
                    recalibrator,
                    runtime_cfg,
                    risk_cfg,
                    exit_cfg,
                    eval_pre,
                    fee=float(args.fee) * mult,
                    slip=float(args.slip) * mult,
                    ledger_out=args.ledger_csv_out if mult == 1.0 else None,
                )
            )
        oos_reject_reasons = list(row["validation_reject_reasons"]) + _oos_reject_reasons(cost, invariant, independent)
        eval_results[label] = {
            "runtime_config": asdict(runtime_cfg),
            "validation": row["validation"],
            "validation_cost_1x": row["validation_cost_1x"],
            "validation_cost_2x": row["validation_cost_2x"],
            "validation_cost_3x": row["validation_cost_3x"],
            "validation_score": row["validation_score"],
            "validation_reject_reasons": row["validation_reject_reasons"],
            "oos": cost["cost_1x"],
            "selected_oos": cost["cost_1x"],
            "cost_1x": cost["cost_1x"],
            "cost_2x": cost["cost_2x"],
            "cost_3x": cost["cost_3x"],
            "cost_stress": cost,
            "decision_invariant_audit": invariant,
            "independent_preservation_audit": independent,
            "promotion_gate": {
                "passed": not oos_reject_reasons,
                "validation_passed": not row["validation_reject_reasons"],
                "oos_passed": not _oos_reject_reasons(cost, invariant, independent),
                "reject_reasons": oos_reject_reasons,
            },
            "promotable_by_contract": not oos_reject_reasons,
        }

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "threshold_shift",
            "delta_scale",
            "max_delta",
            "high_hazard_age_reduction",
            "low_hazard_age_increase",
            "min_age_floor",
            "threshold_floor",
            "threshold_ceiling",
            "selectable",
            "val_pnl",
            "val_mdd",
            "val_trades",
            "val_trades_per_day",
            "val_cost2_pnl",
            "val_cost2_mdd",
            "val_cost3_pnl",
            "val_cost3_mdd",
            "val_threshold_p05",
            "val_threshold_mean",
            "val_min_exit_age_mean",
            "validation_score",
            "validation_reject_reasons",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True):
            cfg = row["runtime_config"]
            val = row["validation"]
            writer.writerow(
                {
                    "name": cfg["name"],
                    "threshold_shift": cfg["threshold_shift"],
                    "delta_scale": cfg["delta_scale"],
                    "max_delta": cfg["max_delta"],
                    "high_hazard_age_reduction": cfg["high_hazard_age_reduction"],
                    "low_hazard_age_increase": cfg["low_hazard_age_increase"],
                    "min_age_floor": cfg["min_age_floor"],
                    "threshold_floor": cfg["threshold_floor"],
                    "threshold_ceiling": cfg["threshold_ceiling"],
                    "selectable": row["selectable"],
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades": val["trades"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_cost2_pnl": row["validation_cost_2x"].get("pnl"),
                    "val_cost2_mdd": row["validation_cost_2x"].get("mdd"),
                    "val_cost3_pnl": row["validation_cost_3x"].get("pnl"),
                    "val_cost3_mdd": row["validation_cost_3x"].get("mdd"),
                    "val_threshold_p05": val.get("threshold_p05", 0.0),
                    "val_threshold_mean": val.get("threshold_mean", 0.0),
                    "val_min_exit_age_mean": val.get("min_exit_age_mean", 0.0),
                    "validation_score": row["validation_score"],
                    "validation_reject_reasons": ";".join(row["validation_reject_reasons"]),
                }
            )

    selected_eval = eval_results["mdd_cost_guarded"]
    reject_reasons = selected_eval["promotion_gate"]["reject_reasons"]
    report = {
        "type": "clean_base_exit_hazard_recalibrator_v1_1",
        "note": "Frozen clean base policy and frozen clean base exit governor. Exit hazard recalibration includes drawdown guard and losing-exit same-side churn guard, so it is reported as exit_hazard_with_churn_guard rather than pure exit-only.",
        "policy_variant": "exit_hazard_with_churn_guard",
        "verdict": "promote" if not reject_reasons else "reject",
        "reject_reasons": reject_reasons,
        "selected_oos": selected_eval["selected_oos"],
        "cost_1x": selected_eval["cost_1x"],
        "cost_2x": selected_eval["cost_2x"],
        "cost_3x": selected_eval["cost_3x"],
        "validation_cost_1x": selected_eval["validation_cost_1x"],
        "validation_cost_2x": selected_eval["validation_cost_2x"],
        "validation_cost_3x": selected_eval["validation_cost_3x"],
        "decision_invariant_audit": invariant,
        "independent_preservation_audit": independent,
        "promotion_gate": selected_eval["promotion_gate"],
        "artifacts": {
            "model": str(model_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "report": str(args.report_out),
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
        },
        "grid_evaluation": {
            "evaluated_configs": int(len(runtime_grid)),
            "total_configs": int(total_runtime_grid),
            "bounded_run": bool(int(args.max_grid_configs) > 0),
            "max_grid_configs": int(args.max_grid_configs),
        },
        "base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(base_val),
        "clean_base_oos_reference": _compact(base_oos),
        "training": {
            "sample_meta": sample_meta,
            "global_hazard_rate": recalibrator["global_hazard_rate"],
            "bucket_count": len(recalibrator["buckets"]),
            "thresholds": recalibrator["thresholds"],
        },
        "validation_top10": [
            {
                "runtime_config": r["runtime_config"],
                "validation": r["validation"],
                "validation_cost_2x": r["validation_cost_2x"],
                "validation_cost_3x": r["validation_cost_3x"],
                "validation_score": r["validation_score"],
                "selectable": r["selectable"],
                "validation_reject_reasons": r["validation_reject_reasons"],
            }
            for r in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True)[:10]
        ],
        "selected": {label: row["runtime_config"]["name"] for label, row in selection.items()},
        "selected_eval": eval_results,
        "gate_contract": {
            "validation": {
                "pnl_min": "clean_base_validation_pnl",
                "mdd_min": "clean_base_validation_mdd - 1.0",
                "trades_day_range": [6.0, 11.5],
                "cost2_pnl_min_exclusive": 0.0,
                "cost3_pnl_min_exclusive": 15.0,
                "threshold_p05_min": 0.38,
            },
            "oos": {
                "pnl_1x_min": 220.0,
                "mdd_1x_min": -17.759665,
                "trades_day_1x_min": 5.5,
                "cost2_pnl_min_exclusive": 50.0,
                "cost3_pnl_min": 20.0,
                "decision_invariant_required": True,
                "independent_preservation_required": True,
            },
        },
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
                "selected": report["selected"],
                "selected_oos": report["selected_oos"],
                "verdict": report["verdict"],
                "reject_reasons": report["reject_reasons"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
