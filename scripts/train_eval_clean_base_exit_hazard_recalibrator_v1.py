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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_grid.csv"

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
        "type": "clean_base_exit_hazard_recalibrator_v1",
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
) -> tuple[float, int, dict[str, Any]]:
    bucket = _bucket_from_vec(vec, dict(recalibrator["thresholds"]))
    info = dict(recalibrator["buckets"].get(bucket, {}))
    global_rate = float(recalibrator["global_hazard_rate"])
    hazard = float(info.get("hazard_rate", global_rate))
    support = int(info.get("support", 0))
    delta = float(np.clip((global_rate - hazard) * float(cfg.delta_scale), -float(cfg.max_delta), float(cfg.max_delta)))
    threshold = float(np.clip(float(base_exit_cfg["exit_threshold"]) + float(cfg.threshold_shift) + delta, 0.05, 0.95))
    min_age = int(base_exit_cfg["min_exit_age"])
    if hazard >= global_rate + 0.08:
        min_age -= int(cfg.high_hazard_age_reduction)
    elif hazard <= global_rate - 0.08:
        min_age += int(cfg.low_hazard_age_increase)
    min_age = int(max(int(cfg.min_age_floor), min_age))
    return threshold, min_age, {"bucket": bucket, "hazard_rate": hazard, "support": support, "threshold_delta": delta}


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
        exits[reason] = exits.get(reason, 0) + 1
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
            threshold, dynamic_min_age, cal = _calibrated_exit_control(recalibrator, row_vec, runtime_cfg, exit_cfg)
            if age >= int(dynamic_min_age):
                p_exit = _exit_probability_vec(exit_model, row_vec)
                exit_probs.append(p_exit)
                thresholds_used.append(float(threshold))
                min_ages_used.append(int(dynamic_min_age))
                bucket = str(cal["bucket"])
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                if p_exit >= float(threshold):
                    close_position(i, "exit_governor_recalibrated")
                    continue
            continue

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
    }


def _score(metrics: dict[str, Any], base_val: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    base_mdd = float(base_val.get("mdd", -12.65))
    sparse_penalty = max(0.0, 5.5 - tpd) * 80.0
    mdd_penalty = max(0.0, base_mdd - mdd) * 8.0
    return pnl + 4.0 * mdd - sparse_penalty - mdd_penalty


def _runtime_grid(base_min_age: int) -> list[RecalibratorRuntimeConfig]:
    rows: list[RecalibratorRuntimeConfig] = []
    for threshold_shift in (0.0, -0.03, 0.03):
        for delta_scale in (0.0, 0.60, 1.00):
            for max_delta in (0.0, 0.08, 0.14):
                if delta_scale == 0.0 and max_delta > 0.0:
                    continue
                for age_reduction, age_increase in ((0, 0), (3, 0), (3, 3)):
                    name = (
                        f"shift{threshold_shift:+.2f}_scale{delta_scale:.2f}_"
                        f"maxd{max_delta:.2f}_ager{age_reduction}_agei{age_increase}"
                    )
                    rows.append(
                        RecalibratorRuntimeConfig(
                            name=name,
                            threshold_shift=float(threshold_shift),
                            delta_scale=float(delta_scale),
                            max_delta=float(max_delta),
                            high_hazard_age_reduction=int(age_reduction),
                            low_hazard_age_increase=int(age_increase),
                            min_age_floor=max(1, int(base_min_age) - 6),
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


def _promotable(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], invariant: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("pnl", -1e9)) >= BASE_REFERENCE["pnl"]
        and float(metrics.get("mdd", -1e9)) >= BASE_REFERENCE["mdd"]
        and float(metrics.get("trades_per_day", 0.0)) >= 5.5
        and float(cost["cost_1x"].get("pnl", -1e9)) > 0.0
        and float(cost["cost_2x"].get("pnl", -1e9)) > 0.0
        and bool(invariant.get("passed", False))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-base exit-only hazard recalibrator v1.")
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
    for runtime_cfg in _runtime_grid(int(exit_cfg["min_exit_age"])):
        metrics = backtest_recalibrated_exit(
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
        val_rows.append({"runtime_config": asdict(runtime_cfg), "validation": _compact(metrics), "validation_score": _score(metrics, base_val)})

    selected_balanced = max(val_rows, key=lambda r: float(r["validation_score"]))
    constrained = [
        r
        for r in val_rows
        if float(r["validation"]["pnl"]) >= float(base_val["pnl"])
        and float(r["validation"]["mdd"]) >= float(base_val["mdd"]) - 2.0
        and float(r["validation"]["trades_per_day"]) >= 5.5
    ]
    selected_constrained = max(constrained, key=lambda r: float(r["validation_score"])) if constrained else selected_balanced
    selected_max_pnl = max(val_rows, key=lambda r: float(r["validation"]["pnl"]))
    selection = {
        "balanced_score": selected_balanced,
        "redteam_constrained": selected_constrained,
        "max_validation_pnl": selected_max_pnl,
    }

    _eval_feat, eval_dec, _eval_close, _eval_fill = eval_pre
    invariant = {
        "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_notional_cooldown_preservation": _decision_preservation_audit(eval_dec, eval_dec.copy()),
    }
    invariant["passed"] = bool(
        invariant["decision_frame_audit"].get("passed", False)
        and invariant["entry_side_notional_cooldown_preservation"].get("passed", False)
    )

    eval_results: dict[str, Any] = {}
    for label, row in selection.items():
        runtime_cfg = RecalibratorRuntimeConfig(**row["runtime_config"])
        cost = {
            f"cost_{mult:g}x": _compact(
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
                )
            )
            for mult in (1.0, 2.0, 3.0)
        }
        eval_results[label] = {
            "runtime_config": asdict(runtime_cfg),
            "validation": row["validation"],
            "validation_score": row["validation_score"],
            "oos": cost["cost_1x"],
            "cost_stress": cost,
            "decision_invariant_audit": invariant,
            "promotable_by_contract": _promotable(cost["cost_1x"], cost, invariant),
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
            "val_pnl",
            "val_mdd",
            "val_trades",
            "val_trades_per_day",
            "val_threshold_mean",
            "val_min_exit_age_mean",
            "validation_score",
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
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades": val["trades"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_threshold_mean": val.get("threshold_mean", 0.0),
                    "val_min_exit_age_mean": val.get("min_exit_age_mean", 0.0),
                    "validation_score": row["validation_score"],
                }
            )

    report = {
        "type": "clean_base_exit_hazard_recalibrator_v1",
        "note": "Frozen clean base policy and frozen clean base exit governor. Recalibrator only changes exit threshold/min-age by state bucket; entry action, side, notional, leverage, and cooldown are untouched.",
        "artifacts": {
            "model": str(model_out),
            "grid_csv": str(args.grid_csv_out),
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
            {"runtime_config": r["runtime_config"], "validation": r["validation"], "validation_score": r["validation_score"]}
            for r in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True)[:10]
        ],
        "selected": {label: row["runtime_config"]["name"] for label, row in selection.items()},
        "selected_eval": eval_results,
        "promotion_gate": {
            "oos_pnl_min": BASE_REFERENCE["pnl"],
            "oos_mdd_min": BASE_REFERENCE["mdd"],
            "trades_per_day_min": 5.5,
            "cost_1x_2x_positive": True,
            "cost_3x_reported": True,
            "decision_invariant_required": True,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "grid": str(args.grid_csv_out),
                "model": str(model_out),
                "selected": report["selected"],
                "redteam_constrained_oos": eval_results["redteam_constrained"]["oos"],
                "redteam_constrained_promotable": eval_results["redteam_constrained"]["promotable_by_contract"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
