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
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_lifecycle_editor_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_grid.csv"

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
    threshold_shift: float
    delta_scale: float
    max_delta: float
    min_age_delta: int
    shrink_margin: float
    shrink_mult: float
    boost_margin: float
    boost_mult: float
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
        "exit_prob_mean",
        "exit_prob_p95",
        "threshold_mean",
        "effective_notional_mean",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _hazard_info(recalibrator: dict[str, Any], vec: np.ndarray) -> tuple[str, float, int]:
    bucket = _bucket_from_vec(vec, dict(recalibrator["thresholds"]))
    info = dict(recalibrator["buckets"].get(bucket, {}))
    return bucket, float(info.get("hazard_rate", recalibrator["global_hazard_rate"])), int(info.get("support", 0))


def _runtime_grid(max_notional: float) -> list[LifecycleRuntimeConfig]:
    rows: dict[str, LifecycleRuntimeConfig] = {}
    for shift in (0.0, 0.03, -0.03):
        for scale, max_delta in ((0.0, 0.0), (0.60, 0.08), (1.0, 0.12)):
            for min_age_delta in (0, 3):
                for shrink_margin, shrink_mult in ((999.0, 1.0), (0.06, 0.80), (0.10, 0.65)):
                    for boost_margin, boost_mult in ((999.0, 1.0), (0.08, 1.08), (0.12, 1.15)):
                        name = (
                            f"shift{shift:+.2f}_scale{scale:.2f}_maxd{max_delta:.2f}_"
                            f"agep{min_age_delta}_sh{shrink_margin:.2f}x{shrink_mult:.2f}_"
                            f"bo{boost_margin:.2f}x{boost_mult:.2f}"
                        )
                        rows[name] = LifecycleRuntimeConfig(
                            name=name,
                            threshold_shift=float(shift),
                            delta_scale=float(scale),
                            max_delta=float(max_delta),
                            min_age_delta=int(min_age_delta),
                            shrink_margin=float(shrink_margin),
                            shrink_mult=float(shrink_mult),
                            boost_margin=float(boost_margin),
                            boost_mult=float(boost_mult),
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
) -> tuple[float, str, dict[str, Any]]:
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
    kind = "noop"
    if hazard >= global_rate + float(cfg.shrink_margin):
        mult = float(cfg.shrink_mult)
        kind = "shrink"
    elif hazard <= global_rate - float(cfg.boost_margin):
        mult = float(cfg.boost_mult)
        kind = "boost"
    effective = float(np.clip(float(trade["base_notional"]) * mult, 0.0, float(cfg.max_notional)))
    return effective, kind, {"entry_bucket": bucket, "entry_hazard": hazard, "entry_support": support}


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
    thresholds: list[float] = []
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
        effective_notional, edit_kind, edit_meta = _entry_edit(recalibrator, cfg, base_values, sides, qualities, confs, trade)
        edit_counts[edit_kind] = edit_counts.get(edit_kind, 0) + 1
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
            if age < max(1, int(exit_cfg["min_exit_age"]) + int(cfg.min_age_delta)):
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
            _bucket, hazard, _support = _hazard_info(recalibrator, vec)
            delta = float(np.clip((float(recalibrator["global_hazard_rate"]) - hazard) * float(cfg.delta_scale), -float(cfg.max_delta), float(cfg.max_delta)))
            threshold = float(np.clip(float(exit_cfg["exit_threshold"]) + float(cfg.threshold_shift) + delta, 0.05, 0.95))
            p_exit = _exit_probability_vec(exit_model, vec)
            exit_probs.append(float(p_exit))
            thresholds.append(float(threshold))
            if p_exit >= threshold:
                exit_i = j
                exit_reason = "lifecycle_early_exit"
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
                "edit": edit_kind,
                "exit_reason": exit_reason,
                **edit_meta,
            }
        )
        active_idx += 1

    trades = len(lifecycle_plan)
    entries = max(trades, 1)
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
        "exit_prob_mean": float(np.mean(exit_probs)) if exit_probs else 0.0,
        "exit_prob_p95": float(np.quantile(exit_probs, 0.95)) if exit_probs else 0.0,
        "threshold_mean": float(np.mean(thresholds)) if thresholds else 0.0,
        "effective_notional_mean": float(notional_sum / entries),
        "lifecycle_plan": lifecycle_plan,
    }


def _score(metrics: dict[str, Any], base_val: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    base_mdd = float(base_val.get("mdd", -12.65))
    sparse_penalty = max(0.0, 5.5 - tpd) * 100.0
    mdd_penalty = max(0.0, base_mdd - mdd) * 10.0
    return pnl + 4.0 * mdd - sparse_penalty - mdd_penalty


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
    }
    for base, edited in zip(base_trades, lifecycle_plan):
        violations["entry_idx_changed"] += int(int(base["entry_idx"]) != int(edited["entry_idx"]))
        violations["side_changed"] += int(int(base["side"]) != int(edited["side"]))
        violations["side_flip"] += int(int(base["side"]) * int(edited["side"]) < 0)
        violations["cooldown_changed"] += int(int(base["cooldown_bars"]) != int(edited["cooldown_bars"]))
        violations["effective_exit_after_base_exit"] += int(int(edited["effective_exit_idx"]) > int(base["exit_idx"]))
        violations["negative_effective_notional"] += int(float(edited["effective_notional"]) < -1e-12)
    base_entries = {int(t["entry_idx"]) for t in base_trades}
    edited_entries = {int(t["entry_idx"]) for t in lifecycle_plan}
    violations["entry_deleted"] = int(len(base_entries - edited_entries))
    return {"passed": bool(sum(violations.values()) == 0), "base_trades": int(len(base_trades)), "edited_trades": int(len(lifecycle_plan)), "violations": violations}


def _promotable(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], invariant: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("pnl", -1e9)) >= BASE_REFERENCE["pnl"]
        and float(metrics.get("mdd", -1e9)) >= BASE_REFERENCE["mdd"]
        and float(metrics.get("trades_per_day", 0.0)) >= 5.5
        and float(cost["cost_1x"].get("pnl", -1e9)) > 0.0
        and float(cost["cost_2x"].get("pnl", -1e9)) > 0.0
        and "cost_3x" in cost
        and bool(invariant.get("passed", False))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean-base lifecycle editor v1.")
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
        metrics = backtest_lifecycle_editor(
            val_df,
            exit_model,
            recalibrator,
            cfg,
            val_base_trades,
            exit_cfg,
            val_pre,
            fee=float(args.fee),
            slip=float(args.slip),
        )
        val_rows.append({"runtime_config": asdict(cfg), "validation": _compact(metrics), "validation_score": _score(metrics, base_val)})
    selected_balanced = max(val_rows, key=lambda r: float(r["validation_score"]))
    constrained = [
        r
        for r in val_rows
        if float(r["validation"]["pnl"]) >= float(base_val["pnl"])
        and float(r["validation"]["mdd"]) >= float(base_val["mdd"]) - 1.0
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
    eval_results: dict[str, Any] = {}
    top_choice = "redteam_constrained"
    for label, row in selection.items():
        cfg = LifecycleRuntimeConfig(**row["runtime_config"])
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
        invariant = {
            "decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
            "entry_side_timing_cooldown_preservation": _preservation_audit(eval_base_trades, full_1x["lifecycle_plan"]),
        }
        invariant["passed"] = bool(
            invariant["decision_frame_audit"].get("passed", False)
            and invariant["entry_side_timing_cooldown_preservation"].get("passed", False)
        )
        eval_results[label] = {
            "runtime_config": asdict(cfg),
            "validation": row["validation"],
            "validation_score": row["validation_score"],
            "oos": cost["cost_1x"],
            "cost_stress": cost,
            "decision_invariant_audit": invariant,
            "promotable_by_contract": _promotable(cost["cost_1x"], cost, invariant),
        }

    selected_eval = eval_results[top_choice]
    verdict = "promote_shadow_candidate" if selected_eval["promotable_by_contract"] else "implemented_but_reject_for_promotion_gate"
    recommendation = (
        "Keep as research artifact; do not promote until one-shot OOS beats the clean-base PnL/MDD gates with positive 1x/2x cost stress."
        if not selected_eval["promotable_by_contract"]
        else "Eligible for shadow review under the clean-base lifecycle-editor contract."
    )

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "threshold_shift",
            "delta_scale",
            "max_delta",
            "min_age_delta",
            "shrink_margin",
            "shrink_mult",
            "boost_margin",
            "boost_mult",
            "val_pnl",
            "val_mdd",
            "val_trades",
            "val_trades_per_day",
            "val_avg_notional",
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
                    "min_age_delta": cfg["min_age_delta"],
                    "shrink_margin": cfg["shrink_margin"],
                    "shrink_mult": cfg["shrink_mult"],
                    "boost_margin": cfg["boost_margin"],
                    "boost_mult": cfg["boost_mult"],
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades": val["trades"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_avg_notional": val["avg_notional"],
                    "validation_score": row["validation_score"],
                }
            )

    model_out = args.model_dir / "lifecycle_editor.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "type": "clean_base_lifecycle_editor_v1",
            "method": "train-only hazard buckets plus validation-selected deterministic lifecycle edits over fixed base trade plan",
            "recalibrator": recalibrator,
            "selected_runtime_config": selected_eval["runtime_config"],
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
        "type": "clean_base_lifecycle_editor_v1",
        "verdict": verdict,
        "recommendation": recommendation,
        "note": "Frozen clean base policy and frozen clean base exit governor. V1 reconstructs the base-admitted trade plan and only edits lifecycle exposure/early exits inside those fixed trades; entry timing, side, and cooldown are audited independently.",
        "cost_1x": selected_eval["cost_stress"]["cost_1x"],
        "cost_2x": selected_eval["cost_stress"]["cost_2x"],
        "cost_3x": selected_eval["cost_stress"]["cost_3x"],
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
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "base_reference": BASE_REFERENCE,
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
            {"runtime_config": r["runtime_config"], "validation": r["validation"], "validation_score": r["validation_score"]}
            for r in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True)[:10]
        ],
        "selected": {label: row["runtime_config"]["name"] for label, row in selection.items()},
        "selected_for_report": top_choice,
        "selected_eval": eval_results,
        "realistic_replay": {
            "run": False,
            "note": "Not run in v1; this experiment uses the canonical simple fixed-base-trade replay to stay lightweight.",
        },
        "promotion_gate": {
            "oos_pnl_min": BASE_REFERENCE["pnl"],
            "oos_mdd_min": BASE_REFERENCE["mdd"],
            "trades_per_day_min": 5.5,
            "cost_1x_positive": True,
            "cost_2x_positive": True,
            "cost_3x_reported": True,
            "decision_invariant_required": True,
            "no_oos_threshold_selection": True,
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
                "verdict": verdict,
                "selected": report["selected"],
                "selected_oos": selected_eval["oos"],
                "promotable": selected_eval["promotable_by_contract"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
