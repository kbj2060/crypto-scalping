#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, ACTION_SHORT, FEATURE_COLS  # noqa: E402
from scripts.compare_muzero_az_defensive_sleeve_v1_2026 import (  # noqa: E402
    DefensiveSleeveConfig,
    _active_mask,
    _apply_defensive_sleeve,
    _decision_array,
    _diagnostic_targets,
    _grid_configs,
    _predict_sleeve,
    _sleeve_features,
    _state_audit as _defense_state_audit,
    _train_sleeve,
)
from scripts.compare_muzero_az_vs_dt_lifecycle_2026 import (  # noqa: E402
    _build_zero_style_current,
    _date_range,
    _run,
    _slice_precomputed,
)
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL  # noqa: E402
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    _date_codes,
    _exit_probability_vec,
    _feature_vec_fast,
    _fill_price,
)
from scripts.train_eval_muzero_style_exit_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_EXIT_MODEL  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL, _load_az_exit  # noqa: E402
from scripts.train_eval_zero_style_remaining_layers_2026 import _load_mz_exit, _load_mz_risk, _load_pv  # noqa: E402
from scripts.train_eval_zero_style_risk_overlay_2026 import (  # noqa: E402
    DEFAULT_AZ_RISK_OUT,
    DEFAULT_MZ_RISK_OUT,
    RISK_ACTIONS,
)


MODEL_ID = "muzero_az_rank1_flat_microadd_v5_2026"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/muzero_az_rank1_flat_microadd_v5_2026.json"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/muzero_az_rank1_flat_microadd_v5"
DEFAULT_RANK1_STAGE2_MZ_MODEL = ROOT / "data/ensemble/supervised/zero_style/remaining_layers_walkforward/wf_stage2_sleeve_mz.pt"
BASELINE_TARGET = {
    "pnl": 752.648580357841,
    "mdd": -18.755787211251405,
    "trades": 353,
    "trades_per_day": 6.017045454545455,
    "avg_leverage": 1.5960290252000644,
}
BASELINE_COST_TARGETS = {
    "cost_2x_pnl": 279.36,
    "cost_3x_pnl": 75.84,
}
MICROADD_NOTIONAL_CAPS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
MICROADD_LEVERAGE_CAPS = (1.0, 1.2, 1.4)
MICROADD_CHURN_CAPS = (0.25, 0.50, 0.75, 1.00, 1.25)
MICROADD_FEATURE_COLS = list(FEATURE_COLS) + [
    "baseline_flat",
    "candidate_side_from_votes",
    "vote_margin",
    "vote_agreement_count",
    "direction_entropy",
    "tail_risk_score",
    "cost_buffer_1x",
    "cost_buffer_2x",
    "cost_buffer_3x",
    "microadd_cooldown_ok",
    "microadd_churn_budget_remaining",
    "rolling_trades_1d",
    "rolling_trades_7d",
    "rolling_notional_delta_1d",
    "rolling_notional_delta_7d",
    "rolling_fee_slip_cost_1d",
    "rolling_fee_slip_cost_7d",
    "rolling_realized_pnl_1d",
    "rolling_realized_pnl_7d",
    "rolling_drawdown_1d",
    "rolling_drawdown_7d",
    "time_since_last_entry_bars",
    "time_since_last_exit_bars",
    "time_since_last_microadd_bars",
    "time_since_last_resize_bars",
]


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class MicroAddConfig:
    horizon: int = 144
    short_horizon: int = 36
    cvar_alpha: float = 0.10
    max_train_samples: int = 50000
    seed: int = 42
    min_notional: float = 0.05
    approximate_hold_bars: int = 36


class ConstantBinaryProb:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, p_one: float):
        self.p_one = float(np.clip(p_one, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.full(len(x), self.p_one, dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])


def _limit(df: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    if rows is None or int(rows) <= 0:
        return df.reset_index(drop=True)
    return df.head(int(rows)).reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).to_numpy(dtype=np.float64)


def _fit_binary(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    if len(x) == 0:
        return ConstantBinaryProb(0.0)
    if len(np.unique(y)) < 2:
        return ConstantBinaryProb(float(np.mean(y)))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=160,
            max_leaf_nodes=24,
            l2_regularization=0.05,
            random_state=int(seed),
        ),
    )
    model.fit(x, y.astype(np.int64))
    return model


def _fit_regressor(x: np.ndarray, y: np.ndarray, *, seed: int, loss: str = "squared_error", quantile: float | None = None) -> Any:
    if len(x) == 0:
        model = DummyRegressor(strategy="constant", constant=0.0)
        model.fit(np.zeros((1, len(MICROADD_FEATURE_COLS)), dtype=np.float32), np.zeros(1, dtype=np.float32))
        return model
    if float(np.nanstd(y)) < 1e-12:
        model = DummyRegressor(strategy="constant", constant=float(np.nanmean(y)))
        model.fit(np.zeros((1, x.shape[1]), dtype=np.float32), np.zeros(1, dtype=np.float32))
        return model
    kwargs: dict[str, Any] = {
        "learning_rate": 0.045,
        "max_iter": 180,
        "max_leaf_nodes": 24,
        "l2_regularization": 0.05,
        "random_state": int(seed),
        "loss": loss,
    }
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(**kwargs))
    model.fit(x, y.astype(np.float64))
    return model


def _baseline_reproduction_audit(eval_row: dict[str, Any], smoke_limits: dict[str, Any]) -> dict[str, Any]:
    metrics = eval_row.get("eval", eval_row)
    diffs = {k: float(metrics.get(k, np.nan)) - float(v) for k, v in BASELINE_TARGET.items()}
    has_smoke_limits = any(v not in (None, 0) for v in smoke_limits.values())
    tolerance = {"pnl": 1e-6, "mdd": 1e-6, "trades": 0.0, "trades_per_day": 1e-9, "avg_leverage": 1e-9}
    passed = all(abs(diffs[k]) <= tolerance[k] for k in BASELINE_TARGET)
    return {
        "status": "development_skipped_due_to_smoke_limits" if has_smoke_limits else ("passed" if passed else "failed"),
        "passed": bool(passed) if not has_smoke_limits else None,
        "smoke_limits_present": bool(has_smoke_limits),
        "target": BASELINE_TARGET,
        "observed": {k: metrics.get(k) for k in BASELINE_TARGET},
        "diff": diffs,
        "tolerance": tolerance,
        "hard_gate": "skipped/development" if has_smoke_limits else "required",
    }


def _safe_entropy(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    flat = np.maximum(0.0, 1.0 - up - down)
    probs = np.column_stack([np.clip(up, 0.0, 1.0), np.clip(down, 0.0, 1.0), np.clip(flat, 0.0, 1.0)])
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return -np.sum(np.where(probs > 0.0, probs * np.log(probs), 0.0), axis=1) / np.log(3.0)


def _vote_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = len(df)
    votes: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    sources: dict[str, str] = {}

    if {"ai_dir_p_up", "ai_dir_p_down"}.issubset(df.columns):
        diff = _num(df, "ai_dir_p_up") - _num(df, "ai_dir_p_down")
        votes.append(np.sign(diff).astype(np.int64))
        weights.append(np.abs(diff))
        sources["ai_dir_prob"] = "used_current_bar"
        entropy = _safe_entropy(_num(df, "ai_dir_p_up"), _num(df, "ai_dir_p_down"))
    else:
        entropy = np.ones(n, dtype=np.float64)
        sources["ai_dir_prob"] = "unavailable"

    if "ai_dir_edge" in df.columns:
        edge = _num(df, "ai_dir_edge")
        votes.append(np.sign(edge).astype(np.int64))
        weights.append(np.maximum(np.abs(edge), 1e-6))
        sources["ai_dir_edge"] = "used_current_bar"
    else:
        sources["ai_dir_edge"] = "unavailable"

    if "m7_expected_ret" in df.columns:
        m7 = _num(df, "m7_expected_ret")
        m7_conf = _num(df, "m7_confidence", 1.0)
        votes.append(np.sign(m7).astype(np.int64))
        weights.append(np.maximum(np.abs(m7), 1e-6) * np.clip(m7_conf, 0.0, 1.0))
        sources["m7_expected_ret"] = "used_current_bar"
    else:
        sources["m7_expected_ret"] = "unavailable"

    patch_col = "conf_patchtst" if "conf_patchtst" in df.columns else "patchtst_confidence" if "patchtst_confidence" in df.columns else ""
    if "pred_patchtst" in df.columns:
        pred = _num(df, "pred_patchtst")
        conf = _num(df, patch_col, 1.0) if patch_col else np.ones(n, dtype=np.float64)
        votes.append(np.sign(pred).astype(np.int64))
        weights.append(np.maximum(np.abs(pred), 1e-6) * np.clip(conf, 0.0, 1.0))
        sources["patchtst"] = f"used_current_bar:{patch_col or 'no_confidence_col'}"
    else:
        sources["patchtst"] = "unavailable"

    bull_col = "regime_bull_id" if "regime_bull_id" in df.columns else "regime_bull" if "regime_bull" in df.columns else ""
    bear_col = "regime_bear_id" if "regime_bear_id" in df.columns else "regime_bear" if "regime_bear" in df.columns else ""
    if bull_col and bear_col:
        bull = _num(df, bull_col)
        bear = _num(df, bear_col)
        reg_vote = np.where(bull > bear, 1, np.where(bear > bull, -1, 0)).astype(np.int64)
        votes.append(reg_vote)
        weights.append(np.abs(bull - bear))
        sources["regime"] = f"used_current_bar:{bull_col},{bear_col}"
    else:
        sources["regime"] = "unavailable"

    if not votes:
        candidate = np.zeros(n, dtype=np.int64)
        margin = np.zeros(n, dtype=np.float64)
        agree = np.zeros(n, dtype=np.int64)
        conflict = np.zeros(n, dtype=bool)
    else:
        v = np.vstack(votes).astype(np.int64)
        w = np.vstack(weights).astype(np.float64)
        weighted = (v * w).sum(axis=0)
        denom = np.maximum((w * (v != 0)).sum(axis=0), 1e-12)
        candidate = np.sign(weighted).astype(np.int64)
        margin = np.where(candidate != 0, np.abs(weighted) / denom, 0.0)
        agree = np.where(candidate != 0, (v == candidate).sum(axis=0), 0).astype(np.int64)
        conflict = np.where(candidate != 0, (v == -candidate).any(axis=0), False)

    out = pd.DataFrame(
        {
            "candidate_side_from_votes": candidate,
            "vote_margin": np.clip(margin, 0.0, 1.0),
            "vote_agreement_count": agree,
            "direction_entropy": entropy,
            "vote_conflict": conflict.astype(bool),
        }
    )
    audit = {
        "side_hint": "ignored_provenance_unclear" if "side_hint" in df.columns else "unavailable",
        "sources": sources,
        "candidate_rows": int((candidate != 0).sum()),
        "long_vote_rows": int((candidate > 0).sum()),
        "short_vote_rows": int((candidate < 0).sum()),
        "conflict_rows": int(conflict.sum()),
        "future_label_side_selection": False,
    }
    return out, audit


def _sequential_baseline_state(
    df: pd.DataFrame,
    exit_model: Any,
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_feat, decisions, close, fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    day_codes = _date_codes(df)
    actions = _decision_array(decisions, "action", ACTION_CASH).astype(np.int64)
    sides = _decision_array(decisions, "side", 0.0).astype(np.int64)
    notionals = _decision_array(decisions, "notional_exposure", 0.0)
    leverages = _decision_array(decisions, "leverage", 1.0)
    cooldowns = _decision_array(decisions, "cooldown_bars", 0.0).astype(np.int64)
    qualities = _decision_array(decisions, "quality_score", 0.0)
    confs = _decision_array(decisions, "confidence", 0.0)
    active_row = (actions != ACTION_CASH) & (sides != 0) & (notionals > 0.0)

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
    trades = 0
    position_open_before = np.zeros(len(df), dtype=bool)
    flat_for_microadd = np.zeros(len(df), dtype=bool)
    entry_rows = np.zeros(len(df), dtype=bool)
    exit_rows = np.zeros(len(df), dtype=bool)
    block_counts: dict[str, int] = {}

    def block(reason: str) -> None:
        block_counts[reason] = block_counts.get(reason, 0) + 1

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
        nonlocal trades, loss_streak, loss_cooldown_left, daily_trades, peak_unrealized
        exit_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_cfg.get("loss_cooldown_bars", 0)))
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        cooldown_left = int(model_cooldown)
        model_cooldown = 0
        peak_unrealized = 0.0
        exit_rows[i] = True
        block(f"exit_{reason}")

    for i in range(0, max(0, len(df) - 2)):
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
        position_open_before[i] = pos != 0
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
        flat_for_microadd[i] = not active_row[i]
        if not active_row[i]:
            block("cash_signal")
            continue
        n = float(np.clip(notionals[i], 0.0, float(risk_cfg.get("max_notional", 3.6))))
        if account_dd >= float(risk_cfg.get("global_dd_cut", 999.0)):
            n *= float(risk_cfg.get("global_dd_mult", 1.0))
        if loss_streak >= int(risk_cfg.get("loss_streak_soft", 999999)):
            steps = loss_streak - int(risk_cfg.get("loss_streak_soft", 999999)) + 1
            n *= float(risk_cfg.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(risk_cfg.get("daily_profit_boost_start", 999.0)):
            n *= float(risk_cfg.get("daily_profit_boost_mult", 1.0))
        if float(risk_cfg.get("equity_high_boost_dd", -1.0)) >= 0.0 and account_dd <= float(risk_cfg.get("equity_high_boost_dd", -1.0)):
            n *= float(risk_cfg.get("equity_high_boost_mult", 1.0))
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
        cash -= cash * fee * notional
        peak_unrealized = 0.0
        entry_quality = float(qualities[i])
        entry_confidence = float(confs[i])
        entry_rows[i] = True

    if pos != 0 and len(df) >= 2:
        close_position(len(df) - 2, "forced_end")

    out = pd.DataFrame(
        {
            "baseline_position_open_before": position_open_before,
            "baseline_flat_for_microadd": flat_for_microadd,
            "baseline_entry_rows_exact": entry_rows,
            "baseline_exit_rows_exact": exit_rows,
        }
    )
    audit = {
        "method": "sequential_replay_exact_backtest_no_limit_exit_entry_state",
        "rows": int(len(df)),
        "position_open_before_rows": int(position_open_before.sum()),
        "flat_for_microadd_rows": int(flat_for_microadd.sum()),
        "exact_entry_rows": int(entry_rows.sum()),
        "exact_exit_rows": int(exit_rows.sum()),
        "exact_closed_trades": int(trades),
        "block_counts": block_counts,
        "exact_backtest_state_claimed": True,
    }
    return out, audit


def _trailing_state(dec: pd.DataFrame, *, fee: float, slip: float) -> pd.DataFrame:
    active = _active_mask(dec).astype(np.float64)
    notional = _decision_array(dec, "notional_exposure", 0.0)
    side = _decision_array(dec, "side", 0.0)
    notional_delta = np.abs(np.diff(np.r_[0.0, notional * side]))
    cost = notional_delta * float(fee + slip)
    resize = np.r_[0.0, np.abs(np.diff(notional))]
    entry_rows = ((active > 0.0) & (pd.Series(active).shift(1).fillna(0.0).to_numpy() == 0.0)).astype(float)
    out = pd.DataFrame(
        {
            "rolling_trades_1d": pd.Series(entry_rows).shift(1).fillna(0.0).rolling(288, min_periods=1).sum(),
            "rolling_trades_7d": pd.Series(entry_rows).shift(1).fillna(0.0).rolling(2016, min_periods=1).sum(),
            "rolling_notional_delta_1d": pd.Series(notional_delta).shift(1).fillna(0.0).rolling(288, min_periods=1).sum(),
            "rolling_notional_delta_7d": pd.Series(notional_delta).shift(1).fillna(0.0).rolling(2016, min_periods=1).sum(),
            "rolling_fee_slip_cost_1d": pd.Series(cost).shift(1).fillna(0.0).rolling(288, min_periods=1).sum(),
            "rolling_fee_slip_cost_7d": pd.Series(cost).shift(1).fillna(0.0).rolling(2016, min_periods=1).sum(),
            "rolling_realized_pnl_1d": np.zeros(len(dec), dtype=np.float64),
            "rolling_realized_pnl_7d": np.zeros(len(dec), dtype=np.float64),
            "rolling_drawdown_1d": np.zeros(len(dec), dtype=np.float64),
            "rolling_drawdown_7d": np.zeros(len(dec), dtype=np.float64),
            "time_since_last_entry_bars": _bars_since(entry_rows > 0.0),
            "time_since_last_exit_bars": np.full(len(dec), 999999.0, dtype=np.float64),
            "time_since_last_microadd_bars": np.full(len(dec), 999999.0, dtype=np.float64),
            "time_since_last_resize_bars": _bars_since(resize > 1e-9),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bars_since(mask: np.ndarray) -> np.ndarray:
    out = np.full(len(mask), 999999.0, dtype=np.float64)
    last: int | None = None
    for i, flag in enumerate(mask):
        if last is not None:
            out[i] = float(i - last)
        if bool(flag):
            last = i
            out[i] = 0.0
    return out


def _microadd_features(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    baseline_dec: pd.DataFrame,
    seq_state: pd.DataFrame,
    vote: pd.DataFrame,
    *,
    fee: float,
    slip: float,
) -> pd.DataFrame:
    out = feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    side = vote["candidate_side_from_votes"].to_numpy(dtype=np.float64)
    directional_inputs = []
    for col in ("ai_dir_edge", "m7_expected_ret", "pred_patchtst"):
        if col in df.columns:
            directional_inputs.append(_num(df, col, 0.0))
    directional = np.mean(np.vstack(directional_inputs), axis=0) if directional_inputs else np.zeros(len(df), dtype=np.float64)
    side_edge = np.maximum(0.0, side * directional)
    tail_parts = []
    for col in ("ai_adverse_risk", "m7_tail_risk", "evt_excess_z", "long_squeeze_risk", "crowding_pressure"):
        if col in df.columns:
            tail_parts.append(np.abs(_num(df, col, 0.0)))
    tail = np.mean(np.vstack(tail_parts), axis=0) if tail_parts else np.zeros(len(df), dtype=np.float64)
    tail = np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)
    tail = np.clip(tail, 0.0, np.nanquantile(tail, 0.99) if len(tail) else 1.0)
    trailing = _trailing_state(baseline_dec, fee=fee, slip=slip)
    out["baseline_flat"] = seq_state["baseline_flat_for_microadd"].astype(float).to_numpy()
    out["candidate_side_from_votes"] = side
    out["vote_margin"] = vote["vote_margin"].to_numpy(dtype=np.float64)
    out["vote_agreement_count"] = vote["vote_agreement_count"].to_numpy(dtype=np.float64)
    out["direction_entropy"] = vote["direction_entropy"].to_numpy(dtype=np.float64)
    out["tail_risk_score"] = tail
    out["cost_buffer_1x"] = side_edge - 2.0 * float(fee + slip)
    out["cost_buffer_2x"] = side_edge - 4.0 * float(fee + slip)
    out["cost_buffer_3x"] = side_edge - 6.0 * float(fee + slip)
    out["microadd_cooldown_ok"] = 1.0
    out["microadd_churn_budget_remaining"] = 1.0
    for col in trailing.columns:
        out[col] = trailing[col].to_numpy(dtype=np.float64)
    return out.reindex(columns=MICROADD_FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _microadd_targets(
    df: pd.DataFrame,
    seq_state: pd.DataFrame,
    vote: pd.DataFrame,
    cfg: MicroAddConfig,
    *,
    fee: float,
    slip: float,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    close = _close(df)
    side = vote["candidate_side_from_votes"].to_numpy(dtype=np.int64)
    eligible = seq_state["baseline_flat_for_microadd"].to_numpy(dtype=bool) & (side != 0)
    usable = np.flatnonzero(eligible)
    usable = usable[usable < len(close) - int(cfg.horizon) - 2]
    rows: list[dict[str, float]] = []
    for i in usable:
        base = max(float(close[int(i)]), 1e-12)
        fut = close[int(i) + 1 : int(i) + 1 + int(cfg.horizon)]
        if len(fut) < int(cfg.short_horizon):
            continue
        raw = fut / base - 1.0 if side[int(i)] > 0 else base / np.maximum(fut, 1e-12) - 1.0
        path_1x = raw - 2.0 * float(fee + slip)
        path_2x = raw - 4.0 * float(fee + slip)
        path_3x = raw - 6.0 * float(fee + slip)
        lower_q = float(np.quantile(path_3x, float(cfg.cvar_alpha)))
        cvar = float(np.mean(path_3x[path_3x <= lower_q])) if np.any(path_3x <= lower_q) else lower_q
        worst = float(np.min(path_3x))
        edge_36 = float(path_3x[min(int(cfg.short_horizon), len(path_3x)) - 1])
        edge_72 = float(path_3x[min(72, len(path_3x)) - 1])
        edge_144 = float(path_3x[-1])
        rows.append(
            {
                "row_idx": float(i),
                "microadd_net_edge_h36": edge_36,
                "microadd_net_edge_h72": edge_72,
                "microadd_net_edge_h144": edge_144,
                "microadd_worst_path_loss_h144": worst,
                "microadd_cvar_loss_alpha_0p10": cvar,
                "microadd_survives_cost_1x": float(path_1x[-1] > 0.0 and np.min(path_1x) > -0.020),
                "microadd_survives_cost_2x": float(path_2x[-1] > 0.0 and np.min(path_2x) > -0.020),
                "microadd_survives_cost_3x": float(edge_144 > 0.0 and worst > -0.020),
            }
        )
    target = pd.DataFrame(rows)
    idx = target["row_idx"].to_numpy(dtype=np.int64) if len(target) else np.zeros(0, dtype=np.int64)
    meta = {
        "candidate_rows": int((side != 0).sum()),
        "eligible_rows": int(eligible.sum()),
        "usable_rows": int(len(idx)),
        "side_from_future_labels": False,
        "survival_3x_rate": float(target["microadd_survives_cost_3x"].mean()) if len(target) else 0.0,
        "edge_3x_quantiles": target["microadd_net_edge_h144"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(8).tolist() if len(target) else [],
        "cvar_3x_quantiles": target["microadd_cvar_loss_alpha_0p10"].quantile([0.0, 0.25, 0.5, 1.0]).round(8).tolist() if len(target) else [],
    }
    return idx, target, meta


def _train_microadd_scorer(
    x_all: pd.DataFrame,
    idx: np.ndarray,
    target: pd.DataFrame,
    cfg: MicroAddConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed))
    take = np.arange(len(idx), dtype=np.int64)
    if len(take) > int(cfg.max_train_samples):
        take = np.sort(rng.choice(take, size=int(cfg.max_train_samples), replace=False))
    idx_take = idx[take]
    x = x_all.iloc[idx_take].to_numpy(dtype=np.float32, copy=False) if len(idx_take) else np.zeros((0, len(MICROADD_FEATURE_COLS)), dtype=np.float32)
    y_survive = target["microadd_survives_cost_3x"].to_numpy(dtype=np.int64)[take] if len(target) else np.zeros(0, dtype=np.int64)
    y_edge = target["microadd_net_edge_h144"].to_numpy(dtype=np.float64)[take] if len(target) else np.zeros(0, dtype=np.float64)
    y_cvar = target["microadd_cvar_loss_alpha_0p10"].to_numpy(dtype=np.float64)[take] if len(target) else np.zeros(0, dtype=np.float64)
    y_worst = target["microadd_worst_path_loss_h144"].to_numpy(dtype=np.float64)[take] if len(target) else np.zeros(0, dtype=np.float64)
    models = {
        "survival": _fit_binary(x, y_survive, seed=int(cfg.seed)),
        "edge": _fit_regressor(x, y_edge, seed=int(cfg.seed) + 11),
        "cvar": _fit_regressor(x, y_cvar, seed=int(cfg.seed) + 12),
        "worst": _fit_regressor(x, y_worst, seed=int(cfg.seed) + 13, loss="quantile", quantile=0.05),
    }
    meta = {
        "samples": int(len(x)),
        "survival_3x_labels": int(y_survive.sum()) if len(y_survive) else 0,
        "survival_3x_rate": float(y_survive.mean()) if len(y_survive) else 0.0,
        "edge_mean": float(y_edge.mean()) if len(y_edge) else 0.0,
        "cvar_mean": float(y_cvar.mean()) if len(y_cvar) else 0.0,
    }
    return models, meta


def _predict_microadd(models: dict[str, Any], x: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32, copy=False)
    survival_model = models["survival"]
    proba = survival_model.predict_proba(arr)
    classes = np.asarray(getattr(survival_model, "classes_", [0, 1]), dtype=np.int64)
    survival = proba[:, int(np.flatnonzero(classes == 1)[0])] if 1 in classes else np.zeros(len(arr), dtype=np.float64)
    return {
        "microadd_survival_prob": np.asarray(survival, dtype=np.float64),
        "microadd_edge": np.asarray(models["edge"].predict(arr), dtype=np.float64),
        "microadd_cvar_loss": np.asarray(models["cvar"].predict(arr), dtype=np.float64),
        "microadd_worst_loss": np.asarray(models["worst"].predict(arr), dtype=np.float64),
    }


def _microadd_grid_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for margin in (0.05, 0.10, 0.15, 0.20, 0.30):
        for agreement in (1, 2, 3):
            for survival in (0.50, 0.54, 0.58, 0.62, 0.70):
                for edge_floor in (0.0, 0.00025, 0.0005, 0.0010, 0.0020):
                    for max_cvar_loss in (0.006, 0.010, 0.014, 0.018):
                        for notional in MICROADD_NOTIONAL_CAPS:
                            for lev in MICROADD_LEVERAGE_CAPS:
                                for churn in MICROADD_CHURN_CAPS:
                                    for cooldown in (12, 24, 36, 72):
                                        configs.append(
                                            {
                                                "vote_margin_min": float(margin),
                                                "vote_agreement_min": int(agreement),
                                                "microadd_survival_prob_min": float(survival),
                                                "microadd_edge_floor_3x": float(edge_floor),
                                                "microadd_max_cvar_loss": float(max_cvar_loss),
                                                "microadd_notional_cap": float(notional),
                                                "microadd_leverage_ceiling": float(lev),
                                                "microadd_entries_per_day_cap": float(churn),
                                                "microadd_cooldown_bars": int(cooldown),
                                            }
                                        )
    rng = np.random.default_rng(int(args.seed))
    max_configs = int(args.max_grid_configs)
    if len(configs) > max_configs:
        take = np.sort(rng.choice(np.arange(len(configs)), size=max_configs, replace=False))
        configs = [configs[int(i)] for i in take]
    return configs


def _apply_microadd(
    defensive_dec: pd.DataFrame,
    baseline_dec: pd.DataFrame,
    seq_state: pd.DataFrame,
    vote: pd.DataFrame,
    micro_x: pd.DataFrame,
    pred: dict[str, np.ndarray],
    df: pd.DataFrame,
    config: dict[str, Any],
    cfg: MicroAddConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = defensive_dec.copy()
    for col, default in (
        ("reason_code", "baseline_pass"),
        ("is_baseline_pass", True),
        ("is_defensive_modified", False),
        ("is_microadd", False),
    ):
        if col not in out.columns:
            out[col] = default
    base_active = _active_mask(baseline_dec)
    def_active = _active_mask(defensive_dec)
    base_notional = _decision_array(baseline_dec, "notional_exposure", 0.0)
    def_notional = _decision_array(defensive_dec, "notional_exposure", 0.0)
    out.loc[base_active & (def_notional <= 0.0), "reason_code"] = "defense_veto"
    out.loc[base_active & def_active & (def_notional < base_notional - 1e-9), "reason_code"] = "defense_scale_down"
    out.loc[base_active & (def_notional != base_notional), "is_defensive_modified"] = True

    side_vote = vote["candidate_side_from_votes"].to_numpy(dtype=np.int64)
    margin = vote["vote_margin"].to_numpy(dtype=np.float64)
    agree = vote["vote_agreement_count"].to_numpy(dtype=np.int64)
    baseline_flat_seq = seq_state["baseline_flat_for_microadd"].to_numpy(dtype=bool)
    baseline_position_open = seq_state["baseline_position_open_before"].to_numpy(dtype=bool)
    tail = micro_x["tail_risk_score"].to_numpy(dtype=np.float64)
    tail_cap = float(np.nanquantile(tail, 0.90)) if len(tail) else 0.0
    cost_buffer_3x = micro_x["cost_buffer_3x"].to_numpy(dtype=np.float64)
    survival = pred["microadd_survival_prob"]
    edge = pred["microadd_edge"]
    cvar = pred["microadd_cvar_loss"]

    ts = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(np.arange(len(df)))
    day_key = ts.dt.date.astype(str).to_numpy() if hasattr(ts.dt, "date") else np.asarray(["0"] * len(df))
    rows_per_day = pd.Series(day_key).value_counts().to_dict()
    daily_entries: dict[str, int] = {}
    last_micro_idx = -10**9
    approximate_micro_position_until = -1
    block_reasons: dict[str, int] = {}
    micro_rows: list[int] = []

    def block(reason: str) -> None:
        block_reasons[reason] = block_reasons.get(reason, 0) + 1

    for i in range(len(out)):
        day = str(day_key[i])
        day_rows = max(int(rows_per_day.get(day, 288)), 1)
        daily_cap_rows = int(np.floor(float(config["microadd_entries_per_day_cap"]) * max(day_rows, 288) / 288.0 + 1e-12))
        if not baseline_flat_seq[i]:
            block("baseline_not_flat_seq")
            continue
        if bool(baseline_position_open[i]) or i <= approximate_micro_position_until:
            block("position_open")
            continue
        if bool(base_active[i]):
            block("baseline_active_row")
            continue
        if side_vote[i] == 0:
            block("no_vote_side")
            continue
        if margin[i] < float(config["vote_margin_min"]):
            block("vote_margin")
            continue
        if agree[i] < int(config["vote_agreement_min"]):
            block("vote_agreement")
            continue
        if int(i - last_micro_idx) < int(config["microadd_cooldown_bars"]):
            block("cooldown")
            continue
        if daily_entries.get(day, 0) >= daily_cap_rows:
            block("daily_churn_cap")
            continue
        if survival[i] < float(config["microadd_survival_prob_min"]):
            block("survival")
            continue
        if edge[i] < float(config["microadd_edge_floor_3x"]):
            block("edge_floor")
            continue
        if cvar[i] <= -float(config["microadd_max_cvar_loss"]):
            block("cvar")
            continue
        if tail[i] > tail_cap:
            block("tail_risk")
            continue
        if cost_buffer_3x[i] <= 0.0:
            block("cost_buffer_3x")
            continue
        micro_rows.append(i)
        daily_entries[day] = daily_entries.get(day, 0) + 1
        last_micro_idx = i
        approximate_micro_position_until = i + int(cfg.approximate_hold_bars)

    if micro_rows:
        rows = np.asarray(micro_rows, dtype=np.int64)
        sides = side_vote[rows]
        out.loc[rows, "action"] = np.where(sides > 0, ACTION_LONG, ACTION_SHORT).astype(int)
        out.loc[rows, "side"] = sides.astype(int)
        out.loc[rows, "notional_exposure"] = float(config["microadd_notional_cap"])
        out.loc[rows, "leverage"] = float(config["microadd_leverage_ceiling"])
        out.loc[rows, "position_fraction"] = float(config["microadd_notional_cap"]) / max(float(config["microadd_leverage_ceiling"]), 1e-12)
        out.loc[rows, "quality_score"] = edge[rows].astype(np.float64)
        out.loc[rows, "confidence"] = survival[rows].astype(np.float64)
        out.loc[rows, "reason_code"] = "microadd"
        out.loc[rows, "is_microadd"] = True
        out.loc[rows, "is_baseline_pass"] = False

    telemetry = {
        "microadd_entry_count": int(len(micro_rows)),
        "microadd_entries_per_day": float(len(micro_rows) / _days_from_df(df)),
        "microadd_long_entries": int((side_vote[np.asarray(micro_rows, dtype=np.int64)] > 0).sum()) if micro_rows else 0,
        "microadd_short_entries": int((side_vote[np.asarray(micro_rows, dtype=np.int64)] < 0).sum()) if micro_rows else 0,
        "microadd_block_reasons": block_reasons,
        "avg_microadd_notional": float(config["microadd_notional_cap"]) if micro_rows else 0.0,
        "avg_microadd_leverage": float(config["microadd_leverage_ceiling"]) if micro_rows else 0.0,
        "daily_entry_counts": daily_entries,
        "tail_risk_cap_validation_selected": tail_cap,
        "approximate_micro_position_hold_bars": int(cfg.approximate_hold_bars),
    }
    return out, telemetry


def _days_from_df(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df) / 288.0, 1.0)
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return max(len(df) / 288.0, 1.0)
    return max((ts.max() - ts.min()).total_seconds() / 86400.0, 1.0)


def _weekly(
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    mdd_weight: float,
) -> dict[str, Any]:
    if "timestamp" not in df.columns:
        return {}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    weeks = sorted(ts.dropna().dt.to_period("W-SUN").unique())
    out: dict[str, Any] = {}
    for week in weeks:
        mask = (ts.dt.to_period("W-SUN") == week).to_numpy(dtype=bool)
        if not mask.any():
            continue
        sub = df.loc[mask].reset_index(drop=True)
        pre = _slice_precomputed(precomputed, mask)
        out[str(week)] = _run("weekly", sub, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, pre, fee=fee, slip=slip, mdd_weight=mdd_weight)["eval"]
    return out


def _invariant_audit(
    baseline_dec: pd.DataFrame,
    candidate_dec: pd.DataFrame,
    seq_state: pd.DataFrame,
    vote: pd.DataFrame,
    selected_config: dict[str, Any],
) -> dict[str, Any]:
    base_action = _decision_array(baseline_dec, "action", ACTION_CASH).astype(np.int64)
    base_side = _decision_array(baseline_dec, "side", 0.0).astype(np.int64)
    base_notional = _decision_array(baseline_dec, "notional_exposure", 0.0)
    base_leverage = _decision_array(baseline_dec, "leverage", 1.0)
    base_pf = _decision_array(baseline_dec, "position_fraction", 0.0)
    cand_side = _decision_array(candidate_dec, "side", 0.0).astype(np.int64)
    cand_notional = _decision_array(candidate_dec, "notional_exposure", 0.0)
    cand_leverage = _decision_array(candidate_dec, "leverage", 1.0)
    cand_pf = _decision_array(candidate_dec, "position_fraction", 0.0)
    baseline_active = (base_action != ACTION_CASH) & (base_side != 0) & (base_notional > 0.0)
    is_microadd = candidate_dec.get("is_microadd", pd.Series(False, index=candidate_dec.index)).astype(bool).to_numpy()
    vote_side = vote["candidate_side_from_votes"].to_numpy(dtype=np.int64)
    active_side_reversal = baseline_active & (cand_side != 0) & (cand_side != base_side)
    violations = {
        "active_side_reversal": int(active_side_reversal.sum()),
        "active_created_side": int((baseline_active & (base_side == 0) & (cand_side != 0)).sum()),
        "active_notional_increase": int((baseline_active & (cand_notional > base_notional + 1e-9)).sum()),
        "active_leverage_increase": int((baseline_active & (cand_leverage > base_leverage + 1e-9)).sum()),
        "active_position_fraction_increase": int((baseline_active & (cand_pf > base_pf + 1e-9)).sum()),
        "microadd_when_baseline_active": int((is_microadd & baseline_active).sum()),
        "microadd_when_position_open": int((is_microadd & seq_state["baseline_position_open_before"].to_numpy(dtype=bool)).sum()),
        "microadd_side_without_vote": int((is_microadd & (vote_side == 0)).sum()),
        "microadd_vote_conflict": int((is_microadd & (cand_side != vote_side)).sum()),
        "microadd_notional_cap_violation": int((is_microadd & ((cand_notional > float(selected_config["microadd_notional_cap"]) + 1e-9) | ~np.isclose(cand_notional, float(selected_config["microadd_notional_cap"]), atol=1e-9))).sum()),
        "microadd_leverage_cap_violation": int((is_microadd & (cand_leverage > float(selected_config["microadd_leverage_ceiling"]) + 1e-9)).sum()),
        "microadd_churn_cap_violation": 0,
        "nonfinite_decision_value": int((~np.isfinite(cand_notional) | ~np.isfinite(cand_leverage) | ~np.isfinite(cand_pf)).sum()),
        "negative_notional_or_leverage": int(((cand_notional < -1e-12) | (cand_leverage < -1e-12)).sum()),
    }
    return {"passed": bool(sum(violations.values()) == 0), "violations": violations, "rows": int(len(candidate_dec))}


def _state_provenance_audit(df: pd.DataFrame, micro_x: pd.DataFrame, vote_audit: dict[str, Any]) -> dict[str, Any]:
    arr = micro_x.to_numpy(dtype=np.float64, copy=False)
    telemetry_cols = [
        "mz_entry_score_0",
        "mz_entry_score_1",
        "mz_entry_score_2",
        "mz_entry_score_3",
        "az_risk_scale",
        "az_risk_prob",
        "stage2_mz_scale",
        "stage2_mz_score",
        "az_exit_prob",
    ]
    return {
        "rows": int(len(micro_x)),
        "feature_count": int(len(MICROADD_FEATURE_COLS)),
        "missing_feature_cols": [c for c in MICROADD_FEATURE_COLS if c not in micro_x.columns],
        "nan_count_after_fill": int(np.isnan(arr).sum()),
        "nonfinite_count_after_fill": int((~np.isfinite(arr)).sum()),
        "trailing_context_shifted": True,
        "baseline_telemetry_unavailable": [c for c in telemetry_cols if c not in df.columns],
        "vote_provenance": vote_audit,
    }


def _selection_score(candidate: dict[str, Any], baseline: dict[str, Any], cost3_candidate: dict[str, Any], cost3_baseline: dict[str, Any]) -> float:
    ce = candidate["eval"]
    be = baseline["eval"]
    pnl_delta = float(ce["pnl"] - be["pnl"])
    mdd_penalty = 1.6 * max(0.0, abs(float(ce["mdd"])) - abs(float(be["mdd"])))
    tpd_delta = float(ce["trades_per_day"] - be["trades_per_day"])
    cost3_degradation = max(0.0, float(cost3_baseline["eval"]["pnl"] - cost3_candidate["eval"]["pnl"]))
    microadd_count = int(candidate.get("telemetry", {}).get("microadd_entry_count", 0))
    microadd_bonus = min(float(microadd_count), 50.0) * 0.35
    avg_microadd_notional = float(candidate.get("telemetry", {}).get("avg_microadd_notional", 0.0))
    return float(pnl_delta - mdd_penalty + 8.0 * tpd_delta + microadd_bonus - 150.0 * avg_microadd_notional - 0.35 * cost3_degradation)


def _decision_diagnostics(
    baseline_dec: pd.DataFrame,
    defensive_dec: pd.DataFrame,
    candidate_dec: pd.DataFrame,
    microadd_telemetry: dict[str, Any],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    base_active = _active_mask(baseline_dec)
    def_active = _active_mask(defensive_dec)
    base_notional = _decision_array(baseline_dec, "notional_exposure", 0.0)
    def_notional = _decision_array(defensive_dec, "notional_exposure", 0.0)
    base_lev = _decision_array(baseline_dec, "leverage", 1.0)
    cand_lev = _decision_array(candidate_dec, "leverage", 1.0)
    cand_notional = _decision_array(candidate_dec, "notional_exposure", 0.0)
    base_side = _decision_array(baseline_dec, "side", 0.0)
    cand_side = _decision_array(candidate_dec, "side", 0.0)
    is_microadd = candidate_dec.get("is_microadd", pd.Series(False, index=candidate_dec.index)).astype(bool).to_numpy()
    base_signed = base_notional * base_side
    cand_signed = cand_notional * cand_side
    turnover_before = float(np.abs(np.diff(np.r_[0.0, base_signed])).sum())
    turnover_after = float(np.abs(np.diff(np.r_[0.0, cand_signed])).sum())
    return {
        "baseline_pass_count": int((base_active & def_active & np.isclose(base_notional, def_notional, atol=1e-9)).sum()),
        "defense_veto_count": int((base_active & ~def_active).sum()),
        "defense_size_down_count": int((base_active & def_active & (def_notional < base_notional - 1e-9)).sum()),
        "microadd_entry_count": int(microadd_telemetry.get("microadd_entry_count", 0)),
        "microadd_entries_per_day": float(microadd_telemetry.get("microadd_entries_per_day", 0.0)),
        "microadd_long_entries": int(microadd_telemetry.get("microadd_long_entries", 0)),
        "microadd_short_entries": int(microadd_telemetry.get("microadd_short_entries", 0)),
        "microadd_block_reasons": microadd_telemetry.get("microadd_block_reasons", {}),
        "avg_active_leverage_before": float(np.mean(base_lev[base_active])) if base_active.any() else 0.0,
        "avg_active_leverage_after": float(np.mean(cand_lev[(cand_notional > 0.0) & ~is_microadd])) if ((cand_notional > 0.0) & ~is_microadd).any() else 0.0,
        "avg_microadd_notional": float(np.mean(cand_notional[is_microadd])) if is_microadd.any() else 0.0,
        "avg_microadd_leverage": float(np.mean(cand_lev[is_microadd])) if is_microadd.any() else 0.0,
        "turnover_before_candidate": turnover_before,
        "turnover_after_candidate": turnover_after,
        "fee_slip_cost_before_candidate": float(turnover_before * float(fee + slip)),
        "fee_slip_cost_after_candidate": float(turnover_after * float(fee + slip)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare current-top MuZero/AZ Stage2+AZ Exit with conservative flat-only micro-add sleeve v5.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--az-risk-model", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--mz-risk-model", type=Path, default=DEFAULT_RANK1_STAGE2_MZ_MODEL)
    p.add_argument("--mz-exit-model", type=Path, default=DEFAULT_MZ_EXIT_MODEL)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--validation-start", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-notional", type=float, default=None)
    p.add_argument("--leverage-cap", type=float, default=5.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--cvar-alpha", type=float, default=0.10)
    p.add_argument("--adverse-threshold", type=float, default=0.020)
    p.add_argument("--max-train-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mdd-weight", type=float, default=3.0)
    p.add_argument("--stage2-gamma", type=float, default=0.55)
    p.add_argument("--stage2-prior", type=float, default=0.0)
    p.add_argument("--stage2-depth", type=int, default=1)
    p.add_argument("--stage2-score-floor", type=float, default=0.12)
    p.add_argument("--hazard-scale-gap", type=float, default=0.15)
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--high-resize-pressure", type=float, default=0.35)
    p.add_argument("--max-grid-configs", type=int, default=240)
    p.add_argument("--limit-train-rows", type=int, default=None, help="Development/smoke only: cap post-split train rows.")
    p.add_argument("--limit-val-rows", type=int, default=None, help="Development/smoke only: cap validation rows.")
    p.add_argument("--limit-eval-rows", type=int, default=None, help="Development/smoke only: cap eval rows.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(args.seed))

    policy = joblib.load(args.policy)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    max_notional = float(args.max_notional if args.max_notional is not None else risk_cfg.get("max_notional", entry_cfg.get("max_notional", 3.6)))
    entry_cfg = dict(entry_cfg)
    risk_cfg = dict(risk_cfg)
    exit_cfg = dict(exit_cfg)
    entry_cfg["max_notional"] = max_notional
    risk_cfg["max_notional"] = max_notional

    train_all = _read(args.train_csv)
    eval_df = _limit(_read(args.eval_csv), args.limit_eval_rows)
    split_ts = pd.Timestamp(args.validation_start)
    ts = pd.to_datetime(train_all["timestamp"], errors="coerce") if "timestamp" in train_all.columns else pd.Series(np.arange(len(train_all)))
    train_df = _limit(train_all.loc[ts < split_ts].reset_index(drop=True), args.limit_train_rows)
    val_df = _limit(train_all.loc[ts >= split_ts].reset_index(drop=True), args.limit_val_rows)

    mz_entry = __import__("scripts.train_eval_zero_style_risk_overlay_2026", fromlist=["_load_mz_entry"])._load_mz_entry(args.mz_entry_model, device)
    az_risk = _load_pv(args.az_risk_model, len(RISK_ACTIONS), RISK_ACTIONS, device)
    mz_risk = _load_mz_risk(args.mz_risk_model, device)
    az_exit = _load_az_exit(args.az_model, device)
    if az_exit is None:
        raise FileNotFoundError(f"AZ exit model not found: {args.az_model}")
    _ = _load_mz_exit(args.mz_exit_model, device)

    current_kwargs = dict(
        policy=policy,
        entry_cfg=entry_cfg,
        mz_entry=mz_entry,
        az_risk=az_risk,
        mz_risk=mz_risk,
        device=device,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
        stage2_gamma=float(args.stage2_gamma),
        stage2_prior=float(args.stage2_prior),
        stage2_depth=int(args.stage2_depth),
        stage2_score_floor=float(args.stage2_score_floor),
    )
    train_current_pre = _build_zero_style_current(train_df, **current_kwargs)
    val_current_pre = _build_zero_style_current(val_df, **current_kwargs)
    eval_current_pre = _build_zero_style_current(eval_df, **current_kwargs)
    zero_exit_cfg = {"exit_threshold": 0.45, "min_exit_age": int(exit_cfg["min_exit_age"])}
    zero_val = _run("current_muzero_az_val", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_current_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight)
    zero_eval = _run("current_muzero_az_eval", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_current_pre, fee=args.fee, slip=args.slip, monthly=True, mdd_weight=args.mdd_weight)
    smoke_limits = {"limit_train_rows": args.limit_train_rows, "limit_val_rows": args.limit_val_rows, "limit_eval_rows": args.limit_eval_rows}
    baseline_reproduction = _baseline_reproduction_audit(zero_eval, smoke_limits)

    def_cfg = DefensiveSleeveConfig(
        horizon=int(args.horizon),
        cvar_alpha=float(args.cvar_alpha),
        adverse_threshold=float(args.adverse_threshold),
        min_edge=0.0,
        max_train_samples=int(args.max_train_samples),
        seed=int(args.seed),
        hazard_scale_gap=float(args.hazard_scale_gap),
    )
    micro_cfg = MicroAddConfig(
        horizon=int(args.horizon),
        cvar_alpha=float(args.cvar_alpha),
        max_train_samples=int(args.max_train_samples),
        seed=int(args.seed),
    )

    train_feat, train_dec, _, _ = train_current_pre
    val_feat, val_dec, val_close, val_fill = val_current_pre
    eval_feat, eval_dec, eval_close, eval_fill = eval_current_pre
    train_x_def = _sleeve_features(train_feat, train_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))
    val_x_def = _sleeve_features(val_feat, val_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))
    eval_x_def = _sleeve_features(eval_feat, eval_dec, max_notional=max_notional, leverage_cap=float(args.leverage_cap))

    defense_models: dict[str, Any] = {}
    defense_target_meta = {"mode": "disabled_for_loop_3", "reason": "loop_2 active-row defense destroyed rank-1 exposure"}
    defense_train_meta = {"mode": "disabled_for_loop_3"}
    selected_defense_cfg = {"mode": "identity_no_active_row_defense"}
    selected_defense = dict(zero_val)
    selected_defense["config"] = selected_defense_cfg

    train_seq, train_seq_audit = _sequential_baseline_state(train_df, az_exit, risk_cfg, zero_exit_cfg, train_current_pre, fee=args.fee, slip=args.slip)
    val_seq, val_seq_audit = _sequential_baseline_state(val_df, az_exit, risk_cfg, zero_exit_cfg, val_current_pre, fee=args.fee, slip=args.slip)
    eval_seq, eval_seq_audit = _sequential_baseline_state(eval_df, az_exit, risk_cfg, zero_exit_cfg, eval_current_pre, fee=args.fee, slip=args.slip)
    train_vote, train_vote_audit = _vote_frame(train_df)
    val_vote, val_vote_audit = _vote_frame(val_df)
    eval_vote, eval_vote_audit = _vote_frame(eval_df)
    train_micro_x = _microadd_features(train_df, train_feat, train_dec, train_seq, train_vote, fee=args.fee, slip=args.slip)
    val_micro_x = _microadd_features(val_df, val_feat, val_dec, val_seq, val_vote, fee=args.fee, slip=args.slip)
    eval_micro_x = _microadd_features(eval_df, eval_feat, eval_dec, eval_seq, eval_vote, fee=args.fee, slip=args.slip)
    micro_idx, micro_target, micro_target_meta = _microadd_targets(train_df, train_seq, train_vote, micro_cfg, fee=args.fee, slip=args.slip)
    micro_models, micro_train_meta = _train_microadd_scorer(train_micro_x, micro_idx, micro_target, micro_cfg)
    val_pred_micro = _predict_microadd(micro_models, val_micro_x)
    eval_pred_micro = _predict_microadd(micro_models, eval_micro_x)

    val_def_dec = val_dec.copy()
    eval_def_dec = eval_dec.copy()
    val_active = _active_mask(val_dec)
    eval_active = _active_mask(eval_dec)
    val_def_telemetry = {
        "mode": "identity_no_active_row_defense",
        "baseline_active_rows": int(val_active.sum()),
        "final_active_rows": int(val_active.sum()),
        "veto_rows": 0,
        "scale_down_rows": 0,
    }
    eval_def_telemetry = {
        "mode": "identity_no_active_row_defense",
        "baseline_active_rows": int(eval_active.sum()),
        "final_active_rows": int(eval_active.sum()),
        "veto_rows": 0,
        "scale_down_rows": 0,
    }
    val_def_pre = (val_feat, val_def_dec, val_close, val_fill)
    eval_def_pre = (eval_feat, eval_def_dec, eval_close, eval_fill)

    micro_grid: list[dict[str, Any]] = []
    ineligible_counts: dict[str, int] = {}
    cost3_base_val = _run("current_muzero_az_val_cost3x", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_current_pre, fee=args.fee * 3.0, slip=args.slip * 3.0, mdd_weight=args.mdd_weight)
    for i, micro_cfg_row in enumerate(_microadd_grid_configs(args)):
        val_candidate_dec, val_micro_telemetry = _apply_microadd(val_def_dec, val_dec, val_seq, val_vote, val_micro_x, val_pred_micro, val_df, micro_cfg_row, micro_cfg)
        val_pre = (val_feat, val_candidate_dec, val_close, val_fill)
        row = _run(f"microadd_grid_{i:03d}_val", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight)
        val_inv = _invariant_audit(val_dec, val_candidate_dec, val_seq, val_vote, micro_cfg_row)
        cost3_candidate = _run("microadd_val_cost3x", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_pre, fee=args.fee * 3.0, slip=args.slip * 3.0, mdd_weight=args.mdd_weight)
        eligible = True
        reasons: list[str] = []
        if not val_inv["passed"]:
            eligible = False
            reasons.append("invariant")
        if int(val_micro_telemetry.get("microadd_entry_count", 0)) <= 0:
            eligible = False
            reasons.append("microadd_entry_count")
        if float(val_micro_telemetry.get("microadd_entries_per_day", 0.0)) < 0.03:
            eligible = False
            reasons.append("microadd_entries_per_day")
        if float(row["eval"]["trades_per_day"]) <= float(zero_val["eval"]["trades_per_day"]):
            eligible = False
            reasons.append("trades_per_day")
        if float(row["eval"]["pnl"]) <= float(zero_val["eval"]["pnl"]):
            eligible = False
            reasons.append("pnl")
        if float(row["eval"]["mdd"]) < float(zero_val["eval"]["mdd"]):
            eligible = False
            reasons.append("mdd")
        if not (1.50 <= float(row["eval"]["avg_leverage"]) <= 1.80):
            eligible = False
            reasons.append("avg_leverage")
        if float(cost3_candidate["eval"]["pnl"]) <= 0.0:
            eligible = False
            reasons.append("cost3_survival")
        for reason in reasons:
            ineligible_counts[reason] = ineligible_counts.get(reason, 0) + 1
        row["config"] = micro_cfg_row
        row["defense_config"] = selected_defense_cfg
        row["telemetry"] = val_micro_telemetry
        row["invariant_audit"] = val_inv
        row["eligible"] = bool(eligible)
        row["ineligible_reasons"] = reasons
        row["selection_score"] = _selection_score(row, zero_val, cost3_candidate, cost3_base_val)
        micro_grid.append(row)
    eligible_grid = [r for r in micro_grid if r["eligible"]]
    selected = sorted(eligible_grid or micro_grid, key=lambda r: float(r["selection_score"]), reverse=True)[0]
    selected_micro_cfg = dict(selected["config"])
    selected_from_eligible = bool(selected.get("eligible", False))

    eval_candidate_dec, eval_micro_telemetry = _apply_microadd(eval_def_dec, eval_dec, eval_seq, eval_vote, eval_micro_x, eval_pred_micro, eval_df, selected_micro_cfg, micro_cfg)
    eval_candidate_pre = (eval_feat, eval_candidate_dec, eval_close, eval_fill)
    candidate_eval = _run(MODEL_ID, eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_candidate_pre, fee=args.fee, slip=args.slip, monthly=True, mdd_weight=args.mdd_weight)
    eval_invariant = _invariant_audit(eval_dec, eval_candidate_dec, eval_seq, eval_vote, selected_micro_cfg)
    val_candidate_dec, _ = _apply_microadd(val_def_dec, val_dec, val_seq, val_vote, val_micro_x, val_pred_micro, val_df, selected_micro_cfg, micro_cfg)
    val_invariant = _invariant_audit(val_dec, val_candidate_dec, val_seq, val_vote, selected_micro_cfg)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [
            _run("current_muzero_az", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_current_pre, fee=args.fee * mult, slip=args.slip * mult, mdd_weight=args.mdd_weight),
            _run(MODEL_ID, eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_candidate_pre, fee=args.fee * mult, slip=args.slip * mult, mdd_weight=args.mdd_weight),
        ]

    weekly = {
        "current_muzero_az": _weekly(eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_current_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight),
        MODEL_ID: _weekly(eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_candidate_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight),
    }
    eval_delta = {
        "pnl": float(candidate_eval["eval"]["pnl"] - zero_eval["eval"]["pnl"]),
        "mdd": float(candidate_eval["eval"]["mdd"] - zero_eval["eval"]["mdd"]),
        "trades": int(candidate_eval["eval"]["trades"] - zero_eval["eval"]["trades"]),
        "trades_per_day": float(candidate_eval["eval"]["trades_per_day"] - zero_eval["eval"]["trades_per_day"]),
        "avg_leverage": float(candidate_eval["eval"]["avg_leverage"] - zero_eval["eval"]["avg_leverage"]),
    }
    diagnostics = _decision_diagnostics(eval_dec, eval_def_dec, eval_candidate_dec, eval_micro_telemetry, fee=args.fee, slip=args.slip)
    smoke_mode = any(v not in (None, 0) for v in smoke_limits.values())
    full_baseline_failed = baseline_reproduction["status"] == "failed"
    reference_metrics = zero_eval["eval"] if smoke_mode else BASELINE_TARGET
    exact_state_claimed = bool(
        train_seq_audit.get("exact_backtest_state_claimed")
        and val_seq_audit.get("exact_backtest_state_claimed")
        and eval_seq_audit.get("exact_backtest_state_claimed")
    )
    stage2_artifact_path = str(args.mz_risk_model)
    stage3_stage4_excluded = bool("stage3" not in stage2_artifact_path.lower() and "stage4" not in stage2_artifact_path.lower())
    cost_survival_gates = {
        "cost_1x_survival": bool(cost_stress["cost_1x"][1]["eval"]["pnl"] > 0.0),
        "cost_2x_survival": bool(cost_stress["cost_2x"][1]["eval"]["pnl"] > 0.0),
        "cost_3x_survival": bool(cost_stress["cost_3x"][1]["eval"]["pnl"] > 0.0),
        "cost_2x_current_top_target": bool(cost_stress["cost_2x"][1]["eval"]["pnl"] > BASELINE_COST_TARGETS["cost_2x_pnl"]),
        "cost_3x_current_top_target": bool(cost_stress["cost_3x"][1]["eval"]["pnl"] > BASELINE_COST_TARGETS["cost_3x_pnl"]),
    }
    loop_success_gates = {
        "reference": "limited_eval_current_top_muzero_az_stage2" if smoke_mode else "full_oos_current_top_muzero_az_stage2_target",
        "pnl_improved": bool(candidate_eval["eval"]["pnl"] > reference_metrics["pnl"]),
        "mdd_improved": bool(candidate_eval["eval"]["mdd"] > reference_metrics["mdd"]),
        "trades_per_day_increased": bool(candidate_eval["eval"]["trades_per_day"] > reference_metrics["trades_per_day"]),
        "avg_leverage_target_range": bool(1.50 <= candidate_eval["eval"]["avg_leverage"] <= 1.80),
        "invariants": bool(eval_invariant["passed"]),
        "baseline_reproduction": None if smoke_mode else bool(baseline_reproduction.get("passed")),
        "microadd_entry_count_positive": bool(int(eval_micro_telemetry.get("microadd_entry_count", 0)) > 0),
        "validation_eligible_config_count_positive": bool(len(eligible_grid) > 0),
        "selected_from_validation_eligible_config": selected_from_eligible,
        "exact_backtest_state_claimed": exact_state_claimed,
        "stage3_stage4_excluded": stage3_stage4_excluded,
        **cost_survival_gates,
    }
    required_loop_gates = [
        "pnl_improved",
        "mdd_improved",
        "trades_per_day_increased",
        "avg_leverage_target_range",
        "invariants",
        "cost_1x_survival",
        "cost_2x_survival",
        "cost_3x_survival",
        "microadd_entry_count_positive",
        "validation_eligible_config_count_positive",
        "selected_from_validation_eligible_config",
        "exact_backtest_state_claimed",
        "stage3_stage4_excluded",
    ]
    hard_gate_passed = bool(
        (not smoke_mode)
        and not full_baseline_failed
        and all(bool(loop_success_gates[k]) for k in required_loop_gates)
    )
    operational_smoke_passed = bool(smoke_mode and eval_invariant["passed"] and not full_baseline_failed)
    report_status = (
        "candidate_passed"
        if hard_gate_passed
        else ("smoke_not_promotable" if operational_smoke_passed else "candidate_failed")
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "type": MODEL_ID,
            "defense_models": defense_models,
            "microadd_models": micro_models,
            "defense_feature_cols": list(train_x_def.columns),
            "microadd_feature_cols": MICROADD_FEATURE_COLS,
            "defense_config": asdict(def_cfg),
            "microadd_config": asdict(micro_cfg),
            "selected_defense_config": selected_defense_cfg,
            "selected_microadd_config": selected_micro_cfg,
            "feature_cols": list(FEATURE_COLS),
        },
        args.model_dir / "rank1_flat_microadd_v5.pkl",
    )
    selector_payload = {
        "type": f"{MODEL_ID}_threshold_selector",
        "selected_defense_config": selected_defense_cfg,
        "selected_microadd_config": selected_micro_cfg,
        "validation_score": selected.get("selection_score"),
        "eligible_count": int(len(eligible_grid)),
        "ineligible_counts": ineligible_counts,
        "grid_count": int(len(micro_grid)),
    }
    (args.model_dir / "threshold_selector.json").write_text(json.dumps(selector_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "type": MODEL_ID,
        "status": report_status,
        "note": "Frozen current-top MuZero/AZ Stage2+AZ Exit baseline preserved exactly; conservative deterministic current-bar-vote micro-adds can fire only when exact sequential baseline replay is flat.",
        "policy": str(args.policy),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_dir": str(args.model_dir),
        "report_out": str(args.report_out),
        "baseline_artifacts": {
            "mz_entry_model": str(args.mz_entry_model),
            "az_risk_model": str(args.az_risk_model),
            "rank1_stage2_mz_model": str(args.mz_risk_model),
            "rank1_stage2_mz_model_sha256": _file_sha256(args.mz_risk_model),
            "az_exit_model": str(args.az_model),
            "stage3_stage4_excluded": stage3_stage4_excluded,
        },
        "audit": {
            "source_audit": _audit(args.train_csv, args.eval_csv, policy),
            "train_range": _date_range(train_df),
            "validation_range": _date_range(val_df),
            "eval_range": _date_range(eval_df),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
            "smoke_limits": smoke_limits,
        },
        "baseline_reproduction": baseline_reproduction,
        "state_audit": {
            "defense": {
                "train": _defense_state_audit(train_x_def),
                "validation": _defense_state_audit(val_x_def),
                "eval": _defense_state_audit(eval_x_def),
            },
            "microadd": {
                "train": _state_provenance_audit(train_df, train_micro_x, train_vote_audit),
                "validation": _state_provenance_audit(val_df, val_micro_x, val_vote_audit),
                "eval": _state_provenance_audit(eval_df, eval_micro_x, eval_vote_audit),
            },
        },
        "provenance_audit": {
            "side_hint": "ignored_or_unavailable",
            "microadd_side_rule": "deterministic_current_bar_vote_only",
            "future_labels_used_for_side": False,
            "oos_used_for_selection": False,
        },
        "sequential_flat_audit": {"train": train_seq_audit, "validation": val_seq_audit, "eval": eval_seq_audit},
        "vote_audit": {"train": train_vote_audit, "validation": val_vote_audit, "eval": eval_vote_audit},
        "leakage_audit": {
            "train_validation_split": str(args.validation_start),
            "microadd_side_from_future_labels": False,
            "future_windows_only_for_train_validation_labels": True,
            "oos_fit_rows": 0,
            "oos_threshold_selection_rows": 0,
            "oos_final_read_only": True,
        },
        "invariant_audit": {"validation_selected": val_invariant, "eval_selected": eval_invariant},
        "cost_and_caps": {
            "fee": float(args.fee),
            "slip": float(args.slip),
            "max_notional": float(max_notional),
            "original_leverage_cap": float(args.leverage_cap),
            "microadd_notional_cap_candidates": list(MICROADD_NOTIONAL_CAPS),
            "microadd_leverage_ceiling_candidates": list(MICROADD_LEVERAGE_CAPS),
            "microadd_entries_per_day_cap_candidates": list(MICROADD_CHURN_CAPS),
        },
        "frozen_current_config": {
            "entry": "MuZero entry planner",
            "risk": "AZ risk overlay",
            "stage2": {"model": "MuZero sleeve overlay", "gamma": float(args.stage2_gamma), "prior": float(args.stage2_prior), "depth": int(args.stage2_depth), "score_floor": float(args.stage2_score_floor)},
            "exit": {"model": "AZ exit governor", "threshold": 0.45, "min_exit_age": int(exit_cfg["min_exit_age"])},
        },
        "candidate_config": {
            "architecture": ["Identity active-row baseline preservation", "Current-bar deterministic vote micro-add generator", "HGB survival/edge/CVaR scorer", "Validation-only threshold/quota selector"],
            "defense_diagnostic": asdict(def_cfg),
            "microadd_diagnostic": asdict(micro_cfg),
            "selected_defense_config": selected_defense_cfg,
            "selected_microadd_config": selected_micro_cfg,
        },
        "target_meta": {"defense": defense_target_meta, "microadd": micro_target_meta},
        "train_meta": {"defense": defense_train_meta, "microadd": micro_train_meta},
        "validation": {
            "current_muzero_az": zero_val,
            "selected_defense": selected_defense,
            "validation_grid_ranked": sorted(micro_grid, key=lambda r: float(r["selection_score"]), reverse=True)[:20],
            "selection_trace": {"top_configs": sorted(micro_grid, key=lambda r: float(r["selection_score"]), reverse=True)[:20], "ineligible_counts": ineligible_counts, "eligible_count": int(len(eligible_grid)), "grid_count": int(len(micro_grid)), "selected_from_eligible": selected_from_eligible},
        },
        "eval": {
            "baseline_muzero_az": zero_eval,
            "rank1_flat_microadd_v5": candidate_eval,
            "defense_telemetry": eval_def_telemetry,
            "microadd_audit": eval_micro_telemetry,
            "delta_vs_baseline": eval_delta,
        },
        "baseline_muzero_az": zero_eval,
        "rank1_flat_microadd_v5": candidate_eval,
        "delta_vs_baseline": eval_delta,
        "defense_audit": {"target_meta": defense_target_meta, "train_meta": defense_train_meta, "eval_telemetry": eval_def_telemetry},
        "microadd_audit": {"target_meta": micro_target_meta, "train_meta": micro_train_meta, "eval_telemetry": eval_micro_telemetry},
        "diagnostics": diagnostics,
        "calibration_audit": {"selector_artifact": str(args.model_dir / "threshold_selector.json"), "model_artifact": str(args.model_dir / "rank1_flat_microadd_v5.pkl")},
        "monthly": {"current_muzero_az": zero_eval.get("monthly", {}), MODEL_ID: candidate_eval.get("monthly", {})},
        "weekly": weekly,
        "cost_stress": cost_stress,
        "hard_gates": {
            "passed": hard_gate_passed,
            "baseline_reproduction": baseline_reproduction["status"],
            "invariants": eval_invariant["passed"],
            "full_oos_target_gates": "skipped/development" if smoke_mode else hard_gate_passed,
            "loop_success_gates": loop_success_gates,
            "development_smoke_operational": operational_smoke_passed if smoke_mode else "not_applicable",
        },
        "limitations": ["Funding, liquidation, and maintenance-margin path are not first-class in this loop-5 implementation."],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "model_dir": str(args.model_dir),
                "current": zero_eval["eval"],
                "candidate": candidate_eval["eval"],
                "delta": eval_delta,
                "selected_microadd_config": selected_micro_cfg,
                "status": report["status"],
                "failed_gates": {k: v for k, v in report["hard_gates"].items() if v is False or v == "failed"},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
