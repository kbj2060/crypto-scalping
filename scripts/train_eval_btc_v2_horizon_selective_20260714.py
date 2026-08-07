#!/usr/bin/env python3
"""Train and evaluate the research-only BTC v2 horizon-aligned parent.

The candidate fixes three BTC v1 contract problems: the label and execution
horizons are identical, non-stationary level features are excluded, and entry
requires agreement from models trained over different temporal environments.
Validation selects the entry policy; 2026 OOS is evaluated exactly once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "btc_v2_direction_quality_lgbm_20260714"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DATA_FILES = tuple(ROOT / f"data/splits/year_oos/btc_features_{year}.csv" for year in (2024, 2025, 2026))

TRAIN_END = pd.Timestamp("2025-07-01")
CALIBRATION_START = pd.Timestamp("2025-07-01")
VALIDATION_START = pd.Timestamp("2025-10-01")
OOS_START = pd.Timestamp("2026-01-01")

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
MAKER_FEE_MULT = 0.20
ATR_WINDOW = 192
TP_MULT = 12.0
SL_MULT = 6.0
MIN_TP = 0.075
MIN_SL = 0.040
MAX_TP = 0.22
MAX_SL = 0.12
DEFAULT_MAX_HOLD_BARS = 4 * 24 * 12
MOMENTUM_DIRECTION_BARS = 24 * 12
NOTIONAL = 1.0
LEVERAGE = 1.0

RAW_LEVEL_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "sum_open_interest_value",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "close_btc",
    "volume_btc",
    "quote_volume_btc",
    "ou_halflife",
}
FORBIDDEN_TOKENS = ("target", "future", "label", "pnl", "zigzag")
PATCH_COLUMNS = (
    "log_return",
    "net_taker_ratio",
    "oi_change_rate",
    "realized_vol_ratio",
    "mean_reversion_z",
    "hma_slope",
    "breakout_strength",
    "crowding_pressure",
)
PATCH_WINDOWS = (12, 48, 288)
TEMPORAL_MEMBER_STARTS = (
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-07-01"),
    pd.Timestamp("2025-01-01"),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _read_market() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for path in DATA_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            raise RuntimeError(f"BTC feature contract differs across years: {path}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise RuntimeError("duplicate BTC timestamps")
    delta = out["timestamp"].diff().dropna()
    if not bool((delta == pd.Timedelta(minutes=5)).all()):
        bad = out.loc[delta.ne(pd.Timedelta(minutes=5)).reindex(out.index, fill_value=False), "timestamp"]
        raise RuntimeError(f"BTC data is not a continuous 5-minute tape: {bad.head(10).tolist()}")
    return out


def _atr_sltp(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    previous = np.roll(close, 1)
    previous[0] = close[0]
    true_range = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    atr = pd.Series(true_range).rolling(ATR_WINDOW, min_periods=1).mean().to_numpy(dtype=np.float64)
    atr_pct = atr / np.maximum(close, 1.0e-12)
    tp = np.clip(np.maximum(MIN_TP, atr_pct * TP_MULT), 0.0, MAX_TP)
    sl = np.clip(np.maximum(MIN_SL, atr_pct * SL_MULT), 0.0, MAX_SL)
    if not np.isfinite(atr_pct).all():
        raise RuntimeError("non-finite BTC ATR")
    return atr_pct, tp, sl


@numba.njit(cache=False)
def _execution_labels_numba(
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    max_hold_bars: int,
    fee_rate: float,
    slip_rate: float,
    maker_fee_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(open_px)
    long_return = np.zeros(n, dtype=np.float64)
    short_return = np.zeros(n, dtype=np.float64)
    long_reason = np.zeros(n, dtype=np.int8)
    short_reason = np.zeros(n, dtype=np.int8)
    long_hold = np.zeros(n, dtype=np.int32)
    short_hold = np.zeros(n, dtype=np.int32)
    entry_fee = fee_rate * maker_fee_mult

    for signal_i in range(n - 2):
        entry_i = signal_i + 1
        last_signal_i = min(entry_i + max_hold_bars, n - 2)
        for side in (1, -1):
            entry_price = open_px[entry_i]
            resolved = False
            reason = 0
            exit_signal_i = last_signal_i
            for row_i in range(entry_i, last_signal_i + 1):
                if side > 0:
                    move = (close[row_i] * (1.0 - slip_rate) - entry_price) / entry_price
                else:
                    move = (entry_price - close[row_i] * (1.0 + slip_rate)) / entry_price
                if move >= tp[signal_i]:
                    reason = 1
                    exit_signal_i = row_i
                    resolved = True
                    break
                if move <= -sl[signal_i]:
                    reason = 2
                    exit_signal_i = row_i
                    resolved = True
                    break
                if row_i - entry_i >= max_hold_bars:
                    reason = 3
                    exit_signal_i = row_i
                    resolved = True
                    break
            if not resolved:
                reason = 4

            fill_i = min(exit_signal_i + 1, n - 1)
            limit_price = open_px[fill_i]
            if side > 0:
                limit_touched = high[fill_i] >= limit_price
            else:
                limit_touched = low[fill_i] <= limit_price
            if limit_touched:
                exit_price = limit_price
                exit_fee = fee_rate * maker_fee_mult
            elif side > 0:
                exit_price = close[fill_i] * (1.0 - slip_rate)
                exit_fee = fee_rate
            else:
                exit_price = close[fill_i] * (1.0 + slip_rate)
                exit_fee = fee_rate

            if side > 0:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            net_return = (1.0 - entry_fee) * (1.0 + raw_return - exit_fee) - 1.0
            hold = exit_signal_i - entry_i
            if side > 0:
                long_return[signal_i] = net_return
                long_reason[signal_i] = reason
                long_hold[signal_i] = hold
            else:
                short_return[signal_i] = net_return
                short_reason[signal_i] = reason
                short_hold[signal_i] = hold
    return long_return, short_return, long_reason, short_reason, long_hold, short_hold


def _build_labels(frame: pd.DataFrame, *, max_hold_bars: int) -> pd.DataFrame:
    _, tp, sl = _atr_sltp(frame)
    arrays = [pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float64) for col in ("open", "high", "low", "close")]
    long_ret, short_ret, long_reason, short_reason, long_hold, short_hold = _execution_labels_numba(
        *arrays,
        tp,
        sl,
        int(max_hold_bars),
        FEE_RATE,
        SLIP_RATE,
        MAKER_FEE_MULT,
    )
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "long_return": long_ret,
            "short_return": short_ret,
            "long_win": (long_ret > 0.0).astype(np.int8),
            "short_win": (short_ret > 0.0).astype(np.int8),
            "long_reason": long_reason,
            "short_reason": short_reason,
            "long_hold": long_hold,
            "short_hold": short_hold,
            "take_profit": tp,
            "stop_loss": sl,
        }
    )


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_columns: list[str] = []
    for column in frame.columns:
        lower = str(column).lower()
        if column == "timestamp" or column in RAW_LEVEL_COLUMNS:
            continue
        if any(token in lower for token in FORBIDDEN_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            base_columns.append(str(column))
    if "ou_halflife" in base_columns or any(column in RAW_LEVEL_COLUMNS for column in base_columns):
        raise RuntimeError("non-stationary BTC feature entered the v2 contract")

    out = frame[base_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    patch_features: dict[str, pd.Series] = {}
    for column in PATCH_COLUMNS:
        if column not in out.columns:
            raise RuntimeError(f"missing BTC v2 patch source feature: {column}")
        source = out[column].astype(np.float64)
        for window in PATCH_WINDOWS:
            rolling = source.rolling(window, min_periods=window)
            mean = rolling.mean()
            patch_features[f"patch_{column}_mean_{window}"] = mean
            patch_features[f"patch_{column}_std_{window}"] = rolling.std(ddof=0)
            patch_features[f"patch_{column}_residual_{window}"] = source - mean
    out = pd.concat([out, pd.DataFrame(patch_features, index=out.index)], axis=1)
    return out, list(out.columns)


def _prepare_matrix(features: pd.DataFrame, fit_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    fit = features.loc[fit_mask]
    median = fit.median(axis=0).fillna(0.0)
    lower = fit.quantile(0.001).fillna(median)
    upper = fit.quantile(0.999).fillna(median)
    invalid = lower > upper
    lower.loc[invalid] = median.loc[invalid]
    upper.loc[invalid] = median.loc[invalid]
    clean = features.fillna(median).clip(lower=lower, upper=upper, axis=1)
    matrix = clean.to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite BTC v2 feature matrix after train-only preprocessing")
    return matrix, {
        "columns": list(features.columns),
        "median": median.to_numpy(dtype=np.float32),
        "clip_lower": lower.to_numpy(dtype=np.float32),
        "clip_upper": upper.to_numpy(dtype=np.float32),
    }


def _balanced_environment_weights(y: np.ndarray, timestamp: pd.Series) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int8)
    positive = max(int((labels == 1).sum()), 1)
    negative = max(int((labels == 0).sum()), 1)
    class_weight = np.where(labels == 1, len(labels) / (2.0 * positive), len(labels) / (2.0 * negative))
    quarter = timestamp.dt.to_period("Q").astype(str).to_numpy()
    _, inverse, counts = np.unique(quarter, return_inverse=True, return_counts=True)
    environment_weight = len(labels) / (len(counts) * counts[inverse])
    weight = class_weight * environment_weight
    weight /= max(float(weight.mean()), 1.0e-12)
    return np.clip(weight, 0.05, 20.0).astype(np.float64)


def _safe_auc(y: np.ndarray, probability: np.ndarray) -> float | None:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None


def _fit_side_ensemble(
    matrix: np.ndarray,
    labels: np.ndarray,
    timestamp: pd.Series,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    *,
    side_name: str,
    seed: int,
    n_estimators: int,
) -> tuple[list[lgb.LGBMClassifier], IsotonicRegression, dict[str, Any]]:
    models: list[lgb.LGBMClassifier] = []
    member_reports: list[dict[str, Any]] = []
    calibration_y = labels[calibration_mask]
    if len(np.unique(calibration_y)) != 2:
        raise RuntimeError(f"{side_name} calibration labels do not contain both classes")
    calibration_member_probability: list[np.ndarray] = []

    for member_index, start in enumerate(TEMPORAL_MEMBER_STARTS):
        member_mask = train_mask & timestamp.ge(start).to_numpy()
        member_y = labels[member_mask]
        if len(member_y) < 10_000 or len(np.unique(member_y)) != 2:
            raise RuntimeError(f"{side_name} temporal member {start.date()} has insufficient labels")
        sample_weight = _balanced_environment_weights(member_y, timestamp.loc[member_mask])
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(n_estimators),
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=160,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.20,
            reg_lambda=2.0,
            random_state=int(seed + member_index),
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            matrix[member_mask],
            member_y,
            sample_weight=sample_weight,
            eval_set=[(matrix[calibration_mask], calibration_y)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
        )
        probability = model.predict_proba(matrix[calibration_mask])[:, 1]
        calibration_member_probability.append(probability)
        models.append(model)
        member_reports.append(
            {
                "train_start": start,
                "rows": int(member_mask.sum()),
                "positive_rate": float(member_y.mean()),
                "best_iteration": int(model.best_iteration_ or n_estimators),
                "calibration_auc": _safe_auc(calibration_y, probability),
                "calibration_brier": float(brier_score_loss(calibration_y, probability)),
            }
        )

    raw_mean = np.mean(np.vstack(calibration_member_probability), axis=0)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_mean, calibration_y)
    calibrated = calibrator.predict(raw_mean)
    report = {
        "side": side_name,
        "members": member_reports,
        "calibration_rows": int(calibration_mask.sum()),
        "calibration_positive_rate": float(calibration_y.mean()),
        "raw_ensemble_auc": _safe_auc(calibration_y, raw_mean),
        "raw_ensemble_brier": float(brier_score_loss(calibration_y, raw_mean)),
        "calibrated_brier": float(brier_score_loss(calibration_y, calibrated)),
    }
    return models, calibrator, report


def _ensemble_probability(
    models: list[lgb.LGBMClassifier],
    calibrator: IsotonicRegression,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.vstack([model.predict_proba(matrix)[:, 1] for model in models])
    calibrated_members = np.vstack([calibrator.predict(row) for row in raw])
    return calibrated_members.mean(axis=0), calibrated_members.std(axis=0), raw.mean(axis=0)


def _causal_direction(close: np.ndarray, *, horizon_bars: int = MOMENTUM_DIRECTION_BARS) -> np.ndarray:
    momentum = pd.Series(np.asarray(close, dtype=np.float64)).pct_change(int(horizon_bars)).to_numpy(dtype=np.float64)
    return np.nan_to_num(np.sign(momentum), nan=0.0).astype(np.int8)


def _policy_side(
    direction: np.ndarray,
    long_probability: np.ndarray,
    long_uncertainty: np.ndarray,
    short_probability: np.ndarray,
    short_uncertainty: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    *,
    min_expected_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_lower = np.clip(long_probability - long_uncertainty, 0.0, 1.0)
    short_lower = np.clip(short_probability - short_uncertainty, 0.0, 1.0)
    round_trip_maker_cost = 2.0 * FEE_RATE * MAKER_FEE_MULT
    long_ev = long_lower * tp - (1.0 - long_lower) * sl - round_trip_maker_cost
    short_ev = short_lower * tp - (1.0 - short_lower) * sl - round_trip_maker_cost
    side = np.asarray(direction, dtype=np.int8).copy()
    selected_ev = np.where(side > 0, long_ev, np.where(side < 0, short_ev, -np.inf))
    active = selected_ev >= float(min_expected_value)
    side[~active] = 0
    return side, long_ev, short_ev


def _exit_fill(arrays: dict[str, np.ndarray], signal_i: int, side: int) -> tuple[int, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(arrays["open"]) - 1)
    limit_price = float(arrays["open"][fill_i])
    touched = bool(arrays["high"][fill_i] >= limit_price) if side > 0 else bool(arrays["low"][fill_i] <= limit_price)
    if touched:
        return fill_i, limit_price, FEE_RATE * MAKER_FEE_MULT, "maker_limit"
    close = float(arrays["close"][fill_i])
    price = close * (1.0 - SLIP_RATE) if side > 0 else close * (1.0 + SLIP_RATE)
    return fill_i, price, FEE_RATE, "market_fallback"


def _fresh_forward_replay(
    frame: pd.DataFrame,
    side: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    *,
    max_hold_bars: int,
    notional: float = NOTIONAL,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    if not (len(frame) == len(side) == len(tp) == len(sl)):
        raise RuntimeError("fresh-forward input length mismatch")
    arrays = {column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close")}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_signal_i = -1
    entry_fill_i = -1
    take_profit = 0.0
    stop_loss = 0.0
    equity_curve = np.ones(len(frame), dtype=np.float64)
    trades: list[dict[str, Any]] = []

    for row_i in range(len(frame) - 1):
        if position != 0:
            close = float(arrays["close"][row_i])
            if position > 0:
                move = (close * (1.0 - SLIP_RATE) - entry_price) / entry_price
            else:
                move = (entry_price - close * (1.0 + SLIP_RATE)) / entry_price
            equity = cash * (1.0 + move * notional)
        else:
            move = 0.0
            equity = cash
        equity_curve[row_i] = equity
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1.0e-12) - 1.0)

        if position != 0:
            hold_bars = row_i - entry_fill_i
            reason = ""
            if move >= take_profit:
                reason = "take_profit"
            elif move <= -stop_loss:
                reason = "stop_loss"
            elif hold_bars >= int(max_hold_bars):
                reason = "time_exit"
            if reason:
                fill_i, exit_price, exit_fee, route = _exit_fill(arrays, row_i, position)
                raw_return = (exit_price - entry_price) / entry_price if position > 0 else (entry_price - exit_price) / entry_price
                before = cash
                cash = cash * (1.0 + raw_return * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                trades.append(
                    {
                        "entry_signal_i": entry_signal_i,
                        "entry_fill_i": entry_fill_i,
                        "exit_signal_i": row_i,
                        "exit_fill_i": fill_i,
                        "entry_timestamp": frame["timestamp"].iloc[entry_signal_i],
                        "entry_fill_timestamp": frame["timestamp"].iloc[entry_fill_i],
                        "exit_timestamp": frame["timestamp"].iloc[row_i],
                        "exit_fill_timestamp": frame["timestamp"].iloc[fill_i],
                        "side": position,
                        "reason": reason,
                        "route": route,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "hold_bars": hold_bars,
                        "raw_return": raw_return,
                        "trade_return": trade_return,
                        "win": int(trade_return > 0.0),
                        "notional": notional,
                        "margin_fraction": notional / LEVERAGE,
                        "leverage": LEVERAGE,
                    }
                )
                position = 0
                equity_curve[fill_i] = cash
                continue

        if position != 0 or int(side[row_i]) == 0:
            continue
        candidate_side = int(side[row_i])
        fill_i = row_i + 1
        entry_price_candidate = float(arrays["open"][fill_i])
        touched = bool(arrays["low"][fill_i] <= entry_price_candidate) if candidate_side > 0 else bool(arrays["high"][fill_i] >= entry_price_candidate)
        if not touched:
            continue
        position = candidate_side
        entry_price = entry_price_candidate
        entry_equity = cash
        entry_signal_i = row_i
        entry_fill_i = fill_i
        take_profit = float(tp[row_i])
        stop_loss = float(sl[row_i])
        cash -= cash * FEE_RATE * MAKER_FEE_MULT * notional

    if position != 0:
        fill_i = len(frame) - 1
        close = float(arrays["close"][fill_i])
        exit_price = close * (1.0 - SLIP_RATE) if position > 0 else close * (1.0 + SLIP_RATE)
        raw_return = (exit_price - entry_price) / entry_price if position > 0 else (entry_price - exit_price) / entry_price
        before = cash
        cash = cash * (1.0 + raw_return * notional)
        cash -= before * FEE_RATE * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        trades.append(
            {
                "entry_signal_i": entry_signal_i,
                "entry_fill_i": entry_fill_i,
                "exit_signal_i": fill_i,
                "exit_fill_i": fill_i,
                "entry_timestamp": frame["timestamp"].iloc[entry_signal_i],
                "entry_fill_timestamp": frame["timestamp"].iloc[entry_fill_i],
                "exit_timestamp": frame["timestamp"].iloc[fill_i],
                "exit_fill_timestamp": frame["timestamp"].iloc[fill_i],
                "side": position,
                "reason": "forced_end",
                "route": "market_end",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "hold_bars": fill_i - entry_fill_i,
                "raw_return": raw_return,
                "trade_return": trade_return,
                "win": int(trade_return > 0.0),
                "notional": notional,
                "margin_fraction": notional / LEVERAGE,
                "leverage": LEVERAGE,
            }
        )
    equity_curve[-1] = cash
    peak_curve = np.maximum.accumulate(equity_curve)
    mdd = min(mdd, float(np.min(equity_curve / np.maximum(peak_curve, 1.0e-12) - 1.0)))
    ledger = pd.DataFrame(trades)
    reason_counts = ledger["reason"].value_counts().to_dict() if len(ledger) else {}
    long_count = int((ledger["side"] > 0).sum()) if len(ledger) else 0
    short_count = int((ledger["side"] < 0).sum()) if len(ledger) else 0
    wins = int(ledger["win"].sum()) if len(ledger) else 0
    duration_days = max((frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).total_seconds() / 86400.0, 1.0e-9)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)) if len(ledger) else 0.0,
        "trades_per_day": float(len(ledger) / duration_days),
        "long_entries": long_count,
        "short_entries": short_count,
        "avg_hold_bars": float(ledger["hold_bars"].mean()) if len(ledger) else 0.0,
        "median_hold_bars": float(ledger["hold_bars"].median()) if len(ledger) else 0.0,
        "max_hold_bars_observed": int(ledger["hold_bars"].max()) if len(ledger) else 0,
        "exit_reasons": {str(key): int(value) for key, value in reason_counts.items()},
        "notional": float(notional),
        "margin_fraction": float(notional / LEVERAGE),
        "leverage": float(LEVERAGE),
    }
    return metrics, ledger, equity_curve


def _select_policy(
    frame: pd.DataFrame,
    direction: np.ndarray,
    long_probability: np.ndarray,
    long_uncertainty: np.ndarray,
    short_probability: np.ndarray,
    short_uncertainty: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    *,
    max_hold_bars: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for min_ev in (0.0, 0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200):
        side, _, _ = _policy_side(
            direction,
            long_probability,
            long_uncertainty,
            short_probability,
            short_uncertainty,
            tp,
            sl,
            min_expected_value=min_ev,
        )
        metrics, _, _ = _fresh_forward_replay(frame, side, tp, sl, max_hold_bars=max_hold_bars)
        eligible = metrics["trades"] >= 12 and metrics["pnl"] > 0.0 and metrics["mdd"] >= -15.0
        rows.append(
            {
                "min_expected_value": min_ev,
                **metrics,
                "eligible": bool(eligible),
                "selection_score": float(metrics["pnl"] + 0.50 * metrics["mdd"]),
            }
        )
    grid = pd.DataFrame(rows)
    eligible_grid = grid.loc[grid["eligible"]]
    selection_pass = len(eligible_grid) > 0
    source = eligible_grid if selection_pass else grid.loc[grid["trades"] >= 5]
    if len(source) == 0:
        raise RuntimeError("BTC v2 validation produced fewer than five trades for every policy")
    selected = source.sort_values(["selection_score", "pnl", "mdd"], ascending=False).iloc[0].to_dict()
    selected["selection_pass"] = bool(selection_pass)
    selected["selection_rule"] = "max(pnl + 0.50*mdd), subject to trades>=12, pnl>0, mdd>=-15"
    return selected, grid


def _slice(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(array)[mask]


def _prediction_summary(probability: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    return {
        "mean_probability": float(np.mean(probability)),
        "p10_probability": float(np.quantile(probability, 0.10)),
        "p50_probability": float(np.quantile(probability, 0.50)),
        "p90_probability": float(np.quantile(probability, 0.90)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "p90_uncertainty": float(np.quantile(uncertainty, 0.90)),
    }


def _label_summary(labels: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    current = labels.loc[mask]
    reason_names = {0: "unavailable", 1: "take_profit", 2: "stop_loss", 3: "time_exit", 4: "forced_end"}
    out: dict[str, Any] = {"rows": int(len(current))}
    for side in ("long", "short"):
        reasons = current[f"{side}_reason"].value_counts().sort_index()
        out[side] = {
            "win_rate": float(current[f"{side}_win"].mean()),
            "mean_return": float(current[f"{side}_return"].mean()),
            "median_hold_bars": float(current[f"{side}_hold"].median()),
            "reasons": {reason_names.get(int(key), str(key)): int(value) for key, value in reasons.items()},
        }
    return out


def _feature_drift(matrix: np.ndarray, columns: list[str], train_mask: np.ndarray, oos_mask: np.ndarray) -> list[dict[str, Any]]:
    train = matrix[train_mask].astype(np.float64)
    oos = matrix[oos_mask].astype(np.float64)
    train_median = np.median(train, axis=0)
    train_iqr = np.quantile(train, 0.75, axis=0) - np.quantile(train, 0.25, axis=0)
    oos_median = np.median(oos, axis=0)
    robust_shift = (oos_median - train_median) / np.maximum(train_iqr, 1.0e-6)
    order = np.argsort(np.abs(robust_shift))[::-1][:20]
    return [{"feature": columns[int(index)], "median_iqr_shift": float(robust_shift[index])} for index in order]


def _feature_importance(models: dict[str, list[lgb.LGBMClassifier]], columns: list[str]) -> list[dict[str, Any]]:
    values = np.mean(
        np.vstack([model.booster_.feature_importance(importance_type="gain") for side_models in models.values() for model in side_models]),
        axis=0,
    )
    total = max(float(values.sum()), 1.0e-12)
    order = np.argsort(values)[::-1][:30]
    return [{"feature": columns[int(index)], "gain_share": float(values[index] / total)} for index in order]


def _write_chart(frame: pd.DataFrame, ledger: pd.DataFrame, equity: np.ndarray, output: Path) -> None:
    timestamp = pd.to_datetime(frame["timestamp"])
    price = pd.to_numeric(frame["close"], errors="raise")
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    axes[0].plot(timestamp, price, color="#20262e", linewidth=0.8, label="BTC close")
    if len(ledger):
        for side_value, marker, color, label in ((1, "^", "#168f5b", "Long"), (-1, "v", "#d04a3a", "Short")):
            current = ledger.loc[ledger["side"] == side_value]
            index = current["entry_fill_i"].to_numpy(dtype=np.int64)
            axes[0].scatter(timestamp.iloc[index], price.iloc[index], marker=marker, color=color, s=34, label=label, zorder=3)
    axes[0].set_ylabel("BTC price (USDT)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(alpha=0.18)
    axes[1].plot(timestamp, (equity - 1.0) * 100.0, color="#2369a2", linewidth=1.1)
    axes[1].axhline(0.0, color="#777777", linewidth=0.7)
    axes[1].set_ylabel("Compound PnL (%)")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].grid(alpha=0.18)
    figure.suptitle("BTC v2 fresh-forward OOS | 5-minute decisions | next-bar fills")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    parser.add_argument("--n-estimators", type=int, default=900)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_hold_bars < 288:
        raise RuntimeError("BTC v2 max_hold_bars must be at least one day")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_market", flush=True)
    frame = _read_market()
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    print("stage=build_horizon_aligned_labels", flush=True)
    labels = _build_labels(frame, max_hold_bars=int(args.max_hold_bars))
    print("stage=build_stationary_patch_features", flush=True)
    features, feature_columns = _feature_frame(frame)

    purge = pd.Timedelta(minutes=5 * (int(args.max_hold_bars) + 2))
    train_label_end = TRAIN_END - purge
    calibration_label_end = VALIDATION_START - purge
    train_mask = timestamp.lt(train_label_end).to_numpy()
    calibration_mask = timestamp.ge(CALIBRATION_START).to_numpy() & timestamp.lt(calibration_label_end).to_numpy()
    validation_mask = timestamp.ge(VALIDATION_START).to_numpy() & timestamp.lt(OOS_START).to_numpy()
    oos_mask = timestamp.ge(OOS_START).to_numpy()
    preprocessing_fit_mask = timestamp.lt(TRAIN_END).to_numpy()
    if not all(mask.any() for mask in (train_mask, calibration_mask, validation_mask, oos_mask)):
        raise RuntimeError("empty BTC v2 temporal split")

    print("stage=train_only_preprocess", flush=True)
    matrix, preprocessing = _prepare_matrix(features, preprocessing_fit_mask)
    print("stage=fit_temporal_environment_ensemble", flush=True)
    models: dict[str, list[lgb.LGBMClassifier]] = {}
    calibrators: dict[str, IsotonicRegression] = {}
    training_report: dict[str, Any] = {}
    for side_name in ("long", "short"):
        side_models, calibrator, side_report = _fit_side_ensemble(
            matrix,
            labels[f"{side_name}_win"].to_numpy(dtype=np.int8),
            timestamp,
            train_mask,
            calibration_mask,
            side_name=side_name,
            seed=int(args.seed) + (0 if side_name == "long" else 100),
            n_estimators=int(args.n_estimators),
        )
        models[side_name] = side_models
        calibrators[side_name] = calibrator
        training_report[side_name] = side_report

    print("stage=predict_validation_oos", flush=True)
    probability: dict[str, np.ndarray] = {}
    uncertainty: dict[str, np.ndarray] = {}
    raw_probability: dict[str, np.ndarray] = {}
    for side_name in ("long", "short"):
        probability[side_name], uncertainty[side_name], raw_probability[side_name] = _ensemble_probability(
            models[side_name], calibrators[side_name], matrix
        )
    _, tp_all, sl_all = _atr_sltp(frame)
    direction_all = _causal_direction(pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64))

    validation_frame = frame.loc[validation_mask].reset_index(drop=True)
    validation_direction = _slice(direction_all, validation_mask)
    validation_tp = _slice(tp_all, validation_mask)
    validation_sl = _slice(sl_all, validation_mask)
    print("stage=validation_only_policy_selection", flush=True)
    selected, grid = _select_policy(
        validation_frame,
        validation_direction,
        _slice(probability["long"], validation_mask),
        _slice(uncertainty["long"], validation_mask),
        _slice(probability["short"], validation_mask),
        _slice(uncertainty["short"], validation_mask),
        validation_tp,
        validation_sl,
        max_hold_bars=int(args.max_hold_bars),
    )

    min_ev = float(selected["min_expected_value"])
    validation_side, validation_long_ev, validation_short_ev = _policy_side(
        validation_direction,
        _slice(probability["long"], validation_mask),
        _slice(uncertainty["long"], validation_mask),
        _slice(probability["short"], validation_mask),
        _slice(uncertainty["short"], validation_mask),
        validation_tp,
        validation_sl,
        min_expected_value=min_ev,
    )
    validation_metrics, validation_ledger, validation_equity = _fresh_forward_replay(
        validation_frame, validation_side, validation_tp, validation_sl, max_hold_bars=int(args.max_hold_bars)
    )

    oos_frame = frame.loc[oos_mask].reset_index(drop=True)
    oos_direction = _slice(direction_all, oos_mask)
    oos_tp = _slice(tp_all, oos_mask)
    oos_sl = _slice(sl_all, oos_mask)
    oos_side, oos_long_ev, oos_short_ev = _policy_side(
        oos_direction,
        _slice(probability["long"], oos_mask),
        _slice(uncertainty["long"], oos_mask),
        _slice(probability["short"], oos_mask),
        _slice(uncertainty["short"], oos_mask),
        oos_tp,
        oos_sl,
        min_expected_value=min_ev,
    )
    print("stage=fresh_forward_oos_once", flush=True)
    oos_metrics, oos_ledger, oos_equity = _fresh_forward_replay(
        oos_frame, oos_side, oos_tp, oos_sl, max_hold_bars=int(args.max_hold_bars)
    )
    q1_mask = pd.to_datetime(oos_frame["timestamp"]).lt(pd.Timestamp("2026-04-01")).to_numpy()
    q1_metrics, _, _ = _fresh_forward_replay(
        oos_frame.loc[q1_mask].reset_index(drop=True),
        oos_side[q1_mask],
        oos_tp[q1_mask],
        oos_sl[q1_mask],
        max_hold_bars=int(args.max_hold_bars),
    )

    prediction_frame = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "long_probability": probability["long"],
            "long_uncertainty": uncertainty["long"],
            "short_probability": probability["short"],
            "short_uncertainty": uncertainty["short"],
            "take_profit": tp_all,
            "stop_loss": sl_all,
            "causal_direction": direction_all,
        }
    )
    prediction_frame.loc[validation_mask, "selected_side"] = validation_side
    prediction_frame.loc[validation_mask, "long_expected_value"] = validation_long_ev
    prediction_frame.loc[validation_mask, "short_expected_value"] = validation_short_ev
    prediction_frame.loc[oos_mask, "selected_side"] = oos_side
    prediction_frame.loc[oos_mask, "long_expected_value"] = oos_long_ev
    prediction_frame.loc[oos_mask, "short_expected_value"] = oos_short_ev

    bundle_path = args.out_dir / "btc_v2_research_bundle.joblib"
    report_path = args.out_dir / "report.json"
    grid_path = args.out_dir / "validation_policy_grid.csv"
    prediction_path = args.out_dir / "validation_oos_predictions.csv"
    validation_ledger_path = args.out_dir / "validation_ledger.csv"
    oos_ledger_path = args.out_dir / "oos_ledger.csv"
    chart_path = args.out_dir / "oos_equity_chart.png"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "models": models,
            "calibrators": calibrators,
            "preprocessing": preprocessing,
            "feature_columns": feature_columns,
            "policy": {
                "direction": "sign of causal 1-day BTC return",
                "direction_horizon_bars": MOMENTUM_DIRECTION_BARS,
                "min_expected_value": min_ev,
            },
            "execution": {
                "max_hold_bars": int(args.max_hold_bars),
                "notional": NOTIONAL,
                "margin_fraction": NOTIONAL / LEVERAGE,
                "leverage": LEVERAGE,
                "atr_window": ATR_WINDOW,
                "tp_mult": TP_MULT,
                "sl_mult": SL_MULT,
                "min_tp": MIN_TP,
                "min_sl": MIN_SL,
                "max_tp": MAX_TP,
                "max_sl": MAX_SL,
            },
        },
        bundle_path,
    )
    grid.to_csv(grid_path, index=False)
    prediction_frame.loc[validation_mask | oos_mask].to_csv(prediction_path, index=False)
    validation_ledger.to_csv(validation_ledger_path, index=False)
    oos_ledger.to_csv(oos_ledger_path, index=False)
    _write_chart(oos_frame, oos_ledger, oos_equity, chart_path)

    validation_beats_v1 = validation_metrics["pnl"] > 6.69 and validation_metrics["mdd"] >= -12.11
    oos_beats_v1 = oos_metrics["pnl"] > 10.52 and oos_metrics["mdd"] >= -16.46
    report = {
        "model_id": MODEL_ID,
        "status": "research_candidate",
        "live_changed": False,
        "design": {
            "parent": "side-specific temporal-environment LightGBM ensemble",
            "temporal_members": [str(value.date()) for value in TEMPORAL_MEMBER_STARTS],
            "probability_calibration": "isotonic on purged 2025-07-01..2025-09-23 calibration",
            "direction_head": "sign of causal 1-day BTC return; independent of side win-rate base rates",
            "quality_head": "side-specific calibrated probability minus one ensemble standard deviation, converted to barrier EV",
            "selective_entry": "enter only when the direction head's side has validation-selected positive barrier EV",
            "paper_inspirations": {
                "TabM": "https://arxiv.org/abs/2410.24210",
                "PatchTST": "https://arxiv.org/abs/2211.14730",
                "FOIL": "https://arxiv.org/abs/2406.09130",
            },
        },
        "splits": {
            "train_feature_fit": {"start": frame["timestamp"].iloc[0], "end_exclusive": TRAIN_END},
            "train_labels": {"start": frame["timestamp"].iloc[0], "end_exclusive": train_label_end, "rows": int(train_mask.sum())},
            "calibration_labels": {"start": CALIBRATION_START, "end_exclusive": calibration_label_end, "rows": int(calibration_mask.sum())},
            "validation": {"start": VALIDATION_START, "end_exclusive": OOS_START, "rows": int(validation_mask.sum())},
            "oos": {"start": OOS_START, "end": frame["timestamp"].iloc[-1], "rows": int(oos_mask.sum())},
            "purge_bars": int(args.max_hold_bars + 2),
        },
        "label_contract": {
            "entry": "decision at 5-minute bar close, maker-limit fill at next bar open",
            "exit": "5-minute close TP/SL check, next-bar maker-limit or market fallback",
            "time_exit_bars": int(args.max_hold_bars),
            "time_exit_days": float(args.max_hold_bars / 288.0),
            "same_execution_contract_used_for_label_and_replay": True,
            "direction_horizon_bars": MOMENTUM_DIRECTION_BARS,
            "train": _label_summary(labels, train_mask),
            "calibration": _label_summary(labels, calibration_mask),
        },
        "feature_contract": {
            "feature_count": len(feature_columns),
            "raw_level_columns_excluded": sorted(RAW_LEVEL_COLUMNS),
            "ou_halflife_excluded": True,
            "patch_sources": list(PATCH_COLUMNS),
            "patch_windows_bars": list(PATCH_WINDOWS),
            "train_only_median_and_clip": True,
            "top_oos_robust_shifts": _feature_drift(matrix, feature_columns, preprocessing_fit_mask, oos_mask),
            "top_gain_features": _feature_importance(models, feature_columns),
        },
        "training": training_report,
        "policy_selection": {
            "oos_used_for_selection": False,
            "selected": selected,
            "grid_rows": int(len(grid)),
        },
        "prediction_summary": {
            "validation": {
                "long": _prediction_summary(_slice(probability["long"], validation_mask), _slice(uncertainty["long"], validation_mask)),
                "short": _prediction_summary(_slice(probability["short"], validation_mask), _slice(uncertainty["short"], validation_mask)),
                "active_decision_ratio": float((validation_side != 0).mean()),
            },
            "oos": {
                "long": _prediction_summary(_slice(probability["long"], oos_mask), _slice(uncertainty["long"], oos_mask)),
                "short": _prediction_summary(_slice(probability["short"], oos_mask), _slice(uncertainty["short"], oos_mask)),
                "active_decision_ratio": float((oos_side != 0).mean()),
            },
        },
        "metrics": {
            "validation": validation_metrics,
            "oos_extended_2026_to_2026_07_12": oos_metrics,
            "oos_frozen_q1_2026": q1_metrics,
        },
        "v1_reference": {
            "validation": {"pnl": 6.69, "mdd": -12.11},
            "oos_extended": {"pnl": 10.52, "mdd": -16.46, "trades": 31, "wr": 0.355},
            "comparison_caveat": "v1 uses a learned risk sidecar and asymmetric L0.5/S2.5 scaling; v2 reports symmetric fixed 1.0 notional",
            "validation_beats_v1_pnl_and_mdd": bool(validation_beats_v1),
            "oos_beats_v1_pnl_and_mdd": bool(oos_beats_v1),
        },
        "promotion": {
            "promotion_ready": False,
            "reason": "research candidate only; live adapter/parity tests and Omega artifact-integrity promotion audit are not yet implemented",
            "validation_selection_pass": bool(selected["selection_pass"]),
            "live_model_remains": "BTC v1",
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "warmup": {
            "performed": True,
            "patch_max_lookback_bars": max(PATCH_WINDOWS),
            "oos_features_include_pre_oos_history": True,
        },
        "artifacts": {
            "bundle": bundle_path,
            "validation_policy_grid": grid_path,
            "predictions": prediction_path,
            "validation_ledger": validation_ledger_path,
            "oos_ledger": oos_ledger_path,
            "oos_chart": chart_path,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": report_path, "metrics": report["metrics"], "promotion": report["promotion"]}, default=_json_default, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
