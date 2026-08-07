#!/usr/bin/env python3
"""Pure BTC v2 research primitives.

The module owns the new candidate contract.  Historical BTC v1 research
scripts stay immutable so their reports remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit


SEEDS = (310713, 310719, 310727)
PURGE_HOURS = 72
MAX_HOLD_BARS = 72
MARGIN_FRACTION = 0.15
LEVERAGE = 2.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
ROUND_TRIP_COST = 0.0014


@dataclass(frozen=True)
class ExecutionContract:
    tp_atr_multiple: float = 8.0
    sl_atr_multiple: float = 4.0
    min_tp: float = 0.008
    max_tp: float = 0.030
    min_sl: float = 0.005
    max_sl: float = 0.015
    max_hold_bars: int = MAX_HOLD_BARS
    notional: float = NOTIONAL


def causal_regime_ids(
    hourly: pd.DataFrame, train_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Four causal BTC states from train-frozen volatility and trailing trend."""
    required = {"rvol_24", "logret_24"}
    missing = required.difference(hourly.columns)
    if missing:
        raise RuntimeError(f"missing regime inputs: {sorted(missing)}")
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(train_mask) != len(hourly) or train_mask.sum() < 100:
        raise RuntimeError("invalid regime training mask")
    volatility = hourly["rvol_24"].to_numpy(dtype=np.float64)
    trend = hourly["logret_24"].to_numpy(dtype=np.float64)
    if not np.isfinite(volatility).all() or not np.isfinite(trend).all():
        raise RuntimeError("non-finite regime input")
    volatility_cut = float(np.median(volatility[train_mask]))
    trend_deadband = float(np.quantile(np.abs(trend[train_mask]), 0.25))
    if volatility_cut <= 0.0 or trend_deadband < 0.0:
        raise RuntimeError("invalid train-frozen regime boundary")
    high_volatility = volatility > volatility_cut
    positive_trend = trend > trend_deadband
    negative_trend = trend < -trend_deadband
    directional_trend = np.where(positive_trend, 1, np.where(negative_trend, 0, trend >= 0.0))
    regime = (2 * high_volatility.astype(np.int8) + directional_trend.astype(np.int8)).astype(np.int8)
    return regime, {
        "volatility_median": volatility_cut,
        "trend_abs_q25_deadband": trend_deadband,
    }


def wave_balanced_weights(action: np.ndarray) -> np.ndarray:
    """Give every contiguous target wave equal total training weight."""
    action = np.asarray(action, dtype=np.int8)
    if not len(action):
        return np.empty(0, dtype=np.float64)
    boundary = np.r_[True, action[1:] != action[:-1]]
    group = np.cumsum(boundary) - 1
    counts = np.bincount(group)
    weights = 1.0 / counts[group]
    return weights / weights.mean()


def fit_classifiers(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seeds: tuple[int, ...] = SEEDS,
    sample_weight: np.ndarray | None = None,
    min_samples_leaf: int = 40,
) -> list[HistGradientBoostingClassifier]:
    models = []
    for seed in seeds:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.04,
            max_iter=220,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=False,
            class_weight="balanced",
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=sample_weight)
        models.append(model)
    return models


def predict_probability(
    models: list[HistGradientBoostingClassifier], x: np.ndarray, classes: int
) -> np.ndarray:
    probability = np.zeros((len(x), classes), dtype=np.float64)
    for model in models:
        raw = model.predict_proba(x)
        for column, cls in enumerate(model.classes_):
            probability[:, int(cls)] += raw[:, column]
    probability /= len(models)
    return probability


def predict_binary(models: list[HistGradientBoostingClassifier], x: np.ndarray) -> np.ndarray:
    return predict_probability(models, x, 2)[:, 1]


def fit_direction_oof(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    balance_waves: bool,
    min_samples_leaf: int = 40,
    sample_weight: np.ndarray | None = None,
) -> tuple[list[HistGradientBoostingClassifier], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) < 1000:
        raise RuntimeError(f"insufficient direction training rows: {len(train_indices)}")
    all_weights = (
        np.asarray(sample_weight, dtype=np.float64)[train_indices].copy()
        if sample_weight is not None
        else np.ones(len(train_indices), dtype=np.float64)
    )
    if balance_waves:
        all_weights *= wave_balanced_weights(y[train_indices])
    all_weights /= all_weights.mean()
    oof = np.full((len(x), 3), np.nan, dtype=np.float64)
    folds: list[dict[str, Any]] = []
    splitter = TimeSeriesSplit(n_splits=5, gap=PURGE_HOURS)
    for fold, (fit_local, test_local) in enumerate(splitter.split(train_indices), start=1):
        fit_idx = train_indices[fit_local]
        test_idx = train_indices[test_local]
        models = fit_classifiers(
            x[fit_idx],
            y[fit_idx],
            seeds=(SEEDS[(fold - 1) % len(SEEDS)],),
            sample_weight=all_weights[fit_local],
            min_samples_leaf=min_samples_leaf,
        )
        oof[test_idx] = predict_probability(models, x[test_idx], 3)
        folds.append(
            {
                "fold": fold,
                "fit_start": int(fit_idx[0]),
                "fit_end": int(fit_idx[-1]),
                "oof_start": int(test_idx[0]),
                "oof_end": int(test_idx[-1]),
                "gap_rows": int(test_idx[0] - fit_idx[-1] - 1),
            }
        )
    final_models = fit_classifiers(
        x[train_indices],
        y[train_indices],
        sample_weight=all_weights,
        min_samples_leaf=min_samples_leaf,
    )
    probability = predict_probability(final_models, x, 3)
    return final_models, probability, oof, folds


def confirmed_events(action: np.ndarray, confirmation_hours: int = 1) -> np.ndarray:
    """Emit an event only after a non-cash side persists causally for N hours."""
    action = np.asarray(action, dtype=np.int8)
    if confirmation_hours < 1:
        raise ValueError("confirmation_hours must be >= 1")
    event = np.zeros(len(action), dtype=bool)
    previous_confirmed = 0
    for i in range(confirmation_hours - 1, len(action)):
        window = action[i - confirmation_hours + 1 : i + 1]
        side = int(window[-1])
        if side == 0:
            previous_confirmed = 0
            continue
        if not np.all(window == side):
            continue
        if side != previous_confirmed:
            event[i] = True
            previous_confirmed = side
    return event


def cadenced_events(
    action: np.ndarray,
    *,
    confirmation_hours: int = 1,
    reentry_hours: int = 0,
) -> np.ndarray:
    """Emit confirmed side changes and optional non-overlapping re-entry events."""
    if reentry_hours < 0:
        raise ValueError("reentry_hours must be >= 0")
    action = np.asarray(action, dtype=np.int8)
    event = np.zeros(len(action), dtype=bool)
    previous_side = 0
    last_event = -10**9
    for i in range(confirmation_hours - 1, len(action)):
        window = action[i - confirmation_hours + 1 : i + 1]
        side = int(window[-1])
        if side == 0:
            previous_side = 0
            continue
        if not np.all(window == side):
            continue
        changed = side != previous_side
        cadence_due = reentry_hours > 0 and side == previous_side and i - last_event >= reentry_hours
        if changed or cadence_due:
            event[i] = True
            last_event = i
        previous_side = side
    return event


def meta_matrix(x: np.ndarray, probability: np.ndarray) -> np.ndarray:
    side = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    margin = np.abs(probability[:, 1] - probability[:, 2])
    signed_side = np.where(side == 1, 1.0, np.where(side == 2, -1.0, 0.0))
    return np.column_stack([x, probability, confidence, margin, signed_side])


def terminal_meta_targets(
    hourly: pd.DataFrame,
    probability: np.ndarray,
    candidate: np.ndarray,
    tape: pd.DataFrame,
    *,
    horizon_bars: int = MAX_HOLD_BARS,
    cost: float = ROUND_TRIP_COST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label an OOF direction event by next-bar, fixed-horizon net return."""
    tape_index = pd.Series(np.arange(len(tape)), index=tape["timestamp"])
    target = np.zeros(len(hourly), dtype=np.int8)
    net_return = np.full(len(hourly), np.nan, dtype=np.float64)
    eligible = np.zeros(len(hourly), dtype=bool)
    action = probability.argmax(axis=1)
    for i in np.flatnonzero(candidate):
        available = hourly["timestamp"].iloc[i] + pd.Timedelta(hours=1)
        if available not in tape_index.index:
            continue
        signal_i = int(tape_index.loc[available])
        entry_i = signal_i + 1
        exit_i = entry_i + horizon_bars
        if exit_i >= len(tape):
            continue
        side = 1 if int(action[i]) == 1 else -1
        raw = side * (float(tape["close"].iloc[exit_i]) / float(tape["open"].iloc[entry_i]) - 1.0)
        net_return[i] = raw - cost
        target[i] = int(net_return[i] > 0.0)
        eligible[i] = True
    return target, net_return, eligible


def execution_meta_targets(
    hourly: pd.DataFrame,
    probability: np.ndarray,
    candidate: np.ndarray,
    tape: pd.DataFrame,
    *,
    contract: ExecutionContract = ExecutionContract(),
    cost: float = ROUND_TRIP_COST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label events with the exact stop-first TP/SL/time-exit execution net return."""
    tape_index = pd.Series(np.arange(len(tape)), index=tape["timestamp"])
    target = np.zeros(len(hourly), dtype=np.int8)
    net_return = np.full(len(hourly), np.nan, dtype=np.float64)
    eligible = np.zeros(len(hourly), dtype=bool)
    action = probability.argmax(axis=1)
    for i in np.flatnonzero(candidate):
        available = hourly["timestamp"].iloc[i] + pd.Timedelta(hours=1)
        if available not in tape_index.index:
            continue
        signal_i = int(tape_index.loc[available])
        entry_i = signal_i + 1
        maximum_exit = entry_i + contract.max_hold_bars
        if maximum_exit >= len(tape):
            continue
        side = 1 if int(action[i]) == 1 else -1
        entry = float(tape["open"].iloc[entry_i])
        atr_pct = float(tape["atr_pct"].iloc[signal_i])
        tp_move = float(np.clip(contract.tp_atr_multiple * atr_pct, contract.min_tp, contract.max_tp))
        sl_move = float(np.clip(contract.sl_atr_multiple * atr_pct, contract.min_sl, contract.max_sl))
        if side > 0:
            tp_price, sl_price = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        else:
            tp_price, sl_price = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
        exit_price = float(tape["close"].iloc[maximum_exit])
        for bar_i in range(entry_i, maximum_exit + 1):
            high = float(tape["high"].iloc[bar_i])
            low = float(tape["low"].iloc[bar_i])
            if side > 0:
                stop_hit, target_hit = low <= sl_price, high >= tp_price
            else:
                stop_hit, target_hit = high >= sl_price, low <= tp_price
            if stop_hit:
                exit_price = float(sl_price)
                break
            if target_hit:
                exit_price = float(tp_price)
                break
        raw = side * (exit_price / entry - 1.0)
        net_return[i] = raw - cost
        target[i] = int(net_return[i] > 0.0)
        eligible[i] = True
    return target, net_return, eligible


def hourly_to_five_signal(hourly: pd.DataFrame, action: np.ndarray, tape: pd.DataFrame) -> np.ndarray:
    event_frame = pd.DataFrame(
        {"timestamp": hourly["timestamp"] + pd.Timedelta(hours=1), "action": np.asarray(action, dtype=np.int8)}
    )
    event_frame = event_frame.loc[event_frame["action"] != 0]
    mapped = tape[["timestamp"]].merge(event_frame, on="timestamp", how="left", validate="one_to_one")
    return mapped["action"].fillna(0).to_numpy(dtype=np.int8)


def replay(
    tape: pd.DataFrame,
    signal: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    contract: ExecutionContract = ExecutionContract(),
    round_trip_cost: float = ROUND_TRIP_COST,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Causal next-bar, one-position replay. Same-bar collision is stop-first."""
    mask = tape["timestamp"].between(start, end).to_numpy()
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("empty replay interval")
    split_start, split_end = int(indices[0]), int(indices[-1])
    open_price = tape["open"].to_numpy(dtype=np.float64)
    high_price = tape["high"].to_numpy(dtype=np.float64)
    low_price = tape["low"].to_numpy(dtype=np.float64)
    close_price = tape["close"].to_numpy(dtype=np.float64)
    atr_pct = tape["atr_pct"].to_numpy(dtype=np.float64)
    equity = peak = 1.0
    mdd = 0.0
    busy_until = split_start - 1
    rows: list[dict[str, Any]] = []
    curve = [{"timestamp": tape["timestamp"].iloc[split_start], "equity": equity}]
    for signal_i in range(split_start, split_end + 1):
        if signal_i <= busy_until or int(signal[signal_i]) == 0 or signal_i + 1 > split_end:
            continue
        entry_i = signal_i + 1
        maximum_exit = min(entry_i + contract.max_hold_bars, split_end)
        if maximum_exit <= entry_i:
            continue
        side = 1 if int(signal[signal_i]) == 1 else -1
        entry = float(open_price[entry_i])
        tp_move = float(np.clip(contract.tp_atr_multiple * atr_pct[signal_i], contract.min_tp, contract.max_tp))
        sl_move = float(np.clip(contract.sl_atr_multiple * atr_pct[signal_i], contract.min_sl, contract.max_sl))
        if side > 0:
            tp_price, sl_price = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        else:
            tp_price, sl_price = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
        exit_i = maximum_exit
        exit_price = float(close_price[exit_i])
        reason = f"max_hold_{contract.max_hold_bars}"
        for bar_i in range(entry_i, maximum_exit + 1):
            if side > 0:
                stop_hit = low_price[bar_i] <= sl_price
                target_hit = high_price[bar_i] >= tp_price
            else:
                stop_hit = high_price[bar_i] >= sl_price
                target_hit = low_price[bar_i] <= tp_price
            if stop_hit:
                exit_i, exit_price, reason = bar_i, float(sl_price), "stop_loss"
                break
            if target_hit:
                exit_i, exit_price, reason = bar_i, float(tp_price), "take_profit"
                break
        raw_return = side * (exit_price / entry - 1.0)
        account_return = float(contract.notional * (raw_return - round_trip_cost))
        equity *= max(1.0 + account_return, 1e-9)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        busy_until = exit_i
        rows.append(
            {
                "signal_timestamp": tape["timestamp"].iloc[signal_i],
                "entry_timestamp": tape["timestamp"].iloc[entry_i],
                "exit_timestamp": tape["timestamp"].iloc[exit_i],
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "raw_return": raw_return,
                "account_return": account_return,
                "meta_score": np.nan,
                "equity": equity,
                "exit_reason": reason,
            }
        )
        curve.append({"timestamp": tape["timestamp"].iloc[exit_i], "equity": equity})
    ledger = pd.DataFrame(rows)
    pnl = equity - 1.0
    metrics = {
        "pnl_pct": 100.0 * pnl,
        "mdd_pct": 100.0 * mdd,
        "calmar": float(pnl / abs(mdd)) if mdd < 0 else 0.0,
        "trades": int(len(ledger)),
        "win_rate": float((ledger["account_return"] > 0).mean()) if len(ledger) else 0.0,
        "long_trades": int((ledger["side"] > 0).sum()) if len(ledger) else 0,
        "short_trades": int((ledger["side"] < 0).sum()) if len(ledger) else 0,
        "exit_reasons": ledger["exit_reason"].value_counts().to_dict() if len(ledger) else {},
    }
    return metrics, ledger, pd.DataFrame(curve)


def monthly_compound(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger.empty:
        return {}
    local = ledger.copy()
    local["month"] = pd.to_datetime(local["exit_timestamp"]).dt.to_period("M").astype(str)
    return {
        str(month): float(100.0 * ((1.0 + rows["account_return"]).prod() - 1.0))
        for month, rows in local.groupby("month", sort=True)
    }


def top_trade_concentration(ledger: pd.DataFrame, top_n: int = 3) -> float | None:
    if ledger.empty:
        return None
    positive = ledger.loc[ledger["account_return"] > 0, "account_return"]
    total = float(ledger["account_return"].sum())
    if total <= 0.0 or positive.empty:
        return None
    return float(positive.nlargest(top_n).sum() / total)
