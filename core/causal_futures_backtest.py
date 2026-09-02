#!/usr/bin/env python3
"""Causal thresholding and single-position futures backtesting primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TailThresholds:
    upper: float
    lower: float
    upper_quantile: float
    lower_quantile: float


@dataclass
class BacktestResult:
    equity: np.ndarray
    ledger: pd.DataFrame
    skipped_while_open: int


def fit_tail_thresholds(
    calibration_scores: np.ndarray,
    *,
    upper_quantile: float,
    lower_quantile: float,
) -> TailThresholds:
    """Fit fixed entry thresholds from a completed calibration split only."""
    scores = np.asarray(calibration_scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("calibration_scores must contain at least one finite value")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
    return TailThresholds(
        upper=float(np.quantile(scores, upper_quantile)),
        lower=float(np.quantile(scores, lower_quantile)),
        upper_quantile=float(upper_quantile),
        lower_quantile=float(lower_quantile),
    )


def purged_decision_mask(
    timestamps: pd.Series | pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    horizon_bars: int,
) -> np.ndarray:
    """Select decisions whose complete forward target remains inside ``[start, end)``."""
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    ts = pd.DatetimeIndex(timestamps)
    target_end = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
    if len(ts) > horizon_bars:
        target_end[:-horizon_bars] = ts[horizon_bars:].to_numpy(dtype="datetime64[ns]")
    return np.asarray((ts >= start) & (ts < end) & (target_end < np.datetime64(end)))


def _resolve_trade(
    *,
    side: int,
    entry: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp_move: float,
    sl_move: float,
) -> tuple[float, str, int]:
    if side > 0:
        tp_level = entry * (1.0 + tp_move)
        sl_level = entry * (1.0 - sl_move)
        for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
            if bar_low <= sl_level:
                return -sl_move, "sl", offset
            if bar_high >= tp_level:
                return tp_move, "tp", offset
        return float(close[-1] / entry - 1.0), "timeout", len(close) - 1

    tp_level = entry * (1.0 - tp_move)
    sl_level = entry * (1.0 + sl_move)
    for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
        if bar_high >= sl_level:
            return -sl_move, "sl", offset
        if bar_low <= tp_level:
            return tp_move, "tp", offset
    return float(1.0 - close[-1] / entry), "timeout", len(close) - 1


def _resolve_trade_trailing(
    *,
    side: int,
    entry: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sl_init_move: float,
    arm_move: float,
    trail_move: float,
) -> tuple[float, str, int]:
    """ATR-trailing-stop variant of _resolve_trade: an initial stop at sl_init_move, which
    starts trailing the best-favorable price (by trail_move) once favorable excursion reaches
    arm_move -- never loosens. Moves are fractions of entry, matching tp_move/sl_move's units."""
    if side > 0:
        stop = entry * (1.0 - sl_init_move)
        peak = entry
        armed = False
        for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
            if bar_low <= stop:
                return stop / entry - 1.0, "trail_sl", offset
            if bar_high > peak:
                peak = bar_high
                if not armed and (peak - entry) / entry >= arm_move:
                    armed = True
                if armed:
                    stop = max(stop, peak * (1.0 - trail_move))
        return float(close[-1] / entry - 1.0), "timeout", len(close) - 1

    stop = entry * (1.0 + sl_init_move)
    peak = entry
    armed = False
    for offset, (bar_high, bar_low) in enumerate(zip(high, low)):
        if bar_high >= stop:
            return 1.0 - stop / entry, "trail_sl", offset
        if bar_low < peak:
            peak = bar_low
            if not armed and (entry - peak) / entry >= arm_move:
                armed = True
            if armed:
                stop = min(stop, peak * (1.0 + trail_move))
    return float(1.0 - close[-1] / entry), "timeout", len(close) - 1


def simulate_single_position(
    *,
    timestamps: pd.Series | pd.DatetimeIndex,
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    decision_indices: np.ndarray,
    scores: np.ndarray,
    tp_moves: np.ndarray,
    sl_moves: np.ndarray,
    upper_threshold: float,
    lower_threshold: float,
    horizon_bars: int,
    margin_fraction: float,
    leverage: float,
    roundtrip_cost_rate: float,
    arm_moves: np.ndarray | None = None,
    trail_moves: np.ndarray | None = None,
) -> BacktestResult:
    """Run a chronological, non-overlapping, mark-to-market futures simulation.

    A decision at bar ``i`` uses information through bar ``i`` and enters at bar
    ``i + 1`` open. ``margin_fraction * leverage`` is the account notional, and
    both the price move and execution cost are multiplied by that same notional.

    By default (``arm_moves``/``trail_moves`` both omitted) exits use the fixed
    ``tp_moves``/``sl_moves`` bracket, unchanged from the original behavior. Passing BOTH
    ``arm_moves`` and ``trail_moves`` switches to an ATR-trailing-stop exit instead:
    ``sl_moves`` becomes the initial stop distance, which starts trailing the best-favorable
    price (by ``trail_moves``) once favorable excursion reaches ``arm_moves`` -- ``tp_moves``
    is ignored in this mode but must still be passed (any finite array of the right length).
    """
    ts = pd.DatetimeIndex(timestamps)
    arrays = [
        np.asarray(open_px, dtype=np.float64),
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
    ]
    if any(len(values) != len(ts) for values in arrays):
        raise ValueError("market arrays and timestamps must have equal length")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be positive")
    if margin_fraction <= 0.0 or leverage <= 0.0:
        raise ValueError("margin_fraction and leverage must be positive")
    if roundtrip_cost_rate < 0.0:
        raise ValueError("roundtrip_cost_rate must be non-negative")

    idxs = np.asarray(decision_indices, dtype=np.int64)
    aligned = [
        np.asarray(scores, dtype=np.float64),
        np.asarray(tp_moves, dtype=np.float64),
        np.asarray(sl_moves, dtype=np.float64),
    ]
    if any(len(values) != len(idxs) for values in aligned):
        raise ValueError("scores/tp_moves/sl_moves must align with decision_indices")
    if len(idxs) and np.any(np.diff(idxs) < 0):
        raise ValueError("decision_indices must be sorted")

    trailing = arm_moves is not None or trail_moves is not None
    if trailing:
        if arm_moves is None or trail_moves is None:
            raise ValueError("arm_moves and trail_moves must both be provided together, or neither")
        arm_values = np.asarray(arm_moves, dtype=np.float64)
        trail_values = np.asarray(trail_moves, dtype=np.float64)
        if len(arm_values) != len(idxs) or len(trail_values) != len(idxs):
            raise ValueError("arm_moves/trail_moves must align with decision_indices")

    open_values, high_values, low_values, close_values = arrays
    score_values, tp_values, sl_values = aligned
    notional = float(margin_fraction * leverage)
    account_cost = float(roundtrip_cost_rate * notional)
    equity = np.ones(len(ts), dtype=np.float64)
    cash = 1.0
    filled_through = -1
    occupied_through = -1
    skipped_while_open = 0
    rows: list[dict] = []

    for i, (decision_i, score, tp_move, sl_move) in enumerate(zip(idxs, score_values, tp_values, sl_values)):
        if not np.isfinite(score) or not np.isfinite(tp_move) or not np.isfinite(sl_move):
            continue
        side = 1 if score >= upper_threshold else -1 if score <= lower_threshold else 0
        if side == 0:
            continue
        entry_i = int(decision_i) + 1
        if entry_i >= len(ts):
            continue
        if entry_i <= occupied_through:
            skipped_while_open += 1
            continue

        final_i = min(entry_i + horizon_bars - 1, len(ts) - 1)
        if final_i < entry_i:
            continue
        if filled_through + 1 < entry_i:
            equity[filled_through + 1 : entry_i] = cash

        entry = float(open_values[entry_i])
        if trailing:
            price_move, reason, exit_offset = _resolve_trade_trailing(
                side=side,
                entry=entry,
                high=high_values[entry_i : final_i + 1],
                low=low_values[entry_i : final_i + 1],
                close=close_values[entry_i : final_i + 1],
                sl_init_move=float(sl_move),
                arm_move=float(arm_values[i]),
                trail_move=float(trail_values[i]),
            )
        else:
            price_move, reason, exit_offset = _resolve_trade(
                side=side,
                entry=entry,
                high=high_values[entry_i : final_i + 1],
                low=low_values[entry_i : final_i + 1],
                close=close_values[entry_i : final_i + 1],
                tp_move=float(tp_move),
                sl_move=float(sl_move),
            )
        exit_i = entry_i + exit_offset

        for bar_i in range(entry_i, exit_i + 1):
            if side > 0:
                unrealized = close_values[bar_i] / entry - 1.0
            else:
                unrealized = 1.0 - close_values[bar_i] / entry
            equity[bar_i] = cash * (1.0 + unrealized * notional - account_cost)

        trade_return = float(price_move * notional - account_cost)
        cash *= 1.0 + trade_return
        equity[exit_i] = cash
        filled_through = exit_i
        occupied_through = exit_i
        rows.append(
            {
                "decision_timestamp": ts[int(decision_i)],
                "entry_timestamp": ts[entry_i],
                "exit_timestamp": ts[exit_i],
                "side": side,
                "score": float(score),
                "reason": reason,
                "bars_held": int(exit_offset + 1),
                "price_move": float(price_move),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "notional": notional,
                "execution_cost": account_cost,
                "trade_return": trade_return,
            }
        )

    if filled_through + 1 < len(equity):
        equity[filled_through + 1 :] = cash
    return BacktestResult(
        equity=equity,
        ledger=pd.DataFrame(rows),
        skipped_while_open=skipped_while_open,
    )
