"""Oracle trajectory labels for the BTC shared-backbone policy experiment.

Future prices are used only to construct supervised targets.  The resulting
teacher PnL is diagnostic-only and must never be reported as model performance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class TeacherConfig:
    leverage: float = 3.0
    max_margin_fraction: float = 0.30
    margin_step: float = 0.10
    horizon_bars: int = 48
    one_way_cost_rate: float = 0.0021
    soft_label_temperature: float = 0.0015

    @property
    def actions(self) -> np.ndarray:
        return np.round(
            np.arange(-self.max_margin_fraction, self.max_margin_fraction + self.margin_step / 2, self.margin_step),
            8,
        )


def exit_label(current: float, target: float) -> str:
    if np.isclose(current, target):
        return "hold"
    if np.isclose(target, 0.0):
        return "exit_full"
    if np.isclose(current, 0.0):
        return "enter"
    if np.sign(current) != np.sign(target):
        return "reverse"
    if abs(target) < abs(current):
        return "exit_partial"
    return "increase"


def first_action_utilities(current_margin: float, future_bar_returns: np.ndarray, config: TeacherConfig) -> np.ndarray:
    """Finite-horizon utility for every possible first target margin action."""
    actions = config.actions
    current_index = int(np.argmin(np.abs(actions - current_margin)))
    # Evaluate the first decision plus the optimal remaining plan.
    if len(future_bar_returns) == 1:
        continuation = -config.one_way_cost_rate * config.leverage * np.abs(actions)
    else:
        continuation = _future_value_by_action(future_bar_returns[1:], config)
    first_reward = actions * config.leverage * future_bar_returns[0]
    first_cost = config.one_way_cost_rate * config.leverage * np.abs(actions - actions[current_index])
    return first_reward - first_cost + continuation


def soft_action_probabilities(action_utilities: np.ndarray, config: TeacherConfig) -> np.ndarray:
    if config.soft_label_temperature <= 0.0:
        raise ValueError("soft_label_temperature must be positive")
    scaled = (action_utilities - np.max(action_utilities)) / config.soft_label_temperature
    weights = np.exp(np.clip(scaled, -700.0, 0.0))
    return weights / weights.sum()


def _future_value_by_action(future_bar_returns: np.ndarray, config: TeacherConfig) -> np.ndarray:
    """Return optimal remaining value indexed by current action after the first bar."""
    actions = config.actions
    value_next = -config.one_way_cost_rate * config.leverage * np.abs(actions)
    for bar_return in future_bar_returns[::-1]:
        values = np.empty(len(actions), dtype=np.float64)
        for previous_index, previous in enumerate(actions):
            transition_cost = config.one_way_cost_rate * config.leverage * np.abs(actions - previous)
            values[previous_index] = np.max(actions * config.leverage * bar_return - transition_cost + value_next)
        value_next = values
    return value_next


@njit(cache=True)
def _build_teacher_arrays(opens, closes, actions, leverage, cost, temperature, horizon):
    n = len(opens) - horizon - 1
    na = len(actions)
    hard = np.empty(n)
    soft = np.empty(n)
    absolute = np.empty(n)
    probability = np.empty((n, na))
    equity = np.empty(n)
    current = 0.0
    cash = 1.0
    for i in range(n):
        continuation = np.empty(na)
        for a in range(na):
            continuation[a] = -cost * leverage * abs(actions[a])
        for offset in range(horizon - 1, 0, -1):
            ret = closes[i + 1 + offset] / opens[i + 1 + offset] - 1.0
            values = np.empty(na)
            for previous in range(na):
                best = -1e300
                for action in range(na):
                    utility = actions[action] * leverage * ret - cost * leverage * abs(actions[action] - actions[previous]) + continuation[action]
                    if utility > best:
                        best = utility
                values[previous] = best
            continuation = values
        first_ret = closes[i + 1] / opens[i + 1] - 1.0
        utilities = np.empty(na)
        best_index = 0
        best_utility = -1e300
        for action in range(na):
            utilities[action] = actions[action] * leverage * first_ret - cost * leverage * abs(actions[action] - current) + continuation[action]
            if utilities[action] > best_utility:
                best_utility = utilities[action]
                best_index = action
        total = 0.0
        for action in range(na):
            probability[i, action] = np.exp(max(-700.0, (utilities[action] - best_utility) / temperature))
            total += probability[i, action]
        for action in range(na):
            probability[i, action] /= total
        chosen = actions[best_index]
        account_return = chosen * leverage * first_ret - abs(chosen - current) * leverage * cost
        cash *= 1.0 + account_return
        hard[i] = chosen
        signed = 0.0
        mag = 0.0
        for action in range(na):
            signed += actions[action] * probability[i, action]
            mag += abs(actions[action]) * probability[i, action]
        soft[i] = signed
        absolute[i] = mag
        equity[i] = cash
        current = chosen
    return hard, soft, absolute, probability, equity


@njit(cache=True)
def _build_state_conditioned_utilities(opens, closes, actions, leverage, cost, horizon):
    """Return first-action utility matrices indexed by row, current, action."""
    n = len(opens) - horizon - 1
    na = len(actions)
    result = np.empty((n, na, na))
    for i in range(n):
        continuation = np.empty(na)
        for action in range(na):
            continuation[action] = -cost * leverage * abs(actions[action])
        for offset in range(horizon - 1, 0, -1):
            ret = closes[i + 1 + offset] / opens[i + 1 + offset] - 1.0
            values = np.empty(na)
            for previous in range(na):
                best = -1e300
                for action in range(na):
                    utility = actions[action] * leverage * ret - cost * leverage * abs(actions[action] - actions[previous]) + continuation[action]
                    if utility > best:
                        best = utility
                values[previous] = best
            continuation = values
        first_ret = closes[i + 1] / opens[i + 1] - 1.0
        for current in range(na):
            for action in range(na):
                result[i, current, action] = actions[action] * leverage * first_ret - cost * leverage * abs(actions[action] - actions[current]) + continuation[action]
    return result


def build_teacher_path(frame: pd.DataFrame, config: TeacherConfig) -> pd.DataFrame:
    """Create a sequential teacher trajectory from timestamp/open/close bars."""
    required = {"timestamp", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher frame missing columns: {sorted(missing)}")
    bars = frame.loc[:, ["timestamp", "open", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    if bars["timestamp"].duplicated().any() or not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("teacher bars must have unique sorted timestamps")
    opens = pd.to_numeric(bars["open"], errors="raise").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="raise").to_numpy(dtype=float)
    if np.any(opens <= 0.0) or np.any(closes <= 0.0):
        raise ValueError("teacher prices must be positive")

    hard, soft, absolute, probability, equity = _build_teacher_arrays(opens, closes, config.actions, config.leverage, config.one_way_cost_rate, config.soft_label_temperature, config.horizon_bars)
    current_margin = 0.0
    rows: list[dict] = []
    for i in range(len(hard)):
        next_returns = closes[i + 1 : i + 1 + config.horizon_bars] / opens[i + 1 : i + 1 + config.horizon_bars] - 1.0
        probabilities = probability[i]
        hard_target_margin = float(hard[i])
        soft_target_margin = float(soft[i])
        next_return = float(next_returns[0])
        turnover = abs(hard_target_margin - current_margin)
        account_return = hard_target_margin * config.leverage * next_return - turnover * config.leverage * config.one_way_cost_rate
        rows.append({
            "decision_timestamp": bars.loc[i, "timestamp"], "execution_timestamp": bars.loc[i + 1, "timestamp"],
            "execution_open": float(opens[i + 1]), "execution_close": float(closes[i + 1]),
            "current_margin_fraction": current_margin, "hard_target_margin_fraction": hard_target_margin,
            "teacher_signed_margin_fraction": soft_target_margin,
            "teacher_margin_fraction": float(absolute[i]),
            "teacher_short_probability": float(probabilities[config.actions < 0.0].sum()),
            "teacher_flat_probability": float(probabilities[np.isclose(config.actions, 0.0)].sum()),
            "teacher_long_probability": float(probabilities[config.actions > 0.0].sum()),
            "direction_label": "long" if hard_target_margin > 0 else "short" if hard_target_margin < 0 else "flat",
            "exit_label": exit_label(current_margin, hard_target_margin), "turnover_margin_fraction": turnover,
            "next_bar_price_return": next_return, "account_return_after_cost": account_return, "equity": float(equity[i]),
        })
        for action, action_probability in zip(config.actions, probabilities):
            label = exit_label(current_margin, float(action))
            rows[-1][f"teacher_{label}_probability"] = rows[-1].get(f"teacher_{label}_probability", 0.0) + float(action_probability)
            rows[-1][f"teacher_action_{action:+.1f}_probability"] = float(action_probability)
        current_margin = hard_target_margin
    labels = pd.DataFrame(rows)
    probability_columns = [column for column in labels if column.startswith("teacher_") and column.endswith("_probability")]
    labels[probability_columns] = labels[probability_columns].fillna(0.0)
    return labels


def build_state_conditioned_teacher_labels(frame: pd.DataFrame, config: TeacherConfig) -> pd.DataFrame:
    """Build one cost-aware soft target for every causal current-margin state.

    Future prices are used only in the target utilities.  Unlike
    :func:`build_teacher_path`, no previous oracle decision is carried into a
    later row: ``current_margin_fraction`` is an explicit input state that is
    available to the deployed policy.
    """
    required = {"timestamp", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"teacher frame missing columns: {sorted(missing)}")
    bars = frame.loc[:, ["timestamp", "open", "close"]].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    if bars["timestamp"].duplicated().any() or not bars["timestamp"].is_monotonic_increasing:
        raise ValueError("teacher bars must have unique sorted timestamps")
    opens = pd.to_numeric(bars["open"], errors="raise").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="raise").to_numpy(dtype=float)
    if np.any(opens <= 0.0) or np.any(closes <= 0.0):
        raise ValueError("teacher prices must be positive")

    actions = config.actions
    utilities_by_state = _build_state_conditioned_utilities(
        opens, closes, actions, config.leverage, config.one_way_cost_rate, config.horizon_bars,
    )
    n, action_count = utilities_by_state.shape[:2]
    flat_utilities = utilities_by_state.reshape(n * action_count, action_count)
    current_indices = np.tile(np.arange(action_count), n)
    best_indices = np.argmax(flat_utilities, axis=1)
    best_utilities = flat_utilities[np.arange(len(flat_utilities)), best_indices]
    hold_utilities = flat_utilities[np.arange(len(flat_utilities)), current_indices]
    scaled = (flat_utilities - best_utilities[:, None]) / config.soft_label_temperature
    weights = np.exp(np.clip(scaled, -700.0, 0.0))
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    next_returns = closes[1 : n + 1] / opens[1 : n + 1] - 1.0
    labels: dict[str, np.ndarray] = {
        "decision_timestamp": np.repeat(bars["timestamp"].iloc[:n].to_numpy(), action_count),
        "execution_timestamp": np.repeat(bars["timestamp"].iloc[1 : n + 1].to_numpy(), action_count),
        "current_margin_fraction": actions[current_indices],
        "next_bar_price_return": np.repeat(next_returns, action_count),
        "teacher_best_target_margin_fraction": actions[best_indices],
        "teacher_hold_utility": hold_utilities,
        "teacher_best_utility": best_utilities,
        "teacher_switch_advantage": best_utilities - hold_utilities,
    }
    for action_index, action in enumerate(actions):
        labels[f"teacher_action_{action:+.2f}_utility"] = flat_utilities[:, action_index]
        labels[f"teacher_action_{action:+.2f}_probability"] = probabilities[:, action_index]
    return pd.DataFrame(labels)
