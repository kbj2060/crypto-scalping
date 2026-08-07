#!/usr/bin/env python3
"""Build full-history hindsight oracle strategy labels for ETHUSDT 5-minute bars.

This is a label generator, not a strategy evaluation. Decision-time features are
causal, while future OHLC is intentionally used to define the target. A dynamic
program chooses the maximum-log-return sequence of non-overlapping trades from a
fixed realistic action grid after fees, slippage, and actual funding.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_hmm_confluence_meta_labels_20260724 as base  # noqa: E402


MODEL_ID = "eth_full_oracle_strategy_labels_v1_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
STOP_ATR_GRID = (0.50, 0.75, 1.00, 1.50)
REWARD_R_GRID = (1.00, 1.50, 2.00, 3.00)
HORIZON_GRID = (12, 24, 48, 96)
MAX_HORIZON = max(HORIZON_GRID)
OUTCOME_NAME = {1: "SL", 2: "TP", 3: "AMBIGUOUS", 4: "TIMEOUT"}
FIVE_MINUTES_NS = int(pd.Timedelta(minutes=base.BAR_MINUTES).value)


@dataclass(frozen=True)
class ActionSpec:
    side: int
    stop_atr: float
    reward_r: float
    horizon_bars: int

    @property
    def side_name(self) -> str:
        return "LONG" if self.side > 0 else "SHORT"


@dataclass
class ActionEvaluation:
    specs: list[ActionSpec]
    returns: np.ndarray
    next_index: np.ndarray
    outcome: np.ndarray
    exit_at_open: np.ndarray
    local_best_action: np.ndarray
    local_best_return: np.ndarray
    local_second_return: np.ndarray
    evaluable_rows: int


def action_grid() -> list[ActionSpec]:
    return [
        ActionSpec(side, stop_atr, reward_r, horizon)
        for side in (1, -1)
        for stop_atr in STOP_ATR_GRID
        for reward_r in REWARD_R_GRID
        for horizon in HORIZON_GRID
    ]


def load_full_frame() -> pd.DataFrame:
    frame_2025 = base.load_frame(base.MARKET_2025, base.HMM_2025, base.RISK_ARTIFACT)
    frame_2026 = base.load_frame(base.MARKET_2026, base.HMM_2026, base.RISK_ARTIFACT)
    frame_2025["source_year"] = 2025
    frame_2026["source_year"] = 2026
    frame = pd.concat([frame_2025, frame_2026], ignore_index=True)
    if frame["timestamp"].duplicated().any() or not frame["timestamp"].is_monotonic_increasing:
        raise RuntimeError("combined oracle market frame violates timestamp contract")
    frame["oracle_context_atr192"] = base.compute_atr(frame)
    frame["oracle_context_vwma100"] = base.compute_vwma(frame["close"], frame["volume"], base.VWMA_FAST)
    frame["oracle_context_vwma288"] = base.compute_vwma(frame["close"], frame["volume"], base.VWMA_SLOW)
    return frame


def _funding_vector(
    tape: base.FundingTape,
    entry_ns: np.ndarray,
    exit_ns: np.ndarray,
    entry_fill: np.ndarray,
    side: int,
) -> np.ndarray:
    left = np.searchsorted(tape.timestamp_ns, entry_ns, side="right")
    right = np.searchsorted(tape.timestamp_ns, exit_ns, side="right")
    value = tape.rate_x_price_cumsum[right] - tape.rate_x_price_cumsum[left]
    return -float(side) * value / np.maximum(entry_fill, 1.0e-12)


def _bar_event_codes(
    open_normalized_favorable: np.ndarray,
    open_normalized_adverse: np.ndarray,
    intrabar_favorable: np.ndarray,
    intrabar_adverse: np.ndarray,
    *,
    stop_atr: float,
    target_atr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return first code/offset and whether the first event is an open event."""
    open_stop = open_normalized_adverse >= stop_atr
    open_target = open_normalized_favorable >= target_atr
    stop_hit = intrabar_adverse >= stop_atr
    target_hit = intrabar_favorable >= target_atr
    code = np.zeros(stop_hit.shape, dtype=np.uint8)
    code[open_stop] = 1
    mask = (code == 0) & open_target
    code[mask] = 2
    mask = (code == 0) & stop_hit & target_hit
    code[mask] = 3
    mask = (code == 0) & stop_hit
    code[mask] = 1
    mask = (code == 0) & target_hit
    code[mask] = 2
    has_event = (code != 0).any(axis=1)
    first_offset = (code != 0).argmax(axis=1).astype(np.int16)
    row = np.arange(len(code))
    first_code = np.where(has_event, code[row, first_offset], 4).astype(np.uint8)
    first_open = np.where(
        has_event,
        open_stop[row, first_offset] | open_target[row, first_offset],
        True,
    )
    return has_event, first_offset, first_code, first_open


def evaluate_actions(frame: pd.DataFrame, tape: base.FundingTape) -> ActionEvaluation:
    specs = action_grid()
    n_rows = len(frame)
    evaluable = n_rows - MAX_HORIZON - 1
    if evaluable <= 0:
        raise RuntimeError("insufficient rows for oracle horizon")
    n_actions = len(specs)
    returns = np.full((evaluable, n_actions), -np.inf, dtype=np.float32)
    next_index = np.zeros((evaluable, n_actions), dtype=np.int32)
    outcome = np.zeros((evaluable, n_actions), dtype=np.uint8)
    exit_at_open = np.ones((evaluable, n_actions), dtype=np.uint8)
    local_best_return = np.zeros(evaluable, dtype=np.float64)
    local_second_return = np.zeros(evaluable, dtype=np.float64)
    local_best_action = np.full(evaluable, -1, dtype=np.int16)

    open_price = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    atr = frame["oracle_context_atr192"].to_numpy(float)[:evaluable]
    timestamp_ns = frame["timestamp"].astype("int64").to_numpy()
    sliding = np.lib.stride_tricks.sliding_window_view
    open_window = sliding(open_price[1:], MAX_HORIZON + 1)[:evaluable]
    high_window = sliding(high[1:], MAX_HORIZON)[:evaluable]
    low_window = sliding(low[1:], MAX_HORIZON)[:evaluable]
    decision_index = np.arange(evaluable, dtype=np.int32)
    row = np.arange(evaluable)
    valid_context = np.isfinite(atr) & (atr > 0.0)

    spec_lookup = {(spec.side, spec.stop_atr, spec.reward_r, spec.horizon_bars): index for index, spec in enumerate(specs)}
    for side in (1, -1):
        entry_fill = open_window[:, 0] * (1.0 + side * base.SLIPPAGE_RATE)
        if side > 0:
            favorable = (high_window - entry_fill[:, None]) / atr[:, None]
            adverse = (entry_fill[:, None] - low_window) / atr[:, None]
            open_favorable = (open_window[:, :MAX_HORIZON] - entry_fill[:, None]) / atr[:, None]
            open_adverse = (entry_fill[:, None] - open_window[:, :MAX_HORIZON]) / atr[:, None]
        else:
            favorable = (entry_fill[:, None] - low_window) / atr[:, None]
            adverse = (high_window - entry_fill[:, None]) / atr[:, None]
            open_favorable = (entry_fill[:, None] - open_window[:, :MAX_HORIZON]) / atr[:, None]
            open_adverse = (open_window[:, :MAX_HORIZON] - entry_fill[:, None]) / atr[:, None]

        for stop_atr in STOP_ATR_GRID:
            stop_price = entry_fill - side * stop_atr * atr
            for reward_r in REWARD_R_GRID:
                target_atr = stop_atr * reward_r
                target_price = entry_fill + side * target_atr * atr
                has_event, first_offset, first_code, first_open = _bar_event_codes(
                    open_favorable,
                    open_adverse,
                    favorable,
                    adverse,
                    stop_atr=stop_atr,
                    target_atr=target_atr,
                )
                for horizon in HORIZON_GRID:
                    action_id = spec_lookup[(side, stop_atr, reward_r, horizon)]
                    within = has_event & (first_offset < horizon)
                    event_offset = np.where(within, first_offset, horizon).astype(np.int32)
                    event_code = np.where(within, first_code, 4).astype(np.uint8)
                    event_open = np.where(within, first_open, True)
                    exit_index = decision_index + 1 + event_offset
                    exit_level = open_window[row, event_offset].copy()
                    stop_event = within & (event_code == 1)
                    target_event = within & (event_code == 2)
                    gap_stop = stop_event & event_open
                    exit_level[stop_event & ~gap_stop] = stop_price[stop_event & ~gap_stop]
                    exit_level[target_event] = target_price[target_event]
                    exit_fill = exit_level * (1.0 - side * base.SLIPPAGE_RATE)
                    gross = side * (exit_fill / entry_fill - 1.0)
                    exit_ns = timestamp_ns[exit_index] + np.where(within & ~event_open, FIVE_MINUTES_NS, 0)
                    funding = _funding_vector(
                        tape,
                        timestamp_ns[decision_index + 1],
                        exit_ns,
                        entry_fill,
                        side,
                    )
                    net = gross - base.FEE_RATE - base.FEE_RATE * exit_fill / entry_fill + funding
                    valid = valid_context & (event_code != 3) & np.isfinite(net) & (net > -0.999)
                    action_return = np.where(valid, net, -np.inf)
                    returns[:, action_id] = action_return.astype(np.float32)
                    next_index[:, action_id] = exit_index
                    outcome[:, action_id] = event_code
                    exit_at_open[:, action_id] = event_open.astype(np.uint8)

                    better = action_return > local_best_return
                    local_second_return = np.where(better, local_best_return, np.maximum(local_second_return, action_return))
                    local_best_return = np.where(better, action_return, local_best_return)
                    local_best_action = np.where(better, action_id, local_best_action).astype(np.int16)
    return ActionEvaluation(
        specs=specs,
        returns=returns,
        next_index=next_index,
        outcome=outcome,
        exit_at_open=exit_at_open,
        local_best_action=local_best_action,
        local_best_return=local_best_return,
        local_second_return=local_second_return,
        evaluable_rows=evaluable,
    )


def dynamic_program(evaluation: ActionEvaluation, n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.zeros(n_rows + 1, dtype=np.float64)
    choice = np.full(n_rows, -1, dtype=np.int16)
    for index in range(evaluation.evaluable_rows - 1, -1, -1):
        row_return = evaluation.returns[index].astype(np.float64)
        valid = np.isfinite(row_return) & (row_return > -0.999)
        best_score = -np.inf
        best_action = -1
        if valid.any():
            action_ids = np.flatnonzero(valid)
            scores = np.log1p(row_return[action_ids]) + value[evaluation.next_index[index, action_ids]]
            position = int(np.argmax(scores))
            best_score = float(scores[position])
            best_action = int(action_ids[position])
        skip_score = value[index + 1]
        if best_action >= 0 and best_score > skip_score + 1.0e-12:
            value[index] = best_score
            choice[index] = best_action
        else:
            value[index] = skip_score

    selected_action = np.full(n_rows, -1, dtype=np.int16)
    selected = np.zeros(n_rows, dtype=np.int8)
    index = 0
    while index < evaluation.evaluable_rows:
        action_id = int(choice[index])
        if action_id < 0:
            index += 1
            continue
        selected[index] = 1
        selected_action[index] = action_id
        next_decision = int(evaluation.next_index[index, action_id])
        if next_decision <= index:
            raise RuntimeError("oracle DP produced a non-forward transition")
        index = next_decision
    return value, selected, selected_action


def _trade_detail(
    frame: pd.DataFrame,
    evaluation: ActionEvaluation,
    decision_index: int,
    action_id: int,
) -> dict[str, Any]:
    spec = evaluation.specs[action_id]
    exit_index = int(evaluation.next_index[decision_index, action_id])
    event_code = int(evaluation.outcome[decision_index, action_id])
    is_open = bool(evaluation.exit_at_open[decision_index, action_id])
    open_price = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    atr = float(frame["oracle_context_atr192"].iloc[decision_index])
    entry_index = decision_index + 1
    entry_fill = float(open_price[entry_index] * (1.0 + spec.side * base.SLIPPAGE_RATE))
    stop = float(entry_fill - spec.side * spec.stop_atr * atr)
    target = float(entry_fill + spec.side * spec.stop_atr * spec.reward_r * atr)
    if event_code == 1:
        exit_level = float(open_price[exit_index]) if is_open else stop
    elif event_code == 2:
        exit_level = target
    else:
        exit_level = float(open_price[exit_index])
    exit_fill = float(exit_level * (1.0 - spec.side * base.SLIPPAGE_RATE))
    path_end = exit_index if is_open else exit_index + 1
    if path_end <= entry_index:
        mfe = 0.0
        mae = 0.0
    elif spec.side > 0:
        mfe = float(high[entry_index:path_end].max() / entry_fill - 1.0)
        mae = float(low[entry_index:path_end].min() / entry_fill - 1.0)
    else:
        mfe = float(1.0 - low[entry_index:path_end].min() / entry_fill)
        mae = float(1.0 - high[entry_index:path_end].max() / entry_fill)
    risk_return = spec.stop_atr * atr / entry_fill
    exit_timestamp = frame["timestamp"].iloc[exit_index] + (pd.Timedelta(minutes=base.BAR_MINUTES) if not is_open else pd.Timedelta(0))
    return {
        "decision_index": decision_index,
        "decision_timestamp": frame["timestamp"].iloc[decision_index],
        "entry_index": entry_index,
        "entry_timestamp": frame["timestamp"].iloc[entry_index],
        "event_end_index": exit_index,
        "event_end_timestamp": exit_timestamp,
        "action_id": action_id,
        "side": spec.side,
        "side_name": spec.side_name,
        "stop_atr": spec.stop_atr,
        "reward_r": spec.reward_r,
        "horizon_bars": spec.horizon_bars,
        "entry_fill_price": entry_fill,
        "planned_stop_price": stop,
        "planned_target_price": target,
        "exit_fill_price": exit_fill,
        "outcome": OUTCOME_NAME[event_code],
        "net_return_per_notional": float(evaluation.returns[decision_index, action_id]),
        "mfe_r": mfe / max(risk_return, 1.0e-12),
        "mae_r": mae / max(risk_return, 1.0e-12),
    }


def build_labels(
    frame: pd.DataFrame,
    evaluation: ActionEvaluation,
    value: np.ndarray,
    selected: np.ndarray,
    selected_action: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(frame)
    labels = pd.DataFrame(
        {
            "decision_index": np.arange(n_rows, dtype=np.int32),
            "decision_timestamp": frame["timestamp"].to_numpy(),
            "source_year": frame["source_year"].to_numpy(np.int16),
            "label_evaluable": np.zeros(n_rows, dtype=np.int8),
            "label_invalid_reason": np.full(n_rows, "", dtype=object),
            "oracle_local_action": np.full(n_rows, "SKIP", dtype=object),
            "oracle_local_action_id": np.full(n_rows, -1, dtype=np.int16),
            "oracle_local_net_return_per_notional": np.zeros(n_rows, dtype=np.float32),
            "oracle_local_advantage": np.zeros(n_rows, dtype=np.float32),
            "oracle_local_outcome": np.full(n_rows, "SKIP", dtype=object),
            "oracle_local_event_end_index": np.full(n_rows, -1, dtype=np.int32),
            "oracle_dp_selected": selected,
            "oracle_action": np.full(n_rows, "SKIP", dtype=object),
            "oracle_action_id": selected_action,
            "oracle_net_return_per_notional": np.zeros(n_rows, dtype=np.float32),
            "oracle_dp_log_value_from_here": value[:n_rows],
        }
    )
    labels.loc[: evaluation.evaluable_rows - 1, "label_evaluable"] = 1
    labels.loc[:191, "label_evaluable"] = 0
    labels.loc[:191, "label_invalid_reason"] = "atr_warmup"
    labels.loc[evaluation.evaluable_rows :, "label_invalid_reason"] = "right_censored_max_horizon"

    local_mask = evaluation.local_best_action >= 0
    local_rows = np.flatnonzero(local_mask)
    local_ids = evaluation.local_best_action[local_mask].astype(int)
    labels.loc[local_rows, "oracle_local_action_id"] = local_ids.astype(np.int16)
    labels.loc[local_rows, "oracle_local_action"] = [evaluation.specs[action].side_name for action in local_ids]
    labels.loc[local_rows, "oracle_local_outcome"] = [
        OUTCOME_NAME[int(evaluation.outcome[row, action])]
        for row, action in zip(local_rows, local_ids)
    ]
    labels.loc[local_rows, "oracle_local_event_end_index"] = evaluation.next_index[local_rows, local_ids]
    labels.loc[: evaluation.evaluable_rows - 1, "oracle_local_net_return_per_notional"] = evaluation.local_best_return.astype(np.float32)
    labels.loc[: evaluation.evaluable_rows - 1, "oracle_local_advantage"] = (
        evaluation.local_best_return - evaluation.local_second_return
    ).astype(np.float32)

    selected_rows = np.flatnonzero(selected)
    details = [
        _trade_detail(frame, evaluation, int(index), int(selected_action[index]))
        for index in selected_rows
    ]
    trades = pd.DataFrame(details)
    labels["oracle_side"] = np.zeros(n_rows, dtype=np.int8)
    labels["oracle_stop_atr"] = np.full(n_rows, np.nan, dtype=np.float32)
    labels["oracle_reward_r"] = np.full(n_rows, np.nan, dtype=np.float32)
    labels["oracle_horizon_bars"] = np.zeros(n_rows, dtype=np.int16)
    labels["oracle_outcome"] = np.full(n_rows, "SKIP", dtype=object)
    labels["oracle_entry_index"] = np.full(n_rows, -1, dtype=np.int32)
    labels["oracle_event_end_index"] = np.full(n_rows, -1, dtype=np.int32)
    labels["oracle_entry_timestamp"] = pd.NaT
    labels["oracle_event_end_timestamp"] = pd.NaT
    labels["oracle_mfe_r"] = np.full(n_rows, np.nan, dtype=np.float32)
    labels["oracle_mae_r"] = np.full(n_rows, np.nan, dtype=np.float32)
    if len(trades):
        labels.loc[selected_rows, "oracle_action"] = trades["side_name"].to_numpy()
        labels.loc[selected_rows, "oracle_net_return_per_notional"] = trades["net_return_per_notional"].to_numpy(np.float32)
        labels.loc[selected_rows, "oracle_side"] = trades["side"].to_numpy(np.int8)
        labels.loc[selected_rows, "oracle_stop_atr"] = trades["stop_atr"].to_numpy(np.float32)
        labels.loc[selected_rows, "oracle_reward_r"] = trades["reward_r"].to_numpy(np.float32)
        labels.loc[selected_rows, "oracle_horizon_bars"] = trades["horizon_bars"].to_numpy(np.int16)
        labels.loc[selected_rows, "oracle_outcome"] = trades["outcome"].to_numpy()
        labels.loc[selected_rows, "oracle_entry_index"] = trades["entry_index"].to_numpy(np.int32)
        labels.loc[selected_rows, "oracle_event_end_index"] = trades["event_end_index"].to_numpy(np.int32)
        labels.loc[selected_rows, "oracle_entry_timestamp"] = trades["entry_timestamp"].to_numpy()
        labels.loc[selected_rows, "oracle_event_end_timestamp"] = trades["event_end_timestamp"].to_numpy()
        labels.loc[selected_rows, "oracle_mfe_r"] = trades["mfe_r"].to_numpy(np.float32)
        labels.loc[selected_rows, "oracle_mae_r"] = trades["mae_r"].to_numpy(np.float32)
        trades["oracle_equity"] = (1.0 + trades["net_return_per_notional"].astype(float)).cumprod()

    local_specs = [evaluation.specs[action] if action >= 0 else None for action in labels["oracle_local_action_id"].to_numpy(int)]
    for column, getter, default, dtype in (
        ("oracle_local_stop_atr", lambda spec: spec.stop_atr, np.nan, np.float32),
        ("oracle_local_reward_r", lambda spec: spec.reward_r, np.nan, np.float32),
        ("oracle_local_horizon_bars", lambda spec: spec.horizon_bars, 0, np.int16),
    ):
        labels[column] = np.asarray([getter(spec) if spec is not None else default for spec in local_specs], dtype=dtype)

    hmm_cols = [
        f"{base.HMM_PREFIX}{name}_prob" for name in base.HMM_CLASSES
    ] + [
        f"{base.HMM_PREFIX}confidence",
        f"{base.HMM_PREFIX}margin",
        f"{base.HMM_PREFIX}entropy",
        "regime3_transition_h6_risk_prob",
        "regime3_churn_h6_risk_score",
    ]
    for column in hmm_cols:
        labels[f"feature_{column}"] = frame[column].to_numpy(float)
    probabilities = frame[[f"{base.HMM_PREFIX}{name}_prob" for name in base.HMM_CLASSES]].to_numpy(float)
    dominant = probabilities.argmax(axis=1)
    labels["feature_hmm_dominant"] = np.asarray(base.HMM_CLASSES, dtype=object)[dominant]
    for class_index, name in enumerate(base.HMM_CLASSES):
        labels[f"feature_hmm_{name}_count6"] = (
            pd.Series(dominant == class_index).rolling(6, min_periods=6).sum().fillna(0).to_numpy(np.int8)
        )
    labels["feature_atr192"] = frame["oracle_context_atr192"].to_numpy(float)
    labels["feature_vwma100"] = frame["oracle_context_vwma100"].to_numpy(float)
    labels["feature_vwma288"] = frame["oracle_context_vwma288"].to_numpy(float)
    return labels, trades


def choose_chart_window(trades: pd.DataFrame, hours: int = 12) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamp = pd.to_datetime(trades["decision_timestamp"]).sort_values().reset_index(drop=True)
    values = timestamp.astype("int64").to_numpy()
    width_ns = int(pd.Timedelta(hours=hours).value)
    best_start = timestamp.iloc[0].floor("h")
    best_count = -1
    right = 0
    for left in range(len(values)):
        start = timestamp.iloc[left].floor("h")
        end_ns = int(start.value) + width_ns
        while right < len(values) and values[right] <= end_ns:
            right += 1
        count = right - left
        if count > best_count:
            best_start, best_count = start, count
    return best_start, best_start + pd.Timedelta(hours=hours)


def plot_chart(frame: pd.DataFrame, trades: pd.DataFrame, path: Path) -> dict[str, Any]:
    start, end = choose_chart_window(trades)
    view = frame.loc[frame["timestamp"].between(start, end)]
    shown = trades.loc[pd.to_datetime(trades["decision_timestamp"]).between(start, end)].copy()
    fig, (axis, equity_axis) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    axis.plot(view["timestamp"], view["close"], color="#334155", linewidth=1.0, label="ETH close")
    for _, trade in shown.iterrows():
        color = "#16a34a" if trade["side"] > 0 else "#dc2626"
        marker = "^" if trade["side"] > 0 else "v"
        axis.scatter(trade["entry_timestamp"], trade["entry_fill_price"], marker=marker, s=55, color=color, zorder=5)
        axis.scatter(trade["event_end_timestamp"], trade["exit_fill_price"], marker="x", s=40, color=color, zorder=5)
        axis.plot(
            [trade["entry_timestamp"], trade["event_end_timestamp"]],
            [trade["entry_fill_price"], trade["exit_fill_price"]],
            color=color,
            linewidth=0.8,
            alpha=0.65,
        )
    window_equity = (1.0 + shown.sort_values("event_end_timestamp")["net_return_per_notional"].astype(float)).cumprod()
    equity_axis.step(
        pd.to_datetime(shown.sort_values("event_end_timestamp")["event_end_timestamp"]),
        window_equity,
        where="post",
        color="#0f766e",
        linewidth=1.5,
    )
    axis.set_title(
        f"Full-history DP oracle labels: {start.strftime('%Y-%m-%d %H:%M')} to {end.strftime('%Y-%m-%d %H:%M')}"
    )
    axis.set_ylabel("ETHUSDT price")
    equity_axis.set_ylabel("Oracle equity")
    equity_axis.set_xlabel("UTC")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.15)
    equity_axis.grid(alpha=0.15)
    equity_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"start": str(start), "end": str(end), "trades": int(len(shown))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_full_frame()
    tape, funding_hashes = base.load_funding_tape(frame)
    print(json.dumps({"stage": "evaluate_action_grid", "rows": len(frame), "actions": len(action_grid())}), flush=True)
    evaluation = evaluate_actions(frame, tape)
    print(json.dumps({"stage": "dynamic_program", "evaluable_rows": evaluation.evaluable_rows}), flush=True)
    value, selected, selected_action = dynamic_program(evaluation, len(frame))
    labels, trades = build_labels(frame, evaluation, value, selected, selected_action)

    label_path = OUT_DIR / "full_oracle_strategy_labels.parquet"
    trade_path = OUT_DIR / "oracle_selected_trades.csv"
    grid_path = OUT_DIR / "action_grid.json"
    labels.to_parquet(label_path, index=False)
    trades.to_csv(trade_path, index=False)
    grid_path.write_text(json.dumps([asdict(spec) for spec in evaluation.specs], indent=2), encoding="utf-8")
    chart_path = OUT_DIR / "oracle_trade_chart.png"
    chart = plot_chart(frame, trades, chart_path)

    total_log_return = float(value[0])
    report = {
        "model_id": MODEL_ID,
        "status": "full_history_oracle_labels_generated",
        "purpose": "hindsight_training_target_not_strategy_performance",
        "asset": "ETHUSDT",
        "bar_minutes": base.BAR_MINUTES,
        "date_range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
        "rows": int(len(labels)),
        "evaluable_rows": int(labels["label_evaluable"].sum()),
        "action_count": len(evaluation.specs),
        "action_grid": {
            "sides": ["LONG", "SHORT"],
            "stop_atr": list(STOP_ATR_GRID),
            "reward_r": list(REWARD_R_GRID),
            "horizon_bars": list(HORIZON_GRID),
            "skip_action": True,
        },
        "execution": {
            "entry": "next_bar_open_with_adverse_slippage",
            "fee_rate_per_side": base.FEE_RATE,
            "slippage_rate_per_side": base.SLIPPAGE_RATE,
            "funding": "actual_ETHUSDT_settlements",
            "same_bar_tp_sl": "action_invalid_AMBIGUOUS",
            "notional": 1.0,
            "leverage_assumed": None,
        },
        "oracle": {
            "objective": "maximum_sum_log1p_net_return_non_overlapping",
            "selected_trades": int(len(trades)),
            "long_trades": int((trades["side"] > 0).sum()),
            "short_trades": int((trades["side"] < 0).sum()),
            "positive_trade_rate": float((trades["net_return_per_notional"] > 0.0).mean()),
            "total_log_return": total_log_return,
            "equity_multiple": math.exp(total_log_return) if total_log_return < 700.0 else None,
            "mean_trade_return": float(trades["net_return_per_notional"].mean()),
            "median_trade_return": float(trades["net_return_per_notional"].median()),
            "outcome_counts": trades["outcome"].value_counts().to_dict(),
        },
        "hmm": base.validate_hmm_artifact(base.HMM_ARTIFACT),
        "transition_risk_artifact": {
            "path": str(base.RISK_ARTIFACT),
            "sha256": base.sha256(base.RISK_ARTIFACT),
        },
        "hmm_role": "causal_decision_time_features_only_no_oracle_constraint",
        "future_rows_used_for_label": True,
        "future_rows_used_for_entry_features": False,
        "future_rows_used_for_entry": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "promotion_eligible": False,
        "promotion_blocker": "hindsight oracle labels require a separately split model-training evaluation",
        "funding_hashes": funding_hashes,
        "artifacts": {
            "labels": {"path": str(label_path), "sha256": base.sha256(label_path)},
            "selected_trades": {"path": str(trade_path), "sha256": base.sha256(trade_path)},
            "action_grid": {"path": str(grid_path), "sha256": base.sha256(grid_path)},
            "chart": {"path": str(chart_path), "sha256": base.sha256(chart_path)},
        },
        "chart": chart,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=base._json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "oracle": report["oracle"], "chart": str(chart_path)}, default=base._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
