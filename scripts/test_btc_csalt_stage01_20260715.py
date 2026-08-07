#!/usr/bin/env python3
"""Run CSALT Stage 0 parity/coverage and Stage 1 oracle tests.

This is a non-promotion research path. It reads raw BTC 5-minute bars and raw
funding events, constructs dollar-activity decisions, and compares an immediate
lifecycle oracle (N0) with the finite-horizon SMDP oracle. It never reads saved
trade ledgers or parent exit timestamps and does not modify the live BTC v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
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
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_csalt_stage01_20260715"
FIVE_MINUTE_DIR = ROOT / "data/splits/year_oos"
FUNDING_DIR = ROOT / "binance_data/funding_rate_other"

BAR_MINUTES = 5
HORIZON_BARS = (288, 864, 2016)
PLANNING_BARS = 4034
COOLDOWN_BARS = 36
MARGIN_FRACTION = 0.30
LEVERAGE = 2.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
STOP_ATR = 2.5
TRAIL_ARM_ATR = 3.333333
TRAIL_GIVEBACK_ATR = 8.333333


@dataclass(frozen=True)
class FoldSpec:
    name: str
    decision_end: str
    outcome_cutoff: str


FOLDS = (
    FoldSpec("T1", "2024-03-16 23:55:00", "2024-03-31 23:55:00"),
    FoldSpec("T2", "2024-06-15 23:55:00", "2024-06-30 23:55:00"),
    FoldSpec("T3", "2024-09-15 23:55:00", "2024-09-30 23:55:00"),
    FoldSpec("T4", "2024-12-16 23:55:00", "2024-12-31 23:55:00"),
    FoldSpec("T5", "2025-03-16 23:55:00", "2025-03-31 23:55:00"),
    FoldSpec("T6", "2025-06-15 23:55:00", "2025-06-30 23:55:00"),
)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    side: int
    horizon_bars: int

    @property
    def contract_max_duration(self) -> int:
        if self.side == 0:
            return 0
        return self.horizon_bars + 2


ACTIONS = (
    ActionSpec("CASH", 0, 0),
    ActionSpec("LONG_H24", 1, HORIZON_BARS[0]),
    ActionSpec("SHORT_H24", -1, HORIZON_BARS[0]),
    ActionSpec("LONG_H72", 1, HORIZON_BARS[1]),
    ActionSpec("SHORT_H72", -1, HORIZON_BARS[1]),
    ActionSpec("LONG_H168", 1, HORIZON_BARS[2]),
    ActionSpec("SHORT_H168", -1, HORIZON_BARS[2]),
)


@dataclass
class FundingTape:
    timestamp_ns: np.ndarray
    rate_x_price_cumsum: np.ndarray


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_range_atr(frame: pd.DataFrame) -> np.ndarray:
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    previous = np.r_[np.nan, close[:-1]]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous), np.abs(low - previous)))
    return pd.Series(true_range).rolling(192, min_periods=48).mean().shift(1).to_numpy(dtype=np.float64)


def load_market_tape() -> tuple[pd.DataFrame, dict[str, str]]:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frames = []
    hashes: dict[str, str] = {}
    for year in (2024, 2025):
        path = FIVE_MINUTE_DIR / f"btc_features_{year}.csv"
        hashes[str(path.relative_to(ROOT))] = _sha256(path)
        frames.append(pd.read_csv(path, usecols=columns, parse_dates=["timestamp"]))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    expected = pd.date_range(frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1], freq="5min")
    if len(expected) != len(frame) or not frame["timestamp"].eq(expected).all():
        raise RuntimeError("BTC 5-minute tape is not continuous and unique")
    numeric = frame[columns[1:]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("BTC tape contains invalid OHLCV")
    frame["atr"] = true_range_atr(frame)
    return frame, hashes


def load_funding_tape(frame: pd.DataFrame) -> tuple[FundingTape, dict[str, str]]:
    parts = []
    hashes: dict[str, str] = {}
    for path in sorted(FUNDING_DIR.glob("BTCUSDT-fundingRate-*.zip")):
        if not ("2024-" in path.name or "2025-" in path.name):
            continue
        hashes[str(path.relative_to(ROOT))] = _sha256(path)
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                part = pd.read_csv(handle, usecols=["calc_time", "last_funding_rate"])
        part["timestamp"] = pd.to_datetime(part.pop("calc_time"), unit="ms")
        parts.append(part)
    funding = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    start, end = frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
    funding = funding.loc[funding["timestamp"].between(start, end, inclusive="both")].reset_index(drop=True)
    gaps = funding["timestamp"].diff().dropna()
    if len(funding) < 3 * 365 or not gaps.between(pd.Timedelta(hours=7, minutes=59), pd.Timedelta(hours=8, minutes=1)).all():
        raise RuntimeError("raw BTC funding events do not satisfy the 8-hour contract")
    bar_ns = frame["timestamp"].astype("int64").to_numpy()
    funding_ns = funding["timestamp"].astype("int64").to_numpy()
    completed_bar = np.searchsorted(bar_ns, funding_ns, side="left") - 1
    valid = completed_bar >= 0
    funding_ns = funding_ns[valid]
    rates = funding.loc[valid, "last_funding_rate"].to_numpy(dtype=np.float64)
    settlement_close = frame["close"].to_numpy(dtype=np.float64)[completed_bar[valid]]
    rate_x_price = rates * settlement_close
    return FundingTape(funding_ns, np.r_[0.0, np.cumsum(rate_x_price)]), hashes


def hourly_activity_threshold(frame: pd.DataFrame, end_index: int) -> float:
    activity = (frame.loc[:end_index, "close"] * frame.loc[:end_index, "volume"]).set_axis(
        frame.loc[:end_index, "timestamp"]
    )
    hourly = activity.resample("1h", label="left", closed="left").sum()
    threshold = float(hourly.median())
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError(f"invalid dollar activity threshold: {threshold}")
    return threshold


def build_dollar_events(activity: np.ndarray, start_index: int, end_index: int, threshold: float) -> np.ndarray:
    events: list[int] = []
    accumulator = 0.0
    for index in range(start_index, end_index + 1):
        accumulator += max(float(activity[index]), 0.0)
        if accumulator >= threshold:
            events.append(index)
            accumulator = 0.0
    if len(events) < 2:
        return np.empty(0, dtype=np.int64)
    return np.asarray(events[1:], dtype=np.int64)


def lifecycle_log_return(
    side: int,
    entry_open: float,
    exit_open: float,
    entry_timestamp_ns: int,
    exit_timestamp_ns: int,
    funding: FundingTape,
    *,
    cost_multiplier: float = 1.0,
) -> float:
    stressed_slippage = SLIPPAGE_RATE * cost_multiplier
    stressed_fee = FEE_RATE * cost_multiplier
    entry_fill = entry_open * (1.0 + side * stressed_slippage)
    exit_fill = exit_open * (1.0 - side * stressed_slippage)
    price_ratio = exit_fill / entry_fill
    gross = side * (price_ratio - 1.0) * NOTIONAL
    entry_fee = stressed_fee * NOTIONAL
    exit_fee = stressed_fee * NOTIONAL * price_ratio
    left = int(np.searchsorted(funding.timestamp_ns, entry_timestamp_ns, side="right"))
    right = int(np.searchsorted(funding.timestamp_ns, exit_timestamp_ns, side="left"))
    funding_rate_x_price = funding.rate_x_price_cumsum[right] - funding.rate_x_price_cumsum[left]
    funding_return = -side * NOTIONAL * funding_rate_x_price / entry_fill
    account_return = gross - entry_fee - exit_fee + funding_return
    if not np.isfinite(account_return) or account_return <= -1.0:
        raise RuntimeError(f"invalid lifecycle account return: {account_return}")
    return float(np.log1p(account_return))


def first_risk_trigger(
    side: int,
    entry_index: int,
    scan_end: int,
    entry_fill: float,
    entry_atr: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[int, str] | None:
    atr_pct = entry_atr / entry_fill
    stop_price = entry_fill * (1.0 - side * STOP_ATR * atr_pct)
    arm = TRAIL_ARM_ATR * atr_pct
    giveback = TRAIL_GIVEBACK_ATR * atr_pct
    armed = False
    peak = -np.inf
    for index in range(entry_index, scan_end + 1):
        stop_hit = low[index] <= stop_price if side > 0 else high[index] >= stop_price
        favorable = side * (close[index] / entry_fill - 1.0)
        trailing_hit = armed and peak - favorable >= giveback
        if stop_hit:
            return index, "stop"
        if trailing_hit:
            return index, "trailing"
        if armed:
            peak = max(peak, favorable)
        elif favorable >= arm:
            armed = True
            peak = favorable
    return None


def simulate_action_table(
    frame: pd.DataFrame,
    events: np.ndarray,
    cutoff_index: int,
    funding: FundingTape,
    *,
    cost_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(events)
    action_count = len(ACTIONS)
    rewards = np.full((count, action_count), np.nan, dtype=np.float64)
    exits = np.full((count, action_count), -1, dtype=np.int64)
    next_event = np.full((count, action_count), -1, dtype=np.int64)
    reasons = np.full((count, action_count), "", dtype=object)
    rewards[:, 0] = 0.0
    next_event[:-1, 0] = np.arange(1, count, dtype=np.int64)

    open_ = frame["open"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    atr = frame["atr"].to_numpy(dtype=np.float64)
    timestamp_ns = frame["timestamp"].astype("int64").to_numpy()

    for position, event_index in enumerate(events):
        entry_index = int(event_index + 1)
        if entry_index >= len(frame) or not np.isfinite(atr[entry_index]):
            continue
        for side in (1, -1):
            entry_fill = open_[entry_index] * (1.0 + side * SLIPPAGE_RATE * cost_multiplier)
            scan_end = min(entry_index + HORIZON_BARS[-1] - 1, cutoff_index - 1)
            risk = first_risk_trigger(side, entry_index, scan_end, entry_fill, atr[entry_index], high, low, close)
            risk_index = risk[0] if risk is not None else sys.maxsize
            risk_reason = risk[1] if risk is not None else ""
            for horizon_index, horizon_bars in enumerate(HORIZON_BARS):
                action_index = 1 + 2 * horizon_index + (0 if side > 0 else 1)
                time_trigger = entry_index + horizon_bars - 1
                trigger_index = min(risk_index, time_trigger)
                exit_index = trigger_index + 1
                if exit_index > cutoff_index or exit_index >= len(frame):
                    continue
                reason = risk_reason if risk_index <= time_trigger else "time"
                reward = lifecycle_log_return(
                    side,
                    open_[entry_index],
                    open_[exit_index],
                    int(timestamp_ns[entry_index]),
                    int(timestamp_ns[exit_index]),
                    funding,
                    cost_multiplier=cost_multiplier,
                )
                rewards[position, action_index] = reward
                exits[position, action_index] = exit_index
                reasons[position, action_index] = reason
                eligible_bar = exit_index + COOLDOWN_BARS
                successor = int(np.searchsorted(events, eligible_bar, side="left"))
                if successor < count:
                    next_event[position, action_index] = successor
    return rewards, exits, next_event, reasons


def immediate_q(rewards: np.ndarray) -> np.ndarray:
    q = rewards.copy()
    q[:, 0] = 0.0
    return q


def finite_smdp_q(events: np.ndarray, rewards: np.ndarray, successors: np.ndarray) -> np.ndarray:
    """Return Q(s, a, B) while carrying the original remaining bar budget."""
    count, action_count = rewards.shape
    output = np.full_like(rewards, np.nan)
    max_duration = np.asarray([action.contract_max_duration for action in ACTIONS], dtype=np.int64)
    for start in range(count):
        boundary = int(events[start] + PLANNING_BARS)
        stop = int(np.searchsorted(events, boundary, side="right"))
        values = np.zeros(stop - start + 1, dtype=np.float64)
        start_q = np.full(action_count, np.nan, dtype=np.float64)
        for position in range(stop - 1, start - 1, -1):
            remaining = boundary - int(events[position])
            candidates = np.full(action_count, -np.inf, dtype=np.float64)
            cash_successor = position + 1
            candidates[0] = values[cash_successor - start] if cash_successor < stop else 0.0
            for action_index in range(1, action_count):
                if max_duration[action_index] > remaining or not np.isfinite(rewards[position, action_index]):
                    continue
                successor = int(successors[position, action_index])
                continuation = values[successor - start] if position < successor < stop else 0.0
                candidates[action_index] = rewards[position, action_index] + continuation
            values[position - start] = float(np.max(candidates))
            if position == start:
                start_q = candidates
        output[start] = start_q
    return output


def replay_labels(
    labels: np.ndarray,
    rewards: np.ndarray,
    successors: np.ndarray,
    exits: np.ndarray,
    events: np.ndarray,
    decision_count: int,
    frame: pd.DataFrame,
    funding: FundingTape,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    equity = 1.0
    position = 0
    close = frame["close"].to_numpy(dtype=np.float64)
    open_ = frame["open"].to_numpy(dtype=np.float64)
    timestamp_ns = frame["timestamp"].astype("int64").to_numpy()
    equity_curve = np.ones(len(frame), dtype=np.float64)
    curve_cursor = int(events[0])
    while position < decision_count:
        action = int(labels[position])
        if action == 0:
            position += 1
            continue
        reward = float(rewards[position, action])
        successor = int(successors[position, action])
        exit_index = int(exits[position, action])
        if not np.isfinite(reward) or exit_index < 0:
            raise RuntimeError("replay selected an unavailable action")
        event_index = int(events[position])
        entry_index = event_index + 1
        equity_curve[curve_cursor:entry_index] = equity
        side = ACTIONS[action].side
        entry_fill = open_[entry_index] * (1.0 + side * SLIPPAGE_RATE)
        mark_indices = np.arange(entry_index, exit_index, dtype=np.int64)
        if len(mark_indices):
            mark_timestamp_ns = timestamp_ns[mark_indices] + pd.Timedelta(minutes=5).value
            funding_left = int(np.searchsorted(funding.timestamp_ns, timestamp_ns[entry_index], side="right"))
            funding_right = np.searchsorted(funding.timestamp_ns, mark_timestamp_ns, side="right")
            funding_rate_x_price = funding.rate_x_price_cumsum[funding_right] - funding.rate_x_price_cumsum[funding_left]
            mark_ratio = close[mark_indices] / entry_fill
            mark_return = (
                side * (mark_ratio - 1.0) * NOTIONAL
                - FEE_RATE * NOTIONAL
                - side * NOTIONAL * funding_rate_x_price / entry_fill
            )
            equity_curve[mark_indices] = equity * (1.0 + mark_return)
        equity *= math.exp(reward)
        equity_curve[exit_index] = equity
        curve_cursor = exit_index + 1
        rows.append(
            {
                "event_position": position,
                "action": ACTIONS[action].name,
                "entry_timestamp": frame["timestamp"].iloc[entry_index],
                "exit_timestamp": frame["timestamp"].iloc[exit_index],
                "exit_bar_index": exit_index,
                "lifecycle_log_return": reward,
                "equity": equity,
            }
        )
        if successor <= position:
            raise RuntimeError("active transition did not advance")
        position = successor if successor >= 0 else decision_count
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "mtm_mdd": 0.0, "calmar": 0.0, "trades": 0}, ledger
    closed_curve = np.r_[1.0, ledger["equity"].to_numpy(dtype=np.float64)]
    closed_peaks = np.maximum.accumulate(closed_curve)
    mdd = float(np.min(closed_curve / closed_peaks - 1.0))
    curve_end = int(ledger["exit_bar_index"].iloc[-1])
    mtm_curve = equity_curve[int(events[0]) : curve_end + 1]
    mtm_peaks = np.maximum.accumulate(mtm_curve)
    mtm_mdd = float(np.min(mtm_curve / mtm_peaks - 1.0))
    start_time = frame["timestamp"].iloc[int(events[0])]
    end_time = ledger["exit_timestamp"].iloc[-1]
    years = max((end_time - start_time).total_seconds() / (365.25 * 86400.0), 1.0 / 365.25)
    annual_return = float(equity ** (1.0 / years) - 1.0)
    calmar = annual_return / abs(mtm_mdd) if mtm_mdd < 0.0 else (math.inf if annual_return > 0.0 else 0.0)
    return {
        "pnl": equity - 1.0,
        "mdd": mdd,
        "mtm_mdd": mtm_mdd,
        "calmar": calmar,
        "trades": len(ledger),
    }, ledger


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {ACTIONS[index].name: int(np.sum(labels == index)) for index in range(len(ACTIONS))}


def plot_fold(
    path: Path,
    family: str,
    frame: pd.DataFrame,
    events: np.ndarray,
    decision_count: int,
    labels: np.ndarray,
    q_values: np.ndarray,
    rewards: np.ndarray,
    ledger: pd.DataFrame,
) -> None:
    decision_events = events[:decision_count]
    cutoff_time = frame["timestamp"].iloc[decision_events[-1]]
    view_start = cutoff_time - pd.Timedelta(days=21)
    view_mask = frame["timestamp"].between(view_start, cutoff_time)
    event_mask = frame["timestamp"].iloc[decision_events].ge(view_start).to_numpy()
    shown_events = decision_events[event_mask]
    shown_labels = labels[:decision_count][event_mask]
    active_values = np.where(np.isfinite(q_values[:decision_count, 1:]), q_values[:decision_count, 1:], -np.inf)
    best_active = np.max(active_values, axis=1)
    best_active[~np.isfinite(best_active)] = np.nan
    edge = best_active - q_values[:decision_count, 0]

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=False, constrained_layout=True)
    axes[0].plot(frame.loc[view_mask, "timestamp"], frame.loc[view_mask, "close"], color="black", linewidth=0.8)
    colors = {0: "gray", 1: "green", 2: "red"}
    for direction in (0, 1, 2):
        mask = np.asarray([0 if value == 0 else (1 if ACTIONS[int(value)].side > 0 else 2) for value in shown_labels]) == direction
        axes[0].scatter(
            frame["timestamp"].iloc[shown_events[mask]],
            frame["close"].iloc[shown_events[mask]],
            s=10,
            color=colors[direction],
            alpha=0.75,
            label=("CASH", "LONG", "SHORT")[direction],
        )
    axes[0].set_title(f"{family}: BTC close and labels (last 21 decision days)")
    axes[0].legend(loc="upper left", ncol=3)

    event_times = frame["timestamp"].iloc[decision_events]
    axes[1].plot(event_times, edge, color="#3b82f6", linewidth=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_title("Best active action value minus CASH")

    selected_reward = np.array(
        [0.0 if label == 0 else rewards[index, label] for index, label in enumerate(labels[:decision_count])],
        dtype=np.float64,
    )
    axes[2].scatter(event_times, selected_reward, c=np.sign(selected_reward), cmap="coolwarm", s=7, alpha=0.65)
    axes[2].axhline(0.0, color="black", linewidth=0.6)
    axes[2].set_title("Selected action immediate lifecycle log return")

    equity_times = [event_times.iloc[0]]
    equity_values = [1.0]
    if not ledger.empty:
        equity_times.extend(ledger["exit_timestamp"].tolist())
        equity_values.extend(ledger["equity"].tolist())
    equity_times.append(event_times.iloc[-1])
    equity_values.append(equity_values[-1])
    axes[3].step(equity_times, equity_values, where="post", color="#7c3aed", linewidth=1.0)
    axes[3].axhline(1.0, color="black", linewidth=0.6)
    axes[3].set_xlim(event_times.iloc[0], event_times.iloc[-1])
    axes[3].set_title("Sequential single-capacity equity")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_fold(
    fold: FoldSpec,
    frame: pd.DataFrame,
    funding: FundingTape,
    out_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    start_index = int(frame["timestamp"].searchsorted(pd.Timestamp("2024-01-01 00:00:00")))
    decision_end_index = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    cutoff_index = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = hourly_activity_threshold(frame, decision_end_index)
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    events = build_dollar_events(activity, start_index, cutoff_index, threshold)
    decision_count = int(np.searchsorted(events, decision_end_index, side="right"))
    if decision_count == 0:
        raise RuntimeError(f"{fold.name}: no decision events")

    rewards, exits, successors, reasons = simulate_action_table(frame, events, cutoff_index, funding)
    coverage = {ACTIONS[index].name: int(np.isfinite(rewards[:decision_count, index]).sum()) for index in range(1, len(ACTIONS))}
    coverage_pass = all(value >= 300 for value in coverage.values())
    n0_q = immediate_q(rewards)
    csalt_q = finite_smdp_q(events, rewards, successors)
    n0_labels = np.nanargmax(n0_q[:decision_count], axis=1).astype(np.int8)
    csalt_labels = np.nanargmax(csalt_q[:decision_count], axis=1).astype(np.int8)
    n0_metrics, n0_ledger = replay_labels(
        n0_labels, rewards, successors, exits, events, decision_count, frame, funding
    )
    csalt_metrics, csalt_ledger = replay_labels(
        csalt_labels, rewards, successors, exits, events, decision_count, frame, funding
    )

    target = pd.DataFrame(
        {
            "event_bar_timestamp": frame["timestamp"].iloc[events[:decision_count]].to_numpy(),
            "entry_available_timestamp": frame["timestamp"].iloc[events[:decision_count] + 1].to_numpy(),
            "n0_action": [ACTIONS[index].name for index in n0_labels],
            "csalt_action": [ACTIONS[index].name for index in csalt_labels],
        }
    )
    for action_index, action in enumerate(ACTIONS):
        target[f"reward_{action.name}"] = rewards[:decision_count, action_index]
        target[f"n0_q_{action.name}"] = n0_q[:decision_count, action_index]
        target[f"csalt_q_{action.name}"] = csalt_q[:decision_count, action_index]
    target.to_parquet(out_dir / "dp_targets" / f"fold_{fold.name}_targets.parquet", index=False)
    n0_ledger.to_csv(out_dir / f"{fold.name}_N0_ledger.csv", index=False)
    csalt_ledger.to_csv(out_dir / f"{fold.name}_CSALT_ledger.csv", index=False)
    plot_fold(
        out_dir / "label_charts" / f"N0_fold_{fold.name}.png",
        f"N0 {fold.name}", frame, events, decision_count, n0_labels, n0_q, rewards, n0_ledger,
    )
    plot_fold(
        out_dir / "label_charts" / f"N3_fold_{fold.name}.png",
        f"CSALT {fold.name}", frame, events, decision_count, csalt_labels, csalt_q, rewards, csalt_ledger,
    )

    report = {
        "fold": fold.name,
        "decision_end": fold.decision_end,
        "outcome_cutoff": fold.outcome_cutoff,
        "dollar_threshold": threshold,
        "events_total": len(events),
        "decision_events": decision_count,
        "action_coverage": coverage,
        "coverage_pass": coverage_pass,
        "n0_label_counts": _label_counts(n0_labels),
        "csalt_label_counts": _label_counts(csalt_labels),
        "label_change_rate": float(np.mean(n0_labels != csalt_labels)),
        "n0_metrics": n0_metrics,
        "csalt_metrics": csalt_metrics,
        "exit_reason_counts": {
            reason: int(np.sum(reasons[:decision_count] == reason)) for reason in ("stop", "trailing", "time")
        },
    }
    return report, target


def build_contact_sheet(out_dir: Path) -> None:
    paths = [out_dir / "label_charts" / f"{family}_fold_{fold.name}.png" for fold in FOLDS for family in ("N0", "N3")]
    images = [plt.imread(path) for path in paths]
    fig, axes = plt.subplots(len(FOLDS), 2, figsize=(18, 5 * len(FOLDS)), constrained_layout=True)
    for axis, image, path in zip(axes.ravel(), images, paths):
        axis.imshow(image)
        axis.set_title(path.stem)
        axis.axis("off")
    fig.savefig(out_dir / "label_charts" / "stage01_contact_sheet.png", dpi=100)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    (out_dir / "dp_targets").mkdir(parents=True, exist_ok=True)
    (out_dir / "label_charts").mkdir(parents=True, exist_ok=True)
    started = time.time()

    frame, market_hashes = load_market_tape()
    funding, funding_hashes = load_funding_tape(frame)
    fold_reports = []
    for fold in FOLDS:
        print(f"[{fold.name}] running Stage 0/1", flush=True)
        fold_report, _ = run_fold(fold, frame, funding, out_dir)
        fold_reports.append(fold_report)
        print(
            f"[{fold.name}] events={fold_report['decision_events']} "
            f"N0 Calmar={fold_report['n0_metrics']['calmar']:.4f} "
            f"CSALT Calmar={fold_report['csalt_metrics']['calmar']:.4f}",
            flush=True,
        )
    build_contact_sheet(out_dir)

    stage0_pass = all(report["coverage_pass"] for report in fold_reports)
    n0_calmar = float(np.mean([report["n0_metrics"]["calmar"] for report in fold_reports]))
    csalt_calmar = float(np.mean([report["csalt_metrics"]["calmar"] for report in fold_reports]))
    continuation_gate_pass = bool(np.isfinite(csalt_calmar) and csalt_calmar >= 1.25 * n0_calmar)
    report = {
        "status": "stage01_complete_non_promotion_research",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "holdout_data_read_before_scheduled_evaluation": False,
        "stage0_coverage_pass": stage0_pass,
        "stage1_n0_continuation_gate_pass": continuation_gate_pass,
        "stage1_causal_supervised_baseline_status": "pending_only_if_n0_continuation_gate_passes",
        "mean_n0_calmar": n0_calmar,
        "mean_csalt_calmar": csalt_calmar,
        "required_csalt_calmar_vs_n0": 1.25 * n0_calmar,
        "folds": fold_reports,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    environment = {
        "market_files_sha256": market_hashes,
        "funding_files_sha256": funding_hashes,
        "execution": {
            "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "notional": NOTIONAL,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "cooldown_bars": COOLDOWN_BARS,
            "planning_bars": PLANNING_BARS,
        },
        "folds": [asdict(fold) for fold in FOLDS],
    }
    (out_dir / "environment_contract.json").write_text(
        json.dumps(environment, indent=2, default=_json_default), encoding="utf-8"
    )
    manifest = {
        str(path.relative_to(out_dir)): _sha256(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.sha256.json"
    }
    (out_dir / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "stage0_coverage_pass", "stage1_n0_continuation_gate_pass", "mean_n0_calmar", "mean_csalt_calmar"
    )}, indent=2), flush=True)
    return 0 if stage0_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
