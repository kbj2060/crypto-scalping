"""Evaluate reusable execution, lifecycle, risk, and allocation layers from ETH v4.

The experiment rebuilds causal one-minute streams and never consumes saved trade
ledgers or parent exit timestamps.  The embedded cash-state Q row is treated as
the frozen parent direction intent because aligned live Omega per-bar predictions
do not exist for the short BTC/SOL microstructure window.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402
import tune_btc_sol_micro_scalp_transfer_adapters_20260718 as tuner  # noqa: E402


transfer = tuner.transfer
REPORT_PATH = ROOT / "data/ensemble/reports/micro_scalp_reuse_layers_test_20260718.json"
FEE_SCENARIOS_BP = (2.0, 4.5, 5.5, 9.0)
DISPLAY_FEE_BP = 4.5
MIN_TUNE_ENTRIES = 2
ASSET_CONFIG = {
    "eth": {
        "symbol": "ETHUSDT",
        "micro_table": "microstructure_1m",
        "book_table": "orderbook_decision_snapshots",
    },
    **transfer.ASSETS,
}
ADAPTER_PATHS = {
    asset: ROOT / f"data/ensemble/{tuner.MODEL_IDS[asset]}/adapter.json"
    for asset in ("btc", "sol")
}
POSITION_INDEX = {-1: 0, 0: 1, 1: 2}
ACTIONS = np.asarray((-1, 0, 1), dtype=np.int8)


@dataclass(frozen=True)
class LifecyclePolicy:
    entry_margin_bp: float
    min_entry_agreement: int
    exit_floor_bp: float
    min_exit_agreement: int
    uncertainty_penalty: float
    risk_veto: bool


def _utc_naive(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _slice_prediction(
    prediction: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[mask] for name, values in prediction.items()}


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = float(np.median(np.abs(finite - center)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(finite))
    return center, scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def _gate_entropy(gate: np.ndarray, experts: int) -> np.ndarray:
    values = np.asarray(gate, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % experts:
        raise RuntimeError("gate output does not match the expert contract")
    reshaped = values.reshape(len(values), -1, experts)
    entropy = -np.sum(reshaped * np.log(np.clip(reshaped, 1e-12, 1.0)), axis=2)
    return np.mean(entropy / np.log(experts), axis=1)


def prepare_asset(
    asset: str,
    frame: pd.DataFrame,
    runtime: Any,
    adapter: dict[str, Any] | None,
    *,
    require_next_return: bool = True,
) -> dict[str, Any]:
    model_frame = tuner.apply_adapter(frame, adapter["adapter"], runtime) if adapter else frame
    prediction, end_indices = binding.observer.infer_stream(model_frame, runtime)
    timestamps = pd.DatetimeIndex(pd.to_datetime(model_frame["timestamp"].iloc[end_indices]))
    closes = model_frame["close"].to_numpy(dtype=np.float64)
    next_indices = end_indices + 1
    has_following = next_indices < len(model_frame)
    cadence = np.zeros(len(end_indices), dtype=bool)
    cadence[has_following] = (
        model_frame["timestamp"].to_numpy()[next_indices[has_following]]
        - model_frame["timestamp"].to_numpy()[end_indices[has_following]]
    ) == np.timedelta64(1, "m")
    valid = has_following & cadence if require_next_return else np.ones(len(end_indices), dtype=bool)
    end_indices = end_indices[valid]
    next_indices = next_indices[valid]
    prediction = _slice_prediction(prediction, valid)
    timestamps = timestamps[valid]
    available = binding.observer._available(model_frame)[end_indices]
    liquidity_healthy = available & (
        model_frame["book_available"].to_numpy(dtype=np.float64)[end_indices] > 0.5
    )
    returns = np.full(len(end_indices), np.nan, dtype=np.float64)
    following = next_indices < len(model_frame)
    returns[following] = closes[next_indices[following]] / closes[end_indices[following]] - 1.0
    feature_names = [*v4.SOURCE_STABLE_FEATURES, *v4.v3.core.MICRO_FEATURES]
    feature_values = model_frame[feature_names].to_numpy(dtype=np.float64)[end_indices]
    feature_hash = np.asarray(
        [hashlib.sha256(row.tobytes()).hexdigest() for row in feature_values]
    )

    cash_q = np.asarray(prediction["q"], dtype=np.float64)[:, 1, :]
    expert_cash_q = np.asarray(prediction["expert_q"], dtype=np.float64)[:, :, 1, :]
    action_index = np.argmax(cash_q, axis=1)
    desired = ACTIONS[action_index]
    desired = np.where(available, desired, 0).astype(np.int8)
    rows = np.arange(len(desired))
    edge_bp = cash_q[rows, action_index] - cash_q[:, 1]
    expert_action = np.argmax(expert_cash_q, axis=2)
    agreement = np.sum(expert_action == action_index[:, None], axis=1)
    chosen_values = expert_cash_q[rows, :, action_index]
    uncertainty = np.std(chosen_values, axis=1)
    entropy = _gate_entropy(prediction["gate"], runtime.config.experts)
    return {
        "asset": asset,
        "timestamps": timestamps,
        "returns": returns,
        "available": available,
        "liquidity_healthy": liquidity_healthy,
        "prediction": prediction,
        "desired": desired,
        "edge_bp": edge_bp,
        "agreement": agreement,
        "uncertainty": uncertainty,
        "gate_entropy": entropy,
        "close": closes[end_indices],
        "feature_hash": feature_hash,
        "source_rows": len(frame),
    }


def fit_risk_detector(data: dict[str, Any], calibration_mask: np.ndarray) -> dict[str, float]:
    uncertainty_center, uncertainty_scale = _robust_scale(data["uncertainty"][calibration_mask])
    disagreement = 1.0 - data["agreement"] / data["prediction"]["expert_q"].shape[1]
    normalized_uncertainty = np.clip(
        (data["uncertainty"] - uncertainty_center) / uncertainty_scale, 0.0, 4.0
    ) / 4.0
    score = (
        0.50 * disagreement
        + 0.30 * normalized_uncertainty
        + 0.20 * data["gate_entropy"]
    )
    eligible = calibration_mask & data["liquidity_healthy"] & np.isfinite(score)
    threshold = float(np.quantile(score[eligible], 0.75))
    return {
        "uncertainty_center": uncertainty_center,
        "uncertainty_scale": uncertainty_scale,
        "risk_threshold": threshold,
    }


def apply_risk_detector(data: dict[str, Any], detector: dict[str, float]) -> None:
    expert_count = data["prediction"]["expert_q"].shape[1]
    disagreement = 1.0 - data["agreement"] / expert_count
    uncertainty = np.clip(
        (data["uncertainty"] - detector["uncertainty_center"])
        / detector["uncertainty_scale"],
        0.0,
        4.0,
    ) / 4.0
    score = 0.50 * disagreement + 0.30 * uncertainty + 0.20 * data["gate_entropy"]
    data["risk_score"] = score
    data["high_risk"] = (~data["liquidity_healthy"]) | (
        score >= detector["risk_threshold"]
    )


def lifecycle_positions(
    data: dict[str, Any],
    policy: LifecyclePolicy,
    *,
    dynamic_exit: bool,
    initial_position: int = 0,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    prediction = data["prediction"]
    q_values = np.asarray(prediction["q"], dtype=np.float64)
    expert_q = np.asarray(prediction["expert_q"], dtype=np.float64)
    continuation = np.asarray(prediction["continuation"], dtype=np.float64)
    expert_continuation = np.asarray(
        prediction["expert_continuation"], dtype=np.float64
    )
    positions = np.zeros(len(data["timestamps"]), dtype=np.int8)
    priority = np.full(len(positions), -np.inf, dtype=np.float64)
    if initial_position not in POSITION_INDEX:
        raise ValueError(f"invalid initial_position: {initial_position}")
    current = int(initial_position)
    counters = {
        "entry_blocks": 0,
        "risk_veto_bars": 0,
        "early_exit_triggers": 0,
        "extended_parent_cash_bars": 0,
    }
    for index in range(len(positions)):
        desired = int(data["desired"][index])
        if not data["available"][index] or (
            policy.risk_veto and data["high_risk"][index]
        ):
            if policy.risk_veto and data["high_risk"][index]:
                counters["risk_veto_bars"] += 1
            current = 0
            positions[index] = 0
            continue
        previous_index = POSITION_INDEX[current]
        state_expert_q = expert_q[index, :, previous_index, :]
        state_q = q_values[index, previous_index] - policy.uncertainty_penalty * np.std(
            state_expert_q, axis=0
        )
        continuation_value = float(continuation[index, previous_index])
        exit_votes = int(
            np.sum(
                expert_continuation[index, :, previous_index]
                < policy.exit_floor_bp
            )
        )
        if current == 0:
            desired_index = POSITION_INDEX[desired]
            improvement = float(state_q[desired_index] - state_q[1])
            votes = int(np.sum(np.argmax(state_expert_q, axis=1) == desired_index))
            if (
                desired != 0
                and int(np.argmax(state_q)) == desired_index
                and improvement >= policy.entry_margin_bp
                and votes >= policy.min_entry_agreement
            ):
                current = desired
                priority[index] = improvement
            elif desired != 0:
                counters["entry_blocks"] += 1
        elif (
            dynamic_exit
            and continuation_value < policy.exit_floor_bp
            and exit_votes >= policy.min_exit_agreement
        ):
            current = 0
            counters["early_exit_triggers"] += 1
        elif desired == current:
            priority[index] = max(float(data["edge_bp"][index]), continuation_value)
        elif desired == 0:
            if dynamic_exit:
                priority[index] = continuation_value
                counters["extended_parent_cash_bars"] += 1
            else:
                current = 0
        else:
            desired_index = POSITION_INDEX[desired]
            improvement = float(state_q[desired_index] - state_q[previous_index])
            votes = int(np.sum(np.argmax(state_expert_q, axis=1) == desired_index))
            if (
                int(np.argmax(state_q)) == desired_index
                and improvement >= policy.entry_margin_bp
                and votes >= policy.min_entry_agreement
            ):
                current = desired
                priority[index] = improvement
            elif dynamic_exit and continuation_value < 0.0:
                current = 0
            else:
                priority[index] = continuation_value
        positions[index] = current
        if current != 0 and not np.isfinite(priority[index]):
            priority[index] = continuation_value
    return positions, counters, priority


def replay(
    positions: np.ndarray,
    data: dict[str, Any],
    fee_bp: float,
) -> dict[str, Any]:
    metrics, _ = v4.v3.core.replay_positions(
        positions,
        data["returns"],
        data["timestamps"],
        fee_bp / 10_000.0,
    )
    return metrics


def select_policy(
    data: dict[str, Any],
    calibration_mask: np.ndarray,
    tune_mask: np.ndarray,
) -> tuple[LifecyclePolicy, list[dict[str, Any]]]:
    continuation = data["prediction"]["continuation"][calibration_mask][:, (0, 2)]
    finite = continuation[np.isfinite(continuation)]
    floors = sorted(
        {
            0.0,
            *(round(float(value), 6) for value in np.quantile(finite, (0.25, 0.5, 0.75))),
        }
    )
    expert_count = data["prediction"]["expert_q"].shape[1]
    agreements = sorted({max(2, expert_count // 2), max(2, expert_count - 2), expert_count})
    candidates: list[dict[str, Any]] = []
    tune_data = slice_data(data, tune_mask)
    for values in itertools.product(
        (0.0, 0.5, 1.0, 2.0), agreements, floors, agreements, (0.0, 1.0), (False, True)
    ):
        policy = LifecyclePolicy(*values)
        positions, counters, _ = lifecycle_positions(tune_data, policy, dynamic_exit=True)
        metrics = replay(positions, tune_data, DISPLAY_FEE_BP)
        net = metrics["compounded_return_pct"] / 100.0
        drawdown = metrics["max_drawdown_pct"] / 100.0
        eligible = metrics["entries_or_reversals"] >= MIN_TUNE_ENTRIES and net > 0.0
        candidates.append(
            {
                "policy": asdict(policy),
                "eligible": bool(eligible),
                "selection_score": net - 0.25 * drawdown if eligible else None,
                "metrics": metrics,
                "counters": counters,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["selection_score"] is not None,
            row["selection_score"] if row["selection_score"] is not None else -1e9,
        ),
        reverse=True,
    )
    if candidates and candidates[0]["eligible"]:
        return LifecyclePolicy(**candidates[0]["policy"]), candidates
    cash = LifecyclePolicy(1e9, expert_count, 0.0, expert_count, 1.0, True)
    return cash, candidates


def slice_data(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = {"asset": data["asset"], "source_rows": data["source_rows"]}
    for name in (
        "timestamps", "returns", "available", "liquidity_healthy", "desired", "edge_bp", "agreement",
        "uncertainty", "gate_entropy", "risk_score", "high_risk", "close", "feature_hash",
    ):
        result[name] = np.asarray(data[name])[mask]
    result["timestamps"] = pd.DatetimeIndex(result["timestamps"])
    result["prediction"] = _slice_prediction(data["prediction"], mask)
    return result


def risk_diagnostics(data: dict[str, Any]) -> dict[str, Any]:
    active = data["desired"] != 0
    high = active & data["high_risk"]
    low = active & ~data["high_risk"]

    def group(mask: np.ndarray) -> dict[str, Any]:
        signed = data["desired"][mask] * data["returns"][mask] * 10_000.0
        absolute = np.abs(data["returns"][mask]) * 10_000.0
        return {
            "bars": int(np.sum(mask)),
            "mean_directional_next_return_bp": float(np.mean(signed)) if len(signed) else None,
            "adverse_beyond_4_5bp_fraction": float(np.mean(signed < -4.5)) if len(signed) else None,
            "mean_absolute_next_move_bp": float(np.mean(absolute)) if len(absolute) else None,
        }

    high_metrics = group(high)
    low_metrics = group(low)
    return {
        "high_risk_fraction": float(np.mean(data["high_risk"])),
        "high_risk": high_metrics,
        "normal_risk": low_metrics,
        "adverse_rate_lift": (
            high_metrics["adverse_beyond_4_5bp_fraction"]
            / low_metrics["adverse_beyond_4_5bp_fraction"]
            if high_metrics["adverse_beyond_4_5bp_fraction"] is not None
            and low_metrics["adverse_beyond_4_5bp_fraction"] not in (None, 0.0)
            else None
        ),
    }


def evaluate_asset_split(
    data: dict[str, Any], policy: LifecyclePolicy
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    baseline = data["desired"].astype(np.int8)
    entry_only, entry_counters, _ = lifecycle_positions(
        data, policy, dynamic_exit=False
    )
    lifecycle, lifecycle_counters, priority = lifecycle_positions(
        data, policy, dynamic_exit=True
    )
    risk_veto_policy = replace(policy, risk_veto=True)
    risk_veto_lifecycle, risk_veto_counters, _ = lifecycle_positions(
        data, risk_veto_policy, dynamic_exit=True
    )
    return (
        {
            "bars": len(data["timestamps"]),
            "start_utc": str(data["timestamps"].min()) if len(data["timestamps"]) else None,
            "end_utc": str(data["timestamps"].max()) if len(data["timestamps"]) else None,
            "parent_immediate": replay(baseline, data, DISPLAY_FEE_BP),
            "entry_execution_only": {
                "metrics": replay(entry_only, data, DISPLAY_FEE_BP),
                "counters": entry_counters,
            },
            "full_lifecycle": {
                "metrics": replay(lifecycle, data, DISPLAY_FEE_BP),
                "counters": lifecycle_counters,
                "cost_stress": {
                    f"{fee:.2f}bp_per_notional_change": replay(lifecycle, data, fee)
                    for fee in FEE_SCENARIOS_BP
                },
            },
            "full_lifecycle_with_risk_veto": {
                "policy": asdict(risk_veto_policy),
                "metrics": replay(risk_veto_lifecycle, data, DISPLAY_FEE_BP),
                "counters": risk_veto_counters,
            },
            "risk_detector": risk_diagnostics(data),
        },
        lifecycle,
        priority,
    )


def replay_portfolio(
    weights: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fee_bp: float,
) -> dict[str, Any]:
    previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    turnover = np.sum(np.abs(weights - previous), axis=1)
    if len(weights):
        turnover[-1] += np.sum(np.abs(weights[-1]))
    gross = np.sum(weights * returns, axis=1)
    cost = fee_bp / 10_000.0 * turnover
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    curve = np.r_[1.0, equity]
    peak = np.maximum.accumulate(curve)
    return {
        "bars": len(weights),
        "compounded_return_pct": float((equity[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "additive_cost_pct": float(cost.sum() * 100.0),
        "max_drawdown_pct": float((1.0 - curve / peak).max() * 100.0),
        "turnover": float(turnover.sum()),
        "exposure_fraction": float(np.mean(np.sum(np.abs(weights), axis=1) > 0.0)),
        "average_gross_notional": float(np.mean(np.sum(np.abs(weights), axis=1))),
        "positive_day_fraction": _positive_day_fraction(timestamps, net),
    }


def _positive_day_fraction(timestamps: pd.DatetimeIndex, net: np.ndarray) -> float:
    frame = pd.DataFrame({"timestamp": timestamps, "net": net})
    daily = frame.groupby(frame["timestamp"].dt.date)["net"].apply(
        lambda values: np.prod(1.0 + values) - 1.0
    )
    return float(np.mean(daily > 0.0)) if len(daily) else 0.0


def allocator_weights(
    candidates: np.ndarray,
    priorities: np.ndarray,
    switch_margin: float,
) -> np.ndarray:
    weights = np.zeros_like(candidates, dtype=np.float64)
    current_asset = -1
    for row in range(len(candidates)):
        eligible = np.flatnonzero(candidates[row] != 0)
        if not len(eligible):
            current_asset = -1
            continue
        best = int(eligible[np.argmax(priorities[row, eligible])])
        if current_asset in eligible:
            if priorities[row, best] - priorities[row, current_asset] <= switch_margin:
                best = current_asset
        weights[row, best] = candidates[row, best]
        current_asset = best
    return weights


def evaluate_portfolio(
    split_assets: dict[str, dict[str, Any]],
    lifecycle: dict[str, np.ndarray],
    priorities: dict[str, np.ndarray],
    switch_margin: float,
) -> dict[str, Any]:
    assets = list(ASSET_CONFIG)
    timestamps = split_assets[assets[0]]["timestamps"]
    for asset in assets[1:]:
        if not timestamps.equals(split_assets[asset]["timestamps"]):
            raise RuntimeError("portfolio asset timestamps are not exactly aligned")
    positions = np.column_stack([lifecycle[asset] for asset in assets])
    priority = np.column_stack([priorities[asset] for asset in assets])
    returns = np.column_stack([split_assets[asset]["returns"] for asset in assets])
    active = np.sum(np.abs(positions), axis=1, keepdims=True)
    concurrent = np.divide(
        positions,
        active,
        out=np.zeros_like(positions, dtype=np.float64),
        where=active > 0,
    )
    allocator = allocator_weights(positions, priority, switch_margin)
    selected_counts = {
        asset: int(np.sum(allocator[:, index] != 0))
        for index, asset in enumerate(assets)
    }
    return {
        "assets": assets,
        "allocator_switch_margin_z": switch_margin,
        "concurrent_unit_gross_baseline": replay_portfolio(
            concurrent, returns, timestamps, DISPLAY_FEE_BP
        ),
        "single_asset_allocator": replay_portfolio(
            allocator, returns, timestamps, DISPLAY_FEE_BP
        ),
        "selected_bars": selected_counts,
        "simultaneous_asset_limit": 1,
    }


def split_masks(timestamps: pd.DatetimeIndex, split_times: dict[str, pd.Timestamp]) -> dict[str, np.ndarray]:
    return {
        "calibration": (timestamps >= split_times["calibration_start"]) & (timestamps < split_times["tune_start"]),
        "tune": (timestamps >= split_times["tune_start"]) & (timestamps < split_times["validation_start"]),
        "validation": (timestamps >= split_times["validation_start"]) & (timestamps < split_times["development_start"]),
        "development": (timestamps >= split_times["development_start"]) & (timestamps <= split_times["development_end"]),
        "fresh_shadow": timestamps >= split_times["fresh_forward_start"],
    }


def run(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    runtime = binding.observer.load_runtime(device_name="cpu")
    adapters = {
        asset: json.loads(path.read_text()) for asset, path in ADAPTER_PATHS.items()
    }
    for asset, artifact in adapters.items():
        if artifact["asset"] != asset or artifact["parent_model_sha256"] != runtime.model_sha256:
            raise RuntimeError(f"{asset} adapter is not bound to the exact v4 parent")
    split_source = adapters["btc"]["split_times"]
    if adapters["sol"]["split_times"] != split_source:
        raise RuntimeError("BTC/SOL adapter split contracts differ")
    split_times = {name: _utc_naive(value) for name, value in split_source.items()}

    with tempfile.TemporaryDirectory(prefix="micro-scalp-reuse-") as directory:
        snapshot_path = Path(directory) / "microstructure.duckdb"
        snapshot = transfer.snapshot_database(transfer.MICRO_DB, snapshot_path)
        connection = duckdb.connect(str(snapshot_path), read_only=True)
        try:
            coverage = {
                asset: {
                    "micro": transfer.table_coverage(connection, config["micro_table"], "ts"),
                    "book": transfer.table_coverage(connection, config["book_table"], "recorded_at_kst"),
                }
                for asset, config in ASSET_CONFIG.items()
            }
        finally:
            connection.close()
        evaluation_end = min(
            values["micro"]["end_utc"] for values in coverage.values()
        ).floor("min")
        source_start = split_times["calibration_start"] - pd.Timedelta(
            hours=transfer.FEATURE_CONTEXT_HOURS + 1
        )
        prepared: dict[str, dict[str, Any]] = {}
        source_reports: dict[str, Any] = {}
        for asset, config in ASSET_CONFIG.items():
            print(f"building {asset.upper()} reuse stream", flush=True)
            frame, source = transfer.build_asset_stream(
                snapshot_path, asset, config, source_start, evaluation_end
            )
            prepared[asset] = prepare_asset(
                asset, frame, runtime, adapters.get(asset)
            )
            source_reports[asset] = source

    common_timestamps = prepared["eth"]["timestamps"]
    for asset in ("btc", "sol"):
        common_timestamps = common_timestamps.intersection(prepared[asset]["timestamps"])
    if len(common_timestamps) < 60:
        raise RuntimeError("common three-asset evaluation interval is too short")
    for asset, data in prepared.items():
        keep = data["timestamps"].isin(common_timestamps)
        prepared[asset] = slice_data_with_optional_risk(data, keep)
    common_timestamps = prepared["eth"]["timestamps"]
    masks = split_masks(common_timestamps, split_times)
    if any(np.sum(masks[name]) == 0 for name in ("calibration", "tune", "validation", "development")):
        raise RuntimeError("a required chronological evaluation split is empty")

    selected_policies: dict[str, LifecyclePolicy] = {}
    asset_results: dict[str, Any] = {}
    lifecycle_by_split: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in masks if name != "calibration"
    }
    priority_by_split: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in masks if name != "calibration"
    }
    split_asset_data: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in masks if name != "calibration"
    }
    for asset, data in prepared.items():
        detector = fit_risk_detector(data, masks["calibration"])
        apply_risk_detector(data, detector)
        policy, candidates = select_policy(data, masks["calibration"], masks["tune"])
        selected_policies[asset] = policy
        splits: dict[str, Any] = {}
        for name, mask in masks.items():
            if name == "calibration":
                continue
            split_data = slice_data(data, mask)
            result, positions, priority = evaluate_asset_split(split_data, policy)
            splits[name] = result
            split_asset_data[name][asset] = split_data
            lifecycle_by_split[name][asset] = positions
            priority_by_split[name][asset] = priority
        asset_results[asset] = {
            "selected_policy": asdict(policy),
            "policy_selected_on_tune_only": True,
            "risk_detector": detector,
            "splits": splits,
            "candidate_count": len(candidates),
            "top_tune_candidates": candidates[:10],
        }

    portfolio_tune_candidates: list[dict[str, Any]] = []
    for margin in (0.0, 0.5, 1.0, 2.0):
        result = evaluate_portfolio(
            split_asset_data["tune"],
            lifecycle_by_split["tune"],
            priority_by_split["tune"],
            margin,
        )
        metrics = result["single_asset_allocator"]
        score = metrics["compounded_return_pct"] - 0.25 * metrics["max_drawdown_pct"]
        portfolio_tune_candidates.append({"margin": margin, "score": score, "result": result})
    portfolio_tune_candidates.sort(key=lambda row: row["score"], reverse=True)
    selected_margin = float(portfolio_tune_candidates[0]["margin"])
    portfolio_results = {
        name: evaluate_portfolio(
            split_asset_data[name],
            lifecycle_by_split[name],
            priority_by_split[name],
            selected_margin,
        )
        for name in split_asset_data
    }

    report = {
        "schema_version": "micro_scalp.reuse_layers_test.v1",
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "parent_direction_source": "embedded causal cash-state Q intent",
        "omega_parent_predictions_used": False,
        "omega_parent_unavailable_reason": "no aligned fresh per-bar Omega prediction history for the BTC/SOL microstructure window",
        "applications": {
            "execution_assist": True,
            "dynamic_early_exit_and_extension": True,
            "multi_asset_allocator": True,
            "market_and_liquidity_risk_detector": True,
        },
        "split_times": {name: str(value) for name, value in split_times.items()},
        "common_interval": {
            "start_utc": str(common_timestamps.min()),
            "end_utc": str(common_timestamps.max()),
            "bars": len(common_timestamps),
        },
        "selection_uses_only_tune_split": True,
        "validation_used_for_selection": False,
        "development_used_for_selection": False,
        "fresh_shadow_used_for_selection": False,
        "assets": asset_results,
        "portfolio": {
            "selected_switch_margin_z": selected_margin,
            "selected_on_tune_only": True,
            "tune_candidates": portfolio_tune_candidates,
            "splits": portfolio_results,
        },
        "source": source_reports,
        "coverage": coverage,
        "microstructure_snapshot": snapshot,
        "fee_scenarios_bp": list(FEE_SCENARIOS_BP),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "fixed_holding_period_used": False,
        "activation_allowed": False,
        "order_submission_supported": False,
        "promotion_pass": False,
        "evidence_class": "short-window reusable-layer research diagnostic; shadow confirmation required",
    }
    transfer.base._write_json_atomic(report_path, report)
    return report


def slice_data_with_optional_risk(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    result = {"asset": data["asset"], "source_rows": data["source_rows"]}
    for name in (
        "timestamps", "returns", "available", "liquidity_healthy", "desired", "edge_bp", "agreement",
        "uncertainty", "gate_entropy", "close", "feature_hash",
    ):
        result[name] = np.asarray(data[name])[mask]
    result["timestamps"] = pd.DatetimeIndex(result["timestamps"])
    result["prediction"] = _slice_prediction(data["prediction"], mask)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.report), indent=2, default=transfer.base._json_default))
