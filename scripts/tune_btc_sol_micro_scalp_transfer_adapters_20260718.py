"""Tune frozen-ETH-v4 transfer adapters for BTC and SOL without weight training."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
from dataclasses import asdict
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


TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "eth_v4_cross_asset_transfer_runtime",
    SCRIPT_DIR / "test_eth_micro_scalp_v4_cross_asset_transfer_20260718.py",
)
if TRANSFER_SPEC is None or TRANSFER_SPEC.loader is None:
    raise RuntimeError("cannot load the cross-asset transfer runtime")
transfer = importlib.util.module_from_spec(TRANSFER_SPEC)
sys.modules[TRANSFER_SPEC.name] = transfer
TRANSFER_SPEC.loader.exec_module(transfer)


REPORT_PATH = (
    ROOT
    / "data/ensemble/reports/btc_sol_micro_scalp_transfer_adapters_v1_20260718.json"
)
ARTIFACT_ROOT = ROOT / "data/ensemble"
MODEL_IDS = {
    "btc": "btc_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
    "sol": "sol_micro_scalp_eth_v4_transfer_adapter_v1_20260718",
}
IDENTITY_MICRO_FEATURES = {
    "micro_data_stale",
    "micro_depth_connected",
    "micro_trade_connected",
    "micro_poll_connected",
    "micro_valid_taker_flow",
    "micro_valid_nif",
    "micro_warmup_30m_ready",
    "micro_available",
    "micro_age_min",
    "book_available",
    "book_age_min",
}
CALIBRATION_HOURS = 24
TUNE_HOURS = 24
VALIDATION_HOURS = 24
MIN_TUNE_ENTRIES = 2


def fit_robust_adapter(
    frame: pd.DataFrame,
    mask: np.ndarray,
    runtime: Any,
) -> dict[str, Any]:
    groups = {
        "base": (
            list(v4.SOURCE_STABLE_FEATURES),
            set(),
        ),
        "micro": (
            list(v4.v3.core.MICRO_FEATURES),
            IDENTITY_MICRO_FEATURES,
        ),
    }
    adapter: dict[str, Any] = {}
    for group, (names, identity_names) in groups.items():
        values = frame[names].to_numpy(dtype=np.float64)
        sample = values[mask]
        center = np.nanmedian(sample, axis=0)
        mad = np.nanmedian(np.abs(sample - center), axis=0) * 1.4826
        std = np.nanstd(sample, axis=0)
        scale = np.where((mad > 1e-8) & np.isfinite(mad), mad, std)
        center = np.where(np.isfinite(center), center, 0.0)
        scale = np.where((scale > 1e-8) & np.isfinite(scale), scale, 1.0)
        adapted = [name not in identity_names for name in names]
        adapter[group] = {
            "feature_names": names,
            "asset_center": center.tolist(),
            "asset_scale": scale.tolist(),
            "adapted": adapted,
        }
    return adapter


def apply_adapter(
    frame: pd.DataFrame,
    adapter: dict[str, Any],
    runtime: Any,
) -> pd.DataFrame:
    result = frame.copy()
    eth_scalers = runtime.checkpoint["scalers"]
    for group in ("base", "micro"):
        spec = adapter[group]
        names = spec["feature_names"]
        values = result[names].to_numpy(dtype=np.float64)
        asset_center = np.asarray(spec["asset_center"], dtype=np.float64)
        asset_scale = np.asarray(spec["asset_scale"], dtype=np.float64)
        eth_center = np.asarray(eth_scalers[f"{group}_center"], dtype=np.float64)
        eth_scale = np.asarray(eth_scalers[f"{group}_scale"], dtype=np.float64)
        adapted = np.asarray(spec["adapted"], dtype=bool)
        z = np.clip((values - asset_center) / asset_scale, -10.0, 10.0)
        transformed = eth_center + z * eth_scale
        values[:, adapted] = transformed[:, adapted]
        result[names] = values
    return result


def slice_prediction(
    prediction: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[mask] for name, values in prediction.items()}


def evaluate_policy(
    prediction: dict[str, np.ndarray],
    available: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    policy: Any,
    fee: float,
) -> dict[str, Any]:
    metrics, _ = v4.v3.replay_policy(
        prediction, available, returns, timestamps, policy, fee
    )
    return metrics


def policy_candidates(expert_count: int) -> list[Any]:
    margins = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0)
    agreements = sorted({2, 4, 6, 9, 12, expert_count})
    penalties = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
    return [
        v4.v3.OpportunityPolicy(
            True,
            margin,
            agreement,
            False,
            0.0,
            expert_count,
            penalty,
        )
        for margin, agreement, penalty in itertools.product(
            margins, agreements, penalties
        )
        if agreement <= expert_count
    ]


def select_tune_policy(
    prediction: dict[str, np.ndarray],
    available: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fee: float,
) -> tuple[Any, list[dict[str, Any]]]:
    expert_count = int(prediction["expert_q"].shape[1])
    rows: list[dict[str, Any]] = []
    for policy in policy_candidates(expert_count):
        metrics = evaluate_policy(
            prediction, available, returns, timestamps, policy, fee
        )
        net = metrics["compounded_return_pct"] / 100.0
        drawdown = metrics["max_drawdown_pct"] / 100.0
        eligible = bool(
            metrics["entries_or_reversals"] >= MIN_TUNE_ENTRIES and net > 0.0
        )
        rows.append(
            {
                "policy": asdict(policy),
                "eligible": eligible,
                "selection_score": net - 0.25 * drawdown if eligible else None,
                "metrics": metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            row["selection_score"] is not None,
            row["selection_score"] if row["selection_score"] is not None else -1e9,
        ),
        reverse=True,
    )
    if rows and rows[0]["eligible"]:
        return v4.v3.OpportunityPolicy(**rows[0]["policy"]), rows
    return v4.v3.OpportunityPolicy(
        False, 0.0, expert_count, False, 0.0, expert_count, 0.0
    ), rows


def cost_stress(
    prediction: dict[str, np.ndarray],
    available: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    policy: Any,
) -> dict[str, Any]:
    return {
        f"{fee:.2f}bp_per_notional_change": evaluate_policy(
            prediction,
            available,
            returns,
            timestamps,
            policy,
            fee / 10_000.0,
        )
        for fee in transfer.FEE_SCENARIOS_BP
    }


def artifact_payload(
    asset: str,
    runtime: Any,
    adapter: dict[str, Any],
    selected_policy: Any,
    split_times: dict[str, pd.Timestamp],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    expert_count = int(len(runtime.models) * runtime.config.experts)
    execution_policy = v4.v3.OpportunityPolicy(
        False, 0.0, expert_count, False, 0.0, expert_count, 0.0
    )
    return {
        "schema_version": "cross_asset_micro_scalp.transfer_adapter.v1",
        "model_id": MODEL_IDS[asset],
        "asset": asset,
        "symbol": transfer.ASSETS[asset]["symbol"],
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "parent_weights_frozen": True,
        "training_performed": False,
        "parameter_updates": 0,
        "adapter": adapter,
        "selected_research_policy": asdict(selected_policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": False,
        "order_submission_supported": False,
        "fixed_holding_period_used": False,
        "fresh_forward_start_utc": str(split_times["fresh_forward_start"]),
        "split_times": {name: str(value) for name, value in split_times.items()},
        "diagnostics": diagnostics,
    }


def run(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    runtime = binding.observer.load_runtime(device_name="cpu")
    with tempfile.TemporaryDirectory(prefix="btc-sol-scalp-tune-") as directory:
        snapshot_path = Path(directory) / "microstructure.duckdb"
        snapshot = transfer.snapshot_database(transfer.MICRO_DB, snapshot_path)
        connection = duckdb.connect(str(snapshot_path), read_only=True)
        try:
            coverage = {
                asset: {
                    "micro": transfer.table_coverage(
                        connection, config["micro_table"], "ts"
                    ),
                    "book": transfer.table_coverage(
                        connection, config["book_table"], "recorded_at_kst"
                    ),
                }
                for asset, config in transfer.ASSETS.items()
            }
        finally:
            connection.close()
        common_start = max(
            max(row["micro"]["start_utc"], row["book"]["start_utc"])
            for row in coverage.values()
        ).ceil("min")
        evaluation_start = common_start + pd.Timedelta(
            minutes=transfer.MODEL_WINDOW_WARMUP_MINUTES
        )
        evaluation_end = min(
            min(row["micro"]["end_utc"], row["book"]["end_utc"])
            for row in coverage.values()
        ).floor("min")
        split_times = {
            "calibration_start": evaluation_start,
            "tune_start": evaluation_start + pd.Timedelta(hours=CALIBRATION_HOURS),
            "validation_start": evaluation_start
            + pd.Timedelta(hours=CALIBRATION_HOURS + TUNE_HOURS),
            "development_start": evaluation_start
            + pd.Timedelta(
                hours=CALIBRATION_HOURS + TUNE_HOURS + VALIDATION_HOURS
            ),
            "development_end": evaluation_end,
            "fresh_forward_start": evaluation_end + pd.Timedelta(minutes=1),
        }
        if split_times["development_end"] <= split_times["development_start"]:
            raise RuntimeError("insufficient rows for chronological adapter splits")
        source_start = common_start - pd.Timedelta(
            hours=transfer.FEATURE_CONTEXT_HOURS
        )
        results: dict[str, Any] = {}
        for asset, config in transfer.ASSETS.items():
            print(f"tuning frozen-weight {asset.upper()} adapter", flush=True)
            frame, source = transfer.build_asset_stream(
                snapshot_path, asset, config, source_start, evaluation_end
            )
            timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
            calibration_mask = np.asarray(
                (timestamps >= split_times["calibration_start"])
                & (timestamps < split_times["tune_start"])
            )
            adapter = fit_robust_adapter(frame, calibration_mask, runtime)
            adapted = apply_adapter(frame, adapter, runtime)
            prediction, end_indices = binding.observer.infer_stream(
                adapted, runtime
            )
            prediction_timestamps = pd.DatetimeIndex(timestamps[end_indices])
            close = frame["close"].to_numpy(dtype=np.float64)
            next_return_all = np.full(len(frame), np.nan, dtype=np.float64)
            next_return_all[:-1] = close[1:] / close[:-1] - 1.0
            returns = next_return_all[end_indices]
            available = binding.observer._available(frame)[end_indices]
            masks = {
                "tune": np.asarray(
                    (prediction_timestamps >= split_times["tune_start"])
                    & (prediction_timestamps < split_times["validation_start"])
                ),
                "validation": np.asarray(
                    (prediction_timestamps >= split_times["validation_start"])
                    & (prediction_timestamps < split_times["development_start"])
                ),
                "development": np.asarray(
                    (prediction_timestamps >= split_times["development_start"])
                    & (prediction_timestamps < split_times["development_end"])
                ),
            }
            split_inputs = {
                name: (
                    slice_prediction(prediction, mask),
                    available[mask],
                    returns[mask],
                    prediction_timestamps[mask],
                )
                for name, mask in masks.items()
            }
            selected_policy, candidates = select_tune_policy(
                *split_inputs["tune"], runtime.config.fee_per_notional_change
            )
            diagnostics = {
                name: {
                    "bars": int(mask.sum()),
                    "metrics_4_5bp": evaluate_policy(
                        *split_inputs[name],
                        selected_policy,
                        runtime.config.fee_per_notional_change,
                    ),
                    "cost_stress": cost_stress(
                        *split_inputs[name], selected_policy
                    ),
                }
                for name, mask in masks.items()
            }
            validation = diagnostics["validation"]["metrics_4_5bp"]
            diagnostics["validation_gate_pass"] = bool(
                selected_policy.enabled
                and validation["entries_or_reversals"] >= 1
                and validation["compounded_return_pct"] > 0.0
            )
            diagnostics["top_tune_candidates"] = candidates[:20]
            payload = artifact_payload(
                asset, runtime, adapter, selected_policy, split_times, diagnostics
            )
            artifact_dir = ARTIFACT_ROOT / MODEL_IDS[asset]
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "adapter.json"
            transfer.base._write_json_atomic(artifact_path, payload)
            results[asset] = {
                "artifact": str(artifact_path),
                "model_id": MODEL_IDS[asset],
                "source": source,
                "selected_research_policy": asdict(selected_policy),
                "diagnostics": diagnostics,
            }
    report = {
        "schema_version": "btc_sol_micro_scalp.transfer_adapter_tuning.v1",
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "parent_weights_frozen": True,
        "training_performed": False,
        "parameter_updates": 0,
        "selection_uses_only_tune_split": True,
        "validation_used_for_reporting_and_gate_only": True,
        "development_used_for_selection": False,
        "activation_allowed": False,
        "order_submission_supported": False,
        "split_times": {name: str(value) for name, value in split_times.items()},
        "microstructure_snapshot": snapshot,
        "assets": results,
        "evidence_class": "short-window transfer-adapter development; fresh shadow required",
        "promotion_pass": False,
    }
    transfer.base._write_json_atomic(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.report)
    print(json.dumps(result, indent=2, default=transfer.base._json_default))
