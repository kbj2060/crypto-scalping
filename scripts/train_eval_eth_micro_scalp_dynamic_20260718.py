"""Train and evaluate a dynamic ETH microstructure scalping policy.

The model forecasts five-minute ETH return, but five minutes is not a holding
period.  At every completed one-minute bar the policy chooses a target position
from SHORT/CASH/LONG.  A position remains open only while the forecast supports
it; there is no fixed/max holding period, TP/SL, or cooldown.

This is a research-only line.  All data through 2026-07-12 has already been
consumed by prior model-family research, so the validation and development
readouts produced here are diagnostics rather than promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "eth_micro_scalp_dynamic_v0_20260718"
CACHE_DIR = ROOT / "data/ensemble/deepscalp_pnl_v1_20260717/cache"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "development_diagnostic_ledger.csv"

BASE_FEATURES = (
    "bar_open_close_logret",
    "bar_range_pct",
    "log_volume",
    "log_trade_count",
    "bar_taker_buy_ratio",
    "log_return",
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "garman_klass_vol",
    "realized_vol_ratio",
    "amihud_illiquidity_z",
    "chop_index",
    "ofi_acceleration",
    "cvd_slope_12",
    "compression_score",
    "liquidity_vacuum",
    "execution_quality",
)

MICRO_CURRENT_FEATURES = (
    "micro_obi",
    "micro_taker_buy_ratio",
    "micro_nif_whale",
    "micro_nif_retail",
    "micro_oi_delta_pct",
    "micro_recent_trade_count_5m",
    "micro_recent_trade_notional_5m",
    "micro_recent_whale_count_5m",
    "micro_age_min",
)

MICRO_ROLLING_FEATURES = (
    "micro_obi",
    "micro_taker_buy_ratio",
    "micro_nif_whale",
    "micro_nif_retail",
)

REQUIRED_HEALTH_FEATURES = (
    "micro_available",
    "micro_data_stale",
    "micro_depth_connected",
    "micro_warmup_30m_ready",
    "micro_age_min",
)


@dataclass(frozen=True)
class Config:
    seed: int = 18
    forecast_horizon_min: int = 5
    fee_per_notional_change: float = 0.00045
    fit_start: str = "2026-05-03 00:00:00"
    tune_start: str = "2026-06-11 00:00:00"
    validation_start: str = "2026-06-21 00:00:00"
    development_start: str = "2026-07-01 00:00:00"
    development_end: str = "2026-07-12 09:01:00"
    min_tune_entries: int = 30
    max_iter: int = 180
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 80
    l2_regularization: float = 1.0


@dataclass(frozen=True)
class DynamicPolicy:
    enabled: bool
    entry_threshold_bp: float
    exit_threshold_bp: float


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_names(available: list[str], required: tuple[str, ...], group: str) -> list[int]:
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(f"{group} feature contract mismatch; missing={missing}")
    return [available.index(name) for name in required]


def load_frozen_cache(cache_dir: Path = CACHE_DIR) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    arrays = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in ("base", "micro", "targets", "next_return", "timestamp_ns")
    }
    if any("btc" in name.lower() for name in metadata["base_feature_names"]):
        raise RuntimeError("base feature contract contains BTC-derived input; fail-fast")
    if len({len(values) for values in arrays.values()}) != 1:
        raise RuntimeError("cache array length mismatch")
    return arrays, metadata


def build_model_frame(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: Config,
) -> tuple[np.ndarray, list[str], pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    timestamp_all = pd.to_datetime(np.asarray(arrays["timestamp_ns"], dtype=np.int64))
    buffer_start = pd.Timestamp(config.fit_start) - pd.Timedelta(minutes=20)
    span = (timestamp_all >= buffer_start) & (timestamp_all < pd.Timestamp(config.development_end))
    indices = np.flatnonzero(span)
    if len(indices) == 0:
        raise RuntimeError("no rows in configured research span")

    base_names = list(metadata["base_feature_names"])
    micro_names = list(metadata["micro_feature_names"])
    base_idx = _require_names(base_names, BASE_FEATURES, "base")
    micro_idx = _require_names(micro_names, MICRO_CURRENT_FEATURES, "micro current")
    rolling_idx = _require_names(micro_names, MICRO_ROLLING_FEATURES, "micro rolling")
    health_idx = _require_names(micro_names, REQUIRED_HEALTH_FEATURES, "micro health")

    raw_base = np.asarray(arrays["base"][indices][:, base_idx], dtype=np.float32)
    raw_micro = np.asarray(arrays["micro"][indices], dtype=np.float32)
    blocks = [raw_base, raw_micro[:, micro_idx]]
    feature_names = list(BASE_FEATURES) + list(MICRO_CURRENT_FEATURES)

    rolling_frame = pd.DataFrame(raw_micro[:, rolling_idx], columns=MICRO_ROLLING_FEATURES)
    for name in MICRO_ROLLING_FEATURES:
        series = rolling_frame[name]
        blocks.append(series.diff(1).to_numpy(dtype=np.float32)[:, None])
        feature_names.append(f"{name}_diff_1")
        blocks.append(series.diff(5).to_numpy(dtype=np.float32)[:, None])
        feature_names.append(f"{name}_diff_5")
        for window in (3, 5, 15):
            mean = series.rolling(window, min_periods=window).mean()
            blocks.append(mean.to_numpy(dtype=np.float32)[:, None])
            feature_names.append(f"{name}_mean_{window}")

    health = {name: raw_micro[:, idx] for name, idx in zip(REQUIRED_HEALTH_FEATURES, health_idx)}
    available = (
        np.isfinite(health["micro_available"])
        & (health["micro_available"] > 0.5)
        & np.isfinite(health["micro_data_stale"])
        & (health["micro_data_stale"] < 0.5)
        & np.isfinite(health["micro_depth_connected"])
        & (health["micro_depth_connected"] > 0.5)
        & np.isfinite(health["micro_warmup_30m_ready"])
        & (health["micro_warmup_30m_ready"] > 0.5)
        & np.isfinite(health["micro_age_min"])
        & (health["micro_age_min"] >= 0.0)
        & (health["micro_age_min"] <= 2.0)
    )

    matrix = np.column_stack(blocks).astype(np.float32, copy=False)
    target_bp = np.asarray(arrays["targets"][indices, 3], dtype=np.float32)
    next_return = np.asarray(arrays["next_return"][indices], dtype=np.float64)
    timestamps = timestamp_all[indices]
    return matrix, feature_names, timestamps, target_bp, next_return, available


def purged_interval_mask(
    timestamps: pd.DatetimeIndex,
    start: str,
    end: str,
    horizon_min: int,
) -> np.ndarray:
    end_exclusive = pd.Timestamp(end) - pd.Timedelta(minutes=horizon_min)
    return np.asarray((timestamps >= pd.Timestamp(start)) & (timestamps < end_exclusive))


def decide_positions(scores_bp: np.ndarray, available: np.ndarray, policy: DynamicPolicy) -> np.ndarray:
    scores = np.asarray(scores_bp, dtype=np.float64)
    usable = np.asarray(available, dtype=bool) & np.isfinite(scores)
    positions = np.zeros(len(scores), dtype=np.int8)
    if not policy.enabled:
        return positions

    current = 0
    entry = float(policy.entry_threshold_bp)
    exit_threshold = float(policy.exit_threshold_bp)
    for idx, score in enumerate(scores):
        if not usable[idx]:
            current = 0
        elif current == 0:
            if score >= entry:
                current = 1
            elif score <= -entry:
                current = -1
        elif current == 1:
            if score <= -entry:
                current = -1
            elif score < exit_threshold:
                current = 0
        else:
            if score >= entry:
                current = 1
            elif score > -exit_threshold:
                current = 0
        positions[idx] = current
    return positions


def holding_lengths(positions: np.ndarray) -> list[int]:
    result: list[int] = []
    current = 0
    length = 0
    for value in np.asarray(positions, dtype=np.int8):
        if value != 0 and value == current:
            length += 1
        else:
            if current != 0:
                result.append(length)
            current = int(value)
            length = 1 if value != 0 else 0
    if current != 0:
        result.append(length)
    return result


def replay_positions(
    positions: np.ndarray,
    next_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fee_per_notional_change: float,
    scores_bp: np.ndarray | None = None,
    available: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    position = np.asarray(positions, dtype=np.int8)
    returns = np.nan_to_num(np.asarray(next_return, dtype=np.float64), nan=0.0)
    previous = np.r_[0, position[:-1]].astype(np.int8)
    turnover = np.abs(position.astype(np.float64) - previous.astype(np.float64))
    if len(position) and position[-1] != 0:
        turnover[-1] += abs(float(position[-1]))
    gross = position.astype(np.float64) * returns
    cost = float(fee_per_notional_change) * turnover
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    curve = np.r_[1.0, equity]
    peak = np.maximum.accumulate(curve)
    drawdown = 1.0 - curve / peak
    lengths = holding_lengths(position)
    changes = position != previous
    entries_or_reversals = int(np.sum(changes & (position != 0)))
    exits_or_reversals = int(np.sum(changes & (previous != 0)))
    if len(position) and position[-1] != 0:
        exits_or_reversals += 1

    daily = pd.DataFrame({"timestamp": timestamps, "net": net})
    daily["date"] = daily["timestamp"].dt.date
    daily_returns = daily.groupby("date", sort=True)["net"].apply(lambda values: np.prod(1.0 + values) - 1.0)
    metrics = {
        "bars": int(len(position)),
        "days": int(len(daily_returns)),
        "compounded_return_pct": float((equity[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "additive_cost_pct": float(cost.sum() * 100.0),
        "max_drawdown_pct": float(drawdown.max() * 100.0) if len(drawdown) else 0.0,
        "entries_or_reversals": entries_or_reversals,
        "exits_or_reversals": exits_or_reversals,
        "turnover": float(turnover.sum()),
        "exposure_fraction": float(np.mean(position != 0)) if len(position) else 0.0,
        "long_fraction": float(np.mean(position > 0)) if len(position) else 0.0,
        "short_fraction": float(np.mean(position < 0)) if len(position) else 0.0,
        "positive_day_fraction": float(np.mean(daily_returns > 0.0)) if len(daily_returns) else 0.0,
        "holding_bars": {
            "count": int(len(lengths)),
            "min": int(min(lengths)) if lengths else 0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "p95": float(np.quantile(lengths, 0.95)) if lengths else 0.0,
            "max": int(max(lengths)) if lengths else 0,
        },
    }
    ledger = pd.DataFrame(
        {
            "timestamp": timestamps,
            "score_bp": np.asarray(scores_bp) if scores_bp is not None else np.nan,
            "available": np.asarray(available, dtype=bool) if available is not None else True,
            "previous_position": previous,
            "position": position,
            "turnover": turnover,
            "next_return": returns,
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "equity": equity,
        }
    )
    return metrics, ledger


def replay_policy(
    scores_bp: np.ndarray,
    next_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    available: np.ndarray,
    policy: DynamicPolicy,
    fee_per_notional_change: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    positions = decide_positions(scores_bp, available, policy)
    return replay_positions(
        positions,
        next_return,
        timestamps,
        fee_per_notional_change,
        scores_bp=scores_bp,
        available=available,
    )


def fit_model(matrix: np.ndarray, target_bp: np.ndarray, mask: np.ndarray, config: Config) -> HistGradientBoostingRegressor:
    fit_mask = np.asarray(mask, dtype=bool) & np.isfinite(target_bp)
    if int(fit_mask.sum()) < 1_000:
        raise RuntimeError(f"insufficient fit rows: {int(fit_mask.sum())}")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=False,
        random_state=config.seed,
    )
    model.fit(matrix[fit_mask], target_bp[fit_mask])
    return model


def select_policy(
    fit_scores_bp: np.ndarray,
    tune_scores_bp: np.ndarray,
    tune_returns: np.ndarray,
    tune_timestamps: pd.DatetimeIndex,
    tune_available: np.ndarray,
    config: Config,
) -> tuple[DynamicPolicy, list[dict[str, Any]]]:
    finite_fit = np.abs(np.asarray(fit_scores_bp, dtype=np.float64))
    finite_fit = finite_fit[np.isfinite(finite_fit)]
    if len(finite_fit) == 0:
        raise RuntimeError("model produced no finite fit scores")
    quantile_entries = np.quantile(finite_fit, (0.80, 0.90, 0.95, 0.975, 0.99))
    absolute_entries = np.asarray((1.0, 2.0, 3.0, 5.0, 8.0, 11.0, 15.0))
    entries = sorted({round(float(value), 6) for value in np.r_[quantile_entries, absolute_entries] if value > 0.0})
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        for ratio in (-1.00, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50):
            policy = DynamicPolicy(True, entry, entry * ratio)
            metrics, _ = replay_policy(
                tune_scores_bp,
                tune_returns,
                tune_timestamps,
                tune_available,
                policy,
                config.fee_per_notional_change,
            )
            net = metrics["compounded_return_pct"] / 100.0
            drawdown = metrics["max_drawdown_pct"] / 100.0
            eligible = metrics["entries_or_reversals"] >= config.min_tune_entries and net > 0.0
            selection_score = net - 0.25 * drawdown if eligible else float("-inf")
            candidates.append(
                {
                    "policy": asdict(policy),
                    "eligible": bool(eligible),
                    "selection_score": selection_score,
                    "metrics": metrics,
                }
            )
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    if not candidates or not np.isfinite(candidates[0]["selection_score"]) or candidates[0]["selection_score"] <= 0.0:
        return DynamicPolicy(False, 0.0, 0.0), candidates
    return DynamicPolicy(**candidates[0]["policy"]), candidates


def evaluate_split(
    model: HistGradientBoostingRegressor,
    matrix: np.ndarray,
    next_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    available: np.ndarray,
    mask: np.ndarray,
    policy: DynamicPolicy,
    config: Config,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    scores = model.predict(matrix[mask])
    metrics, ledger = replay_policy(
        scores,
        next_return[mask],
        timestamps[mask],
        available[mask],
        policy,
        config.fee_per_notional_change,
    )
    stress: dict[str, Any] = {}
    positions = ledger["position"].to_numpy(dtype=np.int8)
    for fee in (0.00020, 0.000325, 0.00045, 0.00055, 0.00090):
        stressed, _ = replay_positions(positions, next_return[mask], timestamps[mask], fee)
        stress[f"{fee * 10_000:.2f}bp_per_notional_change"] = stressed
    return scores, metrics, ledger, stress


def run(config: Config) -> dict[str, Any]:
    arrays, metadata = load_frozen_cache()
    matrix, feature_names, timestamps, target_bp, next_return, available = build_model_frame(arrays, metadata, config)
    fit_mask = purged_interval_mask(timestamps, config.fit_start, config.tune_start, config.forecast_horizon_min)
    tune_mask = purged_interval_mask(timestamps, config.tune_start, config.validation_start, config.forecast_horizon_min)
    validation_mask = purged_interval_mask(
        timestamps, config.validation_start, config.development_start, config.forecast_horizon_min
    )
    development_mask = purged_interval_mask(
        timestamps, config.development_start, config.development_end, config.forecast_horizon_min
    )
    masks = {
        "fit": fit_mask,
        "tune": tune_mask,
        "validation": validation_mask,
        "development": development_mask,
    }
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    if min(counts.values()) <= 0:
        raise RuntimeError(f"empty research split: {counts}")

    model = fit_model(matrix, target_bp, fit_mask & available, config)
    fit_scores = model.predict(matrix[fit_mask])
    tune_scores = model.predict(matrix[tune_mask])
    policy, candidates = select_policy(
        fit_scores,
        tune_scores,
        next_return[tune_mask],
        timestamps[tune_mask],
        available[tune_mask],
        config,
    )

    _, tune_metrics, _, tune_stress = evaluate_split(
        model, matrix, next_return, timestamps, available, tune_mask, policy, config
    )
    _, validation_metrics, validation_ledger, validation_stress = evaluate_split(
        model, matrix, next_return, timestamps, available, validation_mask, policy, config
    )
    _, development_metrics, development_ledger, development_stress = evaluate_split(
        model, matrix, next_return, timestamps, available, development_mask, policy, config
    )

    active_and_positive = (
        policy.enabled
        and validation_metrics["compounded_return_pct"] > 0.0
        and development_metrics["compounded_return_pct"] > 0.0
    )
    execution_policy = policy if active_and_positive else DynamicPolicy(False, 0.0, 0.0)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_id": MODEL_ID,
        "model": model,
        "feature_names": feature_names,
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": active_and_positive,
        "config": asdict(config),
        "cache_contract_sha256": metadata["source_signature"]["contract_sha256"],
        "fixed_holding_period_used": False,
    }
    joblib.dump(bundle, MODEL_PATH)
    validation_ledger.to_csv(VALIDATION_LEDGER_PATH, index=False)
    development_ledger.to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    if not policy.enabled:
        promotion_reason = "No active policy survived tune after modeled execution cost."
    elif validation_metrics["compounded_return_pct"] <= 0.0 or development_metrics["compounded_return_pct"] <= 0.0:
        promotion_reason = (
            "The tune-selected active policy failed locked validation/development; the artifact execution policy "
            "is fail-safe CASH."
        )
    else:
        promotion_reason = (
            "All data through 2026-07-12 is consumed development data; a newly frozen policy requires "
            "post-freeze fresh-forward shadow and execution calibration before promotion."
        )
    report = {
        "model_id": MODEL_ID,
        "status": "research_shadow_candidate" if active_and_positive else "research_no_viable_active_policy",
        "model_family": "single HGB five-minute edge regressor with per-minute stateful hysteresis policy",
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "forecast_horizon_min": config.forecast_horizon_min,
            "forecast_horizon_is_not_holding_period": True,
            "exit_rule": "CASH or opposite target is emitted when the current model score no longer supports the position",
        },
        "config": asdict(config),
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "btc_features_used": False,
            "rule_outputs_used": False,
            "raw_orderbook_direction_features_used": False,
            "availability_gate": "micro available, fresh, depth connected, warmup ready, age in [0,2] minutes",
            "cache_contract_sha256": metadata["source_signature"]["contract_sha256"],
        },
        "data": {
            "cache_dir": str(CACHE_DIR),
            "timestamp_min": str(timestamps.min()),
            "timestamp_max": str(timestamps.max()),
            "split_counts_before_availability_gate": counts,
            "available_counts": {name: int(np.sum(mask & available)) for name, mask in masks.items()},
            "splits": {
                "fit": [config.fit_start, config.tune_start],
                "tune": [config.tune_start, config.validation_start],
                "validation": [config.validation_start, config.development_start],
                "development": [config.development_start, config.development_end],
            },
            "purge_minutes_at_each_boundary": config.forecast_horizon_min,
        },
        "selected_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": active_and_positive,
        "tune": tune_metrics,
        "tune_cost_stress": tune_stress,
        "validation": validation_metrics,
        "validation_cost_stress": validation_stress,
        "development": development_metrics,
        "development_cost_stress": development_stress,
        "top_tune_candidates": candidates[:10],
        "risk_accounting": {
            "margin_fraction": 1.0,
            "leverage": 1.0,
            "notional": 1.0,
            "formula": "notional = margin_fraction * leverage; account_pnl = price_move * notional",
            "live_sizing_authorized": False,
        },
        "artifacts": {
            "model": str(MODEL_PATH),
            "validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "metadata_sha256": _sha256(CACHE_DIR / "metadata.json"),
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "fixed_holding_period_used": False,
            "outer_results_used_for_policy_selection": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": promotion_reason,
            "next_untouched_start": "after this 2026-07-18 model freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=_json_default))
    print(json.dumps({"selected_policy": asdict(policy), "tune": tune_metrics, "validation": validation_metrics, "development": development_metrics}, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(max_iter=20, min_samples_leaf=120) if args.smoke else Config()
    run(cfg)
