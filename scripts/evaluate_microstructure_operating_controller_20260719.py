#!/usr/bin/env python3
"""Evaluate five microstructure controller roles without enabling execution.

ETH uses the frozen v4 per-bar research policy over its existing chronological
fit/tune/validation/development contract.  Controller thresholds are selected on
tune only and reported on locked historical validation/development diagnostics.
BTC/SOL allocation and maker-placement results are imported from their dedicated
causal diagnostic reports.  Nothing in this script can submit an order.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_micro_scalp_reuse_layers_20260718 as reuse  # noqa: E402
import run_eth_micro_scalp_v4_fresh_forward_observer_20260718 as binding  # noqa: E402
import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402
from trading_bot_modules.microstructure_operating_controller import (  # noqa: E402
    CONTROLLER_ID,
    ControllerConfig,
    fragility_score,
    margin_for_entry,
    opportunity_score,
)


REPORT_PATH = ROOT / "data/ensemble/reports/microstructure_operating_controller_v1_20260719.json"
MICRO_DB = ROOT / "data/live/microstructure.duckdb"
TAIL_DB = ROOT / "data/live/tail_risk.duckdb"
EXECUTION_REPORT = ROOT / "data/ensemble/reports/micro_exec_placement_replay_20260718.json"
ALLOCATION_REPORT = ROOT / "data/ensemble/reports/micro_scalp_reuse_layers_test_20260718.json"
FEE_BP = 4.5
FIXED_LEVERAGE = 3.0
BASE_MARGIN_FRACTION = 0.30
BASE_NOTIONAL = BASE_MARGIN_FRACTION * FIXED_LEVERAGE
ACTIONS = np.asarray((-1, 0, 1), dtype=np.int8)


@dataclass(frozen=True)
class OverlayPolicy:
    min_opportunity: float = 0.0
    max_entry_risk: float = 1.01
    min_entry_alignment: float = -1.01
    urgent_exit_risk: float = 1.01
    passive_exit_opportunity: float = -0.01
    passive_exit_alignment: float = -1.01


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _robust_unit(values: np.ndarray, fit_mask: np.ndarray, *, log1p: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if log1p:
        array = np.log1p(np.maximum(array, 0.0))
    sample = array[fit_mask & np.isfinite(array)]
    if not len(sample):
        raise RuntimeError("normalizer fit sample is empty")
    center = float(np.median(sample))
    scale = float(np.median(np.abs(sample - center)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(sample))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.full(len(array), 0.5, dtype=np.float64)
    z = np.clip((array - center) / scale, -6.0, 6.0)
    unit = 1.0 / (1.0 + np.exp(-z))
    return np.where(np.isfinite(array), unit, 0.0)


def _slice_prediction(prediction: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[mask] for name, values in prediction.items()}


def _snapshot_database(source: Path, destination: Path, retries: int = 5) -> None:
    for _attempt in range(retries):
        before = source.stat()
        shutil.copy2(source, destination)
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
            return
        time.sleep(0.2)
    raise RuntimeError(f"database changed during every snapshot attempt: {source}")


def _load_raw_extras(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    start = timestamps.min() - pd.Timedelta(minutes=2)
    end = timestamps.max() - pd.Timedelta(minutes=2)
    with tempfile.TemporaryDirectory(prefix="micro-controller-snapshot-") as directory:
        micro_path = Path(directory) / "microstructure.duckdb"
        tail_path = Path(directory) / "tail_risk.duckdb"
        _snapshot_database(MICRO_DB, micro_path)
        _snapshot_database(TAIL_DB, tail_path)
        connection = duckdb.connect(str(micro_path), read_only=True)
        try:
            micro = connection.execute(
                """
                SELECT timezone('UTC', ts) AS source_ts, taker_buy_ratio, nif_whale, obi, eai,
                       shadow_queue_collapse, shadow_toxicity_score, spoofing_score
                FROM microstructure_1m
                WHERE timezone('UTC', ts) BETWEEN ? AND ?
                ORDER BY source_ts
                """,
                [start.to_pydatetime(), end.to_pydatetime()],
            ).fetchdf()
        finally:
            connection.close()

        connection = duckdb.connect(str(tail_path), read_only=True)
        try:
            tail = connection.execute(
                """
                SELECT timezone('UTC', ts) AS source_ts, shadow_aftershock_prob,
                       valid_liq_stream, schema_version
                FROM tail_risk_1m
                WHERE timezone('UTC', ts) BETWEEN ? AND ?
                ORDER BY source_ts
                """,
                [start.to_pydatetime(), end.to_pydatetime()],
            ).fetchdf()
        finally:
            connection.close()
    micro["timestamp"] = pd.to_datetime(micro.pop("source_ts")) + pd.Timedelta(minutes=2)
    micro.drop_duplicates("timestamp", keep="last", inplace=True)
    micro.set_index("timestamp", inplace=True)

    tail["timestamp"] = pd.to_datetime(tail.pop("source_ts")) + pd.Timedelta(minutes=2)
    tail.drop_duplicates("timestamp", keep="last", inplace=True)
    tail.set_index("timestamp", inplace=True)
    return pd.DataFrame(index=timestamps).join(micro).join(tail)


def load_eth_data(device: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    runtime = binding.observer.load_runtime(device_name=device)
    config = v4.SourceStableConfig(**runtime.checkpoint["config"])
    prepared = v4.prepare_source_stable_data(config)
    for name in ("base_center", "base_scale", "micro_center", "micro_scale"):
        if not np.allclose(
            np.asarray(prepared["scalers"][name]),
            np.asarray(runtime.checkpoint["scalers"][name]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(f"frozen scaler mismatch: {name}")

    predictions: list[dict[str, np.ndarray]] = []
    split_names: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    for split in ("fit", "tune", "validation", "development"):
        indices = np.asarray(prepared["split_indices"][split], dtype=np.int64)
        rows = [
            v4.v3.infer(
                model, prepared["base"], prepared["micro"], indices,
                runtime.config, runtime.device,
            )
            for model in runtime.models
        ]
        predictions.append(v4.v3.aggregate_seed_predictions(rows))
        split_names.append(np.full(len(indices), split, dtype=object))
        source_indices.append(indices)
        print(f"ETH {split}: inferred {len(indices):,} rows", flush=True)

    prediction = {
        name: np.concatenate([row[name] for row in predictions], axis=0)
        for name in predictions[0]
    }
    indices = np.concatenate(source_indices)
    split = np.concatenate(split_names)
    frame = prepared["frame"]
    timestamps = pd.DatetimeIndex(frame["timestamps"][indices])
    order = np.argsort(timestamps.to_numpy())
    timestamps = timestamps[order]
    indices = indices[order]
    split = split[order]
    prediction = {name: values[order] for name, values in prediction.items()}
    available = np.asarray(frame["available"][indices], dtype=bool)
    returns = np.asarray(frame["next_return"][indices], dtype=np.float64)
    micro_names = list(frame["micro_names"])
    micro_raw = np.asarray(frame["micro_raw"][indices], dtype=np.float64)

    def feature(name: str) -> np.ndarray:
        return micro_raw[:, micro_names.index(name)]

    cash_q = np.asarray(prediction["q"], dtype=np.float64)[:, 1, :]
    action_index = np.argmax(cash_q, axis=1)
    rows = np.arange(len(action_index))
    desired = ACTIONS[action_index]
    desired = np.where(available, desired, 0).astype(np.int8)
    expert_cash_q = np.asarray(prediction["expert_q"], dtype=np.float64)[:, :, 1, :]
    agreement = np.sum(np.argmax(expert_cash_q, axis=2) == action_index[:, None], axis=1)
    uncertainty = np.std(expert_cash_q[rows, :, action_index], axis=1)
    gate_entropy = reuse._gate_entropy(prediction["gate"], runtime.config.experts)
    data: dict[str, Any] = {
        "asset": "eth",
        "timestamps": timestamps,
        "returns": returns,
        "available": available,
        "liquidity_healthy": available & (feature("book_available") > 0.5),
        "prediction": prediction,
        "desired": desired,
        "edge_bp": cash_q[rows, action_index] - cash_q[:, 1],
        "agreement": agreement,
        "uncertainty": uncertainty,
        "gate_entropy": gate_entropy,
        "split": split,
        "trade_count": feature("micro_recent_trade_count_5m"),
        "trade_notional": feature("micro_recent_trade_notional_5m"),
        "whale_count": feature("micro_recent_whale_count_5m"),
    }
    masks = {name: split == name for name in ("fit", "tune", "validation", "development")}
    detector = reuse.fit_risk_detector(data, masks["fit"])
    reuse.apply_risk_detector(data, detector)
    extras = _load_raw_extras(timestamps)
    data["extras"] = extras

    notional = _robust_unit(data["trade_notional"], masks["fit"], log1p=True)
    whales = _robust_unit(data["whale_count"], masks["fit"], log1p=True)
    energy = _robust_unit(extras["eai"].to_numpy(), masks["fit"], log1p=True)
    queue = np.clip(extras["shadow_queue_collapse"].fillna(1.0).to_numpy(dtype=float), 0.0, 1.0)
    data["opportunity"] = np.asarray(
        [
            opportunity_score(
                trade_notional=n, whale_activity=w, energy=e, queue_collapse=q
            )
            for n, w, e, q in zip(notional, whales, energy, queue)
        ],
        dtype=np.float64,
    )

    model_risk = _robust_unit(np.asarray(data["risk_score"]), masks["fit"])
    toxicity = np.clip(extras["shadow_toxicity_score"].fillna(1.0).to_numpy(dtype=float), 0.0, 1.0)
    spoofing = np.clip(extras["spoofing_score"].fillna(1.0).to_numpy(dtype=float), 0.0, 1.0)
    valid_tail = (
        extras["schema_version"].fillna(0).to_numpy(dtype=float) >= 3
    ) & extras["valid_liq_stream"].fillna(False).to_numpy(dtype=bool)
    aftershock = np.where(
        valid_tail,
        np.clip(extras["shadow_aftershock_prob"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0),
        0.0,
    )
    data["risk"] = np.asarray(
        [
            fragility_score(
                model_risk=m, toxicity=t, queue_collapse=q,
                aftershock=a, spoofing=s,
            )
            for m, t, q, a, s in zip(model_risk, toxicity, queue, aftershock, spoofing)
        ],
        dtype=np.float64,
    )

    tbr = extras["taker_buy_ratio"].to_numpy(dtype=float) - 0.5
    nif = extras["nif_whale"].to_numpy(dtype=float)
    obi = extras["obi"].to_numpy(dtype=float)
    flow = (
        0.40 * (2.0 * _robust_unit(tbr, masks["fit"]) - 1.0)
        + 0.35 * (2.0 * _robust_unit(nif, masks["fit"]) - 1.0)
        + 0.25 * (2.0 * _robust_unit(obi, masks["fit"]) - 1.0)
    )
    data["contrarian_support"] = np.clip(-flow, -1.0, 1.0)
    return data, runtime, {"risk_detector": detector, "valid_tail_rows": int(valid_tail.sum())}


def apply_overlay(
    parent: np.ndarray,
    opportunity: np.ndarray,
    risk: np.ndarray,
    contrarian_support: np.ndarray,
    policy: OverlayPolicy,
) -> tuple[np.ndarray, dict[str, int]]:
    positions = np.zeros(len(parent), dtype=np.int8)
    current = 0
    counters = {"opportunity_blocks": 0, "risk_blocks": 0, "alignment_blocks": 0, "micro_exits": 0}
    for index, parent_position in enumerate(np.asarray(parent, dtype=np.int8)):
        parent_position = int(parent_position)
        if parent_position == 0:
            current = 0
            continue
        if current and current != parent_position:
            current = 0
        alignment = float(parent_position) * float(contrarian_support[index])
        exited = False
        if current:
            if risk[index] >= policy.urgent_exit_risk or (
                opportunity[index] <= policy.passive_exit_opportunity
                and alignment <= policy.passive_exit_alignment
            ):
                current = 0
                exited = True
                counters["micro_exits"] += 1
        if current == 0 and not exited:
            if opportunity[index] < policy.min_opportunity:
                counters["opportunity_blocks"] += 1
            elif risk[index] >= policy.max_entry_risk:
                counters["risk_blocks"] += 1
            elif alignment < policy.min_entry_alignment:
                counters["alignment_blocks"] += 1
            else:
                current = parent_position
        positions[index] = current
    return positions, counters


def position_weights(
    positions: np.ndarray,
    opportunity: np.ndarray,
    risk: np.ndarray,
    *,
    dynamic: bool,
) -> np.ndarray:
    weights = np.zeros(len(positions), dtype=np.float64)
    current_position = 0
    current_notional = 0.0
    config = ControllerConfig(
        base_margin_fraction=BASE_MARGIN_FRACTION,
        leverage=FIXED_LEVERAGE,
    )
    for index, position in enumerate(np.asarray(positions, dtype=np.int8)):
        position = int(position)
        if position == 0:
            current_position = 0
            current_notional = 0.0
            continue
        if position != current_position:
            if dynamic:
                _, current_notional = margin_for_entry(
                    float(opportunity[index]), float(risk[index]), config
                )
            else:
                current_notional = BASE_NOTIONAL
            current_position = position
        weights[index] = position * current_notional
    return weights


def replay_weights(
    weights: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    fee_bp: float = FEE_BP,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    returns = np.nan_to_num(np.asarray(returns, dtype=np.float64), nan=0.0)
    previous = np.r_[0.0, weights[:-1]]
    turnover = np.abs(weights - previous)
    if len(weights) and weights[-1] != 0.0:
        turnover[-1] += abs(weights[-1])
    gross = weights * returns
    cost = fee_bp / 10_000.0 * turnover
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    curve = np.r_[1.0, equity]
    peak = np.maximum.accumulate(curve)
    daily = pd.DataFrame({"timestamp": timestamps, "net": net})
    day_return = daily.groupby(daily["timestamp"].dt.date)["net"].apply(
        lambda values: np.prod(1.0 + values) - 1.0
    )
    changes = weights != previous
    return {
        "bars": int(len(weights)),
        "compounded_return_pct": float((equity[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "additive_gross_return_pct": float(gross.sum() * 100.0),
        "additive_cost_pct": float(cost.sum() * 100.0),
        "max_drawdown_pct": float((1.0 - curve / peak).max() * 100.0),
        "entries_or_reversals": int(np.sum(changes & (weights != 0.0))),
        "turnover": float(turnover.sum()),
        "exposure_fraction": float(np.mean(weights != 0.0)),
        "average_notional_when_active": float(np.mean(np.abs(weights[weights != 0.0]))) if np.any(weights) else 0.0,
        "average_margin_fraction_when_active": float(np.mean(np.abs(weights[weights != 0.0])) / FIXED_LEVERAGE) if np.any(weights) else 0.0,
        "max_margin_fraction": float(np.max(np.abs(weights)) / FIXED_LEVERAGE) if len(weights) else 0.0,
        "positive_day_fraction": float(np.mean(day_return > 0.0)) if len(day_return) else 0.0,
    }


def _score(metrics: dict[str, Any]) -> float:
    return float(metrics["compounded_return_pct"]) - 0.25 * float(metrics["max_drawdown_pct"])


def select_variant(
    parent: np.ndarray,
    opportunity: np.ndarray,
    risk: np.ndarray,
    support: np.ndarray,
    returns: np.ndarray,
    timestamps: pd.DatetimeIndex,
    candidates: list[OverlayPolicy],
) -> tuple[OverlayPolicy, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for policy in candidates:
        positions, counters = apply_overlay(parent, opportunity, risk, support, policy)
        metrics = replay_weights(
            position_weights(positions, opportunity, risk, dynamic=False),
            returns,
            timestamps,
        )
        rows.append({"policy": asdict(policy), "score": _score(metrics), "metrics": metrics, "counters": counters})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return OverlayPolicy(**rows[0]["policy"]), rows


def _forward_return(one_minute_returns: np.ndarray, horizon: int) -> np.ndarray:
    values = np.nan_to_num(np.asarray(one_minute_returns, dtype=np.float64), nan=0.0)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) >= horizon:
        windows = np.lib.stride_tricks.sliding_window_view(values, horizon)
        result[: len(windows)] = np.prod(1.0 + windows, axis=1) - 1.0
    return result


def opportunity_diagnostic(
    opportunity: np.ndarray,
    returns: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    low = opportunity <= low_threshold
    high = opportunity >= high_threshold
    result: dict[str, Any] = {
        "low_rows": int(np.sum(low)),
        "high_rows": int(np.sum(high)),
    }
    for horizon in (1, 5, 15):
        forward = _forward_return(returns, horizon)
        for name, mask in (("low", low), ("high", high)):
            values = np.abs(forward[mask]) * 10_000.0
            values = values[np.isfinite(values)]
            result[f"{name}_abs_move_{horizon}m_bp"] = float(np.mean(values)) if len(values) else None
            result[f"{name}_prob_abs_gt_9bp_{horizon}m"] = float(np.mean(values > 9.0)) if len(values) else None
    low_move = result["low_abs_move_5m_bp"]
    high_move = result["high_abs_move_5m_bp"]
    result["high_to_low_abs_move_5m_ratio"] = (
        float(high_move / low_move) if low_move and high_move is not None else None
    )
    return result


def _execution_summary() -> dict[str, Any]:
    report = _json(EXECUTION_REPORT)
    rows = [
        row for row in report.get("results", [])
        if row.get("arm") == "naive_join" and int(row.get("deadline_min", 0)) == 5
    ]
    passed = bool(rows) and all(
        float(row["improve_mean_bps"]) > 0.0 and float(row["improve_daily_t"]) >= 3.0
        for row in rows
    )
    return {
        "report": str(EXECUTION_REPORT),
        "fixed_policy_rows": rows,
        "diagnostic_pass": passed,
        "limitation": "one-minute strict trade-through proxy; actual queue position and fills are not observed",
    }


def _allocation_summary() -> dict[str, Any]:
    report = _json(ALLOCATION_REPORT)
    splits = ((report.get("portfolio") or {}).get("splits") or {})
    summary: dict[str, Any] = {"report": str(ALLOCATION_REPORT), "splits": {}}
    passes: list[bool] = []
    for name in ("validation", "development", "fresh_shadow"):
        row = splits.get(name) or {}
        concurrent = row.get("concurrent_unit_gross_baseline") or {}
        allocator = row.get("single_asset_allocator") or {}
        if not concurrent or not allocator:
            continue
        improved = float(allocator["compounded_return_pct"]) > float(concurrent["compounded_return_pct"])
        passes.append(improved)
        summary["splits"][name] = {
            "concurrent_return_pct": concurrent["compounded_return_pct"],
            "allocator_return_pct": allocator["compounded_return_pct"],
            "allocator_mdd_pct": allocator["max_drawdown_pct"],
            "improved": improved,
            "selected_bars": row.get("selected_bars"),
        }
    summary["diagnostic_pass"] = bool(passes) and all(passes)
    summary["evidence_class"] = report.get("evidence_class", "missing")
    return summary


def _tail_readiness() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tail-readiness-snapshot-") as directory:
        snapshot = Path(directory) / "tail_risk.duckdb"
        _snapshot_database(TAIL_DB, snapshot)
        connection = duckdb.connect(str(snapshot), read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT schema_version, count(*) AS rows, min(ts), max(ts),
                       sum(CAST(liq_event_count_1m > 0 AS INTEGER)) AS event_bars,
                       sum(CAST(shadow_aftershock_prob > 0 AS INTEGER)) AS nonzero_aftershock
                FROM tail_risk_1m GROUP BY schema_version ORDER BY schema_version
                """
            ).fetchall()
        finally:
            connection.close()
    payload = [
        {
            "schema_version": int(row[0]), "rows": int(row[1]),
            "start": str(row[2]), "end": str(row[3]),
            "event_bars": int(row[4]), "nonzero_aftershock": int(row[5]),
        }
        for row in rows
    ]
    valid_rows = sum(row["rows"] for row in payload if row["schema_version"] >= 3)
    return {
        "schema_segments": payload,
        "meaningful_schema_v3_rows": valid_rows,
        "performance_testable": valid_rows >= 1_440,
        "historical_schema_v2_eligible": False,
    }


def run(device: str = "cpu", report_path: Path = REPORT_PATH) -> dict[str, Any]:
    data, runtime, diagnostics = load_eth_data(device)
    masks = {name: data["split"] == name for name in ("fit", "tune", "validation", "development")}
    fit_opportunity = data["opportunity"][masks["fit"]]
    low_threshold = float(np.quantile(fit_opportunity, 0.20))
    high_threshold = float(np.quantile(fit_opportunity, 0.80))

    def split_values(name: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        mask = masks[name]
        prediction = _slice_prediction(data["prediction"], mask)
        parent, _ = v4.v3.decide_positions(prediction, data["available"][mask], runtime.policy)
        return (
            parent,
            prediction,
            data["opportunity"][mask],
            data["risk"][mask],
            data["contrarian_support"][mask],
            data["timestamps"][mask],
        )

    tune_parent, _, tune_opp, tune_risk, tune_support, tune_ts = split_values("tune")
    tune_returns = data["returns"][masks["tune"]]
    opportunity_candidates = [
        OverlayPolicy(min_opportunity=value)
        for value in (0.0, 0.50, 0.60, 0.70, 0.80)
    ]
    opportunity_policy, opportunity_rows = select_variant(
        tune_parent, tune_opp, tune_risk, tune_support, tune_returns, tune_ts,
        opportunity_candidates,
    )
    risk_candidates = [
        OverlayPolicy(max_entry_risk=value)
        for value in (0.60, 0.70, 0.80, 0.90, 1.01)
    ]
    risk_policy, risk_rows = select_variant(
        tune_parent, tune_opp, tune_risk, tune_support, tune_returns, tune_ts,
        risk_candidates,
    )
    exit_candidates = [
        OverlayPolicy(
            urgent_exit_risk=risk_cut,
            passive_exit_opportunity=opportunity_cut,
            passive_exit_alignment=alignment_cut,
        )
        for risk_cut, opportunity_cut, alignment_cut in itertools.product(
            (0.70, 0.80, 0.90, 1.01), (0.15, 0.25, 0.35), (-0.75, -0.50, -0.25)
        )
    ]
    exit_policy, exit_rows = select_variant(
        tune_parent, tune_opp, tune_risk, tune_support, tune_returns, tune_ts,
        exit_candidates,
    )
    combined_policy = OverlayPolicy(
        min_opportunity=opportunity_policy.min_opportunity,
        max_entry_risk=risk_policy.max_entry_risk,
        min_entry_alignment=-0.75,
        urgent_exit_risk=exit_policy.urgent_exit_risk,
        passive_exit_opportunity=exit_policy.passive_exit_opportunity,
        passive_exit_alignment=exit_policy.passive_exit_alignment,
    )

    policies = {
        "parent": OverlayPolicy(),
        "opportunity_gate": opportunity_policy,
        "risk_entry_gate": risk_policy,
        "micro_dynamic_exit": exit_policy,
        "combined_controller": combined_policy,
    }
    eth_splits: dict[str, Any] = {}
    for name in ("tune", "validation", "development"):
        parent, _, opportunity, risk, support, timestamps = split_values(name)
        returns = data["returns"][masks[name]]
        variants: dict[str, Any] = {}
        for variant, policy in policies.items():
            positions, counters = apply_overlay(parent, opportunity, risk, support, policy)
            fixed = replay_weights(
                position_weights(positions, opportunity, risk, dynamic=False), returns, timestamps
            )
            variants[variant] = {"policy": asdict(policy), "fixed_sizing": fixed, "counters": counters}
            if variant == "combined_controller":
                variants[variant]["dynamic_risk_sizing"] = replay_weights(
                    position_weights(positions, opportunity, risk, dynamic=True), returns, timestamps
                )
        eth_splits[name] = {
            "start": str(timestamps.min()),
            "end": str(timestamps.max()),
            "bars": len(timestamps),
            "opportunity_magnitude": opportunity_diagnostic(
                opportunity, returns, low_threshold, high_threshold
            ),
            "variants": variants,
        }

    def improved(module: str, sizing: str = "fixed_sizing") -> bool:
        return all(
            float(eth_splits[name]["variants"][module][sizing]["compounded_return_pct"])
            > float(eth_splits[name]["variants"]["parent"]["fixed_sizing"]["compounded_return_pct"])
            for name in ("validation", "development")
        )

    sizing_improved = all(
        _score(eth_splits[name]["variants"]["combined_controller"]["dynamic_risk_sizing"])
        > _score(eth_splits[name]["variants"]["combined_controller"]["fixed_sizing"])
        for name in ("validation", "development")
    )
    opportunity_pass = all(
        float(eth_splits[name]["opportunity_magnitude"]["high_to_low_abs_move_5m_ratio"] or 0.0) > 1.25
        for name in ("validation", "development")
    )
    report = {
        "schema_version": "microstructure.operating_controller.evaluation.v1",
        "controller_id": CONTROLLER_ID,
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "parent_model_id": v4.MODEL_ID,
        "parent_model_sha256": runtime.model_sha256,
        "controller_formula_trained": False,
        "threshold_selection_split": "tune only",
        "locked_reporting_splits": ["validation", "development"],
        "selected_policies": {
            "opportunity_gate": asdict(opportunity_policy),
            "risk_entry_gate": asdict(risk_policy),
            "micro_dynamic_exit": asdict(exit_policy),
            "combined_controller": asdict(combined_policy),
        },
        "top_tune_candidates": {
            "opportunity_gate": opportunity_rows[:5],
            "risk_entry_gate": risk_rows[:5],
            "micro_dynamic_exit": exit_rows[:5],
        },
        "eth": {
            "splits": eth_splits,
            "module_verdicts": {
                "opportunity_magnitude_detector": opportunity_pass,
                "opportunity_entry_gate": improved("opportunity_gate"),
                "risk_entry_gate": improved("risk_entry_gate"),
                "micro_dynamic_exit": improved("micro_dynamic_exit"),
                "combined_controller": improved("combined_controller"),
                "dynamic_margin_sizing_utility": sizing_improved,
            },
            "diagnostics": diagnostics,
            "base_margin_fraction": BASE_MARGIN_FRACTION,
            "fixed_leverage": FIXED_LEVERAGE,
            "base_notional": BASE_NOTIONAL,
            "notional_contract": "notional = margin_fraction * leverage",
        },
        "execution_assist": _execution_summary(),
        "multi_asset_allocator": _allocation_summary(),
        "tail_risk_readiness": _tail_readiness(),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "historical_validation_and_development_are_diagnostic_only": True,
        "activation_allowed": False,
        "order_submission_supported": False,
        "promotion_pass": False,
    }
    binding.observer._write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.device, args.report)
    print(json.dumps(report, indent=2, default=binding.observer._json_default))


if __name__ == "__main__":
    main()
