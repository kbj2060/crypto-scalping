#!/usr/bin/env python3
"""Run the predeclared T1-T4 CSALT DP-advantage development search.

This research-only script never opens T5/T6. It fits purged teachers on each
fold's preceding history, predicts the label fold from causal state only, and
uses realized label-fold paths solely to compare the predeclared candidates.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import test_btc_csalt_causal_baseline_20260715 as baseline
import test_btc_csalt_stage01_20260715 as core
import test_btc_csalt_teacher_t1_smoke_20260715 as smoke


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "dp_advantage_dev"
QUANTILES = (0.10, 0.25, 0.50)
PENALTIES = (0.0, 0.5, 1.0)
MIN_EDGES = (0.0, 0.00025, 0.00050, 0.00100)
MIN_VOTES = (0.60, 0.80)
COST_GATES = (False, True)
TARGETS = ("immediate_advantage", "dp_advantage", "stress15_dp_advantage")
FEATURE_SETS = ("derived11", "btc_native_stationary")
NATIVE_FEATURES = (
    "log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope",
    "wick_ratio", "garman_klass_vol", "realized_vol_ratio", "mtf_trend_1h",
    "mtf_trend_4h", "chop_index", "funding_z_score", "long_squeeze_risk",
    "short_squeeze_risk", "hurst_48", "hurst_288", "regime_trending",
    "ofi_acceleration", "kalman_velocity", "realized_skewness", "funding_pressure",
    "cvd_slope_12", "cvd_slope_48", "bb_width_pct_rank_288",
    "atr_pct_rank_288", "compression_score", "vwap_dist_96",
    "distance_to_day_high_low_pct", "crowding_pressure", "execution_quality",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def load_native_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", *NATIVE_FEATURES]
    parts = [
        pd.read_csv(core.FIVE_MINUTE_DIR / f"btc_features_{year}.csv", usecols=columns, parse_dates=["timestamp"])
        for year in (2024, 2025)
    ]
    features = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    features = features.reset_index(drop=True)
    if len(features) != len(frame) or not features["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError("BTC-native feature timestamps do not exactly match the market tape")
    values = features.loc[:, NATIVE_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("BTC-native stationary feature contract contains non-finite values")
    return features.loc[:, NATIVE_FEATURES]


def advantage(values: np.ndarray) -> np.ndarray:
    result = values - values[:, [0]]
    result[:, 0] = 0.0
    return result


def candidate_key(
    feature_set: str,
    target: str,
    quantile: float,
    penalty: float,
    min_edge: float,
    min_vote: float,
    cost_gate: bool,
) -> str:
    return (
        f"{feature_set}__{target}__q{int(round(quantile * 100)):02d}"
        f"__u{int(round(penalty * 10)):02d}__e{int(round(min_edge * 100000)):03d}"
        f"__v{int(round(min_vote * 100)):02d}__c{int(cost_gate)}"
    )


def lightweight_label_chart(
    path: Path,
    close: np.ndarray,
    event_indices: np.ndarray,
    labels: np.ndarray,
) -> None:
    width = min(1200, max(400, len(labels)))
    sample = np.linspace(0, len(labels) - 1, width).astype(np.int64)
    event_close = close[event_indices[: len(labels)]][sample]
    lo, hi = float(np.min(event_close)), float(np.max(event_close))
    scaled = (event_close - lo) / max(hi - lo, 1e-12)
    image = np.full((96, width, 3), 1.0, dtype=np.float32)
    row = 55 - np.rint(50 * scaled).astype(np.int64)
    image[row, np.arange(width)] = np.array([0.05, 0.05, 0.05])
    sides = np.array([core.ACTIONS[index].side for index in labels[sample]], dtype=np.int8)
    colors = np.empty((width, 3), dtype=np.float32)
    colors[sides == 0] = np.array([0.70, 0.70, 0.70])
    colors[sides > 0] = np.array([0.10, 0.65, 0.20])
    colors[sides < 0] = np.array([0.85, 0.15, 0.15])
    image[64:96] = colors[None, :, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def fit_fold_predictions(
    fold_index: int,
    frame: pd.DataFrame,
    funding: core.FundingTape,
    features: pd.DataFrame,
    *,
    target_names: tuple[str, ...] = TARGETS,
    quantiles: tuple[float, ...] = QUANTILES,
) -> dict[str, Any]:
    fold = core.FOLDS[fold_index]
    window = baseline.LABEL_WINDOWS[fold_index]
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    train_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    train_events_all = core.build_dollar_events(activity, 0, train_cutoff, threshold)
    train_count = int(np.searchsorted(train_events_all, train_end, side="right"))
    train_events = train_events_all[:train_count]
    rewards_all, _, successors_all, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding
    )
    stress_all, _, stress_successors_all, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding, cost_multiplier=1.5
    )
    immediate_target = advantage(rewards_all[:train_count])
    dp_target = advantage(core.finite_smdp_q(train_events_all, rewards_all, successors_all)[:train_count])
    stress_target = advantage(
        core.finite_smdp_q(train_events_all, stress_all, stress_successors_all)[:train_count]
    )

    label_start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    label_end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    label_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    label_events = core.build_dollar_events(activity, label_start, label_cutoff, threshold)
    label_count = int(np.searchsorted(label_events, label_end, side="right"))
    label_rewards, label_exits, label_successors, _ = core.simulate_action_table(
        frame, label_events, label_cutoff, funding
    )
    label_stress, label_stress_exits, label_stress_successors, _ = core.simulate_action_table(
        frame, label_events, label_cutoff, funding, cost_multiplier=1.5
    )
    train_x = features.iloc[train_events].to_numpy(dtype=np.float64)
    label_x = features.iloc[label_events[:label_count]].to_numpy(dtype=np.float64)
    if not np.isfinite(label_x).all():
        raise RuntimeError(f"{fold.name}: non-finite label-fold features")
    bootstrap = [
        smoke.block_bootstrap_indices(frame["timestamp"].iloc[train_events].reset_index(drop=True), seed)
        for seed in smoke.SEEDS
    ]
    targets = {
        "immediate_advantage": immediate_target,
        "dp_advantage": dp_target,
        "stress15_dp_advantage": stress_target,
    }
    predictions: dict[tuple[str, float], np.ndarray] = {}
    for target_name in target_names:
        target_values = targets[target_name]
        for quantile in quantiles:
            print(f"[{fold.name}] fit {target_name} q{quantile:.2f}", flush=True)
            predictions[(target_name, quantile)] = smoke.fit_quantile_predictions(
                train_x, target_values, label_x, bootstrap, quantile
            )
    return {
        "fold": fold.name,
        "window": window,
        "train_events": len(train_events),
        "label_events": label_events,
        "label_count": label_count,
        "threshold": threshold,
        "rewards": label_rewards,
        "exits": label_exits,
        "successors": label_successors,
        "stress_rewards": label_stress,
        "stress_exits": label_stress_exits,
        "stress_successors": label_stress_successors,
        "predictions": predictions,
    }


def labels_from_prediction(
    prediction: np.ndarray,
    stress_prediction: np.ndarray,
    penalty: float,
    min_edge: float,
    min_vote: float,
    cost_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(prediction, axis=0)
    uncertainty = np.std(prediction, axis=0)
    score = median - penalty * uncertainty
    score[:, 0] = 0.0
    best = np.argmax(score, axis=1).astype(np.int8)
    edge = score[np.arange(len(best)), best]
    seed_best = np.argmax(prediction, axis=2)
    vote = np.mean(seed_best == best[None, :], axis=0)
    stress_score = np.median(stress_prediction, axis=0) - penalty * np.std(stress_prediction, axis=0)
    active = (best != 0) & (edge > min_edge) & (vote >= min_vote)
    if cost_gate:
        active &= stress_score[np.arange(len(best)), best] > 0.0
    labels = np.where(active, best, 0).astype(np.int8)
    selected_uncertainty = uncertainty[np.arange(len(best)), best]
    return labels, edge, vote, selected_uncertainty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, market_hashes = core.load_market_tape()
    funding, funding_hashes = core.load_funding_tape(frame)
    feature_frames = {
        "derived11": baseline.build_causal_features(frame),
        "btc_native_stationary": load_native_features(frame),
    }
    close = frame["close"].to_numpy(dtype=np.float64)
    records: dict[str, dict[str, Any]] = {}
    fold_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    grid = list(itertools.product(TARGETS, QUANTILES, PENALTIES, MIN_EDGES, MIN_VOTES, COST_GATES))
    for feature_set in FEATURE_SETS:
        print(f"feature_set={feature_set}", flush=True)
        for fold_index in range(4):
            data = fit_fold_predictions(fold_index, frame, funding, feature_frames[feature_set])
            fold_name = data["fold"]
            for target, quantile, penalty, min_edge, min_vote, cost_gate in grid:
                key = candidate_key(feature_set, target, quantile, penalty, min_edge, min_vote, cost_gate)
                prediction = data["predictions"][(target, quantile)]
                stress_prediction = data["predictions"][("stress15_dp_advantage", quantile)]
                labels, edge, vote, uncertainty = labels_from_prediction(
                    prediction, stress_prediction, penalty, min_edge, min_vote, cost_gate
                )
                metrics, _ = core.replay_labels(
                    labels, data["rewards"], data["successors"], data["exits"],
                    data["label_events"], data["label_count"], frame, funding
                )
                stress_metrics, _ = core.replay_labels(
                    labels, data["stress_rewards"], data["stress_successors"], data["stress_exits"],
                    data["label_events"], data["label_count"], frame, funding
                )
                record = records.setdefault(key, {
                    "key": key,
                    "feature_set": feature_set,
                    "target": target,
                    "quantile": quantile,
                    "uncertainty_penalty": penalty,
                    "min_edge": min_edge,
                    "min_vote": min_vote,
                    "cost_gate": cost_gate,
                    "folds": [],
                })
                record["folds"].append({
                    "fold": fold_name,
                    "metrics": metrics,
                    "cost15_metrics": stress_metrics,
                    "label_counts": core._label_counts(labels),
                })
                lightweight_label_chart(
                    out_dir / "label_charts" / key / f"{fold_name}.png",
                    close, data["label_events"], labels,
                )
                fold_cache[(key, fold_name)] = {
                    "labels": labels,
                    "edge": edge,
                    "vote": vote,
                    "uncertainty": uncertainty,
                    "events": data["label_events"][: data["label_count"]],
                }
            print(f"[{fold_name}] evaluated {len(grid)} candidates", flush=True)

    candidates = []
    for record in records.values():
        pnls = [fold["metrics"]["pnl"] for fold in record["folds"]]
        cost_pnls = [fold["cost15_metrics"]["pnl"] for fold in record["folds"]]
        trades = sum(fold["metrics"]["trades"] for fold in record["folds"])
        record["all_fold_pnl_positive"] = bool(all(value > 0.0 for value in pnls))
        record["aggregate_cost15_pnl"] = float(sum(cost_pnls))
        record["aggregate_trades"] = int(trades)
        record["minimum_fold_pnl"] = float(min(pnls))
        record["development_gate_pass"] = bool(
            record["all_fold_pnl_positive"]
            and record["aggregate_cost15_pnl"] > 0.0
            and trades >= 40
        )
        candidates.append(record)
    eligible = [record for record in candidates if record["development_gate_pass"]]
    eligible.sort(key=lambda record: (
        record["minimum_fold_pnl"],
        record["aggregate_cost15_pnl"],
        -len(NATIVE_FEATURES) if record["feature_set"] == "btc_native_stationary" else -11,
        -int(record["cost_gate"]),
        -record["min_edge"],
    ), reverse=True)
    frozen = eligible[0] if eligible else None

    if frozen is not None:
        packs = []
        for fold_name in ("T1", "T2", "T3", "T4"):
            cached = fold_cache[(frozen["key"], fold_name)]
            event_indices = cached["events"]
            labels = cached["labels"]
            packs.append(pd.DataFrame({
                "fold": fold_name,
                "event_bar_timestamp": frame["timestamp"].iloc[event_indices].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[event_indices + 1].to_numpy(),
                "rl_action_id": labels,
                "rl_action": [core.ACTIONS[index].name for index in labels],
                "rl_predicted_advantage": cached["edge"],
                "rl_seed_vote_ratio": cached["vote"],
                "rl_uncertainty": cached["uncertainty"],
            }))
        pd.concat(packs, ignore_index=True).to_parquet(out_dir / "frozen_T1_T4_oof_label_pack.parquet", index=False)
        (out_dir / "frozen_candidate.json").write_text(
            json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8"
        )

    table = pd.DataFrame([{
        key: value for key, value in record.items() if key != "folds"
    } for record in candidates]).sort_values(
        ["development_gate_pass", "minimum_fold_pnl", "aggregate_cost15_pnl"],
        ascending=[False, False, False],
    )
    table.to_csv(out_dir / "development_candidate_table.csv", index=False)
    report = {
        "status": "development_candidate_frozen" if frozen is not None else "development_fail",
        "research_pass": False,
        "holdout_T5_T6_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "label_fold_realized_outcomes_used_to_change_labels": False,
        "feature_sets": list(FEATURE_SETS),
        "target_types": list(TARGETS),
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "frozen_candidate": frozen,
        "market_hashes": market_hashes,
        "funding_hashes": funding_hashes,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "frozen_key": None if frozen is None else frozen["key"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
