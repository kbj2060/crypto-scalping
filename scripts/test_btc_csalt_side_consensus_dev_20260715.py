#!/usr/bin/env python3
"""Run the predeclared T1-T4 side-consensus DP-label development search."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import test_btc_csalt_causal_baseline_20260715 as baseline
import test_btc_csalt_dp_advantage_dev_20260715 as dev
import test_btc_csalt_stage01_20260715 as core


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "side_consensus_dev"
TARGETS = ("dp_advantage", "stress15_dp_advantage")
QUANTILES = (0.25, 0.50)
PENALTIES = (0.0, 0.5)
MIN_EDGES = (0.0, 0.00025, 0.00050)
MIN_VOTES = (0.40, 0.60)
COST_GATES = (False, True)
LONG_ACTIONS = np.array([1, 3, 5], dtype=np.int8)
SHORT_ACTIONS = np.array([2, 4, 6], dtype=np.int8)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def key_for(
    feature_set: str, target: str, quantile: float, penalty: float,
    min_edge: float, min_vote: float, cost_gate: bool,
) -> str:
    return (
        f"{feature_set}__{target}__q{int(quantile * 100):02d}"
        f"__u{int(penalty * 10):02d}__e{int(round(min_edge * 100000)):03d}"
        f"__sv{int(min_vote * 100):02d}__c{int(cost_gate)}"
    )


def side_consensus_labels(
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
    long_local = np.argmax(score[:, LONG_ACTIONS], axis=1)
    short_local = np.argmax(score[:, SHORT_ACTIONS], axis=1)
    long_action = LONG_ACTIONS[long_local]
    short_action = SHORT_ACTIONS[short_local]
    long_score = score[np.arange(len(score)), long_action]
    short_score = score[np.arange(len(score)), short_action]
    side_score = np.column_stack([np.zeros(len(score)), long_score, short_score])
    best_side = np.argmax(side_score, axis=1).astype(np.int8)
    chosen = np.where(best_side == 1, long_action, np.where(best_side == 2, short_action, 0)).astype(np.int8)

    seed_best = np.argmax(prediction, axis=2)
    seed_side = np.take(np.array([action.side for action in core.ACTIONS]), seed_best)
    desired_side = np.where(best_side == 1, 1, np.where(best_side == 2, -1, 0))
    vote = np.mean(seed_side == desired_side[None, :], axis=0)
    edge = side_score[np.arange(len(score)), best_side]
    active = (best_side != 0) & (edge > min_edge) & (vote >= min_vote)
    if cost_gate:
        stress_score = np.median(stress_prediction, axis=0) - penalty * np.std(stress_prediction, axis=0)
        active &= stress_score[np.arange(len(score)), chosen] > 0.0
    labels = np.where(active, chosen, 0).astype(np.int8)
    selected_uncertainty = uncertainty[np.arange(len(score)), chosen]
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
        "btc_native_stationary": dev.load_native_features(frame),
    }
    close = frame["close"].to_numpy(dtype=np.float64)
    grid = list(itertools.product(TARGETS, QUANTILES, PENALTIES, MIN_EDGES, MIN_VOTES, COST_GATES))
    records: dict[str, dict[str, Any]] = {}
    cached_labels: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    fold_metric_rows: list[dict[str, Any]] = []

    for feature_set, feature_frame in feature_frames.items():
        for fold_index in range(4):
            data = dev.fit_fold_predictions(
                fold_index, frame, funding, feature_frame,
                target_names=TARGETS, quantiles=QUANTILES,
            )
            fold_name = data["fold"]
            for target, quantile, penalty, min_edge, min_vote, cost_gate in grid:
                key = key_for(feature_set, target, quantile, penalty, min_edge, min_vote, cost_gate)
                labels, edge, vote, uncertainty = side_consensus_labels(
                    data["predictions"][(target, quantile)],
                    data["predictions"][("stress15_dp_advantage", quantile)],
                    penalty, min_edge, min_vote, cost_gate,
                )
                metrics, _ = core.replay_labels(
                    labels, data["rewards"], data["successors"], data["exits"],
                    data["label_events"], data["label_count"], frame, funding,
                )
                cost_metrics, _ = core.replay_labels(
                    labels, data["stress_rewards"], data["stress_successors"], data["stress_exits"],
                    data["label_events"], data["label_count"], frame, funding,
                )
                record = records.setdefault(key, {
                    "key": key, "feature_set": feature_set, "target": target,
                    "quantile": quantile, "uncertainty_penalty": penalty,
                    "min_edge": min_edge, "min_side_vote": min_vote,
                    "cost_gate": cost_gate, "folds": [],
                })
                fold_result = {
                    "fold": fold_name, "metrics": metrics, "cost15_metrics": cost_metrics,
                    "label_counts": core._label_counts(labels),
                }
                record["folds"].append(fold_result)
                fold_metric_rows.append({
                    "key": key, "fold": fold_name, "pnl": metrics["pnl"],
                    "cost15_pnl": cost_metrics["pnl"], "trades": metrics["trades"],
                })
                dev.lightweight_label_chart(
                    out_dir / "label_charts" / key / f"{fold_name}.png",
                    close, data["label_events"], labels,
                )
                cached_labels[(key, fold_name)] = {
                    "labels": labels, "edge": edge, "vote": vote,
                    "uncertainty": uncertainty,
                    "events": data["label_events"][: data["label_count"]],
                }
            print(f"[{feature_set} {fold_name}] evaluated {len(grid)} candidates", flush=True)

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
            record["all_fold_pnl_positive"] and record["aggregate_cost15_pnl"] > 0.0 and trades >= 40
        )
        candidates.append(record)
    eligible = [record for record in candidates if record["development_gate_pass"]]
    eligible.sort(key=lambda record: (
        record["minimum_fold_pnl"], record["aggregate_cost15_pnl"],
        -31 if record["feature_set"] == "btc_native_stationary" else -11,
        -int(record["cost_gate"]), -record["min_edge"],
    ), reverse=True)
    frozen = eligible[0] if eligible else None

    if frozen is not None:
        packs = []
        for fold_name in ("T1", "T2", "T3", "T4"):
            cached = cached_labels[(frozen["key"], fold_name)]
            events = cached["events"]
            labels = cached["labels"]
            packs.append(pd.DataFrame({
                "fold": fold_name,
                "event_bar_timestamp": frame["timestamp"].iloc[events].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[events + 1].to_numpy(),
                "rl_action_id": labels,
                "rl_action": [core.ACTIONS[index].name for index in labels],
                "rl_predicted_advantage": cached["edge"],
                "rl_seed_side_vote_ratio": cached["vote"],
                "rl_uncertainty": cached["uncertainty"],
            }))
        pd.concat(packs, ignore_index=True).to_parquet(out_dir / "frozen_T1_T4_oof_label_pack.parquet", index=False)
        (out_dir / "frozen_candidate.json").write_text(
            json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8"
        )

    pd.DataFrame(fold_metric_rows).to_csv(out_dir / "candidate_fold_metrics.csv", index=False)
    pd.DataFrame([{key: value for key, value in record.items() if key != "folds"} for record in candidates]).sort_values(
        ["development_gate_pass", "minimum_fold_pnl", "aggregate_cost15_pnl"], ascending=[False, False, False]
    ).to_csv(out_dir / "development_candidate_table.csv", index=False)
    report = {
        "status": "development_candidate_frozen" if frozen else "development_fail",
        "research_pass": False,
        "holdout_T5_T6_opened": False,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "label_fold_realized_outcomes_used_to_change_labels": False,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "frozen_candidate": frozen,
        "market_hashes": market_hashes,
        "funding_hashes": funding_hashes,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "frozen_key": None if frozen is None else frozen["key"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
