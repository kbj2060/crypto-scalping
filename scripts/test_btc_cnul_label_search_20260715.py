#!/usr/bin/env python3
"""Search and one-shot test non-DP counterfactual net-utility labels."""
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
import test_btc_csalt_policy_distillation_dev_20260715 as policy
import test_btc_csalt_stage01_20260715 as core
import test_btc_csalt_teacher_t1_smoke_20260715 as smoke


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "cnul_label_search"
TARGET_MODES = ("net1", "net15", "consensus")
PROFIT_FLOORS = (0.0, 0.0025)
MIN_PROBS = (0.50, 0.60, 0.70)
MIN_VOTES = (0.40, 0.60)
COST_GATES = (False, True)
FEATURE_SETS = ("derived11", "btc_native_stationary")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def target_key(mode: str, floor: float) -> str:
    return f"{mode}_f{int(round(floor * 10000)):03d}"


def build_targets(rewards: np.ndarray, stress: np.ndarray) -> dict[str, np.ndarray]:
    normal_action = np.argmax(rewards, axis=1).astype(np.int8)
    stress_action = np.argmax(stress, axis=1).astype(np.int8)
    rows = np.arange(len(rewards))
    targets: dict[str, np.ndarray] = {}
    for floor in PROFIT_FLOORS:
        normal = normal_action.copy()
        normal[rewards[rows, normal] <= floor] = 0
        targets[target_key("net1", floor)] = normal

        stressed = stress_action.copy()
        stressed[stress[rows, stressed] <= floor] = 0
        targets[target_key("net15", floor)] = stressed

        consensus = np.where(normal_action == stress_action, normal_action, 0).astype(np.int8)
        consensus_utility = stress[rows, consensus]
        consensus[consensus_utility <= floor] = 0
        targets[target_key("consensus", floor)] = consensus
    return targets


def prepare_fold(
    fold_index: int,
    frame: pd.DataFrame,
    funding: core.FundingTape,
    features: pd.DataFrame,
) -> dict[str, Any]:
    fold = core.FOLDS[fold_index]
    window = baseline.LABEL_WINDOWS[fold_index]
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    train_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    train_all = core.build_dollar_events(activity, 0, train_cutoff, threshold)
    train_count = int(np.searchsorted(train_all, train_end, side="right"))
    train_events = train_all[:train_count]
    train_rewards, _, _, _ = core.simulate_action_table(frame, train_all, train_cutoff, funding)
    train_stress, _, _, _ = core.simulate_action_table(
        frame, train_all, train_cutoff, funding, cost_multiplier=1.5
    )
    targets = build_targets(train_rewards[:train_count], train_stress[:train_count])

    label_start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    label_end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    label_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    label_events = core.build_dollar_events(activity, label_start, label_cutoff, threshold)
    label_count = int(np.searchsorted(label_events, label_end, side="right"))
    label_rewards, label_exits, label_successors, _ = core.simulate_action_table(
        frame, label_events, label_cutoff, funding
    )
    label_stress, stress_exits, stress_successors, _ = core.simulate_action_table(
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
    predictions = {}
    for name, target in targets.items():
        print(f"[{fold.name}] fit CNUL {name}", flush=True)
        predictions[name] = policy.fit_policy_predictions(train_x, target, label_x, bootstrap)
    return {
        "fold": fold.name,
        "events": label_events,
        "count": label_count,
        "rewards": label_rewards,
        "exits": label_exits,
        "successors": label_successors,
        "stress": label_stress,
        "stress_exits": stress_exits,
        "stress_successors": stress_successors,
        "predictions": predictions,
        "target_counts": {name: core._label_counts(value) for name, value in targets.items()},
    }


def candidate_key(
    feature: str, mode: str, floor: float, probability: float, vote: float, cost_gate: bool,
) -> str:
    return (
        f"{feature}__{target_key(mode, floor)}__p{int(probability*100):02d}"
        f"__sv{int(vote*100):02d}__cg{int(cost_gate)}"
    )


def evaluate_labels(
    prediction: np.ndarray,
    stress_prediction: np.ndarray,
    min_probability: float,
    min_vote: float,
    cost_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return policy.policy_labels(
        prediction,
        stress_prediction,
        min_probability,
        0.0,
        min_vote,
        cost_gate,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allow-pnl-fallback", action="store_true")
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
    grid = list(itertools.product(TARGET_MODES, PROFIT_FLOORS, MIN_PROBS, MIN_VOTES, COST_GATES))
    records: dict[str, dict[str, Any]] = {}
    cached: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    fold_rows: list[dict[str, Any]] = []
    target_counts: dict[str, Any] = {}

    for feature_name in FEATURE_SETS:
        for fold_index in range(4):
            data = prepare_fold(fold_index, frame, funding, feature_frames[feature_name])
            target_counts[f"{feature_name}_{data['fold']}"] = data["target_counts"]
            for mode, floor, min_probability, min_vote, cost_gate in grid:
                key = candidate_key(feature_name, mode, floor, min_probability, min_vote, cost_gate)
                name = target_key(mode, floor)
                stress_name = target_key("net15", floor)
                labels, probability, vote, uncertainty = evaluate_labels(
                    data["predictions"][name], data["predictions"][stress_name],
                    min_probability, min_vote, cost_gate,
                )
                metrics, _ = core.replay_labels(
                    labels, data["rewards"], data["successors"], data["exits"],
                    data["events"], data["count"], frame, funding,
                )
                cost_metrics, _ = core.replay_labels(
                    labels, data["stress"], data["stress_successors"], data["stress_exits"],
                    data["events"], data["count"], frame, funding,
                )
                record = records.setdefault(key, {
                    "key": key, "feature_set": feature_name, "target_mode": mode,
                    "profit_floor": floor, "min_probability": min_probability,
                    "min_side_vote": min_vote, "cost_gate": cost_gate, "folds": [],
                })
                record["folds"].append({
                    "fold": data["fold"], "metrics": metrics, "cost15_metrics": cost_metrics,
                    "label_counts": core._label_counts(labels),
                })
                fold_rows.append({
                    "key": key, "fold": data["fold"], "pnl": metrics["pnl"],
                    "cost15_pnl": cost_metrics["pnl"], "trades": metrics["trades"],
                })
                dev.lightweight_label_chart(
                    out_dir / "label_charts" / key / f"{data['fold']}.png",
                    close, data["events"], labels,
                )
                cached[(key, data["fold"])] = {
                    "labels": labels, "probability": probability, "vote": vote,
                    "uncertainty": uncertainty, "events": data["events"][:data["count"]],
                }
            print(f"[{feature_name} {data['fold']}] evaluated {len(grid)} candidates", flush=True)

    candidates = []
    for record in records.values():
        pnls = [fold["metrics"]["pnl"] for fold in record["folds"]]
        cost_pnls = [fold["cost15_metrics"]["pnl"] for fold in record["folds"]]
        trades = sum(fold["metrics"]["trades"] for fold in record["folds"])
        record["all_fold_pnl_positive"] = bool(all(value > 0.0 for value in pnls))
        record["positive_fold_count"] = int(sum(value > 0.0 for value in pnls))
        record["aggregate_pnl"] = float(sum(pnls))
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
        -int(record["cost_gate"]), -record["profit_floor"], -record["min_probability"],
    ), reverse=True)
    frozen = eligible[0] if eligible else None
    selection_rule = "strict_all_four_folds_positive"
    if frozen is None and args.allow_pnl_fallback:
        fallback = [
            record for record in candidates
            if record["positive_fold_count"] >= 3
            and record["minimum_fold_pnl"] > -0.02
            and record["aggregate_trades"] >= 40
            and record["aggregate_cost15_pnl"] > 0.0
        ]
        fallback.sort(key=lambda record: (
            record["aggregate_cost15_pnl"], record["minimum_fold_pnl"],
            -31 if record["feature_set"] == "btc_native_stationary" else -11,
            -int(record["cost_gate"]), -record["profit_floor"],
            -record["min_probability"], -record["min_side_vote"],
        ), reverse=True)
        frozen = fallback[0] if fallback else None
        selection_rule = "pre_holdout_pnl_fallback_3of4_worst_gt_minus2pct"

    holdout_results = []
    if frozen is not None:
        for fold_index in (4, 5):
            data = prepare_fold(fold_index, frame, funding, feature_frames[frozen["feature_set"]])
            name = target_key(frozen["target_mode"], frozen["profit_floor"])
            stress_name = target_key("net15", frozen["profit_floor"])
            labels, probability, vote, uncertainty = evaluate_labels(
                data["predictions"][name], data["predictions"][stress_name],
                frozen["min_probability"], frozen["min_side_vote"], frozen["cost_gate"],
            )
            metrics, ledger = core.replay_labels(
                labels, data["rewards"], data["successors"], data["exits"],
                data["events"], data["count"], frame, funding,
            )
            cost_metrics, _ = core.replay_labels(
                labels, data["stress"], data["stress_successors"], data["stress_exits"],
                data["events"], data["count"], frame, funding,
            )
            ledger.to_csv(out_dir / f"{data['fold']}_diagnostic_ledger.csv", index=False)
            dev.lightweight_label_chart(
                out_dir / "label_charts" / frozen["key"] / f"{data['fold']}.png",
                close, data["events"], labels,
            )
            holdout_results.append({
                "fold": data["fold"], "metrics": metrics, "cost15_metrics": cost_metrics,
                "label_counts": core._label_counts(labels),
            })
            cached[(frozen["key"], data["fold"])] = {
                "labels": labels, "probability": probability, "vote": vote,
                "uncertainty": uncertainty, "events": data["events"][:data["count"]],
            }

        all_folds = ("T1", "T2", "T3", "T4", "T5", "T6")
        packs = []
        for fold_name in all_folds:
            item = cached[(frozen["key"], fold_name)]
            events, labels = item["events"], item["labels"]
            packs.append(pd.DataFrame({
                "fold": fold_name,
                "event_bar_timestamp": frame["timestamp"].iloc[events].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[events + 1].to_numpy(),
                "cnul_action_id": labels,
                "cnul_action": [core.ACTIONS[index].name for index in labels],
                "cnul_side_probability": item["probability"],
                "cnul_seed_side_vote_ratio": item["vote"],
                "cnul_uncertainty": item["uncertainty"],
            }))
        pd.concat(packs, ignore_index=True).to_parquet(out_dir / "frozen_T1_T6_oof_label_pack.parquet", index=False)
        (out_dir / "frozen_candidate.json").write_text(
            json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8"
        )

    holdout_pass = bool(
        len(holdout_results) == 2
        and all(result["metrics"]["pnl"] > 0.0 for result in holdout_results)
        and sum(result["metrics"]["trades"] for result in holdout_results) >= 20
        and sum(result["cost15_metrics"]["pnl"] for result in holdout_results) > 0.0
    )
    pd.DataFrame(fold_rows).to_csv(out_dir / "candidate_fold_metrics.csv", index=False)
    pd.DataFrame([{key: value for key, value in record.items() if key != "folds"} for record in candidates]).sort_values(
        ["development_gate_pass", "minimum_fold_pnl", "aggregate_cost15_pnl"],
        ascending=[False, False, False],
    ).to_csv(out_dir / "development_candidate_table.csv", index=False)
    report = {
        "status": "research_pass" if holdout_pass else ("holdout_fail" if frozen else "development_fail"),
        "research_pass": holdout_pass,
        "holdout_T5_T6_opened": frozen is not None,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "label_fold_realized_outcomes_used_to_change_labels": False,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selection_rule": selection_rule,
        "frozen_candidate": frozen,
        "holdout_results": holdout_results,
        "training_target_counts": target_counts,
        "market_hashes": market_hashes,
        "funding_hashes": funding_hashes,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "frozen_key": None if frozen is None else frozen["key"],
        "holdout_results": holdout_results,
    }, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
