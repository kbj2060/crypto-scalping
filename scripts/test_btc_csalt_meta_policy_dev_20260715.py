#!/usr/bin/env python3
"""Run the predeclared T1-T4 two-stage DP meta-policy search."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import test_btc_csalt_causal_baseline_20260715 as baseline
import test_btc_csalt_dp_advantage_dev_20260715 as dev
import test_btc_csalt_stage01_20260715 as core
import test_btc_csalt_teacher_t1_smoke_20260715 as smoke


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "meta_policy_dev"
TARGETS = ("dp_policy", "stress15_dp_policy")
MIN_ACTIVE = (0.30, 0.40, 0.50)
MIN_SIDE = (0.55, 0.65)
STRESS_GATES = (False, True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def _classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=100, min_samples_leaf=40,
        l2_regularization=1.0, learning_rate=0.05,
        early_stopping=False, random_state=seed,
    )


def fit_meta_predictions(
    train_x: np.ndarray,
    action_target: np.ndarray,
    predict_x: np.ndarray,
    bootstrap: list[np.ndarray],
) -> dict[str, np.ndarray]:
    active_result = np.empty((len(smoke.SEEDS), len(predict_x)), dtype=np.float64)
    long_result = np.empty_like(active_result)
    finite = np.isfinite(train_x).all(axis=1)
    for seed_index, (seed, sampled) in enumerate(zip(smoke.SEEDS, bootstrap)):
        selected = sampled[finite[sampled]]
        active_y = (action_target[selected] != 0).astype(np.int8)
        if len(np.unique(active_y)) != 2:
            raise RuntimeError("meta active target is not binary")
        active_model = _classifier(seed)
        active_model.fit(train_x[selected], active_y)
        active_result[seed_index] = active_model.predict_proba(predict_x)[:, 1]

        active_selected = selected[active_y == 1]
        side_y = np.array([1 if core.ACTIONS[action_target[index]].side > 0 else 0 for index in active_selected])
        classes, counts = np.unique(side_y, return_counts=True)
        if len(classes) != 2:
            raise RuntimeError("meta side target is not binary")
        class_weight = {int(cls): len(side_y) / (2 * count) for cls, count in zip(classes, counts)}
        weight = np.array([class_weight[int(value)] for value in side_y])
        side_model = _classifier(seed + 101)
        side_model.fit(train_x[active_selected], side_y, sample_weight=weight)
        long_result[seed_index] = side_model.predict_proba(predict_x)[:, 1]
    return {"active": active_result, "long": long_result}


def prepare_fold(
    fold_index: int, frame: pd.DataFrame, funding: core.FundingTape, features: pd.DataFrame,
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
    rewards, _, successors, _ = core.simulate_action_table(frame, train_events_all, train_cutoff, funding)
    stress, _, stress_successors, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding, cost_multiplier=1.5
    )
    targets = {
        "dp_policy": np.argmax(core.finite_smdp_q(train_events_all, rewards, successors)[:train_count], axis=1),
        "stress15_dp_policy": np.argmax(
            core.finite_smdp_q(train_events_all, stress, stress_successors)[:train_count], axis=1
        ),
    }
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
    predictions = {}
    for name, target in targets.items():
        print(f"[{fold.name}] fit meta {name}", flush=True)
        predictions[name] = fit_meta_predictions(train_x, target, label_x, bootstrap)
    return {
        "fold": fold.name, "label_events": label_events, "label_count": label_count,
        "rewards": label_rewards, "exits": label_exits, "successors": label_successors,
        "stress_rewards": label_stress, "stress_exits": label_stress_exits,
        "stress_successors": label_stress_successors, "predictions": predictions,
    }


def make_labels(
    prediction: dict[str, np.ndarray], stress_prediction: dict[str, np.ndarray],
    min_active: float, min_side: float, stress_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_probability = np.median(prediction["active"], axis=0)
    long_probability = np.median(prediction["long"], axis=0)
    side_probability = np.maximum(long_probability, 1.0 - long_probability)
    side = np.where(long_probability >= 0.5, 1, -1)
    active = (active_probability >= min_active) & (side_probability >= min_side)
    if stress_gate:
        stress_active = np.median(stress_prediction["active"], axis=0)
        stress_long = np.median(stress_prediction["long"], axis=0)
        stress_side = np.where(stress_long >= 0.5, 1, -1)
        active &= (stress_active >= min_active) & (stress_side == side)
    labels = np.where(active, np.where(side > 0, 1, 2), 0).astype(np.int8)
    seed_side = np.where(prediction["long"] >= 0.5, 1, -1)
    vote = np.mean(seed_side == side[None, :], axis=0)
    uncertainty = np.std(prediction["active"], axis=0)
    return labels, active_probability, vote, uncertainty


def key_for(feature: str, target: str, active: float, side: float, stress: bool) -> str:
    return f"{feature}__{target}__a{int(active*100):02d}__s{int(side*100):02d}__sg{int(stress)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, market_hashes = core.load_market_tape()
    funding, funding_hashes = core.load_funding_tape(frame)
    features = {
        "derived11": baseline.build_causal_features(frame),
        "btc_native_stationary": dev.load_native_features(frame),
    }
    close = frame["close"].to_numpy(dtype=np.float64)
    grid = list(itertools.product(TARGETS, MIN_ACTIVE, MIN_SIDE, STRESS_GATES))
    records: dict[str, dict[str, Any]] = {}
    cached = {}
    fold_rows = []
    for feature_name, feature_frame in features.items():
        for fold_index in range(4):
            data = prepare_fold(fold_index, frame, funding, feature_frame)
            for target, min_active, min_side, stress_gate in grid:
                key = key_for(feature_name, target, min_active, min_side, stress_gate)
                labels, probability, vote, uncertainty = make_labels(
                    data["predictions"][target], data["predictions"]["stress15_dp_policy"],
                    min_active, min_side, stress_gate,
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
                    "key": key, "feature_set": feature_name, "target": target,
                    "min_active_probability": min_active, "min_side_probability": min_side,
                    "stress_gate": stress_gate, "folds": [],
                })
                record["folds"].append({
                    "fold": data["fold"], "metrics": metrics, "cost15_metrics": cost_metrics,
                    "label_counts": core._label_counts(labels),
                })
                fold_rows.append({"key": key, "fold": data["fold"], "pnl": metrics["pnl"],
                                  "cost15_pnl": cost_metrics["pnl"], "trades": metrics["trades"]})
                dev.lightweight_label_chart(out_dir / "label_charts" / key / f"{data['fold']}.png",
                                            close, data["label_events"], labels)
                cached[(key, data["fold"])] = {"labels": labels, "probability": probability,
                                                "vote": vote, "uncertainty": uncertainty,
                                                "events": data["label_events"][:data["label_count"]]}
            print(f"[{feature_name} {data['fold']}] evaluated {len(grid)} candidates", flush=True)
    candidates = []
    for record in records.values():
        pnls = [fold["metrics"]["pnl"] for fold in record["folds"]]
        costs = [fold["cost15_metrics"]["pnl"] for fold in record["folds"]]
        trades = sum(fold["metrics"]["trades"] for fold in record["folds"])
        record["all_fold_pnl_positive"] = bool(all(value > 0 for value in pnls))
        record["aggregate_cost15_pnl"] = float(sum(costs))
        record["aggregate_trades"] = int(trades)
        record["minimum_fold_pnl"] = float(min(pnls))
        record["development_gate_pass"] = bool(record["all_fold_pnl_positive"] and sum(costs) > 0 and trades >= 40)
        candidates.append(record)
    eligible = [record for record in candidates if record["development_gate_pass"]]
    eligible.sort(key=lambda r: (r["minimum_fold_pnl"], r["aggregate_cost15_pnl"],
                                 -31 if r["feature_set"] == "btc_native_stationary" else -11,
                                 -int(r["stress_gate"]), -r["min_active_probability"]), reverse=True)
    frozen = eligible[0] if eligible else None
    if frozen:
        packs = []
        for fold_name in ("T1", "T2", "T3", "T4"):
            item = cached[(frozen["key"], fold_name)]; events = item["events"]; labels = item["labels"]
            packs.append(pd.DataFrame({"fold": fold_name,
                "event_bar_timestamp": frame["timestamp"].iloc[events].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[events + 1].to_numpy(),
                "rl_action_id": labels, "rl_action": [core.ACTIONS[i].name for i in labels],
                "rl_active_probability": item["probability"], "rl_seed_side_vote_ratio": item["vote"],
                "rl_uncertainty": item["uncertainty"]}))
        pd.concat(packs, ignore_index=True).to_parquet(out_dir / "frozen_T1_T4_oof_label_pack.parquet", index=False)
        (out_dir / "frozen_candidate.json").write_text(json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(out_dir / "candidate_fold_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "folds"} for r in candidates]).sort_values(
        ["development_gate_pass", "minimum_fold_pnl", "aggregate_cost15_pnl"], ascending=[False, False, False]
    ).to_csv(out_dir / "development_candidate_table.csv", index=False)
    report = {"status": "development_candidate_frozen" if frozen else "development_fail",
        "research_pass": False, "holdout_T5_T6_opened": False, "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False, "label_fold_realized_outcomes_used_to_change_labels": False,
        "candidate_count": len(candidates), "eligible_candidate_count": len(eligible),
        "frozen_candidate": frozen, "market_hashes": market_hashes, "funding_hashes": funding_hashes,
        "elapsed_seconds": time.time() - started}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidate_count": len(candidates),
                      "eligible_candidate_count": len(eligible),
                      "frozen_key": None if frozen is None else frozen["key"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
