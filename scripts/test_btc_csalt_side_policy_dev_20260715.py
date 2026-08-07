#!/usr/bin/env python3
"""Run the predeclared T1-T4 balanced CASH/LONG/SHORT DP-policy search."""
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


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "side_policy_dev"
TARGETS = ("dp_side", "stress15_dp_side")
MIN_PROBS = (0.50, 0.60, 0.70)
MIN_MARGINS = (0.00, 0.10)
STRESS_GATES = (False, True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def action_to_side_class(actions: np.ndarray) -> np.ndarray:
    return np.array([0 if core.ACTIONS[value].side == 0 else (1 if core.ACTIONS[value].side > 0 else 2)
                     for value in actions], dtype=np.int8)


def fit_side_predictions(
    train_x: np.ndarray, target: np.ndarray, predict_x: np.ndarray, bootstrap: list[np.ndarray],
) -> np.ndarray:
    result = np.zeros((len(smoke.SEEDS), len(predict_x), 3), dtype=np.float64)
    finite = np.isfinite(train_x).all(axis=1)
    for seed_index, (seed, sampled) in enumerate(zip(smoke.SEEDS, bootstrap)):
        selected = sampled[finite[sampled]]; y = target[selected]
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 3:
            raise RuntimeError("side policy target does not contain all three classes")
        weights = {int(cls): len(y) / (3 * count) for cls, count in zip(classes, counts)}
        sample_weight = np.clip(np.array([weights[int(value)] for value in y]), 0.25, 10.0)
        model = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, min_samples_leaf=40, l2_regularization=1.0,
            learning_rate=0.05, early_stopping=False, random_state=seed,
        )
        model.fit(train_x[selected], y, sample_weight=sample_weight)
        result[seed_index][:, model.classes_.astype(int)] = model.predict_proba(predict_x)
    return result


def prepare_fold(index: int, frame: pd.DataFrame, funding: core.FundingTape,
                 features: pd.DataFrame) -> dict[str, Any]:
    fold = core.FOLDS[index]; window = baseline.LABEL_WINDOWS[index]
    activity = (frame["close"] * frame["volume"]).to_numpy(float)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    train_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    train_all = core.build_dollar_events(activity, 0, train_cutoff, threshold)
    train_count = int(np.searchsorted(train_all, train_end, side="right")); train = train_all[:train_count]
    rewards, _, successors, _ = core.simulate_action_table(frame, train_all, train_cutoff, funding)
    stress, _, stress_successors, _ = core.simulate_action_table(frame, train_all, train_cutoff, funding,
                                                                  cost_multiplier=1.5)
    targets = {
        "dp_side": action_to_side_class(np.argmax(core.finite_smdp_q(train_all, rewards, successors)[:train_count], axis=1)),
        "stress15_dp_side": action_to_side_class(np.argmax(
            core.finite_smdp_q(train_all, stress, stress_successors)[:train_count], axis=1)),
    }
    start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    events = core.build_dollar_events(activity, start, cutoff, threshold)
    count = int(np.searchsorted(events, end, side="right"))
    label_rewards, exits, label_successors, _ = core.simulate_action_table(frame, events, cutoff, funding)
    label_stress, stress_exits, label_stress_successors, _ = core.simulate_action_table(
        frame, events, cutoff, funding, cost_multiplier=1.5)
    train_x = features.iloc[train].to_numpy(float); label_x = features.iloc[events[:count]].to_numpy(float)
    if not np.isfinite(label_x).all(): raise RuntimeError(f"{fold.name}: non-finite features")
    bootstrap = [smoke.block_bootstrap_indices(frame["timestamp"].iloc[train].reset_index(drop=True), seed)
                 for seed in smoke.SEEDS]
    predictions = {}
    for name, target in targets.items():
        print(f"[{fold.name}] fit {name}", flush=True)
        predictions[name] = fit_side_predictions(train_x, target, label_x, bootstrap)
    return {"fold": fold.name, "events": events, "count": count, "rewards": label_rewards,
            "exits": exits, "successors": label_successors, "stress": label_stress,
            "stress_exits": stress_exits, "stress_successors": label_stress_successors,
            "predictions": predictions}


def make_labels(prediction: np.ndarray, stress_prediction: np.ndarray, min_prob: float,
                min_margin: float, stress_gate: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(prediction, axis=0); best = np.argmax(median, axis=1).astype(np.int8)
    probability = median[np.arange(len(best)), best]; margin = probability - median[:, 0]
    active = (best != 0) & (probability >= min_prob) & (margin >= min_margin)
    if stress_gate:
        stress_best = np.argmax(np.median(stress_prediction, axis=0), axis=1).astype(np.int8)
        active &= stress_best == best
    labels = np.where(active, np.where(best == 1, 1, 2), 0).astype(np.int8)
    uncertainty = np.std(prediction, axis=0)[np.arange(len(best)), best]
    return labels, probability, uncertainty


def key_for(feature: str, target: str, probability: float, margin: float, stress: bool) -> str:
    return f"{feature}__{target}__p{int(probability*100):02d}__m{int(margin*100):02d}__sg{int(stress)}"


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(); out_dir = args.out_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time(); frame, market_hashes = core.load_market_tape(); funding, funding_hashes = core.load_funding_tape(frame)
    feature_frames = {"derived11": baseline.build_causal_features(frame),
                      "btc_native_stationary": dev.load_native_features(frame)}
    close = frame["close"].to_numpy(float); grid = list(itertools.product(TARGETS, MIN_PROBS, MIN_MARGINS, STRESS_GATES))
    records = {}; cached = {}; fold_rows = []
    for feature_name, features in feature_frames.items():
        for index in range(4):
            data = prepare_fold(index, frame, funding, features)
            for target, min_prob, min_margin, stress_gate in grid:
                key = key_for(feature_name, target, min_prob, min_margin, stress_gate)
                labels, probability, uncertainty = make_labels(data["predictions"][target],
                    data["predictions"]["stress15_dp_side"], min_prob, min_margin, stress_gate)
                metrics, _ = core.replay_labels(labels, data["rewards"], data["successors"], data["exits"],
                                                 data["events"], data["count"], frame, funding)
                costs, _ = core.replay_labels(labels, data["stress"], data["stress_successors"], data["stress_exits"],
                                               data["events"], data["count"], frame, funding)
                record = records.setdefault(key, {"key": key, "feature_set": feature_name, "target": target,
                    "min_probability": min_prob, "min_margin": min_margin, "stress_gate": stress_gate, "folds": []})
                record["folds"].append({"fold": data["fold"], "metrics": metrics, "cost15_metrics": costs,
                                         "label_counts": core._label_counts(labels)})
                fold_rows.append({"key": key, "fold": data["fold"], "pnl": metrics["pnl"],
                                  "cost15_pnl": costs["pnl"], "trades": metrics["trades"]})
                dev.lightweight_label_chart(out_dir / "label_charts" / key / f"{data['fold']}.png",
                                            close, data["events"], labels)
                cached[(key, data["fold"])] = {"labels": labels, "probability": probability,
                                                "uncertainty": uncertainty, "events": data["events"][:data["count"]]}
            print(f"[{feature_name} {data['fold']}] evaluated {len(grid)} candidates", flush=True)
    candidates = []
    for record in records.values():
        pnls = [f["metrics"]["pnl"] for f in record["folds"]]; costs = [f["cost15_metrics"]["pnl"] for f in record["folds"]]
        trades = sum(f["metrics"]["trades"] for f in record["folds"])
        record["all_fold_pnl_positive"] = bool(all(v > 0 for v in pnls)); record["aggregate_cost15_pnl"] = float(sum(costs))
        record["aggregate_trades"] = int(trades); record["minimum_fold_pnl"] = float(min(pnls))
        record["development_gate_pass"] = bool(record["all_fold_pnl_positive"] and sum(costs) > 0 and trades >= 40)
        candidates.append(record)
    eligible = [r for r in candidates if r["development_gate_pass"]]
    eligible.sort(key=lambda r: (r["minimum_fold_pnl"], r["aggregate_cost15_pnl"],
                                 -31 if r["feature_set"] == "btc_native_stationary" else -11,
                                 -int(r["stress_gate"]), -r["min_probability"]), reverse=True)
    frozen = eligible[0] if eligible else None
    if frozen:
        packs = []
        for fold_name in ("T1", "T2", "T3", "T4"):
            item = cached[(frozen["key"], fold_name)]; events = item["events"]; labels = item["labels"]
            packs.append(pd.DataFrame({"fold": fold_name, "event_bar_timestamp": frame["timestamp"].iloc[events].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[events+1].to_numpy(), "rl_action_id": labels,
                "rl_action": [core.ACTIONS[i].name for i in labels], "rl_side_probability": item["probability"],
                "rl_uncertainty": item["uncertainty"]}))
        pd.concat(packs, ignore_index=True).to_parquet(out_dir / "frozen_T1_T4_oof_label_pack.parquet", index=False)
        (out_dir / "frozen_candidate.json").write_text(json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(out_dir / "candidate_fold_metrics.csv", index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k != "folds"} for r in candidates]).sort_values(
        ["development_gate_pass","minimum_fold_pnl","aggregate_cost15_pnl"], ascending=[False,False,False]
    ).to_csv(out_dir / "development_candidate_table.csv", index=False)
    report = {"status": "development_candidate_frozen" if frozen else "development_fail", "research_pass": False,
        "holdout_T5_T6_opened": False, "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "label_fold_realized_outcomes_used_to_change_labels": False, "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible), "frozen_candidate": frozen, "market_hashes": market_hashes,
        "funding_hashes": funding_hashes, "elapsed_seconds": time.time()-started}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidate_count": len(candidates),
                      "eligible_candidate_count": len(eligible),
                      "frozen_key": None if frozen is None else frozen["key"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
