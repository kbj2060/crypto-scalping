#!/usr/bin/env python3
"""Evaluate the predeclared causal-rank gate on frozen T1-T4 OOF probabilities."""
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


SOURCE_DIR = core.DEFAULT_OUT_DIR / "policy_distillation_dev/raw_oof_predictions"
DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "causal_rank_dev"
LOOKBACKS = (288, 576)
QUANTILES = (0.80, 0.85, 0.90)
MIN_SIDE_PROBS = (0.40, 0.50)
STRESS_GATES = (False, True)
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


def load_fold_evaluation(
    fold_index: int, frame: pd.DataFrame, funding: core.FundingTape,
) -> dict[str, Any]:
    fold = core.FOLDS[fold_index]
    window = baseline.LABEL_WINDOWS[fold_index]
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    label_start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    label_end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    label_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    events = core.build_dollar_events(activity, label_start, label_cutoff, threshold)
    count = int(np.searchsorted(events, label_end, side="right"))
    rewards, exits, successors, _ = core.simulate_action_table(frame, events, label_cutoff, funding)
    stress, stress_exits, stress_successors, _ = core.simulate_action_table(
        frame, events, label_cutoff, funding, cost_multiplier=1.5
    )
    return {"fold": fold.name, "events": events, "count": count, "rewards": rewards,
            "exits": exits, "successors": successors, "stress": stress,
            "stress_exits": stress_exits, "stress_successors": stress_successors}


def make_labels(
    prediction: np.ndarray, stress_prediction: np.ndarray, lookback: int,
    quantile: float, min_side_probability: float, stress_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(prediction, axis=0)
    stress_median = np.median(stress_prediction, axis=0)
    long_probability = median[:, LONG_ACTIONS].sum(axis=1)
    short_probability = median[:, SHORT_ACTIONS].sum(axis=1)
    side_probability = np.maximum(long_probability, short_probability)
    side = np.where(long_probability >= short_probability, 1, -1)
    long_action = LONG_ACTIONS[np.argmax(median[:, LONG_ACTIONS], axis=1)]
    short_action = SHORT_ACTIONS[np.argmax(median[:, SHORT_ACTIONS], axis=1)]
    chosen = np.where(side > 0, long_action, short_action).astype(np.int8)
    threshold = np.full(len(side_probability), np.nan, dtype=np.float64)
    active = np.zeros(len(side_probability), dtype=bool)
    for index in range(100, len(side_probability)):
        history = side_probability[max(0, index - lookback):index]
        threshold[index] = float(np.quantile(history, quantile))
        active[index] = side_probability[index] >= max(threshold[index], min_side_probability)
    if stress_gate:
        stress_long = stress_median[:, LONG_ACTIONS].sum(axis=1)
        stress_short = stress_median[:, SHORT_ACTIONS].sum(axis=1)
        stress_side = np.where(stress_long >= stress_short, 1, -1)
        active &= stress_side == side
    labels = np.where(active, chosen, 0).astype(np.int8)
    return labels, side_probability, threshold


def key_for(lookback: int, quantile: float, probability: float, stress: bool) -> str:
    return f"native_dp_policy__lb{lookback}__q{int(quantile*100):02d}__p{int(probability*100):02d}__sg{int(stress)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    source_dir, out_dir = args.source_dir.resolve(), args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, market_hashes = core.load_market_tape()
    funding, funding_hashes = core.load_funding_tape(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    grid = list(itertools.product(LOOKBACKS, QUANTILES, MIN_SIDE_PROBS, STRESS_GATES))
    records = {}
    cached = {}
    fold_rows = []
    for fold_index in range(4):
        data = load_fold_evaluation(fold_index, frame, funding)
        source = np.load(source_dir / f"btc_native_stationary_{data['fold']}.npz")
        if not np.array_equal(source["label_events"], data["events"][:data["count"]]):
            raise RuntimeError(f"{data['fold']}: raw prediction event contract mismatch")
        for lookback, quantile, probability, stress_gate in grid:
            key = key_for(lookback, quantile, probability, stress_gate)
            labels, score, threshold = make_labels(
                source["dp_policy"], source["stress15_dp_policy"],
                lookback, quantile, probability, stress_gate,
            )
            metrics, _ = core.replay_labels(labels, data["rewards"], data["successors"], data["exits"],
                                             data["events"], data["count"], frame, funding)
            cost_metrics, _ = core.replay_labels(labels, data["stress"], data["stress_successors"],
                                                  data["stress_exits"], data["events"], data["count"],
                                                  frame, funding)
            record = records.setdefault(key, {"key": key, "lookback": lookback,
                "quantile": quantile, "min_side_probability": probability,
                "stress_gate": stress_gate, "folds": []})
            record["folds"].append({"fold": data["fold"], "metrics": metrics,
                                     "cost15_metrics": cost_metrics,
                                     "label_counts": core._label_counts(labels)})
            fold_rows.append({"key": key, "fold": data["fold"], "pnl": metrics["pnl"],
                              "cost15_pnl": cost_metrics["pnl"], "trades": metrics["trades"]})
            dev.lightweight_label_chart(out_dir / "label_charts" / key / f"{data['fold']}.png",
                                        close, data["events"], labels)
            cached[(key, data["fold"])] = {"labels": labels, "score": score,
                                            "threshold": threshold,
                                            "events": data["events"][:data["count"]]}
        print(f"[{data['fold']}] evaluated {len(grid)} rank candidates", flush=True)
    candidates = []
    for record in records.values():
        pnls = [fold["metrics"]["pnl"] for fold in record["folds"]]
        costs = [fold["cost15_metrics"]["pnl"] for fold in record["folds"]]
        trades = sum(fold["metrics"]["trades"] for fold in record["folds"])
        record["all_fold_pnl_positive"] = bool(all(value > 0 for value in pnls))
        record["aggregate_cost15_pnl"] = float(sum(costs)); record["aggregate_trades"] = int(trades)
        record["minimum_fold_pnl"] = float(min(pnls))
        record["development_gate_pass"] = bool(record["all_fold_pnl_positive"] and sum(costs) > 0 and trades >= 40)
        candidates.append(record)
    eligible = [record for record in candidates if record["development_gate_pass"]]
    eligible.sort(key=lambda r: (r["minimum_fold_pnl"], r["aggregate_cost15_pnl"],
                                 -r["lookback"], -int(r["stress_gate"]), -r["quantile"]), reverse=True)
    frozen = eligible[0] if eligible else None
    if frozen:
        packs = []
        for fold_name in ("T1", "T2", "T3", "T4"):
            item = cached[(frozen["key"], fold_name)]; events = item["events"]; labels = item["labels"]
            packs.append(pd.DataFrame({"fold": fold_name,
                "event_bar_timestamp": frame["timestamp"].iloc[events].to_numpy(),
                "entry_available_timestamp": frame["timestamp"].iloc[events + 1].to_numpy(),
                "rl_action_id": labels, "rl_action": [core.ACTIONS[i].name for i in labels],
                "rl_side_probability": item["score"], "rl_causal_rank_threshold": item["threshold"]}))
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
