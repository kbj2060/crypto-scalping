#!/usr/bin/env python3
"""Run the T1 CSALT q10/q50/q90 teacher smoke test before full OOF training."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import test_btc_csalt_causal_baseline_20260715 as baseline
import test_btc_csalt_stage01_20260715 as core


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "teacher_T1_smoke"
SEEDS = (310713, 310719, 310727, 310733, 310741)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def block_bootstrap_indices(timestamps: pd.Series, seed: int) -> np.ndarray:
    day = timestamps.dt.floor("D").to_numpy()
    unique_days = np.unique(day)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(unique_days, size=len(unique_days), replace=True)
    groups = {value: np.flatnonzero(day == value) for value in unique_days}
    return np.concatenate([groups[value] for value in sampled])


def fit_quantile_predictions(
    train_x: np.ndarray,
    targets: np.ndarray,
    predict_x: np.ndarray,
    bootstrap: list[np.ndarray],
    quantile: float,
) -> np.ndarray:
    result = np.empty((len(SEEDS), len(predict_x), targets.shape[1]), dtype=np.float64)
    finite_x = np.isfinite(train_x).all(axis=1)
    for seed_index, (seed, sampled) in enumerate(zip(SEEDS, bootstrap)):
        for action_index in range(targets.shape[1]):
            target = targets[:, action_index]
            selected = sampled[finite_x[sampled] & np.isfinite(target[sampled])]
            if len(selected) < 300:
                raise RuntimeError(f"insufficient q{quantile} rows for {core.ACTIONS[action_index].name}")
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_depth=3,
                max_iter=100,
                min_samples_leaf=40,
                l2_regularization=1.0,
                learning_rate=0.05,
                random_state=seed + action_index,
            )
            model.fit(train_x[selected], target[selected])
            result[seed_index, :, action_index] = model.predict(predict_x)
    return result


def fit_quality_predictions(
    train_x: np.ndarray,
    rewards: np.ndarray,
    predict_x: np.ndarray,
    bootstrap: list[np.ndarray],
) -> np.ndarray:
    result = np.full((len(SEEDS), len(predict_x), len(core.ACTIONS)), np.nan, dtype=np.float64)
    finite_x = np.isfinite(train_x).all(axis=1)
    for seed_index, (seed, sampled) in enumerate(zip(SEEDS, bootstrap)):
        for action_index in range(1, len(core.ACTIONS)):
            target = rewards[:, action_index]
            selected = sampled[finite_x[sampled] & np.isfinite(target[sampled])]
            binary = (target[selected] > 0.0).astype(np.int8)
            if len(np.unique(binary)) < 2:
                result[seed_index, :, action_index] = float(binary[0])
                continue
            model = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=100,
                min_samples_leaf=40,
                l2_regularization=1.0,
                learning_rate=0.05,
                random_state=seed + action_index,
            )
            model.fit(train_x[selected], binary)
            result[seed_index, :, action_index] = model.predict_proba(predict_x)[:, 1]
    return result


def lcb(prediction: np.ndarray) -> np.ndarray:
    return np.median(prediction, axis=0) - np.std(prediction, axis=0)


def make_labels(
    immediate_q10: np.ndarray,
    dp_q10: np.ndarray,
    dp_q50: np.ndarray,
    stress15_q10: np.ndarray,
    stress20_q10: np.ndarray,
    quality: np.ndarray,
    train_dp_q50: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    immediate_lcb = lcb(immediate_q10)
    q10_lcb = lcb(dp_q10)
    stress15_lcb = lcb(stress15_q10)
    stress20_lcb = lcb(stress20_q10)
    q50_median = np.median(dp_q50, axis=0)
    n0 = np.argmax(immediate_lcb, axis=1).astype(np.int8)
    n1 = np.argmax(q50_median, axis=1).astype(np.int8)
    n2 = np.argmax(q10_lcb, axis=1).astype(np.int8)

    best = n2.copy()
    cash_value = q10_lcb[:, 0]
    edge = q10_lcb[np.arange(len(best)), best] - cash_value
    seed_best = np.argmax(dp_q10, axis=2)
    vote = np.mean(seed_best == best[None, :], axis=0)
    active = (
        (best != 0)
        & (edge > 0.0010)
        & (vote >= 0.80)
        & (stress15_lcb[np.arange(len(best)), best] > stress15_lcb[:, 0])
        & (stress20_lcb[np.arange(len(best)), best] >= stress20_lcb[:, 0])
    )
    n3 = np.where(active, best, 0).astype(np.int8)

    action_uncertainty = np.std(dp_q10, axis=0)
    best_active = 1 + np.argmax(q10_lcb[:, 1:], axis=1)
    active_margin = q10_lcb[np.arange(len(best)), best_active] - cash_value
    cash_margin = cash_value - np.max(q10_lcb[:, 1:], axis=1)
    label_margin = np.where(active, active_margin, np.maximum(cash_margin, 0.0005))
    selected_for_uncertainty = np.where(active, best, np.where(best == 0, 0, best_active))
    uncertainty = action_uncertainty[np.arange(len(best)), selected_for_uncertainty]
    sample_weight = np.clip(label_margin / np.maximum(uncertainty, 0.0001), 0.25, 10.0)

    train_flat = train_dp_q50[np.isfinite(train_dp_q50)]
    temperature = max(0.5 * float(np.quantile(train_flat, 0.75) - np.quantile(train_flat, 0.25)), 0.0005)
    side_value = np.column_stack(
        [cash_value, np.max(q10_lcb[:, [1, 3, 5]], axis=1), np.max(q10_lcb[:, [2, 4, 6]], axis=1)]
    )
    shifted = (side_value - np.max(side_value, axis=1, keepdims=True)) / temperature
    soft = np.exp(shifted)
    soft /= soft.sum(axis=1, keepdims=True)
    soft[~active] = np.array([1.0, 0.0, 0.0])
    quality[:, :, 0] = 0.0
    quality_median = np.median(quality, axis=0)
    selected_quality = np.full(len(best), np.nan)
    selected_quality[active] = quality_median[np.flatnonzero(active), best[active]]

    artifact = pd.DataFrame(
        {
            "n0_action_id": n0,
            "n1_action_id": n1,
            "n2_action_id": n2,
            "rl_action_id": n3,
            "rl_q10_edge": edge,
            "rl_seed_vote_ratio": vote,
            "rl_uncertainty": uncertainty,
            "rl_sample_weight": sample_weight,
            "rl_quality_score": selected_quality,
            "rl_soft_cash": soft[:, 0],
            "rl_soft_long": soft[:, 1],
            "rl_soft_short": soft[:, 2],
        }
    )
    return {"N0": n0, "N1": n1, "N2": n2, "N3": n3}, artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    (out_dir / "label_charts").mkdir(parents=True, exist_ok=True)
    started = time.time()

    frame, _ = core.load_market_tape()
    funding, _ = core.load_funding_tape(frame)
    feature_frame = baseline.build_causal_features(frame)
    fold = core.FOLDS[0]
    window = baseline.LABEL_WINDOWS[0]
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    train_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    train_events_all = core.build_dollar_events(activity, 0, train_cutoff, threshold)
    train_count = int(np.searchsorted(train_events_all, train_end, side="right"))
    train_rewards_all, _, train_successors, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding
    )
    train_rewards15_all, _, train_successors15, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding, cost_multiplier=1.5
    )
    train_rewards20_all, _, train_successors20, _ = core.simulate_action_table(
        frame, train_events_all, train_cutoff, funding, cost_multiplier=2.0
    )
    train_dp_all = core.finite_smdp_q(train_events_all, train_rewards_all, train_successors)
    train_dp15_all = core.finite_smdp_q(train_events_all, train_rewards15_all, train_successors15)
    train_dp20_all = core.finite_smdp_q(train_events_all, train_rewards20_all, train_successors20)
    train_events = train_events_all[:train_count]
    train_rewards = train_rewards_all[:train_count]
    train_dp = train_dp_all[:train_count]
    train_dp15 = train_dp15_all[:train_count]
    train_dp20 = train_dp20_all[:train_count]

    label_start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    label_end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    label_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    label_events = core.build_dollar_events(activity, label_start, label_cutoff, threshold)
    label_count = int(np.searchsorted(label_events, label_end, side="right"))
    label_rewards, label_exits, label_successors, _ = core.simulate_action_table(
        frame, label_events, label_cutoff, funding
    )
    label_rewards15, label_exits15, label_successors15, _ = core.simulate_action_table(
        frame, label_events, label_cutoff, funding, cost_multiplier=1.5
    )

    train_x = feature_frame.iloc[train_events].to_numpy(dtype=np.float64)
    label_x = feature_frame.iloc[label_events[:label_count]].to_numpy(dtype=np.float64)
    train_timestamps = frame["timestamp"].iloc[train_events].reset_index(drop=True)
    bootstrap = [block_bootstrap_indices(train_timestamps, seed) for seed in SEEDS]
    print("fitting N0 immediate q10", flush=True)
    immediate_q10 = fit_quantile_predictions(train_x, train_rewards, label_x, bootstrap, 0.10)
    print("fitting DP q10/q50/q90", flush=True)
    dp_q10 = fit_quantile_predictions(train_x, train_dp, label_x, bootstrap, 0.10)
    dp_q50 = fit_quantile_predictions(train_x, train_dp, label_x, bootstrap, 0.50)
    dp_q90 = fit_quantile_predictions(train_x, train_dp, label_x, bootstrap, 0.90)
    print("fitting cost-stress q10", flush=True)
    stress15_q10 = fit_quantile_predictions(train_x, train_dp15, label_x, bootstrap, 0.10)
    stress20_q10 = fit_quantile_predictions(train_x, train_dp20, label_x, bootstrap, 0.10)
    quality = fit_quality_predictions(train_x, train_rewards, label_x, bootstrap)
    labels, artifact = make_labels(
        immediate_q10, dp_q10, dp_q50, stress15_q10, stress20_q10, quality, train_dp
    )
    artifact.insert(0, "entry_available_timestamp", frame["timestamp"].iloc[label_events[:label_count] + 1].to_numpy())
    artifact.insert(0, "event_bar_timestamp", frame["timestamp"].iloc[label_events[:label_count]].to_numpy())
    for name in ("N0", "N1", "N2", "N3"):
        artifact[f"{name.lower()}_action"] = [core.ACTIONS[index].name for index in labels[name]]
    artifact.to_parquet(out_dir / "T1_oof_rl_label_pack.parquet", index=False)
    np.savez_compressed(
        out_dir / "T1_teacher_predictions.npz",
        immediate_q10=immediate_q10,
        dp_q10=dp_q10,
        dp_q50=dp_q50,
        dp_q90=dp_q90,
        stress15_q10=stress15_q10,
        stress20_q10=stress20_q10,
        quality=quality,
    )

    family_results: dict[str, Any] = {}
    q_for_chart = {
        "N0": lcb(immediate_q10),
        "N1": np.median(dp_q50, axis=0),
        "N2": lcb(dp_q10),
        "N3": lcb(dp_q10),
    }
    for name, action_labels in labels.items():
        metrics, ledger = core.replay_labels(
            action_labels, label_rewards, label_successors, label_exits, label_events, label_count, frame, funding
        )
        metrics15, _ = core.replay_labels(
            action_labels, label_rewards15, label_successors15, label_exits15, label_events, label_count, frame, funding
        )
        ledger.to_csv(out_dir / f"{name}_T1_diagnostic_ledger.csv", index=False)
        core.plot_fold(
            out_dir / "label_charts" / f"{name}_fold_T1.png",
            f"{name} frozen teacher T1",
            frame,
            label_events,
            label_count,
            action_labels,
            q_for_chart[name],
            label_rewards,
            ledger,
        )
        family_results[name] = {
            "metrics": metrics,
            "cost15_metrics": metrics15,
            "label_counts": core._label_counts(action_labels),
        }
        print(f"{name}: PnL={metrics['pnl']:.4f} MTM_MDD={metrics['mtm_mdd']:.4f} trades={metrics['trades']}", flush=True)

    n0 = family_results["N0"]["metrics"]
    n3 = family_results["N3"]["metrics"]
    smoke_pass = bool(
        n3["pnl"] > 0.0
        and n3["pnl"] > n0["pnl"]
        and n3["trades"] >= 20
        and family_results["N3"]["cost15_metrics"]["pnl"] > 0.0
    )
    report = {
        "status": "T1_teacher_smoke_complete_non_promotion_research",
        "teacher_fit_end": fold.decision_end,
        "label_start": window.start,
        "teacher_predictions_are_purged_oof": True,
        "label_fold_realized_outcomes_used_to_change_labels": False,
        "seeds": list(SEEDS),
        "train_events": len(train_events),
        "label_events": label_count,
        "families": family_results,
        "T1_teacher_smoke_pass": smoke_pass,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"T1_teacher_smoke_pass": smoke_pass}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
