#!/usr/bin/env python3
"""Complete CSALT Stage 1 with a purged causal dollar-event baseline."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import test_btc_csalt_stage01_20260715 as core


DEFAULT_OUT_DIR = core.DEFAULT_OUT_DIR / "causal_baseline"


@dataclass(frozen=True)
class LabelWindow:
    fold: str
    start: str
    end: str
    outcome_cutoff: str


LABEL_WINDOWS = (
    LabelWindow("T1", "2024-04-16 00:00:00", "2024-06-30 23:55:00", "2024-07-15 23:55:00"),
    LabelWindow("T2", "2024-07-16 00:00:00", "2024-09-30 23:55:00", "2024-10-15 23:55:00"),
    LabelWindow("T3", "2024-10-16 00:00:00", "2024-12-31 23:55:00", "2025-01-15 23:55:00"),
    LabelWindow("T4", "2025-01-16 00:00:00", "2025-03-31 23:55:00", "2025-04-15 23:55:00"),
    LabelWindow("T5", "2025-04-16 00:00:00", "2025-06-30 23:55:00", "2025-07-15 23:55:00"),
    LabelWindow("T6", "2025-07-16 00:00:00", "2025-08-31 23:55:00", "2025-09-15 23:55:00"),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(type(value).__name__)


def build_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    log_return = np.log(close).diff()
    features = pd.DataFrame(index=frame.index)
    for bars in (12, 48, 288):
        features[f"return_{bars}"] = close.pct_change(bars)
    features["rv_48"] = log_return.rolling(48).std()
    features["rv_288"] = log_return.rolling(288).std()
    features["atr_pct"] = frame["atr"] / close
    volume_mean = volume.rolling(288).mean()
    volume_std = volume.rolling(288).std()
    features["volume_z_288"] = (volume - volume_mean) / volume_std
    features["range_pct"] = (frame["high"] - frame["low"]) / close
    features["body_pct"] = (frame["close"] - frame["open"]) / frame["open"]
    hour = frame["timestamp"].dt.hour + frame["timestamp"].dt.minute / 60.0
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    return features


def fit_immediate_reward_baseline(
    train_features: np.ndarray,
    train_rewards: np.ndarray,
    predict_features: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros((len(predict_features), len(core.ACTIONS)), dtype=np.float64)
    prediction[:, 0] = 0.0
    for action_index in range(1, len(core.ACTIONS)):
        target = train_rewards[:, action_index]
        valid = np.isfinite(target) & np.isfinite(train_features).all(axis=1)
        if int(valid.sum()) < 300:
            raise RuntimeError(f"insufficient baseline training rows for {core.ACTIONS[action_index].name}")
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=3,
            max_iter=150,
            min_samples_leaf=40,
            l2_regularization=1.0,
            learning_rate=0.05,
            random_state=310713 + action_index,
        )
        model.fit(train_features[valid], target[valid])
        prediction[:, action_index] = model.predict(predict_features)
    return prediction


def run_window(
    window: LabelWindow,
    fold: core.FoldSpec,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    funding: core.FundingTape,
    out_dir: Path,
) -> dict[str, Any]:
    activity = (frame["close"] * frame["volume"]).to_numpy(dtype=np.float64)
    train_end = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.decision_end), side="right") - 1)
    train_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(fold.outcome_cutoff), side="right") - 1)
    threshold = core.hourly_activity_threshold(frame, train_end)
    train_events = core.build_dollar_events(activity, 0, train_cutoff, threshold)
    train_count = int(np.searchsorted(train_events, train_end, side="right"))
    train_rewards, _, _, _ = core.simulate_action_table(frame, train_events, train_cutoff, funding)

    label_start = int(frame["timestamp"].searchsorted(pd.Timestamp(window.start), side="left"))
    label_end = int(frame["timestamp"].searchsorted(pd.Timestamp(window.end), side="right") - 1)
    label_cutoff = int(frame["timestamp"].searchsorted(pd.Timestamp(window.outcome_cutoff), side="right") - 1)
    label_events = core.build_dollar_events(activity, label_start, label_cutoff, threshold)
    label_count = int(np.searchsorted(label_events, label_end, side="right"))
    rewards, exits, successors, _ = core.simulate_action_table(frame, label_events, label_cutoff, funding)

    train_x = features.iloc[train_events[:train_count]].to_numpy(dtype=np.float64)
    label_x = features.iloc[label_events[:label_count]].to_numpy(dtype=np.float64)
    if not np.isfinite(label_x).all():
        raise RuntimeError(f"{window.fold}: non-finite causal baseline features")
    predicted_q = fit_immediate_reward_baseline(train_x, train_rewards[:train_count], label_x)
    baseline_labels = np.argmax(predicted_q, axis=1).astype(np.int8)
    baseline_metrics, baseline_ledger = core.replay_labels(
        baseline_labels, rewards, successors, exits, label_events, label_count, frame, funding
    )

    oracle_q = core.finite_smdp_q(label_events, rewards, successors)
    oracle_labels = np.nanargmax(oracle_q[:label_count], axis=1).astype(np.int8)
    oracle_metrics, oracle_ledger = core.replay_labels(
        oracle_labels, rewards, successors, exits, label_events, label_count, frame, funding
    )

    diagnostic = pd.DataFrame(
        {
            "event_bar_timestamp": frame["timestamp"].iloc[label_events[:label_count]].to_numpy(),
            "baseline_action": [core.ACTIONS[index].name for index in baseline_labels],
            "oracle_action": [core.ACTIONS[index].name for index in oracle_labels],
        }
    )
    for action_index, action in enumerate(core.ACTIONS):
        diagnostic[f"predicted_reward_{action.name}"] = predicted_q[:, action_index]
        diagnostic[f"realized_reward_{action.name}"] = rewards[:label_count, action_index]
    diagnostic.to_parquet(out_dir / f"{window.fold}_baseline_diagnostic.parquet", index=False)
    baseline_ledger.to_csv(out_dir / f"{window.fold}_baseline_ledger.csv", index=False)
    oracle_ledger.to_csv(out_dir / f"{window.fold}_oracle_ledger.csv", index=False)
    core.plot_fold(
        out_dir / "label_charts" / f"B0_fold_{window.fold}.png",
        f"Causal baseline {window.fold}",
        frame,
        label_events,
        label_count,
        baseline_labels,
        predicted_q,
        rewards,
        baseline_ledger,
    )
    core.plot_fold(
        out_dir / "label_charts" / f"N3_samewindow_fold_{window.fold}.png",
        f"CSALT oracle same-window {window.fold}",
        frame,
        label_events,
        label_count,
        oracle_labels,
        oracle_q,
        rewards,
        oracle_ledger,
    )
    return {
        "fold": window.fold,
        "train_events": train_count,
        "label_events": label_count,
        "threshold": threshold,
        "baseline_metrics": baseline_metrics,
        "oracle_metrics": oracle_metrics,
        "baseline_label_counts": core._label_counts(baseline_labels),
        "oracle_label_counts": core._label_counts(oracle_labels),
        "agreement": float(np.mean(baseline_labels == oracle_labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    (out_dir / "label_charts").mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, _ = core.load_market_tape()
    funding, _ = core.load_funding_tape(frame)
    features = build_causal_features(frame)
    results = []
    for window, fold in zip(LABEL_WINDOWS, core.FOLDS):
        print(f"[{window.fold}] causal baseline and same-window oracle", flush=True)
        result = run_window(window, fold, frame, features, funding, out_dir)
        results.append(result)
        print(
            f"[{window.fold}] baseline PnL={result['baseline_metrics']['pnl']:.4f} "
            f"Calmar={result['baseline_metrics']['calmar']:.4f}; "
            f"oracle Calmar={result['oracle_metrics']['calmar']:.4f}",
            flush=True,
        )
    baseline_calmar = float(np.mean([result["baseline_metrics"]["calmar"] for result in results]))
    oracle_calmar = float(np.mean([result["oracle_metrics"]["calmar"] for result in results]))
    gate_pass = bool(np.isfinite(oracle_calmar) and oracle_calmar >= 1.25 * baseline_calmar)
    report = {
        "status": "stage1_causal_baseline_complete_non_promotion_research",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "label_fold_realized_outcomes_used_to_change_labels": False,
        "mean_baseline_calmar": baseline_calmar,
        "mean_samewindow_oracle_calmar": oracle_calmar,
        "required_oracle_calmar": 1.25 * baseline_calmar,
        "stage1_oracle_ceiling_pass": gate_pass,
        "folds": results,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "stage1_oracle_ceiling_pass": gate_pass,
        "mean_baseline_calmar": baseline_calmar,
        "mean_samewindow_oracle_calmar": oracle_calmar,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
