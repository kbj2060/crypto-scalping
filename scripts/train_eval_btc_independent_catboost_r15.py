#!/usr/bin/env python3
"""Independent CatBoost direction and quality heads for the BTC 5-minute policy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import MIN_ACTION_EVENTS, TRAIN_DATA, VAL_DATA, labels_for, simulate  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_independent_catboost_r15"
DEV_END = "2025-08-31 23:59:59+00:00"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
HALF_LIFE_DAYS = 180.0
QUALITY_TEMPERATURE = .002
ENTRY_THRESHOLDS = (.40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90)
LARGE_THRESHOLDS = (.65, .70, .75, .80, .85, .90, .95)


def feature_sets() -> tuple[list[str], list[str]]:
    selected = json.loads(SELECTION.read_text())
    direction, quality = selected["direction_features"], selected["action_features"]
    forbidden = [name for name in [*direction, *quality] if "regime4" in name.lower()]
    if forbidden:
        raise ValueError(f"Regime4 features are forbidden: {forbidden}")
    return direction, quality


def target_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    utilities = frame[[
        "teacher_action_-0.30_utility", "teacher_action_-0.15_utility",
        "teacher_action_+0.00_utility", "teacher_action_+0.15_utility",
        "teacher_action_+0.30_utility",
    ]].to_numpy(dtype=np.float32)
    short_utility = utilities[:, :2].max(axis=1)
    flat_utility = utilities[:, 2]
    long_utility = utilities[:, 3:].max(axis=1)
    direction = long_utility - short_utility
    quality = 1.0 / (1.0 + np.exp(-((np.maximum(short_utility, long_utility) - flat_utility) / QUALITY_TEMPERATURE)))
    return direction.astype(np.float32), quality.astype(np.float32)


def values(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[[*features, "current_margin_fraction"]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def recency_weights(frame: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_datetime(frame["decision_timestamp"], utc=True)
    age_days = (timestamps.max() - timestamps).dt.total_seconds().to_numpy(dtype=np.float32) / 86400.0
    weights = np.exp2(-age_days / HALF_LIFE_DAYS).astype(np.float32)
    return weights / weights.mean()


def fit(train: pd.DataFrame, direction_features: list[str], quality_features: list[str]) -> tuple[CatBoostRegressor, CatBoostRegressor, dict[str, float]]:
    direction_target, quality_target = target_values(train)
    weights = recency_weights(train)
    params = {"iterations": 400, "depth": 7, "learning_rate": .05, "loss_function": "RMSE", "random_seed": 270705, "verbose": False, "thread_count": -1}
    direction_model = CatBoostRegressor(**params).fit(values(train, direction_features), direction_target, sample_weight=weights)
    quality_model = CatBoostRegressor(**params).fit(values(train, quality_features), quality_target, sample_weight=weights)
    return direction_model, quality_model, {"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean())}


def prediction_tables(direction_model: CatBoostRegressor, quality_model: CatBoostRegressor, base: pd.DataFrame, direction_features: list[str], quality_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    states = np.array([-.30, -.15, .0, .15, .30], dtype=np.float32)
    def expand(features: list[str]) -> np.ndarray:
        feature_values = base[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        repeated = np.repeat(feature_values, len(states), axis=0)
        return np.column_stack([repeated, np.tile(states, len(feature_values))])
    direction = direction_model.predict(expand(direction_features)).reshape(len(base), len(states))
    quality = quality_model.predict(expand(quality_features)).reshape(len(base), len(states))
    return direction, np.clip(quality, 0.0, 1.0)


def predict_margins(direction_scores: np.ndarray, quality_scores: np.ndarray, entry: float, large: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = 0.0
    margins, directions, qualities = [], [], []
    states = np.array([-.30, -.15, .0, .15, .30], dtype=np.float32)
    for direction_row, quality_row in zip(direction_scores, quality_scores):
        state_index = int(np.argmin(np.abs(states - current)))
        direction = float(direction_row[state_index])
        quality = float(quality_row[state_index])
        magnitude = .30 if quality >= large else .15 if quality >= entry else .0
        current = float(np.sign(direction) * magnitude)
        margins.append(current); directions.append(direction); qualities.append(quality)
    return np.asarray(margins), np.asarray(directions), np.asarray(qualities)


def market(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    decisions = frame.iloc[:-1].reset_index(drop=True)
    returns = frame["close"].to_numpy(dtype=float)[1:] / frame["open"].to_numpy(dtype=float)[1:] - 1.0
    return decisions, returns


def main() -> int:
    direction_features, quality_features = feature_sets()
    all_features = list(dict.fromkeys([*direction_features, *quality_features]))
    base_2024 = read_window(TRAIN_DATA, all_features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, all_features, "2025-01-01", DEV_END)
    base = pd.concat([base_2024, base_2025], ignore_index=True)
    labels = pd.concat([labels_for(base_2024), labels_for(base_2025)], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    direction_model, quality_model, weight_summary = fit(train, direction_features, quality_features)
    validation, returns = market(read_window(VAL_DATA, all_features, VAL_START, VAL_END))
    direction_scores, quality_scores = prediction_tables(direction_model, quality_model, validation, direction_features, quality_features)
    rows = []
    for entry in ENTRY_THRESHOLDS:
        for large in LARGE_THRESHOLDS:
            if large <= entry:
                continue
            margins, directions, qualities = predict_margins(direction_scores, quality_scores, entry, large)
            metrics = simulate(margins, returns)
            rows.append({"entry_quality_threshold": entry, "large_margin_quality_threshold": large, **metrics, "mean_quality": float(qualities.mean()), "mean_direction_utility": float(directions.mean()), "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0)})
    grid = pd.DataFrame(rows)
    candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(["action_events", "pnl_pct"], ascending=[True, False]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_threshold_grid.csv", index=False)
    feature_contract = {"direction_features": direction_features, "quality_features": quality_features, "current_margin_is_explicit_input": True, "forbidden_features": ["Regime4"], "direction_target": "max(long utilities) - max(short utilities)", "quality_target": "sigmoid((max(non-flat utilities) - flat utility) / 0.002)"}
    (OUT / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2) + "\n")
    report = {"diagnostic_only": True, "architecture": "independent CatBoost direction and quality regressors; no shared backbone", "train_period": ["2024-01-01", DEV_END], "validation_period": [VAL_START, VAL_END], "selection_rule": "positive validation PnL and at least 15 action events; then minimum events, then maximum PnL", "train_base_rows": int(len(base)), "train_state_rows": int(len(train)), "validation_rows": int(len(validation)), "recency_weight_summary": weight_summary, "future_prices_used_only_for_teacher_labels": True, "current_margin_is_explicit_causal_model_input": True, "oos_opened": False, "selected": selected, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
