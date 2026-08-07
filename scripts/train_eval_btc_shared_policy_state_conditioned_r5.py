#!/usr/bin/env python3
"""State-conditioned shared-backbone BTC policy diagnostic.

The teacher uses future prices only to construct labels.  The model receives
the deployed, causal current margin state and selects a new target margin.
Direction is derived from the five action probabilities; zero is full exit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
TRAIN_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2024.csv"
VAL_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2025.csv"
OUT = ROOT / "tmp/btc_shared_policy_v1_r5_state_conditioned"
ACTIONS = np.array([-.30, -.15, .0, .15, .30], dtype=np.float32)
LEVERAGE = 3.0
ONE_WAY_COST_RATE = .0021
TAIL_BARS = 10_000
MIN_ACTION_EVENTS = 15
BUFFERS = (0.0, .0005, .0010, .0015, .0020, .0030)


class SharedPolicy(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.action_logits = nn.Linear(32, len(ACTIONS))
        self.action_utilities = nn.Linear(32, len(ACTIONS))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.backbone(values)
        return self.action_logits(state), self.action_utilities(state)


def read_tail(path: Path, features: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "open", "close", *features], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.tail(TAIL_BARS + 49).reset_index(drop=True)


def labels_for(frame: pd.DataFrame) -> pd.DataFrame:
    from pipeline.btc_trajectory_teacher import TeacherConfig, build_state_conditioned_teacher_labels

    return build_state_conditioned_teacher_labels(
        frame[["timestamp", "open", "close"]],
        TeacherConfig(margin_step=.15),
    )


def model_input(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame[[*features, "current_margin_fraction"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(dtype=np.float32)


def fit(train: pd.DataFrame, features: list[str], epochs: int = 12) -> tuple[SharedPolicy, np.ndarray, np.ndarray, float, float]:
    x = model_input(train, features)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    probability_columns = [f"teacher_action_{action:+.2f}_probability" for action in ACTIONS]
    utility_columns = [f"teacher_action_{action:+.2f}_utility" for action in ACTIONS]
    probabilities = train[probability_columns].to_numpy(dtype=np.float32)
    utilities = train[utility_columns].to_numpy(dtype=np.float32)
    utility_mean, utility_std = float(utilities.mean()), float(utilities.std())
    if utility_std < 1e-8:
        raise ValueError("teacher utility target has zero variance")
    utilities = (utilities - utility_mean) / utility_std

    torch.manual_seed(270705)
    model = SharedPolicy(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor, p_tensor, u_tensor = map(torch.from_numpy, (x, probabilities, utilities))
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), 512):
            batch = order[start : start + 512]
            logits, utility_prediction = model(x_tensor[batch])
            probability_loss = -(p_tensor[batch] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            utility_loss = nn.functional.mse_loss(utility_prediction, u_tensor[batch])
            optimizer.zero_grad()
            (probability_loss + utility_loss).backward()
            optimizer.step()
    return model.eval(), mean, std, utility_mean, utility_std


def predict_policy(model: SharedPolicy, base: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray, utility_mean: float, utility_std: float, buffer: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = 0.0
    chosen, advantages, confidences = [], [], []
    feature_values = base[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    for values in feature_values:
        row = np.concatenate([values, np.array([current], dtype=np.float32)])
        row = (row - mean) / std
        with torch.no_grad():
            logits, utility_prediction = model(torch.from_numpy(row).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        utilities = utility_prediction.numpy()[0] * utility_std + utility_mean
        current_index = int(np.argmin(np.abs(ACTIONS - current)))
        candidate_index = int(np.argmax(utilities))
        advantage = float(utilities[candidate_index] - utilities[current_index])
        if advantage > buffer:
            current = float(ACTIONS[candidate_index])
        chosen.append(current)
        advantages.append(advantage)
        confidences.append(float(probabilities.max()))
    return np.asarray(chosen), np.asarray(advantages), np.asarray(confidences)


def simulate(actions: np.ndarray, returns: np.ndarray) -> dict[str, float | int]:
    cash, current, events, gross, costs = 1.0, 0.0, 0, 0.0, 0.0
    for action, price_return in zip(actions, returns):
        turnover = abs(float(action) - current)
        events += int(not np.isclose(action, current))
        gross += float(action) * LEVERAGE * float(price_return)
        costs += turnover * LEVERAGE * ONE_WAY_COST_RATE
        cash *= 1.0 + float(action) * LEVERAGE * float(price_return) - turnover * LEVERAGE * ONE_WAY_COST_RATE
        current = float(action)
    return {
        "pnl_pct": (cash - 1.0) * 100.0,
        "action_events": events,
        "gross_return_sum_pct": gross * 100.0,
        "turnover_cost_sum_pct": costs * 100.0,
    }


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    train_base, val_base = read_tail(TRAIN_DATA, features), read_tail(VAL_DATA, features)
    train_labels, val_labels = labels_for(train_base), labels_for(val_base)
    train = train_base.merge(train_labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, utility_mean, utility_std = fit(train, features)
    validation_market = val_labels.drop_duplicates("decision_timestamp", keep="first").reset_index(drop=True)
    validation_base = val_base.iloc[:len(validation_market)].reset_index(drop=True)
    if len(validation_base) != len(validation_market):
        raise ValueError("validation feature and market-return rows must align one-to-one")
    if not (validation_base["timestamp"] == validation_market["decision_timestamp"]).all():
        raise ValueError("validation features and next-bar returns have mismatched timestamps")

    validation_rows = []
    selected_actions = None
    for buffer in BUFFERS:
        actions, advantages, confidences = predict_policy(model, validation_base, features, mean, std, utility_mean, utility_std, buffer)
        metrics = simulate(actions, validation_market["next_bar_price_return"].to_numpy())
        validation_rows.append({
            "switch_buffer": buffer,
            **metrics,
            "mean_predicted_switch_advantage": float(advantages.mean()),
            "median_action_probability_confidence": float(np.median(confidences)),
            "meets_minimum_trade_count": bool(metrics["action_events"] >= MIN_ACTION_EVENTS),
            "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0),
        })
    grid = pd.DataFrame(validation_rows)
    eligible = grid.loc[grid["selection_eligible"]]
    selected = None
    if not eligible.empty:
        selected = eligible.sort_values(["pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
        selected_actions, _, _ = predict_policy(model, validation_base, features, mean, std, utility_mean, utility_std, float(selected["switch_buffer"]))

    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_switch_buffer_grid.csv", index=False)
    train_labels.to_csv(OUT / "train_state_conditioned_teacher_labels.csv", index=False)
    val_labels.to_csv(OUT / "validation_state_conditioned_teacher_labels.csv", index=False)
    report = {
        "diagnostic_only": True,
        "selection_scope": "2024 terminal train shard fit; 2025 terminal validation shard buffer selection only",
        "future_prices_used_only_for_teacher_labels": True,
        "current_margin_is_explicit_causal_model_input": True,
        "action_space_margin_fraction": ACTIONS.tolist(),
        "direction_and_full_exit": "derived from selected action sign; action 0.0 is full exit",
        "minimum_action_events": MIN_ACTION_EVENTS,
        "train_base_rows": int(len(train_base)),
        "train_state_rows": int(len(train)),
        "validation_base_rows": int(len(validation_base)),
        "selected": selected,
        "promotion_eligible": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": validation_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
