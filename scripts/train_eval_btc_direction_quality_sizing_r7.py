#!/usr/bin/env python3
"""BTC direction plus 0..1 quality-sized margin diagnostic.

The direction head supplies only the long/short sign.  A scalar quality head
replaces the proposal head: below the entry threshold the target is flat,
between thresholds it is 15% margin, and above the large threshold it is 30%.
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
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import (  # noqa: E402
    MIN_ACTION_EVENTS,
    TRAIN_DATA,
    VAL_DATA,
    labels_for,
    simulate,
)
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r7_direction_quality"
TRAIN_START, TRAIN_END = "2024-01-01", "2024-12-31 23:59:59+00:00"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
QUALITY_TEMPERATURE = .002
ENTRY_THRESHOLDS = (.40, .45, .50, .55, .60, .65)
LARGE_THRESHOLDS = (.65, .70, .75, .80, .85)


class DirectionQualityModel(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.direction = nn.Linear(32, 2)
        self.quality = nn.Linear(32, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.backbone(values)
        return self.direction(state), torch.sigmoid(self.quality(state)).squeeze(1)


def input_values(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return frame[[*features, "current_margin_fraction"]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)


def targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actions = np.array([-.30, -.15, .0, .15, .30])
    utility_cols = [f"teacher_action_{action:+.2f}_utility" for action in actions]
    utilities = frame[utility_cols].to_numpy(dtype=np.float32)
    nonflat = np.array([0, 1, 3, 4])
    best_nonflat_index = nonflat[np.argmax(utilities[:, nonflat], axis=1)]
    direction = (actions[best_nonflat_index] > 0.0).astype(np.int64)
    flat_utility = utilities[:, 2]
    quality = 1.0 / (1.0 + np.exp(-(utilities[np.arange(len(utilities)), best_nonflat_index] - flat_utility) / QUALITY_TEMPERATURE))
    return direction, quality.astype(np.float32)


def fit(train: pd.DataFrame, features: list[str]) -> tuple[DirectionQualityModel, np.ndarray, np.ndarray]:
    x = input_values(train, features)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    direction, quality = targets(train)
    torch.manual_seed(270705)
    model = DirectionQualityModel(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(x)
    direction_tensor, quality_tensor = torch.from_numpy(direction), torch.from_numpy(quality)
    for _ in range(8):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), 512):
            batch = order[start : start + 512]
            direction_logits, quality_prediction = model(x_tensor[batch])
            # Low-quality rows carry no reliable direction target, so they do not steer sign learning.
            direction_loss = (nn.functional.cross_entropy(direction_logits, direction_tensor[batch], reduction="none") * quality_tensor[batch]).sum() / quality_tensor[batch].sum().clamp_min(1e-6)
            quality_loss = nn.functional.binary_cross_entropy(quality_prediction, quality_tensor[batch])
            optimizer.zero_grad()
            (direction_loss + quality_loss).backward()
            optimizer.step()
    return model.eval(), mean, std


def predict_margins(model: DirectionQualityModel, base: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray, entry: float, large: float) -> tuple[np.ndarray, np.ndarray]:
    current = 0.0
    margins, qualities = [], []
    features_np = base[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    for row in features_np:
        value = np.concatenate([row, np.array([current], dtype=np.float32)])
        with torch.no_grad():
            logits, quality = model(torch.from_numpy(((value - mean) / std)).unsqueeze(0))
        direction = 1.0 if int(logits.argmax(dim=1).item()) == 1 else -1.0
        quality_value = float(quality.item())
        magnitude = .30 if quality_value >= large else .15 if quality_value >= entry else .0
        current = direction * magnitude
        margins.append(current)
        qualities.append(quality_value)
    return np.asarray(margins), np.asarray(qualities)


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    train_base = read_window(TRAIN_DATA, features, TRAIN_START, TRAIN_END)
    validation_base_with_horizon = read_window(VAL_DATA, features, VAL_START, VAL_END)
    train_labels, validation_labels = labels_for(train_base), labels_for(validation_base_with_horizon)
    train = train_base.merge(train_labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std = fit(train, features)
    validation_market = validation_labels.drop_duplicates("decision_timestamp", keep="first").reset_index(drop=True)
    validation_base = validation_base_with_horizon.iloc[:len(validation_market)].reset_index(drop=True)
    if not (validation_base["timestamp"] == validation_market["decision_timestamp"]).all():
        raise ValueError("validation features and next-bar returns have mismatched timestamps")
    rows = []
    for entry in ENTRY_THRESHOLDS:
        for large in LARGE_THRESHOLDS:
            if large <= entry:
                continue
            margins, qualities = predict_margins(model, validation_base, features, mean, std, entry, large)
            metrics = simulate(margins, validation_market["next_bar_price_return"].to_numpy())
            rows.append({
                "entry_quality_threshold": entry,
                "large_margin_quality_threshold": large,
                **metrics,
                "mean_quality": float(qualities.mean()),
                "median_quality": float(np.median(qualities)),
                "meets_minimum_trade_count": bool(metrics["action_events"] >= MIN_ACTION_EVENTS),
                "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0),
            })
    grid = pd.DataFrame(rows)
    candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(["pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_quality_threshold_grid.csv", index=False)
    report = {
        "diagnostic_only": True,
        "split": {"train": [TRAIN_START, TRAIN_END], "validation": [VAL_START, VAL_END], "oos_opened": False},
        "architecture": "shared direction and scalar-quality heads; no proposal head",
        "quality_mapping": "quality < entry => 0.0 margin; entry <= quality < large => 0.15; quality >= large => 0.30; direction head supplies sign",
        "quality_target": "sigmoid(best non-flat cost-aware horizon utility minus flat utility)",
        "future_prices_used_only_for_teacher_labels": True,
        "current_margin_is_explicit_causal_model_input": True,
        "train_base_rows": int(len(train_base)),
        "train_state_rows": int(len(train)),
        "validation_base_rows": int(len(validation_base)),
        "minimum_action_events": MIN_ACTION_EVENTS,
        "selected": selected,
        "promotion_eligible": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
