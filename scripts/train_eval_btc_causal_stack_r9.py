#!/usr/bin/env python3
"""Purged expanding-OOF secondary features for BTC direction-quality sizing."""
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
from scripts.train_eval_btc_direction_quality_sizing_r7 import (  # noqa: E402
    ENTRY_THRESHOLDS, LARGE_THRESHOLDS, MIN_ACTION_EVENTS, fit as fit_stage2,
    predict_margins, simulate,
)
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r9_causal_stack"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
OOF_FOLDS = (("2024-04-01", "2024-07-01"), ("2024-07-01", "2024-10-01"), ("2024-10-01", "2025-01-01"))
STACK_COLS = ["stack_p_short", "stack_p_long", "stack_quality"]


class StageOne(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.direction = nn.Linear(32, 2)
        self.quality = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        return self.direction(z), torch.sigmoid(self.quality(z)).squeeze(1)


def primary_targets(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    actions = np.array([-.30, -.15, .0, .15, .30])
    cols = [f"teacher_action_{action:+.2f}_utility" for action in actions]
    values = labels[cols].to_numpy(dtype=np.float32)
    nonflat = np.array([0, 1, 3, 4])
    best = nonflat[np.argmax(values[:, nonflat], axis=1)]
    direction = (actions[best] > 0.0).astype(np.int64)
    quality = 1.0 / (1.0 + np.exp(-(values[np.arange(len(values)), best] - values[:, 2]) / .002))
    return direction, quality.astype(np.float32)


def fit_stage1(base: pd.DataFrame, features: list[str]) -> tuple[StageOne, np.ndarray, np.ndarray]:
    labels = labels_for(base)
    labels = labels.loc[np.isclose(labels["current_margin_fraction"], 0.0)].reset_index(drop=True)
    x = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    values = x[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    mean, std = values.mean(axis=0), values.std(axis=0); std[std < 1e-6] = 1.0
    direction, quality = primary_targets(x)
    torch.manual_seed(270705)
    model = StageOne(values.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xv, yv, qv = map(torch.from_numpy, ((values - mean) / std, direction, quality))
    for _ in range(6):
        order = torch.randperm(len(xv))
        for start in range(0, len(order), 512):
            batch = order[start : start + 512]
            logits, prediction = model(xv[batch])
            direction_loss = (nn.functional.cross_entropy(logits, yv[batch], reduction="none") * qv[batch]).sum() / qv[batch].sum().clamp_min(1e-6)
            loss = direction_loss + nn.functional.binary_cross_entropy(prediction, qv[batch])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval(), mean, std


def score_stage1(model: StageOne, base: pd.DataFrame, features: list[str], mean: np.ndarray, std: np.ndarray) -> pd.DataFrame:
    values = base[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            logits, quality = model(torch.from_numpy((values[start : start + 4096] - mean) / std))
            prob = torch.softmax(logits, dim=1).numpy()
            rows.append(np.column_stack([prob, quality.numpy()]))
    scores = np.vstack(rows)
    return pd.DataFrame({"timestamp": base["timestamp"].to_numpy(), "stack_p_short": scores[:, 0], "stack_p_long": scores[:, 1], "stack_quality": scores[:, 2]})


def market(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = frame.iloc[:-1].reset_index(drop=True)
    returns = frame["close"].to_numpy(dtype=float)[1:] / frame["open"].to_numpy(dtype=float)[1:] - 1.0
    return decisions, pd.DataFrame({"timestamp": decisions["timestamp"], "next_bar_price_return": returns})


def main() -> int:
    raw_features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, raw_features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, raw_features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    oof_scores = []
    for score_start, score_end in OOF_FOLDS:
        fit_base = base_2024.loc[base_2024["timestamp"] < pd.to_datetime(score_start, utc=True)].reset_index(drop=True)
        score_base = base_2024.loc[base_2024["timestamp"].between(pd.to_datetime(score_start, utc=True), pd.to_datetime(score_end, utc=True), inclusive="left")].reset_index(drop=True)
        model, mean, std = fit_stage1(fit_base, raw_features)
        oof_scores.append(score_stage1(model, score_base, raw_features, mean, std))
    stack_2024 = pd.concat(oof_scores, ignore_index=True)
    primary, mean, std = fit_stage1(base_2024, raw_features)
    stack_2025 = score_stage1(primary, base_2025, raw_features, mean, std)

    stage2_features = [*raw_features, *STACK_COLS]
    train_base = base_2024.merge(stack_2024, on="timestamp", how="inner")
    train_labels = labels_for(train_base)
    train = train_base.merge(train_labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    stage2, mean2, std2 = fit_stage2(train, stage2_features)

    validation_raw = base_2025.loc[base_2025["timestamp"].between(pd.to_datetime(VAL_START, utc=True), pd.to_datetime(VAL_END, utc=True))].reset_index(drop=True)
    validation = validation_raw.merge(stack_2025, on="timestamp", how="inner")
    decisions, realized = market(validation)
    rows = []
    for entry in ENTRY_THRESHOLDS:
        for large in LARGE_THRESHOLDS:
            if large <= entry:
                continue
            margins, quality = predict_margins(stage2, decisions, stage2_features, mean2, std2, entry, large)
            metrics = simulate(margins, realized["next_bar_price_return"].to_numpy())
            rows.append({"entry_quality_threshold": entry, "large_margin_quality_threshold": large, **metrics, "mean_quality": float(quality.mean()), "meets_minimum_trade_count": bool(metrics["action_events"] >= MIN_ACTION_EVENTS), "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0)})
    grid = pd.DataFrame(rows); candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(["pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True); grid.to_csv(OUT / "validation_threshold_grid.csv", index=False)
    report = {"diagnostic_only": True, "stage1": "expanding causal OOF predictions on 2024; frozen 2024 model forward-scores 2025", "oof_folds": OOF_FOLDS, "train_base_rows": int(len(train_base)), "train_state_rows": int(len(train)), "validation_rows": int(len(decisions)), "selected": selected, "oos_opened": False, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2))


if __name__ == "__main__":
    main()
