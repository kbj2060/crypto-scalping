#!/usr/bin/env python3
"""Recency-weighted BTC direction-quality validation on 2024 plus 2025 development data."""
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
    DirectionQualityModel, ENTRY_THRESHOLDS, LARGE_THRESHOLDS, MIN_ACTION_EVENTS,
    input_values, predict_margins, simulate, targets,
)
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r11_recency_weighted"
DEV_END = "2025-08-31 23:59:59+00:00"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
HALF_LIFE_DAYS = 180.0


def fit(train: pd.DataFrame, features: list[str]) -> tuple[DirectionQualityModel, np.ndarray, np.ndarray, dict[str, float]]:
    x = input_values(train, features)
    mean, std = x.mean(axis=0), x.std(axis=0); std[std < 1e-6] = 1.0
    direction, quality = targets(train)
    timestamps = pd.to_datetime(train["decision_timestamp"], utc=True)
    age_days = (timestamps.max() - timestamps).dt.total_seconds().to_numpy(dtype=np.float32) / 86400.0
    weights = np.exp2(-age_days / HALF_LIFE_DAYS).astype(np.float32)
    weights /= weights.mean()
    torch.manual_seed(270705)
    model = DirectionQualityModel(x.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xv, yv, qv, wv = map(torch.from_numpy, ((x - mean) / std, direction, quality, weights))
    for _ in range(8):
        order = torch.randperm(len(xv))
        for start in range(0, len(order), 512):
            batch = order[start : start + 512]
            logits, prediction = model(xv[batch])
            direction_weight = qv[batch] * wv[batch]
            direction_loss = (nn.functional.cross_entropy(logits, yv[batch], reduction="none") * direction_weight).sum() / direction_weight.sum().clamp_min(1e-6)
            quality_loss = (nn.functional.binary_cross_entropy(prediction, qv[batch], reduction="none") * wv[batch]).sum() / wv[batch].sum().clamp_min(1e-6)
            optimizer.zero_grad(); (direction_loss + quality_loss).backward(); optimizer.step()
    return model.eval(), mean, std, {"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean())}


def market(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    decisions = frame.iloc[:-1].reset_index(drop=True)
    returns = frame["close"].to_numpy(dtype=float)[1:] / frame["open"].to_numpy(dtype=float)[1:] - 1.0
    return decisions, returns


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025_dev = read_window(VAL_DATA, features, "2025-01-01", DEV_END)
    labels = pd.concat([labels_for(base_2024), labels_for(base_2025_dev)], ignore_index=True)
    base = pd.concat([base_2024, base_2025_dev], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, weight_summary = fit(train, features)
    validation = read_window(VAL_DATA, features, VAL_START, VAL_END)
    decisions, returns = market(validation)
    rows = []
    for entry in ENTRY_THRESHOLDS:
        for large in LARGE_THRESHOLDS:
            if large <= entry:
                continue
            margins, quality = predict_margins(model, decisions, features, mean, std, entry, large)
            metrics = simulate(margins, returns)
            rows.append({"entry_quality_threshold": entry, "large_margin_quality_threshold": large, **metrics, "mean_quality": float(quality.mean()), "meets_minimum_trade_count": bool(metrics["action_events"] >= MIN_ACTION_EVENTS), "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0)})
    grid = pd.DataFrame(rows); candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(["pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True); grid.to_csv(OUT / "validation_threshold_grid.csv", index=False)
    report = {"diagnostic_only": True, "train_period": ["2024-01-01", DEV_END], "validation_period": [VAL_START, VAL_END], "recency_half_life_days": HALF_LIFE_DAYS, "recency_weight_summary": weight_summary, "train_base_rows": int(len(base)), "train_state_rows": int(len(train)), "validation_rows": int(len(decisions)), "selected": selected, "oos_opened": False, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
