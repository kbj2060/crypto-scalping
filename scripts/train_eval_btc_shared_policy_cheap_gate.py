#!/usr/bin/env python3
"""Train-only shard smoke test for separate direction/action feature adapters."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
TRAIN_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2024.csv"
VAL_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2025.csv"
TRAIN_LABELS = ROOT / "tmp/btc_shared_policy_v1_r2_train_2024/btc_shared_policy_teacher_labels_2026.csv"
VAL_LABELS = ROOT / "tmp/btc_shared_policy_v1_r2_train_smoke/btc_shared_policy_teacher_labels_2026.csv"
OUT = ROOT / "tmp/btc_shared_policy_v1_r3_cheap_gate"
ACTIONS = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])


def load_frame(data: Path, labels: Path, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(data, usecols=["timestamp"] + features, low_memory=False)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    lab = pd.read_csv(labels)
    lab["decision_timestamp"] = pd.to_datetime(lab["decision_timestamp"], utc=True)
    return base.merge(lab, left_on="timestamp", right_on="decision_timestamp", how="inner"), lab


def soft_fit(frame: pd.DataFrame, features: list[str], probability_columns: list[str], seed: int) -> LGBMClassifier:
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    expanded_x = np.repeat(x, len(probability_columns), axis=0)
    y = np.tile(np.arange(len(probability_columns)), len(frame))
    weights = frame[probability_columns].to_numpy().reshape(-1)
    model = LGBMClassifier(objective="multiclass", num_class=len(probability_columns), n_estimators=250, learning_rate=.04, num_leaves=31, random_state=seed, verbosity=-1)
    model.fit(expanded_x, y, sample_weight=weights)
    return model


def main() -> int:
    selection = json.loads(SELECTION.read_text())
    direction_features, action_features = selection["direction_features"], selection["action_features"]
    train_dir, _ = load_frame(TRAIN_DATA, TRAIN_LABELS, direction_features)
    val_dir, _ = load_frame(VAL_DATA, VAL_LABELS, direction_features)
    train_act, _ = load_frame(TRAIN_DATA, TRAIN_LABELS, action_features)
    val_act, _ = load_frame(VAL_DATA, VAL_LABELS, action_features)
    direction_cols = ["teacher_short_probability", "teacher_flat_probability", "teacher_long_probability"]
    action_cols = [f"teacher_action_{a:+.1f}_probability" for a in ACTIONS]
    direction = soft_fit(train_dir, direction_features, direction_cols, 270705)
    action = soft_fit(train_act, action_features, action_cols, 270706)
    dir_pred = direction.predict_proba(val_dir[direction_features].fillna(0.0))
    act_pred = action.predict_proba(val_act[action_features].fillna(0.0))
    target = ACTIONS[act_pred.argmax(axis=1)]
    current = 0.0; equity = 1.0; events = 0
    for margin, ret in zip(target, val_act["next_bar_price_return"]):
        events += int(not np.isclose(margin, current))
        equity *= 1.0 + margin * 3.0 * float(ret) - abs(margin - current) * 3.0 * .0021
        current = margin
    report = {"diagnostic_only": True, "train_rows": len(train_act), "validation_rows": len(val_act), "direction_features": len(direction_features), "action_features": len(action_features), "validation_action_kl": float(np.mean(np.sum(val_act[action_cols].to_numpy() * (np.log(val_act[action_cols].to_numpy() + 1e-12) - np.log(act_pred + 1e-12)), axis=1))), "validation_direction_kl": float(np.mean(np.sum(val_dir[direction_cols].to_numpy() * (np.log(val_dir[direction_cols].to_numpy() + 1e-12) - np.log(dir_pred + 1e-12)), axis=1))), "validation_policy_pnl_pct": (equity - 1.0) * 100.0, "validation_action_events": events}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
