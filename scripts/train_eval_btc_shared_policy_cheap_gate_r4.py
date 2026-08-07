#!/usr/bin/env python3
"""5-action soft policy gate with Validation-only confidence abstention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = json.loads((ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json").read_text())["action_features"]
TRAIN_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2024.csv"
VAL_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2025.csv"
TRAIN_LABELS = ROOT / "tmp/btc_shared_policy_v1_r2_train_2024/btc_shared_policy_teacher_labels_2026.csv"
VAL_LABELS = ROOT / "tmp/btc_shared_policy_v1_r2_train_smoke/btc_shared_policy_teacher_labels_2026.csv"
OUT = ROOT / "tmp/btc_shared_policy_v1_r4_cheap_gate"
SOURCE_ACTIONS = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
TARGET_ACTIONS = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
SOURCE_TO_TARGET = np.array([0, 1, 1, 2, 3, 3, 4])


def load(data: Path, labels: Path) -> pd.DataFrame:
    x = pd.read_csv(data, usecols=["timestamp"] + FEATURES, low_memory=False)
    y = pd.read_csv(labels)
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    y["decision_timestamp"] = pd.to_datetime(y["decision_timestamp"], utc=True)
    return x.merge(y, left_on="timestamp", right_on="decision_timestamp", how="inner")


def target_probabilities(frame: pd.DataFrame) -> np.ndarray:
    source = frame[[f"teacher_action_{a:+.1f}_probability" for a in SOURCE_ACTIONS]].to_numpy()
    target = np.zeros((len(frame), len(TARGET_ACTIONS)))
    for source_i, target_i in enumerate(SOURCE_TO_TARGET):
        target[:, target_i] += source[:, source_i]
    return target


def simulate(probabilities: np.ndarray, returns: np.ndarray, threshold: float) -> tuple[float, int]:
    chosen = TARGET_ACTIONS[probabilities.argmax(axis=1)]
    chosen[probabilities.max(axis=1) < threshold] = 0.0
    cash = 1.0; current = 0.0; events = 0
    for action, ret in zip(chosen, returns):
        events += int(not np.isclose(action, current))
        cash *= 1.0 + action * 3.0 * float(ret) - abs(action - current) * 3.0 * .0021
        current = action
    return (cash - 1.0) * 100.0, events


def main() -> int:
    train, val = load(TRAIN_DATA, TRAIN_LABELS), load(VAL_DATA, VAL_LABELS)
    train_p = target_probabilities(train)
    x = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    model = LGBMClassifier(objective="multiclass", num_class=5, n_estimators=250, learning_rate=.04, num_leaves=31, random_state=270705, verbosity=-1)
    model.fit(np.repeat(x, 5, axis=0), np.tile(np.arange(5), len(train)), sample_weight=train_p.reshape(-1))
    pred = model.predict_proba(val[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    rows = []
    for threshold in (.40, .45, .50, .55, .60, .65, .70, .75, .80):
        pnl, events = simulate(pred, val["next_bar_price_return"].to_numpy(), threshold)
        rows.append({"confidence_threshold": threshold, "validation_pnl_pct": pnl, "action_events": events})
    grid = pd.DataFrame(rows)
    selected = grid.sort_values(["validation_pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_confidence_grid.csv", index=False)
    (OUT / "report.json").write_text(json.dumps({"diagnostic_only": True, "selection_scope": "Validation only", "train_rows": len(train), "validation_rows": len(val), "actions": TARGET_ACTIONS.tolist(), "selected": selected}, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
