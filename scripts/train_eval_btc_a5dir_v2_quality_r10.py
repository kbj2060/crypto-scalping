#!/usr/bin/env python3
"""Validation-only ablation: add causal a5dir_v2 to BTC quality sizing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_direction_quality_sizing_r7 import (  # noqa: E402
    ENTRY_THRESHOLDS, LARGE_THRESHOLDS, MIN_ACTION_EVENTS, fit, predict_margins, simulate,
)
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
STACK_DIR = ROOT / "tmp/btc_a5dir_v2_causal"
OUT = ROOT / "tmp/btc_shared_policy_v1_r10_a5dir_v2"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"
STACK_COLS = ["a5dir_v2_flat_prob", "a5dir_v2_long_prob", "a5dir_v2_short_prob", "a5dir_v2_prob_max", "a5dir_v2_edge", "a5dir_v2_margin", "a5dir_v2_side"]


def market(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    decisions = frame.iloc[:-1].reset_index(drop=True)
    returns = frame["close"].to_numpy(dtype=float)[1:] / frame["open"].to_numpy(dtype=float)[1:] - 1.0
    return decisions, returns


def main() -> int:
    raw_features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, raw_features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, raw_features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    oof = pd.read_parquet(STACK_DIR / "a5dir_v2_2024_oof.parquet")
    forward = pd.read_parquet(STACK_DIR / "a5dir_v2_2025_forward.parquet")
    oof["timestamp"] = pd.to_datetime(oof["timestamp"], utc=True); forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    train_base = base_2024.merge(oof[["timestamp", *STACK_COLS]], on="timestamp", how="inner")
    if train_base[STACK_COLS].isna().any().any():
        raise ValueError("a5dir_v2 OOF feature coverage is incomplete")
    labels = labels_for(train_base)
    train = train_base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    features = [*raw_features, *STACK_COLS]
    model, mean, std = fit(train, features)
    validation = base_2025.loc[base_2025["timestamp"].between(pd.to_datetime(VAL_START, utc=True), pd.to_datetime(VAL_END, utc=True))].merge(forward[["timestamp", *STACK_COLS]], on="timestamp", how="inner").reset_index(drop=True)
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
    report = {"diagnostic_only": True, "change_from_r7": "causal a5dir_v2 secondary features only", "train_rows": int(len(train)), "validation_rows": int(len(decisions)), "selected": selected, "oos_opened": False, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "grid": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
