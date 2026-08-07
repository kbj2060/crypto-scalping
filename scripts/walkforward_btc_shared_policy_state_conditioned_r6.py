#!/usr/bin/env python3
"""Fixed-split walk-forward validation for the state-conditioned BTC policy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import (  # noqa: E402
    BUFFERS,
    MIN_ACTION_EVENTS,
    TRAIN_DATA,
    VAL_DATA,
    fit,
    labels_for,
    predict_policy,
    simulate,
)

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r6_walkforward"
TRAIN_START, TRAIN_END = "2024-01-01", "2024-12-31 23:59:59+00:00"
VAL_START, VAL_END = "2025-09-01", "2025-12-31 23:59:59+00:00"


def read_window(path: Path, features: list[str], start: str, end: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "open", "close", *features], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.loc[frame["timestamp"].between(pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True))].reset_index(drop=True)


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    train_base = read_window(TRAIN_DATA, features, TRAIN_START, TRAIN_END)
    validation_base_with_horizon = read_window(VAL_DATA, features, VAL_START, VAL_END)
    train_labels = labels_for(train_base)
    validation_labels = labels_for(validation_base_with_horizon)
    train = train_base.merge(train_labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, utility_mean, utility_std = fit(train, features, epochs=8)

    validation_market = validation_labels.drop_duplicates("decision_timestamp", keep="first").reset_index(drop=True)
    validation_base = validation_base_with_horizon.iloc[:len(validation_market)].reset_index(drop=True)
    if len(validation_base) != len(validation_market):
        raise ValueError("validation feature and market-return rows must align one-to-one")
    if not (validation_base["timestamp"] == validation_market["decision_timestamp"]).all():
        raise ValueError("validation features and next-bar returns have mismatched timestamps")

    rows = []
    for buffer in BUFFERS:
        actions, advantages, confidences = predict_policy(
            model, validation_base, features, mean, std, utility_mean, utility_std, buffer,
        )
        metrics = simulate(actions, validation_market["next_bar_price_return"].to_numpy())
        rows.append({
            "switch_buffer": buffer,
            **metrics,
            "mean_predicted_switch_advantage": float(advantages.mean()),
            "median_action_probability_confidence": float(pd.Series(confidences).median()),
            "meets_minimum_trade_count": bool(metrics["action_events"] >= MIN_ACTION_EVENTS),
            "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0),
        })
    grid = pd.DataFrame(rows)
    candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(["pnl_pct", "action_events"], ascending=[False, True]).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_switch_buffer_grid.csv", index=False)
    report = {
        "diagnostic_only": True,
        "split": {"train": [TRAIN_START, TRAIN_END], "validation": [VAL_START, VAL_END], "oos_opened": False},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "future_rows_used_for_entry": False,
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
