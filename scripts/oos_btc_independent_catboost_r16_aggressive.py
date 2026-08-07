#!/usr/bin/env python3
"""Standard OOS diagnostic for the r15 aggressive validation threshold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_independent_catboost_r15 import (  # noqa: E402
    TRAIN_DATA, VAL_DATA, feature_sets, fit, labels_for, market, prediction_tables,
    predict_margins, read_window,
)

OOS_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"
GRID = ROOT / "tmp/btc_independent_catboost_r15/validation_threshold_grid.csv"
OUT = ROOT / "tmp/btc_independent_catboost_r17_aggressive_standard_oos"
ENTRY_THRESHOLD, LARGE_THRESHOLD = .40, .75
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59+00:00"


def main() -> int:
    grid = pd.read_csv(GRID)
    matching = grid.loc[(grid["entry_quality_threshold"] == ENTRY_THRESHOLD) & (grid["large_margin_quality_threshold"] == LARGE_THRESHOLD)]
    if len(matching) != 1 or not bool(matching.iloc[0]["selection_eligible"]):
        raise ValueError("aggressive threshold is not an eligible r15 validation candidate")
    direction_features, quality_features = feature_sets()
    all_features = list(dict.fromkeys([*direction_features, *quality_features]))
    base_2024 = read_window(TRAIN_DATA, all_features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, all_features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    base = pd.concat([base_2024, base_2025], ignore_index=True)
    labels = pd.concat([labels_for(base_2024), labels_for(base_2025)], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    direction_model, quality_model, weight_summary = fit(train, direction_features, quality_features)
    decisions, returns = market(read_window(OOS_DATA, all_features, OOS_START, OOS_END))
    direction_scores, quality_scores = prediction_tables(direction_model, quality_model, decisions, direction_features, quality_features)
    margins, directions, qualities = predict_margins(direction_scores, quality_scores, ENTRY_THRESHOLD, LARGE_THRESHOLD)
    from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import simulate
    metrics = simulate(margins, returns)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": decisions["timestamp"], "next_bar_price_return": returns, "target_margin_fraction": margins, "direction_utility": directions, "quality": qualities}).to_csv(OUT / "oos_bar_by_bar_decisions.csv", index=False)
    report = {"diagnostic_only": True, "threshold_artifact": str(GRID), "entry_quality_threshold": ENTRY_THRESHOLD, "large_margin_quality_threshold": LARGE_THRESHOLD, "fit_period": ["2024-01-01", "2025-12-31"], "oos_period": [OOS_START, OOS_END], "oos_previously_observed_for_other_architectures": True, "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "teacher_labels_used_as_oos_input": False, "recency_weight_summary": weight_summary, "oos_rows": int(len(decisions)), "metrics": metrics, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
