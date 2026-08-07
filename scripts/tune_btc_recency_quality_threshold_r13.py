#!/usr/bin/env python3
"""Choose a conservative quality threshold on the untouched 2025 validation split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_direction_quality_sizing_r7 import MIN_ACTION_EVENTS, predict_margins, simulate  # noqa: E402
from scripts.train_eval_btc_recency_weighted_quality_r11 import DEV_END, VAL_DATA, VAL_END, VAL_START, fit, market  # noqa: E402
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r13_conservative_threshold"
ENTRY_THRESHOLDS = (.65, .70, .75, .80, .85)
LARGE_THRESHOLDS = (.75, .80, .85, .90, .95)


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, features, "2025-01-01", DEV_END)
    base = pd.concat([base_2024, base_2025], ignore_index=True)
    labels = pd.concat([labels_for(base_2024), labels_for(base_2025)], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, weight_summary = fit(train, features)
    decisions, returns = market(read_window(VAL_DATA, features, VAL_START, VAL_END))
    rows = []
    for entry in ENTRY_THRESHOLDS:
        for large in LARGE_THRESHOLDS:
            if large <= entry:
                continue
            margins, quality = predict_margins(model, decisions, features, mean, std, entry, large)
            metrics = simulate(margins, returns)
            rows.append({
                "entry_quality_threshold": entry,
                "large_margin_quality_threshold": large,
                **metrics,
                "mean_quality": float(quality.mean()),
                "selection_eligible": bool(metrics["action_events"] >= MIN_ACTION_EVENTS and metrics["pnl_pct"] > 0.0),
            })
    grid = pd.DataFrame(rows)
    candidates = grid.loc[grid["selection_eligible"]]
    selected = None if candidates.empty else candidates.sort_values(
        ["action_events", "pnl_pct"], ascending=[True, False]
    ).iloc[0].to_dict()
    OUT.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUT / "validation_threshold_grid.csv", index=False)
    report = {
        "diagnostic_only": True,
        "train_period": ["2024-01-01", DEV_END],
        "validation_period": [VAL_START, VAL_END],
        "selection_rule": "positive validation PnL and at least 15 action events; then minimum events, then maximum PnL",
        "selected": selected,
        "oos_opened": False,
        "recency_weight_summary": weight_summary,
        "promotion_eligible": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
