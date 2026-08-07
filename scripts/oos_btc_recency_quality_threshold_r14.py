#!/usr/bin/env python3
"""Fresh-forward OOS for the conservative r13 threshold selected on validation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_direction_quality_sizing_r7 import predict_margins, simulate  # noqa: E402
from scripts.train_eval_btc_recency_weighted_quality_r11 import fit, market  # noqa: E402
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OOS_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"
THRESHOLD_REPORT = ROOT / "tmp/btc_shared_policy_v1_r13_conservative_threshold/report.json"
OUT = ROOT / "tmp/btc_shared_policy_v1_r14_oos_conservative_threshold"
OOS_START, OOS_END = "2026-04-01", "2026-06-30 23:59:59+00:00"


def main() -> int:
    chosen = json.loads(THRESHOLD_REPORT.read_text()).get("selected")
    if not chosen:
        raise ValueError("r13 has no eligible validation threshold; OOS must remain closed")
    entry, large = chosen["entry_quality_threshold"], chosen["large_margin_quality_threshold"]
    features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    base = pd.concat([base_2024, base_2025], ignore_index=True)
    labels = pd.concat([labels_for(base_2024), labels_for(base_2025)], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, weight_summary = fit(train, features)
    decisions, returns = market(read_window(OOS_DATA, features, OOS_START, OOS_END))
    margins, quality = predict_margins(model, decisions, features, mean, std, entry, large)
    metrics = simulate(margins, returns)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": decisions["timestamp"], "next_bar_price_return": returns, "target_margin_fraction": margins, "quality": quality}).to_csv(OUT / "oos_bar_by_bar_decisions.csv", index=False)
    report = {
        "diagnostic_only": True,
        "threshold_artifact": str(THRESHOLD_REPORT),
        "entry_quality_threshold": entry,
        "large_margin_quality_threshold": large,
        "fit_period": ["2024-01-01", "2025-12-31"],
        "oos_period": [OOS_START, OOS_END],
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "teacher_labels_used_as_oos_input": False,
        "recency_weight_summary": weight_summary,
        "oos_rows": int(len(decisions)),
        "metrics": metrics,
        "promotion_eligible": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
