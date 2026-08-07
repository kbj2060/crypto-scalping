#!/usr/bin/env python3
"""Frozen r11 configuration, final-fit fresh-forward BTC OOS evaluation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_recency_weighted_quality_r11 import fit  # noqa: E402
from scripts.train_eval_btc_direction_quality_sizing_r7 import predict_margins, simulate  # noqa: E402
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OOS_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"
OUT = ROOT / "tmp/btc_shared_policy_v1_r12_oos"
ENTRY_THRESHOLD, LARGE_THRESHOLD = .65, .75
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59+00:00"


def market(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    decisions = frame.iloc[:-1].reset_index(drop=True)
    returns = frame["close"].to_numpy(dtype=float)[1:] / frame["open"].to_numpy(dtype=float)[1:] - 1.0
    return decisions, returns


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    train_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    train_2025 = read_window(VAL_DATA, features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    labels = pd.concat([labels_for(train_2024), labels_for(train_2025)], ignore_index=True)
    base = pd.concat([train_2024, train_2025], ignore_index=True)
    train = base.merge(labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std, weight_summary = fit(train, features)
    oos_base = read_window(OOS_DATA, features, OOS_START, OOS_END)
    decisions, returns = market(oos_base)
    margins, quality = predict_margins(model, decisions, features, mean, std, ENTRY_THRESHOLD, LARGE_THRESHOLD)
    metrics = simulate(margins, returns)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": decisions["timestamp"], "next_bar_price_return": returns, "target_margin_fraction": margins, "quality": quality}).to_csv(OUT / "oos_bar_by_bar_decisions.csv", index=False)
    report = {"diagnostic_only": True, "fit_period": ["2024-01-01", "2025-12-31"], "thresholds_frozen_from": "r11 validation", "entry_quality_threshold": ENTRY_THRESHOLD, "large_margin_quality_threshold": LARGE_THRESHOLD, "oos_period": [OOS_START, OOS_END], "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "teacher_labels_used_as_oos_input": False, "recency_weight_summary": weight_summary, "oos_rows": int(len(decisions)), "metrics": metrics, "promotion_eligible": False}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
