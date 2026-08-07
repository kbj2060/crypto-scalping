#!/usr/bin/env python3
"""Frozen-threshold fresh-forward OOS test for BTC direction-quality sizing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_direction_quality_sizing_r7 import (  # noqa: E402
    OUT as R7_OUT,
    SELECTION,
    fit,
    predict_margins,
)
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import (  # noqa: E402
    TRAIN_DATA,
    VAL_DATA,
    labels_for,
    simulate,
)
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

OOS_DATA = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"
OUT = ROOT / "tmp/btc_shared_policy_v1_r8_oos"
ENTRY_THRESHOLD = .40
LARGE_THRESHOLD = .65
OOS_START, OOS_END = "2026-01-01", "2026-03-31 23:59:59+00:00"


def oos_market(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 2:
        raise ValueError("OOS frame needs at least two bars")
    out = frame.iloc[:-1].loc[:, ["timestamp"]].copy()
    opens = frame["open"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    out["next_bar_price_return"] = closes[1:] / opens[1:] - 1.0
    return out.reset_index(drop=True)


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    train_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    train_2025 = read_window(VAL_DATA, features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    train_labels = pd.concat([labels_for(train_2024), labels_for(train_2025)], ignore_index=True)
    train_base = pd.concat([train_2024, train_2025], ignore_index=True)
    train = train_base.merge(train_labels, left_on="timestamp", right_on="decision_timestamp", how="inner")
    model, mean, std = fit(train, features)

    oos_base = read_window(OOS_DATA, features, OOS_START, OOS_END)
    market = oos_market(oos_base)
    decision_features = oos_base.iloc[:-1].reset_index(drop=True)
    if not (decision_features["timestamp"] == market["timestamp"]).all():
        raise ValueError("OOS decision features and realized next-bar returns have mismatched timestamps")
    margins, qualities = predict_margins(
        model, decision_features, features, mean, std, ENTRY_THRESHOLD, LARGE_THRESHOLD,
    )
    metrics = simulate(margins, market["next_bar_price_return"].to_numpy())
    decisions = market.copy()
    decisions["target_margin_fraction"] = margins
    decisions["quality"] = qualities
    OUT.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(OUT / "oos_bar_by_bar_decisions.csv", index=False)
    report = {
        "diagnostic_only": True,
        "model_fit_period": ["2024-01-01", "2025-12-31"],
        "thresholds_frozen_from": str(R7_OUT / "report.json"),
        "entry_quality_threshold": ENTRY_THRESHOLD,
        "large_margin_quality_threshold": LARGE_THRESHOLD,
        "oos_period": [OOS_START, OOS_END],
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "teacher_labels_used_as_oos_input": False,
        "oos_base_rows": int(len(decision_features)),
        "metrics": metrics,
        "promotion_eligible": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
