#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import train_eval_omega1_2_1_multihorizon_tb_hgb_20260620 as hgb


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "omega1_2_1_dp_trajectory_hgb_no_maxhold_20260620"
SOURCE_ID = "omega1_2_1_dp_trajectory_daytrade_20260620"

hgb.MODEL_ID = "omega1_2_1_dp_trajectory_hgb_20260620"
hgb.RUN_ID = RUN_ID
hgb.SOURCE_ID = SOURCE_ID
hgb.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / RUN_ID
hgb.SOURCE_DIR = ROOT / "tmp/causal_regen_20260516" / SOURCE_ID
hgb.LABEL_2025 = hgb.SOURCE_DIR / "train_2025_multihorizon_tb_labels.csv"
hgb.LABEL_MIN_UTILITY_GRID = [0.00020, 0.00050]
hgb.TRADE_THRESHOLD_GRID = [0.45, 0.55]
hgb.SIDE_EDGE_MIN_GRID = [0.03]
hgb.PARENT_MIN_QUALITY_GRID = [0.0, 0.65]
hgb.MARGIN_FRACTION_GRID = [0.025]


if __name__ == "__main__":
    hgb.main()
