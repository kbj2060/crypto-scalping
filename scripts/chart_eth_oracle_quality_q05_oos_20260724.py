#!/usr/bin/env python3
"""Materialize the validation-selected Oracle Q05 quality model OOS chart."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_split_oracle_3head_20260724 as trained  # noqa: E402


trained.MODEL_ID = "eth_oracle_quality_q05_3head_noleak_20260724"
trained.LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_oracle_entry_quality_q05_20260724"
trained.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / trained.MODEL_ID
trained.QUALITY_TARGET_COLUMN = "oracle_quality_action"

import chart_eth_split_oracle_oos_20260724 as chart  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(chart.main())
