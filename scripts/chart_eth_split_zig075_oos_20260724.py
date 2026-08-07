#!/usr/bin/env python3
"""Materialize the frozen split-local Zig075 student's OOS chart."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_split_oracle_3head_20260724 as trained  # noqa: E402


trained.MODEL_ID = "eth_split_zig075_3head_noleak_20260724"
trained.LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_split_zig075_labels_20260724"
trained.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / trained.MODEL_ID

import chart_eth_split_oracle_oos_20260724 as chart  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(chart.main())
