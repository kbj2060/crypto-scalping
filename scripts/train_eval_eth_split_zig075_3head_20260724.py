#!/usr/bin/env python3
"""Run the exact split-Oracle student pipeline with split-local Zig075 targets."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_split_oracle_3head_20260724 as pipeline  # noqa: E402


pipeline.MODEL_ID = "eth_split_zig075_3head_noleak_20260724"
pipeline.LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_split_zig075_labels_20260724"
pipeline.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / pipeline.MODEL_ID


if __name__ == "__main__":
    raise SystemExit(pipeline.main())
