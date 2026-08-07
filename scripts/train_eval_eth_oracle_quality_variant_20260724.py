#!/usr/bin/env python3
"""Train one sparse Oracle quality variant with OOS optionally withheld."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

if "--quality-variant" not in sys.argv:
    raise SystemExit("--quality-variant is required: q20, q10, or q05")
position = sys.argv.index("--quality-variant")
try:
    variant = sys.argv[position + 1]
except IndexError as exc:
    raise SystemExit("--quality-variant requires a value") from exc
if variant not in {"q20", "q10", "q05"}:
    raise SystemExit(f"invalid --quality-variant: {variant}")
del sys.argv[position : position + 2]

import train_eval_eth_split_oracle_3head_20260724 as pipeline  # noqa: E402


pipeline.MODEL_ID = f"eth_oracle_quality_{variant}_3head_noleak_20260724"
pipeline.LABEL_DIR = ROOT / f"tmp/causal_regen_20260516/eth_oracle_entry_quality_{variant}_20260724"
pipeline.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / pipeline.MODEL_ID
pipeline.QUALITY_TARGET_COLUMN = "oracle_quality_action"


if __name__ == "__main__":
    raise SystemExit(pipeline.main())
