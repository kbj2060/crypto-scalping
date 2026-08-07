#!/usr/bin/env python3
"""Retrain the current ETH Omega4 3-head TabM with oracle trajectory targets."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as trainer  # noqa: E402


MODEL_ID = "eth_omega4_3head_oracle_trajectory_20260724"
LABEL_DIR = (
    ROOT / "tmp/causal_regen_20260516/eth_oracle_trajectory_action_labels_20260724"
)

trainer.MODEL_ID = MODEL_ID
trainer.LABEL_DIR = LABEL_DIR
trainer.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def add_default_args() -> None:
    defaults = {
        "--epochs": "2",
        "--exit-label-mode": "entry_label_terminal_giveback",
        "--quality-thresholds": "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
        "--direction-label-dir": str(LABEL_DIR),
        "--quality-mode": "same_as_direction",
        "--max-exit-samples": "30000",
        "--max-train-rows": "0",
        "--cost-mult": "1.0",
        "--seed": "260724",
        "--out-suffix": "e2_fulltrain_exit30k_cost1",
        "--device": "cuda",
    }
    for key, value in defaults.items():
        if key not in sys.argv:
            sys.argv.extend([key, value])


if __name__ == "__main__":
    add_default_args()
    raise SystemExit(trainer.main())
