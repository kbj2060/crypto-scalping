#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_omega1_direction_head_direction_only_20260602 as direction_base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


MODEL_ID = "omega1_2_true_3head_tabm_fast_horizon_20260620"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_fast_horizon_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def main() -> int:
    if not LABEL_DIR.exists():
        raise RuntimeError(f"missing fast-horizon label dir: {LABEL_DIR}")
    direction_base.LABEL_DIR = LABEL_DIR
    parent.MODEL_ID = MODEL_ID
    parent.OUT_DIR = OUT_DIR
    sys.argv = [
        sys.argv[0],
        "--epochs",
        "4",
        "--max-train-rows",
        "30000",
        "--max-exit-samples",
        "12000",
        "--quality-threshold",
        "0.45",
        "--thresholds",
        "0.45,0.55,0.65,0.75,0.85,0.95",
        "--out-suffix",
        "smoke_e4_train30k_exit12k",
        "--device",
        "cpu",
    ]
    return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())
