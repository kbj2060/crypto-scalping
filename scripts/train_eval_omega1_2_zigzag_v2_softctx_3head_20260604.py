#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import train_eval_omega1_2_zigzag_softctx_3head_20260604 as base


ROOT = Path(__file__).resolve().parents[1]

base.MODEL_ID = "omega1_2_zigzag_v2_execaware_softctx_3head_20260604"
base.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / base.MODEL_ID
base.LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_v2_execaware_20260604"


if __name__ == "__main__":
    raise SystemExit(base.main())
