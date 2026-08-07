#!/usr/bin/env python3
"""Train the current ETH Omega4 risk sidecar on the frozen q040 oracle parent."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as risk  # noqa: E402


MODEL_ID = "eth_omega4_oracle_risk_sidecar_20260724"
PARENT_DIR = (
    ROOT
    / "tmp/causal_regen_20260516/eth_omega4_3head_oracle_trajectory_20260724_e2_fulltrain_exit30k_cost1"
)
LABEL_DIR = (
    ROOT / "tmp/causal_regen_20260516/eth_oracle_trajectory_action_labels_20260724"
)
TRAIN_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48"
    / "02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025"
    / "trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
)
EVAL_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
    / "trade_candidates_2026_alpha6_current_tail111_exact.csv"
)

risk.MODEL_ID = MODEL_ID
risk.OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def add_default_args() -> None:
    valued = {
        "--baseline-bundle": str(PARENT_DIR / "true_3head_tabm_bundle.pt"),
        "--precomputed-prediction-dir": str(PARENT_DIR),
        "--precomputed-prediction-tag": "q040",
        "--direction-label-dir": str(LABEL_DIR),
        "--quality-mode": "same_as_direction",
        "--train-csv": str(TRAIN_CSV),
        "--eval-csv": str(EVAL_CSV),
        "--quality-threshold": "0.40",
        "--exit-threshold": "0.95",
        "--cost-mult": "1.0",
        "--risk-feature-mode": "parent_outputs",
        "--selection-objective": "log_risk",
        "--selection-scope": "validation_only",
        "--log-tail-penalty": "0.5",
        "--max-validation-mdd-abs": "25.0",
        "--min-validation-avg-notional": "0.45",
        "--max-validation-avg-notional": "0.95",
        "--full-replay-top-k": "1",
        "--seed": "260724",
        "--device": "cuda",
        "--out-suffix": "q040_valonly_logrisk_livegrid_cost1",
    }
    flags = {
        "--side-split-model",
        "--dynamic-leverage",
        "--require-dynamic-leverage-mapping",
        "--live-exposure-grid",
    }
    for key, value in valued.items():
        if key not in sys.argv:
            sys.argv.extend([key, value])
    for flag in flags:
        if flag not in sys.argv:
            sys.argv.append(flag)


if __name__ == "__main__":
    add_default_args()
    raise SystemExit(risk.main())
