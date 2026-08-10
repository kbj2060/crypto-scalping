#!/usr/bin/env python3
"""Run one split-correct SOL dual-component risk-sidecar candidate WITHOUT a frozen mapping.

Same as run_sol_dual_sidecar_candidate_20260729.py (same SPLIT_TS=2025-09-01 monkeypatch on the
parent/omega modules, same fixed hyperparameters), except --fixed-mapping-report is optional
instead of required -- omitting it lets the risk sidecar's own margin/leverage mapping grid
search run for real, instead of reusing one seed's (possibly overfit) frozen mapping. Built for
scripts/eval_sol_dual_router_seed_ensemble_retune_20260810.py's "genuine retune on the
seed-ensemble-averaged predictions" step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_sol_20260707 as sidecar  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--out-suffix", required=True)
    parser.add_argument("--direction-label-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fixed-mapping-report", type=Path, default=None)
    args = parser.parse_args()

    split = pd.Timestamp("2025-09-01")
    parent.SPLIT_TS = split
    omega.SPLIT_TS = split
    argv = [
        "train_eval_omega4_2_risk_sidecar_sol_20260707.py",
        "--baseline-bundle", str(args.parent_dir / "true_3head_tabm_bundle.pt"),
        "--precomputed-prediction-dir", str(args.parent_dir),
        "--precomputed-prediction-tag", str(args.tag),
        "--direction-label-dir", str(args.direction_label_dir),
        "--quality-threshold", str(args.quality_threshold),
        "--exit-threshold", "0.95",
        "--selection-scope", "validation_only",
        "--full-replay-top-k", "5",
        "--out-suffix", str(args.out_suffix),
        "--seed", str(args.seed),
        "--device", "cuda",
    ]
    if args.fixed_mapping_report is not None:
        argv.extend(("--fixed-mapping-report", str(args.fixed_mapping_report)))
    sys.argv = argv
    return int(sidecar.main())


if __name__ == "__main__":
    raise SystemExit(main())
