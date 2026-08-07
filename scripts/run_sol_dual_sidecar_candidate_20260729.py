#!/usr/bin/env python3
"""Run one split-correct SOL dual-component risk-sidecar candidate."""
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
    parser.add_argument("--fixed-mapping-report", type=Path, required=True)
    args = parser.parse_args()

    split = pd.Timestamp("2025-09-01")
    parent.SPLIT_TS = split
    omega.SPLIT_TS = split
    sys.argv = [
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
        "--fixed-mapping-report", str(args.fixed_mapping_report),
        "--device", "cuda",
    ]
    return int(sidecar.main())


if __name__ == "__main__":
    raise SystemExit(main())
