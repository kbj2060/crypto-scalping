#!/usr/bin/env python3
"""Train one SOL dual-component parent with the required Sep-2025 split."""
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
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as trainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction-label-dir", type=Path, required=True)
    parser.add_argument("--quality-mode", choices=("same_as_direction", "quality_label_action"), required=True)
    parser.add_argument("--quality-label-dir", type=Path)
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-suffix", required=True)
    args = parser.parse_args()
    if args.quality_mode == "quality_label_action" and args.quality_label_dir is None:
        raise ValueError("--quality-label-dir is required for quality_label_action")
    if args.quality_mode == "same_as_direction" and args.quality_label_dir is not None:
        raise ValueError("--quality-label-dir must be omitted for same_as_direction")

    split = pd.Timestamp("2025-09-01")
    parent.SPLIT_TS = split
    omega.SPLIT_TS = split
    argv = [
        "train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py",
        "--direction-label-dir", str(args.direction_label_dir),
        "--quality-mode", args.quality_mode,
        "--quality-thresholds", str(args.quality_threshold),
        "--max-exit-samples", "30000",
        "--seed", str(args.seed),
        "--out-suffix", args.out_suffix,
        "--device", "cuda",
    ]
    if args.quality_label_dir is not None:
        argv.extend(("--quality-label-dir", str(args.quality_label_dir)))
    sys.argv = argv
    return int(trainer.main())


if __name__ == "__main__":
    raise SystemExit(main())
