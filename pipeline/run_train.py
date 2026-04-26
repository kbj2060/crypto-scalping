#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _run(cmd: list[str], *, dry_run: bool, label: str) -> None:
    print(f"[{label}] {_cmd_str(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-level training entrypoint: build RL dataset, then train RL."
    )
    p.add_argument("--python", default=sys.executable)

    p.add_argument("--features-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--sup-year", type=int, default=2024)
    p.add_argument("--rl-year", type=int, default=2025)
    p.add_argument("--split-dir", default="data/splits/year_oos")
    p.add_argument("--output-rl-path", default="")
    p.add_argument("--generate-rl-base", action="store_true")

    p.add_argument("--target", choices=["all", "supervised", "unsupervised"], default="all")
    p.add_argument("--use-optuna-runner", action="store_true")
    p.add_argument("--xgb-trials", type=int, default=40)
    p.add_argument("--supervised-trials", type=int, default=30)
    p.add_argument("--unsupervised-trials", type=int, default=25)
    p.add_argument("--vae-trials", type=int, default=20)
    p.add_argument("--vae-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--force-retune", action="store_true")
    p.add_argument("--skip-startup-check", action="store_true")
    p.add_argument("--startup-check-seconds", type=float, default=8.0)
    p.add_argument("--skip-completed", dest="skip_completed", action="store_true")
    p.set_defaults(skip_completed=False)
    p.add_argument("--skip-ensemble-train", action="store_true")
    p.add_argument("--skip-augment", action="store_true")

    p.add_argument("--rl-trainer", choices=["dsac"], default="dsac")
    p.add_argument("--rl-train-ratio", type=float, default=0.8)
    p.add_argument("--rl-episodes", type=int, default=1000)
    p.add_argument("--rl-startup-check-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    build_cmd = [
        args.python,
        "pipeline/build_rl_dataset.py",
        "--features-path",
        args.features_path,
        "--rl-path",
        args.rl_path,
        "--timestamp-col",
        args.timestamp_col,
        "--sup-year",
        str(args.sup_year),
        "--rl-year",
        str(args.rl_year),
        "--split-dir",
        args.split_dir,
        "--target",
        args.target,
        "--xgb-trials",
        str(args.xgb_trials),
        "--supervised-trials",
        str(args.supervised_trials),
        "--unsupervised-trials",
        str(args.unsupervised_trials),
        "--vae-trials",
        str(args.vae_trials),
        "--vae-device",
        args.vae_device,
        "--startup-check-seconds",
        str(args.startup_check_seconds),
    ]
    if args.output_rl_path:
        build_cmd.extend(["--output-rl-path", args.output_rl_path])
    if args.generate_rl_base:
        build_cmd.append("--generate-rl-base")
    if args.use_optuna_runner:
        build_cmd.append("--use-optuna-runner")
    if args.force_retune:
        build_cmd.append("--force-retune")
    if args.skip_startup_check:
        build_cmd.append("--skip-startup-check")
    if args.skip_completed:
        build_cmd.append("--skip-completed")
    if args.skip_ensemble_train:
        build_cmd.append("--skip-ensemble-train")
    if args.skip_augment:
        build_cmd.append("--skip-augment")
    if args.dry_run:
        build_cmd.append("--dry-run")

    _run(build_cmd, dry_run=args.dry_run, label="BUILD-RL-DATASET")

    out_rl_path = args.output_rl_path or f"{args.split_dir}/rl_training_{args.rl_year}_m7.csv"
    rl_script = "ensemble/train_rl_dsac_agent.py"
    rl_cmd = [
        args.python,
        rl_script,
        "--csv-path",
        out_rl_path,
        "--train-ratio",
        str(args.rl_train_ratio),
        "--episodes",
        str(args.rl_episodes),
    ]
    if args.rl_startup_check_only:
        rl_cmd.append("--startup-check-only")

    _run(rl_cmd, dry_run=args.dry_run, label="RL-TRAIN")

    print("[DONE] Unified training pipeline completed.")
    print(f"       trainer={args.rl_trainer}")
    print(f"       dataset={out_rl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
