#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    name: str
    script: str
    args: List[str]
    results_file: str | None = None


def _cmd_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(cmd: List[str], dry_run: bool) -> None:
    print(f"[RUN] {_cmd_str(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _remove_results(path_str: str | None, dry_run: bool) -> None:
    if not path_str:
        return
    p = ROOT / path_str
    if not p.exists():
        return
    print(f"[CLEAN] remove {p}")
    if dry_run:
        return
    p.unlink()


def _build_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []

    if args.target in ("all", "supervised"):
        jobs.extend(
            [
                Job(
                    name="supervised/train_trend_xgb",
                    script="ensemble/supervised/train_trend_xgb.py",
                    args=["--n-trials", str(args.xgb_trials)],
                    results_file="data/trend_xgb/training_results.json",
                ),
                Job(
                    name="supervised/train_catboost_triple_barrier",
                    script="ensemble/supervised/train_catboost_triple_barrier.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    results_file="data/ensemble/supervised/catboost_triple_barrier_training_results.json",
                ),
                Job(
                    name="supervised/train_multitarget_lgbm",
                    script="ensemble/supervised/train_multitarget_lgbm.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    results_file="data/ensemble/supervised/multitarget_lgbm_training_results.json",
                ),
                Job(
                    name="supervised/train_two_stage_stacking",
                    script="ensemble/supervised/train_two_stage_stacking.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    results_file="data/ensemble/supervised/two_stage_stacking_training_results.json",
                ),
                Job(
                    name="supervised/train_quantile_forest",
                    script="ensemble/supervised/train_quantile_forest.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    results_file="data/ensemble/supervised/quantile_forest_training_results.json",
                ),
                Job(
                    name="supervised/train_tabnet_triple_barrier",
                    script="ensemble/supervised/train_tabnet_triple_barrier.py",
                    args=["--n-trials", str(args.tabnet_trials)],
                    results_file="data/ensemble/supervised/tabnet_triple_barrier_training_results.json",
                ),
            ]
        )

    if args.target in ("all", "unsupervised"):
        jobs.extend(
            [
                Job(
                    name="unsupervised/train_gmm_volatility",
                    script="ensemble/unsupervised/train_gmm_volatility.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    results_file="data/ensemble/unsupervised/gmm_volatility_training_results.json",
                ),
                Job(
                    name="unsupervised/train_hdbscan_regime",
                    script="ensemble/unsupervised/train_hdbscan_regime.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    results_file="data/ensemble/unsupervised/hdbscan_regime_training_results.json",
                ),
                Job(
                    name="unsupervised/train_isolation_forest",
                    script="ensemble/unsupervised/train_isolation_forest.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    results_file="data/ensemble/unsupervised/isolation_forest_training_results.json",
                ),
                Job(
                    name="unsupervised/train_pca_umap_mapper",
                    script="ensemble/unsupervised/train_pca_umap_mapper.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    results_file="data/ensemble/unsupervised/pca_umap_mapper_training_results.json",
                ),
                Job(
                    name="unsupervised/train_vae_anomaly",
                    script="ensemble/unsupervised/train_vae_anomaly.py",
                    args=["--n-trials", str(args.vae_trials), "--device", args.vae_device],
                    results_file="data/ensemble/unsupervised/vae_anomaly_training_results.json",
                ),
            ]
        )

    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Optuna tuning + training for all ensemble supervised/unsupervised models"
    )
    p.add_argument("--target", choices=["all", "supervised", "unsupervised"], default="all")
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--xgb-trials", type=int, default=100)
    p.add_argument("--supervised-trials", type=int, default=100)
    p.add_argument("--tabnet-trials", type=int, default=100)
    p.add_argument("--unsupervised-trials", type=int, default=100)
    p.add_argument("--vae-trials", type=int, default=100)
    p.add_argument("--vae-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help="Do not delete existing *_training_results.json before running",
    )
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jobs = _build_jobs(args)

    if not jobs:
        print("No jobs selected.")
        return 1

    if not args.reuse_existing_results:
        print("[INFO] existing training_results files will be removed (fresh Optuna tuning)")
        for job in jobs:
            _remove_results(job.results_file, args.dry_run)
    else:
        print("[INFO] reuse_existing_results=True (hash matched jobs may skip Optuna)")

    failures: List[str] = []
    for i, job in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {job.name}")
        cmd = [args.python, job.script, *job.args]
        try:
            _run(cmd, args.dry_run)
        except subprocess.CalledProcessError as e:
            msg = f"{job.name} failed (exit={e.returncode})"
            print(f"[ERROR] {msg}")
            failures.append(msg)
            if not args.continue_on_error:
                return e.returncode

    if failures:
        print("\n[SUMMARY] completed with failures:")
        for msg in failures:
            print(f"- {msg}")
        return 1

    print("\n[SUMMARY] all jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
