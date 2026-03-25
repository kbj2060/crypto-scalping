#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    name: str
    script: str
    args: List[str]
    model_files: List[str]
    results_file: str | None = None


def _cmd_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    prev_pythonpath = env.get("PYTHONPATH", "")
    root_path = str(ROOT)
    env["PYTHONPATH"] = f"{root_path}{os.pathsep}{prev_pythonpath}" if prev_pythonpath else root_path
    return env


def _run(cmd: List[str], dry_run: bool, label: str = "RUN") -> None:
    print(f"[{label}] {_cmd_str(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=_build_env())


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    proc.wait(timeout=5)


def _startup_check(job: Job, cmd: List[str], dry_run: bool, timeout_sec: float) -> None:
    print(f"[STARTUP-CHECK] {job.name} ({timeout_sec:.1f}s)")
    print(f"[CHECK-CMD] {_cmd_str(cmd)}")
    if dry_run:
        return

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_build_env(),
        stdout=None,
        stderr=None,
        start_new_session=True,
    )
    start = time.monotonic()
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
            print(f"[STARTUP-CHECK] {job.name} exited cleanly during smoke window")
            break
        if time.monotonic() - start >= timeout_sec:
            print(f"[STARTUP-CHECK] {job.name} started successfully")
            break
        time.sleep(0.2)

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

    _remove_model_files(job.model_files, dry_run=False)
    _remove_results(job.results_file, dry_run=False)


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


def _remove_model_files(paths: List[str], dry_run: bool) -> None:
    for path_str in paths:
        p = ROOT / path_str
        if not p.exists():
            continue
        print(f"[CLEAN] remove {p}")
        if dry_run:
            continue
        p.unlink()


def _exists(path_str: str | None) -> bool:
    if not path_str:
        return False
    return (ROOT / path_str).exists()


def _job_completed(job: Job) -> bool:
    if not _exists(job.results_file):
        return False
    if not job.model_files:
        return False
    return all(_exists(p) for p in job.model_files)


def _build_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []

    if args.target in ("all", "supervised"):
        jobs.extend(
            [
                Job(
                    name="supervised/train_catboost_triple_barrier",
                    script="ensemble/supervised/train_catboost_triple_barrier.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    model_files=[
                        "data/ensemble/supervised/catboost_triple_barrier.pkl",
                        "data/ensemble/supervised/catboost_triple_barrier.json",
                    ],
                    results_file="data/ensemble/supervised/catboost_triple_barrier_training_results.json",
                ),
                Job(
                    name="supervised/train_multitarget_lgbm",
                    script="ensemble/supervised/train_multitarget_lgbm.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    model_files=[
                        "data/ensemble/supervised/multi_target_lgbm.pkl",
                        "data/ensemble/supervised/multi_target_lgbm.json",
                    ],
                    results_file="data/ensemble/supervised/multitarget_lgbm_training_results.json",
                ),
                Job(
                    name="supervised/train_trend_xgb",
                    script="ensemble/supervised/train_trend_xgb.py",
                    args=["--n-trials", str(args.xgb_trials)],
                    model_files=[
                        "data/trend_xgb/trend_xgb.json",
                        "data/trend_xgb/trend_xgb.pkl",
                    ],
                    results_file="data/trend_xgb/training_results.json",
                ),
                Job(
                    name="supervised/train_two_stage_stacking",
                    script="ensemble/supervised/train_two_stage_stacking.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    model_files=[
                        "data/ensemble/supervised/two_stage_stacking.pkl",
                        "data/ensemble/supervised/two_stage_stacking.json",
                    ],
                    results_file="data/ensemble/supervised/two_stage_stacking_training_results.json",
                ),
                Job(
                    name="supervised/train_quantile_forest",
                    script="ensemble/supervised/train_quantile_forest.py",
                    args=["--n-trials", str(args.supervised_trials)],
                    model_files=[
                        "data/ensemble/supervised/quantile_forest.pkl",
                        "data/ensemble/supervised/quantile_forest.json",
                    ],
                    results_file="data/ensemble/supervised/quantile_forest_training_results.json",
                ),
                Job(
                    name="supervised/train_tabnet_triple_barrier",
                    script="ensemble/supervised/train_tabnet_triple_barrier.py",
                    args=["--n-trials", str(args.tabnet_trials)],
                    model_files=[
                        "data/ensemble/supervised/tabnet_triple_barrier.json",
                        "data/ensemble/supervised/tabnet_triple_barrier.pkl",
                    ],
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
                    model_files=[
                        "data/ensemble/unsupervised/gmm_volatility.pkl",
                        "data/ensemble/unsupervised/gmm_volatility.json",
                    ],
                    results_file="data/ensemble/unsupervised/gmm_volatility_training_results.json",
                ),
                Job(
                    name="unsupervised/train_hdbscan_regime",
                    script="ensemble/unsupervised/train_hdbscan_regime.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    model_files=[
                        "data/ensemble/unsupervised/hdbscan_regime.pkl",
                        "data/ensemble/unsupervised/hdbscan_regime.json",
                    ],
                    results_file="data/ensemble/unsupervised/hdbscan_regime_training_results.json",
                ),
                Job(
                    name="unsupervised/train_isolation_forest",
                    script="ensemble/unsupervised/train_isolation_forest.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    model_files=[
                        "data/ensemble/unsupervised/isolation_forest.pkl",
                        "data/ensemble/unsupervised/isolation_forest.json",
                    ],
                    results_file="data/ensemble/unsupervised/isolation_forest_training_results.json",
                ),
                Job(
                    name="unsupervised/train_pca_umap_mapper",
                    script="ensemble/unsupervised/train_pca_umap_mapper.py",
                    args=["--n-trials", str(args.unsupervised_trials)],
                    model_files=[
                        "data/ensemble/unsupervised/pca_umap_mapper.pkl",
                        "data/ensemble/unsupervised/pca_umap_mapper.json",
                    ],
                    results_file="data/ensemble/unsupervised/pca_umap_mapper_training_results.json",
                ),
                Job(
                    name="unsupervised/train_vae_anomaly",
                    script="ensemble/unsupervised/train_vae_anomaly.py",
                    args=["--n-trials", str(args.vae_trials), "--device", args.vae_device],
                    model_files=[
                        "data/ensemble/unsupervised/vae_anomaly.pkl",
                        "data/ensemble/unsupervised/vae_anomaly.json",
                    ],
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
    p.add_argument("--xgb-trials", type=int, default=60)
    p.add_argument("--supervised-trials", type=int, default=60)
    p.add_argument("--tabnet-trials", type=int, default=60)
    p.add_argument("--unsupervised-trials", type=int, default=60)
    p.add_argument("--vae-trials", type=int, default=60)
    p.add_argument("--vae-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help="deprecated: no-op (existing files are reused by default)",
    )
    p.add_argument(
        "--force-retune",
        action="store_true",
        help="Delete existing model/results files and run all selected jobs again",
    )
    p.add_argument(
        "--no-skip-completed",
        dest="skip_completed",
        action="store_false",
        help="Run jobs even when model/results files already exist",
    )
    p.set_defaults(skip_completed=True)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument(
        "--skip-startup-check",
        action="store_true",
        help="Skip per-job startup smoke tests before full training",
    )
    p.add_argument(
        "--startup-check-seconds",
        type=float,
        default=8.0,
        help="Seconds to wait while verifying each job starts without immediate errors",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    jobs = _build_jobs(args)

    if not jobs:
        print("No jobs selected.")
        return 1

    if args.force_retune:
        print("[INFO] force_retune=True -> existing model/results files will be removed")
        for job in jobs:
            _remove_model_files(job.model_files, args.dry_run)
            _remove_results(job.results_file, args.dry_run)
    else:
        print("[INFO] existing model/results files are kept")

    failures: List[str] = []
    skipped: List[str] = []

    if not args.skip_startup_check:
        print(f"[INFO] startup smoke check enabled ({args.startup_check_seconds:.1f}s per job)")
        for i, job in enumerate(jobs, start=1):
            print(f"\n[CHECK {i}/{len(jobs)}] {job.name}")
            cmd = [args.python, job.script, *job.args]
            try:
                _startup_check(job, cmd, args.dry_run, args.startup_check_seconds)
            except subprocess.CalledProcessError as e:
                msg = f"{job.name} startup check failed (exit={e.returncode})"
                print(f"[ERROR] {msg}")
                failures.append(msg)
                if not args.continue_on_error:
                    return e.returncode
    else:
        print("[INFO] startup smoke check skipped")

    for i, job in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {job.name}")
        if args.skip_completed and _job_completed(job):
            print("[SKIP] model + results files already exist")
            skipped.append(job.name)
            continue
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
        if skipped:
            print("[SUMMARY] skipped jobs:")
            for name in skipped:
                print(f"- {name}")
        return 1

    if skipped:
        print("\n[SUMMARY] skipped jobs:")
        for name in skipped:
            print(f"- {name}")

    print("\n[SUMMARY] all jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
