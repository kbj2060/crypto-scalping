#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    root = str(ROOT)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{prev}" if prev else root
    return env


def _run(cmd: list[str], dry_run: bool, label: str = "RUN") -> None:
    print(f"[{label}] {_cmd_str(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), env=_build_env(), check=True)


def _split_year(df: pd.DataFrame, year: int, ts_col: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    out = df.loc[ts.dt.year == int(year)].copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    out = out.dropna(subset=[ts_col]).sort_values(ts_col)
    out = out.drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    return out


def _save_df(df: pd.DataFrame, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    ts_min = pd.to_datetime(df["timestamp"], errors="coerce").min() if len(df) else None
    ts_max = pd.to_datetime(df["timestamp"], errors="coerce").max() if len(df) else None
    print(f"[SAVE] {name}: rows={len(df):,} cols={len(df.columns)} path={path}")
    print(f"       range={ts_min} -> {ts_max}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leakage-free pipeline: train SUP/UNSUP on 2024, build RL set on 2025, then train RL."
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--features-path", default="data/training_features_5m.csv")
    p.add_argument("--rl-path", default="data/rl_training_data_full.csv")
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--sup-year", type=int, default=2024)
    p.add_argument("--rl-year", type=int, default=2025)
    p.add_argument("--split-dir", default="data/splits/year_oos")
    p.add_argument("--output-rl-path", default="")

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

    p.add_argument("--rl-trainer", choices=["none", "dsac"], default="none")
    p.add_argument("--rl-train-ratio", type=float, default=0.8)
    p.add_argument("--rl-episodes", type=int, default=1000)
    p.add_argument("--rl-startup-check-only", action="store_true")

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    features_path = ROOT / args.features_path
    rl_path = ROOT / args.rl_path
    split_dir = ROOT / args.split_dir

    if not features_path.exists():
        raise FileNotFoundError(f"features csv not found: {features_path}")
    if not rl_path.exists():
        raise FileNotFoundError(f"rl csv not found: {rl_path}")

    feat_df = pd.read_csv(features_path)
    rl_df = pd.read_csv(rl_path)
    ts_col = args.timestamp_col
    if ts_col not in feat_df.columns:
        raise ValueError(f"timestamp col missing in features csv: {ts_col}")
    if ts_col not in rl_df.columns:
        raise ValueError(f"timestamp col missing in rl csv: {ts_col}")

    rl_no_m7 = rl_df.drop(columns=[c for c in rl_df.columns if c.startswith("m7_")], errors="ignore")

    feat_sup = _split_year(feat_df, args.sup_year, ts_col)
    feat_rl = _split_year(feat_df, args.rl_year, ts_col)
    rl_sup = _split_year(rl_no_m7, args.sup_year, ts_col)
    rl_base = _split_year(rl_no_m7, args.rl_year, ts_col)

    if len(feat_sup) == 0 or len(rl_sup) == 0:
        raise RuntimeError(f"empty SUP/UNSUP split for year={args.sup_year}")
    if len(feat_rl) == 0 or len(rl_base) == 0:
        raise RuntimeError(f"empty RL split for year={args.rl_year}")

    feat_sup_path = split_dir / f"training_features_{args.sup_year}.csv"
    feat_rl_path = split_dir / f"training_features_{args.rl_year}.csv"
    rl_sup_path = split_dir / f"rl_base_{args.sup_year}.csv"
    rl_base_path = split_dir / f"rl_base_{args.rl_year}.csv"
    out_rl_path = Path(args.output_rl_path) if args.output_rl_path else (split_dir / f"rl_training_{args.rl_year}_m7.csv")
    if not out_rl_path.is_absolute():
        out_rl_path = ROOT / out_rl_path

    _save_df(feat_sup, feat_sup_path, f"features_{args.sup_year}")
    _save_df(rl_sup, rl_sup_path, f"rl_base_{args.sup_year}")
    _save_df(feat_rl, feat_rl_path, f"features_{args.rl_year}")
    _save_df(rl_base, rl_base_path, f"rl_base_{args.rl_year}")

    if not args.skip_ensemble_train:
        runner = "scripts/train_all_ensemble_optuna.py" if args.use_optuna_runner else "scripts/train_all_ensemble.py"
        cmd = [
            args.python, runner,
            "--target", args.target,
            "--data-path", str(feat_sup_path.relative_to(ROOT)),
            "--rl-path", str(rl_sup_path.relative_to(ROOT)),
            "--xgb-trials", str(args.xgb_trials),
            "--supervised-trials", str(args.supervised_trials),
            "--unsupervised-trials", str(args.unsupervised_trials),
            "--vae-trials", str(args.vae_trials),
            "--vae-device", args.vae_device,
            "--startup-check-seconds", str(args.startup_check_seconds),
        ]
        if args.force_retune:
            cmd.append("--force-retune")
        if args.skip_startup_check:
            cmd.append("--skip-startup-check")
        if not args.skip_completed:
            cmd.append("--no-skip-completed")
        _run(cmd, args.dry_run, label="ENSEMBLE-TRAIN")

    if not args.skip_augment:
        cmd = [
            args.python,
            "pipeline/augment_m7_dataset.py",
            "--rl-path", str(rl_base_path.relative_to(ROOT)),
            "--feature-path", str(feat_rl_path.relative_to(ROOT)),
            "--output-path", str(out_rl_path.relative_to(ROOT)),
        ]
        _run(cmd, args.dry_run, label="AUGMENT-RL")

    if args.rl_trainer != "none":
        rl_script = "ensemble/train_rl_dsac_agent.py"
        cmd = [
            args.python,
            rl_script,
            "--csv-path", str(out_rl_path.relative_to(ROOT)),
            "--train-ratio", str(args.rl_train_ratio),
            "--episodes", str(args.rl_episodes),
        ]
        if args.rl_startup_check_only:
            cmd.append("--startup-check-only")
        _run(cmd, args.dry_run, label="RL-TRAIN")

    print("[DONE] Leakage-free year split pipeline completed.")
    print(f"       SUP/UNSUP year={args.sup_year}, RL year={args.rl_year}")
    print(f"       RL dataset={out_rl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
