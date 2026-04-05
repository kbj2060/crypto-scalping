#!/usr/bin/env python3
"""
데이터 파이프라인 통합 관리 스크립트.

스테이지:
  0  generate-rl-data  — elite/regime/volatility 피처 계산 → rl_training_data_full.csv
  1  split-year        — CSV를 연도별로 분리 (supervised 2024, RL 2025)
  2  train-ensemble    — 앙상블 M7 모델 9종 학습 (supervised + unsupervised)
  3  augment-rl        — SevenModelEnsemble 예측값 추가 → rl_training_2025_m7.csv

사용 예:
  # 전체 실행
  python scripts/run_pipeline.py

  # 2번 스테이지부터 실행
  python scripts/run_pipeline.py --from-stage 2

  # 특정 스테이지만
  python scripts/run_pipeline.py --stage 3

  # 명령어만 출력 (실제 실행 없음)
  python scripts/run_pipeline.py --dry-run

  # 커스텀 설정 파일
  python scripts/run_pipeline.py --config config/pipeline.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _ROOT / "config" / "pipeline.yaml"

STAGE_NAMES = {
    0: "generate-rl-data",
    1: "split-year",
    2: "train-ensemble",
    3: "augment-rl",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# 설정 헬퍼
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"pipeline.yaml must be a mapping, got {type(cfg)}")
    return cfg


def _p(cfg: dict, *keys: str) -> Any:
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"pipeline.yaml 누락 키: {'.'.join(keys)}")
        node = node[k]
    return node


def _resolve_date(val: str) -> str:
    return str(date.today()) if str(val).strip().lower() == "today" else str(val)


# ---------------------------------------------------------------------------
# 서브프로세스 실행
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, dry_run: bool = False, label: str = "") -> None:
    display = " ".join(str(c) for c in cmd)
    log.info("[%s] %s", label, display)
    if dry_run:
        return
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    result = subprocess.run(cmd, env=env, cwd=str(_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"명령 실패 (exit {result.returncode}): {display}")


# ---------------------------------------------------------------------------
# 검증 게이트
# ---------------------------------------------------------------------------

def _validate_csv(path: str, min_rows: int = 0, min_cols: int = 0, label: str = "") -> None:
    import csv

    p = _ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"검증 실패 [{label}]: 파일 없음: {p}")

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"검증 실패 [{label}]: 빈 파일: {p}")
        ncols = len(header)
        nrows = sum(1 for _ in reader)

    if min_rows and nrows < min_rows:
        raise ValueError(f"검증 실패 [{label}]: {p.name} rows={nrows} < 최소 {min_rows}")
    if min_cols and ncols < min_cols:
        raise ValueError(f"검증 실패 [{label}]: {p.name} cols={ncols} < 최소 {min_cols}")
    log.info("[validate] %-45s  rows=%-7d  cols=%d  ✓", p.name, nrows, ncols)


# ---------------------------------------------------------------------------
# 스테이지 구현
# ---------------------------------------------------------------------------

def _stage_generate_rl_data(cfg: dict, *, dry_run: bool) -> None:
    sc = _p(cfg, "generate_rl_data")
    _run(
        [
            sys.executable, "scripts/generate_training_data.py",
            "--input",  sc["input_csv"],
            "--output", sc["output_csv"],
        ],
        dry_run=dry_run,
        label="stage-0",
    )
    if not dry_run:
        val = _p(cfg, "validation")
        _validate_csv(
            sc["output_csv"],
            min_rows=val.get("rl_full_csv_min_rows", 0),
            label="rl_full_csv",
        )


def _stage_split_year(cfg: dict, *, dry_run: bool) -> None:
    sc = _p(cfg, "split_year")
    paths = _p(cfg, "paths")
    sup_year: int = int(sc["supervised_year"])
    rl_year: int = int(sc["rl_year"])
    ts_col: str = sc["timestamp_col"]

    features_csv = _ROOT / paths["features_csv"]
    rl_full_csv  = _ROOT / paths["rl_full_csv"]
    split_dir    = _ROOT / paths["split_dir"]

    if dry_run:
        log.info(
            "[stage-1] (dry-run) %s / %s 를 %d / %d 년도로 분리",
            features_csv.name, rl_full_csv.name, sup_year, rl_year,
        )
        return

    import pandas as pd

    split_dir.mkdir(parents=True, exist_ok=True)

    def _split_save(src: Path, year_a: int, name_a: str, year_b: int, name_b: str) -> None:
        log.info("[stage-1] 로드 중: %s ...", src.name)
        if not src.exists():
            raise FileNotFoundError(f"분리 대상 파일 없음: {src}")
        df = pd.read_csv(src)
        if ts_col not in df.columns:
            raise KeyError(f"타임스탬프 컬럼 '{ts_col}' 없음: {src.name}")
        df[ts_col] = pd.to_datetime(df[ts_col], errors="raise")
        for year, name in [(year_a, name_a), (year_b, name_b)]:
            out = split_dir / name
            sub = df[df[ts_col].dt.year == year].reset_index(drop=True)
            sub.to_csv(out, index=False)
            log.info("[stage-1] 저장: %-50s  rows=%d", out.name, len(sub))

    _split_save(
        features_csv,
        sup_year, f"training_features_{sup_year}.csv",
        rl_year,  f"training_features_{rl_year}.csv",
    )
    _split_save(
        rl_full_csv,
        sup_year, f"rl_base_{sup_year}.csv",
        rl_year,  f"rl_base_{rl_year}.csv",
    )


def _stage_train_ensemble(cfg: dict, *, dry_run: bool) -> None:
    sc = _p(cfg, "train_ensemble")
    paths = _p(cfg, "paths")

    script = (
        "scripts/train_all_ensemble_optuna.py"
        if sc.get("use_optuna", True)
        else "scripts/train_all_ensemble.py"
    )
    cmd = [
        sys.executable, script,
        "--target",              sc.get("target", "all"),
        "--data-path",           paths["features_sup_csv"],
        "--rl-path",             paths["rl_base_sup_csv"],
        "--xgb-trials",          str(sc.get("xgb_trials", 40)),
        "--supervised-trials",   str(sc.get("supervised_trials", 30)),
        "--unsupervised-trials", str(sc.get("unsupervised_trials", 25)),
        "--vae-trials",          str(sc.get("vae_trials", 20)),
        "--vae-device",          sc.get("vae_device", "auto"),
    ]
    if sc.get("skip_completed", True):
        cmd.append("--skip-completed")
    if sc.get("continue_on_error", False):
        cmd.append("--continue-on-error")

    _run(cmd, dry_run=dry_run, label="stage-2")


def _stage_augment_rl(cfg: dict, *, dry_run: bool) -> None:
    sc = _p(cfg, "augment_rl")
    paths = _p(cfg, "paths")
    output = sc.get("output_path") or paths["rl_m7_csv"]

    _run(
        [
            sys.executable, "scripts/augment_rl_training_with_model7.py",
            "--rl-path",      sc["rl_path"],
            "--feature-path", sc["feature_path"],
            "--output-path",  output,
        ],
        dry_run=dry_run,
        label="stage-3",
    )
    if not dry_run:
        val = _p(cfg, "validation")
        _validate_csv(
            output,
            min_rows=val.get("rl_m7_csv_min_rows", 0),
            min_cols=val.get("rl_m7_csv_min_cols", 0),
            label="rl_m7_csv",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="데이터 파이프라인 통합 관리 (stage 0-3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k}  {v}" for k, v in STAGE_NAMES.items()),
    )
    p.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="pipeline.yaml 경로 (기본값: config/pipeline.yaml)",
    )
    p.add_argument(
        "--stage", type=int, default=None,
        help="단일 스테이지만 실행 (0-3)",
    )
    p.add_argument(
        "--from-stage", type=int, default=0,
        help="시작 스테이지 (포함, 기본값 0)",
    )
    p.add_argument(
        "--to-stage", type=int, default=3,
        help="종료 스테이지 (포함, 기본값 3)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="실제 실행 없이 명령어만 출력",
    )
    p.add_argument(
        "--list-stages", action="store_true",
        help="스테이지 목록 출력 후 종료",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.list_stages:
        for k, v in STAGE_NAMES.items():
            print(f"  {k}  {v}")
        return 0

    cfg = _load_config(Path(args.config))

    if args.stage is not None:
        from_stage = to_stage = args.stage
    else:
        from_stage = args.from_stage
        to_stage   = args.to_stage

    max_stage = max(STAGE_NAMES)
    if from_stage < 0 or to_stage > max_stage or from_stage > to_stage:
        log.error("유효하지 않은 스테이지 범위: %d-%d  (0-%d)", from_stage, to_stage, max_stage)
        return 1

    log.info("설정: %s", args.config)
    log.info("실행 범위: stage %d → %d  dry_run=%s", from_stage, to_stage, args.dry_run)

    stage_fns = {
        0: lambda: _stage_generate_rl_data(cfg, dry_run=args.dry_run),
        1: lambda: _stage_split_year(cfg, dry_run=args.dry_run),
        2: lambda: _stage_train_ensemble(cfg, dry_run=args.dry_run),
        3: lambda: _stage_augment_rl(cfg, dry_run=args.dry_run),
    }

    for stage_id in range(from_stage, to_stage + 1):
        name = STAGE_NAMES[stage_id]
        log.info("=" * 60)
        log.info("STAGE %d: %s", stage_id, name)
        log.info("=" * 60)
        try:
            stage_fns[stage_id]()
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            log.error("Stage %d (%s) 실패: %s", stage_id, name, exc)
            return 1

    log.info("완료.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
