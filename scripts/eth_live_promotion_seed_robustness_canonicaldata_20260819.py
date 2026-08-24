#!/usr/bin/env python3
"""eth_live_promotion_seed_robustness_prefix_snapshot_20260819(=git HEAD, exit_head 버그수정
이전 원본 코드)의 omega.TRAIN_CSV/EVAL_CSV를 canonical 데이터로 오버라이드한 판.

배경: 원본코드를 오늘(2026-08-19) 그대로(legacy CSV 기본값) 재실행하면 omega._numeric_
feature_cols가 라이브 번들 자신의 102 base_cols 중 7개(fibonacci_level/funding_roc_12/
funding_roc_48/funding_z_score/hurst_288/regime_persistence/short_squeeze_risk)를 못 찾아
RuntimeError -- legacy EVAL_CSV(2026-02-28까지)가 이 피쳐들의 overlay 커버리지 교집합에서
빠지는 것으로 추정(train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py가
이미 동일 문제를 겪고 해결한 전례와 정확히 같은 증상). 그 기존 해법(canonical TRAIN_CSV/
EVAL_CSV로 오버라이드 + cmamba/risk placeholder를 canonical 타임스탬프로 재생성)을 그대로
복제하되, wrapping 대상만 posfix(버그수정본) 대신 이 스냅샷(원본)으로 바꿨다 -- 로직은 재구현
아님, 상수 재할당 패턴 동일.

공유 모듈(`omega`) 기본값 자체는 안 건드림(로컬 오버라이드), placeholder 파일은 posfix 라인과
별도 디렉토리에 생성(동시세션 충돌 회피).

⚠️ 2026-08-19: 이 4개 placeholder 파일을 h48qual/zig075 x 신규시드 2개, 총 4개 프로세스가
동시에(handoff.sh launch 4회 병렬) import하면서 전부 같은 경로에 동시 쓰기 -- pandas가 부분
기록된 파일을 읽다가 파싱에러(2/4 발생, 재현됨). os.replace() 원자적 rename으로 수정:
각 프로세스가 자기 pid로 유니크한 임시파일에 쓴 뒤 최종경로로 원자적 교체 -- 어느 프로세스가
마지막에 rename하든 내용은 100% 동일(같은 정적 입력에서 같은 계산)하므로 순서 무관하게 안전."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_live_promotion_seed_robustness_prefix_snapshot_20260819 as parent_script  # noqa: E402

omega = parent_script.omega

_RAW_TRAIN_CSV = ROOT / "data/splits/year_oos/training_features_2025.csv"
_RAW_EVAL_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

_PLACEHOLDER_DIR = ROOT / "tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

# train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py와 동일 근거(REGIME3_
# CURRENT_2026의 95bar gap을 _overlay_required의 edge-only 허용치가 못 넘김) -- EVAL_CSV를
# 그 커버리지에 정확히 맞춰 사전필터링.
_current_2026 = pd.read_csv(omega.REGIME3_CURRENT_2026, usecols=["timestamp"], parse_dates=["timestamp"])
_current_2026_ts = set(_current_2026["timestamp"])

_eval_raw = pd.read_csv(_RAW_EVAL_CSV, low_memory=False)
_eval_raw["timestamp"] = pd.to_datetime(_eval_raw["timestamp"])
_eval_filtered = _eval_raw[_eval_raw["timestamp"].isin(_current_2026_ts)].sort_values("timestamp").reset_index(drop=True)
if len(_eval_filtered) != len(_current_2026_ts):
    raise RuntimeError(f"EVAL_CSV filtering did not reach full REGIME3_CURRENT_2026 coverage: {len(_eval_filtered)} vs {len(_current_2026_ts)}")
_FILTERED_EVAL_CSV = _PLACEHOLDER_DIR / "eval_2026_canonical_filtered_to_regime3_current_coverage.csv"
_tmp_path = _PLACEHOLDER_DIR / f"eval_2026_canonical_filtered_to_regime3_current_coverage.csv.tmp{os.getpid()}"
_eval_filtered.to_csv(_tmp_path, index=False)
os.replace(_tmp_path, _FILTERED_EVAL_CSV)
print(f"EVAL_CSV pre-filter: {len(_eval_raw)} raw rows -> {len(_eval_filtered)} rows matching REGIME3_CURRENT_2026 coverage "
      f"(range {_eval_filtered['timestamp'].min()}..{_eval_filtered['timestamp'].max()})", flush=True)

omega.TRAIN_CSV = _RAW_TRAIN_CSV
omega.EVAL_CSV = _FILTERED_EVAL_CSV


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    out_path = _PLACEHOLDER_DIR / out_name
    tmp_path = _PLACEHOLDER_DIR / f"{out_name}.tmp{os.getpid()}"
    ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
    for c in cols:
        ts[c] = 0.0
    ts.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    return out_path


omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2025_canonical_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2026_canonical_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_2025_canonical_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_2026_canonical_zero.csv")

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
