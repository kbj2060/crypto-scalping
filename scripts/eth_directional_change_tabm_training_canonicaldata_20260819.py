#!/usr/bin/env python3
"""ETH Directional-Change(DC) 라벨로 TabM 3-head를 학습시키기 위한 canonical 데이터 오버라이드
래퍼. `train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py`와 로직이 동일
하다(재구현 아님, 그 파일이 정확히 wrapping해야 할 대상 -- 작업트리 현재본
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py` -- 을 이미 wrapping하고
있어 대상 교체가 필요 없음) -- `_PLACEHOLDER_DIR`만 별도 경로로 분리해 h48qual/zig075
재현작업과의 동시 실행 시 placeholder CSV 쓰기 경합을 피한다.

공유 모듈(`omega`) 기본값 자체는 안 건드림(로컬 오버라이드). omega.TRAIN_CSV/EVAL_CSV 기본값
(legacy alpha6 trade-candidate CSV)은 feature drift가 있고 2026년이 02-28에서 끊겨 canonical
Fresh-Forward OOS(2026-01~03)를 커버 못 한다 -- canonical `data/splits/year_oos/
training_features_{2025,2026_rebuilt}.csv`(DC 라벨을 빌드할 때 쓴 것과 동일 파일)로 대체한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

_RAW_TRAIN_CSV = ROOT / "data/splits/year_oos/training_features_2025.csv"
_RAW_EVAL_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

_PLACEHOLDER_DIR = ROOT / "tmp/causal_regen_20260516/eth_directional_change_tabm_training_20260819/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

# REGIME3_CURRENT_2026 커버리지(2026-01-01..06-30, 95bar gap 有)에 맞춰 EVAL_CSV를 사전 필터링
# -- _overlay_required의 tolerance가 edge-only라 raw canonical EVAL_CSV를 그대로 주면
# "non-edge missing timestamps"로 RuntimeError. 2025/TRAIN_CSV는 REGIME3_CURRENT_2025와
# 이미 완전히 일치해 필터링 불필요(posfix 래퍼가 이미 확인한 사실, 재확인 안 함).
_current_2026 = pd.read_csv(omega.REGIME3_CURRENT_2026, usecols=["timestamp"], parse_dates=["timestamp"])
_current_2026_ts = set(_current_2026["timestamp"])

_eval_raw = pd.read_csv(_RAW_EVAL_CSV, low_memory=False)
_eval_raw["timestamp"] = pd.to_datetime(_eval_raw["timestamp"])
_eval_filtered = _eval_raw[_eval_raw["timestamp"].isin(_current_2026_ts)].sort_values("timestamp").reset_index(drop=True)
if len(_eval_filtered) != len(_current_2026_ts):
    raise RuntimeError(f"EVAL_CSV filtering did not reach full REGIME3_CURRENT_2026 coverage: {len(_eval_filtered)} vs {len(_current_2026_ts)}")
_FILTERED_EVAL_CSV = _PLACEHOLDER_DIR / "eval_2026_canonical_filtered_to_regime3_current_coverage.csv"
_eval_filtered.to_csv(_FILTERED_EVAL_CSV, index=False)
print(f"EVAL_CSV pre-filter: {len(_eval_raw)} raw rows -> {len(_eval_filtered)} rows matching REGIME3_CURRENT_2026 coverage "
      f"(range {_eval_filtered['timestamp'].min()}..{_eval_filtered['timestamp'].max()})", flush=True)

omega.TRAIN_CSV = _RAW_TRAIN_CSV
omega.EVAL_CSV = _FILTERED_EVAL_CSV


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    out_path = _PLACEHOLDER_DIR / out_name
    ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
    for c in cols:
        ts[c] = 0.0
    ts.to_csv(out_path, index=False)
    return out_path


# REGIME3_CMAMBA/RISK는 진짜 데이터가 canonical 타임스탬프 범위를 못 덮어 0-fill placeholder로
# 대체(cmamba/risk 피쳐 자체를 실제로 쓰는 라이브 번들이 없다는 게 이미 확인돼 있음 -- posfix
# 래퍼 docstring 24-25행 참고). REGIME3_CURRENT(방금 위에서 다룬 실제 HMM 레짐 확률)는 건드리지
# 않는다 -- 단일모델+레짐피쳐 설계(계획 문서 참고)가 바로 이 컬럼을 피쳐로 쓴다.
omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2025_canonical_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2026_canonical_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_2025_canonical_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_2026_canonical_zero.csv")

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
