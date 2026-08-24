#!/usr/bin/env python3
"""`eth_directional_change_tabm_training_ilias_anchored_20260821.py`의 세 번째 변형 -- 그
파일의 "h1"/"full" variant 둘 다 2026-08-22에 확정된 유일 split 규약(TRAIN 2024-01-01~
2026-03-31 / VAL 2026 Q2 / OOS 2026-07-01~데이터상 최근일, 계약서 "## Dataset Split" 절)과
정확히 안 맞는다 -- "h1"은 VAL+OOS를 2026H1 하나로 뭉쳤고, "full"은 TRAIN을 2026H1까지 늘려
VAL 개념이 없어진다. 로직은 원본과 동일(TRAIN/EVAL/레짐오버레이 4개 경로만 로컬 오버라이드),
날짜 경계만 계약서 규약대로 새로 자른다.

**⚠️ 1차 정정(같은 날)**: 처음엔 "EVAL_CSV=VAL+OOS 병합, 후속 분석에서 날짜필터로 분리"로
설계했으나 실제로 단일 zigzag/seed 테스트를 돌려보고 **파이프라인이 2-way(train/eval)가
아니라는 걸 발견**했다 -- `_prepare_frames()`(train_eval_omega4_3head_parent72_loose_entry_
quality_20260620.py:377-389)가 `train_raw = train_all[ts < parent.SPLIT_TS]` /
`val_raw = train_all[ts >= parent.SPLIT_TS]`로 **TRAIN_CSV 자체를 내부에서 fit/validation
2조각으로 다시 쪼개고**, EVAL_CSV 전체를 그대로 "oos"라 부른다. `parent.SPLIT_TS`는
`train_eval_omega1_2_tabm_3head_20260603.py:33`에 `2025-10-01`로 **하드코딩**돼 있다(레짐/DC
공유 스크립트라 원본 수정 안 함, 이미 문서화된 관행). 이걸 모르고 그냥 TRAIN_CSV/EVAL_CSV
경계만 계약서 날짜로 바꾸면 파이프라인이 내부에서 다시 옛 경계(2025-10-01)로 잘라버려
"validation"이 실제론 2025-10~2026-03-31, "oos"가 실제론 VAL(Q2)+OOS(7월~) 병합이 되는 조용한
오염이 생긴다(실제로 1차 테스트 실행에서 이렇게 나온 걸 확인함).

**수정**: TRAIN_CSV = 2024-01-01~2026-06-30(fit+VAL-Q2 통째로 담음) + `parent.SPLIT_TS`를
2026-04-01로 로컬 오버라이드(공유 파일 무수정, TRAIN_CSV/EVAL_CSV와 같은 방식) → 파이프라인이
내부에서 알아서 fit=2024-01~2026-03-31 / "validation"=2026-04~06-30(Q2)으로 정확히 쪼갠다.
EVAL_CSV = 2026-07-01~데이터상 최근일만 순수하게 담아 "oos"=의도한 그대로가 되게 한다.

REGIME3_CURRENT 오버레이는 2026-08-21에 이미 재계산된 최신 pick(states=24/sticky=0.90)을
그대로 재사용한다(`tmp/ilias_labellogic_recheck_20260821/train_2024_2026H1_regime3_current_
states24_sticky090.csv` + `oos_20260701_20260819_regime3_current_states24_sticky090.csv` concat,
2024-01-01~2026-08-19 무결) -- DC154 트랜스포머 스모크테스트가 썼던 공유 canonical 경로의
구pick보다 낫다.

⚠️ `_prepare_frames()`의 `_read_labels(dir, 2025, ...)` 연도 하드코딩 문제는 원본 20260821
파일이 이미 우회해뒀다(라벨 디렉토리를 "2025"라는 이름 아래 병합데이터로 위장) -- 이 파일은
그 우회를 그대로 상속하고 데이터소스 4개 경로만 다시 갈아끼운다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import eth_dc_engineered_features_canonicaldata_20260820 as feat154  # noqa: E402

omega = feat154.omega
parent_script = feat154.parent_script
parent = parent_script.parent   # train_eval_omega1_2_tabm_3head_20260603 -- SPLIT_TS 소유 모듈

SCRATCH = ROOT / "tmp/ilias_labellogic_recheck_20260821"
SCRATCH.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2026-06-30 23:59:59"      # TRAIN_CSV 끝 = fit+VAL(Q2) 통째로 담음
EVAL_START = "2026-07-01"               # EVAL_CSV 시작 = 순수 OOS만(VAL과 안 섞음)
SPLIT_TS = pd.Timestamp("2026-04-01")   # parent.SPLIT_TS 로컬 오버라이드 -- 파이프라인이
                                          # TRAIN_CSV를 이 기준으로 fit(<)/validation(>=) 분리

_raw = pd.concat(
    [pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", low_memory=False)
     for y in ("2024", "2025", "2026_rebuilt")],
    ignore_index=True,
)
_raw["timestamp"] = pd.to_datetime(_raw["timestamp"])
_raw = _raw.sort_values("timestamp").reset_index(drop=True)
if _raw["timestamp"].duplicated().any():
    raise RuntimeError("2024+2025+2026 concat에 중복 timestamp 존재")

_train_raw = _raw[_raw["timestamp"] <= TRAIN_END].reset_index(drop=True)
_eval_raw = _raw[_raw["timestamp"] >= EVAL_START].reset_index(drop=True)

_RAW_TRAIN_CSV = SCRATCH / "train_2024_to_2026q2_canonical_raw.csv"
_RAW_EVAL_CSV = SCRATCH / "eval_2026q3_partial_canonical_raw.csv"
_train_raw.to_csv(_RAW_TRAIN_CSV, index=False)
_eval_raw.to_csv(_RAW_EVAL_CSV, index=False)
print(f"[ilias_anchored_canonical] TRAIN_CSV(fit+VAL-Q2): {len(_train_raw)} rows "
      f"[{_train_raw['timestamp'].min()}..{_train_raw['timestamp'].max()}], "
      f"SPLIT_TS={SPLIT_TS} -- fit=<SPLIT_TS, validation(Q2)=>=SPLIT_TS", flush=True)
print(f"[ilias_anchored_canonical] EVAL_CSV(순수 OOS): {len(_eval_raw)} rows "
      f"[{_eval_raw['timestamp'].min()}..{_eval_raw['timestamp'].max()}]", flush=True)

parent.SPLIT_TS = SPLIT_TS

_regime3_concat = pd.concat(
    [pd.read_csv(SCRATCH / "train_2024_2026H1_regime3_current_states24_sticky090.csv"),
     pd.read_csv(SCRATCH / "oos_20260701_20260819_regime3_current_states24_sticky090.csv")],
    ignore_index=True,
)
_regime3_concat["timestamp"] = pd.to_datetime(_regime3_concat["timestamp"])
_regime3_concat = _regime3_concat.sort_values("timestamp").reset_index(drop=True)
_regime3_train = _regime3_concat[_regime3_concat["timestamp"] <= TRAIN_END]
_regime3_eval = _regime3_concat[_regime3_concat["timestamp"] >= EVAL_START]
_REGIME_TRAIN_SIDE = SCRATCH / "train_2024_to_2026q1_regime3_current_states24_sticky090.csv"
_REGIME_EVAL_SIDE = SCRATCH / "eval_2026q2_plus_partial_q3_regime3_current_states24_sticky090.csv"
_regime3_train.to_csv(_REGIME_TRAIN_SIDE, index=False)
_regime3_eval.to_csv(_REGIME_EVAL_SIDE, index=False)

_PLACEHOLDER_DIR = SCRATCH / "placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

omega.TRAIN_CSV = _RAW_TRAIN_CSV
omega.EVAL_CSV = _RAW_EVAL_CSV
omega.REGIME3_CURRENT_2025 = _REGIME_TRAIN_SIDE
omega.REGIME3_CURRENT_2026 = _REGIME_EVAL_SIDE

print(f"[ilias_anchored_canonical] TRAIN_CSV={omega.TRAIN_CSV}", flush=True)
print(f"[ilias_anchored_canonical] EVAL_CSV={omega.EVAL_CSV}", flush=True)


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    out_path = _PLACEHOLDER_DIR / out_name
    ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
    for c in cols:
        ts[c] = 0.0
    ts.to_csv(out_path, index=False)
    return out_path


omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_train_canonical_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_eval_canonical_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_train_canonical_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_eval_canonical_zero.csv")

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
