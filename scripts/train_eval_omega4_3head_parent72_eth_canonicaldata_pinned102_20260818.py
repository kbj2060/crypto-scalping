#!/usr/bin/env python3
"""h48qual/zig075 posfix 재학습 -- canonical 데이터 + base_cols를 원본(6/29·6/30, 102개)에
정확히 고정한 버전.

배경: canonical 데이터로 재학습한 posfix 번들(base_cols=158)을 baseline(원본, 102)과 Fresh-
Forward로 비교했더니 oos_confirm 2창 중 oos_q1에서 크게 악화되어 REJECTED_SIGN_MISMATCH
판정을 받았음(docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_
20260818.md 후속 세션 8). 그런데 이 비교는 "버그수정" 하나만 격리된 게 아니라 "버그수정 +
56개 신규피쳐(canonical 파이프라인 자연진화분)"가 섞인 결과였음 -- 사용자 지시로 피쳐셋
자체를 원본과 동일하게 고정해서 다시 격리 비교한다.

`train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py`(canonical TRAIN_CSV/
EVAL_CSV 오버라이드 + cmamba/risk placeholder 재생성)를 모듈로 그대로 import해서 재사용하고,
그 위에 `omega._numeric_feature_cols`만 추가로 monkey-patch -- 자동유도된 전체 피쳐 목록을
원본 bundle의 102개 컬럼으로 필터링한다(reduced80의 --base-cols-allowlist-file과 동일한
발상, 다만 main() 내부를 복제하지 않고 그 값의 소스 함수 자체를 감싸는 방식 -- 이후 main()의
`base_cols = list(frames["feature_cols"])`(line 1124)로 자연스럽게 전파됨. 공유 모듈
`omega`/`train_eval_omega4_3head_parent72_loose_entry_quality_20260620`의 소스 자체는
안 건드림, 이번 세션 내내 쓴 로컬오버라이드 패턴 그대로."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818 as canon  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

parent_script = canon.parent_script
omega = canon.omega

_ORIGINAL_BUNDLE = torch.load(sweep.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)
_ORIGINAL_102_COLS = list(_ORIGINAL_BUNDLE["base_cols"])
# zig075's original bundle uses the identical 102-column set (verified 2026-08-18, same order too)
_zig_bundle = torch.load(sweep.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
if list(_zig_bundle["base_cols"]) != _ORIGINAL_102_COLS:
    raise RuntimeError("h48qual/zig075 original base_cols differ -- pinning to a single shared list is invalid, split this script per-component")

_real_numeric_feature_cols = omega._numeric_feature_cols


def _pinned_numeric_feature_cols(train, eval_df):
    full = _real_numeric_feature_cols(train, eval_df)
    full_set = set(full)
    missing = sorted(set(_ORIGINAL_102_COLS) - full_set)
    if missing:
        raise RuntimeError(f"pinned allowlist references columns not present in canonical-data auto-derived feature set: {missing}")
    pinned = [c for c in full if c in set(_ORIGINAL_102_COLS)]
    if len(pinned) != len(_ORIGINAL_102_COLS):
        raise RuntimeError(f"pinned count mismatch: got {len(pinned)}, expected {len(_ORIGINAL_102_COLS)}")
    return pinned


omega._numeric_feature_cols = _pinned_numeric_feature_cols

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
