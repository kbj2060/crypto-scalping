#!/usr/bin/env python3
"""본 학습(수 분 소요) 돌리기 전에, adaptive_squeeze 데이터에서 SOL 라이브 zig075 v2 번들의 147
base_cols가 자동유도 피쳐 집합에 전부 존재하는지만 가볍게 확인 (모델 학습 없음,
omega._load_omega_frames + omega._numeric_feature_cols만 호출).

ETH 쪽(scripts/eth_live_promotion_seed_robustness_precheck_20260819.py)과 달리 SOL의 HEAD 스냅샷
학습스크립트는 --base-feature-contract-bundle CLI 옵션을 이미 내장하고 있어(피쳐-pin을 스크립트
자체가 처리) 이 프리체크는 실제 학습 커맨드가 성공할지 사전에 값싸게(CSV 로드+컬럼 diff만) 확인하는
용도다. 여기서 missing=0이 아니면 seed_variant 학습이 RuntimeError로 실패하므로, 학습 잡을 서버에
띄우기 전에 여기서 먼저 잡아낸다.

sol_live_promotion_seed_robustness_canonicaldata_20260819를 import해서 TRAIN_CSV/EVAL_CSV
adaptive_squeeze 오버라이드 + REGIME3_CURRENT_2025/2026 재생성(공유 정식 경로에 2025년 파일이
아예 없어서 발견된 이슈, 그 모듈 docstring 참고)을 둘 다 적용한 뒤 피쳐 자동유도를 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402

omega = canon_wrap.omega
LIVE_BUNDLE = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt"

train_all, eval_df, overlay_report = omega._load_omega_frames()
full = omega._numeric_feature_cols(train_all, eval_df)
full_set = set(full)
print(f"auto-derived full feature count (adaptive_squeeze data): {len(full)}", flush=True)
print(f"overlay_report: {overlay_report}", flush=True)

live_cols = list(torch.load(LIVE_BUNDLE, map_location="cpu", weights_only=False)["base_cols"])
missing = sorted(set(live_cols) - full_set)
print(f"live_zig075_v2: original_base_cols_count={len(live_cols)} missing={len(missing)} {missing[:20]}", flush=True)

if missing:
    print("PRECHECK_FAIL: --base-feature-contract-bundle pin would raise RuntimeError", flush=True)
    raise SystemExit(1)
print("PRECHECK_DONE: pin will succeed, all live base_cols present in fresh auto-derivation", flush=True)
