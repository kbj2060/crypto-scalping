#!/usr/bin/env python3
"""154피쳐 엔지니어링셋(`eth_dc_engineered_features_canonicaldata_20260820.py`)에서 사용자
지시로 regime3_current_sensitive_wide24_{bull_prob,bear_prob,confidence} 3개를 제거한 151개
확정판(일리아스 이관용). 제거 사유: 이 3개의 원본 데이터(REGIME3_CURRENT_2024)가 2024년엔
존재하지 않아 "154개 고정 + 2024-01~2026-06 데이터"라는 이관 목표와 충돌 -- 사용자가 완전제거
선택(151개로, 2024/2025/2026 전 구간에서 오버레이 없이 동일하게 정의됨).

154→151 피쳐변경이 이전에 보고한 zigzag/h48qual/cusum OOS 결과(154피쳐 기준)를 뒤집을 수
있는지 사용자가 직접 문제제기 -- 이 저장소 전체가 시드/피쳐의 작은 변화에도 OOS 부호가
잘 뒤집힌다는 걸 반복 확인해온 터라 근거있는 우려. 이 wrapper로 동일 seed=133725056
재학습해 154피쳐 결과와 직접 대조 검증한다(eth_tabm_label_logic_151feature_verification_20260821.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
sys.path.insert(0, str(ROOT / "scripts"))

import eth_directional_change_tabm_training_canonicaldata_20260819 as base_canon  # noqa: E402
import eth_dc_financial_ml_feature_construction_20260820 as finml  # noqa: E402

omega = base_canon.omega
parent_script = base_canon.parent_script

VIF_CLEAN_109 = json.loads((SCRATCH / "ilias_vif_clean_109_20260821.json").read_text())
COMBO_FEATURES = json.loads((SCRATCH / "dc_combo_feature_names_20260820.json").read_text())
FINML_NAMES = json.loads((SCRATCH / "dc_financial_ml_feature_names_20260820.json").read_text())
assert len(VIF_CLEAN_109) == 109 and len(COMBO_FEATURES) == 30 and len(FINML_NAMES) == 12

FINAL_FEATURE_LIST = sorted(VIF_CLEAN_109) + sorted(c["name"] for c in COMBO_FEATURES) + sorted(FINML_NAMES)
assert len(FINAL_FEATURE_LIST) == 151

_orig_load_omega_frames = omega._load_omega_frames


def _attach_engineered_columns(df):
    import pandas as pd
    out = df.copy()
    for c in COMBO_FEATURES:
        out[c["name"]] = pd.to_numeric(out[c["a"]], errors="coerce") * pd.to_numeric(out[c["b"]], errors="coerce")
    finml_feats = finml.build_financial_ml_features(out)
    for name, arr in finml_feats.items():
        out[name] = arr
    return out


def _load_omega_frames_engineered():
    result = _orig_load_omega_frames()
    train, eval_df, *rest = result
    return (_attach_engineered_columns(train), _attach_engineered_columns(eval_df), *rest)


omega._load_omega_frames = _load_omega_frames_engineered

_real_numeric_feature_cols = omega._numeric_feature_cols


def _numeric_feature_cols_engineered(train, eval_df):
    full = _real_numeric_feature_cols(train, eval_df)
    missing_base = set(VIF_CLEAN_109) - set(full)
    if missing_base:
        raise RuntimeError(f"VIF-clean(151판) 피쳐가 auto-derived 목록에 없음: {missing_base}")
    new_cols = {c["name"] for c in COMBO_FEATURES} | set(FINML_NAMES)
    missing_new = new_cols - set(train.columns)
    if missing_new:
        raise RuntimeError(f"신규 컬럼이 프레임에 없음(_load_omega_frames 오버라이드 실패?): {missing_new}")
    out = list(FINAL_FEATURE_LIST)
    if len(out) != 151:
        raise RuntimeError(f"최종 피쳐 개수가 151이 아님: {len(out)}")
    return out


omega._numeric_feature_cols = _numeric_feature_cols_engineered

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
