#!/usr/bin/env python3
"""사용자 지시로 구축한 "정리+조합+문헌표준피쳐" 최종 엔지니어링 피쳐셋 -- 실제 학습에 바로
꽂을 수 있는 canonicaldata 래퍼. ⚠️ 이 파일 자체는 피쳐셋을 배선만 한다 -- 아직 어떤 학습/
신호계산 스크립트도 이 래퍼를 실행하지 않았다(사용자가 "신호는 아직 계산하지 마" 지시).

구성 = VIF-clean 112개(정리, eth_dc_feature_vif_iterative_elimination_20260820.py)
     + 신규 조합 30개(RIT식 트리구조 발견, eth_dc_combination_feature_construction_20260820.py)
     + 신규 financial-ML 12개(문헌갭분석, eth_dc_financial_ml_feature_construction_20260820.py)
     = 154개.

기존 `eth_directional_change_tabm_training_canonicaldata_20260819.py`(TRAIN_CSV/EVAL_CSV/
오버레이 전부 그대로) 위에 두 가지만 추가:
1. `omega._load_omega_frames`를 감싸 원본 프레임에 30개 조합컬럼 + 12개 financial-ML 컬럼을
   추가로 붙임.
2. `omega._numeric_feature_cols`를 112+30+12=154개만 반환하도록 교체."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "tmp/dc_engineered_feature_specs_20260820"
sys.path.insert(0, str(ROOT / "scripts"))

import eth_directional_change_tabm_training_canonicaldata_20260819 as base_canon  # noqa: E402
import eth_dc_financial_ml_feature_construction_20260820 as finml  # noqa: E402

omega = base_canon.omega
parent_script = base_canon.parent_script

VIF_CLEAN_112 = json.loads((SCRATCH / "dc_vif_clean_features_20260820.json").read_text())
COMBO_FEATURES = json.loads((SCRATCH / "dc_combo_feature_names_20260820.json").read_text())
FINML_NAMES = json.loads((SCRATCH / "dc_financial_ml_feature_names_20260820.json").read_text())
assert len(VIF_CLEAN_112) == 112 and len(COMBO_FEATURES) == 30 and len(FINML_NAMES) == 12

FINAL_FEATURE_LIST = sorted(VIF_CLEAN_112) + sorted(c["name"] for c in COMBO_FEATURES) + sorted(FINML_NAMES)
assert len(FINAL_FEATURE_LIST) == 154

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
    missing_base = set(VIF_CLEAN_112) - set(full)
    if missing_base:
        raise RuntimeError(f"VIF-clean 피쳐가 auto-derived 목록에 없음: {missing_base}")
    new_cols = {c["name"] for c in COMBO_FEATURES} | set(FINML_NAMES)
    missing_new = new_cols - set(train.columns)
    if missing_new:
        raise RuntimeError(f"신규 컬럼이 프레임에 없음(_load_omega_frames 오버라이드 실패?): {missing_new}")
    out = list(FINAL_FEATURE_LIST)
    if len(out) != 154:
        raise RuntimeError(f"최종 피쳐 개수가 154가 아님: {len(out)}")
    return out


omega._numeric_feature_cols = _numeric_feature_cols_engineered

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
