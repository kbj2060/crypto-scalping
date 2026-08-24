#!/usr/bin/env python3
""""정리"(리던던시 감사, eth_dc_feature_redundancy_audit_20260820.py) 결과로 나온 133개 피쳐만
쓰도록 원본 DC canonicaldata 래퍼(eth_directional_change_tabm_training_canonicaldata_20260819.py)
위에 allow-list를 얹는다. TRAIN_CSV/EVAL_CSV/오버레이는 전부 원본 그대로 재사용(변경 없음) --
바뀌는 건 omega._numeric_feature_cols()가 반환하는 피쳐 목록 하나뿐이라, 158피쳐 DC 원본
베이스라인과 "피쳐셋 하나만" 다른 단일변수 비교가 된다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
sys.path.insert(0, str(ROOT / "scripts"))

import eth_directional_change_tabm_training_canonicaldata_20260819 as base_canon  # noqa: E402

omega = base_canon.omega
parent_script = base_canon.parent_script

PRUNED_133 = json.loads((SCRATCH / "dc_pruned_features_20260820.json").read_text())
assert len(PRUNED_133) == 133

_real_numeric_feature_cols = omega._numeric_feature_cols


def _numeric_feature_cols_pruned133(train, eval_df):
    full = _real_numeric_feature_cols(train, eval_df)
    missing = set(PRUNED_133) - set(full)
    if missing:
        raise RuntimeError(f"pruned list references columns not present in auto-derived feature set: {missing}")
    out = [c for c in full if c in PRUNED_133]
    if len(out) != len(PRUNED_133):
        raise RuntimeError(f"expected {len(PRUNED_133)} pruned cols, got {len(out)}")
    return out


omega._numeric_feature_cols = _numeric_feature_cols_pruned133

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
