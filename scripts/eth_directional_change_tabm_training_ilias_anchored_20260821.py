#!/usr/bin/env python3
"""ETH label-logic 5-way(zigzag/h48qual/cusum) N=1 빠른 재확인용 -- 154피쳐 엔지니어링 체인
(`eth_dc_engineered_features_canonicaldata_20260820.py`, financial-ML 12개+조합 30개+VIF-clean
112개 로직 전부 무수정 재사용)은 그대로 두고, 그 뒤에 로드되는 데이터소스 4개만 앵커드
walk-forward 버전으로 로컬 오버라이드한다.

`omega._load_omega_frames`(154피쳐 계층이 이미 감싸놓은 버전)는 `TRAIN_CSV`/`EVAL_CSV`/
`REGIME3_CURRENT_2025`/`REGIME3_CURRENT_2026`을 **호출 시점에** omega 모듈 전역에서 읽는다
(패치 정의 시점에 캡처 안 됨) -- 그래서 154피쳐 몽키패치가 이미 걸린 뒤에 이 4개 경로만
재할당해도 실제 실행에는 새 값이 그대로 반영된다. 재구현이 아니라 정확히 이 지점(경로 4개)만
갈아끼우는 것.

원본과 다른 점: TRAIN을 2025 단독 -> 2024+2025 전체(앵커드 walk-forward, 오늘 레짐분류기
재스윕과 동일 관례). ⚠️ EVAL은 원래 계획(오늘 레짐분류기가 쓴 2026-07-01~08-19 single-touch
OOS)을 그대로 못 씀 -- zigzag/h48qual의 direction 라벨소스(zigzag_action_labels_20260531)가
**2026-02-28 16:00에서 물리적으로 끊겨있어**(Open Issue (f), 별도 재빌드 필요, 이 스크립트
범위 밖) 07-01 이후 구간엔 라벨 자체가 없다 -- 실제로 돌려보고 확인됨(empty timestamp
intersection). cusum만 그 구간을 커버해서 라벨별로 다른 길이의 EVAL을 쓰면, 이 축이 예전에
발견하고 고친 바로 그 "OOS 기간 불공정" 버그(위 계약서 CRITICAL 절)를 재현하게 된다. 그래서
EVAL은 기존 N=3 예비결과와 동일한 2026-01~06(3개 라벨 전부 커버 가능한 유일한 공정한 구간)을
유지하고, TRAIN만 2024를 추가하는 것으로 절충했다. REGIME3_CURRENT 오버레이는
states=24/sticky=0.90(2026-08-21 재확정 pick) 기준 최신값으로 교체. 원본 공유 파일
(data/ensemble/supervised/.../20260530/)은 건드리지 않는다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_dc_engineered_features_canonicaldata_20260820 as feat154  # noqa: E402

omega = feat154.omega
parent_script = feat154.parent_script

# ILIAS_EVAL_VARIANT selects which EVAL window this run uses (env var, default "h1"):
#   "h1"  -- 2026-01~06, works for all 3 labels (h48qual's quality barrier caps out here)
#   "full"-- 2026-07-01~08-19, matches today's regime-classifier OOS exactly (zigzag/cusum only,
#            since both now have data extended that far -- h48qual's quality label does not)
_VARIANT = os.environ.get("ILIAS_EVAL_VARIANT", "h1")
if _VARIANT == "full":
    _RAW_TRAIN_CSV = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821/train_merged_2024_to_2026h1.csv"
    _RAW_EVAL_CSV = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821/oos_2026q3_partial.csv"
    _REGIME_TRAIN_SIDE = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_2024_2026H1_regime3_current_states24_sticky090.csv"
    _REGIME_EVAL_SIDE = ROOT / "tmp/ilias_labellogic_recheck_20260821/oos_20260701_20260819_regime3_current_states24_sticky090.csv"
elif _VARIANT == "h1":
    _RAW_TRAIN_CSV = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_2024_2025full_raw.csv"
    _RAW_EVAL_CSV = ROOT / "tmp/ilias_labellogic_recheck_20260821/eval_2026H1_raw.csv"
    _REGIME_TRAIN_SIDE = ROOT / "tmp/ilias_labellogic_recheck_20260821/train_2024_2025full_regime3_current_states24_sticky090.csv"
    _REGIME_EVAL_SIDE = ROOT / "tmp/ilias_labellogic_recheck_20260821/eval_2026H1_regime3_current_states24_sticky090.csv"
else:
    raise ValueError(f"unknown ILIAS_EVAL_VARIANT={_VARIANT!r}, expected 'h1' or 'full'")

_PLACEHOLDER_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

omega.TRAIN_CSV = _RAW_TRAIN_CSV
omega.EVAL_CSV = _RAW_EVAL_CSV
omega.REGIME3_CURRENT_2025 = _REGIME_TRAIN_SIDE
omega.REGIME3_CURRENT_2026 = _REGIME_EVAL_SIDE

print(f"[ilias_anchored] TRAIN_CSV={omega.TRAIN_CSV}", flush=True)
print(f"[ilias_anchored] EVAL_CSV={omega.EVAL_CSV}", flush=True)
print(f"[ilias_anchored] REGIME3_CURRENT_2025(train overlay)={omega.REGIME3_CURRENT_2025}", flush=True)
print(f"[ilias_anchored] REGIME3_CURRENT_2026(eval overlay)={omega.REGIME3_CURRENT_2026}", flush=True)


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    import pandas as pd
    out_path = _PLACEHOLDER_DIR / out_name
    ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
    for c in cols:
        ts[c] = 0.0
    ts.to_csv(out_path, index=False)
    return out_path


omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_train_ilias_anchored_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_eval_ilias_anchored_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_train_ilias_anchored_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_eval_ilias_anchored_zero.csv")

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
