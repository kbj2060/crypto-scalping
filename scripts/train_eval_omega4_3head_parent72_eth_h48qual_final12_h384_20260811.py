"""h48qual(quality_head) 재설계판: 이 세션의 detrend/diff1 재검증 + mRMR/knockoff 교차 dedup으로
확정된 direction7+quality9(sig_whale->sig_whale_dt288 치환, whale_retail_ratio_dt288 탈락) 통합
12피쳐로 direction_head/quality_head를 함께 학습한다. quality_head 타겟은 h48_conservative 배리어
공식(tp_mult=1.2/sl_mult=0.8/min_tp=0.006/min_sl=0.004)에서 horizon만 48bar->384bar로 재검토한
버전(zigzag_action과의 방향일치 89.5%->92.1%, build_eth_h384_conservative_triple_barrier_label_
20260811.py + pad_eth_h384_conservative_labels_to_zigzag_timestamps_20260811.py로 생성).

Direct fork of scripts/train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_
20260811.py -- direction 쪽 배선(REGIME3_CURRENT 오버레이, --direction-label-dir, cmamba/risk
overlay skip, vwap_dist_24/funding_roc_48 bridge)은 그대로 재사용하고 두 가지만 얹는다:
  1. `_numeric_feature_cols`를 15개 대신 이 12개로 교체.
  2. `_load_omega_frames`가 반환한 프레임에 raw 소스 컬럼(funding_pressure/m7_vae_error/sig_whale/
     sum_toptrader_long_short_ratio)으로부터 diff1/dt288 파생 4개를 추가로 계산해서 얹는다(전부
     이 세션의 mrmr_quality384_compress.py/reverify_sig_whale.py/finalize_union_featureset.py와
     동일 규칙: diff1=.diff(1), dt288=값-rolling(288,min_periods=96).mean(), 둘 다 fillna(0.0)).
  3. --quality-mode를 same_as_direction(zig075) 대신 quality_label_action으로, --quality-label-dir을
     새 h384 패딩 디렉토리로 바꾼다. (--pin-component는 이 base 스크립트에 존재하지 않는 플래그다 --
     pinned102 계열은 2026-08-10 커밋에서 삭제된 모듈에 의존해 현재 깨져 있음, 여기선 안 씀.)

architecture note: 이 harness는 direction/quality/exit 3-head를 bull/bear/chop regime별로 3개
독립 인스턴스 학습 후 regime3-current 확률로 라우팅한다(train_omega1_regime3_expert_direction_head_
volpca_20260602.py의 EXPERT_NAMES/route 로직, Omega4도 그대로 재사용) -- "공유 인코더"는 direction_head
+quality_head가 같은 인코더를 쓴다는 뜻이지, 3-regime 인스턴스를 하나로 합친다는 뜻이 아니다. 또한
여기서 쓰는 ThreeHeadTabM은 train_eval_omega1_2_tabm_3head_20260603.py의 기존(라이브 공유) 클래스다
-- scratchpad의 tabm_corrected.py(논문 정확 구현+불확실성 가중손실)는 라이브 전체가 공유하는 파일을
직접 고치지 않고 효과를 먼저 검증한다는 그 파일 자신의 설계 방침에 따라 아직 여기 연결하지 않았다;
이번 실행은 "새 피쳐셋+새 quality horizon"만 격리해서 검증하는 베이스라인이고, tabm_corrected 교체는
이 결과를 본 뒤 별도 통제비교로 진행한다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811 as zig075_final15  # noqa: E402

parent_script = zig075_final15.parent_script
omega = zig075_final15.omega

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]

# derived name -> (raw source column, transform). raw columns are all native to TRAIN_CSV/EVAL_CSV
# (verified present); vwap_dist_24/funding_roc_48 arrive already-bridged from zig075_final15's
# _load_omega_frames_no_cmamba_risk, which this script reuses unmodified below.
DERIVED = {
    "funding_pressure_diff1": ("funding_pressure", "diff1"),
    "m7_vae_error_dt288": ("m7_vae_error", "dt288"),
    "sig_whale_dt288": ("sig_whale", "dt288"),
    "sum_toptrader_long_short_ratio_dt288": ("sum_toptrader_long_short_ratio", "dt288"),
}

_base_load_omega_frames = zig075_final15._load_omega_frames_no_cmamba_risk


def _add_derived_columns(df):
    for derived, (raw, kind) in DERIVED.items():
        src = df[raw].astype(np.float64)
        if kind == "diff1":
            df[derived] = src.diff(1).fillna(0.0)
        elif kind == "dt288":
            df[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)
        else:
            raise RuntimeError(f"unknown transform kind: {kind}")
    return df


def _load_omega_frames_final12():
    train, eval_df, audit = _base_load_omega_frames()
    train = _add_derived_columns(train)
    eval_df = _add_derived_columns(eval_df)
    return train, eval_df, audit


def _numeric_feature_cols_final12(train, eval_df):
    missing_train = [c for c in FINAL12 if c not in train.columns]
    missing_eval = [c for c in FINAL12 if c not in eval_df.columns]
    if missing_train or missing_eval:
        raise RuntimeError(f"final12 feature(s) missing -- train:{missing_train} eval:{missing_eval}")
    bad_dtype = [c for c in FINAL12 if not pd.api.types.is_numeric_dtype(train[c])
                 or not pd.api.types.is_numeric_dtype(eval_df[c])]
    if bad_dtype:
        raise RuntimeError(f"final12 feature(s) non-numeric: {bad_dtype}")
    return list(FINAL12)


omega._load_omega_frames = _load_omega_frames_final12
omega._numeric_feature_cols = _numeric_feature_cols_final12

if __name__ == "__main__":
    defaults = [
        ("--out-suffix", "h48qual_final12_h384_20260811"),
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "quality_label_action"),
        ("--quality-label-dir", "tmp/eth_h384_conservative_padded_to_zigzag_timestamps_20260811"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
