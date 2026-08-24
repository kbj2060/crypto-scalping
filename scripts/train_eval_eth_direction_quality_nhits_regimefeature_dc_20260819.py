#!/usr/bin/env python3
"""ETH Directional-Change(DC) dense-CASH-fill 라벨 + 레짐 피쳐(라우팅 아님) + N-HiTS 백본
사전확인.

`train_eval_eth_direction_quality_nhits_moderntcn_20260816.py`(base_nt)를 import로 재사용한다
(수정 없음 -- `train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818.py`가 이미
쓰는 "import ... as base_nt" 패턴과 동일). 이 파일이 새로 하는 건 3가지뿐이다:

1. **레짐을 라우팅이 아니라 피쳐로**: base_nt는 원래 레짐을 아예 모른다(MoE 없음, 08-18
   regime-hardsplit 라인과 달리 라우팅 코드 자체가 없어 disable할 것도 없음). `SEQ_COLS`(8개
   고정 컬럼)에 HMM 레짐 확률 6개 컬럼(regime3_current_sensitive_wide24_{bull,bear,chop}_prob/
   _confidence/_entropy/_margin, data/ensemble/supervised/regime3_current_hmm_sensitive_
   balancedish_20260530/training_features_{2025,2026_rebuilt}_regime3_current_sensitive_hmm_
   wide24.csv 유래 -- TabM의 omega._load_omega_frames()가 쓰는 것과 동일 소스/동일 컬럼명)를
   concat한다. `len(SEQ_COLS)`를 참조하는 모든 다운스트림 코드(build_backbone 등)가 이미
   `SEQ_COLS`(module-level 리스트) 자체를 통해 채널 수를 얻으므로, 이 리스트를 늘리는 것만으로
   backbone 입력 차원이 자동으로 맞춰진다 -- n_vars를 참조하는 모든 지점을 개별 패치할 필요 없음.
   레짐 CSV가 2025-01-01부터만 커버해(그 이전은 신뢰 불가로 이미 문서화됨) 2024-06~2025-01
   구간은 0.0으로 채운다 -- 이 구간 bar들은 그냥 "레짐 정보 없음"으로 학습되는 것으로,
   TRAIN_START(2024-06-01)를 건드리지 않는 대신 감안해야 할 단순화다.
2. **DC 라벨로 교체**: `DIRECTION_LABEL_DIR`/`QUALITY_LABEL_DIR` 둘 다 DC dense-cashfill
   라벨 디렉토리로 오버라이드(quality=direction과 동일 파일 -- TabM의 same_as_direction과
   같은 발상, quality head는 h48_conservative가 아니라 direction 그대로를 다시 배움).
3. **`--stage sanity`로 사전확인**: 사용자가 승인한 "N-HiTS 단일시드 사전확인부터"에 정확히
   대응하는 이 스크립트의 기존 스테이지(로컬 CPU, 4개월 소표본, 2epoch, seed=1 고정, "크래시
   없이 도나"만 확인 -- PnL 결과는 안 냄). base_nt.stage_sanity()는 --arch 값과 무관하게
   ModernTCN/N-HiTS 둘 다 돌리므로 이 실행 하나로 두 아키텍처 모두의 배선을 확인할 수 있다.
   sanity가 통과하면(코드 크래시 없음 확인) 실제 VAL/OOS PnL이 딸린 좀 더 무거운 단일시드
   확인은 후속 단계(이번 스크립트 범위 밖, 결과 보고 후 논의).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402

DC_DENSE_LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

REGIME_ROUTE_FILES = {
    2025: ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv",
    2026: ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv",
}
REGIME_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

_PLACEHOLDER_DIR = ROOT / "tmp/causal_regen_20260516/eth_directional_change_nhits_training_20260819"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. 레짐 확률을 패널에 merge한 새 CSV 생성 (base_nt.PANEL_PATH는 안 건드림, 오버라이드만) ---
_panel = pd.read_csv(base_nt.PANEL_PATH, low_memory=False)
_panel["timestamp"] = pd.to_datetime(_panel["timestamp"])

_route = pd.concat(
    [pd.read_csv(p, usecols=["timestamp", *REGIME_COLS], parse_dates=["timestamp"]) for p in REGIME_ROUTE_FILES.values()],
    ignore_index=True,
).drop_duplicates("timestamp", keep="last")

_merged = _panel.merge(_route, on="timestamp", how="left")
if len(_merged) != len(_panel):
    raise RuntimeError(f"레짐 merge 후 행수({len(_merged)})가 패널({len(_panel)})과 다름 -- timestamp 중복 의심")
_n_missing_regime = int(_merged[REGIME_COLS[0]].isna().sum())
for c in REGIME_COLS:
    _merged[c] = _merged[c].fillna(0.0)

_MERGED_PANEL_PATH = _PLACEHOLDER_DIR / "eth_features_2024_2026_analysis_with_regime3_current_wide24.csv"
_merged.to_csv(_MERGED_PANEL_PATH, index=False)
print(f"레짐 merge: 패널 {len(_panel):,}행, 레짐확률 결측(2025-01-01 이전 등, 0.0으로 채움)="
      f"{_n_missing_regime:,}행 ({_n_missing_regime/len(_panel)*100:.1f}%) -> {_MERGED_PANEL_PATH}", flush=True)

base_nt.PANEL_PATH = _MERGED_PANEL_PATH
base_nt.SEQ_COLS = [*base_nt.SEQ_COLS, *REGIME_COLS]
print(f"SEQ_COLS 확장: {len(base_nt.SEQ_COLS)}채널 (원본 8 + 레짐 6)", flush=True)

# --- 2. DC dense-cashfill 라벨로 direction/quality 둘 다 교체 (quality=direction, TabM의
# same_as_direction과 동일 발상) ---
base_nt.DIRECTION_LABEL_DIR = DC_DENSE_LABEL_DIR
base_nt.QUALITY_LABEL_DIR = DC_DENSE_LABEL_DIR

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "sanity", "--device", "cpu"]
    print("stage=start mode=sanity(both archs, DC dense-cashfill labels, regime-as-feature)", flush=True)
    raise SystemExit(base_nt.main())
