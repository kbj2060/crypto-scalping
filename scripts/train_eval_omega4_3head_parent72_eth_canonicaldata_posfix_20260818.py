#!/usr/bin/env python3
"""h48qual/zig075 posfix 재학습 -- canonical 데이터 소스 강제판.

배경: 08-18 posfix 재학습(pos_tp/pos_sl 등 버그수정)이 실제로는 `omega.TRAIN_CSV`/`EVAL_CSV`
기본값(tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_
*_alpha6_current_tail111_exact.csv)을 통해 "legacy" 피쳐 파일로 학습됐다는 게 뒤늦게 발견됨 -- 이
파일은 (a) 2026년 데이터가 02-28에서 끊기고(Fresh-Forward OOS 3/4월 전체 불가), (b) 다른 스크립트
(build_omega4_6_1_extended_parent_predictions_20260706.py)가 이미 "legacy, 일부 피쳐 drift"로
문서화한 파일. base_cols가 102(원본)->172(posfix)로 늘어난 것도 이 legacy 파일 자체가 그새
커진 것이지 "현재 표준 파이프라인 진화"가 아니었음.

이 래퍼는 `omega.TRAIN_CSV`/`omega.EVAL_CSV`를 sweep.load_frame/6-window gate가 실제로 쓰는
canonical 파일(data/splits/year_oos/training_features_2025.csv / training_features_2026_
rebuilt.csv, 2026년 07-20까지 커버)로 로컬 오버라이드한다 -- 공유 모듈(`omega`)의 기본값 자체는
안 건드림, 50개 스크립트가 그 로더에 의존하므로([[omega_cmamba_risk_overlay_dead_code]] 참고)
와이드 블라스트 반경 변경 회피. 패턴은 기존 train_eval_omega4_3head_parent72_eth_zig075_
liverecipe_20260812.py와 동일(모듈 import 후 상수 재할당, main() 호출 전).

REGIME3_CURRENT_2025/2026은 건드리지 않음 -- 이미 sweep.WIDE24_2025/2026과 동일 canonical 경로가
기본값(직접 대조 확인). REGIME3_CMAMBA/RISK 2025/2026은 재생성 필요 -- 기존 placeholder가 legacy
EVAL_CSV 타임스탬프(2026-02-28까지)로 만들어져 있어 canonical EVAL_CSV(2026-07-20까지)의
57,601행 중 40,704행(71%)을 못 덮음 -- 그대로 두면 `_overlay_required`의 tail-edge-drop 로직이
그 71%를 조용히 잘라내 결국 canonical로 바꾼 의미가 없어짐. 새 canonical 타임스탬프로 새
placeholder를 만든다(cmamba/risk 자체는 여전히 진짜 데이터가 없어 0으로 채움 -- 이미 확인된 사실:
어떤 라이브 번들도 cmamba/risk 피쳐를 실제로 쓰지 않음, 08-12 아키텍처 감사로 재확인됨).
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

_PLACEHOLDER_DIR = ROOT / "tmp/causal_regen_20260516/eth_canonicaldata_posfix_20260818/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

# REGIME3_CURRENT_2025/2026 (== sweep.WIDE24_2025/2026, confirmed by direct diff 2026-08-18) is the
# tightest-coverage overlay `_load_omega_frames`'s _overlay_required calls actually need to join
# against for the 2026 side: it spans 2026-01-01..06-30 but has a 95-bar gap on 2026-02-28
# 16:05..23:55 (already documented + handled via row-dropping, not erroring, in
# eth_omega461_multiwindow_confirmation_gate_20260814.py's _drop_route_nan -- but
# _overlay_required's own tolerance is edge-only, not gap-tolerant, so a raw canonical EVAL_CSV
# raises "non-edge missing timestamps" here). Pre-filter EVAL_CSV to that same window+gap BEFORE
# _load_omega_frames ever sees it, so the overlay join has zero non-edge misses. 2025 side has zero
# missing rows against REGIME3_CURRENT_2025 (directly checked) -- TRAIN_CSV needs no filtering.
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


omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2025_canonical_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2026_canonical_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_2025_canonical_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_2026_canonical_zero.csv")

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
