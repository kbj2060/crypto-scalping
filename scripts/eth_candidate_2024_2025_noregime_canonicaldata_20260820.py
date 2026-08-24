#!/usr/bin/env python3
"""사용자 요청 신규 축 -- 레짐 피쳐 전부 제거 + 학습 데이터를 2025만이 아니라 2024+2025로
확장. 기존 canonicaldata 래퍼(`eth_directional_change_tabm_training_canonicaldata_20260819.py`)
와 다른 점 3가지만 변경(나머지 배선은 그대로 재사용 -- parent_script는 여전히
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`):

1. **TRAIN_CSV 확장**: `data/splits/year_oos/training_features_{2024,2025}.csv`를 concat한
   `tmp/eth_candidate_2024_2025_noregime_20260820/training_features_2024_2025_combined.csv`
   (210,481행, 2024-01-01~2025-12-31, 컬럼 142개 완전일치 확인됨)로 교체.
2. **레짐 피쳐 21개 deny-list**: `omega._numeric_feature_cols`를 monkey-patch해 자동유도된
   전체 피쳐 목록에서 레짐 관련 21개(`chop_index`/`cvp_regime`/`regime_trending`/
   `regime_persistence` 패널네이티브 4개 + `regime3_current_sensitive_wide24_*` HMM 6개 +
   `regime3_cmamba_h6_sidecar_*`/`regime3_stability_h6_score`/`regime3_transition_h6_risk_*`/
   `regime3_churn_h6_risk_score` 11개, 이 중 cmamba/stability/transition/churn 11개는 어차피
   기존에도 0-fill이었음)를 전부 제외. 이걸 빼는 만큼 REGIME3_CURRENT/CMAMBA/RISK 오버레이
   전부 0-fill로 바꿔도 무방해짐(진짜 2025 HMM 데이터를 쓰던 기존 래퍼와 다른 점) --
   덕분에 "2024엔 REGIME3_CURRENT 실데이터가 없다"는 문제 자체가 사라지고, EVAL_CSV를
   REGIME3_CURRENT_2026 커버리지에 맞춰 사전필터링할 필요도 없어짐(raw 57,601행 그대로 사용
   가능, 기존 래퍼는 51,746행으로 축소했었음).
3. **`_read_labels` 2025년 라벨을 2024+2025 결합으로 치환**: `main()`이 `_read_labels(dir, 2025,
   ...)`를 하드코딩 호출하는데(소스 319행), 이 함수를 그대로 두면 `train_all`(2024+2025
   피쳐)과 `omega._align()`할 때 2025년치 라벨만 있어 2024 피쳐 행이 전부 조용히 드롭된다
   (inner-join). `_read_labels`를 monkey-patch해 `year==2025`일 때만 2024+2025 라벨을 concat해
   반환하도록(파일시스템에 새 라벨 CSV를 쓰지 않고 메모리에서만 결합) -- `year==2026`(eval)
   호출은 그대로 통과."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

_RAW_TRAIN_CSV = ROOT / "tmp/eth_candidate_2024_2025_noregime_20260820/training_features_2024_2025_combined.csv"
_RAW_EVAL_CSV = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"

_PLACEHOLDER_DIR = ROOT / "tmp/eth_candidate_2024_2025_noregime_20260820/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)

omega.TRAIN_CSV = _RAW_TRAIN_CSV
omega.EVAL_CSV = _RAW_EVAL_CSV


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    out_path = _PLACEHOLDER_DIR / out_name
    ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
    for c in cols:
        ts[c] = 0.0
    ts.to_csv(out_path, index=False)
    return out_path


# 레짐 피쳐 21개를 아래 REGIME_DENY_COLS로 걸러낼 것이므로 CURRENT/CMAMBA/RISK 오버레이
# 전부 0-fill로 통일(진짜 데이터 필요 없음 -- 다 걸러지는 컬럼들이라 값 자체가 학습에 안 들어감).
omega.REGIME3_CURRENT_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CURRENT_COLS, "current_train_zero.csv")
omega.REGIME3_CURRENT_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CURRENT_COLS, "current_eval_zero.csv")
omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_train_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_eval_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_train_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_eval_zero.csv")

REGIME_DENY_COLS = {
    "chop_index", "cvp_regime", "regime_trending", "regime_persistence",
    "regime3_current_sensitive_wide24_bull_prob", "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob", "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy", "regime3_current_sensitive_wide24_margin",
    "regime3_cmamba_h6_sidecar_bull_prob", "regime3_cmamba_h6_sidecar_bear_prob",
    "regime3_cmamba_h6_sidecar_chop_prob", "regime3_cmamba_h6_sidecar_class_id",
    "regime3_cmamba_h6_sidecar_confidence", "regime3_cmamba_h6_sidecar_transition_prob",
    "regime3_cmamba_h6_sidecar_stability_score", "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob", "regime3_transition_h6_risk_pred", "regime3_churn_h6_risk_score",
}
_real_numeric_feature_cols = omega._numeric_feature_cols


def _numeric_feature_cols_no_regime(train, eval_df):
    full = _real_numeric_feature_cols(train, eval_df)
    missing = REGIME_DENY_COLS - set(full)
    if missing:
        raise RuntimeError(f"deny-list references columns not present in auto-derived feature set: {missing}")
    out = [c for c in full if c not in REGIME_DENY_COLS]
    if len(full) - len(out) != len(REGIME_DENY_COLS):
        raise RuntimeError(f"expected to drop {len(REGIME_DENY_COLS)} cols, actually dropped {len(full) - len(out)}")
    return out


omega._numeric_feature_cols = _numeric_feature_cols_no_regime

_orig_read_labels = parent_script._read_labels


def _read_labels_2024_2025_combined(label_dir, year, *, require_diagnostics):
    """year==2025 호출을 2024+2025 결합으로 치환. 2024 라벨도 반드시 같은 label_dir(호출부가
    --direction-label-dir로 넘긴 그 디렉토리)에서 읽는다 -- 다른 라벨 계열(DC vs CUSUM)로
    바꿔 재사용할 때 2024만 엉뚱한 라벨 소스에서 섞여 들어가는 걸 방지."""
    if int(year) != 2025:
        return _orig_read_labels(label_dir, year, require_diagnostics=require_diagnostics)
    lbl_2024 = _orig_read_labels(label_dir, 2024, require_diagnostics=require_diagnostics)
    lbl_2025 = _orig_read_labels(label_dir, 2025, require_diagnostics=require_diagnostics)
    combined = pd.concat([lbl_2024, lbl_2025], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if combined["timestamp"].duplicated().any():
        raise RuntimeError("2024+2025 라벨 결합 후 timestamp 중복")
    return combined


parent_script._read_labels = _read_labels_2024_2025_combined

if __name__ == "__main__":
    raise SystemExit(parent_script.main())
