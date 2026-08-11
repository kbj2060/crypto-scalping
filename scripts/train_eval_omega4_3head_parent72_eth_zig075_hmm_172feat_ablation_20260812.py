"""Regime/feature 효과 분리를 위한 대조군 (HMM 레짐 + "넓은" 자동선택 피쳐, cmamba/risk 제외).

원래는 jmlam4_20260809(172개 피쳐, cmamba/risk 포함)의 정확한 fork로 설계했으나, 그 172개 중
cmamba/risk 오버레이가 요구하는 2025년 소스 CSV가 (jmlam4 학습 이후 어느 시점의 gitignore
정리로) 디스크에서 사라져 재현 불가능함을 실행 중 확인함([[omega_cmamba_risk_overlay_dead_code]]
와 동일한 근본 원인) -- HMM 레짐 CSV(jmlam4/final15가 안 쓰지만 이 대조군엔 필요)는 얼려둔
joblib으로 재생성 가능했지만, cmamba/risk는 원본 입력 자체가 없어 재생성 대상이 아님.

그래서 final15/jmredesign과 동일하게(`_load_omega_frames_no_cmamba_risk` 패턴 재사용)
cmamba/risk 오버레이만 skip하고, FINAL15 15개 제한은 걸지 않아 "넓은 자동선택" 성격은 유지한
버전으로 대체함 -- jmlam4의 정확히 172개는 아니지만(cmamba/risk 관련 피쳐 개수만큼 적음), 이
세션에서 재현 가능한 가장 가까운 "넓은 피쳐셋 + HMM 레짐" 대조군.

2x2 설계의 (HMM, 넓은피쳐) 셀:
  (HMM, 넓은피쳐) = 이 스크립트          (JM,  172feat) = jmlam4_20260809 (cmamba/risk 포함, 172개)
  (HMM, 15feat)  = hmm_final15_ablation (JM,  15feat)  = jmredesign_final15_20260811
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega
# HMM 재생성 CSV는 공유 위치(data/ensemble/supervised)에 안 남기고 스크래치패드에만 임시 생성 --
# 학습 직후 삭제 예정(사용자 지시, 2026-08-12).
_SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/b1629bd0-65ee-4b87-a198-cb521ad4a6ef/scratchpad")
omega.REGIME3_CURRENT_2025 = _SCRATCH / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
omega.REGIME3_CURRENT_2026 = _SCRATCH / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"


def _load_omega_frames_no_cmamba_risk():
    train = omega._read(omega.TRAIN_CSV)
    eval_df = omega._read(omega.EVAL_CSV)
    train, train_current = omega._overlay_required(
        train, omega.REGIME3_CURRENT_2025, omega.REGIME3_CURRENT_COLS, tag="train_regime3_current")
    eval_df, eval_current = omega._overlay_required(
        eval_df, omega.REGIME3_CURRENT_2026, omega.REGIME3_CURRENT_COLS, tag="eval_regime3_current")
    return train, eval_df, {
        "train_current": train_current, "eval_current": eval_current,
        "cmamba_risk_overlay": "skipped -- source 2025 CSVs missing from disk, see omega-cmamba-risk-overlay-dead-code memory",
    }


omega._load_omega_frames = _load_omega_frames_no_cmamba_risk

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "zig075_hmm_broadfeat_nocmambarisk_ablation_20260812"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    raise SystemExit(parent_script.main())
