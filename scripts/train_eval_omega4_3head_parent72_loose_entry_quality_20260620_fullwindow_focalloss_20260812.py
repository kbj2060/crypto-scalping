"""h48qual direction_head focal-loss 재학습용 wrapper -- 라이브 번들과 비교 가능하도록 진짜
2024-2025 학습 소스로 고정한다. train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py
(공유 트레이너, --direction-focal-gamma 이미 구현됨)를 그대로 감싼다
-- scripts/train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py와
동일한 monkey-patch wrapper 패턴.

왜 필요한가: 공유 모듈(train_eval_omega1_2_tabm_diffusion_risk_20260603.py)의 TRAIN_CSV/
REGIME3_*_2025 전역변수 기본값은 2025년만 커버(2026-08-07 커밋, 2026-06-30 학습 시점과 다름) --
그냥 돌리면 라이브 번들(2024-2025 전체로 학습)과 비교 불가능한 2025-only 모델이 된다. 진짜 소스는
tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/ (포렌식 재구성,
scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py에서 행수 183,936 정확
일치로 검증 완료). cmamba/risk 오버레이는 내부 NaN 때문에 ffill/bfill 처리한 사본을 재사용
(라이브 번들 base_cols에 cmamba/risk 컬럼 0개 확인됨 -- 예측에 영향 없음, edge-case 검증 통과 목적만).

사용례: --seed/--direction-focal-gamma만 바꿔가며 반복 실행 (예: --seed 481003
--direction-focal-gamma 2.0 --out-suffix ...). --out-suffix를 매번 다르게 지정할 것 --
안 그러면 서로 다른 시드/감마 실행이 같은 디렉터리를 덮어쓴다."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

SRC_DIR = ROOT / "tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629"
SCRATCH = ROOT / "tmp/causal_regen_20260516/_scratch_fullwindow_recheck_20260812"

omega.TRAIN_CSV = SRC_DIR / "trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
omega.REGIME3_CURRENT_2025 = SRC_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_wide24.csv"
omega.REGIME3_CMAMBA_2025 = SCRATCH / "cmamba_2025_filled.csv"
omega.REGIME3_RISK_2025 = SCRATCH / "risk_2025_filled.csv"
omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv.bak_pre_extend_20260704"
omega.REGIME3_CMAMBA_2026 = SCRATCH / "cmamba_2026_filled.csv"
omega.REGIME3_RISK_2026 = SCRATCH / "risk_2026_filled.csv"

for p in [omega.TRAIN_CSV, omega.REGIME3_CURRENT_2025, omega.REGIME3_CMAMBA_2025, omega.REGIME3_RISK_2025,
          omega.REGIME3_CURRENT_2026, omega.REGIME3_CMAMBA_2026, omega.REGIME3_RISK_2026]:
    if not Path(p).exists():
        raise FileNotFoundError(p)

if __name__ == "__main__":
    defaults = [
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "quality_label_action"),
        ("--quality-label-dir",
         "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/"
         "label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
