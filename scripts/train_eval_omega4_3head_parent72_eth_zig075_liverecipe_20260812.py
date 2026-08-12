"""zig075(quality_head) 실제 라이브 레시피 재현판 -- 라이브 번들(true_3head_tabm_bundle.pt,
2026-06-29 학습, quality_threshold=0.75)이 실제로 쓰는 라벨 레시피(direction/quality 모두
zigzag_action_labels_20260531, quality_mode=same_as_direction)로 기본 3-head 학습 스크립트를
그대로 재사용(h48qual의 h48orig 재현판과 동일 패턴 -- FINAL12 피쳐 축소는 없음, 전체 피쳐 그대로).

목적: zig075의 short-only 격리 테스트(2026-08-12, N=1 라이브 가중치 단일 인스턴스에서는
short_only가 always_short을 이김)를 N>=5 진짜 무작위 시드로 통계적으로 확정하기 위한 재학습.
--seed/--out-suffix는 호출자가 CLI로 넘겨서 시드별로 구분(이 파일은 direction-label-dir/
quality-mode/quality-thresholds/exit-label-mode만 라이브 레시피로 고정)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega

# 인프라 발견(2026-08-12): data/ensemble/supervised/의 여러 정식 overlay training_features
# 파일이 dev/서버 양쪽 모두에서 사라져 있음(regime3_current/cmamba/risk 전부, .bak_pre_extend
# 접미사 백업만 남거나 그마저도 없음) -- zig075와 무관한 별도 인프라 이슈로 사용자에게 플래그됨,
# 여기서는 임시 우회만 한다.
#
# (1) regime3_current: 나중의 재튜닝 작업 tmp 디렉터리에 동일 파일명 사본이 dev/서버 둘 다에서
#     발견됨(row수/날짜범위 확인: 2025년 전체, 5분봉, 105,102행) -- 진짜 데이터라서 그대로 사용.
parent_script.omega.REGIME3_CURRENT_2025 = (
    ROOT / "tmp/causal_regen_20260516/eth_regime3_current_hmm_tuning_20260721/sensitive/"
    "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
)
parent_script.omega.REGIME3_CURRENT_2026 = (
    ROOT / "tmp/causal_regen_20260516/eth_regime3_current_hmm_tuning_20260721/sensitive/"
    "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
)

# (2) cmamba/risk: 어디에도 백업 사본이 없음. 이미 확인된 사실(memory: omega_cmamba_risk_overlay_
#     dead_code, 오늘 아키텍처 조사로 재확인: omega4_6_1_live.py는 cmamba/risk_col을 전혀
#     참조하지 않음)로 -- 라이브 모델은 이 컬럼들을 어차피 쓰지 않는다. 그래서 실제 데이터를
#     복구하는 대신, TRAIN_CSV/EVAL_CSV와 같은 타임스탬프에 0으로 채운 placeholder overlay를
#     만들어 쓴다 -- 이건 "가짜 데이터 주입"이 아니라 라이브 모델이 실제로 보는 것(빈 컬럼)을
#     그대로 재현하는 것.
_PLACEHOLDER_DIR = ROOT / "tmp/eth_zig075_liverecipe_20260812/placeholder_overlays"
_PLACEHOLDER_DIR.mkdir(parents=True, exist_ok=True)


def _zero_fill_overlay(csv_path: Path, cols: list[str], out_name: str) -> Path:
    out_path = _PLACEHOLDER_DIR / out_name
    if not out_path.exists():
        ts = pd.read_csv(csv_path, usecols=["timestamp"], parse_dates=["timestamp"])
        for c in cols:
            ts[c] = 0.0
        ts.to_csv(out_path, index=False)
    return out_path


omega.REGIME3_CMAMBA_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2025_zero.csv")
omega.REGIME3_CMAMBA_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_CMAMBA_COLS, "cmamba_2026_zero.csv")
omega.REGIME3_RISK_2025 = _zero_fill_overlay(omega.TRAIN_CSV, omega.REGIME3_RISK_COLS, "risk_2025_zero.csv")
omega.REGIME3_RISK_2026 = _zero_fill_overlay(omega.EVAL_CSV, omega.REGIME3_RISK_COLS, "risk_2026_zero.csv")

if __name__ == "__main__":
    defaults = [
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "same_as_direction"),
        ("--quality-thresholds", "0.75"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
