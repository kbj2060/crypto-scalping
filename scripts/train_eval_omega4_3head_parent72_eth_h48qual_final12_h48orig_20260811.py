"""h48qual(quality_head) 라이브 라벨 대조판: FINAL12 피쳐+구조는 h384판(train_eval_omega4_3head_
parent72_eth_h48qual_final12_h384_20260811.py)과 완전히 동일하게 두고, quality_head 타겟만
이 세션의 384bar 재설계 대신 라이브 실제 배포판(trading_bot_modules/omega4_6_1_live.py가 로드하는
runtime_config.py::FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH 번들, 2026-06-30 학습)이 쓰는
원본 h48_conservative(horizon=48bar, quality_threshold=0.50)로 되돌린다.

목적: "quality_head 게이트가 추세 편향을 들여온다"는 이번 세션의 발견이 우리가 만든 384bar
재설계에서만 나오는 현상인지, 라이브가 실제로 쓰는 48bar 원본 레시피에서도 똑같이 나오는지 확인.
라이브 번들 자체(102피쳐, 다른 인코더)의 가중치는 우리 12피쳐 모델에 그대로 옮길 수 없어서, 대신
"동일 피쳐/구조 + 라이브와 동일한 라벨 레시피"로 재학습해서 비교한다.

pad_eth_h48_conservative_orig_labels_to_zigzag_timestamps_20260811.py로 만든 라벨 디렉토리 사용
(barrier 자체는 build_omega1_2_triple_barrier_labels_20260619.py가 원래부터 계산해두던
h48_conservative 설정 그대로, 새로 만들지 않고 재사용)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811 as h384_version  # noqa: E402

parent_script = h384_version.parent_script
omega = h384_version.omega
FINAL12 = h384_version.FINAL12

if __name__ == "__main__":
    defaults = [
        ("--out-suffix", "h48qual_final12_h48orig_20260811"),
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "quality_label_action"),
        ("--quality-label-dir", "tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811"),
        ("--quality-thresholds", "0.50"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
