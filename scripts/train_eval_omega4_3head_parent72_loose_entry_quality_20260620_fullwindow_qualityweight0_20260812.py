"""h48qual quality_loss_weight=0 실험 -- 사용자 질문: quality_head가 실현수익률과 무관하다고
4갈래 시도에서 반복 확인됐는데, 공유 TabM trunk(in_proj/blocks)에 quality_head loss(가중치
0.80, 고정 상수)가 계속 gradient를 흘려보내는 게 direction_head가 쓰는 공유 표현 자체를
오염시키고 있는 건 아닌가? -- 라는 가설을 검증하는 별도 실험 조건. gamma는 0(순수 CE, 원래
방식)으로 고정해 focal loss 변수와 섞이지 않게 격리한다.

scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620_fullwindow_focalloss_20260812.py
를 그대로 import해서(2024-2025 전체구간 TRAIN_CSV/overlay override를 재사용) quality_loss_weight
만 추가로 0.0으로 낮춘다."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620_fullwindow_focalloss_20260812 as base_wrapper  # noqa: E402

parent_script = base_wrapper.parent_script
parent_script.parent.CFG = dataclasses.replace(parent_script.parent.CFG, quality_loss_weight=0.0)

if __name__ == "__main__":
    defaults = [
        ("--direction-focal-gamma", "0.0"),
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
