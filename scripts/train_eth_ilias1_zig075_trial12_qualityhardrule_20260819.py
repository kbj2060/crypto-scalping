#!/usr/bin/env python3
"""trial12(Optuna 우승 레시피: lr=9.98e-4, wd=1.32e-4, direction_focal_gamma=7.0, patience=10,
epochs cap=40)과 완전히 동일하되, quality_mode만 same_as_direction -> hard_rule로 교체.

hard_rule은 이미 구현된 코드 경로(train_eval_omega4_3head_parent72_loose_entry_quality_
20260620.py::_quality_target_hard_rule)로, zigzag_action_labels_20260531이 이미 계산해둔
zigzag_soft_long/short(각 bar에서 진입했다면 세그먼트 끝까지 얼마나 벌었을지 기반 소프트확률
-- 세그먼트 초반=높음, 후반=낮음)과 path_edge/mae/mfe를 4개 조건(side_soft>=0.70, edge>0,
mae<=0.010, mfe/mae>=1.50)으로 AND해서 quality 타겟을 만든다. same_as_direction은 이 정보를
전혀 안 쓰고 zigzag_action 그대로 quality 타겟으로 썼었다 -- 오늘 발견한 "확신도가 스윙 후반
(나쁜 타이밍)과 약하게 상관된다"는 문제를 quality head 학습단에서 직접 겨냥.

첫 테스트는 단일시드(260620, trial12 baseline과 동일 시드)로 저렴하게 확인."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

parent_script = canon.parent_script

_orig_fit_expert = parent_script._fit_expert_omega4


def _fit_expert_omega4_logged(*args, **kwargs):
    import time
    expert_idx = kwargs.get("expert_idx")
    t0 = time.time()
    print(f"  expert_idx={expert_idx} start", flush=True)
    payload = _orig_fit_expert(*args, **kwargs)
    print(f"  expert_idx={expert_idx} done epochs_ran={payload.get('epochs_ran')} "
          f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
    return payload


parent_script._fit_expert_omega4 = _fit_expert_omega4_logged

SEED = 260620
LR = 9.98e-4
WEIGHT_DECAY = 1.32e-4
FOCAL_GAMMA = 7.0
PATIENCE = 10
EPOCHS_CAP = 40

parent.CFG = replace(parent.CFG, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE)

BASE_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
    "--quality-mode", "hard_rule",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", str(EPOCHS_CAP),
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
    "--seed", str(SEED),
    "--out-suffix", "trial12_qualityhardrule_20260819",
    "--direction-focal-gamma", str(FOCAL_GAMMA),
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *BASE_ARGS]
    print(f"stage=start seed={SEED} lr={LR} weight_decay={WEIGHT_DECAY} focal_gamma={FOCAL_GAMMA} quality_mode=hard_rule", flush=True)
    result = parent_script.main()
    print("=== STUDY DONE ===", flush=True)
    raise SystemExit(result)
