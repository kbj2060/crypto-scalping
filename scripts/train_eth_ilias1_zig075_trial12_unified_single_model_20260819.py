#!/usr/bin/env python3
"""trial12(Optuna 우승 레시피) 그대로, 단 bull/bear/chop 3개 expert 대신 **레짐 가중치 없이
전체 데이터로 학습한 모델 1개**를 만들어서 3개 슬롯에 전부 재사용. 사용자 지시: "레짐을 어차피
잘 못 맞추는거 같다"(방금 확인한 차트: argmax 기준 chop 1,160 -> 0.5확정 규칙 적용시 1,308로
증가, 라우터가 자주 애매함) -- 레짐분할 자체가 유효한지 직접 검증하는 최소변경 테스트.

구현: parent._route_probs를 전체 1.0(레짐 무관 균등가중치)로 몬키패치하고, _fit_expert_omega4는
bull(expert_idx=0) 한 번만 진짜로 학습시킨 뒤 그 payload를 bear/chop 슬롯에 그대로 복사
저장(재학습 안 함) -- 이러면 서빙/eval 코드(generate_predictions, greedy_replay 등)는 전혀
안 건드리고 그대로 재사용 가능(번들 포맷이 여전히 3-expert 모양이라 라우팅해도 항상 같은
가중치가 응답 = 사실상 레짐무관 단일모델). 학습 자체도 1/3 비용(expert 1개만 실제 학습).

첫 테스트는 단일시드(260620, trial12 baseline과 동일)로 저렴하게 확인."""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = canon.parent_script

_orig_route_probs = parent._route_probs


def _uniform_route_probs(frame):
    return np.ones((len(frame), len(hard.EXPERT_NAMES)), dtype=np.float64)


parent._route_probs = _uniform_route_probs

_orig_fit_expert = parent_script._fit_expert_omega4
_cache: dict[str, object] = {}


def _fit_expert_omega4_unified(*args, **kwargs):
    expert_idx = kwargs.get("expert_idx")
    model_path = kwargs.get("model_path")
    if expert_idx == 0:
        t0 = time.time()
        print(f"  expert_idx=0 (bull) -- ACTUALLY TRAINING (uniform regime weight) start", flush=True)
        payload = _orig_fit_expert(*args, **kwargs)
        print(f"  expert_idx=0 done epochs_ran={payload.get('epochs_ran')} "
              f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
        _cache["payload"] = payload
        return payload
    cached = copy.deepcopy(_cache["payload"])
    cached["expert"] = hard.EXPERT_NAMES[int(expert_idx)]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, model_path)
    print(f"  expert_idx={expert_idx} ({hard.EXPERT_NAMES[int(expert_idx)]}) -- SKIPPED, reused bull's weights "
          f"(same best_validation_loss={cached.get('best_validation_loss')})", flush=True)
    return cached


parent_script._fit_expert_omega4 = _fit_expert_omega4_unified

SEED = 260620
LR = 9.98e-4
WEIGHT_DECAY = 1.32e-4
FOCAL_GAMMA = 7.0
PATIENCE = 10
EPOCHS_CAP = 40

parent.CFG = replace(parent.CFG, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE)

BASE_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", str(EPOCHS_CAP),
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
    "--seed", str(SEED),
    "--out-suffix", "trial12_unified_single_model_20260819",
    "--direction-focal-gamma", str(FOCAL_GAMMA),
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *BASE_ARGS]
    print(f"stage=start seed={SEED} lr={LR} weight_decay={WEIGHT_DECAY} focal_gamma={FOCAL_GAMMA} "
          f"mode=unified_single_model(no regime reweighting, bull-trained weights reused for bear/chop)", flush=True)
    result = parent_script.main()
    print("=== STUDY DONE ===", flush=True)
    raise SystemExit(result)
