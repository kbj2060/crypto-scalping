#!/usr/bin/env python3
"""154피쳐 엔지니어링 셋(`eth_dc_engineered_features_canonicaldata_20260820.py`) 위에서 라벨
로직만 바꿔가며 TabM 단일모델(레짐피쳐+MoE라우팅無)로 재학습하는 공유 러너.

사용자 지시("tabm 단일 모델로 라벨로직만 바꿔서 5개 테스트, 시드는 3개")에 따라 zigzag(zig075
production)/h48qual/dc(directional-change)/cusum 4개를 동일 스택(154피쳐+canonical 2025/2026
TRAIN·EVAL+단일모델 monkeypatch+exit-label-mode=independent_entry_hold_offsets+epochs=2)에서
`--direction-label-dir`/`--quality-mode`/`--quality-label-dir`만 바꿔 비교한다. (5번째 라벨인
분포적회귀는 이산 classification 스키마 자체가 달라 이 러너에 꽂을 수 없다 -- 별도 regression
head 스크립트에서 다룬다.)

exit-label-mode를 4개 전부 independent_entry_hold_offsets로 통일한 이유: entry_label_terminal_
giveback(zig075/h48qual 라이브가 실제로 쓰는 모드)은 연속 same-value 세그먼트를 전제하는데 dc/
cusum의 dense-cashfill 라벨은 고립된 단일-bar 이벤트라 그 모드에서 RuntimeError로 죽는다(DC
학습 착수 시 이미 확인된 제약, 계획 문서 참고). 4개 모두 같은 모드를 쓰지 않으면 "라벨 로직"과
"exit-head 학습 방식"이라는 두 축이 동시에 바뀌어 순수 라벨 비교가 아니게 되므로, 유일하게
4개 다 통과 가능한 independent_entry_hold_offsets(CLI 기본값)로 통일했다 -- 이건 zig075/h48qual
라이브 프로모션 설정과는 다른 선택이며, 그 결과를 라이브 프로모션 근거로 쓰지 않는다(스크리닝
전용).

zigzag/h48qual의 --direction-label-dir(zigzag_action_labels_20260531)는
eth_live_promotion_seed_robustness_{zig075,h48qual}_seed_variant_20260819.py가 이미 검증한
경로를 그대로 재사용(재현 아님, 파일 자체가 동일) -- 단 그 두 스크립트는 원본 코드 스냅샷+
102-pin+3-expert MoE를 쓰는 반면 이 러너는 현재 작업트리 코드(exit_head 버그수정 반영)+154피쳐+
단일모델을 쓴다는 점이 다르다(이 서브프로젝트 자체가 "코드/데이터가 바뀐 현재 환경에서
재검증"하겠다는 방침으로 시작됐음, 메모리 참고)."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eth_dc_engineered_features_canonicaldata_20260820 as feat154  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = feat154.parent_script


def _uniform_route_probs(frame):
    return np.ones((len(frame), len(hard.EXPERT_NAMES)), dtype=np.float64)


parent._route_probs = _uniform_route_probs

_cache: dict[str, object] = {}


def _fit_expert_omega4_unified(*args, **kwargs):
    expert_idx = kwargs.get("expert_idx")
    model_path = kwargs.get("model_path")
    orig = _fit_expert_omega4_unified._orig
    if expert_idx == 0:
        t0 = time.time()
        print(f"  expert_idx=0 (bull) -- ACTUALLY TRAINING (uniform regime weight) start", flush=True)
        payload = orig(*args, **kwargs)
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


_fit_expert_omega4_unified._orig = parent_script._fit_expert_omega4
parent_script._fit_expert_omega4 = _fit_expert_omega4_unified

ZIGZAG_ACTION_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
H48_CONSERVATIVE_QUALITY_DIR = ROOT / "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"
DC_DENSE_LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
CUSUM_DENSE_LABEL_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820"

LABEL_CONFIGS = {
    "zigzag": {
        "direction_label_dir": str(ZIGZAG_ACTION_DIR),
        "quality_mode": "same_as_direction",
        "quality_label_dir": None,
    },
    "h48qual": {
        "direction_label_dir": str(ZIGZAG_ACTION_DIR),
        "quality_mode": "quality_label_action",
        "quality_label_dir": str(H48_CONSERVATIVE_QUALITY_DIR),
    },
    "dc": {
        "direction_label_dir": str(DC_DENSE_LABEL_DIR),
        "quality_mode": "same_as_direction",
        "quality_label_dir": None,
    },
    "cusum": {
        "direction_label_dir": str(CUSUM_DENSE_LABEL_DIR),
        "quality_mode": "same_as_direction",
        "quality_label_dir": None,
    },
}

COMMON_ARGS = [
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "2",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(LABEL_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()

    cfg = LABEL_CONFIGS[known.label]
    args = ["--direction-label-dir", cfg["direction_label_dir"], "--quality-mode", cfg["quality_mode"]]
    if cfg["quality_label_dir"]:
        args += ["--quality-label-dir", cfg["quality_label_dir"]]
    args += COMMON_ARGS

    out_suffix = f"label5way_{known.label}_154feat_unified_single_model_seed{known.seed}_20260820"
    sys.argv = [sys.argv[0], *args, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start label={known.label} seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done label={known.label} seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
