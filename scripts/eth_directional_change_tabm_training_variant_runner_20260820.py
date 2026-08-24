#!/usr/bin/env python3
"""hp/aswa/bag 변형 CLI 러너 -- canonicaldata+단일모델(레짐피쳐) 배선은 기존
`eth_directional_change_tabm_training_unified_single_model_20260819.py`와 동일, 학습함수만
`eth_directional_change_tabm_training_variants_20260820.py::fit_expert_omega4_variant`로
교체(bull만 실학습, bear/chop는 payload 복사 -- 동일 monkeypatch 기법).

`--epochs 30`(원본 스크리닝의 2가 아님) -- ASWA는 burn-in(epoch>=2) 이후 최소 여러 에폭을
평균해야 의미가 있고, hp(warmup+낮은 lr)도 수렴할 시간이 필요해서 세 변형 다 epoch30
테스트에서 이미 검증된 예산(patience=8이 실제로 도는)을 공정하게 준다."""
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

import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
import eth_directional_change_tabm_training_variants_20260820 as variants  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = canon.parent_script


def _uniform_route_probs(frame):
    return np.ones((len(frame), len(hard.EXPERT_NAMES)), dtype=np.float64)


parent._route_probs = _uniform_route_probs

_cache: dict[str, object] = {}


def _make_fit_fn(variant: str):
    def _fit_expert_omega4_unified(*args, **kwargs):
        expert_idx = kwargs.get("expert_idx")
        model_path = kwargs.get("model_path")
        if expert_idx == 0:
            t0 = time.time()
            print(f"  expert_idx=0 (bull) -- ACTUALLY TRAINING variant={variant} start", flush=True)
            payload = variants.fit_expert_omega4_variant(*args, **kwargs, variant=variant)
            print(f"  expert_idx=0 done epochs_ran={payload.get('epochs_ran')} "
                  f"best_validation_loss={payload.get('best_validation_loss')} "
                  f"used_aswa_weights={payload.get('used_aswa_weights')} "
                  f"aswa_averaged_checkpoints={payload.get('aswa_averaged_checkpoints')} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
            _cache["payload"] = payload
            return payload
        cached = copy.deepcopy(_cache["payload"])
        cached["expert"] = hard.EXPERT_NAMES[int(expert_idx)]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cached, model_path)
        print(f"  expert_idx={expert_idx} ({hard.EXPERT_NAMES[int(expert_idx)]}) -- SKIPPED, reused bull's weights", flush=True)
        return cached
    return _fit_expert_omega4_unified


DC_DENSE_LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

BASE_ARGS = [
    "--direction-label-dir", str(DC_DENSE_LABEL_DIR),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "30",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["hp", "aswa", "bag"], required=True)
    known, _ = ap.parse_known_args()

    parent_script._fit_expert_omega4 = _make_fit_fn(known.variant)

    out_suffix = f"dc_dense_cashfill_unified_single_model_{known.variant}_seed{known.seed}_20260820"
    sys.argv = [sys.argv[0], *BASE_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} variant={known.variant} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} variant={known.variant} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
