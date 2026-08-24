#!/usr/bin/env python3
"""정리된 133피쳐(리던던시 감사 결과) DC TabM 단일모델 학습 CLI 러너. 레짐 라우팅 무력화
monkeypatch는 eth_directional_change_tabm_training_unified_single_model_20260819.py와 완전
동일(재구현 아님) -- canonicaldata 래퍼만 eth_dc_pruned133_canonicaldata_20260820.py로 교체.
epochs=2로 DC 원본(158피쳐) 베이스라인과 동일하게 맞춰 피쳐셋 하나만 다른 단일변수 비교를 유지."""
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

import eth_dc_pruned133_canonicaldata_20260820 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = canon.parent_script


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
        print("  expert_idx=0 (bull) -- ACTUALLY TRAINING (uniform regime weight) start", flush=True)
        payload = _orig_fit_expert(*args, **kwargs)
        print(f"  expert_idx=0 done epochs_ran={payload.get('epochs_ran')} "
              f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
        _cache["payload"] = payload
        return payload
    cached = copy.deepcopy(_cache["payload"])
    cached["expert"] = hard.EXPERT_NAMES[int(expert_idx)]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, model_path)
    print(f"  expert_idx={expert_idx} ({hard.EXPERT_NAMES[int(expert_idx)]}) -- SKIPPED, reused bull's weights", flush=True)
    return cached


parent_script._fit_expert_omega4 = _fit_expert_omega4_unified

DC_DENSE_LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

BASE_ARGS = [
    "--direction-label-dir", str(DC_DENSE_LABEL_DIR),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "2",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"dc_pruned133_unified_single_model_seed{known.seed}_20260820"
    sys.argv = [sys.argv[0], *BASE_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix} mode=pruned133_features", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
