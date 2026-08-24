#!/usr/bin/env python3
"""TabM DC 언학습 부족(under-training) 배제용 -- epoch만 2 -> 30(patience=8 그대로, N-HiTS
MAX_EPOCHS_FINAL=30과 맞춤)으로 올려 재학습.

`eth_directional_change_tabm_training_unified_single_model_20260819.py`와 완전히 동일한
배선(canonicaldata + 단일모델 monkeypatch)이고 BASE_ARGS의 `--epochs` 값 하나만 다르다 --
나머지 전부 동일해야 "epoch만 바꾼 효과"를 격리해서 볼 수 있다. patience=8은
`train_eval_omega1_2_tabm_3head_20260603.py`의 CFG 기본값 그대로(안 건드림) -- 검증손실이
8에폭 연속 개선 안 되면 30 다 안 돌고 알아서 멈춘다.

대상 시드 2개만: 758616172(원본 OOS 최우수 +60.70), 573123622(원본 OOS 최악 -9.33) --
"epoch를 늘리면 조건부 방향정확도가 chance에서 벗어나는가"를 가장 잘 드러낼 두 극단."""
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

DC_DENSE_LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

BASE_ARGS = [
    "--direction-label-dir", str(DC_DENSE_LABEL_DIR),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "30",  # 2 -> 30, patience=8(CFG 기본값, 미변경)이 조기종료 담당
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"dc_dense_cashfill_unified_single_model_epoch30test_seed{known.seed}_20260820"
    sys.argv = [sys.argv[0], *BASE_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix} "
          f"mode=unified_single_model(epoch30test, no regime routing, regime kept as feature)", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
