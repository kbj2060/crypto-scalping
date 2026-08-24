#!/usr/bin/env python3
"""eth_tabm_label_logic_5way_seed_variant_20260820.py의 151피쳐판 -- 154→151(regime3_current
3개 제거) 변경이 zigzag/h48qual/cusum OOS 결과를 뒤집는지 직접 검증하기 위한 재학습 러너.
그 외 전부(단일모델 monkeypatch, exit-label-mode, epochs, 라벨소스별 direction-label-dir/
quality-mode) 동일 유지 -- 피쳐셋 하나만 통제된 변수로 격리."""
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

import ilias_eth_engineered151_features_canonicaldata_20260821 as feat151  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = feat151.parent_script


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
CUSUM_DENSE_LABEL_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820"

LABEL_CONFIGS = {
    "zigzag": {"direction_label_dir": str(ZIGZAG_ACTION_DIR), "quality_mode": "same_as_direction", "quality_label_dir": None},
    "h48qual": {"direction_label_dir": str(ZIGZAG_ACTION_DIR), "quality_mode": "quality_label_action", "quality_label_dir": str(H48_CONSERVATIVE_QUALITY_DIR)},
    "cusum": {"direction_label_dir": str(CUSUM_DENSE_LABEL_DIR), "quality_mode": "same_as_direction", "quality_label_dir": None},
}

COMMON_ARGS = ["--exit-label-mode", "independent_entry_hold_offsets", "--epochs", "2", "--device", "cpu"]

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

    out_suffix = f"label151feat_{known.label}_unified_single_model_seed{known.seed}_20260821"
    sys.argv = [sys.argv[0], *args, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start label={known.label} seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done label={known.label} seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
