#!/usr/bin/env python3
"""일리아스1 zig075 pinned102 parent(L2 전체: direction+quality+exit head, encoder 포함)를
다른 시드로 재학습 -- N>=5 시드 확장(always-benchmark 재현성 검증용). 원본
(.../pinned102_20260818, seed=260620)의 report.json에 기록된 실제 설정을 그대로 복제,
seed와 out-suffix만 바꾼다:
  --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_
    parent_screen_20260629/label_contracts/zigzag_action_labels_20260531
  --quality-mode same_as_direction
  --exit-label-mode entry_label_terminal_giveback
  --max-exit-samples 30000
  --epochs 2  (원본 epochs_ran=2와 정확히 일치, report.json에서 확인)
  --quality-thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 (원본 risk_sidecar_
    precomputed_prediction_tag_values와 정확히 일치)
train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818(canonical데이터+
base_cols 102개 고정)을 모듈로 재사용, 재구현 없음. 사용: --seed <int> 필수."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402

parent_script = canon.parent_script

_orig_fit_expert = parent_script._fit_expert_omega4


def _fit_expert_omega4_logged(*args, **kwargs):
    expert_idx = kwargs.get("expert_idx")
    t0 = time.time()
    print(f"  expert_idx={expert_idx} start", flush=True)
    payload = _orig_fit_expert(*args, **kwargs)
    print(f"  expert_idx={expert_idx} done epochs_ran={payload.get('epochs_ran')} "
          f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
    return payload


parent_script._fit_expert_omega4 = _fit_expert_omega4_logged

FIXED_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", "2",
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_{known.seed}"
    sys.argv = [sys.argv[0], *FIXED_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
