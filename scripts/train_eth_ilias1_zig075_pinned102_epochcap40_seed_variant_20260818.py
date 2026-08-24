#!/usr/bin/env python3
"""사용자 진단("2 에폭 학습이 문제") 직접 검증 -- train_eth_ilias1_zig075_pinned102_seed_
variant_20260818.py와 완전히 동일한 원본 설정(FIXED_ARGS 그대로)이지만 --epochs만 2->40으로
올린다. _fit_expert_omega4는 이미 parent.CFG.patience=8 early-stopping을 내장하고 있는데
(val loss 8epoch 연속 무개선시 stop, best checkpoint 자동저장), epochs=2 cap이 patience가
발동할 기회 자체를 원천봉쇄하고 있었다는 게 가설 -- 이 스크립트는 그 가설을 최소 변경(epoch cap
하나)으로 직접 검증한다. lr/optimizer/loss/patience 등 나머지는 전부 기존 검증된 기본값
그대로(CLAUDE.md 무관 변경 금지 원칙, feedback_modern_dl_training_checklist가 이미 GCE/
AdaBelief/cosine schedule 등은 이 TabM 계열에서 전부 CLOSED negative임을 확인했으므로 이번
비교에 섞지 않음)."""
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
    "--epochs", "40",
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_epochcap40_seedvariant_{known.seed}"
    sys.argv = [sys.argv[0], *FIXED_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
