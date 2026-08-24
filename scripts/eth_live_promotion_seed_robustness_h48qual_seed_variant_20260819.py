#!/usr/bin/env python3
"""라이브 h48qual 번들(FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH, seed=260620, report.json
label_contract에서 확인된 원본설정)을 원본(git HEAD, exit_head pos_tp/pos_sl 버그수정 이전 --
실제 라이브를 학습시킨 그 코드)로 다른 시드 재학습. 원본과 동일 설정, seed/out-suffix만 바꾼다:
  --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_
    parent_screen_20260629/label_contracts/zigzag_action_labels_20260531
  --quality-mode quality_label_action
  --quality-label-dir tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/
    label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps
  --exit-label-mode entry_label_terminal_giveback (exit-terminal-window/adverse-unreal/
    min-mfe-for-giveback/giveback-min은 전부 argparse 기본값 그대로 -- 라이브와 동일)
  --max-exit-samples 30000 --epochs 2
  --quality-thresholds 0.35,...,0.95 (13개, q050 포함 -- 라이브 risk sidecar가 참조하는
    precomputed_prediction_tag=q050 생성 위해 0.50 포함 필수)

⚠️ 2026-08-19 정정 (두 차례): (1) 최초 시도는 omega.TRAIN_CSV/EVAL_CSV 기본값(legacy)을
그대로 뒀는데, 원본 코드를 오늘 재실행하면 자동유도 base_cols가 179개로 커짐(ai_*/m7_*/
patchtst_* 등) -- 그래서 원본 라이브 번들의 102 base_cols로 고정(pin)하는 로직을 추가했다.
(2) 그런데 pin을 걸어도 legacy CSV 자체에는 원본 102개 중 7개(fibonacci_level/funding_roc_12/
funding_roc_48/funding_z_score/hurst_288/regime_persistence/short_squeeze_risk)가 아예 없어서
(legacy EVAL_CSV가 2026-02-28까지만 있어 overlay 커버리지 교집합에서 빠지는 것으로 추정) pin
자체가 RuntimeError로 실패했다 -- eth_live_promotion_seed_robustness_canonicaldata_20260819.py
(= train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818.py가 이미 검증한 canonical
TRAIN_CSV/EVAL_CSV 오버라이드를, 버그수정본이 아니라 이 원본 스냅샷에 적용한 판)를 대신 import해
canonical 데이터 위에서 102-pin을 건다. 최종 스택: 원본코드 + canonical데이터 + 102-pin.
사용: --seed <int> 필수."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

parent_script = canon_wrap.parent_script  # == eth_live_promotion_seed_robustness_prefix_snapshot_20260819, with omega.TRAIN_CSV/EVAL_CSV already overridden to canonical as an import side effect

_ORIGINAL_102_COLS = list(torch.load(sweep.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)["base_cols"])
_real_numeric_feature_cols = parent_script.omega._numeric_feature_cols


def _pinned_numeric_feature_cols(train, eval_df):
    full = _real_numeric_feature_cols(train, eval_df)
    full_set = set(full)
    missing = sorted(set(_ORIGINAL_102_COLS) - full_set)
    if missing:
        raise RuntimeError(f"pinned allowlist references columns not present in auto-derived feature set: {missing}")
    pinned = [c for c in full if c in set(_ORIGINAL_102_COLS)]
    if len(pinned) != len(_ORIGINAL_102_COLS):
        raise RuntimeError(f"pinned count mismatch: got {len(pinned)}, expected {len(_ORIGINAL_102_COLS)}")
    return pinned


parent_script.omega._numeric_feature_cols = _pinned_numeric_feature_cols

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
    "--quality-mode", "quality_label_action",
    "--quality-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"),
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", "2",
    "--quality-thresholds", "0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_livepromo_seedvariant_{known.seed}"
    sys.argv = [sys.argv[0], *FIXED_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
