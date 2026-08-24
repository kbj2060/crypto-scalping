#!/usr/bin/env python3
"""ETH Directional-Change(DC) dense-CASH-fill 라벨로 TabM 3-head를 **단일 모델**(레짐당 별도
expert 없음, 레짐은 피쳐로만 사용)로 학습시키는 시드별 CLI 러너.

레짐 라우팅 제거 기법은 `train_eth_ilias1_zig075_trial12_unified_single_model_20260819.py`와
동일하다(재구현 아님) -- `parent._route_probs`를 전체 1.0(균등가중치)로 monkeypatch하고,
`_fit_expert_omega4`는 bull(expert_idx=0) 한 번만 진짜로 학습시킨 뒤 그 payload를 bear/chop
슬롯에 그대로 복사 저장한다(재학습 안 함, 3배 비용 절감이자 세 슬롯이 수학적으로 완전히
동일한 모델임을 보장). 번들 포맷은 여전히 3-expert 모양이라 서빙/eval 코드는 무수정 재사용
가능.

⚠️ 레짐 라우팅은 껐지만 레짐 **피쳐**는 그대로 살아있다 -- `omega._load_omega_frames()`가
무조건 오버레이하는 `regime3_current_sensitive_wide24_{bull,bear,chop}_prob/_confidence/
_entropy/_margin`(+ 패널 자체의 `chop_index`/`cvp_regime`/`regime_trending`/
`regime_persistence`)가 `omega._numeric_feature_cols()`에 이미 자동 포함되어(DENY_PREFIX에
안 걸림, 확인됨) 모델 입력으로 들어간다 -- 이게 "레짐당 모델 대신 단일모델+레짐피쳐"라는
설계 그 자체다. 새로 계산한 피쳐는 없다.

zig075/trial12 원본 레시피(Optuna 튜닝된 lr/weight_decay/focal_gamma, exit-label-mode=
entry_label_terminal_giveback)는 그대로 베끼지 않는다 -- DC는 그 튜닝의 대상이 아니었던
새 후보이고, entry_label_terminal_giveback은 DC의 고립된 단일-bar 이벤트에서 세그먼트
스캐너가 거의 전부 skip해 RuntimeError로 죽는다(dense-fill 후에도 LONG/SHORT는 여전히 개별
bar이지 zigzag_action류의 연속 구간이 아니므로). 대신 CLI 기본값(independent_entry_hold_
offsets)과 zig075 프로덕션의 튜닝 안 된 베이스라인 하이퍼파라미터(epochs=2 등)를 쓴다."""
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
    "--epochs", "2",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"dc_dense_cashfill_unified_single_model_seed{known.seed}_20260819"
    sys.argv = [sys.argv[0], *BASE_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix} "
          f"mode=unified_single_model(no regime routing, regime kept as feature)", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
