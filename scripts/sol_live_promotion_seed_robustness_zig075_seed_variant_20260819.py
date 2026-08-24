#!/usr/bin/env python3
"""라이브 SOL zig075 v2(adaptive_squeeze) 번들(FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH,
`sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/
true_3head_tabm_bundle.pt`, report.json에서 확인된 원본설정)을 원본(git HEAD, exit_head
pos_tp/pos_sl/risk_margin/risk_leverage 버그수정 이전 -- 실제 라이브를 학습시킨 그 코드)으로
다른 시드 재학습. 원본과 동일 설정, seed/out-suffix만 바꾼다:

  omega.TRAIN_CSV/EVAL_CSV -> data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_{2025,2026}.csv
    (docs/model_contracts/sol_adaptive_squeeze_v2_20260720.md 1/2단계, 기존 committed 래퍼
    scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py와
    동일한 오버라이드 -- 단 그 래퍼는 워킹트리의 dirty sol_20260707 모듈을 import하므로 여기서는
    직접 재구현하고 prefix_snapshot(HEAD)을 대상으로 적용한다)
  --quality-mode same_as_direction
  --quality-thresholds 0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75 (8개, q070 포함 -- 라이브 risk
    sidecar가 참조하는 precomputed_prediction_tag=q070 생성 위해 0.70 포함 필수. adaptive_squeeze
    래퍼의 __main__ 블록과 report.json 둘 다에서 확인된 정확히 이 8개 값)
  --exit-label-mode entry_label_terminal_giveback (SOL HEAD 스냅샷은 이 값이 유일한 choice이자
    기본값 -- 명시적으로 넘겨도 무해)
  --epochs 4 --max-exit-samples 12000 --max-train-rows 30000 --cost-mult 3.0 (전부 스크립트
    자체의 argparse 기본값과 동일 -- report.json summaries.*.epochs_ran=4,
    exit_label.diag.rows=12000로 실측 대조 확인됨. 명시적으로 지정해 이 스냅샷의 향후 변경과
    무관하게 재현성 고정)
  --base-feature-contract-bundle <라이브 번들 경로> (⚠️ ETH와 다른 점: ETH는 이 메커니즘이 없어
    omega._numeric_feature_cols를 직접 monkey-patch해야 했지만, SOL의 HEAD 스냅샷 스크립트는
    이미 --base-feature-contract-bundle CLI 옵션을 내장하고 있다 -- 지정된 번들의 base_cols로
    자동유도 feature_cols를 pin하고, 부족하면 RuntimeError. 원본 라이브 학습 자체는 이 플래그
    없이 돌았지만(report.json: input_contract.base_feature_contract_bundle=None), 오늘 재학습은
    피쳐드리프트 방지를 위해 라이브 번들 자신에게 pin한다 -- ETH 세션과 동일한 목적, 다른 구현.
    scripts/sol_live_promotion_seed_robustness_precheck_20260819.py로 이 pin이 성공함을 이미
    사전 확인.

⚠️ 2026-08-19 정정 (동시성 + 데이터 결측): 최초 계획은 "ETH와 달리 SOL은 omega.TRAIN_CSV/
EVAL_CSV를 기존 정적 파일로 바꿔치기만 할 뿐 새 파일을 전혀 생성하지 않아 동시쓰기 경합이 없다"
였으나, 실측 중 공유 정식 경로(data/ensemble/supervised/sol_regime3_current_hmm_sensitive_
wide24_20260707/)에 REGIME3_CURRENT_2025 파일이 아예 없고 2026 파일도 `.bak_pre_extend_20260721`
백업만 남아있는 것을 발견했다(dev/server 둘 다 동일). 이 오버레이를 별도 scratch 경로에 재생성하는
로직이 필요해졌고, 그 재생성 자체가 새 파일을 만들기 때문에 ETH와 동일한 os.replace() 원자적
쓰기가 다시 필요해졌다 -- sol_live_promotion_seed_robustness_canonicaldata_20260819.py에 위치
(TRAIN_CSV/EVAL_CSV adaptive_squeeze 오버라이드 + REGIME3_CURRENT_2025/2026 재생성을 모두 처리).
이 스크립트는 prefix_snapshot을 직접 import하지 않고 그 canonicaldata 래퍼를 통해 import한다.

사용: --seed <int> 필수."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sol_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402

parent_script = canon_wrap.parent_script  # == sol_live_promotion_seed_robustness_prefix_snapshot_20260819, with omega.TRAIN_CSV/EVAL_CSV/REGIME3_CURRENT_2025/2026 already overridden as an import side effect

LIVE_BUNDLE = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt"

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
    "--quality-mode", "same_as_direction",
    "--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--epochs", "4",
    "--max-exit-samples", "12000",
    "--max-train-rows", "30000",
    "--cost-mult", "3.0",
    "--base-feature-contract-bundle", str(LIVE_BUNDLE),
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()
    out_suffix = f"adaptive_squeeze_seedvariant_{known.seed}"
    sys.argv = [sys.argv[0], *FIXED_ARGS, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
