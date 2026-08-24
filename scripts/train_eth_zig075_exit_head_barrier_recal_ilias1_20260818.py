#!/usr/bin/env python3
"""zig075 exit_head 배리어 재보정(adverse_unreal/min_mfe_for_giveback/giveback_min) 재시도
-- Phase 0 감사([[eth_odyssey4_exit_head_tpsl_closed_lines_bug_relevance_audit_20260818]])가
찾은 가장 직접적인 오염 사례: 원본 시도(`train_eth_zig075_exit_head_barrier_recal_20260818.py`,
전날 밤)는 이 세 값을 "새 ATR floor 비율"에 맞게 재보정했지만, freeze한 zig075 인코더가
pos_tp/pos_sl 버그가 있던 원본(`..._20260629`) 그대로였다 -- exit_head가 무슨 adverse/
giveback 값으로 재학습되든, 그 exit_head가 참조하는 `pos_tp`/`pos_sl` 입력 피쳐 자체가
여전히 2.6%/1.4% 고정값이라 자기 진짜 배리어를 볼 수 없는 상태였음.

이 스크립트는 원본 스크립트를 모듈로 그대로 import해서 재사용(로직 재구현 없음) -- 딱 2가지만
바꾼다:
  1. `--parent-bundle`을 일리아스 1의 zig075(pinned102, canonical데이터, base_cols=102,
     pos_tp/pos_sl 버그 수정됨)로 지정.
  2. `_risk_sizing_for_component`가 내부적으로 참조하는 `sweep.COMPONENTS["zig075"]`를
     일리아스 1의 번들+새로 학습한 전용 risk sidecar(q080)로 임시 오버라이드(모듈 attribute
     수준, 공유 모듈 소스 자체는 안 건드림 -- 이번 세션 내내 쓴 패턴과 동일). 이게 없으면
     `_risk_sizing_for_component`가 여전히 원본(버그있는) 번들의 risk sidecar를 참조해서
     새 인코더와 무관한 리스크사이징이 섞여 들어감.

adverse_unreal/min_mfe_for_giveback/giveback_min 재보정값 자체는 원본 시도와 동일하게
유지(-0.020/+0.015/0.45) -- "새 ATR floor 비율에 맞춘다"는 근거는 내 pos_tp/pos_sl 수정과
무관하게 floor 값(7.5%/4.0%) 자체가 안 바뀌었으므로 여전히 유효, 두 시도를 같은 재보정값으로
직접 비교 가능하게 유지."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818 as canon  # noqa: E402  (side effect: patches shared `omega` module to canonical TRAIN_CSV/EVAL_CSV/REGIME3_*)
import train_eth_zig075_exit_head_barrier_recal_20260818 as barrier_recal  # noqa: E402

assert barrier_recal.liveatr.omega4.omega is canon.omega, "module-cache sharing assumption broken"

ILIAS1_ZIG075_BUNDLE = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt"
ILIAS1_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl"

# _risk_sizing_for_component("zig075", ...) internally reads h48cons.sweep.COMPONENTS["zig075"]
# (hardcoded, not parameterized) to find the bundle/sidecar/q_tag -- override it to Ilias 1's own,
# so training-time risk sizing comes from a sidecar that actually matches this encoder, not the
# original bundle's sidecar. `sweep` here is the SAME module object referenced throughout the
# liveatr->h48cons->sweep import chain (module-cache sharing, same pattern as `canon.omega` above).
_sweep = barrier_recal.liveatr.h48cons.sweep
_sweep.COMPONENTS["zig075"] = {
    **_sweep.COMPONENTS["zig075"],
    "bundle": ILIAS1_ZIG075_BUNDLE,
    "sidecar_pkl": ILIAS1_ZIG075_SIDECAR,
    "q_tag": "q080",
    "quality_threshold": 0.80,
}

# retrain_exit_head stage prints nothing until all 3 experts finish (no per-expert progress),
# which twice looked indistinguishable from a hang and burned two timeout budgets (per
# [[feedback_always_log_and_monitor_epoch_metrics]] -- monitor training progress live, don't just
# wait for a final report). Wrap (not reimplement) the unmodified original per-expert fit function
# with start/end timing + the val_loss/epochs_ran it already returns -- zero logic change.
_orig_fit_exit_head_only = barrier_recal.pricemove_retrain._fit_exit_head_only


def _fit_exit_head_only_logged(baseline_payload, x_exit, y_exit, exit_route_frame, *, expert_idx, seed, epochs, device, model_path, **kwargs):
    expert_name = barrier_recal.hard.EXPERT_NAMES[int(expert_idx)]
    t0 = barrier_recal.time.time()
    n_experts = len(barrier_recal.hard.EXPERT_NAMES)
    # Two prior runs were interrupted by session/harness teardown after already saving some
    # per-expert checkpoints. --seed is fixed and the dataset-build stage was independently
    # verified byte-identical across separate process launches (total_rows_built matched to the
    # row exactly at every checkpoint between two runs) -- so a checkpoint already on disk for
    # this exact model_path is reproducible, not stale, and safe to reuse instead of re-paying
    # ~8-33min of retrain compute per expert.
    if model_path.exists():
        payload = barrier_recal.torch.load(model_path, map_location=device, weights_only=False)
        print(f"  expert={expert_name} ({int(expert_idx) + 1}/{n_experts}) SKIP (checkpoint already on disk from an "
              f"earlier interrupted run, seed={seed} is deterministic) epochs_ran={payload['exit_epochs_ran']} "
              f"best_val_loss={payload['best_exit_validation_loss']:.5f}", flush=True)
        return payload
    print(f"  expert={expert_name} ({int(expert_idx) + 1}/{n_experts}) start", flush=True)
    payload = _orig_fit_exit_head_only(
        baseline_payload, x_exit, y_exit, exit_route_frame,
        expert_idx=expert_idx, seed=seed, epochs=epochs, device=device, model_path=model_path, **kwargs,
    )
    print(f"  expert={expert_name} done epochs_ran={payload['exit_epochs_ran']} "
          f"best_val_loss={payload['best_exit_validation_loss']:.5f} elapsed={barrier_recal.time.time() - t0:.1f}s", flush=True)
    return payload


barrier_recal.pricemove_retrain._fit_exit_head_only = _fit_exit_head_only_logged

if __name__ == "__main__":
    if "--parent-bundle" not in sys.argv:
        sys.argv += ["--parent-bundle", str(ILIAS1_ZIG075_BUNDLE)]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "ilias1_encoder"]
    raise SystemExit(barrier_recal.main())
