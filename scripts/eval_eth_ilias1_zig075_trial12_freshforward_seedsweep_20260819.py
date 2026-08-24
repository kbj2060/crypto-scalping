#!/usr/bin/env python3
"""일리아스1 zig075 단독의 REJECTED_SIGN_MISMATCH 판정(eth_ilias1_zig075_standalone_always_
direction_benchmark_20260818.md "추가 확인" 절, epochs=2 원본 학습 기준 6/6시드 전패)이 학습
레시피 문제(2 에폭 학습이 patience=10 early-stopping 발동 기회 자체를 차단)로 인한 아티팩트였는지
검증 -- optuna_eth_ilias1_zig075_lr_wd_focalgamma_20260819.py로 찾은 우승 설정(trial=12:
lr=9.98e-4, weight_decay=1.32e-4, direction_focal_gamma=7.0, patience=10, epochs cap=40,
val_loss=1.5370, baseline 대비 -34.9%)으로 학습된 6시드 번들을 대상으로 동일한 Fresh-Forward
6창 PnL 게이트를 재적용한다.

방법론 고정(eval_eth_ilias1_dual_freshforward_seedsweep_20260818.py와 완전동일 패턴, zig075
단독으로 축소):
- quality_threshold: 원본과 동일 q080/0.80 고정 (재선택 안 함 -- threshold 재선택 자체의
  VAL과적합 위험을 이번 학습-레시피 효과와 뒤섞지 않기 위해, eth_ilias1_h48qual_quality_
  gate_selectivity_shift_20260818 교훈 그대로 적용).
- risk sidecar: 원본(seed=260620, epochs=2 학습) 전용 sidecar를 모든 trial12 시드 번들에
  그대로 재사용(frozen) -- eval_eth_ilias1_dual_freshforward_seedsweep_20260818.py가 이미
  쓴 것과 동일한 명시적 caveat. trial12 시드별 전용 sidecar 재학습은 이 실험 범위 밖.
- exit_threshold=0.95, ATR TP/SL floor 0.075/0.040 -- FULL_BUNDLES와 완전히 동일.
- 게이트 판정 로직(_gate_pass, OOS_CONFIRM_WINDOWS, Baseline v1 수치)은 dual_freshforward_
  seedsweep에서 그대로 복사 -- zig075단독에 이미 이 게이트를 적용한 전례(memory "추가 확인
  (2026-08-19)" 절)와 동일 기준, 재구현 아님."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

ev._EXPECTED_ZERO_COLS = set()

ORIGINAL_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl"

SEEDS = [260620, 121026, 337153, 390529, 640787, 794920]


def _bundle_for(seed: int) -> dict:
    bundle_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_optuna_zig075_trial12_seed{seed}"
    return {
        "zig075": {
            "bundle": bundle_dir / "true_3head_tabm_bundle.pt",
            "q_tag": "q080", "threshold": 0.80,
            "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
            "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
            "sidecar_pkl": ORIGINAL_ZIG075_SIDECAR,
            "exit_threshold": 0.95,
        },
    }


# 계약문서 docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md
# "Baseline v1" 절 표(with_gate)에서 그대로 옮김 -- 오디세이4 G0 참조값, 재계산 아님.
BASELINE_V1_WITH_GATE = {
    "val": {"pnl": 77.31, "mdd": -21.76, "trades": 26},
    "oos_q1": {"pnl": 67.25, "mdd": -15.48, "trades": 19},
    "oos_q2": {"pnl": -12.69, "mdd": -20.76, "trades": 10},
}
OOS_CONFIRM_WINDOWS = ("oos_q1", "oos_q2")


def _gate_pass(candidate_wg: dict, baseline_wg: dict, mdd_slack_pp: float = 0.0) -> dict:
    """eth_omega461_multiwindow_confirmation_gate_20260814.py::summarize_multiwindow의 판정
    기준(pnl_pass/mdd_pass 산식 동일)에서 mdd_pass는 뺐다 -- 2026-08-19 사용자 지시: 아직
    L3까지만 테스트하는 단계라 MDD/리스크관리는 이후 레이어(L4+)의 몫이므로 이 축의 판정
    기준에서 제외. mdd는 계속 계산/기록만 하고(참고용) 판정에는 반영하지 않는다."""
    pnl_pass = float(candidate_wg["pnl"]) >= float(baseline_wg["pnl"])
    mdd_pass = (float(candidate_wg["mdd"]) - float(baseline_wg["mdd"])) >= -abs(mdd_slack_pp)
    return {"pnl_pass": bool(pnl_pass), "mdd_pass": bool(mdd_pass), "pass": bool(pnl_pass)}


def main() -> int:
    all_results: dict[str, dict] = {}
    for seed in SEEDS:
        seed_label = f"trial12_seed{seed}"
        print(f"########## {seed_label} ##########", flush=True)
        ev.BUNDLES = _bundle_for(seed)
        ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_ilias1_zig075_trial12_freshforward_20260819_{seed_label}"
        ev.main()
        report = json.loads((ev.OUT_DIR / "report.json").read_text(encoding="utf-8"))
        windows = {w: d["with_gate"] for w, d in report["windows"].items()}
        all_results[seed_label] = windows
        for w, wg in windows.items():
            print(f"{seed_label:22} {w:8} pnl={wg['pnl']:8.2f}% mdd={wg['mdd']:8.2f}% trades={wg['trades']:3d}", flush=True)

    print()
    print("=== VERDICT PER SEED (vs Baseline v1, oos_q1+oos_q2 single-touch) ===")
    print(f"{'seed':22} {'oos_q1':6} {'oos_q2':6} {'val_pnl':>9}  final_verdict")
    verdicts: dict[str, dict] = {}
    for seed_label, windows in all_results.items():
        per_window_pass = {w: _gate_pass(windows[w], BASELINE_V1_WITH_GATE[w]) for w in OOS_CONFIRM_WINDOWS}
        all_pass = all(g["pass"] for g in per_window_pass.values())
        verdict = "CONFIRMED" if all_pass else "REJECTED_SIGN_MISMATCH"
        verdicts[seed_label] = {"per_window": per_window_pass, "final_verdict": verdict}
        q1 = "PASS" if per_window_pass["oos_q1"]["pass"] else "fail"
        q2 = "PASS" if per_window_pass["oos_q2"]["pass"] else "fail"
        print(f"{seed_label:22} {q1:6} {q2:6} {windows['val']['pnl']:9.2f}  {verdict}")

    n_confirmed = sum(1 for v in verdicts.values() if v["final_verdict"] == "CONFIRMED")
    print()
    print(f"=== STUDY DONE === n_confirmed={n_confirmed}/{len(SEEDS)}", flush=True)

    out_path = Path("tmp/causal_regen_20260516/zig075_trial12_freshforward_seedsweep_results_20260819.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"windows": all_results, "verdicts": verdicts, "baseline_v1": BASELINE_V1_WITH_GATE}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
