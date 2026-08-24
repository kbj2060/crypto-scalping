#!/usr/bin/env python3
"""일리아스1 dual(h48qual+zig075 단일슬롯 포트폴리오)의 REJECTED_SIGN_MISMATCH 판정이 Baseline
v1(오디세이4 G0) 대비 단일시드(260620) 노이즈인지 검증 -- Seed-Diversity Ensemble Promotion
Gate 적용. zig075는 이미 학습된 5개 신규시드 pinned102 번들
(train_eth_ilias1_zig075_pinned102_seed_variant_20260818.py), h48qual도 동일 5개 시드로
신규학습(train_eth_ilias1_h48qual_pinned102_seed_variant_20260818.py)한 번들을 페어링해
6시드(원본+신규5) x dual 6창 Fresh-Forward 평가를 반복한다.

방법론 고정(인코더 시드만 단독 변수로 격리, 나머지는 공식 일리아스1 FULL_BUNDLES와 완전동일
-- eval_eth_ilias1_standalone_components_freshforward_20260818.py 참고):
- quality_threshold: 원본 VAL-best 재선택값 그대로 고정(h48qual=0.40, zig075=0.80) -- 시드마다
  재선택하면 이미 이 세션에서 확인된 "threshold 재선택 자체의 VAL과적합 위험"
  ([[eth_ilias1_h48qual_quality_gate_selectivity_shift_20260818]])이 인코더 시드효과와
  뒤섞여 원인 분리가 불가능해진다.
- risk sidecar: 원본(seed=260620) 전용 sidecar를 모든 신규시드 번들에 그대로 재사용(frozen)
  -- zig075 단독 always-benchmark(N=5, [[eth_ilias1_zig075_standalone_always_direction_
  benchmark_20260818]])에서 이미 쓴 것과 동일한 단순화. sidecar 재학습(시드별 전용)은 그
  자체로 인코더 재학습만큼 무거운 별도 축이라 이 실험 범위 밖 -- 명시적 caveat으로 남긴다.
- exit_threshold=0.95, ATR TP/SL floor 0.075/0.040 -- FULL_BUNDLES와 완전히 동일.

판정 로직은 eth_omega461_multiwindow_confirmation_gate_20260814.py::summarize_multiwindow의
기준(with_gate PnL non-worse AND MDD non-worse, OOS-Q1/OOS-Q2 동시통과 필요, mdd_slack_pp=0.0
기본값)을 그대로 인라인 재현한다 -- 그 함수 자체는 baseline_results도 (no_gate,with_gate)
튜플로 요구하는데 Baseline v1의 no_gate 수치가 계약문서에 없어(with_gate만 기록됨) 억지로
튜플을 맞추는 대신, 검증된 판정 3줄(pnl_pass/mdd_pass/final_verdict)만 소스에서 그대로
복사했다(재구현 아님, 로직 identical)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

ev._EXPECTED_ZERO_COLS = set()

ORIGINAL_SIDECARS = {
    "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/risk_sidecar.pkl",
    "zig075": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
}

SEED_BUNDLE_DIRS = {
    "seed260620_original": {
        "h48qual": "omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818",
        "zig075": "omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818",
    },
}
for _seed in (121026, 337153, 390529, 640787, 794920):
    SEED_BUNDLE_DIRS[f"seed{_seed}"] = {
        "h48qual": f"omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_{_seed}",
        "zig075": f"omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818_seedvariant_{_seed}",
    }


def _bundles_for(seed_label: str) -> dict:
    dirs = SEED_BUNDLE_DIRS[seed_label]
    return {
        "h48qual": {
            "bundle": ROOT / f"tmp/causal_regen_20260516/{dirs['h48qual']}/true_3head_tabm_bundle.pt",
            "q_tag": "q040", "threshold": 0.40,
            "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
            "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
            "sidecar_pkl": ORIGINAL_SIDECARS["h48qual"],
            "exit_threshold": 0.95,
        },
        "zig075": {
            "bundle": ROOT / f"tmp/causal_regen_20260516/{dirs['zig075']}/true_3head_tabm_bundle.pt",
            "q_tag": "q080", "threshold": 0.80,
            "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
            "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
            "sidecar_pkl": ORIGINAL_SIDECARS["zig075"],
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
    기준을 그대로 복사 (pnl_pass/mdd_pass 산식 동일)."""
    pnl_pass = float(candidate_wg["pnl"]) >= float(baseline_wg["pnl"])
    mdd_pass = (float(candidate_wg["mdd"]) - float(baseline_wg["mdd"])) >= -abs(mdd_slack_pp)
    return {"pnl_pass": bool(pnl_pass), "mdd_pass": bool(mdd_pass), "pass": bool(pnl_pass and mdd_pass)}


def main() -> int:
    all_results: dict[str, dict] = {}
    for seed_label in SEED_BUNDLE_DIRS:
        print(f"########## {seed_label} ##########", flush=True)
        ev.BUNDLES = _bundles_for(seed_label)
        ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_ilias1_dual_seedsweep_20260818_{seed_label}"
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

    out_path = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/2a48cbbb-f754-4dea-b53f-3262b328fe32/scratchpad/dual_seedsweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"windows": all_results, "verdicts": verdicts, "baseline_v1": BASELINE_V1_WITH_GATE}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
