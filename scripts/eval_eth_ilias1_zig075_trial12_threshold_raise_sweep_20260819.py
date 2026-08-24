#!/usr/bin/env python3
"""trial12(Optuna 우승 레시피: lr=9.98e-4, wd=1.32e-4, direction_focal_gamma=7.0, patience=10,
epochs cap=40) 6시드 번들로 quality_threshold를 0.80보다 높여보면(0.85/0.90/0.95) OOS-Q1
REJECTED 판정이 바뀌는지 확인 -- 사용자 질문("threshold를 높이면 어떻게 되지?")에 대한 직접
실증. 재학습 불필요: 각 trial12 번들은 학습 시 --quality-thresholds 0.55~0.95 그리드 전체를
이미 스윕해서 train/validation/oos_predictions_qXXX.csv를 전부 갖고 있으므로, threshold만
바꿔 리플레이 재실행하면 된다.

방법론(eval_eth_ilias1_zig075_trial12_freshforward_seedsweep_20260819.py와 완전동일 패턴,
threshold만 바깥쪽 축으로 추가):
- risk sidecar: q080에서 학습된 원본 sidecar를 그대로 재사용(frozen) -- 이미 시드축에 대해
  쓰던 것과 동일한 단순화를 threshold축에도 적용. sidecar 자체의 risk-sizing 피쳐는 후보bar의
  모델출력(quality/confidence 등)에서 계산되고 후보 판정(threshold)과는 별도 단계이므로, 이미
  받아들인 시드간 재사용과 같은 성격의 caveat으로 취급 -- 정확한 내부 결합까지 검증하지는
  않았음을 명시.
- 게이트: pnl_pass만 판정(2026-08-19 사용자 지시, MDD는 이후 레이어(L4+) 담당이라 이 축
  판정에서 제외 -- eval_eth_ilias1_zig075_trial12_freshforward_seedsweep_20260819.py와 동일).
- threshold=0.80 지점은 재실행하지 않음 -- 같은 스크립트로 이미 계산됨
  (tmp/causal_regen_20260516/zig075_trial12_freshforward_seedsweep_results_20260819.json)."""
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
THRESHOLDS = [0.85, 0.90, 0.95]  # 0.80은 이미 계산됨 (trial12_freshforward_seedsweep 결과 재사용)


def _bundle_for(seed: int, threshold: float) -> dict:
    bundle_dir = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_optuna_zig075_trial12_seed{seed}"
    q_tag = f"q{round(threshold * 100):03d}"
    return {
        "zig075": {
            "bundle": bundle_dir / "true_3head_tabm_bundle.pt",
            "q_tag": q_tag, "threshold": threshold,
            "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
            "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
            "sidecar_pkl": ORIGINAL_ZIG075_SIDECAR,
            "exit_threshold": 0.95,
        },
    }


BASELINE_V1_WITH_GATE = {
    "val": {"pnl": 77.31, "mdd": -21.76, "trades": 26},
    "oos_q1": {"pnl": 67.25, "mdd": -15.48, "trades": 19},
    "oos_q2": {"pnl": -12.69, "mdd": -20.76, "trades": 10},
}
OOS_CONFIRM_WINDOWS = ("oos_q1", "oos_q2")


def _gate_pass_pnl_only(candidate_wg: dict, baseline_wg: dict) -> bool:
    return float(candidate_wg["pnl"]) >= float(baseline_wg["pnl"])


def main() -> int:
    all_results: dict[str, dict] = {}
    for threshold in THRESHOLDS:
        q_tag = f"q{round(threshold * 100):03d}"
        for seed in SEEDS:
            run_label = f"trial12_{q_tag}_seed{seed}"
            print(f"########## {run_label} ##########", flush=True)
            ev.BUNDLES = _bundle_for(seed, threshold)
            ev.OUT_DIR = ROOT / f"tmp/causal_regen_20260516/eth_ilias1_zig075_trial12_thrsweep_20260819_{run_label}"
            ev.main()
            report = json.loads((ev.OUT_DIR / "report.json").read_text(encoding="utf-8"))
            windows = {w: d["with_gate"] for w, d in report["windows"].items()}
            all_results[run_label] = windows
            for w, wg in windows.items():
                print(f"{run_label:28} {w:8} pnl={wg['pnl']:8.2f}% mdd={wg['mdd']:8.2f}% trades={wg['trades']:3d}", flush=True)

    print()
    print("=== VERDICT PER (threshold, seed), pnl-only gate, oos_q1+oos_q2 single-touch ===")
    print(f"{'run':28} {'oos_q1':6} {'oos_q2':6} {'val_pnl':>9}  final_verdict")
    verdicts: dict[str, dict] = {}
    n_confirmed = 0
    for run_label, windows in all_results.items():
        q1_pass = _gate_pass_pnl_only(windows["oos_q1"], BASELINE_V1_WITH_GATE["oos_q1"])
        q2_pass = _gate_pass_pnl_only(windows["oos_q2"], BASELINE_V1_WITH_GATE["oos_q2"])
        all_pass = q1_pass and q2_pass
        verdict = "CONFIRMED" if all_pass else "REJECTED_SIGN_MISMATCH"
        verdicts[run_label] = {"oos_q1_pass": q1_pass, "oos_q2_pass": q2_pass, "final_verdict": verdict}
        if all_pass:
            n_confirmed += 1
        print(f"{run_label:28} {'PASS' if q1_pass else 'fail':6} {'PASS' if q2_pass else 'fail':6} {windows['val']['pnl']:9.2f}  {verdict}")

    print()
    print(f"=== STUDY DONE === n_confirmed={n_confirmed}/{len(all_results)}", flush=True)

    out_path = Path("tmp/causal_regen_20260516/zig075_trial12_threshold_raise_sweep_results_20260819.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"windows": all_results, "verdicts": verdicts, "baseline_v1": BASELINE_V1_WITH_GATE}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"results={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
