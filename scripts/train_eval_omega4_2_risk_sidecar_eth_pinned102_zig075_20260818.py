#!/usr/bin/env python3
"""zig075 pinned102(canonical데이터, base_cols=102 원본과 동일) 번들 전용 진짜 risk sidecar
학습 -- h48qual판(train_eval_omega4_2_risk_sidecar_eth_pinned102_h48qual_20260818.py)과
완전히 동일한 패턴, parent_dir/tag/threshold/out-suffix만 zig075용으로 교체.

quality_threshold=0.80(q080)은 zig075 pinned102 재학습의 quality_threshold_ranking.csv
VAL-1위 값(원본 zig075의 0.75가 아님)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_canonicaldata_posfix_20260818 as canon  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402

assert sidecar_script.omega is canon.omega, "sidecar_script.omega is not the same module object as canon.omega -- module-cache sharing assumption broken, overrides will not propagate"

_PARENT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818"

if __name__ == "__main__":
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_PARENT_DIR / "true_3head_tabm_bundle.pt")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_PARENT_DIR)]
    if "--precomputed-prediction-tag" not in sys.argv:
        sys.argv += ["--precomputed-prediction-tag", "q080"]
    if "--quality-threshold" not in sys.argv:
        sys.argv += ["--quality-threshold", "0.80"]
    if "--exit-threshold" not in sys.argv:
        sys.argv += ["--exit-threshold", "0.95"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--risk-feature-mode" not in sys.argv:
        sys.argv += ["--risk-feature-mode", "parent_outputs"]
    if "--side-split-model" not in sys.argv:
        sys.argv += ["--side-split-model"]
    if "--dynamic-leverage" not in sys.argv:
        sys.argv += ["--dynamic-leverage"]
    if "--require-dynamic-leverage-mapping" not in sys.argv:
        sys.argv += ["--require-dynamic-leverage-mapping"]
    if "--live-exposure-grid" not in sys.argv:
        sys.argv += ["--live-exposure-grid"]
    if "--min-validation-avg-notional" not in sys.argv:
        sys.argv += ["--min-validation-avg-notional", "0.0"]
    if "--max-validation-avg-notional" not in sys.argv:
        sys.argv += ["--max-validation-avg-notional", "0.0"]
    # Same relaxation as the h48qual pinned102 wrapper -- first attempt failed on the MDD floor
    # ("no eligible risk mapping after full validation replay: trades >= 20, validation_mdd >=
    # -8.0000") even with the notional band already unconstrained. See that wrapper's comment.
    if "--max-validation-mdd-abs" not in sys.argv:
        sys.argv += ["--max-validation-mdd-abs", "50.0"]
    if "--selection-objective" not in sys.argv:
        sys.argv += ["--selection-objective", "log_risk"]
    if "--selection-scope" not in sys.argv:
        sys.argv += ["--selection-scope", "validation_only"]
    if "--log-tail-penalty" not in sys.argv:
        sys.argv += ["--log-tail-penalty", "0.5"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "zig075_pinned102_q080_20260818"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
