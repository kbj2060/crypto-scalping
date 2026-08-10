"""ETH zig075 risk sidecar retrain matching the live contract, pointed at the regime_jmlam4 zig075
parent's bundle/predictions (VAL-selected quality_threshold=0.40, the best of {0.40,0.45,0.50,0.55,0.60}
on validation_pnl -- see train_eval_omega4_3head_parent72_loose_entry_quality_eth_zig075_regime_jmlam4_20260809.py's
report) and the maskedname JM-lambda4-as-wide24 regime overlay. Mirrors the h48qual sidecar fork's
structure and the docs42 precedent's CLI contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402

sidecar_script.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
sidecar_script.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_cleansource_maskedname.csv"
sidecar_script.omega.REGIME3_RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6_gapfilled_20260809.csv"

if __name__ == "__main__":
    _parent_dir = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmlam4_20260809"
    if "--train-csv" not in sys.argv:
        sys.argv += ["--train-csv",
                      "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"]
    if "--eval-csv" not in sys.argv:
        sys.argv += ["--eval-csv",
                      "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_parent_dir)]
    if "--precomputed-prediction-tag" not in sys.argv:
        sys.argv += ["--precomputed-prediction-tag", "q040"]
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
    if "--max-validation-mdd-abs" not in sys.argv:
        sys.argv += ["--max-validation-mdd-abs", "100"]
    if "--selection-objective" not in sys.argv:
        sys.argv += ["--selection-objective", "log_risk"]
    if "--selection-scope" not in sys.argv:
        sys.argv += ["--selection-scope", "validation_only"]
    if "--log-tail-penalty" not in sys.argv:
        sys.argv += ["--log-tail-penalty", "0.5"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "zig075_regime_jmlam4_q040_20260809"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
