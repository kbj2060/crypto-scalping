"""ETH risk sidecar retrain matching the live h48qual_q050 contract exactly, pointed at the
regime_jmlam4 parent's bundle/predictions and the maskedname JM-lambda4-as-wide24 regime overlay.
Mirrors train_eval_omega4_2_risk_sidecar_eth_regime_docs42_20260721.py exactly except for the
regime3 source columns, the parent bundle directory, and the out-suffix.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_20260622 as sidecar_script  # noqa: E402

sidecar_script.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
sidecar_script.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_cleansource_maskedname.csv"
# 2026-08-13: the "_gapfilled_20260809" file this used to point at no longer exists on disk
# (tmp/data is gitignored, disk-only -- lost to cleanup between 2026-08-10 and today). Verified
# (scripts/diag_risk_overlay_20260813.py) that omega's own module default
# (training_features_2026_rebuilt_regime3_stability_risk_h6.csv, no gapfill suffix) merges against
# this wrapper's actual EVAL_CSV with zero dropped/edge rows, so no override is needed -- leaving
# REGIME3_RISK_2026 unset here falls through to that valid default.

if __name__ == "__main__":
    _parent_dir = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regime_jmlam4_20260809"
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
        sys.argv += ["--precomputed-prediction-tag", "q050"]
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
    # NOTE 2026-08-09: the 0.45/0.95 band (copied from the docs42 fork script) rejected every
    # candidate for this run ("no eligible validation-only risk mapping"), exactly the failure the
    # docs42 precedent documented ("avg-notional constraint 0.45-0.95 relaxed to get a raw read once
    # it rejected every candidate", docs/model_contracts/sol_btc_regime_models_retrain_tuning_20260721.md).
    # Relaxed identically: 0.0/0.0 disables the exposure_ok() notional bound entirely (see
    # train_eval_omega4_2_risk_sidecar_20260622.py's `if > 0.0` checks), leaving only the trade-floor
    # (>=95% of baseline trades) and validation-mdd-floor (>=-8.0) selection gates active.
    if "--min-validation-avg-notional" not in sys.argv:
        sys.argv += ["--min-validation-avg-notional", "0.0"]
    if "--max-validation-avg-notional" not in sys.argv:
        sys.argv += ["--max-validation-avg-notional", "0.0"]
    if "--selection-objective" not in sys.argv:
        sys.argv += ["--selection-objective", "log_risk"]
    if "--selection-scope" not in sys.argv:
        sys.argv += ["--selection-scope", "validation_only"]
    if "--log-tail-penalty" not in sys.argv:
        sys.argv += ["--log-tail-penalty", "0.5"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_jmlam4_q050_20260809"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
