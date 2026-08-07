"""SOL risk sidecar retrain matching the live v2 (adaptive_squeeze) contract exactly
(risk_feature_mode=parent_outputs, side_split_model, dynamic_leverage, selection_objective=pnl,
selection_scope=validation_oos_guard), pointed at the regime_docs42 parent's bundle/predictions
and the same maskedname docs42-as-wide24 regime overlay used for that parent retrain.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_sol_20260707 as sidecar_script  # noqa: E402

sidecar_script.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2025_regime3_current_hmm_docs42_maskedname.csv"
sidecar_script.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2026_regime3_current_hmm_docs42_maskedname.csv"

if __name__ == "__main__":
    _parent_dir = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_regime_docs42_20260721"
    if "--train-csv" not in sys.argv:
        sys.argv += ["--train-csv", "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"]
    if "--eval-csv" not in sys.argv:
        sys.argv += ["--eval-csv", "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"]
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_parent_dir)]
    if "--precomputed-prediction-tag" not in sys.argv:
        sys.argv += ["--precomputed-prediction-tag", "q070"]
    if "--risk-feature-mode" not in sys.argv:
        sys.argv += ["--risk-feature-mode", "parent_outputs"]
    if "--side-split-model" not in sys.argv:
        sys.argv += ["--side-split-model"]
    if "--dynamic-leverage" not in sys.argv:
        sys.argv += ["--dynamic-leverage"]
    if "--selection-objective" not in sys.argv:
        sys.argv += ["--selection-objective", "pnl"]
    if "--selection-scope" not in sys.argv:
        sys.argv += ["--selection-scope", "validation_oos_guard"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_docs42_q070_20260721"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
