"""BTC risk sidecar retrain matching the live h48qual_q055 contract exactly, pointed at the
regime_docs42 parent's bundle/predictions and the maskedname docs42-as-wide24 regime overlay.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar_script  # noqa: E402

sidecar_script.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_docs42_20260720/btc_features_2025_regime3_current_hmm_docs42_maskedname.csv"
sidecar_script.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_docs42_20260720/btc_features_2026_regime3_current_hmm_docs42_maskedname.csv"

if __name__ == "__main__":
    _parent_dir = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_regime_docs42_20260721"
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_parent_dir)]
    if "--precomputed-prediction-tag" not in sys.argv:
        sys.argv += ["--precomputed-prediction-tag", "q055"]
    if "--quality-threshold" not in sys.argv:
        sys.argv += ["--quality-threshold", "0.55"]
    if "--exit-threshold" not in sys.argv:
        sys.argv += ["--exit-threshold", "0.95"]
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
    if "--selection-objective" not in sys.argv:
        sys.argv += ["--selection-objective", "log_risk"]
    if "--selection-scope" not in sys.argv:
        sys.argv += ["--selection-scope", "validation_only"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_docs42_q055_20260721"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
