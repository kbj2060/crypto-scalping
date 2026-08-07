"""Same risk-sidecar training/contract as train_eval_omega4_2_risk_sidecar_btc_20260708.py's live
h48qual_q055 run, pointed at the adaptive_squeeze parent bundle/predictions and feature files
instead of the originals. Mirrors the SOL analogue's use of module-level monkeypatching plus
explicit CLI defaults matching the live contract exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar_script  # noqa: E402

sidecar_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2025.csv"
sidecar_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"

if __name__ == "__main__":
    # Match the live h48qual_q055 sidecar's exact contract (its own report.json "risk_model"/
    # "contract" sections), pointed at the adaptive_squeeze parent's bundle/predictions.
    _adaptive_parent_dir = (ROOT / "tmp/causal_regen_20260516/"
                             "btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_adaptive_squeeze_20260720")
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_adaptive_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_adaptive_parent_dir)]
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
        sys.argv += ["--out-suffix", "h48qual_q055_adaptive_squeeze_20260720"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(sidecar_script.main())
