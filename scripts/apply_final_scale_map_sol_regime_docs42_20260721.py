"""Same final scale-map search as apply_final_scale_map_sol_20260707.py, pointed at the
regime_docs42 parent/risk-sidecar artifacts and the maskedname docs42-as-wide24 regime overlay.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_sol_20260707 as scale_map_script  # noqa: E402

scale_map_script.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2025_regime3_current_hmm_docs42_maskedname.csv"
scale_map_script.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2026_regime3_current_hmm_docs42_maskedname.csv"

if __name__ == "__main__":
    _parent_dir = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_regime_docs42_20260721"
    _sidecar_dir = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_regime_docs42_q070_20260721"
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--sidecar-pkl" not in sys.argv:
        sys.argv += ["--sidecar-pkl", str(_sidecar_dir / "risk_sidecar.pkl")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_parent_dir)]
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_regime_docs42_20260721")]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(scale_map_script.main())
