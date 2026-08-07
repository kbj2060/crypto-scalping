"""Same final scale-map search as apply_final_scale_map_sol_20260707.py, pointed at the
adaptive-squeeze parent/risk-sidecar artifacts and feature files instead of the originals.
Only the feature-file paths (module-level omega.TRAIN_CSV/EVAL_CSV) need monkeypatching; the
bundle/sidecar/precomputed-prediction-dir/out-dir are passed via this script's own CLI flags.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_sol_20260707 as scale_map_script  # noqa: E402

scale_map_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
scale_map_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

if __name__ == "__main__":
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle",
                     str(ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt")]
    if "--sidecar-pkl" not in sys.argv:
        sys.argv += ["--sidecar-pkl",
                     str(ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir",
                     str(ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720")]
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720")]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(scale_map_script.main())
