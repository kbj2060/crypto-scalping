"""Same final scale-map replay as apply_final_scale_map_btc_20260708.py, pointed at the
adaptive_squeeze parent/risk-sidecar artifacts instead of the originals. Reuses BTC v1's own
already-selected long_scale=0.5/short_scale=2.5 (this script's own CLI defaults) rather than
re-running the broad scale grid -- this is a same-asset feature-variant comparison, not a
cross-asset transfer, so the v1 scale choice is the correct fixed point to isolate the
adaptive_squeeze effect against.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_btc_20260708 as scale_map_script  # noqa: E402

scale_map_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2025.csv"
scale_map_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_btc_20260720/btc_features_2026.csv"

if __name__ == "__main__":
    _adaptive_parent_dir = (ROOT / "tmp/causal_regen_20260516/"
                             "btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_adaptive_squeeze_20260720")
    _adaptive_sidecar_dir = (ROOT / "tmp/causal_regen_20260516/"
                              "btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_adaptive_squeeze_20260720")
    if "--baseline-bundle" not in sys.argv:
        sys.argv += ["--baseline-bundle", str(_adaptive_parent_dir / "true_3head_tabm_bundle.pt")]
    if "--sidecar-pkl" not in sys.argv:
        sys.argv += ["--sidecar-pkl", str(_adaptive_sidecar_dir / "risk_sidecar.pkl")]
    if "--precomputed-prediction-dir" not in sys.argv:
        sys.argv += ["--precomputed-prediction-dir", str(_adaptive_parent_dir)]
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_adaptive_squeeze_20260720")]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(scale_map_script.main())
