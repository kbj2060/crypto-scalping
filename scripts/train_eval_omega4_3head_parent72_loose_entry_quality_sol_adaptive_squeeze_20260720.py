"""Same zig075 parent architecture/training as train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py,
pointed at the adaptive_squeeze feature files (build_sol_features_adaptive_squeeze_20260720.py)
instead of the original ones. Only the input feature values differ (long_squeeze_risk,
short_squeeze_risk, crowding_pressure use a per-symbol funding_z_score instead of ETH's fixed
0.0002 divisor) -- same labels, same architecture, same hyperparameters, same regime3 sidecar.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as parent_script  # noqa: E402

parent_script.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
parent_script.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

if __name__ == "__main__":
    # Match the production zig075 run's exact args (train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py's
    # own report.json label_contract), which are NOT this script's own CLI defaults
    # (default quality-mode is "hard_rule", production used "same_as_direction"; default
    # quality-thresholds scan stops at 0.60, production/live uses 0.70).
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "adaptive_squeeze_20260720"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75"]
    raise SystemExit(parent_script.main())
