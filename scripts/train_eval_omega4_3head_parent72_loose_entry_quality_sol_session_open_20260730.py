"""Same zig075 parent architecture/training/hyperparameters as
train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py
(the live-model recipe), pointed at the session-open-dummy feature files
(build_sol_features_session_open_20260730.py) instead -- the only difference
is 5 new input columns: session_europe, session_japan, session_europe_open,
session_us_open, session_japan_open (added to features/engineering.py +
features/schema.py 2026-07-30). Same labels, same architecture, same
hyperparameters, same regime3 sidecar. Writes to a separate out-suffix so the
live-supporting adaptive_squeeze_20260720 artifacts are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as parent_script  # noqa: E402

parent_script.omega.TRAIN_CSV = ROOT / "data/splits/session_open_dummy_sol_20260730/sol_features_2025.csv"
parent_script.omega.EVAL_CSV = ROOT / "data/splits/session_open_dummy_sol_20260730/sol_features_2026.csv"

if __name__ == "__main__":
    # Match the production zig075 run's exact args (same rationale as the
    # adaptive_squeeze wrapper this is copied from).
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "session_open_20260730"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75"]
    raise SystemExit(parent_script.main())
