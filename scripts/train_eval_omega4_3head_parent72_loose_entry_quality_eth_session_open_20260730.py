"""Same zig075 parent architecture/training/hyperparameters as the live ETH
recipe (train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py with
the exact args recorded in its own report.json), pointed at TRAIN_CSV/EVAL_CSV
augmented with the 4 new session-open columns (session_japan,
session_europe_open, session_us_open, session_japan_open -- added to
features/engineering.py + features/schema.py 2026-07-30, joined onto the
existing trade_candidates files by build_eth_features_session_open_20260730.py).
Same labels, same architecture, same hyperparameters. Writes to a separate
out-suffix so the live-supporting zig075 artifacts are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

SESSION_DIR = ROOT / "data/splits/session_open_dummy_eth_20260730"
parent_script.omega.TRAIN_CSV = SESSION_DIR / "candidate100_trade_candidates_2025.csv"
parent_script.omega.EVAL_CSV = SESSION_DIR / "candidate100_trade_candidates_2026.csv"

if __name__ == "__main__":
    # Match the production zig075 run's exact args, per its own report.json
    # label_contract / input_contract (found 2026-07-30).
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                     str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531")]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "session_open_20260730"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(parent_script.main())
