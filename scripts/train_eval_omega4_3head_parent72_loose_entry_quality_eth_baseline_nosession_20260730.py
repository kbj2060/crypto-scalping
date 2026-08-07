"""Control run for the ETH session-open-feature experiment: identical recipe
to train_eval_omega4_3head_parent72_loose_entry_quality_eth_session_open_20260730.py
(same architecture/hyperparameters/labels/quality-threshold sweep), but
pointed at the ORIGINAL (unmodified) zig075 TRAIN_CSV/EVAL_CSV -- i.e. without
the 4 new session-open columns. Used to get an apples-to-apples baseline swept
over the same extended quality-threshold list (0.40..0.75), since the live
report.json only swept to 0.60 by default.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

SESSION_DIR = ROOT / "data/splits/session_open_dummy_eth_20260730"
parent_script.omega.TRAIN_CSV = SESSION_DIR / "baseline96_trade_candidates_2025.csv"
parent_script.omega.EVAL_CSV = SESSION_DIR / "baseline96_trade_candidates_2026.csv"

if __name__ == "__main__":
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                     str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531")]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "baseline_nosession_20260730"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    if "--device" not in sys.argv:
        sys.argv += ["--device", "cpu"]
    raise SystemExit(parent_script.main())
