"""CORRECTED zig075 JM lambda=4 retrain: pins base_cols to the LIVE 102-column contract and swaps
regime3 to JM-lambda4. See train_eval_omega4_3head_parent72_pinned102_regime_jmlam4_20260809.py
(the h48qual sibling) for the bug this fixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402

pinned.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
pinned.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"

if __name__ == "__main__":
    if "--pin-component" not in sys.argv:
        sys.argv += ["--pin-component", "zig075"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "pinned102_regime_jmlam4_20260809_zig075"]
    if "--epochs" not in sys.argv:
        sys.argv += ["--epochs", "2"]
    if "--max-train-rows" not in sys.argv:
        sys.argv += ["--max-train-rows", "0"]
    if "--max-exit-samples" not in sys.argv:
        sys.argv += ["--max-exit-samples", "30000"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"]
    raise SystemExit(pinned.main())
