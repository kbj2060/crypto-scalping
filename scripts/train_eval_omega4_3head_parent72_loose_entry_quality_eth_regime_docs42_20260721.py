"""ETH parent retrain on top of the LIVE h48qual feature files, with the regime3-current overlay
swapped from wide24 (VAL balanced_accuracy 0.7638, sensitive label mode) to docs42 (VAL balanced_accuracy
0.7756 -- see tmp/causal_regen_20260516/eth_regime3_current_hmm_tuning_20260721/sensitive_report.json).
Same architecture/labels/hyperparameters as the live h48qual parent; only the 6 regime3-current
input columns change. Uses maskedname docs42-as-wide24 columns (same trick as the SOL/BTC version
of this test) so the hardcoded wide24 column-name contract in shared modules (omega, hard, cat_dq)
keeps working unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega
omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_docs42_20260721_2025_maskedname.csv"
omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_docs42_20260721_2026_maskedname.csv"

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_docs42_20260721"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir",
                      "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    raise SystemExit(parent_script.main())
