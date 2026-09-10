"""CORRECTED h48qual JM lambda=4 retrain: pins base_cols to the LIVE 102-column contract and swaps
regime3 to JM-lambda4. Sibling of train_eval_omega4_3head_parent72_pinned102_zig075_regime_jmlam4_20260809.py
-- this file didn't exist yet (only reconstructable from that script's docstring reference plus the
existing tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_h48qual_ext/
bundle's report.json, both cross-checked against train_eval_omega4_3head_parent72_pinned102_20260727.py's
own usage docstring example for h48qual). Recreated 2026-08-13 to run N=5-seed robustness retrains
against the same recipe as the existing single-seed (260620) h48qual_ext bundle.

Label recipe matches the EXISTING h48qual_ext bundle exactly (quality_label_action against
sltp_h48_conservative_padded_to_zigzag_timestamps under omega_zigzag_fix_all_solutions_20260630) --
this predates the 2026-08-11 h48qual label-recipe correction (h48orig,
tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811); intentionally NOT switched to
the corrected recipe here, so this run isolates the regime3 HMM->JM seed-robustness question only.
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
        sys.argv += ["--pin-component", "h48qual"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "pinned102_regime_jmlam4_20260809_h48qual_ext"]
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
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir",
                      "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.70,0.75,0.80,0.85,0.90"]
    raise SystemExit(pinned.main())
