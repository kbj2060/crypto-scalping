"""ETH zig075 parent retrain with regime3-current swapped from wide24 (12-state HMM) to the
Statistical Jump Model (k=3, lambda=4, scripts/build_eth_regime3_jm_lam4_20260809.py). CORRECTION
2026-08-09: zig075's live bundle DOES consume regime3_current_sensitive_wide24_* (6 of its 102
base_cols, verified via torch.load(...)['base_cols']) -- an earlier claim in this session that
zig075 "doesn't use regime3 at all" was wrong, based on grepping the wrong (newer, unrelated)
zig075 script. The live zig075 bundle is built from this SAME base parent script family
(train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py), just with a different
direction-label-dir/quality-mode than h48qual -- reconstructed from
tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/report.json's
recorded label_contract (direction_label_dir=zigzag_action_labels_20260531, quality_mode=same_as_direction).
Uses the SAME base-script-default hyperparameters (epochs/max-train-rows/max-exit-samples) as the
h48qual JM fork for internal consistency, NOT the live bundle's exact e2/fulltrain/exit30k settings
-- matches this session's h48qual methodology rather than chasing full hyperparameter fidelity to
the live artifact, which the docs42 precedent this replicates also did not attempt.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

omega = parent_script.omega
omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "zig075_regime_jmlam4_20260809"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir",
                      "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    raise SystemExit(parent_script.main())
