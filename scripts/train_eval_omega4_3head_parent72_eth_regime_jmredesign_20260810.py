"""ETH parent retrain with the regime3-current overlay swapped to the redesigned JM detector.

Direct fork of scripts/train_eval_omega4_3head_parent72_loose_entry_quality_eth_regime_jmlam4_
20260809.py -- the lambda=4 swap this supersedes -- with only the regime CSV paths and the
out-suffix changed, so the comparison against that run isolates the detector.

Architecture, labels and hyperparameters stay at the live h48qual contract; only the six
regime3-current input columns change. The CSVs keep the `regime3_current_sensitive_wide24_*`
column names (the "maskedname" device) because that column-name contract is hardcoded across
omega, hard and cat_dq.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"

omega = parent_script.omega
omega.REGIME3_CURRENT_2025 = SUP / f"eth_regime3_current_hmm_{TAG}_2025_maskedname.csv"
omega.REGIME3_CURRENT_2026 = SUP / f"eth_regime3_current_hmm_{TAG}_2026_maskedname.csv"

if __name__ == "__main__":
    defaults = [
        ("--out-suffix", f"regime_{TAG}"),
        ("--direction-label-dir",
         "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/"
         "label_contracts/zigzag_action_labels_20260531"),
        ("--quality-mode", "quality_label_action"),
        ("--quality-label-dir",
         "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/"
         "sltp_h48_conservative_padded_to_zigzag_timestamps"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
