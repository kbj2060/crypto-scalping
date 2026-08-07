"""BTC parent retrain on top of the LIVE v1 feature files, with the regime3-current overlay
swapped from wide24 (VAL balanced_accuracy 0.840, current label mode) to docs42 (VAL balanced_accuracy
0.8455 -- see tmp/causal_regen_20260516/btc_regime3_current_hmm_tuning_20260720/current_report.json).
Same architecture/labels/hyperparameters as the live h48qual parent; only the 6 regime3-current
input columns change. Uses renamed ("maskedname") docs42 sidecar CSVs so the wide24-named
column-string contract hardcoded across multiple shared modules (omega, hard, cat_dq -- found
during the SOL version of this same retrain) keeps working unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as parent_script  # noqa: E402

omega = parent_script.omega
omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_docs42_20260720/btc_features_2025_regime3_current_hmm_docs42_maskedname.csv"
omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_docs42_20260720/btc_features_2026_regime3_current_hmm_docs42_maskedname.csv"

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_docs42_20260721"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir",
                      "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"]
    raise SystemExit(parent_script.main())
