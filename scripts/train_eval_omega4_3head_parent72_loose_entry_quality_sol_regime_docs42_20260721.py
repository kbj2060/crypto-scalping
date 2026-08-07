"""SOL parent retrain on top of the LIVE v2 (adaptive_squeeze) feature files, with the
regime3-current overlay swapped from wide24 (VAL balanced_accuracy 0.759, current label mode) to
docs42 (VAL balanced_accuracy 0.7845 -- see tmp/causal_regen_20260516/sol_regime3_current_hmm_tuning_20260720/current_report.json).
Same architecture/labels/hyperparameters as the live v2 parent; only the 6 regime3-current input
columns change (still 147 base_cols total, same count, just a better upstream classifier feeding
the same 6 slots). Isolates whether a materially more accurate regime classifier actually improves
the downstream trading model once fully retrained -- doesn't assume it does (BTC's adaptive_squeeze
test this session showed an isolated improvement can make the full retrained pipeline WORSE).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720 as base_script  # noqa: E402

parent_script = base_script.parent_script
omega = parent_script.omega

# The "regime3_current_sensitive_wide24_*" column-name contract turns out to be hardcoded as
# literal strings in at least 3 separate shared modules across this pipeline (omega's
# REGIME3_CURRENT_COLS, hard.ROUTE_COLS/ROUTE_EXTRA_COLS in
# train_omega1_regime3_expert_direction_head_volpca_20260602.py, and a literal string inside
# train_omega1_regime3_routed_expert_direction_quality_20260602.py's _prediction_output) --
# patching every occurrence individually risks silently missing one. Instead, point at renamed
# copies of the docs42 sidecar CSVs whose columns are relabeled to the wide24 names (semantically
# identical contract: bull/bear/chop_prob + confidence/entropy/margin from a "current regime"
# classifier -- just a more accurate one). No constant overrides needed anywhere else in the chain.
omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2025_regime3_current_hmm_docs42_maskedname.csv"
omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_docs42_20260720/sol_features_2026_regime3_current_hmm_docs42_maskedname.csv"

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "regime_docs42_20260721"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75"]
    raise SystemExit(parent_script.main())
