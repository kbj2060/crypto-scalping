"""zig075, pinned to live's 102-column contract, regime3 = JM lambda=4, EVAL_CSV extended to
2026-01-01..06-30. See the h48qual sibling script for the rationale."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402

pinned.omega.REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2025_maskedname.csv"
pinned.omega.REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_cleansource_maskedname.csv"
pinned.omega.REGIME3_RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6_gapfilled_20260809.csv"
pinned.omega.EVAL_CSV = ROOT / "data/ensemble/supervised/eth_eval_2026_extended_20260809/eth_eval_2026_extended_raw.csv"

if __name__ == "__main__":
    if "--pin-component" not in sys.argv:
        sys.argv += ["--pin-component", "zig075"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "pinned102_extended_jmlam4_20260809_zig075"]
    if "--epochs" not in sys.argv:
        sys.argv += ["--epochs", "2"]
    if "--max-train-rows" not in sys.argv:
        sys.argv += ["--max-train-rows", "0"]
    if "--max-exit-samples" not in sys.argv:
        sys.argv += ["--max-exit-samples", "30000"]
    if "--exit-label-mode" not in sys.argv:
        sys.argv += ["--exit-label-mode", "entry_label_terminal_giveback"]
    if "--direction-label-dir" not in sys.argv:
        sys.argv += ["--direction-label-dir", "tmp/zigzag_action_labels_extended_20260809"]
    if "--quality-mode" not in sys.argv:
        sys.argv += ["--quality-mode", "same_as_direction"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.60,0.65,0.70,0.75,0.80,0.85,0.90"]
    raise SystemExit(pinned.main())
