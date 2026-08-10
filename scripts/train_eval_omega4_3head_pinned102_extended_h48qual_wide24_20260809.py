"""h48qual, pinned to live's 102-column contract, regime3 = DEFAULT wide24 (live-equivalent
control), EVAL_CSV extended to 2026-01-01..06-30. This is the extended-window baseline to compare
the JM sibling script against -- NOT a claim of bit-identical live weights (retrained with a fixed
seed on the same TRAIN_CSV, so should closely reproduce, but is not literally the live checkpoint)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_20260727 as pinned  # noqa: E402

pinned.omega.REGIME3_RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6_gapfilled_20260809.csv"
pinned.omega.EVAL_CSV = ROOT / "data/ensemble/supervised/eth_eval_2026_extended_20260809/eth_eval_2026_extended_raw.csv"

if __name__ == "__main__":
    if "--pin-component" not in sys.argv:
        sys.argv += ["--pin-component", "h48qual"]
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "pinned102_extended_wide24_20260809_h48qual"]
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
        sys.argv += ["--quality-mode", "quality_label_action"]
    if "--quality-label-dir" not in sys.argv:
        sys.argv += ["--quality-label-dir", "tmp/eth_h48_conservative_padded_to_zigzag_timestamps_extended_20260809"]
    if "--quality-thresholds" not in sys.argv:
        sys.argv += ["--quality-thresholds", "0.45,0.50,0.55,0.60"]
    raise SystemExit(pinned.main())
