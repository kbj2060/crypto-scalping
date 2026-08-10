"""BTC parent retrain with the regime3-current overlay swapped to the redesigned JM detector.

Everything except the six regime3-current input columns is held at the LIVE swingtransition
contract, read back out of that bundle's own report.json rather than reconstructed from memory:

  direction labels  tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708
  quality mode      quality_label_action, labels from
                    tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708
  exit label        entry_label_terminal_giveback
  architecture      train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806
                    (h48qual + swing_transition_prob), unchanged

The regime source is passed through the script's own --regime3-current-* CLI arguments rather than
by monkeypatching the omega module, so the swap is recorded in the run's own report.json.

The new CSVs keep the `regime3_current_sensitive_wide24_*` column names even though the detector
uses an 8-feature mRMR panel, because the wide24 column-name contract is hardcoded across omega,
hard and cat_dq -- the same "maskedname" device every previous regime swap on this project used.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806 as parent_script  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"

if __name__ == "__main__":
    defaults = [
        ("--regime3-current-2025", str(SUP / f"btc_regime3_current_hmm_{TAG}_2025_maskedname.csv")),
        ("--regime3-current-2026", str(SUP / f"btc_regime3_current_hmm_{TAG}_2026_maskedname.csv")),
        ("--direction-label-dir", "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708"),
        ("--quality-mode", "quality_label_action"),
        ("--quality-label-dir",
         "tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708"),
        ("--exit-label-mode", "entry_label_terminal_giveback"),
        ("--out-suffix", f"h48qual_regime_{TAG}"),
        ("--device", "cpu"),
    ]
    for flag, value in defaults:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    raise SystemExit(parent_script.main())
