"""Same as train_eval_omega4_3head_catboost_parent_sol_dp_labels_20260715.py, but pointed at
the lower-frequency DP labels (convert_sol_dp_labels_lowfreq_to_zigzag_schema_20260715.py)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_catboost_parent_sol_20260713 as catboost_parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as tabm  # noqa: E402

DP_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_dp_trajectory_action_labels_lowfreq_20260715"

catboost_parent.MODEL_ID = "sol_omega4_3head_catboost_parent_dp_labels_lowfreq_20260715"
tabm.LABEL_DIR = DP_LABEL_DIR

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "dp_lowfreq_q070"]
    raise SystemExit(catboost_parent.main())
