"""Same CatBoost 3-head parent architecture, feature contract, regime routing, and exit-label
logic as train_eval_omega4_3head_catboost_parent_sol_20260713.py, but with the direction label
source swapped from SOL's zigzag/triple-barrier labels to the DP oracle trajectory labels
(build_sol_dp_trajectory_labels_20260715.py, converted by
convert_sol_dp_labels_to_zigzag_schema_20260715.py). Everything else (quality target mode,
exit-head construction, regime experts, Fresh-Forward VAL/OOS split boundary) is unchanged, so
the resulting parent_only_metrics are a like-for-like comparison against the zigzag baseline's
report at the same "parent-only, simple screening metric" level.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_catboost_parent_sol_20260713 as catboost_parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as tabm  # noqa: E402

DP_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_dp_trajectory_action_labels_20260715"

catboost_parent.MODEL_ID = "sol_omega4_3head_catboost_parent_dp_labels_20260715"
tabm.LABEL_DIR = DP_LABEL_DIR

if __name__ == "__main__":
    if "--out-suffix" not in sys.argv:
        sys.argv += ["--out-suffix", "dp_q070"]
    raise SystemExit(catboost_parent.main())
