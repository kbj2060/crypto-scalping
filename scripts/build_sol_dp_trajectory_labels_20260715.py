"""SOL port, step: build DP (backward-induction, full-price-path oracle) trajectory labels
for SOL using the IDENTICAL solver as build_omega1_2_1_dp_trajectory_daytrade_20260620.py
(build_omega1_2_1_dp_trajectory_labels_20260620.py), applied to SOL's year_oos feature frames
(2024/2025/2026) instead of the ETH alpha6 candidate CSVs.

Design note: this is an *offline label generator*, not a live/trained RL policy. Because the
full future price path is known at label-build time, the optimal position sequence that
maximizes cumulative reward is obtainable exactly via Bellman backward induction (a known-model
MDP) -- there is no need to train a SAC/DSAC agent to approximate it. Labels are direction-only
(LONG/SHORT/CASH via dp_action / label_side), matching the existing zigzag_action label schema
so they plug into the existing supervised trainers unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_omega1_2_1_dp_trajectory_labels_20260620 as dp  # noqa: E402

MODEL_ID = "sol_omega1_2_1_dp_trajectory_daytrade_20260715"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLITS = ROOT / "data/splits/year_oos"

dp.MODEL_ID = MODEL_ID
dp.OUT_DIR = OUT_DIR


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    diags: dict[str, Any] = {}
    for year, split_name in ((2024, "train_2024"), (2025, "train_2025"), (2026, "oos_2026")):
        path = SPLITS / f"sol_features_{year}.csv"
        frame = dp._read_frame(path)
        labels, action_path, diag = dp._build_dp_labels(frame, split_name)
        entry_out = OUT_DIR / f"sol_{split_name}_dp_trajectory_labels.csv"
        path_out = OUT_DIR / f"sol_{split_name}_dp_action_path_labels.csv"
        labels.to_csv(entry_out, index=False)
        action_path.to_csv(path_out, index=False)
        artifacts[f"{split_name}_entry_labels"] = str(entry_out.relative_to(ROOT))
        artifacts[f"{split_name}_path_labels"] = str(path_out.relative_to(ROOT))
        diags[split_name] = diag
        print(f"{split_name}: {len(labels)} rows, side_counts={diag['side_counts']} -> {entry_out}", flush=True)

    report = {
        "model_id": MODEL_ID,
        "status": "labels_built",
        "label_mode": "finite_state_dp_optimal_trajectory",
        "asset": "SOL",
        "solver_source": "build_omega1_2_1_dp_trajectory_labels_20260620.py (unmodified, ETH DP oracle reused verbatim)",
        "dp_actions": ["ENTER_LONG", "ENTER_SHORT", "HOLD", "EXIT", "CASH"],
        "params": {
            "max_age": dp.MAX_AGE,
            "leverage": dp.LEVERAGE,
            "margin_fraction_for_label": dp.MARGIN_FRACTION_FOR_LABEL,
            "notional_for_label": dp.NOTIONAL,
            "fee_per_side": dp.FEE_PER_SIDE,
            "hold_penalty": dp.HOLD_PENALTY,
            "min_entry_edge": dp.MIN_ENTRY_EDGE,
            "tp_bounds": dp.TP_BOUNDS,
            "sl_bounds": dp.SL_BOUNDS,
        },
        "risk_contract": {
            "notional": "margin_fraction * leverage",
            "pnl": "price_move * notional - fees",
            "tp_sl_targets": "price_move targets, not leverage-multiplied account thresholds",
        },
        "note": "params inherited from the ETH DP labeler as a first pass; not yet re-derived for SOL's cost/vol profile.",
        "splits": diags,
        "artifacts": artifacts,
    }
    (OUT_DIR / "label_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=dp._json_default) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
