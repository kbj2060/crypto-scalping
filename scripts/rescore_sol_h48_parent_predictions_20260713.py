#!/usr/bin/env python3
"""Regenerate SOL h48qual parent OOS predictions on the extended feature frame."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as omega4  # noqa: E402


PARENT_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_h48qual_20260707"
TAG = "q055"
QUALITY_DIR = ROOT / "tmp/causal_regen_20260516/sol_h48_conservative_padded_to_zigzag_timestamps_20260707"


def main() -> int:
    device = parent._device("cpu")
    bundle = torch.load(PARENT_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    oos_raw = frames["oos_raw"]
    x_oos = parent._base_input(oos_raw, base_cols)
    models = bundle["models"]
    preds = {expert: parent._predict_payload(models[expert], x_oos, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(oos_raw)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    out = parent._prediction_output(oos_raw, direction, quality, threshold=0.55, prefix="omega1_regime3_expertdq")
    path = PARENT_DIR / f"oos_predictions_{TAG}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows: {oos_raw['timestamp'].min()}..{oos_raw['timestamp'].max()} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
