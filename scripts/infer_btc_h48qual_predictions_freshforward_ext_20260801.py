#!/usr/bin/env python3
"""Inference-only regeneration of BTC h48qual (q055) OOS predictions on the
fresh-forward extended window.

This does NOT retrain anything. It loads the FROZEN
true_3head_tabm_bundle.pt (same bundle apply_final_scale_map_btc_20260708.py
uses) and runs pure forward inference (direction/quality softmax heads) over
oos_raw frames built from the extended zigzag/quality label directories
(tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260801
and btc_h48_conservative_padded_freshforward_ext_20260801), which themselves
were produced by exactly reproducing the canonical zigzag label builder
(build_wave3_action_labels_20260531.py) on BTC feature csvs that already
extend further (through ~2026-07-21) than the original 2026-07-08/13 frozen
run. Output schema matches oos_predictions_q055.csv exactly so it's a drop-in
replacement for --precomputed-prediction-dir in apply_final_scale_map_btc_20260708.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708 as omega4  # noqa: E402

BASELINE_BUNDLE = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708/true_3head_tabm_bundle.pt"
NEW_ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260801"
NEW_QUALITY_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_freshforward_ext_20260801"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_freshforward_ext_20260801"
Q_THRESHOLD = 0.55
Q_TAG = "q055"


def main() -> int:
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("stage=load_frozen_bundle", flush=True)
    bundle = torch.load(BASELINE_BUNDLE, map_location=device, weights_only=False)
    models = bundle["models"]
    base_cols = list(bundle["base_cols"])
    print(f"loaded {len(models)} frozen expert payloads, {len(base_cols)} base cols", flush=True)

    print("stage=prepare_frames_extended_labels", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=NEW_ZIGZAG_DIR,
        quality_mode="quality_label_action",
        quality_label_dir=NEW_QUALITY_LABEL_DIR,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    oos_raw = frames["oos_raw"]
    print("oos_raw range:", oos_raw["timestamp"].min(), "->", oos_raw["timestamp"].max(), len(oos_raw), flush=True)

    missing = [c for c in base_cols if c not in oos_raw.columns]
    if missing:
        raise RuntimeError(f"frozen bundle base_cols missing from new oos_raw: {missing[:20]}")

    print("stage=inference_only_forward_pass", flush=True)
    x_oos = parent._base_input(oos_raw, base_cols)
    oos_preds = {expert: parent._predict_payload(models[expert], x_oos, device=device) for expert in hard.EXPERT_NAMES}
    oos_route = hard._route_id(oos_raw)
    oos_direction = parent._routed(oos_preds, oos_route, "direction", 3)
    oos_quality = parent._routed(oos_preds, oos_route, "quality", 3)

    oos_src_oof = parent._prediction_output(oos_raw, oos_direction, oos_quality, threshold=Q_THRESHOLD, prefix="omega1_regime3_expertdq_oof")
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    out_path = OUT_DIR / f"oos_predictions_{Q_TAG}.csv"
    oos_src.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(oos_src)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
