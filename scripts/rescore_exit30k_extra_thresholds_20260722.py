#!/usr/bin/env python3
"""Re-score an already-trained exit30k parent bundle at an extra quality threshold
(outside the default 0.40-0.60 sweep), without retraining. Produces
train_predictions_q{tag}.csv / validation_predictions_q{tag}.csv / oos_predictions_q{tag}.csv
inside the bundle's own out_dir, matching the format the parent script's own sweep produces,
so downstream risk-sidecar training can point --precomputed-prediction-tag at it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

parent = importlib.import_module("train_eval_omega1_2_tabm_3head_20260603")
hard = importlib.import_module("train_omega1_regime3_expert_direction_head_volpca_20260602")

ASSET_MODULE = {
    "sol": "train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707",
    "btc": "train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSET_MODULE), required=True)
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--quality-mode", choices=["same_as_direction", "quality_label_action"], required=True)
    ap.add_argument("--quality-label-dir", type=Path, default=None)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    omega4 = importlib.import_module(ASSET_MODULE[args.asset])
    device = parent._device(str(args.device))

    bundle = torch.load(args.parent_dir / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = bundle["models"]

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode=str(args.quality_mode),
        quality_label_dir=args.quality_label_dir,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )

    # train/validation keep the _oof_ prefix (parent._to_decisions(..., oof=True)); only oos gets
    # the prefix stripped (oof=False) -- matches the original training script's own convention.
    for split_name, raw, strip_oof in (
        ("train", frames["train_raw"], False),
        ("validation", frames["val_raw"], False),
        ("oos", frames["oos_raw"], True),
    ):
        x = parent._base_input(raw, base_cols)
        preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(raw)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        out_oof = parent._prediction_output(raw, direction, quality, threshold=float(args.threshold), prefix="omega1_regime3_expertdq_oof")
        if strip_oof:
            out = out_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in out_oof.columns})
            final_col = "omega1_regime3_expertdq_final_action"
        else:
            out = out_oof
            final_col = "omega1_regime3_expertdq_oof_final_action"
        out_path = args.parent_dir / f"{split_name}_predictions_{args.tag}.csv"
        out.to_csv(out_path, index=False)
        nonzero = float((out[final_col] != 0).mean())
        print(f"{args.asset} {split_name}: wrote {len(out)} rows -> {out_path}, nonzero_action_rate={nonzero:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
