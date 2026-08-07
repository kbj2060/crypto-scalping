#!/usr/bin/env python3
"""Re-score SOL/BTC's frozen parent bundle on the extended (through 2026-07-12) features file,
WITHOUT retraining -- mirrors build_omega4_6_1_extended_parent_predictions_20260706.py for ETH.
Only the `oos_predictions_{tag}.csv` file needs regenerating (VAL window, 2025-10-01..12-31, is
unaffected by extending 2026 data); writes it in place inside the existing parent bundle's out_dir
so replay_omega4_6_1_two_component_router_assets_20260708.py's
sidecar._load_precomputed_prediction() picks it up transparently.
"""
from __future__ import annotations

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

ASSETS = {
    "sol": {
        "date": "20260707",
        "parent_dir": ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707",
        "tag": "q070",
        "threshold": 0.70,
    },
    "btc": {
        "date": "20260708",
        "parent_dir": ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708",
        "tag": "q055",
        "threshold": 0.55,
    },
}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(ASSETS), required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    cfg = ASSETS[args.asset]
    omega4 = importlib.import_module(f"train_eval_omega4_3head_parent72_loose_entry_quality_{args.asset}_{cfg['date']}")
    device = parent._device(str(args.device))

    bundle = torch.load(cfg["parent_dir"] / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = bundle["models"]

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction" if args.asset == "sol" else "quality_label_action",
        quality_label_dir=None if args.asset == "sol" else ROOT / f"tmp/causal_regen_20260516/btc_h48_conservative_padded_to_zigzag_timestamps_20260708",
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    oos_raw = frames["oos_raw"]
    print(f"{args.asset} extended OOS frame: {len(oos_raw)} rows, {oos_raw['timestamp'].min()}..{oos_raw['timestamp'].max()}", flush=True)

    x_oos = parent._base_input(oos_raw, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x_oos, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(oos_raw)
    oos_direction = parent._routed(preds, route, "direction", 3)
    oos_quality = parent._routed(preds, route, "quality", 3)
    oos_src_oof = parent._prediction_output(oos_raw, oos_direction, oos_quality, threshold=float(cfg["threshold"]), prefix="omega1_regime3_expertdq_oof")
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})

    out_path = cfg["parent_dir"] / f"oos_predictions_{cfg['tag']}.csv"
    nonzero = float((oos_src[f"omega1_regime3_expertdq_final_action"] != 0).mean())
    oos_src.to_csv(out_path, index=False)
    print(f"{args.asset}: wrote {len(oos_src)} rows -> {out_path}, nonzero_action_rate={nonzero:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
