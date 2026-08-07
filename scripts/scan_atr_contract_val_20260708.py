#!/usr/bin/env python3
"""Asset-agnostic VAL-only scan of the ATR TP/SL barrier contract
(atr_window/tp_mult/sl_mult/min_tp/min_sl/max_tp/max_sl), which every SOL/BTC
build so far has copied byte-for-byte from ETH's own tuned values
(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040,
max_tp=0.22, max_sl=0.12) without ever re-checking whether a different
asset's volatility calls for a different barrier width. Since BTC's realized
5m volatility is materially lower than ETH's, the ETH-tuned floor
(min_tp=7.5%/min_sl=4%) may be too wide relative to BTC's ATR, producing the
very sparse trade counts observed (12-37 trades) -- this scans a uniform
"barrier scale" multiplier applied to tp_mult/sl_mult/min_tp/min_sl/max_tp/
max_sl together (keeping their ratios fixed) on VAL only, using the REAL
production replay (parent forward pass + ATR contract + trained exit-head +
baseline BASE_TEMPLATE sizing, matching _replay_with_risk) -- not the parent
trainer's simplified fixed-TP/SL screening metric, which we already learned
misranks configs relative to the real stack.

Only the baseline (BASE_TEMPLATE) sizing is scanned here -- no risk-sizing
grid search -- to keep the scan fast; a promising scale should be re-run
through the full sidecar/duration-gate/scale-map pipeline afterward.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

atr_eval = importlib.import_module("eval_omega4_1_atr_safety_sltp_20260622")
parent = importlib.import_module("train_eval_omega1_2_tabm_3head_20260603")

BASE_ATR = {"atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0, "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega-module", required=True, help="e.g. train_eval_omega1_2_tabm_diffusion_risk_btc_20260708")
    ap.add_argument("--omega4-module", required=True, help="e.g. train_eval_omega4_3head_parent72_loose_entry_quality_btc_20260708")
    ap.add_argument("--sidecar-module", required=True, help="e.g. train_eval_omega4_2_risk_sidecar_btc_20260708 (reused only for helper functions)")
    ap.add_argument("--baseline-bundle", type=Path, required=True)
    ap.add_argument("--precomputed-prediction-dir", type=Path, required=True)
    ap.add_argument("--precomputed-prediction-tag", required=True)
    ap.add_argument("--direction-label-dir", type=Path, required=True)
    ap.add_argument("--quality-mode", choices=["same_as_direction", "quality_label_action"], default="same_as_direction")
    ap.add_argument("--quality-label-dir", type=Path, default=None)
    ap.add_argument("--quality-threshold", type=float, required=True)
    ap.add_argument("--exit-threshold", type=float, default=0.95)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--scales", default="0.35,0.5,0.7,1.0,1.4")
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    omega = importlib.import_module(args.omega_module)
    omega4 = importlib.import_module(args.omega4_module)
    sidecar = importlib.import_module(args.sidecar_module)
    hard = importlib.import_module("train_omega1_regime3_expert_direction_head_volpca_20260602")

    device = parent._device(str(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle", flush=True)
    bundle = torch.load(args.baseline_bundle, map_location=device, weights_only=False)
    models: dict[str, Any] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=args.direction_label_dir,
        quality_mode=str(args.quality_mode),
        quality_label_dir=args.quality_label_dir,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    pred_dir = Path(args.precomputed_prediction_dir)
    tag = str(args.precomputed_prediction_tag)
    val_src = sidecar._load_precomputed_prediction(pred_dir, "validation", tag, frames["val_raw"])
    oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", tag, frames["oos_raw"])
    x_val = parent._base_input(frames["val_raw"], base_cols)
    x_oos = parent._base_input(frames["oos_raw"], base_cols)
    val_dec_base = parent._to_decisions(val_src, oof=True)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)

    scales = [float(s.strip()) for s in str(args.scales).split(",") if s.strip()]
    rows: list[dict[str, Any]] = []
    for scale in scales:
        cfg = {
            "atr_window": int(args.atr_window),
            "tp_mult": BASE_ATR["tp_mult"] * scale,
            "sl_mult": BASE_ATR["sl_mult"] * scale,
            "min_tp": BASE_ATR["min_tp"] * scale,
            "min_sl": BASE_ATR["min_sl"] * scale,
            "max_tp": BASE_ATR["max_tp"] * scale,
            "max_sl": BASE_ATR["max_sl"] * scale,
        }
        val_dec, val_atr_diag = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], **cfg)
        oos_dec, oos_atr_diag = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **cfg)
        val_m, _ = sidecar._replay_with_risk(
            frames["val_raw"], x_val, val_dec, loaded,
            risk_margin_fraction=None, risk_leverage=None,
            exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult),
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device,
        )
        oos_m, _ = sidecar._replay_with_risk(
            frames["oos_raw"], x_oos, oos_dec, loaded,
            risk_margin_fraction=None, risk_leverage=None,
            exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult),
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device,
        )
        row = {
            "scale": scale, **cfg,
            "validation_pnl": val_m["pnl"], "validation_mdd": val_m["mdd"], "validation_trades": val_m["trades"], "validation_wr": val_m["wr"],
            "validation_exit_reasons": val_m["exit_reasons"],
            "oos_pnl": oos_m["pnl"], "oos_mdd": oos_m["mdd"], "oos_trades": oos_m["trades"], "oos_wr": oos_m["wr"],
            "oos_exit_reasons": oos_m["exit_reasons"],
        }
        rows.append(row)
        print(json.dumps(row, default=_json_default), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "atr_scale_scan.csv", index=False)
    print(f"\nWrote {args.out_dir / 'atr_scale_scan.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
