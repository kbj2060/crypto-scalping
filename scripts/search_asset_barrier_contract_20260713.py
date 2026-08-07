#!/usr/bin/env python3
"""Validation-only asset-specific TP/SL floor search for single-component stacks."""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
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


CONFIG = {
    "sol": {
        "date": "20260707", "component": "zig075", "tag": "q070", "quality": 0.70,
        "parent": "sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707",
        "sidecar": "sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707",
        "quality_mode": "same_as_direction", "quality_dir": None,
        "tp_floors": (0.020, 0.040, 0.060, 0.075), "sl_floors": (0.015, 0.025, 0.040, 0.055),
    },
    "btc": {
        "date": "20260708", "component": "h48qual", "tag": "q055", "quality": 0.55,
        "parent": "btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708",
        "sidecar": "btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708",
        "quality_mode": "quality_label_action", "quality_dir": "btc_h48_conservative_padded_to_zigzag_timestamps_20260708",
        "tp_floors": (0.020, 0.040, 0.060, 0.075), "sl_floors": (0.015, 0.025, 0.040, 0.055),
    },
}


def _compound(ledger: pd.DataFrame) -> dict:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=float):
        equity *= 1.0 + float(ret)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        wins += int(ret > 0)
    return {"pnl": (equity - 1.0) * 100.0, "mdd": mdd * 100.0, "trades": len(ledger), "wr": wins / len(ledger)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", choices=sorted(CONFIG), required=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = CONFIG[args.asset]
    parent = importlib.import_module("train_eval_omega1_2_tabm_3head_20260603")
    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{args.asset}_{cfg['date']}")
    omega4 = importlib.import_module(f"train_eval_omega4_3head_parent72_loose_entry_quality_{args.asset}_{cfg['date']}")
    sidecar = importlib.import_module(f"train_eval_omega4_2_risk_sidecar_{args.asset}_{cfg['date']}")
    atr = importlib.import_module("eval_omega4_1_atr_safety_sltp_20260622")
    device = parent._device(args.device)
    base = ROOT / "tmp/causal_regen_20260516"
    parent_dir = base / cfg["parent"]
    sidecar_dir = base / cfg["sidecar"]
    bundle = torch.load(parent_dir / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    loaded = parent._load_payloads(bundle["models"], device=device)
    with (sidecar_dir / "risk_sidecar.pkl").open("rb") as f:
        pkl = pickle.load(f)
    frames = omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=omega4.LABEL_DIR,
        quality_mode=cfg["quality_mode"],
        quality_label_dir=(base / cfg["quality_dir"]) if cfg["quality_dir"] else None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    pred_dir = parent_dir
    val_src = sidecar._load_precomputed_prediction(pred_dir, "validation", cfg["tag"], frames["val_raw"])
    oos_src = sidecar._load_precomputed_prediction(pred_dir, "oos", cfg["tag"], frames["oos_raw"])
    base_cols = list(bundle["base_cols"])
    x_val = parent._base_input(frames["val_raw"], base_cols)
    x_oos = parent._base_input(frames["oos_raw"], base_cols)
    val_base = parent._to_decisions(val_src, oof=True)
    oos_base = parent._to_decisions(oos_src, oof=False)
    fee, slip = omega._load_fee_slip()
    mapping = pkl["selected_mapping"]

    def prepare(frame, src, dec_base, base_x):
        dec, _ = atr._apply_atr_safety_sltp(dec_base, frame, atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=current_tp, min_sl=current_sl, max_tp=0.22, max_sl=0.12)
        atr_pct = atr._atr_pct(frame, 192)
        features = sidecar._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
        x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
        side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        score = sidecar._predict_side_split_models(pkl["model"], x_all, side) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=float)
        margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
        leverage = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))
        return frame, dec, base_x, margin, leverage

    candidates = []
    for current_tp in cfg["tp_floors"]:
        for current_sl in cfg["sl_floors"]:
            vf, vd, vx, vm, vl = prepare(frames["val_raw"], val_src, val_base, x_val)
            metrics, ledger = sidecar._replay_with_risk(vf, vx, vd, loaded, risk_margin_fraction=vm, risk_leverage=vl, exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)
            cm = _compound(ledger)
            candidates.append({"min_tp": current_tp, "min_sl": current_sl, "validation": cm, "log_risk_utility": float(metrics.get("log_risk_utility", -np.inf))})
    eligible = [c for c in candidates if c["validation"]["trades"] > 0 and c["validation"]["mdd"] >= -35.0]
    selected = max(eligible, key=lambda c: (c["log_risk_utility"], c["validation"]["pnl"])) if eligible else max(candidates, key=lambda c: c["validation"]["pnl"])
    current_tp, current_sl = selected["min_tp"], selected["min_sl"]
    vf, vd, vx, vm, vl = prepare(frames["val_raw"], val_src, val_base, x_val)
    of, od, ox, om, ol = prepare(frames["oos_raw"], oos_src, oos_base, x_oos)
    val_metrics, val_ledger = sidecar._replay_with_risk(vf, vx, vd, loaded, risk_margin_fraction=vm, risk_leverage=vl, exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)
    oos_metrics, oos_ledger = sidecar._replay_with_risk(of, ox, od, loaded, risk_margin_fraction=om, risk_leverage=ol, exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0, notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)
    report = {"asset": args.asset, "component": cfg["component"], "selected_barrier": selected, "validation": _compound(val_ledger), "oos": _compound(oos_ledger), "candidates": candidates, "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    out = args.out_dir or (base / f"{args.asset}_asset_barrier_search_20260713")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    val_ledger.to_csv(out / "validation_ledger.csv", index=False)
    oos_ledger.to_csv(out / "oos_ledger.csv", index=False)
    print(json.dumps({k: report[k] for k in ("asset", "component", "selected_barrier", "validation", "oos")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
