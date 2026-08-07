"""Sweep a post-hoc notional multiplier on top of the adaptive-squeeze scale-map's selected
config (long_scale=1.0, short_scale=3.0, gate off), mirroring how trading_bot.py's real live path
applies FINAL_GOVERNOR_OMEGA4_6_1_{ETH,SOL}_NOTIONAL_MULTIPLIER: notional *= multiplier with
margin_fraction held fixed and NOT re-clamped against NOTIONAL_CAP=1.8 (that cap only applies
inside the scale-map stage itself) -- equivalently leverage *= multiplier since notional =
margin_fraction * leverage.

Reuses apply_final_scale_map_sol_20260707.py's own loading/scoring/_scaled_margin_leverage/_replay
machinery unmodified, just adds one more multiplication step before the final replay.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_sol_20260707 as sm  # noqa: E402

sm.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
sm.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

BASELINE_BUNDLE = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt"
SIDECAR_PKL = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl"
PRECOMPUTED_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
OUT_PATH = ROOT / "tmp/causal_regen_20260516/sol_adaptive_squeeze_notional_multiplier_sweep_20260720/report.json"
SELECTED_LONG_SCALE = 1.0
SELECTED_SHORT_SCALE = 1.75
MULTIPLIER_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def main() -> int:
    import argparse
    import pickle

    import torch

    device = sm.parent._device("cpu")
    bundle = torch.load(BASELINE_BUNDLE, map_location=device, weights_only=False)
    models = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = sm.parent._load_payloads(models, device=device)
    with open(SIDECAR_PKL, "rb") as f:
        pkl = pickle.load(f)

    frames = sm.omega4._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIRECTION_LABEL_DIR,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0, quality_max_mae=0.0, quality_min_mfe_mae=0.0, quality_max_hold_bars=0,
    )
    fee, slip = sm.omega._load_fee_slip()
    tag = "q070"
    val_src = sm.sidecar._load_precomputed_prediction(PRECOMPUTED_DIR, "validation", tag, frames["val_raw"])
    oos_src = sm.sidecar._load_precomputed_prediction(PRECOMPUTED_DIR, "oos", tag, frames["oos_raw"])
    x_val = sm.parent._base_input(frames["val_raw"], base_cols)
    x_oos = sm.parent._base_input(frames["oos_raw"], base_cols)
    val_dec = sm.parent._to_decisions(val_src, oof=True)
    oos_dec = sm.parent._to_decisions(oos_src, oof=False)

    val_dec, _ = sm.atr_eval._apply_atr_safety_sltp(val_dec, frames["val_raw"], atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
    oos_dec, _ = sm.atr_eval._apply_atr_safety_sltp(oos_dec, frames["oos_raw"], atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
    val_atr = sm.atr_eval._atr_pct(frames["val_raw"], 192)
    oos_atr = sm.atr_eval._atr_pct(frames["oos_raw"], 192)

    val_features = sm.sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=pkl["risk_feature_mode"])
    oos_features = sm.sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_val_all, _ = sm.sidecar._feature_matrix(val_features, pkl["feature_columns"])
    x_oos_all, _ = sm.sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    val_side = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    oos_side = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    val_score = sm.sidecar._predict_side_split_models(pkl["model"], x_val_all, val_side)
    oos_score = sm.sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side)
    mapping = pkl["selected_mapping"]
    val_base_margin = sm.sidecar._risk_margins(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sm.sidecar.MARGIN_CFG_KEYS})
    oos_base_margin = sm.sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sm.sidecar.MARGIN_CFG_KEYS})
    val_base_leverage = sm.sidecar._risk_leverage(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sm.sidecar.LEVERAGE_CFG_KEYS})
    oos_base_leverage = sm.sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sm.sidecar.LEVERAGE_CFG_KEYS})

    val_margin, val_leverage_scaled = sm._scaled_margin_leverage(val_dec, val_base_margin, val_base_leverage, long_scale=SELECTED_LONG_SCALE, short_scale=SELECTED_SHORT_SCALE)
    oos_margin, oos_leverage_scaled = sm._scaled_margin_leverage(oos_dec, oos_base_margin, oos_base_leverage, long_scale=SELECTED_LONG_SCALE, short_scale=SELECTED_SHORT_SCALE)

    def _replay(dec, frame, x, margin, leverage):
        return sm.sidecar._replay_with_risk(
            frame, x, dec, loaded, risk_margin_fraction=margin, risk_leverage=leverage,
            exit_threshold=0.95, fee=fee, slip=slip, cost_mult=3.0,
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device,
        )

    results = []
    for mult in MULTIPLIER_GRID:
        val_m, val_ledger = _replay(val_dec, frames["val_raw"], x_val, val_margin, val_leverage_scaled * mult)
        oos_m, oos_ledger = _replay(oos_dec, frames["oos_raw"], x_oos, oos_margin, oos_leverage_scaled * mult)
        val_metrics = sm._compound_metrics(val_ledger)
        oos_metrics = sm._compound_metrics(oos_ledger)
        results.append({"multiplier": mult, "validation": val_metrics, "oos_extended": oos_metrics})
        print(f"multiplier={mult} val={val_metrics} oos={oos_metrics}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "base_scale": {"long_scale": SELECTED_LONG_SCALE, "short_scale": SELECTED_SHORT_SCALE},
        "results": results,
        "fresh_forward_bar_by_bar": True,
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
