"""Combined sweep: for each notional multiplier, replay the ledger (same as
sweep_notional_multiplier_sol_adaptive_squeeze_20260720.py) then ALSO apply the ETH chop
soft-sizing rule (shadow_trade_return = trade_return * max(0, 1 - chop_prob)) on top, reporting
both the "real" (multiplier only) and "chop-sized" (multiplier + chop soft-sizing) VAL/OOS
pnl/mdd. Goal: find whether combining a higher multiplier with chop soft-sizing's leverage-
reduction effect can land back inside the -25% OOS MDD gate at a higher PnL than the 1.0x/
no-chop-sizing baseline (OOS +57.94%/mdd-21.35%).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import apply_final_scale_map_sol_20260707 as sm  # noqa: E402

sm.omega.TRAIN_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
sm.omega.EVAL_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"

BASELINE_BUNDLE = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720/true_3head_tabm_bundle.pt"
SIDECAR_PKL = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl"
PRECOMPUTED_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720"
DIRECTION_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707"
OUT_PATH = ROOT / "tmp/causal_regen_20260516/sol_adaptive_squeeze_multiplier_x_chop_sweep_20260720/report.json"
SELECTED_LONG_SCALE = 1.0
SELECTED_SHORT_SCALE = 1.75
MULTIPLIER_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
CHOP_COL = "regime3_current_sensitive_wide24_chop_prob"


def _compound(returns: np.ndarray) -> dict:
    cash, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        cash *= 1.0 + float(r)
        peak = max(peak, cash)
        mdd = min(mdd, (cash - peak) / peak)
    return {"pnl": (cash - 1.0) * 100.0, "mdd": mdd * 100.0, "trades": int(len(returns))}


def main() -> int:
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

    val_chop = frames["val_raw"][["timestamp", CHOP_COL]].rename(columns={"timestamp": "entry_timestamp"})
    oos_chop = frames["oos_raw"][["timestamp", CHOP_COL]].rename(columns={"timestamp": "entry_timestamp"})

    results = []
    for mult in MULTIPLIER_GRID:
        val_m, val_ledger = _replay(val_dec, frames["val_raw"], x_val, val_margin, val_leverage_scaled * mult)
        oos_m, oos_ledger = _replay(oos_dec, frames["oos_raw"], x_oos, oos_margin, oos_leverage_scaled * mult)

        val_ledger["entry_timestamp"] = pd.to_datetime(val_ledger["entry_timestamp"])
        oos_ledger["entry_timestamp"] = pd.to_datetime(oos_ledger["entry_timestamp"])
        val_ledger = val_ledger.merge(val_chop, on="entry_timestamp", how="left", validate="one_to_one")
        oos_ledger = oos_ledger.merge(oos_chop, on="entry_timestamp", how="left", validate="one_to_one")

        val_real = _compound(val_ledger["trade_return"].to_numpy(dtype=np.float64))
        oos_real = _compound(oos_ledger["trade_return"].to_numpy(dtype=np.float64))
        val_shadow_mult = np.maximum(0.0, 1.0 - val_ledger[CHOP_COL].to_numpy(dtype=np.float64))
        oos_shadow_mult = np.maximum(0.0, 1.0 - oos_ledger[CHOP_COL].to_numpy(dtype=np.float64))
        val_chopped = _compound(val_ledger["trade_return"].to_numpy(dtype=np.float64) * val_shadow_mult)
        oos_chopped = _compound(oos_ledger["trade_return"].to_numpy(dtype=np.float64) * oos_shadow_mult)

        row = {"multiplier": mult, "val_real": val_real, "oos_real": oos_real,
               "val_chop_sized": val_chopped, "oos_chop_sized": oos_chopped}
        results.append(row)
        print(f"mult={mult} real: val={val_real} oos={oos_real}", flush=True)
        print(f"mult={mult} chop-sized: val={val_chopped} oos={oos_chopped}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"base_scale": {"long_scale": SELECTED_LONG_SCALE, "short_scale": SELECTED_SHORT_SCALE}, "results": results}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
