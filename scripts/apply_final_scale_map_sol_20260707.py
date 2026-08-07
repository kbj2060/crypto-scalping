#!/usr/bin/env python3
"""SOL single-component analogue of the ETH greedy router's final rescaling
stage (scripts/replay_omega4_6_1_greedy_router_20260706.py lines ~150-158):
after the risk sidecar's own margin/leverage output, ETH multiplies leverage
by a per-component-per-side SCALE_MAP factor, then clamps at
LEVERAGE_CAP=5.0 / NOTIONAL_CAP=1.8 (recomputing leverage to match the
clamped notional). That SCALE_MAP is ETH-tuned with no evidence of ever being
re-derived for another asset, so this script VAL-only grid-searches SOL's own
long/short scale factors instead of reusing ETH's {h48qual_L:0.38,
h48qual_S:2.499, zig075_L:2.446, zig075_S:2.478}.

Reuses the SOL risk sidecar's own `_replay_with_risk` bar-by-bar simulator
(not a ledger post-hoc rescale) since notional/leverage are also exit-head
input features -- a different scale can change exit timing, not just PnL
magnitude.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_sol_20260707 as sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(wins / len(ledger)) if len(ledger) else 0.0}


def _scaled_margin_leverage(dec: pd.DataFrame, base_margin: np.ndarray, base_leverage: np.ndarray, *, long_scale: float, short_scale: float) -> tuple[np.ndarray, np.ndarray]:
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    leverage = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional = np.minimum(base_margin * leverage, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = np.where(base_margin > 0.0, notional / np.maximum(base_margin, 1e-12), leverage)
    margin = base_margin
    return margin, leverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707/true_3head_tabm_bundle.pt")
    ap.add_argument("--sidecar-pkl", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707/risk_sidecar.pkl")
    ap.add_argument("--precomputed-prediction-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707")
    ap.add_argument("--precomputed-prediction-tag", default="q070")
    ap.add_argument("--direction-label-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_zigzag_action_labels_20260707")
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.95)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--duration-gate-threshold", type=float, default=0.0055208323)
    ap.add_argument("--max-validation-mdd-abs", type=float, default=30.0)
    ap.add_argument("--scale-grid", default="1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0")
    ap.add_argument("--fixed-long-scale", type=float, default=None)
    ap.add_argument("--fixed-short-scale", type=float, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/sol_final_scale_map_20260707")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    device = parent._device(str(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle_and_sidecar", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, Any] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    with open(args.sidecar_pkl, "rb") as f:
        pkl = pickle.load(f)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode="same_as_direction",
        quality_label_dir=None,
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

    print("stage=apply_atr_contract", flush=True)
    val_dec, _ = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult), min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl))
    oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult), min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl))
    val_atr = atr_eval._atr_pct(frames["val_raw"], int(args.atr_window))
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], int(args.atr_window))

    print("stage=score_and_base_sizing", flush=True)
    val_features = sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=pkl["risk_feature_mode"])
    oos_features = sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_val_all, _ = sidecar._feature_matrix(val_features, pkl["feature_columns"])
    x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    val_side_all = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    if pkl["side_split_model"]:
        val_score = sidecar._predict_side_split_models(pkl["model"], x_val_all, val_side_all)
        oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side_all)
    else:
        val_score = np.asarray(pkl["model"].predict(x_val_all), dtype=np.float64)
        oos_score = np.asarray(pkl["model"].predict(x_oos_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    val_base_margin = sidecar._risk_margins(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    oos_base_margin = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    val_base_leverage = sidecar._risk_leverage(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(val_dec))
    oos_base_leverage = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(oos_dec))

    def _replay(dec: pd.DataFrame, frame: pd.DataFrame, x: pd.DataFrame, margin: np.ndarray, leverage: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
        return sidecar._replay_with_risk(
            frame, x, dec, loaded,
            risk_margin_fraction=margin, risk_leverage=leverage,
            exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult),
            notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device,
        )

    scale_values = [float(x) for x in str(args.scale_grid).split(",") if x.strip()]
    candidates: list[dict[str, Any]] = []
    if args.fixed_long_scale is not None or args.fixed_short_scale is not None:
        if args.fixed_long_scale is None or args.fixed_short_scale is None:
            raise RuntimeError("--fixed-long-scale and --fixed-short-scale must be provided together")
        selected = {"long_scale": float(args.fixed_long_scale), "short_scale": float(args.fixed_short_scale), "validation": None, "eligible": True}
        print("stage=fixed_scale_map_replay", flush=True)
    else:
        print("stage=grid_search_scale_map", flush=True)
        for long_scale in scale_values:
            for short_scale in scale_values:
                val_margin, val_leverage = _scaled_margin_leverage(val_dec, val_base_margin, val_base_leverage, long_scale=long_scale, short_scale=short_scale)
                val_m, val_ledger = _replay(val_dec, frames["val_raw"], x_val, val_margin, val_leverage)
                val_ledger_m = _compound_metrics(val_ledger)
                eligible = float(val_ledger_m["mdd"]) >= -abs(float(args.max_validation_mdd_abs)) and int(val_ledger_m["trades"]) > 0
                candidates.append({"long_scale": long_scale, "short_scale": short_scale, "validation": val_ledger_m, "eligible": bool(eligible)})

        eligible = [c for c in candidates if c["eligible"]]
        if not eligible:
            raise RuntimeError("no eligible scale-map candidate under the validation MDD constraint")
        selected = max(eligible, key=lambda c: float(c["validation"]["pnl"]))

    print("stage=final_replay_with_selected_scale", flush=True)
    val_margin, val_leverage = _scaled_margin_leverage(val_dec, val_base_margin, val_base_leverage, long_scale=selected["long_scale"], short_scale=selected["short_scale"])
    oos_margin, oos_leverage = _scaled_margin_leverage(oos_dec, oos_base_margin, oos_base_leverage, long_scale=selected["long_scale"], short_scale=selected["short_scale"])
    val_m, val_ledger = _replay(val_dec, frames["val_raw"], x_val, val_margin, val_leverage)
    oos_m, oos_ledger = _replay(oos_dec, frames["oos_raw"], x_oos, oos_margin, oos_leverage)

    val_feats_ou = frames["val_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    oos_feats_ou = frames["oos_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    val_ledger["entry_timestamp"] = pd.to_datetime(val_ledger["entry_timestamp"])
    oos_ledger["entry_timestamp"] = pd.to_datetime(oos_ledger["entry_timestamp"])
    val_ledger_g = val_ledger.merge(val_feats_ou, on="entry_timestamp", how="left", validate="one_to_one")
    oos_ledger_g = oos_ledger.merge(oos_feats_ou, on="entry_timestamp", how="left", validate="one_to_one")
    val_gated = val_ledger_g.loc[val_ledger_g["ou_halflife"] > float(args.duration_gate_threshold)].reset_index(drop=True)
    oos_gated = oos_ledger_g.loc[oos_ledger_g["ou_halflife"] > float(args.duration_gate_threshold)].reset_index(drop=True)
    oos_gated_q1 = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)

    report = {
        "method": "sol_single_component_final_scale_map_leverage5x_notional1p8_grid",
        "leverage_cap": LEVERAGE_CAP,
        "notional_cap": NOTIONAL_CAP,
        "scale_grid": scale_values,
        "candidates": candidates,
        "selected_scale": {"long_scale": selected["long_scale"], "short_scale": selected["short_scale"]},
        "no_duration_gate": {
            "validation": _compound_metrics(val_ledger),
            "oos_extended": _compound_metrics(oos_ledger),
        },
        "with_duration_gate": {
            "validation": _compound_metrics(val_gated),
            "oos_extended": _compound_metrics(oos_gated),
            "oos_frozen_q1_2026": _compound_metrics(oos_gated_q1),
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    val_ledger.to_csv(args.out_dir / "validation_ledger.csv", index=False)
    oos_ledger.to_csv(args.out_dir / "oos_ledger.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
