#!/usr/bin/env python3
"""BTC Omega4.6.1 replica final replay for the selected single component.

This is the BTC analogue of `apply_final_scale_map_sol_20260707.py`, but it
does not rerun a broad scale grid by default. The broad BTC search was already
done in `fast_search_omega4_6_1_asset_params_20260708.py`; this script takes
that selected candidate and performs the real bar-by-bar replay with the final
scale-map applied before exit-head evaluation.
"""
from __future__ import annotations

import argparse
import json
import pickle
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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_btc_20260708 as sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_swingtransition_20260806 as omega4  # noqa: E402

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
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
    }


def _scaled_margin_leverage(dec: pd.DataFrame, base_margin: np.ndarray, base_leverage: np.ndarray, *, long_scale: float, short_scale: float) -> tuple[np.ndarray, np.ndarray]:
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    scale = np.where(side > 0, float(long_scale), np.where(side < 0, float(short_scale), 1.0))
    leverage = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional = np.minimum(base_margin * leverage, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = np.where(base_margin > 0.0, notional / np.maximum(base_margin, 1e-12), leverage)
    return base_margin, leverage


def _duration_search(ledger: pd.DataFrame, *, min_trade_ratio: float = 0.50, max_mdd_abs: float = 30.0) -> dict[str, Any]:
    baseline = _compound_metrics(ledger)
    candidates: list[dict[str, Any]] = [
        {"threshold": 0.0, "quantile": None, "validation": baseline, "eligible": baseline["trades"] > 0, "priority_score": baseline["pnl"]}
    ]
    if ledger.empty:
        return {"selected": candidates[0], "candidates": candidates}
    floor = max(1, int(np.floor(len(ledger) * float(min_trade_ratio))))
    for q in np.arange(0.05, 0.85, 0.05):
        th = float(np.quantile(ledger["ou_halflife"].to_numpy(dtype=np.float64), q))
        gated = ledger.loc[ledger["ou_halflife"] > th].reset_index(drop=True)
        m = _compound_metrics(gated)
        eligible = int(m["trades"]) >= floor and float(m["mdd"]) >= -abs(float(max_mdd_abs))
        candidates.append({"threshold": th, "quantile": float(q), "validation": m, "eligible": bool(eligible), "priority_score": float(m["pnl"]) if eligible else float("-inf")})
    selected = max([c for c in candidates if c["eligible"]], key=lambda c: float(c["priority_score"]))
    return {"selected": selected, "candidates": candidates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_20260806_swingtransition/true_3head_tabm_bundle.pt")
    ap.add_argument("--sidecar-pkl", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708/risk_sidecar.pkl")
    ap.add_argument("--precomputed-prediction-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_swingtransition_20260806_h48qual_freshforward_ext_20260806")
    ap.add_argument("--precomputed-prediction-tag", default="q055")
    ap.add_argument("--quality-threshold", type=float, default=0.55)
    ap.add_argument("--long-scale", type=float, default=0.5)
    ap.add_argument("--short-scale", type=float, default=2.5)
    ap.add_argument("--exit-threshold", type=float, default=0.95)
    ap.add_argument("--duration-gate-threshold", type=float, default=None)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_swingtransition_freshforward_ext_20260806")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    device = parent._device(str(args.device))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle_and_sidecar", flush=True)
    bundle = torch.load(args.baseline_bundle, map_location=device, weights_only=False)
    models: dict[str, Any] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    with open(args.sidecar_pkl, "rb") as f:
        pkl = pickle.load(f)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_freshforward_ext_20260802",
        quality_mode="quality_label_action",
        quality_label_dir=ROOT / "tmp/causal_regen_20260516/btc_h48_conservative_padded_freshforward_ext_20260802",
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
    atr_kwargs = dict(atr_window=192, tp_mult=12.0, sl_mult=6.0, min_tp=0.075, min_sl=0.040, max_tp=0.22, max_sl=0.12)
    val_dec, _ = atr_eval._apply_atr_safety_sltp(val_dec_base, frames["val_raw"], **atr_kwargs)
    oos_dec, _ = atr_eval._apply_atr_safety_sltp(oos_dec_base, frames["oos_raw"], **atr_kwargs)
    val_atr = atr_eval._atr_pct(frames["val_raw"], 192)
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], 192)

    print("stage=score_and_base_sizing", flush=True)
    val_features = sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=pkl["risk_feature_mode"])
    oos_features = sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=pkl["risk_feature_mode"])
    x_val_all, _ = sidecar._feature_matrix(val_features, pkl["feature_columns"])
    x_oos_all, _ = sidecar._feature_matrix(oos_features, pkl["feature_columns"])
    val_side = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    oos_side = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    val_score = sidecar._predict_side_split_models(pkl["model"], x_val_all, val_side) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_val_all), dtype=np.float64)
    oos_score = sidecar._predict_side_split_models(pkl["model"], x_oos_all, oos_side) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_oos_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    val_base_margin = sidecar._risk_margins(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    oos_base_margin = sidecar._risk_margins(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    val_base_leverage = sidecar._risk_leverage(val_dec, val_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(val_dec))
    oos_base_leverage = sidecar._risk_leverage(oos_dec, oos_score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(oos_dec))

    val_margin, val_leverage = _scaled_margin_leverage(val_dec, val_base_margin, val_base_leverage, long_scale=args.long_scale, short_scale=args.short_scale)
    oos_margin, oos_leverage = _scaled_margin_leverage(oos_dec, oos_base_margin, oos_base_leverage, long_scale=args.long_scale, short_scale=args.short_scale)

    print("stage=final_replay", flush=True)
    val_m, val_ledger = sidecar._replay_with_risk(frames["val_raw"], x_val, val_dec, loaded, risk_margin_fraction=val_margin, risk_leverage=val_leverage, exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)
    oos_m, oos_ledger = sidecar._replay_with_risk(frames["oos_raw"], x_oos, oos_dec, loaded, risk_margin_fraction=oos_margin, risk_leverage=oos_leverage, exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), notional_scaled_sltp=False, exit_sizing_input_mode="actual", device=device)

    val_ou = frames["val_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    oos_ou = frames["oos_raw"][["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp"})
    val_ledger["entry_timestamp"] = pd.to_datetime(val_ledger["entry_timestamp"])
    oos_ledger["entry_timestamp"] = pd.to_datetime(oos_ledger["entry_timestamp"])
    val_ledger_g = val_ledger.merge(val_ou, on="entry_timestamp", how="left", validate="one_to_one")
    oos_ledger_g = oos_ledger.merge(oos_ou, on="entry_timestamp", how="left", validate="one_to_one")
    duration = _duration_search(val_ledger_g)
    selected_threshold = float(args.duration_gate_threshold) if args.duration_gate_threshold is not None else float(duration["selected"]["threshold"])
    val_gated = val_ledger_g.loc[val_ledger_g["ou_halflife"] > selected_threshold].reset_index(drop=True)
    oos_gated = oos_ledger_g.loc[oos_ledger_g["ou_halflife"] > selected_threshold].reset_index(drop=True)
    oos_q1 = oos_gated.loc[oos_gated["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)

    report = {
        "method": "btc_single_component_final_scale_map_exact_replay",
        "component": "h48qual",
        "quality_threshold": float(args.quality_threshold),
        "precomputed_prediction_tag": tag,
        "long_scale": float(args.long_scale),
        "short_scale": float(args.short_scale),
        "leverage_cap": LEVERAGE_CAP,
        "notional_cap": NOTIONAL_CAP,
        "duration_gate": duration,
        "selected_duration_threshold": selected_threshold,
        "no_duration_gate": {"validation": _compound_metrics(val_ledger), "oos_extended": _compound_metrics(oos_ledger)},
        "with_duration_gate": {
            "validation": _compound_metrics(val_gated),
            "oos_extended": _compound_metrics(oos_gated),
            "oos_frozen_q1_2026": _compound_metrics(oos_q1),
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
