#!/usr/bin/env python3
"""Parent-swap test: reuse the FROZEN tuned SOL TCN sequence-entry-gate (trained/tuned
against the fresh-retrain SOL Omega4.6.1 zig075 parent in
scripts/train_eval_omega462_tcn_gate_sol_tuning_20260722.py, weights at
tmp/causal_regen_20260516/omega462_tcn_gate_sol_tuning_20260722/
sol_tcn_seq_gate_FROZEN_L144_ep24_lr0.0016_20260722.pt) but feed it candidates from the
CURRENTLY-LIVE SOL "adaptive_squeeze" parent instead.

Gate is loaded verbatim (no retraining). Only the parent-decision source is swapped:
  - fresh-retrain parent: sol_omega4_6_1_fresh_retrain_20260722 (risk_feature_mode="all",
    side_split_model=False, dynamic_leverage=False, final scale long=1.0/short=3.0)
  - live adaptive_squeeze parent: TabM bundle
    sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707 + risk sidecar
    sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720 (risk_feature_mode=
    "parent_outputs", side_split_model=True, dynamic_leverage=True) + final scale map from
    sol_final_scale_map_adaptive_squeeze_20260720 (long_scale=1.0, short_scale=1.75) + duration
    gate threshold 0.0055208323 (docs/model_contracts/sol_omega4_6_1_full_stack_20260707_contract.md).
    Feature source: data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2024_2026.csv
    (contains the funding-divisor fix referenced in project memory, extends through 2026-07-21).

All heavy building blocks (slice_bundle, replay_with_gate, build_static_tape,
predict_direction_quality, compound_metrics, the risk-sidecar's parent_outputs /
side_split_model / dynamic_leverage code paths) are REUSED VERBATIM from
scripts/train_eval_omega462_tcn_gate_sol_20260722.py (imported as `base`, not modified) and
scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py (imported inside `base` as `sidecar`).
Only prepare_frame() is reimplemented here (as prepare_frame_live()) because the live parent's
bundle/sidecar paths and risk-scoring contract differ from the fresh-retrain parent's.

Fresh-forward contract: fixed VAL 2025-09-01..2025-12-31 / OOS 2026-01-01..2026-03-31 / fresh
2026-04-01..2026-07-21 (12:00, matching data availability), causal bar-by-bar walk-forward.
fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.

Read-only w.r.t. all fresh-retrain / gate-tuning / adaptive_squeeze-integrity artifact dirs.
New artifacts only under tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_20260722/.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
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

import scripts.train_eval_omega462_tcn_gate_sol_20260722 as base  # noqa: E402
from train_eval_omega462_live_native_sequence_entry_gate_20260703 import load_artifact  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_liveparent_20260722"

FEATURES_PATH_LIVE = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2024_2026.csv"
# NOTE: trading_bot_modules/runtime_config.py's FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH is the
# authoritative live wiring source -- it points at the adaptive_squeeze-retrained bundle below, NOT
# the original (pre-fix) sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707
# bundle. The risk sidecar's own report.json has a "baseline_bundle" field that still shows the old
# zig075_20260707 path, but that field is a stale/unused argparse default left over from the
# --precomputed-prediction-dir training path (risk_feature_mode="parent_outputs" scores off
# precomputed predictions, not a live re-run of the bundle, so that default was never overridden) --
# verified against runtime_config.py's OMEGA4_6_1_SHADOW_ASSET_CONFIG["sol"]["bundle_path"], which is
# what actually runs live.
PARENT_DIR_LIVE = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720"
BUNDLE_PATH_LIVE = PARENT_DIR_LIVE / "true_3head_tabm_bundle.pt"
SIDECAR_PATH_LIVE = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_adaptive_squeeze_q070_20260720/risk_sidecar.pkl"

QUALITY_THRESHOLD = 0.70
DURATION_THRESHOLD = 0.0055208323  # docs/model_contracts/sol_omega4_6_1_full_stack_20260707_contract.md Phase 7
LONG_SCALE = 1.0
SHORT_SCALE = 1.75  # tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720/report.json selected_scale
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
EXIT_THRESHOLD = 0.95
ATR_WINDOW = 192
TP_MULT, SL_MULT = 12.0, 6.0
MIN_TP, MIN_SL, MAX_TP, MAX_SL = 0.075, 0.040, 0.22, 0.12
COST_MULT = 3.0

FRAME_START = base.FRAME_START
FRAME_END_EXTENDED = "2026-07-21 12:00:00"
VAL_START, VAL_END = base.VAL_START, base.VAL_END
OOS_START, OOS_END = base.OOS_START, base.OOS_END
FRESH_START = "2026-04-01 00:00:00"
FRESH_END = "2026-07-21 12:00:00"

FROZEN_GATE_PATH = ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_tuning_20260722/sol_tcn_seq_gate_FROZEN_L144_ep24_lr0.0016_20260722.pt"
FROZEN_LOOKBACK = 144


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=base.json_default) + "\n", encoding="utf-8")


def load_bundle_and_sidecar_live(device: torch.device) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    bundle = torch.load(BUNDLE_PATH_LIVE, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = dict(bundle["models"])
    missing = sorted(set(base.hard.EXPERT_NAMES) - set(models))
    if missing:
        raise RuntimeError(f"live bundle missing experts: {missing}")
    loaded = base.parent._load_payloads(models, device=device)
    with open(SIDECAR_PATH_LIVE, "rb") as f:
        pkl = pickle.load(f)
    return base_cols, loaded, pkl


def prepare_frame_live(device: torch.device) -> dict[str, Any]:
    print("stage=load_frame_live", flush=True)
    df = pd.read_csv(FEATURES_PATH_LIVE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= pd.Timestamp(FRAME_START)) & (df["timestamp"] < pd.Timestamp(FRAME_END_EXTENDED))].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("empty SOL live-parent frame after slicing")

    print("stage=append_regime3", flush=True)
    regime3 = base.Regime3CurrentLiveFeatures(current_path=base.REGIME3_PATH)
    frame = regime3.append(df.copy())

    print("stage=load_bundle_sidecar_live", flush=True)
    base_cols, loaded, pkl = load_bundle_and_sidecar_live(device)
    if pkl.get("risk_feature_mode") != "parent_outputs":
        raise RuntimeError(f"unexpected live sidecar risk_feature_mode: {pkl.get('risk_feature_mode')}")
    if not pkl.get("side_split_model"):
        raise RuntimeError("expected side_split_model=True for live adaptive_squeeze sidecar")
    if not pkl.get("dynamic_leverage"):
        raise RuntimeError("expected dynamic_leverage=True for live adaptive_squeeze sidecar")
    missing_cols = [c for c in base_cols if c not in frame.columns]
    if missing_cols:
        raise RuntimeError(f"frame missing base_cols: {missing_cols[:20]}")

    base_x = base.parent._base_input(frame, base_cols)

    print("stage=route", flush=True)
    route = base.hard._route_id(frame)

    print("stage=tabm_inference", flush=True)
    direction_by_expert, quality_by_expert, direction_arr, quality_arr = base.predict_direction_quality(loaded, base_x, device)
    for idx, expert in enumerate(base.hard.EXPERT_NAMES):
        mask = route == idx
        direction_arr[mask] = direction_by_expert[expert][mask]
        quality_arr[mask] = quality_by_expert[expert][mask]

    dir_action = direction_arr.argmax(axis=1)
    n = len(frame)
    qual_for_action = np.where(dir_action > 0, quality_arr[np.arange(n), dir_action], quality_arr[:, 0])
    final_action = np.where((dir_action != 0) & (qual_for_action >= QUALITY_THRESHOLD), dir_action, 0).astype(np.int64)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0)).astype(np.int64)
    active = final_action != 0
    router_expert_raw = np.asarray(base.hard.EXPERT_NAMES, dtype=object)[route]
    router_expert_scale_key = np.where(router_expert_raw == "chop", "chop_expert", router_expert_raw)

    dec = pd.DataFrame(
        {
            "action": final_action,
            "side": side,
            "notional_exposure": np.where(active, float(base.omega_eth.BASE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(base.omega_eth.BASE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(base.omega_eth.BASE_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(base.omega_eth.BASE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(base.omega_eth.BASE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(base.omega_eth.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(base.omega_eth.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": qual_for_action,
            "confidence": direction_arr.max(axis=1),
            "router_expert": router_expert_scale_key,
        }
    )
    for expert, scale in base.omega_eth.EXPERT_SCALES.items():
        m = active & dec["router_expert"].eq(expert)
        dec.loc[m, "notional_exposure"] = dec.loc[m, "notional_exposure"].astype(float) * float(scale)
        dec.loc[m, "position_fraction"] = dec.loc[m, "position_fraction"].astype(float) * float(scale)

    print("stage=atr_safety_sltp", flush=True)
    dec_atr, _ = base.atr_eval._apply_atr_safety_sltp(
        dec, frame, atr_window=ATR_WINDOW, tp_mult=TP_MULT, sl_mult=SL_MULT, min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL
    )
    atr = base.atr_eval._atr_pct(frame, ATR_WINDOW)

    print("stage=risk_sidecar_score_parent_outputs_side_split", flush=True)
    route_probs = frame[base.hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    router_confidence = route_probs.max(axis=1)
    router_margin = pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64)
    src = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "omega1_regime3_expertdq_router_expert": router_expert_raw,
            "omega1_regime3_expertdq_router_confidence": router_confidence,
            "omega1_regime3_expertdq_router_margin": router_margin,
            "omega1_regime3_expertdq_dir_p_cash": direction_arr[:, 0],
            "omega1_regime3_expertdq_dir_p_long": direction_arr[:, 1],
            "omega1_regime3_expertdq_dir_p_short": direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_confidence": direction_arr.max(axis=1),
            "omega1_regime3_expertdq_dir_side_edge": direction_arr[:, 1] - direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_trade_prob": direction_arr[:, 1] + direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_action": dir_action,
            "omega1_regime3_expertdq_quality_p_cash": quality_arr[:, 0],
            "omega1_regime3_expertdq_quality_p_long": quality_arr[:, 1],
            "omega1_regime3_expertdq_quality_p_short": quality_arr[:, 2],
            "omega1_regime3_expertdq_quality_for_action": qual_for_action,
            "omega1_regime3_expertdq_quality_threshold": np.full(n, QUALITY_THRESHOLD),
            "omega1_regime3_expertdq_final_action": final_action,
        }
    )
    risk_features = base.sidecar._risk_feature_frame(frame, src, dec_atr, base_cols, atr_pct=atr, feature_mode="parent_outputs")
    x_risk, _ = base.sidecar._feature_matrix(risk_features, pkl["feature_columns"])
    side_arr_for_score = dec_atr["side"].to_numpy(dtype=np.int64)
    score = base.sidecar._predict_side_split_models(pkl["model"], x_risk, side_arr_for_score)
    mapping = pkl["selected_mapping"]
    base_margin = base.sidecar._risk_margins(
        dec_atr, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base.sidecar.MARGIN_CFG_KEYS}
    )
    base_leverage = base.sidecar._risk_leverage(
        dec_atr, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base.sidecar.LEVERAGE_CFG_KEYS}
    )

    print("stage=final_scale_map_and_duration_gate", flush=True)
    side_arr = dec_atr["side"].to_numpy(dtype=np.int64)
    scale = np.where(side_arr > 0, LONG_SCALE, np.where(side_arr < 0, SHORT_SCALE, 1.0))
    leverage_scaled = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional_scaled = np.minimum(base_margin * leverage_scaled, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage_scaled = np.where(base_margin > 0.0, notional_scaled / np.maximum(base_margin, 1e-12), leverage_scaled)
    margin_final = base_margin.copy()
    ou_halflife = pd.to_numeric(frame["ou_halflife"], errors="raise").to_numpy(dtype=np.float64)
    duration_ok = ou_halflife > DURATION_THRESHOLD
    margin_final = np.where(duration_ok, margin_final, 0.0)

    print("stage=static_tape", flush=True)
    static_tape, feature_names = base.build_static_tape(frame, dec_atr, atr, margin_final, leverage_scaled)

    return {
        "frame": frame,
        "base_x": base_x,
        "dec_atr": dec_atr,
        "loaded": loaded,
        "margin": margin_final,
        "leverage": leverage_scaled,
        "static_tape": static_tape,
        "feature_names": feature_names,
        "fee_slip": base.omega_sol._load_fee_slip(),
        "sidecar_contract": {
            "risk_feature_mode": pkl.get("risk_feature_mode"),
            "side_split_model": pkl.get("side_split_model"),
            "dynamic_leverage": pkl.get("dynamic_leverage"),
        },
    }


def eval_split(bundle: dict[str, Any], start: str, end: str, artifact: Any, fee: float, slip: float, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    base.LOOKBACK = FROZEN_LOOKBACK
    sl = base.slice_bundle(bundle, start, end)
    parent_metrics, parent_ledger, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=None, collect_labels=False,
    )
    gated_metrics, gated_ledger, _ = base.replay_with_gate(
        frame=sl["frame"], base_x=sl["base_x"], dec=sl["dec_atr"], loaded=bundle["loaded"],
        margin=sl["margin"], leverage=sl["leverage"], static_tape=sl["static_tape"],
        fee=fee, slip=slip, device=device, gate_artifact=artifact, collect_labels=False,
    )
    return parent_metrics, gated_metrics, parent_ledger, gated_ledger


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # sanity: the frozen gate's own original config must match what we claim to reuse
    if not FROZEN_GATE_PATH.exists():
        raise RuntimeError(f"frozen gate weights not found: {FROZEN_GATE_PATH}")

    print("stage=prepare_frame_live", flush=True)
    bundle = prepare_frame_live(device)
    fee, slip = bundle["fee_slip"]

    print("stage=load_frozen_gate", flush=True)
    gate_artifact = load_artifact(FROZEN_GATE_PATH)
    if gate_artifact.lookback != FROZEN_LOOKBACK:
        raise RuntimeError(f"frozen gate lookback mismatch: {gate_artifact.lookback} != {FROZEN_LOOKBACK}")
    if list(gate_artifact.feature_cols) != list(bundle["feature_names"]):
        raise RuntimeError(
            f"feature_cols mismatch between frozen gate and live-parent static tape:\n"
            f"gate={gate_artifact.feature_cols}\nlive={bundle['feature_names']}"
        )

    results: dict[str, Any] = {}
    for split, start, end in (("validation", VAL_START, VAL_END), ("oos_canonical", OOS_START, OOS_END), ("fresh_forward", FRESH_START, FRESH_END)):
        print(f"stage=eval_{split}", flush=True)
        parent_metrics, gated_metrics, parent_ledger, gated_ledger = eval_split(bundle, start, end, gate_artifact, fee, slip, device)
        parent_ledger.to_csv(OUT_DIR / f"{split}_liveparent_alone_ledger.csv", index=False)
        gated_ledger.to_csv(OUT_DIR / f"{split}_liveparent_plus_frozen_tcn_gate_ledger.csv", index=False)
        results[split] = {
            "start": start,
            "end_exclusive": end,
            "parent_alone": parent_metrics,
            "parent_plus_tcn_gate": gated_metrics,
        }
        print(json.dumps({split: results[split]}, default=base.json_default), flush=True)

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.liveparent_swap.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "omega462_tcn_sequence_entry_gate_sol_liveparent_20260722",
        "purpose": "Parent-swap research test: same frozen TCN gate, live adaptive_squeeze SOL parent instead of fresh-retrain SOL parent. Gate NOT retrained.",
        "gate_source": {
            "frozen_weights_path": str(FROZEN_GATE_PATH),
            "trained_against": "sol_omega4_6_1_fresh_retrain_20260722 (zig075-only) candidate stream",
            "lookback": FROZEN_LOOKBACK,
            "epochs": 24,
            "lr": 0.0016,
            "batch_size": 128,
            "seed": 260722,
            "gate_train_end": "2025-06-15 00:00:00",
            "threshold": gate_artifact.threshold,
        },
        "live_parent_config": {
            "bundle_path": str(BUNDLE_PATH_LIVE),
            "sidecar_path": str(SIDECAR_PATH_LIVE),
            "features_path": str(FEATURES_PATH_LIVE),
            "quality_threshold": QUALITY_THRESHOLD,
            "duration_gate_threshold": DURATION_THRESHOLD,
            "final_scale_map": {"long_scale": LONG_SCALE, "short_scale": SHORT_SCALE},
            "exit_threshold": EXIT_THRESHOLD,
            "leverage_cap": LEVERAGE_CAP,
            "notional_cap": NOTIONAL_CAP,
            "cost_mult": COST_MULT,
            "sidecar_contract": bundle["sidecar_contract"],
            "artifact_integrity_audit_source": str(
                ROOT / "tmp/causal_regen_20260516/sol_adaptive_squeeze_artifact_integrity_20260720/omega_artifact_integrity_audit_20260630.json"
            ),
        },
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "validation_window_canonical": [VAL_START, VAL_END],
        "oos_window_canonical": [OOS_START, OOS_END],
        "fresh_forward_window": [FRESH_START, FRESH_END],
        "caveat_gate_distribution_shift": (
            "The frozen gate was trained on counterfactual entry candidates proposed by the "
            "FRESH-RETRAIN parent (different bundle/sidecar/sizing than the live adaptive_squeeze "
            "parent used here). Swapping parents shifts the candidate distribution the gate now "
            "scores at inference; the gate itself is unmodified. See report text for whether veto "
            "behavior here looks consistent with its fresh-retrain-parent behavior or looks "
            "out-of-distribution mismatched."
        ),
        "not_reproduced": (
            "This script's parent-alone numbers are its OWN independent bar-by-bar replay of the "
            "live adaptive_squeeze config on the canonical VAL/OOS/fresh windows using this "
            "harness's TabM/ATR/exit-head/sizing code paths, not a byte-identical replay of "
            "tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720/report.json's "
            "own numbers (that report's own window boundaries and precomputed-prediction-file "
            "windowing differ slightly from this harness's continuous single-frame walk). The "
            "parent-alone vs parent+TCN-gate DELTA within this report is internally consistent "
            "since both use the identical replay harness."
        ),
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(OUT_DIR / "report.json")},
    }
    write_json(OUT_DIR / "report.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["results"], ensure_ascii=False, indent=2, default=base.json_default), flush=True)
