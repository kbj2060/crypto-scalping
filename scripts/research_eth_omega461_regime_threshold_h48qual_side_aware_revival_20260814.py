#!/usr/bin/env python3
"""Odyssey2 follow-up to #1/#1-followup (regime-specific quality_threshold, h48qual-only asymmetric,
2026-08-13/14 -- rejected: no_gate clean win (PnL+MDD both improved) but with_gate PnL missed baseline
by one metric, gate_pass=False, OOS never opened).

User request (2026-08-14): don't discard it -- diagnose exactly why with_gate PnL fell short and see
if it can be revived. Direct ledger-level diagnosis (interactive, not scripted, see chat transcript)
found the cause: lowering h48qual's quality_threshold by regime opens up BOTH long and short h48qual
entries. The short side is pure value-add (VAL with_gate-active h48qual SHORT trades sum_ret=+0.589),
but the long side is pure noise/wash (VAL with_gate-active h48qual LONG trades sum_ret=+0.005 across 14
trades) -- diluting the portfolio without adding value. This matches the project's long-established,
independently-confirmed-many-times finding that h48qual/direction_head has no real LONG-side skill
(see docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md and
[[h48qual_standalone_replay_invalid]] memory) -- so disabling h48qual's LONG entries specifically is a
principled application of an ALREADY-validated finding, not a new untested hypothesis.

Fix tested: keep h48qual's SHORT threshold at the already-found VAL-selected regime map
(bull=0.30, bear=0.30, chop=0.35, unchanged from the 08-14 h48qual-only-asymmetric experiment), but
make h48qual's LONG threshold regime-INDEPENDENT and swept separately, holding SHORT fixed. Full
0.30-0.80 grid (+1.01 = hard shutoff) run once, side-effect-free (no retraining, cached predictions
only). Result: NOT monotonic across the whole grid -- LONG=0.35 produces an isolated spike (no_gate
+114.46%) bracketed by FAILURES on both sides (0.30 and 0.40 both fail the gate) -- a classic
single-point overfitting artifact (same red-flag pattern as this session's own "MDD identical across
seeds = shared event, not robustness" lesson) and is explicitly REJECTED, not chosen, despite its
attractive headline number. The trustworthy region is LONG>=0.60, which is monotonically increasing
and reaches a STABLE PLATEAU from LONG=0.65 through 1.01 (h48qual LONG trades=0 in all of them,
byte-identical portfolio numbers) -- a wide, robust plateau, not a lucky point. The chosen candidate is
the plateau itself: h48qual LONG effectively disabled (any threshold >=0.65 is equivalent; this script
uses 1.01, an explicit "never passes" sentinel, for clarity over picking an arbitrary grid point).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. Does not touch trading_bot.py/omega4_6_1_live.py/runtime_config.py/.env.
zig075 completely untouched (global flat 0.75 in every regime, same as every prior experiment in this line).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_regime_specific_quality_threshold_20260813 as base  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814"
SHORT_MAP = {"bull": 0.30, "bear": 0.30, "chop": 0.35}  # unchanged from the 08-14 h48qual-only-asymmetric experiment
LONG_SHUTOFF = {"bull": 1.01, "bear": 1.01, "chop": 1.01}  # any value >=0.65 is equivalent on VAL (see docstring)
ZIG075_MAP = {r: 0.75 for r in base.REGIMES}


def log(msg: str) -> None:
    print(f"[side_aware_revival] {msg}", flush=True)


def build_final_action_side_aware(src: pd.DataFrame, prefix: str, thr_short: dict, thr_long: dict) -> np.ndarray:
    dir_action = pd.to_numeric(src[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    quality_for_action = pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    router_expert = src[f"{prefix}router_expert"].astype(str).to_numpy()
    thr_s = np.array([thr_short[r] for r in router_expert], dtype=np.float64)
    thr_l = np.array([thr_long[r] for r in router_expert], dtype=np.float64)
    threshold_per_bar = np.where(dir_action == -1, thr_s, thr_l)
    final_action = dir_action.copy()
    final_action[(dir_action != 0) & (quality_for_action < threshold_per_bar)] = 0
    return final_action


def evaluate_h48qual_side_aware(frame: pd.DataFrame, src: pd.DataFrame, prefix: str,
                                 thr_short: dict, thr_long: dict, *, oof: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = base.base_sweep.COMPONENTS["h48qual"]
    src2 = src.copy()
    src2[f"{prefix}final_action"] = build_final_action_side_aware(src, prefix, thr_short, thr_long)
    keep_ts = set(src2["timestamp"])
    frame2 = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src2 = src2[src2["timestamp"].isin(set(frame2["timestamp"]))].reset_index(drop=True)
    if len(src2) != len(frame2) or not src2["timestamp"].equals(frame2["timestamp"]):
        raise RuntimeError("h48qual: prediction/frame timestamp mismatch")
    x = base.base_sweep.parent._base_input(frame2, base.base_sweep.torch.load(cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    dec_base = base.base_sweep.parent._to_decisions(src2, oof=oof)
    dec, _ = base.base_sweep.atr_eval._apply_atr_safety_sltp(
        dec_base, frame2, atr_window=cfg["atr_window"], tp_mult=cfg["tp_mult"], sl_mult=cfg["sl_mult"],
        min_tp=cfg["min_tp"], min_sl=cfg["min_sl"], max_tp=cfg["max_tp"], max_sl=cfg["max_sl"],
    )
    atr_pct = base.base_sweep.atr_eval._atr_pct(frame2, cfg["atr_window"])
    fee, slip = base.base_sweep.omega._load_fee_slip()
    bundle = base.base_sweep.torch.load(cfg["bundle"], map_location="cpu", weights_only=False)
    loaded = base.base_sweep.parent._load_payloads(bundle["models"], device=base.base_sweep.DEVICE)
    import pickle
    with open(cfg["sidecar_pkl"], "rb") as f:
        pkl = pickle.load(f)
    features = base.base_sweep.rs._risk_feature_frame(frame2, src2, dec, bundle["base_cols"], atr_pct=atr_pct, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = base.base_sweep.rs._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = base.base_sweep.rs._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = base.base_sweep.rs._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base.base_sweep.rs.MARGIN_CFG_KEYS})
    leverage = None
    if pkl["dynamic_leverage"]:
        leverage = base.base_sweep.rs._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in base.base_sweep.rs.LEVERAGE_CFG_KEYS})
    p = dict(frame=frame2, x=x, dec=dec, loaded=loaded, margin=margin, leverage=leverage,
             fee=fee, slip=slip, notional_scaled_sltp=pkl["notional_scaled_sltp"])
    m, _ledger = base.base_sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=p["margin"], risk_leverage=p["leverage"],
        exit_threshold=base.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base.base_sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=base.base_sweep.DEVICE,
    )
    return {k: v for k, v in m.items() if k != "exit_reasons"}, p


def build_portfolio(window: dict, thr_short: dict, thr_long: dict, *, oof: bool) -> tuple[dict, dict, dict]:
    """NOTE: window["frame"] (from gate.load_all_windows) is NOT pre-aligned to either component's
    predictions -- 2025q1/2025q3 have documented coverage gaps vs the base+wide24 frame (see
    eth_omega461_multiwindow_confirmation_gate_20260814.verify_windows docstring; VAL/OOS-Q1/OOS-Q2/
    2025Q2 happen to have 100% coverage so this was never triggered before). portfolio_eval's
    greedy_replay indexes positionally into each component's route/decision arrays using the SAME
    `frame` it's given, so frame must be pre-intersected with BOTH components' raw prediction
    timestamps first, or a coverage-gapped window overruns the shorter component array (confirmed by
    direct reproduction: IndexError inside greedy_replay on 2025q1). This is a pre-existing latent gap
    in research_eth_omega461_regime_specific_quality_threshold_20260813.portfolio_eval's calling
    convention, not new logic -- it was never previously exercised against a coverage-gapped window."""
    prefix = base.base_sweep.omega._tabm_prefix(oof)
    common_ts = (set(window["frame"]["timestamp"]) & set(window["raw"]["h48qual"]["timestamp"]) & set(window["raw"]["zig075"]["timestamp"]))
    frame_common = window["frame"][window["frame"]["timestamp"].isin(common_ts)].reset_index(drop=True)

    m_h48, p_h48 = evaluate_h48qual_side_aware(frame_common, window["raw"]["h48qual"], prefix, thr_short, thr_long, oof=oof)
    baseline_zig_map = {r: base.GLOBAL_BASELINE["zig075"] for r in base.REGIMES}
    m_zig, p_zig = base.evaluate_component("zig075", base.base_sweep.COMPONENTS["zig075"], frame_common, window["raw"]["zig075"], prefix, baseline_zig_map, oof=oof)
    if len(p_h48["frame"]) != len(p_zig["frame"]) or not p_h48["frame"]["timestamp"].equals(p_zig["frame"]["timestamp"]):
        raise RuntimeError("build_portfolio: h48qual/zig075 aligned frames diverge after common-timestamp pre-intersection")
    no_gate, with_gate = base.portfolio_eval(p_h48["frame"], {"h48qual": p_h48, "zig075": p_zig})
    return m_h48, no_gate, with_gate


def run_val() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = gate.load_all_windows()
    val = windows["val"]

    log("stage=G0 baseline (both components at global flat threshold, must reproduce known numbers)")
    baseline_map = {"h48qual": {r: base.GLOBAL_BASELINE["h48qual"] for r in base.REGIMES}, "zig075": {r: base.GLOBAL_BASELINE["zig075"] for r in base.REGIMES}}
    _, no_gate_base, with_gate_base = build_portfolio(val, baseline_map["h48qual"], baseline_map["h48qual"], oof=True)
    g0_ok = (abs(no_gate_base["pnl"] - 36.82) < 0.5 and abs(no_gate_base["mdd"] - (-24.34)) < 0.5 and no_gate_base["trades"] == 29
             and abs(with_gate_base["pnl"] - 54.88) < 0.5 and abs(with_gate_base["mdd"] - (-31.11)) < 0.5)
    log(f"  G0 no_gate={no_gate_base} with_gate={with_gate_base} -> {'PASS' if g0_ok else 'FAIL'}")
    if not g0_ok:
        raise RuntimeError("G0 self-consistency failed")

    log("stage=G0b side-aware code path at LONG=SHORT=0.50 must ALSO reduce exactly to baseline")
    flat50 = {r: 0.50 for r in base.REGIMES}
    m_h48_g0b, no_gate_g0b, with_gate_g0b = build_portfolio(val, flat50, flat50, oof=True)
    g0b_ok = (abs(no_gate_g0b["pnl"] - no_gate_base["pnl"]) < 1e-6 and abs(with_gate_g0b["pnl"] - with_gate_base["pnl"]) < 1e-6)
    log(f"  G0b no_gate={no_gate_g0b} with_gate={with_gate_g0b} -> {'PASS' if g0b_ok else 'FAIL'}")
    if not g0b_ok:
        raise RuntimeError("G0b side-aware-code-path self-consistency failed")

    log("stage=full LONG-threshold grid sweep (SHORT held fixed at found regime map)")
    grid_rows = []
    for thr in base.THRESHOLD_GRID + [1.01]:
        thr_long = {r: thr for r in base.REGIMES}
        m_h48, no_gate, with_gate = build_portfolio(val, SHORT_MAP, thr_long, oof=True)
        beats = (no_gate["pnl"] >= no_gate_base["pnl"] and no_gate["mdd"] >= no_gate_base["mdd"] and
                 with_gate["pnl"] >= with_gate_base["pnl"] and with_gate["mdd"] >= with_gate_base["mdd"])
        row = {"long_threshold": thr, "h48qual_component": m_h48, "no_gate": no_gate, "with_gate": with_gate, "gate_pass": bool(beats)}
        grid_rows.append(row)
        log(f"  LONG={thr:.2f} h48qual_n={m_h48['trades']:2d} no_gate={no_gate['pnl']:+.2f}%/{no_gate['mdd']:+.2f}% "
            f"with_gate={with_gate['pnl']:+.2f}%/{with_gate['mdd']:+.2f}% gate={beats}")

    log("stage=selection -- reject isolated single-point spikes, require a stable neighborhood")
    passing = [r for r in grid_rows if r["gate_pass"]]
    robust = []
    for i, r in enumerate(grid_rows):
        if not r["gate_pass"]:
            continue
        neighbors_pass = []
        if i > 0:
            neighbors_pass.append(grid_rows[i - 1]["gate_pass"])
        if i < len(grid_rows) - 1:
            neighbors_pass.append(grid_rows[i + 1]["gate_pass"])
        if neighbors_pass and all(neighbors_pass):
            robust.append(r)
    log(f"  passing candidates: {[r['long_threshold'] for r in passing]}")
    log(f"  robust (both grid neighbors also pass): {[r['long_threshold'] for r in robust]}")

    chosen_long = LONG_SHUTOFF
    m_h48_final, no_gate_final, with_gate_final = build_portfolio(val, SHORT_MAP, chosen_long, oof=True)
    chosen_beats = (no_gate_final["pnl"] >= no_gate_base["pnl"] and no_gate_final["mdd"] >= no_gate_base["mdd"] and
                     with_gate_final["pnl"] >= with_gate_base["pnl"] and with_gate_final["mdd"] >= with_gate_base["mdd"])
    log(f"  CHOSEN (LONG effectively disabled, thr=1.01): h48qual_n={m_h48_final['trades']} "
        f"no_gate={no_gate_final['pnl']:+.2f}%/{no_gate_final['mdd']:+.2f}% "
        f"with_gate={with_gate_final['pnl']:+.2f}%/{with_gate_final['mdd']:+.2f}% gate={chosen_beats}")

    result = {
        "g0_ok": g0_ok, "g0b_ok": g0b_ok,
        "baseline_no_gate": no_gate_base, "baseline_with_gate": with_gate_base,
        "short_map": SHORT_MAP, "zig075_map": ZIG075_MAP,
        "grid_rows": grid_rows,
        "rejected_isolated_spike": {"long_threshold": 0.35, "reason": "bracketed by FAILURES at 0.30 and 0.40 on both sides -- classic single-point overfitting artifact, explicitly not chosen despite attractive headline number"},
        "chosen_long_map": chosen_long,
        "chosen_component_h48qual": m_h48_final,
        "chosen_no_gate": no_gate_final, "chosen_with_gate": with_gate_final,
        "chosen_gate_pass": bool(chosen_beats),
        "oos_run": False,
    }
    (OUT_DIR / "val_report.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = run_val()
    if r["chosen_gate_pass"]:
        log("VAL gate PASSED -- see companion OOS script for the mandatory dual-window (OOS-Q1+OOS-Q2) single-touch confirmation")
    else:
        log("VAL gate FAILED")
