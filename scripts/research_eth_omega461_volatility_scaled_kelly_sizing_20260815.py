#!/usr/bin/env python3
"""RESEARCH ONLY -- volatility-scaled fractional-Kelly sizing benchmark for zig075 (2026-08-15),
follow-up to research_eth_omega461_fractional_kelly_sizing_benchmark_20260815.py after the user
asked to research a different non-RL sizing rule (plain Kelly did not beat the deployed HGB
sidecar on VAL, even after the grid was widened past production's own explored range).

=== Why this candidate, not another blind guess ===
Diagnosed (not assumed) reason plain Kelly (f = p - (1-p)/b) underperformed: the deployed zig075
sidecar's own report.json atr_diag shows take_profit/stop_loss pinned EXACTLY at their ATR floors
(min_tp=0.075, min_sl=0.040) at BOTH the 50th and 90th percentile, in every split (train/
validation/oos) -- i.e. decision_rr = take_profit/|stop_loss| is a near-constant 0.075/0.040=1.875
for at least 90% of rows. Kelly's b-term therefore carries almost no cross-sectional information
for the great majority of trades, so plain Kelly degenerates to (nearly) a monotonic function of
p=decision_quality_score alone -- a single scalar with a narrow spread (train_iqr=0.073 on this
component, per the prior benchmark's calibration diagnostic). HGB, by contrast, has direct access
to atr_pct_runtime itself (continuous, NOT floor/cap-clipped) as one of its ~12 parent_outputs
features. This candidate restores that lost information with an explicit inverse-volatility
multiplicative term on the Kelly fraction -- the same "volatility-scaled position sizing"
principle the RL literature review already cited as well-established (Zhang, Zohren, Roberts,
"Deep RL for Trading", arXiv:1911.10107, cited in docs/experiments/
eth_odyssey4_rl_layer_integration_literature_research_20260815.md S2) -- not a new, unmotivated
guess.

=== Formula ===
score = kelly_score * clip(atr_ref / atr_pct_runtime, VOL_SCALE_FLOOR, VOL_SCALE_CAP)
kelly_score = p - (1-p)/b (identical to the plain-Kelly candidate, imported unmodified from
research_eth_omega461_fractional_kelly_sizing_benchmark_20260815._kelly_score). atr_ref = median
atr_pct_runtime over ACTIVE rows of the SAME 3 pre-VAL calibration windows (2025q1+q2+q3) the
plain-Kelly candidate's train_q50/iqr used -- causal, no lookahead. VOL_SCALE_FLOOR/CAP = (0.5,
2.0), FIXED (not grid-searched) -- deliberately parameter-light: this candidate's whole premise is
"restore one specific, diagnosed missing signal with the fewest possible new free parameters", so
it should not itself become a second multi-axis grid search on top of the margin-mapping grid.

All downstream mechanics (margin-mapping grid over min_scale/max_scale/temp/floor/cap -- reuses
the prior script's v2 WIDENED grid directly, since the margin-mapping search space is orthogonal
to the score-generation question this candidate changes -- VAL-only two-tier selection,
guardrails, six-window component+portfolio confirmation) are REUSED UNMODIFIED from
research_eth_omega461_fractional_kelly_sizing_benchmark_20260815.py by import: those functions
(_margin_leverage, _component_ledger, _portfolio_replay, LEVERAGE_CFG, MARGIN_GRID_AXES,
VAL_TRADE_FLOOR, VAL_MDD_FLOOR_ASSUMED_PP, LOG_RISK_PARAMS) only consume an already-computed
`score` array (or reuse config constants) and do not care how that score was produced. NEW code in
this file: _volatility_kelly_score, _prep_zig075_score (renamed copy of the imported module's own
_prep_zig075_score -- identical except the final score computation), _atr_ref_from_calibration,
and this file's own main() orchestration (G0/calibration/VAL-grid/six-window stages, mirroring the
imported module's own stage structure).

=== Compliance ===
Same as research_eth_omega461_fractional_kelly_sizing_benchmark_20260815.py (fresh_forward_bar_by_
bar=true, no saved ledgers/exit timestamps as input, zig075-only intervention, h48qual frozen, no
GPU, no live files touched) -- inherited via the reused functions; re-verified here only for the
one new component (the score computation itself, via the G0 self-consistency check below).
"""
from __future__ import annotations

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

import research_eth_omega461_fractional_kelly_sizing_benchmark_20260815 as fk  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_volatility_scaled_kelly_sizing_20260815"
DEVICE = fk.DEVICE
ZIG075_CFG = fk.ZIG075_CFG
LEVERAGE_CFG = fk.LEVERAGE_CFG
MARGIN_GRID_AXES = fk.MARGIN_GRID_AXES
VAL_TRADE_FLOOR = fk.VAL_TRADE_FLOOR
VAL_MDD_FLOOR_ASSUMED_PP = fk.VAL_MDD_FLOOR_ASSUMED_PP
LOG_RISK_PARAMS = fk.LOG_RISK_PARAMS
CALIBRATION_WINDOWS = fk.CALIBRATION_WINDOWS
COMPONENT_NAMES = fk.COMPONENT_NAMES
VOL_SCALE_FLOOR = 0.5
VOL_SCALE_CAP = 2.0


def log(msg: str) -> None:
    print(f"[vol_kelly] {msg}", flush=True)


def _volatility_kelly_score(features: pd.DataFrame, *, atr_ref: float) -> np.ndarray:
    kelly = fk._kelly_score(features)
    atr = pd.to_numeric(features["atr_pct_runtime"], errors="raise").to_numpy(dtype=np.float64)
    vol_scale = np.clip(atr_ref / np.maximum(atr, 1.0e-8), VOL_SCALE_FLOOR, VOL_SCALE_CAP)
    return kelly * vol_scale


def _prep_zig075_score(frame: pd.DataFrame, pred_csv: Path, *, oof: bool, atr_ref: float) -> dict[str, Any]:
    bundle = torch.load(ZIG075_CFG["bundle"], map_location="cpu", weights_only=False)
    base_cols = bundle["base_cols"]
    models = bundle["models"]

    src_raw = pd.read_csv(pred_csv)
    for c in src_raw.columns:
        if str(src_raw[c].dtype).lower().startswith("str"):
            src_raw[c] = src_raw[c].astype(object)
    src_raw["timestamp"] = pd.to_datetime(src_raw["timestamp"])
    keep_ts = set(src_raw["timestamp"])
    frame = frame[frame["timestamp"].isin(keep_ts)].reset_index(drop=True)
    src = src_raw[src_raw["timestamp"].isin(set(frame["timestamp"]))].reset_index(drop=True)
    if len(src) != len(frame) or not src["timestamp"].equals(frame["timestamp"]):
        raise RuntimeError(f"zig075: prediction/frame timestamp mismatch ({len(src)} vs {len(frame)})")

    x = base_sweep.parent._base_input(frame, base_cols)
    dec_base = base_sweep.parent._to_decisions(src, oof=oof)
    dec, _atr_diag = base_sweep.atr_eval._apply_atr_safety_sltp(
        dec_base, frame, atr_window=ZIG075_CFG["atr_window"], tp_mult=ZIG075_CFG["tp_mult"], sl_mult=ZIG075_CFG["sl_mult"],
        min_tp=ZIG075_CFG["min_tp"], min_sl=ZIG075_CFG["min_sl"], max_tp=ZIG075_CFG["max_tp"], max_sl=ZIG075_CFG["max_sl"],
    )
    atr_pct = base_sweep.atr_eval._atr_pct(frame, ZIG075_CFG["atr_window"])
    fee, slip = base_sweep.omega._load_fee_slip()
    loaded = base_sweep.parent._load_payloads(models, device=DEVICE)

    features = base_sweep.rs._risk_feature_frame(frame, src, dec, base_cols, atr_pct=atr_pct, feature_mode="parent_outputs")
    score = _volatility_kelly_score(features, atr_ref=atr_ref)

    return dict(frame=frame, x=x, dec=dec, loaded=loaded, fee=fee, slip=slip, score=score,
                notional_scaled_sltp=bool(fk._DEPLOYED_PKL["notional_scaled_sltp"]))


def _atr_ref_from_calibration(windows: dict[str, Any], aligned: dict[str, tuple[pd.DataFrame, dict[str, Path]]]) -> float:
    pooled: list[np.ndarray] = []
    for wname in CALIBRATION_WINDOWS:
        aligned_frame, aligned_paths = aligned[wname]
        p = fk._prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=windows[wname]["oof"])
        atr_pct = base_sweep.atr_eval._atr_pct(p["frame"], ZIG075_CFG["atr_window"])
        active = np.asarray(base_sweep.omega._active(p["dec"]))
        pooled.append(atr_pct[active])
    return float(np.median(np.concatenate(pooled)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = base_sweep.omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": __doc__,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "score_formula": "kelly_score * clip(atr_ref/atr_pct_runtime, VOL_SCALE_FLOOR, VOL_SCALE_CAP); kelly_score = p-(1-p)/b",
        "vol_scale_floor": VOL_SCALE_FLOOR, "vol_scale_cap": VOL_SCALE_CAP,
        "margin_grid_axes_reused_from_v2": MARGIN_GRID_AXES,
        "leverage_cfg_reused_from_deployed_pkl": LEVERAGE_CFG,
        "val_trade_floor": VAL_TRADE_FLOOR, "val_mdd_floor_assumed_pp": VAL_MDD_FLOOR_ASSUMED_PP,
        "log_risk_params_reused_from_deployed_pkl": LOG_RISK_PARAMS,
        "diagnosis_motivating_this_candidate": (
            "deployed zig075 report.json atr_diag: tp_p50==tp_p90==min_tp(0.075) and "
            "sl_p50==sl_p90==min_sl(0.040) in train/validation/oos -- decision_rr is a near-constant "
            "1.875 for >=90% of rows, so plain Kelly's b-term carries almost no information and it "
            "degenerates to (nearly) a function of quality_score alone. This candidate restores "
            "atr_pct_runtime (continuous, unclipped) as an explicit inverse-volatility multiplier."
        ),
    }

    log("=== stage=load_and_align ===")
    windows = gate.load_all_windows()
    q_tags = {name: base_sweep.COMPONENTS[name]["q_tag"] for name in COMPONENT_NAMES}
    aligned: dict[str, tuple[pd.DataFrame, dict[str, Path]]] = {}
    for wname, wd in gate.WINDOW_DEFS.items():
        aligned_frame, aligned_paths = gate.align_frame_and_predictions(windows[wname]["frame"], q_tags, wd["split"], OUT_DIR / wname)
        aligned[wname] = (aligned_frame, aligned_paths)
        log(f"  aligned {wname}: rows={len(aligned_frame)}")

    # =====================================================================================
    # stage=G0 -- self-consistency: this file's own _prep_zig075_score must produce the
    # IDENTICAL `dec` as fk._prep_zig075_score (already G0-verified against the trusted
    # sweep.prep_component in the prior script's run) on the same window -- a valid transitive
    # check (mine==fk's, fk's==trusted original => mine==trusted original). Also recomputes the
    # fresh deployed-HGB-under-this-pipeline baseline (not hand-copied from the prior run, to
    # avoid citing stale/mistyped numbers).
    # =====================================================================================
    log("=== stage=G0_self_consistency_and_fresh_hgb_baseline ===")
    atr_ref = _atr_ref_from_calibration(windows, aligned)
    log(f"  atr_ref (median atr_pct_runtime, active rows, 2025q1-q3)={atr_ref:.6f}")

    g0: dict[str, Any] = {}
    hgb_component_by_window: dict[str, Any] = {}
    for wname in ("val", "oos_q1", "oos_q2"):
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        hgb_p = base_sweep.prep_component("zig075", ZIG075_CFG, aligned_frame, aligned_paths["zig075"], oof=w["oof"])
        fk_p = fk._prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"])
        vk_p_check = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"], atr_ref=atr_ref)
        cmp_cols = ["side", "quality_score", "notional_exposure", "leverage", "take_profit", "stop_loss"]
        dec_match = bool(fk_p["dec"][cmp_cols].equals(vk_p_check["dec"][cmp_cols]))
        hgb_m, hgb_ledger = fk._component_ledger(hgb_p, hgb_p["margin"], hgb_p["leverage"])
        hgb_component_by_window[wname] = {"p": hgb_p, "metrics": hgb_m, "ledger": hgb_ledger}
        g0[wname] = {"dec_match_vs_plain_kelly_prep": dec_match, "hgb_fresh_baseline": hgb_m}
        log(f"  {wname}: dec_match={dec_match} hgb_fresh_baseline pnl={hgb_m['pnl']:.2f}% mdd={hgb_m['mdd']:.2f}% trades={hgb_m['trades']}")
    g0_pass = all(row["dec_match_vs_plain_kelly_prep"] for row in g0.values())
    report["g0"] = {"windows": g0, "pass": g0_pass, "atr_ref": atr_ref}
    log(f"stage=G0_result pass={g0_pass}")
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 self-consistency check failed. Aborting before trusting any volatility-Kelly number."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
        return 1

    # =====================================================================================
    # stage=calibration -- volatility-Kelly score train_q50/train_iqr from ACTIVE rows pooled
    # across 2025q1+2025q2+2025q3 (same discipline as the plain-Kelly candidate; a fresh
    # calibration is required because multiplying by vol_scale changes the score distribution).
    # =====================================================================================
    log("=== stage=calibration ===")
    calib_scores: list[np.ndarray] = []
    for wname in CALIBRATION_WINDOWS:
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        p = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"], atr_ref=atr_ref)
        active_mask = np.asarray(base_sweep.omega._active(p["dec"]))
        calib_scores.append(p["score"][active_mask])
        log(f"  {wname}: active_rows={int(active_mask.sum())}")
    pooled = np.concatenate(calib_scores)
    train_q50 = float(np.quantile(pooled, 0.50))
    train_iqr = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    report["calibration"] = {"n_active_pooled": int(len(pooled)), "train_q50": train_q50, "train_iqr": train_iqr,
                              "score_min": float(pooled.min()), "score_max": float(pooled.max())}
    log(f"  train_q50={train_q50:.6f} train_iqr={train_iqr:.6f} (n={len(pooled)}, range=[{pooled.min():.4f},{pooled.max():.4f}])")

    # =====================================================================================
    # stage=VAL_grid -- same two-tier margin-mapping grid search as the plain-Kelly candidate,
    # reusing the v2 widened MARGIN_GRID_AXES directly.
    # =====================================================================================
    log("=== stage=VAL_grid ===")
    val_frame, val_paths = aligned["val"]
    val_p = _prep_zig075_score(val_frame, val_paths["zig075"], oof=windows["val"]["oof"], atr_ref=atr_ref)
    hgb_val_m = hgb_component_by_window["val"]["metrics"]
    hgb_val_reference_ledger = hgb_component_by_window["val"]["ledger"]
    log(f"  fresh HGB baseline (VAL, this pipeline): pnl={hgb_val_m['pnl']:.2f}% mdd={hgb_val_m['mdd']:.2f}% trades={hgb_val_m['trades']}")

    candidates: list[dict[str, Any]] = []
    for min_scale in MARGIN_GRID_AXES["min_scale"]:
        for max_scale in MARGIN_GRID_AXES["max_scale"]:
            if max_scale <= min_scale:
                continue
            for temp in MARGIN_GRID_AXES["temp"]:
                for floor in MARGIN_GRID_AXES["floor"]:
                    for cap in MARGIN_GRID_AXES["cap"]:
                        if cap <= floor:
                            continue
                        margin_cfg = {"min_scale": min_scale, "max_scale": max_scale, "temp": temp, "floor": floor, "cap": cap, "long_scale": 1.0, "short_scale": 1.0}
                        margin, leverage = fk._margin_leverage(val_p, margin_cfg, train_q50=train_q50, train_iqr=train_iqr)
                        cheap_m, _ = base_sweep.rs._ledger_metrics_with_margins(val_frame, hgb_val_reference_ledger, margin, leverage, **LOG_RISK_PARAMS)
                        candidates.append({"margin_cfg": margin_cfg, "cheap_metrics": cheap_m, "log_risk_utility": cheap_m["log_risk_utility"]})

    eligible = [c for c in candidates if c["cheap_metrics"]["trades"] >= VAL_TRADE_FLOOR and c["cheap_metrics"]["mdd"] >= VAL_MDD_FLOOR_ASSUMED_PP]
    log(f"  grid candidates={len(candidates)} eligible={len(eligible)} (trade_floor={VAL_TRADE_FLOOR}, mdd_floor={VAL_MDD_FLOOR_ASSUMED_PP})")
    report["val_grid"] = {
        "n_candidates": len(candidates), "n_eligible": len(eligible),
        "fresh_hgb_baseline_val": hgb_val_m,
        "top10_by_log_risk_utility_cheap": sorted(candidates, key=lambda c: c["log_risk_utility"], reverse=True)[:10],
    }

    if not eligible:
        report["stage_reached"] = "VAL_grid"
        report["val_winner"] = None
        report["final_verdict"] = "REJECTED_VAL_GATE_NO_ELIGIBLE_CANDIDATE"
        report["gate_pass"] = True
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
        log("stage=done -- no VAL candidate passed the guardrails (cheap tier), stopping (negative result).")
        return 0

    winner_cheap = max(eligible, key=lambda c: (c["log_risk_utility"], c["cheap_metrics"]["mdd"], c["cheap_metrics"]["pnl"]))
    winner_margin, winner_leverage = fk._margin_leverage(val_p, winner_cheap["margin_cfg"], train_q50=train_q50, train_iqr=train_iqr)
    winner_exact_m, winner_exact_ledger = fk._component_ledger(val_p, winner_margin, winner_leverage)
    winner_passes_exact = bool(winner_exact_m["trades"] >= VAL_TRADE_FLOOR and winner_exact_m["mdd"] >= VAL_MDD_FLOOR_ASSUMED_PP)
    beats_hgb_component_val = bool(winner_exact_m["pnl"] >= hgb_val_m["pnl"] and winner_exact_m["mdd"] >= hgb_val_m["mdd"])
    report["val_winner"] = {
        "margin_cfg": winner_cheap["margin_cfg"], "cheap_metrics": winner_cheap["cheap_metrics"], "exact_metrics": winner_exact_m,
        "passes_guardrails_under_exact_replay": winner_passes_exact,
        "beats_fresh_hgb_baseline_val_component_level": beats_hgb_component_val,
    }
    log(f"  VAL winner (cheap-selected): {winner_cheap['margin_cfg']}")
    log(f"    cheap:  pnl={winner_cheap['cheap_metrics']['pnl']:.2f}% mdd={winner_cheap['cheap_metrics']['mdd']:.2f}% trades={winner_cheap['cheap_metrics']['trades']}")
    log(f"    exact:  pnl={winner_exact_m['pnl']:.2f}% mdd={winner_exact_m['mdd']:.2f}% trades={winner_exact_m['trades']} "
        f"passes_guardrails={winner_passes_exact} beats_hgb_component_val={beats_hgb_component_val}")

    # =====================================================================================
    # stage=six_window -- winner's margin_cfg applied to all 6 windows, both component-level
    # (zig075 alone) and portfolio-level (h48qual frozen, zig075 on this candidate), single-
    # touch oos_q1+oos_q2 via gate.summarize_multiwindow.
    # =====================================================================================
    log("=== stage=six_window_component_and_portfolio ===")
    component_table: dict[str, Any] = {}
    portfolio_baseline: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    portfolio_candidate: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for wname in gate.ALL_WINDOWS:
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]

        if wname in hgb_component_by_window:
            hgb_p, hgb_m = hgb_component_by_window[wname]["p"], hgb_component_by_window[wname]["metrics"]
        else:
            hgb_p = base_sweep.prep_component("zig075", ZIG075_CFG, aligned_frame, aligned_paths["zig075"], oof=w["oof"])
            hgb_m, _ = fk._component_ledger(hgb_p, hgb_p["margin"], hgb_p["leverage"])

        vk_p = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"], atr_ref=atr_ref)
        vk_margin, vk_leverage = fk._margin_leverage(vk_p, winner_cheap["margin_cfg"], train_q50=train_q50, train_iqr=train_iqr)
        vk_m, _vk_ledger = fk._component_ledger(vk_p, vk_margin, vk_leverage)
        vk_p_sized = dict(vk_p, margin=vk_margin, leverage=vk_leverage)
        component_table[wname] = {"tier": gate.WINDOW_DEFS[wname]["tier"], "hgb_fresh_baseline": hgb_m, "volatility_kelly_candidate": vk_m}
        log(f"  {wname} [{gate.WINDOW_DEFS[wname]['tier']}] component zig075: hgb={hgb_m['pnl']:.2f}%/{hgb_m['mdd']:.2f}%/{hgb_m['trades']}  "
            f"vol_kelly={vk_m['pnl']:.2f}%/{vk_m['mdd']:.2f}%/{vk_m['trades']}")

        h48qual_p = base_sweep.prep_component("h48qual", gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], aligned_frame, aligned_paths["h48qual"], oof=w["oof"])
        base_no_gate, base_with_gate, _ = fk._portfolio_replay(aligned_frame, {"h48qual": h48qual_p, "zig075": hgb_p}, fee=fee, slip=slip)
        cand_no_gate, cand_with_gate, _ = fk._portfolio_replay(aligned_frame, {"h48qual": h48qual_p, "zig075": vk_p_sized}, fee=fee, slip=slip)
        portfolio_baseline[wname] = (base_no_gate, base_with_gate)
        portfolio_candidate[wname] = (cand_no_gate, cand_with_gate)
        log(f"  {wname} portfolio: baseline no_gate={base_no_gate['pnl']:.2f}%/{base_no_gate['mdd']:.2f}%/{base_no_gate['trades']} with_gate={base_with_gate['pnl']:.2f}%/{base_with_gate['mdd']:.2f}%/{base_with_gate['trades']}  |  "
            f"candidate no_gate={cand_no_gate['pnl']:.2f}%/{cand_no_gate['mdd']:.2f}%/{cand_no_gate['trades']} with_gate={cand_with_gate['pnl']:.2f}%/{cand_with_gate['mdd']:.2f}%/{cand_with_gate['trades']}")

    ref_val_ng, ref_val_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["val"]
    ref_oos_ng, ref_oos_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR["oos_q1"]
    portfolio_baseline_check = {
        "val": {"no_gate": gate._close(portfolio_baseline["val"][0], ref_val_ng), "with_gate": gate._close(portfolio_baseline["val"][1], ref_val_wg)},
        "oos_q1": {"no_gate": gate._close(portfolio_baseline["oos_q1"][0], ref_oos_ng), "with_gate": gate._close(portfolio_baseline["oos_q1"][1], ref_oos_wg)},
    }
    log(f"  portfolio baseline cross-check vs published asymmetric_tabm_liveatr reference: {portfolio_baseline_check}")

    summary_strict = gate.summarize_multiwindow(portfolio_baseline, portfolio_candidate, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(portfolio_baseline, portfolio_candidate, mdd_slack_pp=3.0)
    log(f"portfolio verdict: strict(mdd0pp)={summary_strict['final_verdict']} relaxed(mdd3pp)={summary_relaxed['final_verdict']}")

    report["six_window_component_table"] = component_table
    report["portfolio_baseline_cross_check_vs_published_reference"] = portfolio_baseline_check
    report["portfolio_summary_strict_mdd0pp"] = summary_strict
    report["portfolio_summary_relaxed_mdd3pp"] = summary_relaxed
    report["final_verdict"] = summary_strict["final_verdict"] if summary_strict["oos_confirm_all_pass_single_touch"] else (
        summary_relaxed["final_verdict"] if summary_relaxed["oos_confirm_all_pass_single_touch"] else "REJECTED_SIGN_MISMATCH")
    report["stage_reached"] = "six_window_component_and_portfolio"
    report["gate_pass"] = True
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
    log(f"stage=done final_verdict={report['final_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
