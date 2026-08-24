#!/usr/bin/env python3
"""RESEARCH ONLY -- non-RL fractional-Kelly sizing benchmark for zig075 (2026-08-15).

Motivated by docs/experiments/eth_odyssey4_rl_layer_integration_literature_research_20260815.md
Section 3.3 step 1: before spending effort on an RL-based margin_fraction sizer, check whether a
simple closed-form rule can already beat the existing trained HGB risk-sidecar
(train_eval_omega4_2_risk_sidecar_20260622.py) on the SAME parent_outputs features. Neither this
project's own 2026-06-23 RL-sidecar experiment (docs/model_contracts/
omega4_4_rl_risk_sidecar_v1_full_20260623_contract.md) nor any paper found in that literature
review ran this comparison.

=== Formula ===
Fractional Kelly for a binary win/lose bet with payoff ratio b:1 (Kelly, 1956; Thorp's trading
adaptation): f* = p - (1-p)/b. p = decision_quality_score (the parent's own quality-class
probability for the CHOSEN action -- the same quantity the entry gate itself thresholds against,
an imperfect but directly-available P(good trade) proxy; recent bet-sizing-RL literature (Macri/
Jaimungal/Lillo, arXiv:2511.00190, cited in the RL research doc S2) finds feeding a posterior
probability like this outperforms feeding raw features or a point forecast). b = decision_rr =
take_profit / |stop_loss| price-move ratio, already computed by train_eval_omega4_2_risk_sidecar_
20260622._risk_feature_frame from this component's own ATR-derived TP/SL. Both columns verified
present under risk_feature_mode="parent_outputs" (read directly, not assumed). No fitting, no
random_state -- a closed-form function of two existing columns, zero seed variance.

=== Pipeline-source correction (found this session, disclosed) ===
train_eval_omega4_2_risk_sidecar_20260622.py's own report.json (the deployed zig075 sidecar's
"omega4_2_replayed_baseline") reports OOS pnl/trades from a DIFFERENT underlying feature/frame
pipeline (omega4._prepare_frames, backed by "alpha6/7-lineage" trade_candidates_*.csv, whose OOS
slice is 2026-01-01..02-28 only -- confirmed via docs/experiments/
eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md's "feature_drift" finding) than
the one every other Odyssey2 sizing/exit-timing candidate tonight (queue-pressure, risk-controlled,
conformal-kelly, Gittins) uses (research_eth_omega461_exit_sweep_20260721.load_frame, backed by
data/splits/year_oos/training_features_*.csv + regime3 wide24 overlay, oos_q1=2026-01-01..03-31).
This script therefore does NOT try to reproduce the deployed report.json's exact OOS numbers (a
different pipeline, not directly comparable) -- instead it computes a FRESH deployed-HGB-under-the-
gate-pipeline baseline itself (stage=G0 below) and compares the Kelly candidate against THAT, both
under the identical pipeline, which is the only apples-to-apples comparison available and also
what makes this result directly comparable to every other candidate in tonight's Odyssey2 series.

=== Design ===
zig075 ONLY -- h48qual stays on its currently-deployed HGB-sidecar-driven sizing throughout (same
"freeze everything except the one thing under test" discipline as every other Odyssey2 candidate).

train_q50/train_iqr (the z-score normalization constants _risk_margins/_risk_leverage require) are
recomputed for the Kelly score's own distribution (the deployed pkl's HGB-specific constants do
not apply to a differently-scaled score) from the Kelly score over ACTIVE (quality-gated) rows
pooled across the 3 pre-VAL calibration windows (2025q1+2025q2+2025q3, oof=True/train_predictions
-- same calibration windows research_eth_omega461_conformal_kelly_sizing_scale_20260814.py used).
Simplification vs. the original sidecar script's own convention (train_score_q50/iqr computed over
actual FILLED TRADE rows, not all active rows): using active rows avoids needing a throwaway
baseline replay just to find entry indices, and this is a normalization-only step (not a fitted
target), so the small distributional difference between "active" and "filled" rows should not
matter -- disclosed, not hidden.

Margin mapping (min_scale/max_scale/temp/floor/cap) is grid-searched, VAL-only, small grid (108
combinations: 3x3x3x2x2, long_scale/short_scale fixed at 1.0 matching the deployed mapping's own
selected values) -- deliberately far smaller than the 2304-combination "live_exposure_grid" that
produced the deployed HGB mapping: a "simple, low-researcher-degrees-of-freedom" benchmark that
itself exploited a large grid search would undercut the point of the comparison. Leverage mapping
is NOT re-searched -- fixed, reused verbatim from the deployed zig075 pkl's own selected leverage
config (isolates the score-generation question as the only free variable).

Two-tier grid evaluation, copied structurally from train_eval_omega4_2_risk_sidecar_20260622.py's
own stage=grid_risk_mapping -> selected_full_replay pattern (that script's own reason for this
split applies identically here: margin/leverage changes can shift exit_head-triggered exit timing
by a few bars, per research_eth_omega461_conformal_kelly_sizing_scale_20260814.py's own causality
finding, so a full bar-by-bar replay per grid candidate is the exact evaluation but expensive; a
cheap re-score of one FIXED reference ledger's entries is the standard approximation for the broad
sweep): (1) cheap -- re-score the deployed-HGB reference ledger (same entries/exits, already
computed in G0) at each candidate's margin/leverage via rs._ledger_metrics_with_margins, select by
log_risk_utility under guardrails; (2) exact -- one full base_sweep.replay_exit_variant call for
the winning candidate only, re-checked against the same guardrails.

Guardrails: trades >= floor(0.95 * 28) = 26 (28 = deployed zig075 VAL baseline trade count, from
report.json read this session, i.e. the standard 95%-of-baseline-trades rule the original script
itself enforces); validation_mdd >= -15.0pp (ASSUMPTION, disclosed: the original script's CLI
default is -8.0pp, but the actually-deployed mapping's own VAL MDD is -11.59% (worse than -8%), so
the real promotion run must have used a looser, unlogged value -- -15.0pp was chosen to comfortably
admit the deployed mapping's own observed magnitude rather than silently applying a stricter, wrong
default). Selection objective: log_risk_utility (same log_risk_params as the deployed pkl:
tail_budget=0.02, tail_penalty=0.5, liquidation_buffer=0.12, liquidation_penalty=0.25), VAL only,
matching --selection-scope validation_only enforced by the original script.

Reused UNMODIFIED (imported only): research_eth_omega461_exit_sweep_20260721 (prep_component,
replay_exit_variant, COMPONENTS, COST_MULT, BASELINE_EXIT_THRESHOLD, and via its own .rs/.parent/
.omega/.atr_eval re-exports, train_eval_omega4_2_risk_sidecar_20260622's _risk_margins/
_risk_leverage/_risk_feature_frame/_ledger_metrics_with_margins/MARGIN_CFG_KEYS/LEVERAGE_CFG_KEYS),
eth_omega461_multiwindow_confirmation_gate_20260814 (load_all_windows, align_frame_and_predictions,
run_portfolio_variant, summarize_multiwindow, COMP_CFGS_ASYMMETRIC_TABM_LIVEATR,
REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR, WINDOW_DEFS, ALL_WINDOWS, _close),
research_eth_omega461_exit_head_portfolio_asymmetric_20260813 (_ledger_metrics),
research_eth_omega461_live_sltp_mfe_width_20260813 (_duration_gated, _as_router_component),
replay_omega4_6_1_greedy_router_20260706 (greedy_replay, DURATION_THRESHOLD). NEW code in this
file: _kelly_score, _prep_zig075_score (trimmed sibling of sweep.prep_component / conformal-
kelly's _prep_component_with_score -- computes the Kelly score instead of loading+predicting the
sidecar pkl's HGB model), the two-tier margin-grid search, and the G0/portfolio-confirmation
wiring. direction_head/quality_head/exit_head decision logic is NEVER touched -- h48qual/zig075
labels, quality_threshold, and exit_head weights are all frozen identically to the certified
asymmetric_tabm_liveatr baseline; only zig075's margin_fraction (and, downstream of that via the
frozen-reused leverage mapping, its leverage) is replaced.

=== Compliance ===
fresh_forward_bar_by_bar=true (every ledger comes from an unmodified single forward call to
replay_exit_variant). trade_ledgers_used_as_input=false (no pre-existing ledger CSV is loaded from
disk as an input; the Kelly score needs no trade history at all, unlike Conformal Kelly's rolling
residual pool -- it is a pure per-row function of that row's own quality_score/decision_rr).
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. Does NOT touch
trading_bot.py, trading_bot_modules/omega4_6_1_live.py, runtime_config.py, .env. Does NOT modify
any imported module. No retraining, no GPU (DEVICE=cpu, matching research_eth_omega461_exit_sweep_
20260721's own DEVICE).

quality_threshold caveat (inherited, same as every other Odyssey2 sizing/exit candidate tonight):
zig075's quality_threshold=0.75 was itself OOS-pnl-primary selected against 2026-01-01..02-28 (see
docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The relative
comparison (Kelly candidate vs. freshly-computed HGB baseline within this run) remains meaningful
since both share the identical contaminated entry-selection layer; absolute OOS PnL/MDD are not
clean unbiased forward performance and must not be over-interpreted as such.
"""
from __future__ import annotations

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

import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_fractional_kelly_sizing_benchmark_20260815"
DEVICE = portfolio.DEVICE
ZIG075_CFG = base_sweep.COMPONENTS["zig075"]
COMPONENT_NAMES = ("h48qual", "zig075")
CALIBRATION_WINDOWS = ("2025q1", "2025q2", "2025q3")

with open(ZIG075_CFG["sidecar_pkl"], "rb") as _f:
    _DEPLOYED_PKL = pickle.load(_f)
if _DEPLOYED_PKL["risk_feature_mode"] != "parent_outputs":
    raise RuntimeError("deployed zig075 sidecar risk_feature_mode changed from the verified parent_outputs -- abort")
LEVERAGE_CFG = {k: float(_DEPLOYED_PKL["selected_mapping"][k]) for k in base_sweep.rs.LEVERAGE_CFG_KEYS}

# v2 (widened, 2026-08-15 follow-up): the v1 grid's VAL winner sat at the single most aggressive
# tested value on ALL FIVE axes (min_scale=1.0=lowest, max_scale=2.5=highest, temp=1.7=highest,
# floor=0.18=lowest, cap=0.45=highest) -- a boundary optimum flags that the true optimum may lie
# outside the tested range. Checked (not assumed) that this wasn't just an arbitrarily-narrow
# choice on this script's part: the deployed HGB mapping's own "live_exposure_grid" (train_eval_
# omega4_2_risk_sidecar_20260622.py's production grid) has the IDENTICAL endpoints on every one of
# these axes (min_scale up to 1.0 floor, max_scale/temp/cap capped at 2.50/1.70/0.45) -- so v1's
# grid matched the full plausible region production itself ever explored; Kelly's raw score wants
# to push past even THAT. v2 extends one step beyond each hit boundary in the direction that was
# hit, roughly doubling the combination count (108 -> ~720) rather than reproducing the full
# 2,304-combination production grid (still deliberately smaller than production, to keep this a
# "simple, low-researcher-degrees-of-freedom" benchmark). cap's upward extension is bounded at 0.55
# (not pushed further): with the fixed, reused LEVERAGE_CFG capped at leverage_cap=3.0, cap=0.55
# keeps margin_fraction*leverage <= 1.65 even at the leverage ceiling, safely under the live
# NOTIONAL_CAP=1.8 (trading_bot_modules/omega4_6_1_runtime_contract.py) -- a materially higher cap
# would let this research comparison "win" using notional exposure the live finalize_sizing() gate
# would clip anyway, an unrealistic/undeployable advantage this benchmark should not manufacture.
MARGIN_GRID_AXES: dict[str, tuple[float, ...]] = {
    "min_scale": (0.5, 0.75, 1.0, 1.25, 1.5),
    "max_scale": (1.75, 2.5, 3.0, 3.5),
    "temp": (1.0, 1.7, 2.2, 2.7),
    "floor": (0.08, 0.14, 0.18, 0.26),
    "cap": (0.36, 0.45, 0.55),
}
VAL_BASELINE_TRADES = 28  # deployed zig075 VAL baseline trades, report.json read this session
VAL_TRADE_FLOOR = int(np.floor(VAL_BASELINE_TRADES * 0.95))
VAL_MDD_FLOOR_ASSUMED_PP = -15.0
LOG_RISK_PARAMS = {"tail_budget": 0.02, "tail_penalty": 0.5, "liquidation_buffer": 0.12, "liquidation_penalty": 0.25}


def log(msg: str) -> None:
    print(f"[fractional_kelly] {msg}", flush=True)


def _kelly_score(features: pd.DataFrame) -> np.ndarray:
    p = pd.to_numeric(features["decision_quality_score"], errors="raise").to_numpy(dtype=np.float64)
    b = pd.to_numeric(features["decision_rr"], errors="raise").to_numpy(dtype=np.float64)
    return p - (1.0 - p) / np.maximum(b, 1.0e-8)


def _prep_zig075_score(frame: pd.DataFrame, pred_csv: Path, *, oof: bool) -> dict[str, Any]:
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
    score = _kelly_score(features)

    return dict(frame=frame, x=x, dec=dec, loaded=loaded, fee=fee, slip=slip, score=score,
                notional_scaled_sltp=bool(_DEPLOYED_PKL["notional_scaled_sltp"]))


def _margin_leverage(p: dict[str, Any], margin_cfg: dict[str, float], *, train_q50: float, train_iqr: float) -> tuple[np.ndarray, np.ndarray]:
    margin = base_sweep.rs._risk_margins(p["dec"], p["score"], train_q50=train_q50, train_iqr=train_iqr, **margin_cfg)
    leverage = base_sweep.rs._risk_leverage(p["dec"], p["score"], train_q50=train_q50, train_iqr=train_iqr, **LEVERAGE_CFG)
    return margin, leverage


def _component_ledger(p: dict[str, Any], margin: np.ndarray, leverage: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    return base_sweep.replay_exit_variant(
        p["frame"], p["x"], p["dec"], p["loaded"], risk_margin_fraction=margin, risk_leverage=leverage,
        exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD, fee=p["fee"], slip=p["slip"], cost_mult=base_sweep.COST_MULT,
        notional_scaled_sltp=p["notional_scaled_sltp"], device=DEVICE,
    )


def _portfolio_replay(frame: pd.DataFrame, comp_ps: dict[str, dict[str, Any]], *, fee: float, slip: float) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    router_components = {name: mfe_width._as_router_component(p, exit_threshold=base_sweep.BASELINE_EXIT_THRESHOLD) for name, p in comp_ps.items()}
    _diag, ledger = greedy.greedy_replay(frame, router_components, fee=fee, slip=slip, cost_mult=base_sweep.COST_MULT, device=DEVICE)
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate, ledger


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = base_sweep.omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": __doc__,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "kelly_formula": "f = p - (1-p)/b, p=decision_quality_score, b=decision_rr (take_profit/|stop_loss|)",
        "leverage_cfg_reused_from_deployed_pkl": LEVERAGE_CFG,
        "margin_grid_axes": MARGIN_GRID_AXES,
        "val_trade_floor": VAL_TRADE_FLOOR,
        "val_mdd_floor_assumed_pp": VAL_MDD_FLOOR_ASSUMED_PP,
        "log_risk_params_reused_from_deployed_pkl": LOG_RISK_PARAMS,
        "oos_caveat_quality_threshold_contamination": (
            "zig075 quality_threshold=0.75 was itself OOS-pnl-primary selected against "
            "2026-01-01..02-28 (see docs/experiments/"
            "eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). Relative comparison "
            "(Kelly vs freshly-computed HGB baseline) remains meaningful; absolute OOS PnL/MDD are not."
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
    # stage=G0 -- self-consistency: _prep_zig075_score must produce the IDENTICAL `dec` (side/
    # quality_score/notional_exposure/leverage/take_profit/stop_loss) as the trusted, unmodified
    # sweep.prep_component on the same window (both build `dec` via the exact same
    # parent._to_decisions + atr_eval._apply_atr_safety_sltp calls; only the risk-model layer,
    # which comes AFTER `dec`, differs). Also establishes a FRESH deployed-HGB-under-this-
    # pipeline VAL/OOS baseline+reference-ledger for zig075 alone (see docstring "pipeline-
    # source correction" -- report.json's own numbers are from a different, incompatible
    # feature pipeline and cannot be reproduced here).
    # =====================================================================================
    log("=== stage=G0_self_consistency_and_fresh_hgb_baseline ===")
    g0: dict[str, Any] = {}
    hgb_component_by_window: dict[str, Any] = {}
    for wname in ("val", "oos_q1", "oos_q2"):
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        hgb_p = base_sweep.prep_component("zig075", ZIG075_CFG, aligned_frame, aligned_paths["zig075"], oof=w["oof"])
        kelly_p_check = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"])
        cmp_cols = ["side", "quality_score", "notional_exposure", "leverage", "take_profit", "stop_loss"]
        dec_match = bool(hgb_p["dec"][cmp_cols].equals(kelly_p_check["dec"][cmp_cols]))
        hgb_m, hgb_ledger = _component_ledger(hgb_p, hgb_p["margin"], hgb_p["leverage"])
        hgb_component_by_window[wname] = {"p": hgb_p, "metrics": hgb_m, "ledger": hgb_ledger}
        g0[wname] = {"dec_match": dec_match, "hgb_fresh_baseline": hgb_m}
        log(f"  {wname}: dec_match={dec_match} hgb_fresh_baseline pnl={hgb_m['pnl']:.2f}% mdd={hgb_m['mdd']:.2f}% trades={hgb_m['trades']}")
    g0_pass = all(row["dec_match"] for row in g0.values())
    report["g0"] = {"windows": g0, "pass": g0_pass}
    log(f"stage=G0_result pass={g0_pass}")
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 self-consistency check failed -- _prep_zig075_score's dec construction diverges from the trusted sweep.prep_component on at least one window. Aborting before trusting any Kelly number."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base_sweep.omega._json_default) + "\n", encoding="utf-8")
        return 1

    # =====================================================================================
    # stage=calibration -- Kelly score train_q50/train_iqr from ACTIVE rows pooled across
    # 2025q1+2025q2+2025q3 (see docstring "Design" for why active rows, not filled-trade rows).
    # =====================================================================================
    log("=== stage=calibration ===")
    calib_scores: list[np.ndarray] = []
    for wname in CALIBRATION_WINDOWS:
        w = windows[wname]
        aligned_frame, aligned_paths = aligned[wname]
        p = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"])
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
    # stage=VAL_grid -- two-tier margin-mapping grid (see docstring), VAL-only, log_risk_utility
    # selection with guardrails, matching train_eval_omega4_2_risk_sidecar_20260622.py's own
    # selection discipline (validation_only scope, trades>=0.95*baseline, MDD floor).
    # =====================================================================================
    log("=== stage=VAL_grid ===")
    val_frame, val_paths = aligned["val"]
    val_kelly_p = _prep_zig075_score(val_frame, val_paths["zig075"], oof=windows["val"]["oof"])
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
                        margin, leverage = _margin_leverage(val_kelly_p, margin_cfg, train_q50=train_q50, train_iqr=train_iqr)
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
    winner_margin, winner_leverage = _margin_leverage(val_kelly_p, winner_cheap["margin_cfg"], train_q50=train_q50, train_iqr=train_iqr)
    winner_exact_m, winner_exact_ledger = _component_ledger(val_kelly_p, winner_margin, winner_leverage)
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
    # (zig075 alone, vs the FRESH HGB baseline established in G0) and portfolio-level (h48qual
    # frozen on asymmetric_tabm_liveatr, zig075 on the Kelly winner, vs the true
    # asymmetric_tabm_liveatr baseline), single-touch oos_q1+oos_q2 via gate.summarize_multiwindow
    # (same criterion every other Odyssey2 candidate tonight was judged by).
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
            hgb_m, _ = _component_ledger(hgb_p, hgb_p["margin"], hgb_p["leverage"])

        kelly_p = _prep_zig075_score(aligned_frame, aligned_paths["zig075"], oof=w["oof"])
        kelly_margin, kelly_leverage = _margin_leverage(kelly_p, winner_cheap["margin_cfg"], train_q50=train_q50, train_iqr=train_iqr)
        kelly_m, _kelly_ledger = _component_ledger(kelly_p, kelly_margin, kelly_leverage)
        kelly_p_sized = dict(kelly_p, margin=kelly_margin, leverage=kelly_leverage)
        component_table[wname] = {"tier": gate.WINDOW_DEFS[wname]["tier"], "hgb_fresh_baseline": hgb_m, "kelly_candidate": kelly_m}
        log(f"  {wname} [{gate.WINDOW_DEFS[wname]['tier']}] component zig075: hgb={hgb_m['pnl']:.2f}%/{hgb_m['mdd']:.2f}%/{hgb_m['trades']}  "
            f"kelly={kelly_m['pnl']:.2f}%/{kelly_m['mdd']:.2f}%/{kelly_m['trades']}")

        h48qual_p = base_sweep.prep_component("h48qual", gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], aligned_frame, aligned_paths["h48qual"], oof=w["oof"])
        base_no_gate, base_with_gate, _ = _portfolio_replay(aligned_frame, {"h48qual": h48qual_p, "zig075": hgb_p}, fee=fee, slip=slip)
        cand_no_gate, cand_with_gate, _ = _portfolio_replay(aligned_frame, {"h48qual": h48qual_p, "zig075": kelly_p_sized}, fee=fee, slip=slip)
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
