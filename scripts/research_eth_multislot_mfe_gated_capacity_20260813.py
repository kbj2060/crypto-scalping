#!/usr/bin/env python3
"""ETH multi-slot (N=3) MFE-GATED capacity ADMISSION test -- pre-registered.

Contract: docs/experiments/eth_multislot_mfe_gated_capacity_20260813.json (written FIRST, before any
number below was computed).

eth_multislot_capacity_transfer_20260808.json already tested N=3 with EQUAL-BUDGET sizing and NO
admission-policy change (every slot uses h48qual/zig075's own quality_threshold, unchanged) and
FAILED its G2 VAL falsification gate (PnL +36.82%->+14.15%, -62%) despite exposure being nearly
unchanged (time-weighted notional only -7.8% lower) -- diagnosed there as a MARGINAL-SIGNAL-QUALITY
problem, not exposure dilution. That contract explicitly forbids re-sweeping N or budget allocation
under its own terms; this is therefore a NEW, separate contract testing a qualitatively different
slot-ADMISSION policy, not a re-parameterization of the old one.

The ONE variable this script changes relative to BOTH the incumbent (N=1) and 08-08's failed N=3 arm:
entries attempted while >=1 other slot is already occupied (i.e. this admission is INCREMENTAL over
what a single slot could have taken) must additionally clear
  direction_matched_predicted_MFE(bar) >= quantile(TRAIN prediction distribution, 0.70)
using this sub-project's MFE quantile-regression head -- the ONLY signal in the whole Odyssey/h48qual
research line ever to clear its own pre-registered MI/R^2 gate
(docs/experiments/eth_h48qual_mfe_quantile_quality_regression_20260812.md). The base slot
(occupied_count==0 at entry) is completely untouched: same signal, same priority order, same 1/3
margin sizing as 08-08's own first-filled slot.

Reuses, does NOT reimplement:
  - research_eth_multislot_capacity_transfer_20260808.multislot_replay (lines 59-166 of that file) --
    imported UNMODIFIED, used directly for every N=1 baseline computation below, and copied then
    minimally extended (exactly one new admission condition) into multislot_replay_mfe_gated() here.
  - research_eth_multislot_capacity_transfer_20260808.run_window / metrics / incremental_split /
    G0_EXPECTED / OOS_PRED_DIR / OUT_DIR -- reused unmodified.
  - research_eth_omega461_live_sltp_mfe_width_20260813.base102_panel / train_mfe_models /
    val_sanity_gate / _load_tb_labels -- reused unmodified (the exact MFE-regressor training recipe
    that passed this sub-project's MI/R^2 gate, already used once tonight for a different mechanism).
  - replay_omega4_6_1_greedy_router_20260706.prepare_component / PRIORITY / SCALE_MAP /
    LEVERAGE_CAP / NOTIONAL_CAP, retest_omega4_6_1_extended_oos_20260706.COMPONENTS /
    load_frame_current / DEVICE / COST_MULT, replay_omega4_6_1_greedy_val_20260706.load_val_frame /
    VAL_PRED, research_eth_omega461_exit_sweep_20260721.load_frame / BASE_2025 / WIDE24_2025 (TRAIN
    frame only) -- all reused unmodified.

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

VAL-FIRST BY DEFAULT: with no --stage flag (or --stage val), this script runs G0 -> G0b -> G0c ->
(per-seed) G1 -> G2 and stops there, writing result.json regardless of pass/fail. That VAL-only run
(2026-08-13) passed both G1 and G2 aggregate (5/5 seeds each) -- result backed up verbatim at
tmp/eth_multislot_mfe_gated_capacity_20260813/result_val_only_20260813.json before this file was
touched again. The coordinator reviewed that result directly and explicitly authorized spending the
contract's pre-registered, ONE-TIME OOS read (G3/G4): passing --stage all additionally builds OOS
components (with the same regime3 NaN-gap forward-fill patch as G0), re-scores each of the 5 SEEDS'
already-fitted MFE models on OOS (no retraining -- same in-memory models used for VAL), runs
multislot_replay_mfe_gated on the OOS extended window (2026-01-01..06-30, the 08-08 line's own
window, NOT the shorter 2026-01-01..03-31 window sibling exit_sweep-based scripts use tonight), and
evaluates G3 (per contract: OOS PnL/MDD/both-quarters bar, >=4/5 seeds) then G4 (pooled effect size,
only if G3 passes). --stage all first re-verifies (hard assertion) that this run's own VAL numbers
reproduce the already-published, coordinator-reviewed result exactly, before trusting anything it
computes on OOS -- protecting the one-time read from being silently spent on a subtly-changed
pipeline. No re-tuning after seeing the OOS result is permitted, per the contract.

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as gval  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as base_sweep  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_src  # noqa: E402
import research_eth_multislot_capacity_transfer_20260808 as base_multi  # noqa: E402

CONTRACT = ROOT / "docs/experiments/eth_multislot_mfe_gated_capacity_20260813.json"
OUT_DIR = ROOT / "tmp/eth_multislot_mfe_gated_capacity_20260813"
N_SLOTS = 3
SEEDS = [454090186, 918777617, 130430114, 828152837, 415921410]
MFE_CUTOFF_QUANTILE = 0.70
TRAIN_START, TRAIN_END = "2025-01-01", "2025-09-30"
G0_TOL_PP = 0.05
G2_MDD_SLACK_PP = 3.0
G3_PNL_MARGIN_PP = 3.0  # OOS PnL must beat N=1 OOS PnL by this much -- matches 08-08's own G3 margin
G3_MDD_FLOOR = -18.5  # matches 08-08's own G3 MDD floor
R_PERM = 20000
AGGREGATE_MIN_PASS = 4  # out of 5 seeds -- matches this session's own established seed-robustness bar
PERM_RNG_SEED = 20260813  # permutation-test RNG only, distinct from the 5 MFE-training seeds above
OOS_START, OOS_END = "2026-01-01", "2026-06-30"  # 08-08's extended window -- NOT the shorter Q1-only
# window (2026-01-01..03-31) sibling exit_sweep-based scripts use tonight; see contract window_honesty.


# --------------------------------------------------------------------------------- MFE-gated replay
@torch.no_grad()
def multislot_replay_mfe_gated(frame: pd.DataFrame, components: dict, *, n_slots: int,
                                mfe_pred: dict[str, np.ndarray], mfe_cutoff: dict[str, float],
                                fee: float, slip: float, cost_mult: float, device) -> pd.DataFrame:
    """Copy of research_eth_multislot_capacity_transfer_20260808.multislot_replay, extended with
    exactly ONE new condition. Every other line is unchanged from that function.

    New condition: an entry attempted while >=1 other slot is already occupied (occupied_count>=1,
    i.e. this admission is incremental over what a single slot could take) must additionally clear
    mfe_pred[direction][i] >= mfe_cutoff[direction], direction in {"long","short"} matching the
    candidate's side. Entries into a fully-flat account (occupied_count==0) are never gated -- same
    eligibility as the unmodified function's first-filled slot.

    `mfe_pred` = {"long": np.ndarray, "short": np.ndarray}, precomputed once per seed outside this
    loop, aligned 1:1 by row position to `frame` (causal, point-in-time features only -- no lookahead
    beyond what the unmodified function's own signals already use).
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64)
              for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)

    pre = {}
    for name, comp in components.items():
        act = omega._active(comp["dec"])
        pre[name] = {
            "active": np.asarray(act.to_numpy() if hasattr(act, "to_numpy") else act, dtype=bool),
            "side": pd.to_numeric(comp["dec"]["side"], errors="raise").to_numpy(dtype=np.int64),
            "tp": pd.to_numeric(comp["dec"]["take_profit"], errors="raise").to_numpy(dtype=np.float64),
            "sl": pd.to_numeric(comp["dec"]["stop_loss"], errors="raise").to_numpy(dtype=np.float64),
            "margin": np.asarray(comp["margin"], dtype=np.float64),
            "leverage": np.asarray(comp["leverage"], dtype=np.float64),
        }

    slots: list[dict[str, Any] | None] = [None] * int(n_slots)
    rows: list[dict[str, Any]] = []

    for i in range(0, n - 2):
        exited_this_bar = False
        for k in range(n_slots):
            s = slots[k]
            if s is None:
                continue
            comp = components[s["comp"]]
            pos, entry_price, notional = s["pos"], s["entry_price"], s["notional"]
            move = ((arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0
                    else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price)
            s["mfe"], s["mae"] = max(s["mfe"], move), min(s["mae"], move)

            reason = ""
            if s["tp"] > 0.0 and move >= s["tp"]:
                reason = "take_profit"
            elif s["sl"] > 0.0 and move <= -abs(s["sl"]):
                reason = "stop_loss"
            if not reason:
                hold = max(i - s["entry_i"], 0)
                giveback = (s["mfe"] - move) / max(abs(s["mfe"]), 1e-8) if s["mfe"] > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(s["mfe"]), float(s["mae"]),
                                float(np.clip(giveback, 0.0, 10.0)), float(s["tp"] - move),
                                float(move + abs(s["sl"])), float(notional), float(s["leverage"]),
                                float(notional * s["leverage"]), float(s["tp"]), float(s["sl"])],
                    device=device)
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if not reason:
                continue

            exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
            raw_exit = ((exit_px - entry_price) / entry_price if pos > 0
                        else (entry_price - exit_px) / entry_price)
            factor = (1.0 - fee_eff * notional) * (1.0 + raw_exit * notional - fee_eff * notional)
            rows.append({
                "entry_signal_i": s["entry_signal_i"], "entry_i": s["entry_i"], "exit_i": int(i),
                "entry_timestamp": str(frame["timestamp"].iloc[s["entry_signal_i"]]),
                "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                "side": int(pos), "source_component": s["comp"], "reason": reason,
                "win": int(factor > 1.0), "trade_return": float(factor - 1.0),
                "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(s["mfe"]),
                "mae_price_move": float(s["mae"]), "notional": float(notional),
                "margin_fraction": float(s["margin"]), "leverage": float(s["leverage"]), "slot": int(k),
                "occupied_at_entry": int(s["occupied_at_entry"]),
            })
            slots[k] = None
            exited_this_bar = True

        if exited_this_bar:
            continue
        free = next((k for k in range(n_slots) if slots[k] is None), None)
        if free is None:
            continue
        occupied_count = sum(1 for s in slots if s is not None)

        for name in router.PRIORITY:
            if name not in components:
                continue
            p = pre[name]
            side = int(p["side"][i])
            if side == 0 or not bool(p["active"][i]):
                continue
            row_margin = float(p["margin"][i])
            if row_margin <= 0.0:
                continue
            if occupied_count >= 1:
                direction = "long" if side > 0 else "short"
                if float(mfe_pred[direction][i]) < float(mfe_cutoff[direction]):
                    continue
            row_margin = row_margin / float(n_slots)
            scale = router.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(float(p["leverage"][i]) * scale, router.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, router.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            slots[free] = {
                "comp": name, "pos": side, "entry_price": float(entry_px),
                "entry_i": min(i + 1, n - 1), "entry_signal_i": i,
                "margin": row_margin, "leverage": row_leverage, "notional": row_notional,
                "tp": float(p["tp"][i]), "sl": float(p["sl"][i]), "mfe": 0.0, "mae": 0.0,
                "occupied_at_entry": occupied_count,
            }
            break

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- MFE machinery
def train_seed_mfe(seed: int, panel_train: pd.DataFrame, feature_cols: list[str],
                    train_labels: pd.DataFrame) -> dict[str, Any]:
    models, diag = mfe_src.train_mfe_models(panel_train, feature_cols, train_labels, seed=seed)
    pred_long_train = np.clip(models["long"].predict(panel_train[feature_cols]), 0.0, None)
    pred_short_train = np.clip(models["short"].predict(panel_train[feature_cols]), 0.0, None)
    cutoff = {"long": float(np.quantile(pred_long_train, MFE_CUTOFF_QUANTILE)),
              "short": float(np.quantile(pred_short_train, MFE_CUTOFF_QUANTILE))}
    return {"models": models, "diag": diag, "cutoff": cutoff}


def score_mfe(models: dict[str, Any], feature_cols: list[str], panel: pd.DataFrame) -> dict[str, np.ndarray]:
    x = panel[feature_cols]
    return {"long": np.clip(models["long"].predict(x), 0.0, None),
            "short": np.clip(models["short"].predict(x), 0.0, None)}


def _ffill_regime3_gap(frame: pd.DataFrame, split_label: str) -> pd.DataFrame:
    """Harness-only data-hygiene patch, NOT a mechanism change -- copied from the identical, already-
    documented fix in research_eth_omega461_live_sltp_wide_calibration_oos_confirm_20260813.py
    (itself citing the same pattern in research_eth_omega461_tpsl_floor_portfolio_check_20260728.py's
    _truncated_pred_csv comment). The live WIDE24_2026 overlay CSV currently has a genuine, contiguous
    gap in the 6 regime3_current_sensitive_wide24_* columns (regenerated by other concurrent work in
    this shared repo since this script's own components were last frozen). hard._route_id() raises on
    any non-finite value in its route-probability columns with zero tolerance, so this forward-fills
    (causal, strictly using already-elapsed data only) all 6 regime3_current_sensitive_wide24_*
    columns before the frame is used for anything else. Verified below that no non-finite values
    remain after the fill; raises rather than silently proceeding otherwise."""
    cols = [c for c in frame.columns if c.startswith("regime3_current_sensitive_wide24_")]
    n_before = int(frame[cols].isna().any(axis=1).sum())
    if n_before == 0:
        return frame
    out = frame.copy()
    out[cols] = out[cols].ffill()
    n_after = int(out[cols].isna().any(axis=1).sum())
    print(f"  [{split_label}] regime3 overlay gap patch: {n_before} rows had NaN in {cols}, forward-filled, {n_after} remaining", flush=True)
    if n_after:
        raise RuntimeError(f"{split_label}: {n_after} rows still non-finite in regime3 columns after ffill (gap at series start?) -- refusing to silently proceed")
    return out


def compounded_pct(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((np.cumprod(1.0 + returns)[-1] - 1.0) * 100.0)


def permutation_incremental_compounded(base: np.ndarray, incr: np.ndarray, rng) -> dict:
    """Same reassignment methodology as base_multi.permutation_incremental, but on the COMPOUNDED
    statistic (this contract's G1 gate statistic) rather than the mean (08-08's original gate
    statistic, which its own report flagged as disagreeing with the compounded number)."""
    if len(incr) == 0 or len(base) == 0:
        return {"percentile": None, "note": "no incremental trades"}
    pool = np.concatenate([base, incr])
    obs = compounded_pct(incr)
    k = len(incr)
    null = np.array([compounded_pct(rng.permutation(pool)[:k]) for _ in range(R_PERM)])
    return {"observed_incremental_compounded_pct": round(obs, 4),
            "null_compounded_median_pct": round(float(np.median(null)), 4),
            "percentile": round(float((null < obs).mean()), 4), "R": R_PERM}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["g0", "val", "all"], default="val",
                     help="g0 = regression check only; val = full G0/G0b/G0c/G1/G2 (default, does NOT "
                          "touch OOS); all = additionally runs the pre-registered G3 (OOS single look) "
                          "and G4 (effect size, only if G3 passes). Spending the OOS read via --stage "
                          "all is a deliberate, explicit, one-time-only decision -- not the default.")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    device = retest.DEVICE
    rng = np.random.default_rng(PERM_RNG_SEED)
    fee, slip = omega._load_fee_slip()
    out: dict[str, Any] = {"contract": str(CONTRACT.relative_to(ROOT)), "n_slots": N_SLOTS, "seeds": SEEDS,
                            "mfe_cutoff_quantile": MFE_CUTOFF_QUANTILE,
                            "outcome_ceiling": contract["pre_registered_gates"]["G5_outcome_ceiling"]["rule"]}

    # ---- precondition: base_cols identical across bundles (contract's stated precondition) -------
    bundle_h48 = torch.load(retest.COMPONENTS["h48qual"]["bundle"], map_location="cpu", weights_only=False)
    bundle_zig = torch.load(retest.COMPONENTS["zig075"]["bundle"], map_location="cpu", weights_only=False)
    base_cols = list(bundle_h48["base_cols"])
    if list(bundle_zig["base_cols"]) != base_cols:
        raise RuntimeError("h48qual/zig075 base_cols differ -- cannot share one MFE model")
    print(json.dumps({"base_cols_n": len(base_cols), "base_cols_identical_across_bundles": True}), flush=True)

    # ---- G0: N=1 regression against the published incumbent, OOS extended window ------------------
    print("=== G0 regression: OOS N=1 must reproduce the incumbent (08-08's own function, unmodified)", flush=True)
    oos_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_frame = _ffill_regime3_gap(oos_frame, "oos")
    oos_preds = {n: base_multi.OOS_PRED_DIR / n / f"oos_predictions_{c['q_tag']}.csv" for n, c in retest.COMPONENTS.items()}
    # oos_leds1/oos_frame_al are kept (not discarded) so --stage all can reuse them for G3 without a
    # second, potentially-divergent alignment pass -- purely a "keep the reference" change, computes
    # nothing differently; G0's own pass/fail check below is untouched.
    oos_res1, oos_leds1, oos_frame_al = base_multi.run_window(oos_frame, oos_preds, device, "oos", [1])
    m1 = oos_res1[1]["no_gate"]
    g0 = {"expected": base_multi.G0_EXPECTED, "got": m1,
          "pass": bool(abs(m1["pnl"] - base_multi.G0_EXPECTED["pnl"]) <= G0_TOL_PP
                       and abs(m1["mdd"] - base_multi.G0_EXPECTED["mdd"]) <= G0_TOL_PP
                       and m1["trades"] == base_multi.G0_EXPECTED["trades"])}
    out["G0_regression"] = g0
    print(json.dumps({"G0": g0}, indent=2), flush=True)
    if not g0["pass"]:
        out["verdict"] = "HALT -- G0 failed; no number in this contract's arm may be reported"
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(out["verdict"])
        return 1
    if args.stage == "g0":
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    # ---- VAL: baseline N=1 (via run_window, unmodified) + independently-rebuilt aligned components
    print("\n=== VAL 2025Q4: baseline N=1 + component reconstruction", flush=True)
    val_pred_paths = {k: Path(v) for k, v in gval.VAL_PRED.items()}
    val_res1, val_leds1, val_frame_aligned = base_multi.run_window(gval.load_val_frame(), val_pred_paths, device, "val", [1])
    out["val_n1_baseline"] = val_res1[1]
    print(json.dumps({"val_n1_baseline": val_res1[1]["no_gate"]}, indent=2), flush=True)

    comps_val = {
        name: router.prepare_component(val_frame_aligned, base_multi.OUT_DIR / f"_aligned_val_{name}.csv",
                                        retest.COMPONENTS[name], device)
        for name in retest.COMPONENTS
    }

    # ---- G0b: independently-rebuilt VAL components, replayed at n=1 via the UNMODIFIED function,
    # must exactly match run_window's own n=1 result (proves the rebuild is faithful before it is
    # trusted for every downstream MFE-gated arm) -------------------------------------------------
    led1_check = base_multi.multislot_replay(val_frame_aligned, comps_val, n_slots=1, fee=fee, slip=slip,
                                              cost_mult=retest.COST_MULT, device=device)
    m1_check = base_multi.metrics(led1_check["trade_return"].to_numpy(float))
    g0b = {"run_window_n1": val_res1[1]["no_gate"], "rebuilt_comps_n1": m1_check,
           "pass": bool(abs(m1_check["pnl"] - val_res1[1]["no_gate"]["pnl"]) < 1e-6
                        and abs(m1_check["mdd"] - val_res1[1]["no_gate"]["mdd"]) < 1e-6
                        and m1_check["trades"] == val_res1[1]["no_gate"]["trades"])}
    out["G0b_val_comps_reconstruction_selfcheck"] = g0b
    print(json.dumps({"G0b": g0b}, indent=2), flush=True)
    if not g0b["pass"]:
        out["verdict"] = "HALT -- G0b failed; independently-rebuilt VAL components do not match run_window's own n=1 result"
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(out["verdict"])
        return 1

    # ---- G0c: gate-disabled structural equivalence at n_slots=3 ------------------------------------
    n_rows_val = len(val_frame_aligned)
    zero_pred = {"long": np.zeros(n_rows_val), "short": np.zeros(n_rows_val)}
    disabled_cutoff = {"long": -1.0e18, "short": -1.0e18}
    led3_gated_disabled = multislot_replay_mfe_gated(val_frame_aligned, comps_val, n_slots=N_SLOTS,
                                                       mfe_pred=zero_pred, mfe_cutoff=disabled_cutoff,
                                                       fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    led3_original = base_multi.multislot_replay(val_frame_aligned, comps_val, n_slots=N_SLOTS, fee=fee, slip=slip,
                                                  cost_mult=retest.COST_MULT, device=device)
    m3_gd = base_multi.metrics(led3_gated_disabled["trade_return"].to_numpy(float))
    m3_orig = base_multi.metrics(led3_original["trade_return"].to_numpy(float))
    g0c = {"gate_disabled_n3": m3_gd, "original_n3_equal_budget_ungated_08_08_arm": m3_orig,
           "pass": bool(abs(m3_gd["pnl"] - m3_orig["pnl"]) <= G0_TOL_PP
                        and abs(m3_gd["mdd"] - m3_orig["mdd"]) <= G0_TOL_PP
                        and m3_gd["trades"] == m3_orig["trades"])}
    out["G0c_gate_disabled_structural_equivalence"] = g0c
    print(json.dumps({"G0c": g0c}, indent=2), flush=True)
    if not g0c["pass"]:
        out["verdict"] = "HALT -- G0c failed; multislot_replay_mfe_gated is not a faithful superset of the original when the gate is a no-op"
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(out["verdict"])
        return 1
    print(f"reference only (unmodified 08-08 N=3, equal-budget, UNGATED -- this is the arm that already "
          f"failed G2 on 2026-08-08, NOT this contract's arm): {json.dumps(m3_orig)}", flush=True)

    # ---- OOS component reconstruction + self-check, ONLY when spending the one-time G3/G4 OOS read -
    comps_oos = None
    prior_val_result = None
    if args.stage == "all":
        print("\n=== --stage all: preparing OOS components for the one-time G3/G4 read", flush=True)
        comps_oos = {
            name: router.prepare_component(oos_frame_al, base_multi.OUT_DIR / f"_aligned_oos_{name}.csv",
                                            retest.COMPONENTS[name], device)
            for name in retest.COMPONENTS
        }
        led1_oos_check = base_multi.multislot_replay(oos_frame_al, comps_oos, n_slots=1, fee=fee, slip=slip,
                                                       cost_mult=retest.COST_MULT, device=device)
        m1_oos_check = base_multi.metrics(led1_oos_check["trade_return"].to_numpy(float))
        g0b_oos = {"run_window_n1": oos_res1[1]["no_gate"], "rebuilt_comps_n1": m1_oos_check,
                   "pass": bool(abs(m1_oos_check["pnl"] - oos_res1[1]["no_gate"]["pnl"]) < 1e-6
                                and abs(m1_oos_check["mdd"] - oos_res1[1]["no_gate"]["mdd"]) < 1e-6
                                and m1_oos_check["trades"] == oos_res1[1]["no_gate"]["trades"])}
        out["G0b_oos_comps_reconstruction_selfcheck"] = g0b_oos
        print(json.dumps({"G0b_oos": g0b_oos}, indent=2), flush=True)
        if not g0b_oos["pass"]:
            out["verdict"] = "HALT -- G0b_oos failed; independently-rebuilt OOS components do not match run_window's own n=1 result; refusing to spend the one-time OOS read"
            (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
            print(out["verdict"])
            return 1

        # Pre-flight safety check before spending the one-time OOS read: this run must reproduce the
        # ALREADY-PUBLISHED, coordinator-reviewed VAL result (tmp/.../result_val_only_20260813.json)
        # to a tight tolerance. This script's VAL code path (G0/G0b/G0c/G1/G2 above) is UNCHANGED by
        # this --stage all extension -- this assertion is insurance against any accidental drift, not
        # an expected source of new information.
        prior_path = OUT_DIR / "result_val_only_20260813.json"
        if not prior_path.exists():
            raise RuntimeError(f"{prior_path} not found -- cannot pre-flight-check against the published VAL result before spending the one-time OOS read")
        prior_val_result = json.loads(prior_path.read_text())
        print(f"stage=all preflight: loaded prior VAL-only result from {prior_path} for a reproduction self-check", flush=True)

    # ---- TRAIN frame + labels for the MFE regressor -----------------------------------------------
    print("\n=== building TRAIN frame + panel for MFE regressor", flush=True)
    train_frame = base_sweep.load_frame(TRAIN_START, TRAIN_END, base_csv=base_sweep.BASE_2025, wide24_csv=base_sweep.WIDE24_2025)
    missing_train = [c for c in base_cols if c not in train_frame.columns]
    if missing_train:
        raise RuntimeError(f"base_cols missing from TRAIN frame: {missing_train}")
    print(f"train_frame rows={len(train_frame)} range=[{train_frame['timestamp'].min()}, {train_frame['timestamp'].max()}]", flush=True)

    train_labels = mfe_src._load_tb_labels("train")
    val_labels = mfe_src._load_tb_labels("validation")

    panel_train, feature_cols = mfe_src.base102_panel(base_cols, train_frame)
    panel_val, feature_cols_val = mfe_src.base102_panel(base_cols, val_frame_aligned)
    if feature_cols != feature_cols_val:
        raise RuntimeError("feature_cols differ between TRAIN and VAL panels -- base102_panel is not frame-content-invariant as assumed")
    print(f"MFE feature panel: base102, n_features={len(feature_cols)}", flush=True)

    panel_oos = None
    if args.stage == "all":
        panel_oos, feature_cols_oos = mfe_src.base102_panel(base_cols, oos_frame_al)
        if feature_cols != feature_cols_oos:
            raise RuntimeError("feature_cols differ between TRAIN and OOS panels -- base102_panel is not frame-content-invariant as assumed")

    # ---- per-seed: train MFE, score VAL, replay MFE-gated N=3, evaluate G1/G2 ---------------------
    per_seed: dict[str, Any] = {}
    g1_pass_count = g2_pass_count = 0
    g3_pass_count = 0
    g3_pnl_only_count = g3_mdd_only_count = g3_quarters_only_count = 0
    oos_incr_pooled: list[np.ndarray] = []
    oos_led3_pooled: list[pd.DataFrame] = []
    v1 = val_res1[1]["no_gate"]
    o1 = oos_res1[1]["no_gate"] if args.stage == "all" else None
    for seed in SEEDS:
        print(f"\n=== seed={seed}", flush=True)
        fit = train_seed_mfe(seed, panel_train, feature_cols, train_labels)
        print(f"  train_diag: {json.dumps(fit['diag'])}", flush=True)
        val_diag = mfe_src.val_sanity_gate(fit["models"], panel_val, feature_cols, val_labels)
        print(f"  val_sanity_gate: {json.dumps(val_diag)}", flush=True)
        print(f"  cutoff (q={MFE_CUTOFF_QUANTILE}): {json.dumps(fit['cutoff'])}", flush=True)

        mfe_pred_val = score_mfe(fit["models"], feature_cols, panel_val)
        led3 = multislot_replay_mfe_gated(val_frame_aligned, comps_val, n_slots=N_SLOTS, mfe_pred=mfe_pred_val,
                                           mfe_cutoff=fit["cutoff"], fee=fee, slip=slip, cost_mult=retest.COST_MULT,
                                           device=device)
        led3.to_csv(OUT_DIR / f"ledger_val_n3_mfegated_seed{seed}.csv", index=False)
        m3 = base_multi.metrics(led3["trade_return"].to_numpy(float))
        print(f"  n3_mfe_gated VAL: {json.dumps(m3)}", flush=True)

        base_r, incr_r, incr_led = base_multi.incremental_split(val_leds1[1], led3)
        g1_compounded = compounded_pct(incr_r)
        g1_perm = permutation_incremental_compounded(base_r, incr_r, rng)
        g1_pass = bool(len(incr_r) > 0 and g1_compounded > 0.0)
        g1_pass_count += int(g1_pass)

        g2_pnl_ok = bool(m3["pnl"] >= v1["pnl"])
        g2_mdd_ok = bool(m3["mdd"] >= v1["mdd"] - G2_MDD_SLACK_PP)
        g2_pass = bool(g2_pnl_ok and g2_mdd_ok)
        g2_pass_count += int(g2_pass)

        # ---- pre-flight self-check (--stage all only): this seed's freshly-recomputed VAL numbers
        # must reproduce the ALREADY-PUBLISHED, coordinator-reviewed VAL result exactly, before this
        # run is trusted to spend the one-time OOS read for this seed -----------------------------
        seed_g3: dict[str, Any] | None = None
        if args.stage == "all":
            prior_seed = prior_val_result["per_seed"][str(seed)]
            mismatches = []
            for gate_key, fresh_val in (("pnl", m3["pnl"]), ("mdd", m3["mdd"]), ("trades", m3["trades"])):
                prior_val_num = prior_seed["n3_mfe_gated"][gate_key]
                if abs(float(fresh_val) - float(prior_val_num)) > 1.0e-6 * max(1.0, abs(float(prior_val_num))):
                    mismatches.append(f"seed={seed} n3_mfe_gated.{gate_key}: fresh={fresh_val} prior_published={prior_val_num}")
            if mismatches:
                raise RuntimeError("VAL self-check FAILED for seed " + str(seed) + " -- refusing to spend the "
                                    "one-time OOS read until this is understood: " + "; ".join(mismatches))
            print(f"  [seed={seed}] VAL self-check vs published result: OK (exact match)", flush=True)

            mfe_pred_oos = score_mfe(fit["models"], feature_cols, panel_oos)
            led3_oos = multislot_replay_mfe_gated(oos_frame_al, comps_oos, n_slots=N_SLOTS, mfe_pred=mfe_pred_oos,
                                                   mfe_cutoff=fit["cutoff"], fee=fee, slip=slip, cost_mult=retest.COST_MULT,
                                                   device=device)
            led3_oos.to_csv(OUT_DIR / f"ledger_oos_n3_mfegated_seed{seed}.csv", index=False)
            led3_oos["entry_ts"] = pd.to_datetime(led3_oos["entry_timestamp"])
            m3_oos = base_multi.metrics(led3_oos["trade_return"].to_numpy(float))
            q3_oos = base_multi.quarterly(led3_oos)
            print(f"  [seed={seed}] n3_mfe_gated OOS: {json.dumps(m3_oos)} quarters={json.dumps(q3_oos)}", flush=True)

            g3_pnl_ok = bool(m3_oos["pnl"] >= o1["pnl"] + G3_PNL_MARGIN_PP)
            g3_mdd_ok = bool(m3_oos["mdd"] >= G3_MDD_FLOOR)
            q1_ok = "2026Q1" in q3_oos and q3_oos["2026Q1"]["pnl"] > 0
            q2_ok = "2026Q2" in q3_oos and q3_oos["2026Q2"]["pnl"] > 0
            g3_quarters_ok = bool(q1_ok and q2_ok)
            g3_seed_pass = bool(g3_pnl_ok and g3_mdd_ok and g3_quarters_ok)
            g3_pass_count += int(g3_seed_pass)
            g3_pnl_only_count += int(g3_pnl_ok)
            g3_mdd_only_count += int(g3_mdd_ok)
            g3_quarters_only_count += int(g3_quarters_ok)

            oos_base_r, oos_incr_r, oos_incr_led = base_multi.incremental_split(oos_leds1[1], led3_oos)
            oos_incr_pooled.append(oos_incr_r)
            oos_led3_pooled.append(led3_oos)
            if len(oos_incr_led):
                oos_incr_led.to_csv(OUT_DIR / f"oos_incremental_trades_seed{seed}.csv", index=False)

            seed_g3 = {
                "rule": f"OOS PnL >= N1 OOS PnL + {G3_PNL_MARGIN_PP}pp AND MDD >= {G3_MDD_FLOOR}% AND BOTH 2026Q1/Q2 positive",
                "n1_oos_baseline": o1, "n3_mfe_gated_oos": m3_oos, "quarters": q3_oos,
                "pnl_ok": g3_pnl_ok, "mdd_ok": g3_mdd_ok, "quarters_ok": g3_quarters_ok, "pass": g3_seed_pass,
                "oos_incremental_trades": int(len(oos_incr_r)),
                "oos_incremental_compounded_pct": round(compounded_pct(oos_incr_r), 4),
            }
            if len(oos_incr_led):
                seed_g3["oos_incremental_side_mix"] = oos_incr_led["side"].value_counts().to_dict()
                seed_g3["oos_incremental_component_mix"] = oos_incr_led["source_component"].value_counts().to_dict()

        seed_result = {
            "mfe_train_diag": fit["diag"], "mfe_val_sanity_gate": val_diag, "mfe_cutoff": fit["cutoff"],
            "n1_baseline": v1, "n3_mfe_gated": m3,
            "G1": {"rule": "compounded contribution of incremental trades > 0 (NOT mean)",
                   "incremental_trades": int(len(incr_r)),
                   "incremental_compounded_pct": round(g1_compounded, 4),
                   "incremental_mean_pct": round(float(incr_r.mean() * 100), 4) if len(incr_r) else None,
                   "kept_mean_pct": round(float(base_r.mean() * 100), 4) if len(base_r) else None,
                   "permutation": g1_perm, "pass": g1_pass},
            "G2": {"rule": f"n3_mfe_gated PnL >= n1 PnL AND MDD >= n1 MDD - {G2_MDD_SLACK_PP}pp",
                   "pnl_ok": g2_pnl_ok, "mdd_ok": g2_mdd_ok, "pass": g2_pass},
        }
        if len(incr_led):
            incr_led.to_csv(OUT_DIR / f"val_incremental_trades_seed{seed}.csv", index=False)
            seed_result["G1_incremental_side_mix"] = incr_led["side"].value_counts().to_dict()
            seed_result["G1_incremental_component_mix"] = incr_led["source_component"].value_counts().to_dict()
            seed_result["G1_incremental_occupied_at_entry_mix"] = incr_led["occupied_at_entry"].value_counts().to_dict()
        if seed_g3 is not None:
            seed_result["G3"] = seed_g3
        per_seed[str(seed)] = seed_result
        summary = {"n3_mfe_gated": m3, "G1_pass": g1_pass, "G2_pass": g2_pass, "G1_compounded_pct": round(g1_compounded, 4)}
        if seed_g3 is not None:
            summary["n3_mfe_gated_oos"] = seed_g3["n3_mfe_gated_oos"]
            summary["G3_pass"] = seed_g3["pass"]
        print(json.dumps({f"seed_{seed}_summary": summary}, indent=2), flush=True)

    out["per_seed"] = per_seed
    out["G1_aggregate"] = {"rule": f">= {AGGREGATE_MIN_PASS}/{len(SEEDS)} seeds individually pass G1",
                            "pass_count": g1_pass_count, "of": len(SEEDS), "min_required": AGGREGATE_MIN_PASS,
                            "pass": bool(g1_pass_count >= AGGREGATE_MIN_PASS)}
    out["G2_aggregate"] = {"rule": f">= {AGGREGATE_MIN_PASS}/{len(SEEDS)} seeds individually pass G2",
                            "pass_count": g2_pass_count, "of": len(SEEDS), "min_required": AGGREGATE_MIN_PASS,
                            "pass": bool(g2_pass_count >= AGGREGATE_MIN_PASS)}
    print(json.dumps({"G1_aggregate": out["G1_aggregate"], "G2_aggregate": out["G2_aggregate"]}, indent=2), flush=True)

    if not (out["G1_aggregate"]["pass"] and out["G2_aggregate"]["pass"]):
        out["verdict"] = ("CLOSE -- " +
                           ("G1 aggregate premise failed (fewer than "
                            f"{AGGREGATE_MIN_PASS}/{len(SEEDS)} seeds show positive COMPOUNDED incremental "
                            "contribution on VAL)" if not out["G1_aggregate"]["pass"] else
                            f"G2 aggregate VAL falsification failed (fewer than {AGGREGATE_MIN_PASS}/{len(SEEDS)} "
                            "seeds beat N=1 on PnL+MDD)") +
                           "; no OOS read was spent, per the contract and per explicit instruction to stop after G2")
    else:
        out["verdict"] = (f"G1/G2 aggregate PASS on VAL (>= {AGGREGATE_MIN_PASS}/{len(SEEDS)} seeds each) -- per "
                           "explicit instruction this script stops here and does not spend the OOS read. "
                           "Orchestrator decides separately whether/when to run G3.")
    out["oos_read_spent"] = False

    # ---- G3 (OOS single look) + G4 (effect size), ONLY under --stage all, ONLY reached because
    # G1/G2 already passed above (this is the coordinator's explicit, one-time decision to spend the
    # pre-registered OOS read -- no retuning after seeing this result is permitted by the contract) --
    if args.stage == "all" and out["G1_aggregate"]["pass"] and out["G2_aggregate"]["pass"]:
        out["oos_read_spent"] = True
        out["oos_window"] = [OOS_START, OOS_END]
        out["oos_window_convention_note"] = ("this contract inherits the 08-08 multislot line's own extended "
                                              "OOS window (Jan-Jun, 2 quarters), NOT the shorter 2026-01-01..03-31 "
                                              "window tonight's sibling exit_sweep-based scripts use -- do not "
                                              "compare these OOS numbers directly against those scripts' OOS numbers")
        g3_aggregate_pass = bool(g3_pass_count >= AGGREGATE_MIN_PASS)
        out["G3_aggregate"] = {
            "rule": f">= {AGGREGATE_MIN_PASS}/{len(SEEDS)} seeds individually satisfy ALL THREE of "
                    f"(OOS PnL >= N1+{G3_PNL_MARGIN_PP}pp) AND (OOS MDD >= {G3_MDD_FLOOR}%) AND (both 2026Q1/Q2 positive)",
            "interpretation_note": ("the contract's G3 text lists three separate '>=4/5' clauses; this script "
                                     "reads them as a per-seed conjunction (all three must hold for the SAME seed) "
                                     "then aggregates that conjunction at >=4/5 seeds, for consistency with how "
                                     "this same contract's G1/G2 were implemented -- the three sub-gates' own "
                                     "individual pass-counts are also reported below so the alternative "
                                     "(disjoint-membership) reading can be checked directly"),
            "pass_count": g3_pass_count, "of": len(SEEDS), "min_required": AGGREGATE_MIN_PASS, "pass": g3_aggregate_pass,
            "sub_gate_pass_counts": {"pnl_ok": g3_pnl_only_count, "mdd_ok": g3_mdd_only_count, "quarters_ok": g3_quarters_only_count},
        }
        print(json.dumps({"G3_aggregate": out["G3_aggregate"]}, indent=2), flush=True)

        if g3_aggregate_pass:
            print("\n=== G3 aggregate PASSED -- proceeding to G4 (effect size), per the contract", flush=True)
            pooled_incr = np.concatenate(oos_incr_pooled) if oos_incr_pooled else np.array([])
            base_full_oos = oos_leds1[1]["trade_return"].to_numpy(float)
            g4_perm = permutation_incremental_compounded(base_full_oos, pooled_incr, rng)

            led3_oos_pooled = pd.concat(oos_led3_pooled, ignore_index=True) if oos_led3_pooled else pd.DataFrame(columns=oos_leds1[1].columns)
            g4_cmp = base_multi.compare(led3_oos_pooled, oos_leds1[1], rng) if len(led3_oos_pooled) else None
            gate_cfg = contract["selection"]["effect_size_gate"]
            g4_t_ok = bool(g4_cmp is not None and g4_cmp["per_trade_welch_t"] is not None
                           and abs(g4_cmp["per_trade_welch_t"]) >= gate_cfg["min_abs_t"])
            g4_perm_ok = bool(g4_perm.get("percentile") is not None and g4_perm["percentile"] >= gate_cfg["min_permutation_percentile"])
            g4_pass = bool(g4_t_ok and g4_perm_ok)
            out["G4_effect_size"] = {
                "rule": gate_cfg, "pooling_note": "pooled across all 5 seeds' OOS ledgers (contract explicitly allows pooled or per-seed)",
                "compare_pooled_n3_vs_n1": g4_cmp, "incremental_permutation_pooled": g4_perm,
                "t_ok": g4_t_ok, "perm_ok": g4_perm_ok, "pass": g4_pass,
            }
            print(json.dumps({"G4_effect_size": out["G4_effect_size"]}, indent=2), flush=True)

            g4_verdict = ("PASS -> stand up a parallel ETH multi-slot MFE-gated SHADOW (never a live swap; "
                          "one-way-mode blocker still applies)" if g4_pass else
                          "FAIL (effect size too weak/inconsistent) -> CLOSE, no re-tuning")
            out["verdict"] = (out["verdict"] + f" || G3 OOS single look: PASS ({g3_pass_count}/{len(SEEDS)} seeds) "
                               f"-> G4 effect size: {g4_verdict}")
        else:
            out["verdict"] = (out["verdict"] + f" || G3 OOS single look: FAIL ({g3_pass_count}/{len(SEEDS)} seeds, "
                               f"need >={AGGREGATE_MIN_PASS}) -- CLOSE the axis. No re-tuning of slot count, sizing, "
                               "cutoff quantile, or feature panel on OOS, per the contract.")
        print("\n" + out["verdict"], flush=True)

    out["fresh_forward"] = contract["fresh_forward"]
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + out["verdict"])
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
