#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 #18 SOFT VARIANT (2026-08-15), the explicitly-proposed-but-untried next
step from docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md's own "다음 단계"
section: instead of an unconditional forced exit when the evidence-veto signal fires on an open
h48qual SHORT, RELAX the exit_head probability threshold (0.95 -> a lower value) for the next N bars,
so the position still exits through the NORMAL exit_head check (prob >= threshold) -- just an easier
one -- rather than being clipped outright. Hypothesis (stated in the precedent doc): this should avoid
the hard version's per-firing damage (VAL with_gate PnL 77.31%->47.39%, -29.92pp, REJECTED) because it
never forces an exit a winning short wouldn't have taken anyway.

=== Precedent this is INCREMENTAL to (read in full before this script; not re-derived here) ===
docs/experiments/eth_omega461_evidence_veto_exit_overlay_20260814.md (Odyssey2 #18, "Candidate C",
the HARD variant): while h48qual holds an open SHORT, orthogonal_combo (adaptive Williams-%R/Slow-%K
both in rolling-864 bottom decile AND same-bar net-aggressive-sell-volume z<=-2) forced an immediate
exit. REJECTED on VAL: only 6 firings out of 26,209 VAL bars, but each firing was so costly that VAL
with_gate PnL fell 77.31%->47.39% (-29.92pp); Q1/Q2 also degraded (-18.83pp/-13.28pp with_gate). The
targeted Q3 turnover-acceleration failure mode DID recover (no_gate 145%, with_gate 130% of the
liveATR-vs-original gap), but that upside never offset the VAL/Q1/Q2 damage. That script's own
docstring/report record the exact signal formula (orthogonal_combo, no new threshold), G0a/G0b
self-checks, and the multiwindow gate module's exact numbers reproduced here.

=== What THIS script tests (the soft variant) ===
Reuses the HARD variant script's signal construction and component preparation FUNCTIONS UNMODIFIED
(imported, not reimplemented): research_eth_omega461_evidence_veto_exit_overlay_20260814.build_signal
/ .prepare_evidence_veto_components / ._prep_liveatr_only / .G0_REQUIRED. The ONLY new logic is a
renamed copy of that script's own greedy_replay_evidence_veto_exit loop
(greedy_replay_evidence_veto_soft_exit below) where the "--- evidence veto ---" block no longer force-
exits on firing; instead it starts (or restarts) an N-bar countdown during which the SAME bar's
exit_head check uses `relax_threshold` instead of `comp["exit_threshold"]` (0.95, static, from
trading_bot_modules/omega4_6_1_live.py EXIT_THRESHOLD). Re-firing while already inside a countdown
resets it to N (does not stack). The countdown decrements once per held bar and is cleared whenever
the position closes (via TP/SL/relaxed-exit_head) or a fresh position opens. Every hard barrier
(take_profit/stop_loss) still takes priority, exactly as in the hard variant and in unmodified
greedy.greedy_replay.

=== Pre-registered grid (fixed BEFORE any cell was run; see docs/experiments/
eth_omega461_evidence_veto_exit_overlay_soft_variant_20260815.md for the committed grid) ===
RELAX_N_BARS_GRID = [3, 6, 12] (15/30/60 minutes on 5m bars -- short/medium/long relative to
h48qual's own typical hold, chosen as round multiples, not tuned against any outcome).
RELAX_THRESHOLD_GRID = [0.80, 0.85, 0.90] -- all three are pre-existing grid points from
research_eth_omega461_exit_sweep_20260721.py's own exit_grid = [0.999,0.99,0.97,0.95,0.90,0.85,0.80,
0.70] (this script introduces no new threshold values, only reuses already-swept ones). 3x3 = 9 cells
total, run on VAL ONLY first. Tie-break (pre-registered, in case >1 cell clears the VAL gate): pick the
cell with the highest with_gate PnL; ties broken by fewest veto-triggered relaxation windows opened
(prefer the smaller/simpler intervention). Only that single winner (if any) is carried into the
single-touch OOS-Q1+OOS-Q2 pass -- no re-running OOS per cell, no OOS peeking before VAL is judged. If
zero of the 9 cells clear VAL, this script stops at VAL and OOS is never opened.

VAL gate criterion (same row-level logic eth_omega461_multiwindow_confirmation_gate_20260814.
summarize_multiwindow already applies to its oos_confirm rows, applied here to the "val" window
instead): with_gate PnL >= baseline (asymmetric_tabm_liveatr) with_gate PnL, AND with_gate MDD within
`mdd_slack_pp` of baseline MDD. Checked at both mdd_slack_pp=0.0 (strict) and mdd_slack_pp=3.0
(relaxed, matching every other Odyssey2 candidate's dual-criterion reporting this session) -- a cell
"clears VAL" if it passes under EITHER criterion (same "non_regression_ok_either_criterion" convention
the hard variant's own report used).

fresh_forward_bar_by_bar=true (renamed copy of the hard variant's own already-verified single causal
forward pass; only the reason-determination block changes, i increasing, only bar i and closed history
used at bar i). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module --
research_eth_omega461_evidence_veto_exit_overlay_20260814.py,
eth_omega461_multiwindow_confirmation_gate_20260814.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
replay_omega4_6_1_greedy_router_20260706.py, research_eth_omega461_exit_sweep_20260721.py,
research_eth_omega461_live_sltp_mfe_width_20260813.py,
train_eval_omega4_2_risk_sidecar_20260622.py, analyze_eth_creative_reversal_evidence_signals_
20260814.py, backtest_eth_slowk_williamsr_persistence_confluence_20260814.py are all imported and read
only. No retraining, no GPU (DEVICE=cpu, matching every script in this lineage).
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as rs  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_evidence_veto_exit_overlay_20260814 as veto_hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_evidence_veto_exit_overlay_soft_variant_20260815"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05

# =====================================================================================================
# Pre-registered grid -- fixed here, before any cell is run. See module docstring for the anchor of
# each value (RELAX_THRESHOLD_GRID values reuse research_eth_omega461_exit_sweep_20260721.py's own
# exit_grid points, no new threshold introduced).
# =====================================================================================================
RELAX_N_BARS_GRID: list[int] = [3, 6, 12]
RELAX_THRESHOLD_GRID: list[float] = [0.80, 0.85, 0.90]
MDD_SLACK_STRICT_PP = 0.0
MDD_SLACK_RELAXED_PP = 3.0

G0_REQUIRED = veto_hard.G0_REQUIRED


def log(msg: str) -> None:
    print(f"[evidence_veto_exit_soft] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP, check_trades: bool = True) -> bool:
    ok = bool(abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp)
    if check_trades and "trades" in expected:
        ok = ok and int(actual["trades"]) == int(expected["trades"])
    return ok


# =====================================================================================================
# Renamed copy of veto_hard.greedy_replay_evidence_veto_exit (itself a renamed copy of
# greedy.greedy_replay). Every line is unchanged except the "--- evidence veto (soft) ---" block: on
# firing, instead of an unconditional forced exit, an N-bar relaxed-exit_threshold countdown is
# (re)started; the exit_head check that already existed in greedy_replay runs every bar regardless,
# just against a relaxed threshold while the countdown is active.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_evidence_veto_soft_exit(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    relax_n_bars: int,
    relax_threshold: float,
    veto_component: str = "h48qual",
) -> tuple[dict, pd.DataFrame]:
    """While `veto_component` (h48qual) holds an OPEN SHORT position (pos<0), if
    components[veto_component]['evidence_veto_mask'][i] is True on bar i, an N-bar (relax_n_bars)
    relaxed-exit_threshold (relax_threshold, < comp['exit_threshold']) countdown is (re)started --
    re-firing while already inside a countdown resets it to N, does not stack. While the countdown is
    active (including the firing bar itself), the SAME exit_head probability check that runs every
    bar in unmodified greedy_replay uses relax_threshold instead of comp['exit_threshold']; no
    unconditional forced exit exists in this variant. Hard barriers (take_profit/stop_loss) always
    take priority, unaffected by the countdown. LONG positions on veto_component, and any other
    active component (zig075), are unaffected -- the countdown only ever starts/applies when
    active_comp == veto_component AND pos < 0 AND a 'evidence_veto_mask' key is present.
    """
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    active_comp = None
    entry_price = entry_equity = 1.0
    entry_i = entry_signal_i = 0
    notional = leverage_v = margin_fraction = 0.0
    take_profit = stop_loss = 0.0
    mfe = mae = 0.0
    armed = False
    trailing_enabled = False
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    veto_short_hold_bars = 0
    veto_fire_bars = 0
    veto_relax_active_bars = 0
    veto_relax_remaining = 0

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == veto_component and pos < 0:
                veto_short_hold_bars += 1
            move = (arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0 else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price
            unreal = move * notional
            mfe, mae = max(mfe, move), min(mae, move)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

            reason = ""
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            if not reason and trailing_enabled:
                pass  # trailing not used by this candidate (mirrors greedy_replay default off)
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                # --- evidence veto (soft): only new logic vs greedy_replay/greedy_replay_evidence_veto_exit ---
                is_veto_eligible = active_comp == veto_component and pos < 0
                mask = comp.get("evidence_veto_mask")
                if is_veto_eligible and mask is not None and bool(mask[i]):
                    veto_relax_remaining = int(relax_n_bars)
                    veto_fire_bars += 1
                relaxed_active = is_veto_eligible and veto_relax_remaining > 0
                if relaxed_active:
                    veto_relax_active_bars += 1
                active_threshold = float(relax_threshold) if relaxed_active else float(comp["exit_threshold"])
                prob = rs._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=pos_values, device=device,
                )
                if prob >= active_threshold:
                    reason = "exit_head_relaxed" if relaxed_active else "exit_head"
                elif relaxed_active:
                    veto_relax_remaining -= 1
                # --- end evidence veto (soft) block ---
            if reason:
                exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
                raw_exit = (exit_px - entry_price) / entry_price if pos > 0 else (entry_price - exit_px) / entry_price
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee_eff * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"entry_signal_i": entry_signal_i, "entry_i": entry_i, "exit_i": i,
                             "entry_timestamp": str(frame["timestamp"].iloc[entry_signal_i]),
                             "exit_timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos),
                             "source_component": active_comp, "reason": reason,
                             "win": int(cash > entry_equity), "trade_return": float(trade_return),
                             "notional": float(notional), "margin_fraction": float(margin_fraction),
                             "leverage": float(leverage_v)})
                pos, active_comp = 0, None
                veto_relax_remaining = 0
                continue
            continue

        # flat: try priority order
        for name in greedy.PRIORITY:
            if name not in components:
                continue
            comp = components[name]
            side = int(comp["dec"]["side"].iloc[i])
            if side == 0 or not bool(omega._active(comp["dec"]).iloc[i] if hasattr(omega._active(comp["dec"]), "iloc") else omega._active(comp["dec"])[i]):
                continue
            row_margin, row_leverage = float(comp["margin"][i]), float(comp["leverage"][i])
            if row_margin <= 0.0:
                continue
            scale = greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(row_leverage * scale, greedy.LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, greedy.NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            pos, active_comp = side, name
            entry_price, entry_equity = float(entry_px), cash
            entry_i, entry_signal_i = min(i + 1, n - 1), i
            margin_fraction, leverage_v, notional = row_margin, row_leverage, row_notional
            take_profit = float(comp["dec"]["take_profit"].iloc[i])
            stop_loss = float(comp["dec"]["stop_loss"].iloc[i])
            cash -= cash * fee_eff * notional
            mfe = mae = 0.0
            armed = False
            veto_relax_remaining = 0
            break

    diag = {
        "reason_counts": reasons,
        f"{veto_component}_short_hold_bars": veto_short_hold_bars,
        f"{veto_component}_veto_fire_bars": veto_fire_bars,
        f"{veto_component}_veto_relax_active_bars": veto_relax_active_bars,
    }
    return diag, pd.DataFrame(rows)


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def _row_pass(candidate_wg: dict[str, Any], baseline_wg: dict[str, Any], *, mdd_slack_pp: float) -> tuple[bool, bool, bool]:
    """Same per-window pass logic as gate.summarize_multiwindow's row computation, applied here to
    an arbitrary (candidate, baseline) with_gate pair instead of only oos_confirm rows."""
    pnl_pass = float(candidate_wg["pnl"]) >= float(baseline_wg["pnl"])
    mdd_pass = (float(candidate_wg["mdd"]) - float(baseline_wg["mdd"])) >= -abs(mdd_slack_pp)
    return pnl_pass, mdd_pass, bool(pnl_pass and mdd_pass)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 #18 SOFT VARIANT -- instead of an unconditional forced exit on evidence-veto "
            "firing, relax h48qual's exit_head threshold (0.95 -> relax_threshold) for relax_n_bars "
            "bars, so the position still exits via the normal exit_head check, just more easily. "
            "Pre-registered 3x3 grid, VAL-gated before any OOS touch."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "pre_registered_grid": {
            "relax_n_bars": RELAX_N_BARS_GRID,
            "relax_threshold": RELAX_THRESHOLD_GRID,
            "n_cells": len(RELAX_N_BARS_GRID) * len(RELAX_THRESHOLD_GRID),
            "val_gate_criterion": "with_gate PnL >= baseline(asymmetric_tabm_liveatr) with_gate PnL AND with_gate MDD within mdd_slack_pp of baseline MDD, at mdd_slack_pp in {0.0, 3.0} (cell clears if either passes)",
            "tie_break_if_multiple_clear_val": "highest with_gate PnL; ties broken by fewest veto_fire_bars",
            "oos_policy": "only the single pre-selected VAL winner (if any) is run on OOS-Q1+OOS-Q2 together, single touch",
        },
        "trigger_signal": "orthogonal_combo (bottom side only, unchanged from Odyssey2 #18 hard variant) -- (p_fast<=0.10)&(p_slow<=0.10)&(delta_z<=-2.0)",
        "trigger_scope": "h48qual SHORT positions only; LONG h48qual and zig075 (both sides) untouched",
    }

    # =================================================================================================
    # stage=load_windows
    # =================================================================================================
    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=G0a -- reproduce reference numbers via the already-validated gate module (unmodified import)
    # =================================================================================================
    log("=== stage=G0a_reference_via_gate_module ===")
    g0a: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = G0_REQUIRED[wname]
        ok_ng, ok_wg = _close(result["no_gate"], ref_ng), _close(result["with_gate"], ref_wg)
        g0a[wname] = {"no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg}}
        log(f"  {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
            f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}")
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0a_reference_via_gate_module"] = {"windows": g0a, "pass": g0a_pass}
    log(f"stage=G0a_result pass={g0a_pass}")

    # =================================================================================================
    # stage=G0b -- soft-replay-forced-inactive identity check: this script's own soft-replay function,
    # run on plain asymmetric_tabm_liveatr components (no evidence_veto_mask key at all -- so the
    # countdown can never start), must reproduce the SAME 4 reference numbers exactly as unmodified
    # greedy_replay. Proves the copy is faithful outside the intentionally-changed block, same
    # discipline as the hard variant's own G0b.
    # =================================================================================================
    log("=== stage=G0b_soft_replay_forced_inactive_identity ===")
    g0b: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        aligned_frame, components = veto_hard._prep_liveatr_only(wname, windows, OUT_DIR, device)
        diag, ledger = greedy_replay_evidence_veto_soft_exit(
            aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            relax_n_bars=6, relax_threshold=0.80,  # arbitrary grid values -- must be a no-op since mask key is absent
        )
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_REQUIRED[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg},
                      "veto_fire_bars_expected_zero": diag["h48qual_veto_fire_bars"],
                      "veto_relax_active_bars_expected_zero": diag["h48qual_veto_relax_active_bars"]}
        log(f"  {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg} "
            f"veto_fire_bars={diag['h48qual_veto_fire_bars']} veto_relax_active_bars={diag['h48qual_veto_relax_active_bars']}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] and g0b[w]["veto_fire_bars_expected_zero"] == 0 and g0b[w]["veto_relax_active_bars_expected_zero"] == 0 for w in ("val", "oos_q1"))
    report["g0b_soft_replay_forced_inactive_identity"] = {"windows": g0b, "pass": g0b_pass}
    log(f"stage=G0b_result pass={g0b_pass}")

    g0_pass = bool(g0a_pass and g0b_pass)
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (reference reproduction and/or soft-replay-forced-inactive identity check). Aborting before trusting any candidate number."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=signal_build -- reuse the hard variant's own build_signal() UNMODIFIED (same evidence-veto
    # definition, no new threshold, no calibration).
    # =================================================================================================
    log("=== stage=signal_build ===")
    score_by_base = veto_hard.build_signal()

    # =================================================================================================
    # stage=val_grid -- run all 9 pre-registered (relax_n_bars, relax_threshold) cells on VAL ONLY.
    # VAL must be judged before OOS is ever opened (the hard variant's own self-flagged process gap).
    # =================================================================================================
    log("=== stage=val_grid (9 pre-registered cells, VAL ONLY) ===")
    val_baseline = gate.run_portfolio_variant("val", windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
    baseline_ng, baseline_wg = val_baseline["no_gate"], val_baseline["with_gate"]

    val_grid: dict[str, Any] = {}
    for n_bars in RELAX_N_BARS_GRID:
        for thr in RELAX_THRESHOLD_GRID:
            cell_key = f"n{n_bars}_thr{thr:.2f}"
            aligned_frame, components, prep_diag = veto_hard.prepare_evidence_veto_components("val", windows, score_by_base, OUT_DIR, device)
            diag, ledger = greedy_replay_evidence_veto_soft_exit(
                aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
                relax_n_bars=n_bars, relax_threshold=thr,
            )
            ledger_path = OUT_DIR / f"portfolio_ledger_val_soft_{cell_key}.csv"
            ledger.to_csv(ledger_path, index=False)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            pnl_pass_s, mdd_pass_s, pass_s = _row_pass(with_gate, baseline_wg, mdd_slack_pp=MDD_SLACK_STRICT_PP)
            pnl_pass_r, mdd_pass_r, pass_r = _row_pass(with_gate, baseline_wg, mdd_slack_pp=MDD_SLACK_RELAXED_PP)
            clears_val = bool(pass_s or pass_r)
            val_grid[cell_key] = {
                "relax_n_bars": n_bars, "relax_threshold": thr,
                "no_gate": no_gate, "with_gate": with_gate,
                "veto_fire_bars": diag["h48qual_veto_fire_bars"], "veto_relax_active_bars": diag["h48qual_veto_relax_active_bars"],
                "pass_strict_mdd0pp": pass_s, "pass_relaxed_mdd3pp": pass_r, "clears_val": clears_val,
                "ledger_path": str(ledger_path),
            }
            log(f"  {cell_key}: no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
                f"fire_bars={diag['h48qual_veto_fire_bars']:3d} relax_active_bars={diag['h48qual_veto_relax_active_bars']:4d}  "
                f"clears_val={clears_val} (strict={pass_s} relaxed={pass_r})")

    report["val_baseline_asymmetric_tabm_liveatr"] = {"no_gate": baseline_ng, "with_gate": baseline_wg}
    report["val_grid"] = val_grid

    clearing_cells = {k: v for k, v in val_grid.items() if v["clears_val"]}
    log(f"stage=val_grid_result n_clearing={len(clearing_cells)}/{len(val_grid)}")

    if not clearing_cells:
        report["stage_reached"] = "val_grid"
        report["gate_pass"] = True
        report["final_verdict"] = "REJECTED_ALL_VAL_CELLS_FAIL"
        report["note"] = "All 9 pre-registered grid cells failed the VAL gate (with_gate PnL nonworse, either MDD criterion) vs asymmetric_tabm_liveatr baseline. OOS was never opened, per the pre-registered VAL-first policy."
        _write_report(report)
        log("stage=done final_verdict=REJECTED_ALL_VAL_CELLS_FAIL (OOS not opened)")
        return 0

    # Pre-registered tie-break: highest with_gate PnL, ties broken by fewest veto_fire_bars.
    winner_key = max(clearing_cells.keys(), key=lambda k: (clearing_cells[k]["with_gate"]["pnl"], -clearing_cells[k]["veto_fire_bars"]))
    winner = clearing_cells[winner_key]
    log(f"stage=val_winner_selected key={winner_key} relax_n_bars={winner['relax_n_bars']} relax_threshold={winner['relax_threshold']} with_gate_pnl={winner['with_gate']['pnl']:.2f}%")
    report["val_winner"] = {"key": winner_key, "relax_n_bars": winner["relax_n_bars"], "relax_threshold": winner["relax_threshold"]}

    # =================================================================================================
    # stage=oos_single_touch -- open OOS-Q1+OOS-Q2 TOGETHER, single touch, ONLY for the pre-selected
    # VAL winner. Also compute 2025q1/q2/q3 context for the winner (never gates, shown for context).
    # =================================================================================================
    log(f"=== stage=oos_single_touch (winner={winner_key} only) ===")
    win_n, win_thr = winner["relax_n_bars"], winner["relax_threshold"]
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        baseline = gate.run_portfolio_variant(wname, windows, gate.COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        aligned_frame, components, prep_diag = veto_hard.prepare_evidence_veto_components(wname, windows, score_by_base, OUT_DIR, device)
        diag, ledger = greedy_replay_evidence_veto_soft_exit(
            aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            relax_n_bars=win_n, relax_threshold=win_thr,
        )
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_soft_winner_{winner_key}.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "asymmetric_tabm_liveatr": {"no_gate": baseline["no_gate"], "with_gate": baseline["with_gate"]},
            "soft_veto_winner": {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)},
            "veto_fire_bars": diag["h48qual_veto_fire_bars"], "veto_relax_active_bars": diag["h48qual_veto_relax_active_bars"],
        }
        log(f"  {wname:8s} liveatr    no_gate={baseline['no_gate']['pnl']:7.2f}%/{baseline['no_gate']['mdd']:7.2f}%/{baseline['no_gate']['trades']:3d}  with_gate={baseline['with_gate']['pnl']:7.2f}%/{baseline['with_gate']['mdd']:7.2f}%/{baseline['with_gate']['trades']:3d}")
        log(f"  {wname:8s} soft_veto  no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  fire_bars={diag['h48qual_veto_fire_bars']}")

    baseline_tuples = {w: (comparison[w]["asymmetric_tabm_liveatr"]["no_gate"], comparison[w]["asymmetric_tabm_liveatr"]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["soft_veto_winner"]["no_gate"], comparison[w]["soft_veto_winner"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=MDD_SLACK_STRICT_PP)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=MDD_SLACK_RELAXED_PP)
    oos_confirmed = bool(summary_strict["oos_confirm_all_pass_single_touch"] or summary_relaxed["oos_confirm_all_pass_single_touch"])
    log(f"  OOS-Q1+OOS-Q2 single-touch verdict: strict={summary_strict['final_verdict']} relaxed_mdd3pp={summary_relaxed['final_verdict']} -> oos_confirmed={oos_confirmed}")

    report["comparison_all_6_windows_winner_only"] = comparison
    report["oos_single_touch_summary"] = {"strict_mdd0pp": summary_strict, "relaxed_mdd3pp": summary_relaxed, "oos_confirmed": oos_confirmed}
    report["final_verdict"] = "CONFIRMED" if oos_confirmed else "REJECTED_OOS_SIGN_MISMATCH"
    report["stage_reached"] = "oos_single_touch"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done final_verdict={report['final_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
