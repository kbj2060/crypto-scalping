#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey3 #2: zig075 SHORT entry veto during detected sustained uptrends.

=== Scope change, stated explicitly (per docs/experiments/eth_omega461_zig075_sustained_uptrend_
guard_20260814.md's own closing note) ===
Odyssey(1)/Odyssey2 operated under a task instruction of "post-entry interventions ONLY", and the
zig075 sustained-uptrend diagnosis (Odyssey3 execution log #1) therefore concluded
`diagnosed_no_valid_design`: zig075 SHORT's exit_head never once appears as an exit reason in any
2025 quarter (0/53 trades), the principled exit_threshold range (0.80~0.99) is inert in Q3, and the
sub-0.80 range is post-hoc selection. The USER has now explicitly lifted the post-entry-only
constraint for this experiment (2026-08-14 session). This script is the first sanctioned
entry-side intervention of the Odyssey line.

=== Why THIS entry-side design and not the 29 that failed ===
Every failed entry-side attempt cataloged in the Odyssey1/Odyssey2 contracts tried to create or
re-select direction/quality SKILL from model heads (retraining, relabeling, regating, recalibrating
-- TabM/GBDT/AE/TCN/CNN x zigzag/trend-scanning/MFE labels, all OOS-defeated by always-short;
"direction_head has no direction skill" is contract-level settled fact). This design does NOT touch
any model head and does not attempt subset selection by model internals (which the diagnosis showed
cannot separate Q3 winners from losers: dir_p_short 0.718~0.825 and quality_for_action 0.751~0.825
for BOTH the 16 losers and 3 winners). It manages regime BETA with an external, causal,
already-locked detector: the Odyssey3-baseline sustained-uptrend detector (rolling 1-week fraction
of dual_momentum>0, threshold = p90 of the 2025-Q1+Q2-ONLY calibration sample = 0.802579...,
Q3/VAL/OOS never used to derive it -- research_eth_omega461_regime_aware_exit_head_uptrend_guard_
20260814.build_detector, reused verbatim, ZERO new free parameters).

Mechanism evidence (from the execution-log-#1 diagnosis walk + a detector overlay on those same
trades' SIGNAL bars): Q3 zig075 SHORT entries with detector ACTIVE at the signal bar account for
10/19 union trades and -0.4089 of the -0.5440 union loss (9 stop_loss, 1 take_profit); Q1 overlap
is 0/10 and Q2 overlap is 1/16 (a winner). The Q3 losses are entry-timing losses (median MFE only
41% of the SL distance; TP-share collapses 70%->44%->16% across Q1->Q2->Q3), so removing the
entries themselves is the only handle left -- and the naive per-trade accounting above ignores the
shared-slot dynamics (a vetoed zig075 entry frees the single slot for h48qual or a later signal),
which is exactly why this script runs a full fresh-forward portfolio replay instead of ledger
arithmetic.

=== The intervention ===
In the flat-state entry loop, iff the candidate entry is (component == "zig075" AND side == SHORT)
and the sustained-uptrend detector is ACTIVE at the signal bar, skip that entry. Nothing else
changes: zig075 LONG untouched, h48qual untouched (it keeps its Odyssey3 regime-aware exit guard),
all model heads / thresholds / TP/SL / sizing / priority / caps untouched, exit side untouched.
Baseline for ALL comparisons = the locked Odyssey3 baseline (asymmetric_tabm_liveatr + h48qual
regime-aware exit guard at p90).

=== Verification protocol (pre-registered before running) ===
- G0a: reproduce the Odyssey3 baseline on val+oos_q1 via the guard module's OWN unmodified
  function -- environment/data drift check.
- G0b: this script's renamed replay copy (veto machinery present, NO veto mask attached) must
  reproduce the contract's G0 reference numbers on ALL 6 windows -- proves the copy is faithful
  outside the intentionally-added veto block, and doubles as the baseline tuples for the verdict.
- Candidate: veto at the baseline detector threshold (p90) on all 6 windows, single execution.
- Verdict: gate.summarize_multiwindow (with_gate PnL AND MDD non-worse), strict + relaxed(3pp),
  VAL gate first, then OOS-Q1+OOS-Q2 single touch. 2025q1/q2/q3 stay context-tier (the Q3
  improvement is the experiment's raison d'etre but NEVER enters the verdict).
- Robustness (context only, pre-registered percentiles from the guard experiment): veto threshold
  at p75/p95 on the three 2025 quarters -- checks the Q3 effect is not a p90 artifact. The h48qual
  exit guard stays at p90 everywhere (it is the locked baseline, not part of this axis).

fresh_forward_bar_by_bar=true (single causal forward pass, i increasing; detector is a plain
backward-looking rolling mean; veto reads mask[i] at the signal bar only).
trade_ledgers_used_as_input=false (ledgers are write-only outputs; the diagnosis overlay above is
cited as motivation, not consumed as input). saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module. No retraining,
no GPU (DEVICE=cpu), conda env quant_ai.
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
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
VETO_COMPONENT = "zig075"

# Locked Odyssey3 baseline threshold, copied from the guard experiment's report.json
# ["detector"]["threshold_used"] for an exact-match assertion (the number is REUSED, never
# re-derived here -- build_detector recomputes it from the same Q1+Q2-only sample and we assert
# equality to prove nothing drifted).
EXPECTED_PRIMARY_THRESHOLD = 0.8025793650793651

# G0 reference -- the Odyssey3 baseline (regime_aware_guard) numbers for all 6 windows, copied
# verbatim from docs/model_contracts/odyssey3_eth_regime_guard_baseline_contract_20260814.md G0
# table (itself from the guard experiment's report.json). (no_gate, with_gate) per window.
G0_ODYSSEY3 = {
    "2025q1": ({"pnl": 97.70, "mdd": -20.62, "trades": 28}, {"pnl": 44.98, "mdd": -20.62, "trades": 20}),
    "2025q2": ({"pnl": 106.45, "mdd": -13.23, "trades": 31}, {"pnl": 31.49, "mdd": -15.85, "trades": 19}),
    "2025q3": ({"pnl": -37.43, "mdd": -51.25, "trades": 27}, {"pnl": -15.86, "mdd": -44.37, "trades": 21}),
    "val": ({"pnl": 46.59, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
    "oos_q2": ({"pnl": -9.55, "mdd": -20.76, "trades": 13}, {"pnl": -12.69, "mdd": -20.76, "trades": 10}),
}


def log(msg: str) -> None:
    print(f"[zig075_short_entry_veto] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _attach_veto_mask(components: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Return a components dict whose zig075 entry carries the veto mask. Shallow-copies only the
    zig075 dict (never mutates the input) so the SAME prepared components can serve both the
    no-veto baseline run and the veto candidate run."""
    out = dict(components)
    zig = dict(out[VETO_COMPONENT])
    zig["short_entry_veto_mask"] = mask
    out[VETO_COMPONENT] = zig
    return out


# =====================================================================================================
# Renamed copy of research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.
# greedy_replay_regime_aware_exit_guard (itself a validated copy of greedy.greedy_replay). Neither
# source module is edited -- only imported and read. Every line is unchanged except the block marked
# "--- zig075 SHORT entry veto: only new logic vs greedy_replay_regime_aware_exit_guard ---" and the
# veto diagnostic counters threaded through.
# =====================================================================================================


@torch.no_grad()
def greedy_replay_entry_veto(
    frame: pd.DataFrame,
    components: dict,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    guard_component: str = "h48qual",
    trailing_activate_frac: float | None = None,
    trailing_trail_frac: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Identical to guard.greedy_replay_regime_aware_exit_guard (h48qual regime-aware exit guard
    fully preserved -- Odyssey3 baseline), plus ONE new rule in the flat-state entry loop: if a
    component carries a 'short_entry_veto_mask' and its candidate entry this bar is SHORT while
    mask[i] is True at the signal bar, that entry is skipped. No mask attached -> byte-identical to
    the guard replay's own behaviour."""
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
    trailing_enabled = trailing_activate_frac is not None and trailing_trail_frac is not None
    rows: list[dict] = []
    reasons: dict[str, int] = {}
    guard_hold_bars = 0
    guard_active_bars = 0
    guard_decision_differs_bars = 0
    veto_bars = 0  # diagnostic: flat-state bars where a SHORT entry signal was suppressed by the
    # veto. Consecutive signal bars of one persistent signal each count once per bar -- the ledger
    # diff (computed in main()) is the per-trade view.
    veto_events: list[dict] = []

    for i in range(0, n - 2):
        if pos != 0:
            comp = components[active_comp]
            if active_comp == guard_component:
                guard_hold_bars += 1
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
                if (not armed) and take_profit > 0.0 and mfe >= float(trailing_activate_frac) * take_profit:
                    armed = True
                if armed and mfe > 0.0 and move <= mfe - float(trailing_trail_frac) * abs(stop_loss):
                    reason = "trailing_stop"
            if not reason:
                hold = max(i - entry_i, 0)
                giveback = (mfe - move) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
                giveback_clipped = float(np.clip(giveback, 0.0, 10.0))
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                pos_values = [float(pos), float(hold), float(move), float(mfe), float(mae),
                              giveback_clipped, float(take_profit - move),
                              float(move + abs(stop_loss)), float(notional), float(leverage_v),
                              float(notional * leverage_v), float(take_profit), float(stop_loss)]
                use_guard = False
                mask = comp.get("sustained_uptrend_mask")
                if active_comp == guard_component and mask is not None and bool(mask[i]):
                    use_guard = True
                if use_guard:
                    guard_active_bars += 1
                    prob = rs._predict_exit_prob_one(
                        comp["guard_base_np"], comp["guard_exit_runtime"], comp["guard_pos_idx"], row_i=int(i),
                        expert=expert, pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp.get("guard_exit_threshold", comp["exit_threshold"]))
                    default_prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    if (prob >= active_threshold) != (default_prob >= float(comp["exit_threshold"])):
                        guard_decision_differs_bars += 1
                else:
                    prob = rs._predict_exit_prob_one(
                        comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                        pos_values=pos_values, device=device,
                    )
                    active_threshold = float(comp["exit_threshold"])
                if prob >= active_threshold:
                    reason = "exit_head"
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
            # --- zig075 SHORT entry veto: only new logic vs greedy_replay_regime_aware_exit_guard ---
            veto_mask = comp.get("short_entry_veto_mask")
            if veto_mask is not None and side < 0 and bool(veto_mask[i]):
                veto_bars += 1
                veto_events.append({"i": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "component": name})
                continue
            # --- end zig075 SHORT entry veto block ---
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
            break

    diag = {
        "reason_counts": reasons,
        f"{guard_component}_hold_bars": guard_hold_bars,
        f"{guard_component}_guard_active_bars": guard_active_bars,
        f"{guard_component}_guard_decision_differs_bars": guard_decision_differs_bars,
        "veto_bars": veto_bars,
        "veto_events": veto_events,
    }
    return diag, pd.DataFrame(rows)


def _ledger_diff(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Per-trade view of what the veto changed: trades keyed by (entry_signal_i, side,
    source_component). Removed = in baseline only; added = in candidate only (slot freed by a
    vetoed entry being taken by another/later signal)."""
    def _key(df: pd.DataFrame) -> set[tuple]:
        if df.empty:
            return set()
        return set(zip(df["entry_signal_i"].astype(int), df["side"].astype(int), df["source_component"]))

    bk, ck = _key(baseline), _key(candidate)
    removed_keys, added_keys = bk - ck, ck - bk

    def _rows(df: pd.DataFrame, keys: set[tuple]) -> list[dict]:
        if df.empty or not keys:
            return []
        sel = df[[k in keys for k in zip(df["entry_signal_i"].astype(int), df["side"].astype(int), df["source_component"])]]
        cols = ["entry_signal_i", "entry_timestamp", "exit_timestamp", "side", "source_component", "reason", "trade_return"]
        return sel[cols].to_dict("records")

    removed = _rows(baseline, removed_keys)
    added = _rows(candidate, added_keys)
    return {
        "n_removed": len(removed), "n_added": len(added),
        "removed_trades": removed, "added_trades": added,
        "removed_return_sum": float(sum(r["trade_return"] for r in removed)),
        "added_return_sum": float(sum(r["trade_return"] for r in added)),
    }


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Odyssey3 #2 -- zig075 SHORT entry veto during detected sustained uptrends. Reuses the "
            "locked Odyssey3-baseline detector (rolling 1-week fraction of dual_momentum>0, "
            "threshold=p90 of 2025-Q1+Q2-only calibration, Q3 never used) verbatim as an entry-side "
            "veto: iff component==zig075 AND side==SHORT AND detector active at the signal bar, "
            "skip that entry. Zero new free parameters. First sanctioned entry-side intervention "
            "after the user lifted the post-entry-only constraint (2026-08-14)."
        ),
        "scope_change_note": "post-entry-only constraint lifted by user instruction this session; entry-side now sanctioned for this experiment.",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    # =================================================================================================
    # stage=load_windows
    # =================================================================================================
    log("=== stage=load_windows ===")
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=detector_build -- reuse the guard module's OWN builder; assert the recomputed primary
    # threshold equals the locked Odyssey3 value exactly (the number is inherited, not re-chosen).
    # =================================================================================================
    log("=== stage=detector_build (reused from guard module) ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()
    if abs(threshold - EXPECTED_PRIMARY_THRESHOLD) > 1e-12:
        report["stage_reached"] = "detector_build"
        report["gate_pass"] = False
        report["note"] = f"recomputed p90 threshold {threshold!r} != locked Odyssey3 value {EXPECTED_PRIMARY_THRESHOLD!r} -- data drift, aborting."
        _write_report(report)
        log("stage=ABORT threshold drift")
        return 1
    log(f"  thresholds (Q1+Q2-only): {robustness_thresholds}  primary(p90)={threshold:.10f} == locked Odyssey3 value")
    report["detector"] = {
        "reused_from": "research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector (unmodified import)",
        "calibration_window": [guard.CALIBRATION_START, guard.CALIBRATION_END],
        "calibration_excludes_2025q3": True,
        "thresholds_q1q2_only": robustness_thresholds,
        "threshold_used": threshold,
        "threshold_matches_locked_odyssey3_value": True,
        "new_free_parameters": 0,
    }

    # =================================================================================================
    # stage=G0a -- environment/data sanity: the guard module's OWN replay (unmodified import) must
    # reproduce the Odyssey3 baseline on val+oos_q1.
    # =================================================================================================
    log("=== stage=G0a_odyssey3_baseline_via_guard_module ===")
    g0a: dict[str, Any] = {}
    prepared: dict[str, tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]] = {}
    for wname in ("val", "oos_q1"):
        aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        prepared[wname] = (aligned_frame, components, prep_diag)
        diag, ledger = guard.greedy_replay_regime_aware_exit_guard(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_ODYSSEY3[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        g0a[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg}}
        log(f"  {wname}: no_gate={no_gate['pnl']:.2f}%/{no_gate['mdd']:.2f}%/{no_gate['trades']} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:.2f}%/{with_gate['mdd']:.2f}%/{with_gate['trades']} match={ok_wg}")
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0a_odyssey3_baseline_via_guard_module"] = {"windows": g0a, "pass": g0a_pass}
    log(f"stage=G0a_result pass={g0a_pass}")

    # =================================================================================================
    # stage=G0b -- copy-fidelity check AND baseline production: THIS script's replay copy with NO
    # veto mask attached must reproduce the Odyssey3 baseline on ALL 6 windows. The resulting
    # ledgers/metrics ARE the baseline tuples for the verdict (same code path as the candidate,
    # differing only by the veto mask -- the cleanest possible ceteris paribus).
    # =================================================================================================
    log("=== stage=G0b_copy_fidelity_all6_no_veto ===")
    g0b: dict[str, Any] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}
    for wname in gate.ALL_WINDOWS:
        if wname not in prepared:
            prepared[wname] = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        aligned_frame, components, prep_diag = prepared[wname]
        diag, ledger = greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_odyssey3_baseline.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        ref_ng, ref_wg = G0_ODYSSEY3[wname]
        ok_ng, ok_wg = _close(no_gate, ref_ng), _close(with_gate, ref_wg)
        veto_zero = int(diag["veto_bars"]) == 0
        g0b[wname] = {"no_gate": {"actual": no_gate, "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": with_gate, "reference": ref_wg, "match": ok_wg},
                      "veto_bars_expected_zero": int(diag["veto_bars"])}
        baseline_runs[wname] = {"no_gate": no_gate, "with_gate": with_gate, "ledger": ledger, "ledger_path": str(ledger_path), "diag": {k: v for k, v in diag.items() if k != "veto_events"}}
        log(f"  {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d} match={ok_ng}  "
            f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d} match={ok_wg} veto_bars={diag['veto_bars']}")
    g0b_pass = all(g0b[w]["no_gate"]["match"] and g0b[w]["with_gate"]["match"] and g0b[w]["veto_bars_expected_zero"] == 0 for w in gate.ALL_WINDOWS)
    report["g0b_copy_fidelity_all6_no_veto"] = {"windows": g0b, "pass": g0b_pass}
    log(f"stage=G0b_result pass={g0b_pass}")

    g0_pass = bool(g0a_pass and g0b_pass)
    report["gate_pass_g0"] = g0_pass
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 failed (Odyssey3 baseline reproduction and/or copy fidelity). Aborting before trusting any candidate number."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=entry_overlap_static -- cheap pre-replay diagnostic: per window, how many bars carry an
    # active zig075 SHORT signal, and how many of those coincide with the detector. Informative
    # only; the replay (slot dynamics) decides what actually changes.
    # =================================================================================================
    log("=== stage=entry_overlap_static ===")
    overlap: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = prepared[wname]
        zig = components[VETO_COMPONENT]
        side = pd.to_numeric(zig["dec"]["side"], errors="raise").to_numpy()
        active = omega._active(zig["dec"])
        active = active.to_numpy() if hasattr(active, "to_numpy") else np.asarray(active)
        mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
        short_sig = (side < 0) & active.astype(bool)
        overlap[wname] = {
            "zig075_short_signal_bars": int(short_sig.sum()),
            "short_signal_bars_detector_active": int((short_sig & mask).sum()),
            "detector_active_frac": float(mask.mean()),
        }
        log(f"  {wname:8s} short_signal_bars={overlap[wname]['zig075_short_signal_bars']:5d}  "
            f"overlap_with_detector={overlap[wname]['short_signal_bars_detector_active']:5d}  "
            f"detector_active={overlap[wname]['detector_active_frac'] * 100:5.1f}%")
    report["entry_overlap_static"] = overlap

    # =================================================================================================
    # stage=candidate_run -- veto at the locked p90 threshold, all 6 windows, single execution.
    # =================================================================================================
    log("=== stage=candidate_run (veto @ p90, all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = prepared[wname]
        mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
        veto_components = _attach_veto_mask(components, mask)
        diag, ledger = greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_zig075_short_entry_veto_p90.csv"
        ledger.to_csv(ledger_path, index=False)
        no_gate = portfolio._ledger_metrics(ledger)
        with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
        diff = _ledger_diff(baseline_runs[wname]["ledger"], ledger)
        comparison[wname] = {
            "tier": gate.WINDOW_DEFS[wname]["tier"],
            "odyssey3_baseline": {"no_gate": baseline_runs[wname]["no_gate"], "with_gate": baseline_runs[wname]["with_gate"], "ledger_path": baseline_runs[wname]["ledger_path"]},
            "entry_veto_p90": {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path)},
            "detector_diag": prep_diag,
            "veto_replay_diag": {k: v for k, v in diag.items() if k != "veto_events"},
            "veto_events": diag["veto_events"],
            "ledger_diff": diff,
        }
        b_ng, b_wg = baseline_runs[wname]["no_gate"], baseline_runs[wname]["with_gate"]
        log(f"  {wname:8s} baseline  no_gate={b_ng['pnl']:7.2f}%/{b_ng['mdd']:7.2f}%/{b_ng['trades']:3d}  with_gate={b_wg['pnl']:7.2f}%/{b_wg['mdd']:7.2f}%/{b_wg['trades']:3d}")
        log(f"  {wname:8s} veto_p90  no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  "
            f"veto_bars={diag['veto_bars']}  removed={diff['n_removed']}(ret {diff['removed_return_sum']:+.4f})  added={diff['n_added']}(ret {diff['added_return_sum']:+.4f})")
    report["comparison"] = comparison

    # =================================================================================================
    # stage=robustness -- veto threshold at p75/p95 (pre-registered percentiles of the SAME
    # Q1+Q2-only sample), 2025 quarters only, context tier. h48qual exit guard stays at p90.
    # =================================================================================================
    log("=== stage=robustness (veto @ p75/p95, 2025 quarters, context only) ===")
    robustness: dict[str, Any] = {}
    for plabel in ("p75", "p95"):
        thr = robustness_thresholds[plabel]
        robustness[plabel] = {"threshold": thr}
        for wname in gate.CONTEXT_WINDOWS:
            aligned_frame, components, prep_diag = prepared[wname]
            mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, thr)
            veto_components = _attach_veto_mask(components, mask)
            diag, ledger = greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            no_gate = portfolio._ledger_metrics(ledger)
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            diff = _ledger_diff(baseline_runs[wname]["ledger"], ledger)
            robustness[plabel][wname] = {"no_gate": no_gate, "with_gate": with_gate,
                                         "veto_bars": diag["veto_bars"], "n_removed": diff["n_removed"], "n_added": diff["n_added"],
                                         "removed_return_sum": diff["removed_return_sum"], "added_return_sum": diff["added_return_sum"]}
            log(f"  {plabel} {wname:8s} no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}  removed={diff['n_removed']} added={diff['n_added']}")
    report["robustness_context_only"] = robustness

    # =================================================================================================
    # stage=summarize -- VAL gate + OOS-Q1/OOS-Q2 single touch vs the Odyssey3 baseline, strict and
    # relaxed(3pp). 2025 quarters context-only.
    # =================================================================================================
    log("=== stage=summarize ===")
    baseline_tuples = {w: (baseline_runs[w]["no_gate"], baseline_runs[w]["with_gate"]) for w in gate.ALL_WINDOWS}
    candidate_tuples = {w: (comparison[w]["entry_veto_p90"]["no_gate"], comparison[w]["entry_veto_p90"]["with_gate"]) for w in gate.ALL_WINDOWS}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    val_gate_pass_strict = bool(summary_strict["rows"]["val"]["with_gate_pass"])
    val_gate_pass_relaxed = bool(summary_relaxed["rows"]["val"]["with_gate_pass"])
    log(f"  VAL gate: strict={val_gate_pass_strict} relaxed={val_gate_pass_relaxed}")
    log(f"  OOS single touch: strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']}")

    q3_base = comparison["2025q3"]["odyssey3_baseline"]
    q3_veto = comparison["2025q3"]["entry_veto_p90"]
    q3_effect = {
        "no_gate": {"baseline_pnl": q3_base["no_gate"]["pnl"], "veto_pnl": q3_veto["no_gate"]["pnl"],
                    "baseline_mdd": q3_base["no_gate"]["mdd"], "veto_mdd": q3_veto["no_gate"]["mdd"]},
        "with_gate": {"baseline_pnl": q3_base["with_gate"]["pnl"], "veto_pnl": q3_veto["with_gate"]["pnl"],
                      "baseline_mdd": q3_base["with_gate"]["mdd"], "veto_mdd": q3_veto["with_gate"]["mdd"]},
    }
    log(f"  2025q3 effect: {q3_effect}")

    report["summary"] = {
        "val_gate_pass_strict": val_gate_pass_strict,
        "val_gate_pass_relaxed": val_gate_pass_relaxed,
        "multiwindow_strict_mdd0pp": summary_strict,
        "multiwindow_relaxed_mdd3pp": summary_relaxed,
        "q3_effect_context_tier_never_gated": q3_effect,
    }
    report["stage_reached"] = "summarize"
    report["gate_pass"] = True
    _write_report(report)
    log(f"stage=done strict={summary_strict['final_verdict']} relaxed={summary_relaxed['final_verdict']} val_strict={val_gate_pass_strict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
