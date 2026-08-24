#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey4 #2: h48qual SHORT entry veto during detected sustained uptrends.

=== Why this script exists ===
Odyssey4's zig075-SHORT entry veto (docs/experiments/eth_omega461_zig075_short_entry_veto_
sustained_uptrend_20260814.md) CONFIRMED strict on VAL + single-touch OOS-Q1/OOS-Q2. Both Odyssey3
and Odyssey4's own "다음 단계" notes flagged the SAME mechanism applied to h48qual SHORT (the OTHER
live component, priority-first in the shared single-slot arbitration -- replay_omega4_6_1_greedy_
router_20260706.PRIORITY = ("h48qual", "zig075")) as explicitly UNTRIED. This script is that
extension: literally the SAME detector formula, SAME calibration window, SAME locked threshold
(0.8025793650793651) -- the ONLY thing that changes vs the zig075 version is the veto target
(component="h48qual", side=SHORT instead of component="zig075", side=SHORT).

Zero new free parameters: the detector (rolling 1-week fraction of dual_momentum>0, threshold=p90
of the 2025-Q1+Q2-only calibration sample, Q3/VAL/OOS never touched for calibration) is imported
verbatim from research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector --
not recomputed with any new choice. The veto-application machinery (greedy_replay_entry_veto,
_ledger_diff) is imported verbatim from research_eth_omega461_zig075_short_entry_veto_sustained_
uptrend_20260814 -- that function is already generic (it reads `comp.get("short_entry_veto_mask")`
off WHICHEVER component dict carries it, in priority order, not hardcoded to zig075), so this script
adds no new replay logic at all, only a different mask attachment target.

Note h48qual already carries a DIFFERENT mask key from the Odyssey3 baseline -- "sustained_uptrend_
mask" (drives its regime-aware EXIT guard: original pre-liveATR-relabel exit head while an uptrend
is detected during an open h48qual position). This script additionally attaches "short_entry_veto_
mask" to h48qual for a completely different codepath (the flat-state ENTRY loop). The two masks are
the SAME boolean series (same detector, same threshold) but gate unrelated decisions and coexist as
distinct dict keys without conflict -- h48qual's exit guard stays fully intact in every run here.

=== The intervention ===
In the flat-state entry loop, iff the candidate entry is (component == "h48qual" AND side == SHORT)
and the sustained-uptrend detector is ACTIVE at the signal bar, skip that entry. Nothing else
changes: h48qual LONG untouched, h48qual's own regime-aware exit guard untouched, zig075 (LONG and
SHORT) untouched, all model heads / thresholds / TP/SL / sizing / priority / caps untouched.
Baseline for ALL comparisons = the locked Odyssey3 baseline (asymmetric_tabm_liveatr + h48qual
regime-aware exit guard at p90) -- identical comparator to the zig075 version.

=== Verification protocol (pre-registered before running, mirrors the zig075 script exactly) ===
- G0a: reproduce the Odyssey3 baseline on val+oos_q1 via the guard module's OWN unmodified function.
- G0b: greedy_replay_entry_veto (imported, no mask attached) must reproduce the Odyssey3 baseline on
  ALL 6 windows -- proves the imported function is faithful with zero veto attached, and doubles as
  the baseline tuples for the verdict.
- Candidate: veto at the baseline detector threshold (p90) on all 6 windows, single execution, mask
  attached to h48qual instead of zig075.
- Verdict: gate.summarize_multiwindow (with_gate PnL AND MDD non-worse), strict + relaxed(3pp), VAL
  gate first, then OOS-Q1+OOS-Q2 single touch. 2025q1/q2/q3 stay context-tier.
- Robustness (context only, pre-registered percentiles from the guard experiment): veto threshold at
  p75/p95 on the three 2025 quarters. h48qual's own exit guard stays at p90 everywhere.

fresh_forward_bar_by_bar=true (single causal forward pass, i increasing; detector is a plain
backward-looking rolling mean; veto reads mask[i] at the signal bar only -- unchanged from the
imported greedy_replay_entry_veto).
trade_ledgers_used_as_input=false (ledgers are write-only outputs). saved_parent_exit_timestamps_
used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module (including the
zig075 entry-veto script and the regime-guard module, both imported and read only). No retraining,
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
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as zveto  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_h48qual_short_entry_veto_sustained_uptrend_20260815"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
VETO_COMPONENT = "h48qual"

# Locked Odyssey3 baseline threshold, copied from the guard experiment's report.json
# ["detector"]["threshold_used"] -- reused, never re-derived here.
EXPECTED_PRIMARY_THRESHOLD = 0.8025793650793651

# G0 reference -- the Odyssey3 baseline (regime_aware_guard) numbers for all 6 windows, identical to
# the zig075 entry-veto script's own G0_ODYSSEY3 (same comparator, no zig075-specific content).
G0_ODYSSEY3 = zveto.G0_ODYSSEY3


def log(msg: str) -> None:
    print(f"[h48qual_short_entry_veto] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _attach_veto_mask(components: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Return a components dict whose h48qual entry carries the veto mask. Shallow-copies only the
    h48qual dict (never mutates the input) -- h48qual already carries its own 'sustained_uptrend_
    mask' key (exit guard) which is untouched; this adds the separate 'short_entry_veto_mask' key
    that greedy_replay_entry_veto's flat-state loop reads."""
    out = dict(components)
    h48 = dict(out[VETO_COMPONENT])
    h48["short_entry_veto_mask"] = mask
    out[VETO_COMPONENT] = h48
    return out


def _ledger_diff(baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    return zveto._ledger_diff(baseline, candidate)


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
            "Odyssey4 #2 -- h48qual SHORT entry veto during detected sustained uptrends. Reuses the "
            "locked Odyssey3-baseline detector (rolling 1-week fraction of dual_momentum>0, "
            "threshold=p90 of 2025-Q1+Q2-only calibration, Q3 never used) verbatim as an entry-side "
            "veto: iff component==h48qual AND side==SHORT AND detector active at the signal bar, "
            "skip that entry. Zero new free parameters. Direct extension of the CONFIRMED zig075 "
            "entry-veto experiment (2026-08-14) to the other live component, per that experiment's "
            "own flagged-untried next step."
        ),
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
    # stage=detector_build -- reuse the guard module's OWN builder; assert exact-match to the locked
    # Odyssey3 value.
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
    # stage=G0a -- environment/data sanity via the guard module's own unmodified replay.
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
    # stage=G0b -- copy-fidelity check AND baseline production: the IMPORTED greedy_replay_entry_veto
    # (from the zig075 script, unmodified) with NO veto mask attached must reproduce the Odyssey3
    # baseline on ALL 6 windows.
    # =================================================================================================
    log("=== stage=G0b_copy_fidelity_all6_no_veto ===")
    g0b: dict[str, Any] = {}
    baseline_runs: dict[str, dict[str, Any]] = {}
    for wname in gate.ALL_WINDOWS:
        if wname not in prepared:
            prepared[wname] = guard.prepare_regime_aware_components(wname, windows, score_by_base, threshold, OUT_DIR, device)
        aligned_frame, components, prep_diag = prepared[wname]
        diag, ledger = zveto.greedy_replay_entry_veto(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
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
    # active h48qual SHORT signal, and how many of those coincide with the detector.
    # =================================================================================================
    log("=== stage=entry_overlap_static ===")
    overlap: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = prepared[wname]
        h48 = components[VETO_COMPONENT]
        side = pd.to_numeric(h48["dec"]["side"], errors="raise").to_numpy()
        active = omega._active(h48["dec"])
        active = active.to_numpy() if hasattr(active, "to_numpy") else np.asarray(active)
        mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
        short_sig = (side < 0) & active.astype(bool)
        overlap[wname] = {
            "h48qual_short_signal_bars": int(short_sig.sum()),
            "short_signal_bars_detector_active": int((short_sig & mask).sum()),
            "detector_active_frac": float(mask.mean()),
        }
        log(f"  {wname:8s} short_signal_bars={overlap[wname]['h48qual_short_signal_bars']:5d}  "
            f"overlap_with_detector={overlap[wname]['short_signal_bars_detector_active']:5d}  "
            f"detector_active={overlap[wname]['detector_active_frac'] * 100:5.1f}%")
    report["entry_overlap_static"] = overlap

    # =================================================================================================
    # stage=candidate_run -- veto at the locked p90 threshold, all 6 windows, single execution, mask
    # attached to h48qual.
    # =================================================================================================
    log("=== stage=candidate_run (veto @ p90, all 6 windows) ===")
    comparison: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        aligned_frame, components, prep_diag = prepared[wname]
        mask, _ = guard._detector_mask_for_frame(aligned_frame, wname, score_by_base, threshold)
        veto_components = _attach_veto_mask(components, mask)
        diag, ledger = zveto.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_h48qual_short_entry_veto_p90.csv"
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
    # stage=robustness -- veto threshold at p75/p95 (pre-registered percentiles of the SAME Q1+Q2-only
    # sample), 2025 quarters only, context tier. h48qual's own exit guard stays at p90 everywhere.
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
            diag, ledger = zveto.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
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
