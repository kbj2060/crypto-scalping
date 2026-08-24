#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias regime-gated hybrid session (2026-08-17): pre-registered success/kill
re-evaluation of the SAME side-blind new exit signal
(tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/
new_exit_signal_bundle_sideblind.pkl, threshold=0.5), this time only governing h48qual's held-position
exit decision when the ALREADY-VALIDATED "sustained uptrend" detector
(research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.build_detector, rolling 1-week
dual_momentum>0 fraction, p90-of-2025Q1+Q2-only threshold, zero new free parameters) is INACTIVE. When
the detector is ACTIVE, h48qual's ORIGINAL (pre-liveATR-relabel) exit head, threshold=0.95, governs
instead -- byte-identical to the deployed guard's own ON branch
(research_eth_odyssey4_h48qual_exit_guard_ranging_misfire_test_20260817.py already showed this exact
detector+ON-branch pairing is causally INERT against the guard's EXISTING (liveATR) OFF-branch in all 6
judged/ranging windows -- this script asks whether swapping the OFF branch for Ilias's own side-blind
signal breaks that inertness, i.e. does the hybrid do anything the side-blind signal running unguarded
everywhere (arm_eval_report_sideblind.json, reused not rerun) does not already do, especially in the 3
windows where the ungated side-blind signal itself failed criterion 2 (VAL, OOS-Q2, ranging①
2025-05-12..07-07 -- see docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md §8).

=== Zero new free variables ===
Detector (rolling window, percentile, calibration sample), ON-branch model (h48qual's original frozen
exit head, threshold=0.95, hardcoded in research_eth_omega461_exit_head_portfolio_asymmetric_20260813.
_component_cfg regardless of bundle_override), and OFF-branch model (the side-blind classifier,
threshold=0.5, the SAME frozen bundle file arm_eval_report_sideblind.json already scored) are ALL
pre-existing and reused verbatim -- nothing in this script is retrained, refit, or threshold-swept.

=== Method ===
- (a) real_g0 (currently deployed, BOTH branches original -- ON=original exit head 0.95, OFF=liveATR-
  relabeled exit head 0.95): `research_eth_odyssey4_random_direction_risk_management_ablation_20260817.
  run_arm`, unmodified, recomputed here fresh (cheap, and doubles as this script's own G0 reference).
- (b) side-blind signal alone (UNGATED, governs 100% of held-position bars): reused verbatim from
  tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/arm_eval_report_sideblind.
  json's `criterion2_by_window` -- NOT rerun, per the task's explicit instruction (identical classifier,
  identical replay mechanism already on file).
- (c) THIS session's regime-gated hybrid: `research_ilias_eth_adaptive_exit_signal_common_regime_gated_
  20260817.greedy_replay_new_exit_signal_regime_gated` on the SAME always_long/always_short arms.

=== Criterion 1 note (why NOT recomputed here) ===
Criterion 1 (firing-rate/precision-by-direction-quality, N=30, |t|>2) is a property of the classifier
BUNDLE alone, measured via `simulate_private_barrier_trades` + `score_arm_trades` -- a pure offline
label/feature-vs-classifier scoring that never invokes ANY live replay, gate, or guard mechanism. The
regime-gating hybrid tested here changes ONLY when the classifier is consulted during a live replay (the
OFF branch), never the classifier itself or how criterion 1 scores it -- so criterion 1's numbers for
the hybrid are, by construction, IDENTICAL to criterion 1's numbers already computed and reported for
the ungated side-blind signal (arm_eval_report_sideblind.json's `criterion1_by_window`). This script
reuses that dict verbatim (with a label noting the equivalence) rather than re-running an identical
computation.

=== G0 identity check (this script's own validation gate, run BEFORE trusting any hybrid number) ===
`greedy_replay_new_exit_signal_regime_gated` run on h48qual components carrying the detector mask but
NO `new_exit_model` must reduce, bar for bar, to the SAME branching
research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.greedy_replay_regime_aware_exit_
guard already implements (ON -> guard_*, OFF -> base_np/exit_runtime/pos_idx) -- i.e. it must reproduce
`abl.run_arm`'s own always_long/always_short with_gate numbers exactly. If this check fails, the report
aborts before any candidate number is trusted (same discipline as this script's sibling guard/veto
research scripts' own G0 gates).

Background detector activation rate per window is reported alongside every result (interpretation
context: a window where the detector is active on very few bars will show the hybrid behaving almost
identically to the ungated side-blind signal by construction, not because gating "did nothing smart").
Trivial-pass detection (docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md §8's
own pitfall: a "pass" that fires zero real interventions) is checked directly from each arm's raw
`reason_counts` (`new_exit_signal_regime_off` count vs total always_long trades).

fresh_forward_bar_by_bar=true for every replay in this script. simulate_private_barrier_trades (reused,
not called here) is offline label construction, not a live decision -- see its own docstring.
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify any
imported module (including both prior ilias arm-eval scripts, whose report JSONs are read-only inputs
here). No retraining, no GPU (DEVICE=cpu), conda env quant_ai.
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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_odyssey4_random_direction_risk_management_ablation_20260817 as abl  # noqa: E402
import research_eth_odyssey4_random_direction_large_n_reverification_20260817 as abl_large  # noqa: E402
import research_ilias_eth_adaptive_exit_signal_common_regime_gated_20260817 as common_rg  # noqa: E402

BASELINE_OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_regime_gated_20260817"
DEVICE = portfolio.DEVICE
JUDGED_DOWNTREND_WINDOWS = ("val", "oos_q1", "oos_q2")
RANGING_WINDOW_KEYS = [c["key"] for c in abl_large.RANGING_CANDIDATES]
ALL_EVAL_WINDOWS = list(JUDGED_DOWNTREND_WINDOWS) + RANGING_WINDOW_KEYS
GUARDRAIL_MAX_RELATIVE_DEGRADATION = 0.50  # reused convention, unchanged from the sideblind script
G0_TOLERANCE_PP = 0.05


def log(msg: str) -> None:
    common_rg.log(msg)


def _guardrail_pass(baseline_pnl: float, candidate_pnl: float) -> bool:
    """Verbatim copy of the sideblind arm-eval script's _guardrail_pass. Unchanged here."""
    if baseline_pnl == 0.0:
        return bool(candidate_pnl >= 0.0)
    if baseline_pnl > 0.0 and candidate_pnl <= 0.0:
        return False
    relative_degradation = (baseline_pnl - candidate_pnl) / abs(baseline_pnl)
    return bool(relative_degradation <= GUARDRAIL_MAX_RELATIVE_DEGRADATION)


def load_windows() -> dict[str, Any]:
    windows = dict(gate.load_all_windows())
    for cand in abl_large.RANGING_CANDIDATES:
        windows[cand["key"]] = abl_large.load_custom_window(cand)
    return windows


def _close(a: dict[str, Any], b: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(a["pnl"]) - float(b["pnl"])) <= tol_pp
        and abs(float(a["mdd"]) - float(b["mdd"])) <= tol_pp
        and int(a["trades"]) == int(b["trades"])
    )


def _build_hybrid_components(window_name: str, windows: dict, score_by_base: dict, threshold: float,
                              device: torch.device, side_val: int, *, attach_new_model: dict | None) -> tuple[pd.DataFrame, dict]:
    aligned_frame, components = abl.build_ablation_components(
        window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device,
        side_selector=lambda n, _s=side_val: abl._side_selector_constant(n, _s),
    )
    if attach_new_model is not None:
        h48_new = dict(components["h48qual"])
        h48_new["new_exit_model"] = attach_new_model
        h48_new["new_exit_threshold"] = float(attach_new_model["threshold"])
        components = dict(components)
        components["h48qual"] = h48_new
    return aligned_frame, components


def _run_hybrid(window_name: str, windows: dict, score_by_base: dict, threshold: float, device: torch.device,
                 fee: float, slip: float, side_val: int, *, attach_new_model: dict | None) -> dict[str, Any]:
    aligned_frame, components = _build_hybrid_components(
        window_name, windows, score_by_base, threshold, device, side_val, attach_new_model=attach_new_model,
    )
    diag, ledger = common_rg.greedy_replay_new_exit_signal_regime_gated(
        aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
    )
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    no_gate = portfolio._ledger_metrics(ledger)
    return {"no_gate": no_gate, "with_gate": with_gate, "reason_counts": diag.get("reason_counts"), "veto_bars": diag.get("veto_bars")}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()

    log("=== stage=load_frozen_side_blind_bundle (SAME bundle as the ungated side-blind arm-eval) ===")
    with open(BASELINE_OUT_DIR / "new_exit_signal_bundle_sideblind.pkl", "rb") as f:
        bundle = pickle.load(f)
    log(f"  model={bundle['model_name']} threshold={bundle['threshold']} n_train_trades={bundle['n_train_trades']} "
        f"feature_columns={bundle['feature_columns']}")

    log("=== stage=load_sideblind_arm_eval_report (arm b -- reused verbatim, not rerun) ===")
    with open(BASELINE_OUT_DIR / "arm_eval_report_sideblind.json", "r", encoding="utf-8") as f:
        sideblind_report = json.load(f)
    criterion1_by_window = sideblind_report["criterion1_by_window"]  # reused: classifier-only property, gating-independent (see module docstring)
    sideblind_criterion2 = sideblind_report["criterion2_by_window"]

    log("=== stage=load_windows (3 downtrend judged + 3 ranging, same 6 as every prior ilias script) ===")
    windows = load_windows()

    log("=== stage=detector_build (reused, zero new free parameters) ===")
    score_by_base, _robustness, threshold = guard.build_detector()

    # =================================================================================================
    # G0 identity check: hybrid replay WITHOUT new_exit_model attached must reduce byte-for-byte to
    # abl.run_arm's own always_long/always_short with_gate numbers (both call, through different code
    # paths, the SAME mask-gated ON=guard/OFF=default branching for h48qual).
    # =================================================================================================
    log("=== stage=G0_identity_check (hybrid replay minus new_exit_model vs abl.run_arm reference) ===")
    g0: dict[str, Any] = {}
    g0_pass_all = True
    for window_name in ALL_EVAL_WINDOWS:
        ref_long = abl.run_arm("always_long", window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device, fee, slip,
                                side_selector=lambda n: abl._side_selector_constant(n, 1))["with_gate"]
        ref_short = abl.run_arm("always_short", window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device, fee, slip,
                                 side_selector=lambda n: abl._side_selector_constant(n, -1))["with_gate"]
        hyb_long = _run_hybrid(window_name, windows, score_by_base, threshold, device, fee, slip, 1, attach_new_model=None)["with_gate"]
        hyb_short = _run_hybrid(window_name, windows, score_by_base, threshold, device, fee, slip, -1, attach_new_model=None)["with_gate"]
        ok_long, ok_short = _close(hyb_long, ref_long), _close(hyb_short, ref_short)
        g0[window_name] = {"always_long": {"reference": ref_long, "hybrid_no_new_model": hyb_long, "match": ok_long},
                            "always_short": {"reference": ref_short, "hybrid_no_new_model": hyb_short, "match": ok_short}}
        g0_pass_all = g0_pass_all and ok_long and ok_short
        log(f"  {window_name}: AL match={ok_long} (ref={ref_long['pnl']:.2f}%/{ref_long['mdd']:.2f}%/{ref_long['trades']} "
            f"hyb={hyb_long['pnl']:.2f}%/{hyb_long['mdd']:.2f}%/{hyb_long['trades']})  "
            f"AS match={ok_short} (ref={ref_short['pnl']:.2f}%/{ref_short['mdd']:.2f}%/{ref_short['trades']} "
            f"hyb={hyb_short['pnl']:.2f}%/{hyb_short['mdd']:.2f}%/{hyb_short['trades']})")

    if not g0_pass_all:
        report_abort = {"stage_reached": "G0_identity_check", "gate_pass": False, "g0_identity_check": g0,
                         "note": "G0 identity check failed -- hybrid replay function does not reduce to the deployed guard's own behaviour when no_new_exit_model is attached. Aborting before trusting any candidate number."}
        (OUT_DIR / "arm_eval_report_regime_gated.json").write_text(json.dumps(report_abort, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log("stage=ABORT G0 identity check failed")
        return 1
    log("stage=G0_identity_check_result pass=True")

    # =================================================================================================
    # Background detector activation rate per window (context for interpreting the hybrid's results --
    # a low-activation window makes the ON branch nearly irrelevant by construction).
    # =================================================================================================
    log("=== stage=background_detector_activation_rate ===")
    activation: dict[str, float] = {}
    for window_name in ALL_EVAL_WINDOWS:
        _af, _comp, prep_diag = guard.prepare_regime_aware_components(window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device)
        activation[window_name] = float(prep_diag["detector_active_frac"])
        log(f"  {window_name}: detector_active_frac={activation[window_name]*100:.2f}%")

    # =================================================================================================
    # Criterion 1: reused verbatim from the ungated side-blind report (see module docstring for why).
    # =================================================================================================
    passing_windows = [w for w, v in criterion1_by_window.items() if v["criterion1_pass"]]
    log(f"criterion1 passing windows (reused from arm_eval_report_sideblind.json, unchanged): {passing_windows}")

    # =================================================================================================
    # Criterion 2: hybrid replay (arm c) on always_long/always_short, all criterion1-passing windows.
    # =================================================================================================
    log("=== stage=criterion2_replay_hybrid (arm c) ===")
    criterion2: dict[str, Any] = {}
    three_way: dict[str, Any] = {}
    for window_name in passing_windows:
        log(f"  window={window_name}")
        base_long = abl.run_arm("always_long", window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device, fee, slip,
                                 side_selector=lambda n: abl._side_selector_constant(n, 1))["with_gate"]
        base_short = abl.run_arm("always_short", window_name, windows, score_by_base, threshold, BASELINE_OUT_DIR, device, fee, slip,
                                  side_selector=lambda n: abl._side_selector_constant(n, -1))["with_gate"]

        hyb_long = _run_hybrid(window_name, windows, score_by_base, threshold, device, fee, slip, 1, attach_new_model=bundle)
        hyb_short = _run_hybrid(window_name, windows, score_by_base, threshold, device, fee, slip, -1, attach_new_model=bundle)

        mdd_improves_always_long = bool(hyb_long["with_gate"]["mdd"] >= base_long["mdd"])
        guardrail_always_short = _guardrail_pass(base_short["pnl"], hyb_short["with_gate"]["pnl"])
        criterion2_pass = bool(mdd_improves_always_long and guardrail_always_short)

        # Trivial-pass check (docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md
        # §8's own pitfall): did the OFF-branch signal actually fire on the always_long arm, or did
        # mdd_improves_always_long pass purely because the >= comparison tolerates zero intervention?
        al_reason_counts = hyb_long["reason_counts"] or {}
        al_total_trades = int(hyb_long["no_gate"]["trades"])
        al_off_fired = int(al_reason_counts.get("new_exit_signal_regime_off", 0))
        al_on_fired = int(al_reason_counts.get("exit_head_regime_on", 0))
        al_off_firing_rate = float(al_off_fired / al_total_trades) if al_total_trades else float("nan")
        trivial_pass = bool(criterion2_pass and al_off_fired == 0 and al_on_fired == 0)

        criterion2[window_name] = {
            "baseline_always_long_with_gate": base_long, "baseline_always_short_with_gate": base_short,
            "hybrid_always_long": hyb_long, "hybrid_always_short": hyb_short,
            "mdd_improves_always_long": mdd_improves_always_long,
            "guardrail_pass_always_short_pnl": guardrail_always_short,
            "criterion2_pass": criterion2_pass,
            "always_long_off_branch_fired_trades": al_off_fired,
            "always_long_on_branch_fired_trades": al_on_fired,
            "always_long_total_trades_no_gate": al_total_trades,
            "always_long_off_branch_firing_rate": al_off_firing_rate,
            "trivial_pass": trivial_pass,
        }
        log(f"    baseline(a=real_g0): AL pnl={base_long['pnl']:+.2f}% mdd={base_long['mdd']:.2f}%  AS pnl={base_short['pnl']:+.2f}% mdd={base_short['mdd']:.2f}%")
        log(f"    hybrid(c):   AL pnl={hyb_long['with_gate']['pnl']:+.2f}% mdd={hyb_long['with_gate']['mdd']:.2f}%  AS pnl={hyb_short['with_gate']['pnl']:+.2f}% mdd={hyb_short['with_gate']['mdd']:.2f}%")
        log(f"    mdd_improves_always_long={mdd_improves_always_long}  guardrail_always_short={guardrail_always_short}  criterion2_pass={criterion2_pass}  "
            f"AL off-branch fired {al_off_fired}/{al_total_trades} trades ({al_off_firing_rate*100 if al_total_trades else float('nan'):.1f}%)  trivial_pass={trivial_pass}")

        sb = sideblind_criterion2.get(window_name, {})
        three_way[window_name] = {
            "detector_active_frac": activation[window_name],
            "a_real_g0": {"always_long": base_long, "always_short": base_short},
            "b_sideblind_alone_ungated": {
                "always_long": sb.get("new_signal_always_long", {}).get("with_gate"),
                "always_short": sb.get("new_signal_always_short", {}).get("with_gate"),
                "criterion2_pass": sb.get("criterion2_pass"),
            },
            "c_regime_gated_hybrid": {
                "always_long": hyb_long["with_gate"], "always_short": hyb_short["with_gate"],
                "criterion2_pass": criterion2_pass, "trivial_pass": trivial_pass,
            },
        }

    overall_pass = bool(passing_windows) and any(v["criterion2_pass"] for v in criterion2.values())
    downtrend_pass = [w for w in passing_windows if w in JUDGED_DOWNTREND_WINDOWS]
    ranging_pass = [w for w in passing_windows if w in RANGING_WINDOW_KEYS]

    report = {
        "design": __doc__,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "on_branch": "h48qual ORIGINAL (pre-liveATR-relabel) exit head, threshold=0.95 (unchanged from the deployed guard)",
        "off_branch": "Ilias side-blind new exit signal (new_exit_signal_bundle_sideblind.pkl), threshold=0.5",
        "new_exit_signal_bundle": {"model_name": bundle["model_name"], "threshold": bundle["threshold"],
                                     "n_train_trades": bundle["n_train_trades"], "n_train_rows": bundle["n_train_rows"],
                                     "feature_columns": bundle["feature_columns"]},
        "g0_identity_check": g0,
        "g0_identity_check_pass": g0_pass_all,
        "background_detector_activation_rate_by_window": activation,
        "criterion1_by_window_REUSED_FROM_arm_eval_report_sideblind": criterion1_by_window,
        "criterion1_passing_windows": passing_windows,
        "criterion1_passing_downtrend_windows": downtrend_pass,
        "criterion1_passing_ranging_windows": ranging_pass,
        "criterion2_by_window": criterion2,
        "three_way_comparison_by_window": three_way,
        "final_verdict": "SUCCESS" if overall_pass else "KILL",
        "kill_reason": None if overall_pass else (
            "criterion1_failed_all_windows" if not passing_windows else "criterion2_failed_all_criterion1_passing_windows"
        ),
    }
    report_path = OUT_DIR / "arm_eval_report_regime_gated.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.bool_)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={report_path}")
    log(f"FINAL_VERDICT={report['final_verdict']}  passing_windows={passing_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
