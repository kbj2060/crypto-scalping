#!/usr/bin/env python3
"""RESEARCH ONLY -- Odyssey2 priority-queue item #6 (docs/model_contracts/odyssey2_eth_live_
injection_contract_20260813.md, "점검 결과" row #6): "zig075 exit_head 개선 -- 같은 live-ATR
relabel 레시피는 이미 악화로 닫힘(Odyssey1). 다른 접근(개별 재라벨 파라미터, 별도 exit_threshold
등) 미탐색."

Every post-entry candidate run tonight (GBDT/TCN full replacement, queue-pressure, risk-controlled,
selective-conformal, regime-aware guard) held EXIT_THRESHOLD=0.95 (research_eth_omega461_exit_
sweep_20260721.BASELINE_EXIT_THRESHOLD) fixed and identical for BOTH h48qual and zig075 -- only
h48qual's exit_head *model* (which set of weights answers "hold or exit?") was ever varied. The one
axis nobody has touched: zig075's exit_threshold NUMBER itself, with zig075's model completely
untouched (no retrain, no relabel). This is the cheapest unexplored axis in the queue -- it requires
zero retraining, because replay_omega4_6_1_greedy_router_20260706.greedy_replay already reads a
PER-COMPONENT exit_threshold from each component's own prepared dict (`comp["exit_threshold"]`,
line "if prob >= comp['exit_threshold']:") -- varying it is a pure config change, not a code change.

=== Hypothesis (BOTH directions left open, no directional prior) ===
Tonight's h48qual investigation (docs/experiments/eth_omega461_exit_head_liveatr_sustained_uptrend_
vulnerability_20260814.md) found: lowering exit_threshold (or relabeling toward live-ATR) speeds up
turnover. For h48qual -- a component with NO validated direction-side skill (see [[h48qual_
standalone_replay_invalid]]) -- faster turnover was net-positive: cutting a mediocre trade early and
recycling the shared slot mattered more than letting any one trade run. zig075 is the opposite
profile: it is this project's one component with a validated, direction-specific edge (short-side
"bear beta"). Two competing predictions follow, and this script does not pick one in advance:
  (a) SLOWER turnover (raise zig075's exit_threshold above 0.95, exit only when the model is MORE
      confident) could let zig075's good short trades run further before an early exit-head flip
      cuts them off -- net positive.
  (b) FASTER turnover (lower zig075's exit_threshold below 0.95) could still help by freeing the
      shared h48qual>zig075 slot sooner, the same slot-recycling mechanism GBDT/TCN/queue-pressure
      all found on the h48qual side -- net positive via a completely different channel.
Both are tested on the SAME symmetric grid; the grid itself does not favor either direction.

=== What is held fixed vs varied ===
- h48qual: EXACTLY today's confirmed shadow baseline (asymmetric_tabm_liveatr) -- live-ATR-relabeled
  exit_head bundle (research_eth_omega461_exit_head_portfolio_asymmetric_20260813.NEW_H48QUAL_
  BUNDLE), exit_threshold=0.95, unchanged in every single run in this script. h48qual is NEVER swept.
- zig075: direction_head/quality_head/encoder/exit_head WEIGHTS are its ORIGINAL frozen live bundle
  (research_eth_omega461_exit_sweep_20260721.COMPONENTS["zig075"]["bundle"], no relabel, no
  retrain) -- identical to the current live/shadow config. ONLY zig075's exit_threshold (the number
  compared against the frozen exit_head's output probability) is swept:
  {0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99} -- dense on both the narrow (raise-threshold) and
  wide (lower-threshold) sides of the 0.95 anchor, symmetric per the task's explicit instruction not
  to pre-commit to a direction.

=== Reuse discipline (no modification of any shared function/module) ===
research_eth_omega461_exit_sweep_20260721.py (prep_component/replay_exit_variant/run_grid),
replay_omega4_6_1_greedy_router_20260706.py (greedy_replay/prepare_component),
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py (_component_cfg/_prepare_
component_val -- the asymmetric-baseline builder), research_eth_omega461_live_sltp_mfe_width_
20260813.py (_duration_gated), research_eth_omega461_risk_controlled_exit_fallback_20260814.py
(_guardrail_ok, reused verbatim for the relaxed-gate guardrail formula), and
eth_omega461_multiwindow_confirmation_gate_20260814.py (load_all_windows/run_portfolio_variant/
summarize_multiwindow/_close/ALL_WINDOWS) are all IMPORTED AND READ ONLY -- never edited. This
script needs no "renamed copy" of greedy_replay at all (unlike GBDT/TCN/queue-pressure/risk-
controlled, which each needed one because their intervention added NEW conditional logic inside the
exit-decision block) -- exit_threshold is already a first-class per-component config value in the
unmodified harness, so a plain config-dict override is the entire "intervention".

=== Gate criteria (same dual-criterion vocabulary as every other Odyssey2 post-entry candidate
tonight, e.g. #7/#8/#9/#14) ===
zig075 (not h48qual) is the intervention target here, so "component" below means zig075's own
standalone full-capital replay (research_eth_omega461_exit_sweep_20260721.prep_component +
replay_exit_variant on zig075 alone -- the same harness research_eth_omega461_exit_sweep_20260721's
own main() Experiment A already uses for exactly this kind of per-component exit_threshold sweep).
  (a) ORIGINAL: zig075-component NO_GATE PnL+MDD AND portfolio NO_GATE PnL+MDD all non-worse than
      the et=0.95 baseline.
  (b) RELAXED (docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md): portfolio WITH_
      GATE PnL strictly improved AND portfolio WITH_GATE MDD within 3pp of baseline AND the
      component guardrail (research_eth_omega461_risk_controlled_exit_fallback_20260814._guardrail_
      ok, reused unmodified: zig075-component PnL must not flip sign nor worsen >50% relative).
A grid point "passes" if EITHER criterion passes (passes_any), matching #7/#8/#9/#14's convention.

=== Robustness / isolated-spike rejection (per task instruction, citing tonight's #12 side-aware-
revival precedent, research_eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.py) ===
A passing grid point is only treated as a "robust" candidate if EVERY existing immediate grid
neighbor (by threshold order) ALSO passes (passes_any). A passing point bracketed by failing
neighbors on both sides is an isolated spike -- explicitly rejected with reason logged, never
promoted regardless of its headline number, exactly as #12 rejected LONG=0.35.

=== OOS confirmation ===
If (and only if) a robust VAL candidate exists, this script opens OOS-Q1+OOS-Q2 ONCE, together, via
eth_omega461_multiwindow_confirmation_gate_20260814.summarize_multiwindow (single touch, not
sequential) for the single best (highest portfolio WITH_GATE PnL) robust candidate. 2025 Q1/Q2/Q3
are also computed and shown for context (via the same 6-window loop, ALL_WINDOWS) but never gate
pass/fail, per this project's now-standard multiwindow policy.

fresh_forward_bar_by_bar=true (every replay here is greedy.greedy_replay / sweep.replay_exit_variant,
single causal forward passes, i increasing, only bar i and already-closed history used at bar i --
this script adds no new bar-by-bar simulation logic of its own). trade_ledgers_used_as_input=false
(ledgers are written-only outputs). saved_parent_exit_timestamps_used=false. future_rows_used_for_
entry=false. No retraining, no GPU. Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_
live.py / trading_bot_modules/runtime_config.py / .env. h48qual is untouched in every window.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_risk_controlled_exit_fallback_20260814 as rc_mod  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_zig075_exit_threshold_recalibration_20260814"
DEVICE = portfolio.DEVICE

ZIG075_EXIT_THRESHOLD_GRID = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.98, 0.99]
BASELINE_ET = 0.95
MDD_SLACK_PP = 3.0  # docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md, reused verbatim
G0_TOLERANCE_PP = 0.05

# Independent historical cross-check only (NOT this script's G0, which is the portfolio-level check
# below) -- research_eth_omega461_exit_sweep_20260721.py's OWN main() already ran a per-component
# exit_threshold sweep on 2026-07-21 (tmp/research_20260721/exit_threshold_sweep_VAL.csv), before
# zig075 was ever touched by this project (zig075's bundle/config have not changed since). If this
# script's freshly-computed zig075-standalone et=0.95 number reproduces that file's row, it is
# independent evidence the component-only harness composition here is correct.
HISTORICAL_CROSS_CHECK_CSV = ROOT / "tmp/research_20260721/exit_threshold_sweep_VAL.csv"


def log(msg: str) -> None:
    print(f"[zig075_exit_threshold] {msg}", flush=True)


def _comp_cfgs(zig075_exit_threshold: float) -> dict[str, dict[str, Any]]:
    """h48qual: today's confirmed asymmetric_tabm_liveatr baseline, exit_threshold=0.95, NEVER
    varied. zig075: original frozen bundle (unchanged), exit_threshold overridden to the candidate
    value -- the entire "intervention" in this script."""
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = dict(portfolio._component_cfg("zig075"))
    zig075_cfg["exit_threshold"] = float(zig075_exit_threshold)
    return {"h48qual": h48qual_cfg, "zig075": zig075_cfg}


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8"
    )
    log(f"report={OUT_DIR / 'report.json'}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Odyssey2 #6 (zig075 exit_head improvement): zig075's exit_head MODEL is untouched "
            "(original frozen bundle, no retrain, no relabel) -- only the exit_threshold NUMBER "
            "compared against its already-frozen exit-head probability is swept. h48qual is held "
            "fixed at today's confirmed asymmetric_tabm_liveatr baseline (exit_threshold=0.95) in "
            "every single run in this script."
        ),
        "hypothesis": (
            "Both directions left open, no directional prior: (a) raising zig075's exit_threshold "
            "above 0.95 could let its validated short-side edge run further before an early exit "
            "cuts a good trade off; (b) lowering it could still help via the same shared-slot "
            "recycling mechanism GBDT/TCN/queue-pressure found on the h48qual side. Grid is "
            "symmetric around 0.95 and does not favor either direction."
        ),
        "zig075_exit_threshold_grid": ZIG075_EXIT_THRESHOLD_GRID,
        "baseline_et": BASELINE_ET,
        "mdd_slack_pp": MDD_SLACK_PP,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }

    # =================================================================================================
    # stage=load_windows
    # =================================================================================================
    log("stage=load_windows")
    windows = gate.load_all_windows()

    # =================================================================================================
    # stage=G0 -- portfolio-level self-reproduction: zig075 exit_threshold=0.95 (the grid's own
    # anchor point) must exactly reproduce tonight's repeatedly-verified asymmetric_tabm_liveatr
    # baseline on VAL and OOS-Q1 (task-mandated reference numbers, sourced from
    # gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR, itself read directly from tmp/causal_regen_
    # 20260516/eth_omega461_risk_controlled_exit_fallback_20260814/report.json).
    # =================================================================================================
    log("stage=G0 (zig075 et=0.95 must reproduce asymmetric_tabm_liveatr exactly)")
    g0: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = gate.run_portfolio_variant(
            wname, windows, _comp_cfgs(BASELINE_ET), fee=fee, slip=slip, device=device,
            out_dir=OUT_DIR, variant_label="g0_zig075_et095",
        )
        ref_ng, ref_wg = gate.REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng = gate._close(result["no_gate"], ref_ng)
        ok_wg = gate._close(result["with_gate"], ref_wg)
        g0[wname] = {
            "no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
            "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg},
        }
        log(f"  {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
            f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}")
    g0_pass = all(g0[w]["no_gate"]["match"] and g0[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0"] = g0
    report["g0_pass"] = g0_pass
    log(f"stage=G0_result pass={g0_pass}")
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["final_verdict"] = "ABORTED_G0_FAIL"
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    # =================================================================================================
    # stage=component_baseline -- zig075 STANDALONE (full-capital, isolated) exit_threshold sweep on
    # VAL, via the original, unmodified sweep.prep_component/sweep.run_grid harness (exactly the
    # composition research_eth_omega461_exit_sweep_20260721.py's own main() Experiment A already
    # uses). This is the "component" leg of the ORIGINAL gate criterion.
    # =================================================================================================
    log("stage=component_baseline (zig075 standalone exit_threshold grid, VAL)")
    val_frame_component = sweep.load_frame(sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025)
    zig_pred_val = sweep.EXT_PRED_DIR / "zig075" / f"validation_predictions_{sweep.COMPONENTS['zig075']['q_tag']}.csv"
    zig_prepped_val = sweep.prep_component("zig075", sweep.COMPONENTS["zig075"], val_frame_component, zig_pred_val, oof=True)
    component_grid_df = sweep.run_grid({"zig075": zig_prepped_val}, exit_thresholds=ZIG075_EXIT_THRESHOLD_GRID)
    component_grid: dict[float, dict[str, Any]] = {}
    for et in ZIG075_EXIT_THRESHOLD_GRID:
        row = component_grid_df[component_grid_df["exit_threshold"] == et].iloc[0]
        component_grid[et] = {"pnl": float(row["pnl"]), "mdd": float(row["mdd"]), "trades": int(row["trades"]), "wr": float(row["wr"]), "avg_hold_bars": float(row["avg_hold_bars"])}
        log(f"  component et={et:.2f}: pnl={component_grid[et]['pnl']:.2f}% mdd={component_grid[et]['mdd']:.2f}% trades={component_grid[et]['trades']}")

    historical_cross_check: dict[str, Any] = {"available": False}
    if HISTORICAL_CROSS_CHECK_CSV.exists():
        hist = pd.read_csv(HISTORICAL_CROSS_CHECK_CSV)
        hist_row = hist[(hist["component"] == "zig075") & (hist["exit_threshold"].round(2) == BASELINE_ET)]
        if len(hist_row):
            hr = hist_row.iloc[0]
            match = (abs(float(hr["pnl"]) - component_grid[BASELINE_ET]["pnl"]) < 0.01
                      and abs(float(hr["mdd"]) - component_grid[BASELINE_ET]["mdd"]) < 0.01
                      and int(hr["trades"]) == component_grid[BASELINE_ET]["trades"])
            historical_cross_check = {
                "available": True, "source": str(HISTORICAL_CROSS_CHECK_CSV), "date": "2026-07-21 (pre-dates all Odyssey1/2 zig075 work; zig075 bundle/config unchanged since)",
                "historical": {"pnl": float(hr["pnl"]), "mdd": float(hr["mdd"]), "trades": int(hr["trades"])},
                "fresh": component_grid[BASELINE_ET], "match": bool(match),
            }
    report["component_baseline_historical_cross_check"] = historical_cross_check
    log(f"  historical cross-check (independent, not this script's G0): {historical_cross_check.get('match')}")

    # =================================================================================================
    # stage=val_grid -- full portfolio grid sweep, dual gate criterion (original / relaxed) per point.
    # =================================================================================================
    log("stage=val_grid (portfolio, full 8-point grid)")
    baseline_no_gate = g0["val"]["no_gate"]["actual"]
    baseline_with_gate = g0["val"]["with_gate"]["actual"]
    baseline_component = component_grid[BASELINE_ET]

    val_grid_rows: list[dict[str, Any]] = []
    for et in ZIG075_EXIT_THRESHOLD_GRID:
        result = gate.run_portfolio_variant(
            "val", windows, _comp_cfgs(et), fee=fee, slip=slip, device=device,
            out_dir=OUT_DIR, variant_label=f"val_zig075_et{et:.2f}",
        )
        comp = component_grid[et]
        gate_original = {
            "component_pnl_nonworse": bool(comp["pnl"] >= baseline_component["pnl"]),
            "component_mdd_nonworse": bool(comp["mdd"] >= baseline_component["mdd"]),
            "portfolio_pnl_nonworse": bool(result["no_gate"]["pnl"] >= baseline_no_gate["pnl"]),
            "portfolio_mdd_nonworse": bool(result["no_gate"]["mdd"] >= baseline_no_gate["mdd"]),
        }
        gate_original["pass"] = bool(all(gate_original.values()))
        guardrail_ok = rc_mod._guardrail_ok(float(baseline_component["pnl"]), float(comp["pnl"]))
        gate_relaxed = {
            "portfolio_with_gate_pnl_improved": bool(result["with_gate"]["pnl"] > baseline_with_gate["pnl"]),
            "portfolio_with_gate_mdd_within_3pp": bool((result["with_gate"]["mdd"] - baseline_with_gate["mdd"]) >= -MDD_SLACK_PP),
            "component_guardrail_ok": bool(guardrail_ok),
        }
        gate_relaxed["pass"] = bool(all(gate_relaxed.values()))
        # A grid point BYTE-IDENTICAL to baseline (same no_gate AND with_gate pnl/mdd/trades) is a
        # structural no-op -- zig075's exit-head probability never actually crosses this threshold
        # value while holding a position, so nothing in the replay differs from baseline. Original
        # gate's ">=" ("non-worse") trivially passes an exact tie (matching this project's own
        # established convention -- #11's byte-identical Q1/Q2 no-op windows were correctly counted
        # as "non-worse" too), but a tie is NOT an improvement worth spending the single-touch OOS
        # opportunity on. Tracked separately so winner-selection below excludes ties explicitly
        # instead of silently promoting one via Python's arbitrary max()-tie-breaking order.
        is_degenerate_tie = bool(
            abs(result["no_gate"]["pnl"] - baseline_no_gate["pnl"]) < 1.0e-6 and abs(result["no_gate"]["mdd"] - baseline_no_gate["mdd"]) < 1.0e-6
            and int(result["no_gate"]["trades"]) == int(baseline_no_gate["trades"])
            and abs(result["with_gate"]["pnl"] - baseline_with_gate["pnl"]) < 1.0e-6 and abs(result["with_gate"]["mdd"] - baseline_with_gate["mdd"]) < 1.0e-6
            and int(result["with_gate"]["trades"]) == int(baseline_with_gate["trades"])
        )
        row = {
            "zig075_exit_threshold": et,
            "is_baseline_point": bool(et == BASELINE_ET),
            "is_degenerate_tie_with_baseline": is_degenerate_tie,
            "component_zig075_no_gate": comp,
            "portfolio_no_gate": result["no_gate"],
            "portfolio_with_gate": result["with_gate"],
            "gate_original": gate_original,
            "gate_relaxed": gate_relaxed,
            "passes_any": bool(gate_original["pass"] or gate_relaxed["pass"]),
        }
        val_grid_rows.append(row)
        log(f"  et={et:.2f} component={comp['pnl']:+.2f}%/{comp['mdd']:.2f}%/{comp['trades']}  "
            f"portfolio_no_gate={result['no_gate']['pnl']:+.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']}  "
            f"portfolio_with_gate={result['with_gate']['pnl']:+.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']}  "
            f"gate_original={gate_original['pass']} gate_relaxed={gate_relaxed['pass']} degenerate_tie={is_degenerate_tie}")
    report["val_grid"] = val_grid_rows

    # =================================================================================================
    # stage=robustness -- reject isolated single-point spikes (per #12's established principle:
    # research_eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.py), require every
    # existing immediate grid neighbor to also pass.
    # =================================================================================================
    log("stage=robustness (reject isolated spikes, require grid-neighbor confirmation)")
    passing = [r for r in val_grid_rows if r["passes_any"]]
    robust: list[dict[str, Any]] = []
    isolated_rejected: list[dict[str, Any]] = []
    for i, r in enumerate(val_grid_rows):
        if not r["passes_any"]:
            continue
        neighbors_pass = []
        if i > 0:
            neighbors_pass.append(val_grid_rows[i - 1]["passes_any"])
        if i < len(val_grid_rows) - 1:
            neighbors_pass.append(val_grid_rows[i + 1]["passes_any"])
        if neighbors_pass and all(neighbors_pass):
            robust.append(r)
        else:
            isolated_rejected.append({
                "zig075_exit_threshold": r["zig075_exit_threshold"],
                "reason": "bracketed by a failing neighbor on at least one side -- isolated single-point spike, rejected regardless of headline number (same principle as #12's LONG=0.35 rejection)",
            })
    log(f"  passing (either gate): {[r['zig075_exit_threshold'] for r in passing]}")
    log(f"  robust (all existing neighbors also pass): {[r['zig075_exit_threshold'] for r in robust]}")
    log(f"  isolated spikes rejected: {[r['zig075_exit_threshold'] for r in isolated_rejected]}")

    # A "candidate" must be robust (neighbor-confirmed), not the baseline anchor point itself, and
    # NOT a degenerate tie with baseline -- an exact tie is not an improvement, and promoting one to
    # the single-touch OOS opportunity would waste it on a config that is VAL-indistinguishable from
    # simply not touching zig075 at all (see is_degenerate_tie_with_baseline comment above).
    candidates = [r for r in robust if not r["is_baseline_point"] and not r["is_degenerate_tie_with_baseline"]]
    degenerate_robust = [r["zig075_exit_threshold"] for r in robust if r["is_degenerate_tie_with_baseline"] and not r["is_baseline_point"]]
    val_winner = max(candidates, key=lambda r: r["portfolio_with_gate"]["pnl"]) if candidates else None
    report["val_robustness"] = {
        "passing_any_gate": [r["zig075_exit_threshold"] for r in passing],
        "robust_neighbor_confirmed": [r["zig075_exit_threshold"] for r in robust],
        "robust_but_degenerate_tie_excluded_from_candidacy": degenerate_robust,
        "isolated_spikes_rejected": isolated_rejected,
    }
    report["val_winner"] = val_winner
    log(f"  robust but degenerate-tie (excluded from candidacy): {degenerate_robust}")
    log(f"  VAL winner (robust AND genuinely non-tied improvement): {val_winner['zig075_exit_threshold'] if val_winner else None}")

    if val_winner is None:
        report["oos_opened"] = False
        report["stage_reached"] = "val_grid"
        report["gate_pass"] = True
        report["final_verdict"] = "REJECTED_VAL_GATE"
        _write_report(report)
        log("FINAL VERDICT: REJECTED_VAL_GATE (no robust VAL candidate -- OOS not opened)")
        return 0

    # =================================================================================================
    # stage=oos_single_touch -- OOS-Q1+OOS-Q2 opened TOGETHER, once, for the single VAL winner, via
    # eth_omega461_multiwindow_confirmation_gate_20260814.summarize_multiwindow (reused unmodified).
    # 2025 Q1/Q2/Q3 computed in the same 6-window loop for context only (never gates).
    # =================================================================================================
    winner_et = val_winner["zig075_exit_threshold"]
    log(f"stage=oos_single_touch (winner et={winner_et:.2f}, all 6 windows via gate.ALL_WINDOWS)")
    baseline_cfgs = _comp_cfgs(BASELINE_ET)
    candidate_cfgs = _comp_cfgs(winner_et)
    baseline_tuples: dict[str, tuple[dict, dict]] = {}
    candidate_tuples: dict[str, tuple[dict, dict]] = {}
    six_window_table: dict[str, Any] = {}
    for wname in gate.ALL_WINDOWS:
        rb = gate.run_portfolio_variant(wname, windows, baseline_cfgs, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="baseline_zig075_et095")
        rc = gate.run_portfolio_variant(wname, windows, candidate_cfgs, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label=f"candidate_zig075_et{winner_et:.2f}")
        baseline_tuples[wname] = (rb["no_gate"], rb["with_gate"])
        candidate_tuples[wname] = (rc["no_gate"], rc["with_gate"])
        six_window_table[wname] = {"tier": gate.WINDOW_DEFS[wname]["tier"], "baseline_no_gate": rb["no_gate"], "baseline_with_gate": rb["with_gate"], "candidate_no_gate": rc["no_gate"], "candidate_with_gate": rc["with_gate"]}
        log(f"  {wname} ({gate.WINDOW_DEFS[wname]['tier']}): baseline no_gate={rb['no_gate']['pnl']:+.2f}%/{rb['no_gate']['mdd']:.2f}%/{rb['no_gate']['trades']} with_gate={rb['with_gate']['pnl']:+.2f}%/{rb['with_gate']['mdd']:.2f}%/{rb['with_gate']['trades']}  |  "
            f"candidate no_gate={rc['no_gate']['pnl']:+.2f}%/{rc['no_gate']['mdd']:.2f}%/{rc['no_gate']['trades']} with_gate={rc['with_gate']['pnl']:+.2f}%/{rc['with_gate']['mdd']:.2f}%/{rc['with_gate']['trades']}")

    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=MDD_SLACK_PP)
    log(f"  multiwindow verdict: strict={summary_strict['final_verdict']} relaxed_mdd3pp={summary_relaxed['final_verdict']}")

    final_verdict = "CONFIRMED" if (summary_strict["final_verdict"] == "CONFIRMED" or summary_relaxed["final_verdict"] == "CONFIRMED") else "REJECTED_SIGN_MISMATCH"

    report["oos_opened"] = True
    report["winner_zig075_exit_threshold"] = winner_et
    report["six_window_table"] = six_window_table
    report["multiwindow_summary_strict"] = summary_strict
    report["multiwindow_summary_relaxed"] = summary_relaxed
    report["stage_reached"] = "oos_single_touch"
    report["gate_pass"] = True
    report["final_verdict"] = final_verdict
    _write_report(report)
    log(f"FINAL VERDICT: {final_verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
