#!/usr/bin/env python3
"""RESEARCH ONLY -- follow-up to docs/experiments/eth_yush_orderflow_strategy_absorption_study_20260815.md's
second untested leftover: Yush's risk-discipline rule #2, "never let a winning trade turn into a
loser" (breakeven stop), flagged there as "testable in the Omega TP/SL frame as an exit overlay...
this document does not measure it."

Mechanism: once a trade's MFE reaches `ACTIVATE_FRAC` of its own take_profit target, arm a stop at
breakeven (entry price) -- from that point, if price falls back to entry or below, exit there
instead of continuing to hold toward the original wider TP/SL. This is exactly
research_eth_omega461_exit_sweep_20260721.replay_exit_variant's existing PROPORTIONAL trailing-stop
hook (`trailing_activate_frac` + `trailing_retain_frac`) at `trailing_retain_frac=0.0` -- no new
simulation logic was written for the component level. The portfolio-level router
(replay_omega4_6_1_greedy_router_20260706.greedy_replay) only supported the FIXED-DISTANCE trailing
variant (`trailing_trail_frac`) before this experiment; a `trailing_retain_frac` parameter mirroring
the sibling function was added to it in the same commit as this script (small, additive, all
existing call sites unaffected since it's a new keyword-only arg defaulting to None) because a fixed
multiple of |stop_loss| cannot reproduce "exit at exactly breakeven" for trades whose MFE varies.

ACTIVATE_FRAC=0.5 (arm once MFE reaches 50% of the trade's own take-profit target) is the ONE
pre-registered choice -- not swept/tuned, chosen because it uses the model's own existing TP as the
reference scale (no new free constant) and is the simplest "well into profit" reading of Yush's own
undated description. No second attempt at a different threshold is made regardless of outcome.

Tested against the full 6-window gate (eth_omega461_multiwindow_confirmation_gate_20260814,
VAL + OOS-Q1 + OOS-Q2 single-touch + 2025-Q1/Q2/Q3 reference), not just VAL->single-OOS, because
Odyssey2's own finding (VAL has the least exit-improvement headroom of the 6 windows) makes VAL-only
screening for an EXIT-side candidate specifically unreliable in this repo.

Component-level (h48qual alone, zig075 alone, other component absent from that specific replay) AND
portfolio-level (both components simultaneously, in the router's own slot-competition simulation --
the more decision-relevant number, since exit-side changes in one component have repeatedly been
found in this repo's history to alter which OTHER component's signals get a trade slot) are both
reported; the gate verdict is decided on portfolio with_gate only, matching every other Odyssey-era
candidate in this repo.

fresh_forward_bar_by_bar=true (single forward bar-by-bar pass, replay_exit_variant/greedy_replay
unmodified in control flow). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. No retraining -- ACTIVATE_FRAC/RETAIN_FRAC are runtime execution
constants like TP/SL, not learned weights, so this is a pure deterministic backtest replay.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio_mod  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as helpers  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as router  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_yush_breakeven_stop_exit_overlay_20260815"
BASELINE_EXIT_THRESHOLD = 0.95  # unchanged -- isolate the breakeven-stop axis only, matches sibling scripts.
ACTIVATE_FRAC = 0.5             # pre-registered, not swept. See module docstring.
RETAIN_FRAC = 0.0               # exact breakeven.

# Known VAL reference (component h48qual alone, baseline exit_threshold=0.95, no trailing) --
# reproduced independently by research_eth_omega461_atr_tpsl_floor_independent_percomponent_20260815.py's
# own G0 check; reused here as this harness's G0 too.
G0_EXPECTED_PORTFOLIO_VAL_NO_GATE = {"pnl": 36.82, "mdd": -24.34, "trades": 29}
G0_TOL_PP = 0.5


def log(msg: str) -> None:
    print(f"[yush_breakeven] {msg}", flush=True)


def _component_metrics(name: str, cfg: dict, frame: pd.DataFrame, pred_path: Path, *, oof: bool,
                        breakeven: bool) -> dict[str, Any]:
    p = sweep.prep_component(name, cfg, frame, pred_path, oof=oof)
    kwargs: dict[str, Any] = dict(
        risk_margin_fraction=p["margin"], risk_leverage=p["leverage"], exit_threshold=BASELINE_EXIT_THRESHOLD,
        fee=p["fee"], slip=p["slip"], cost_mult=sweep.COST_MULT, notional_scaled_sltp=p["notional_scaled_sltp"],
        device=sweep.DEVICE,
    )
    if breakeven:
        kwargs["trailing_activate_frac"] = ACTIVATE_FRAC
        kwargs["trailing_retain_frac"] = RETAIN_FRAC
    m, _ledger = sweep.replay_exit_variant(p["frame"], p["x"], p["dec"], p["loaded"], **kwargs)
    return {**{k: v for k, v in m.items() if k != "exit_reasons"}, "exit_reasons": json.dumps(m["exit_reasons"])}, p


def _portfolio_metrics(aligned_frame: pd.DataFrame, aligned_paths: dict[str, Path], *, oof: bool, breakeven: bool,
                        ledger_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    # Mirrors gate.run_portfolio_variant's own preparation path EXACTLY (greedy.prepare_component /
    # portfolio._prepare_component_val chosen by the window's oof flag, same as that function) --
    # required because sweep.prep_component does its OWN per-component row intersection (documented
    # in load_all_windows' docstring: 2025q1/2025q3 have ~66/30-row prediction-coverage gaps versus
    # the raw feature frame), which silently desyncs component-internal arrays from an un-aligned
    # `frame` passed straight to greedy_replay (hit as a live IndexError during this experiment on
    # the 2025q1 window -- fixed by switching to this path, not by adding a truncate/pad workaround).
    components: dict[str, Any] = {}
    for cname, base_cfg in sweep.COMPONENTS.items():
        cfg = dict(base_cfg, exit_threshold=BASELINE_EXIT_THRESHOLD)
        if oof:
            components[cname] = portfolio_mod._prepare_component_val(aligned_frame, aligned_paths[cname], cfg, sweep.DEVICE)
        else:
            components[cname] = router.prepare_component(aligned_frame, aligned_paths[cname], cfg, sweep.DEVICE)
    fee, slip = omega_mod._load_fee_slip()
    kwargs: dict[str, Any] = dict(fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=sweep.DEVICE)
    if breakeven:
        kwargs["trailing_activate_frac"] = ACTIVATE_FRAC
        kwargs["trailing_retain_frac"] = RETAIN_FRAC
    _diag, ledger = router.greedy_replay(aligned_frame, components, **kwargs)
    ledger.to_csv(ledger_path, index=False)
    no_gate = portfolio_mod._ledger_metrics(ledger)
    with_gate = helpers._duration_gated(ledger, aligned_frame, router.DURATION_THRESHOLD)
    return no_gate, with_gate


def run_window(wname: str, w: dict[str, Any]) -> dict[str, Any]:
    frame = w["frame"]
    log(f"=== window={wname} tier={w['tier']} rows={len(frame)} ===")
    out: dict[str, Any] = {"component": {}, "portfolio": {}}

    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in sweep.COMPONENTS}
    aligned_frame, aligned_paths = gate.align_frame_and_predictions(frame, q_tags, w["split"], OUT_DIR)

    for breakeven, label in ((False, "baseline"), (True, "breakeven")):
        comp_metrics: dict[str, Any] = {}
        for name, cfg in sweep.COMPONENTS.items():
            m, _p = _component_metrics(name, cfg, frame, w["raw_paths"][name], oof=w["oof"], breakeven=breakeven)
            comp_metrics[name] = m
            log(f"  [{label}] component={name} pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} "
                f"wr={m['wr']*100:.1f}% avg_hold={m['avg_hold_bars']:.1f} exit_reasons={m['exit_reasons']}")
        out["component"][label] = comp_metrics

        ledger_path = OUT_DIR / f"portfolio_ledger_{wname}_{label}.csv"
        no_gate, with_gate = _portfolio_metrics(aligned_frame, aligned_paths, oof=w["oof"], breakeven=breakeven,
                                                 ledger_path=ledger_path)
        out["portfolio"][label] = (no_gate, with_gate)
        log(f"  [{label}] PORTFOLIO no_gate pnl={no_gate['pnl']:.2f}% mdd={no_gate['mdd']:.2f}% trades={no_gate['trades']} | "
            f"with_gate pnl={with_gate['pnl']:.2f}% mdd={with_gate['mdd']:.2f}% trades={with_gate['trades']}")

    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = gate.load_all_windows()

    results: dict[str, Any] = {}
    for wname, w in windows.items():
        results[wname] = run_window(wname, w)

    # G0: this harness's own baseline portfolio no_gate on VAL must reproduce the independently-known
    # reference (same check the ATR-floor sibling script runs).
    g0_measured = results["val"]["portfolio"]["baseline"][0]
    g0_ok = (abs(g0_measured["pnl"] - G0_EXPECTED_PORTFOLIO_VAL_NO_GATE["pnl"]) < G0_TOL_PP and
             abs(g0_measured["mdd"] - G0_EXPECTED_PORTFOLIO_VAL_NO_GATE["mdd"]) < G0_TOL_PP and
             g0_measured["trades"] == G0_EXPECTED_PORTFOLIO_VAL_NO_GATE["trades"])
    log(f"G0 self-consistency: pnl={g0_measured['pnl']:.2f} (expect {G0_EXPECTED_PORTFOLIO_VAL_NO_GATE['pnl']}) "
        f"mdd={g0_measured['mdd']:.2f} (expect {G0_EXPECTED_PORTFOLIO_VAL_NO_GATE['mdd']}) "
        f"trades={g0_measured['trades']} (expect {G0_EXPECTED_PORTFOLIO_VAL_NO_GATE['trades']}) -> "
        f"{'PASS' if g0_ok else 'FAIL'}")

    baseline_tuples = {wname: r["portfolio"]["baseline"] for wname, r in results.items()}
    candidate_tuples = {wname: r["portfolio"]["breakeven"] for wname, r in results.items()}
    summary_strict = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=0.0)
    summary_relaxed = gate.summarize_multiwindow(baseline_tuples, candidate_tuples, mdd_slack_pp=3.0)
    log(f"VERDICT (strict, 0pp mdd slack): {summary_strict['final_verdict']} "
        f"per-window={summary_strict['oos_confirm_per_window_pass']}")
    log(f"VERDICT (relaxed, 3pp mdd slack): {summary_relaxed['final_verdict']} "
        f"per-window={summary_relaxed['oos_confirm_per_window_pass']}")

    report = {
        "design": {"activate_frac": ACTIVATE_FRAC, "retain_frac": RETAIN_FRAC,
                   "baseline_exit_threshold": BASELINE_EXIT_THRESHOLD},
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "g0_self_consistency": {"ok": bool(g0_ok), "measured": g0_measured,
                                 "expected": G0_EXPECTED_PORTFOLIO_VAL_NO_GATE},
        "windows": {wname: {"component": r["component"],
                             "portfolio_no_gate": {lbl: r["portfolio"][lbl][0] for lbl in ("baseline", "breakeven")},
                             "portfolio_with_gate": {lbl: r["portfolio"][lbl][1] for lbl in ("baseline", "breakeven")}}
                    for wname, r in results.items()},
        "verdict_strict": summary_strict, "verdict_relaxed": summary_relaxed,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"wrote {OUT_DIR / 'report.json'}")
    return 0 if g0_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
