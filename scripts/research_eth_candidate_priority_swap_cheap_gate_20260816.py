#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate for a new candidate idea: swap the fixed slot-priority order between
h48qual and zig075 (currently PRIORITY=("h48qual","zig075") in replay_omega4_6_1_greedy_router_20260706.py)
to zig075-first, motivated by a frequency check (2026-08-16, ad hoc, not persisted as a script) that found:
- h48qual and zig075 never signal in opposite directions simultaneously across any of the 6 windows
  (opp_dir=0 always) -- removes the hedge/netting-conflict concern for any slot-sharing redesign.
- 15.8% of zig075 quality-gate-passing signal episodes (765/4844 pooled across all 6 windows) occur while
  h48qual is holding the shared slot and are therefore fully blocked -- concentrated in 2025-Q3 (38.7%)
  and VAL (23.4%), much rarer in OOS-Q2 (0.8%).
Given h48qual has repeatedly shown weak-to-no direction skill this session (independent of this check),
a cheap first test is simply swapping which component gets first refusal on the shared slot -- a single-bit
config change, not a new model or new free parameter, distinct from disabling h48qual entirely (which
was already tested via the conformal-veto cheap_gate and flagged as colliding with a globally-closed
quality-threshold-retuning axis; this test does NOT touch quality thresholds at all).

VAL ONLY. OOS-Q1/OOS-Q2 not opened by this script.

fresh_forward_bar_by_bar=true (same causal replay loop, only the priority-order constant differs).
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module (PRIORITY is
monkeypatched at runtime and restored). No GPU (DEVICE=cpu).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as o4  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_priority_swap_cheap_gate_20260816"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
WINDOW = "val"

G0_ODYSSEY4_VAL_WITH_GATE = {"pnl": 77.31, "mdd": -21.76, "trades": 26}
G0_ODYSSEY4_VAL_NO_GATE = {"pnl": 41.13, "mdd": -21.70, "trades": 35}


def log(msg: str) -> None:
    print(f"[priority_swap_cheap_gate] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _metrics_pair(ledger, aligned_frame):
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": "cheap_gate for swapping shared-slot priority order to zig075-first (single-bit config change, no new model/threshold).",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window": WINDOW,
        "motivating_frequency_check": {
            "opp_dir_simultaneous_signals": 0,
            "zig075_episodes_blocked_by_h48qual_pooled": "765/4844 (15.8%)",
            "note": "ad hoc check, not persisted as a script -- see docs/experiments write-up for exact per-window breakdown",
        },
    }

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()

    log("=== stage=prepare_val ===")
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(WINDOW, gate.load_all_windows(), score_by_base, threshold, OUT_DIR, device)
    mask, _ = guard._detector_mask_for_frame(aligned_frame, WINDOW, score_by_base, threshold)
    veto_components = o4._attach_veto_mask(components, mask)

    log("=== stage=G0_reproduce (PRIORITY unchanged) ===")
    assert tuple(greedy.PRIORITY) == ("h48qual", "zig075"), f"expected live PRIORITY=('h48qual','zig075'), found {greedy.PRIORITY} -- aborting, config drift"
    diag0, ledger0 = o4.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    no_gate0, with_gate0 = _metrics_pair(ledger0, aligned_frame)
    g0_ok = _close(no_gate0, G0_ODYSSEY4_VAL_NO_GATE) and _close(with_gate0, G0_ODYSSEY4_VAL_WITH_GATE)
    report["g0_reproduce"] = {"no_gate": no_gate0, "with_gate": with_gate0, "pass": g0_ok}
    log(f"  no_gate={no_gate0['pnl']:.2f}%/{no_gate0['mdd']:.2f}%/{no_gate0['trades']}  with_gate={with_gate0['pnl']:.2f}%/{with_gate0['mdd']:.2f}%/{with_gate0['trades']}  match={g0_ok}")
    if not g0_ok:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 reproduction failed -- aborting before trusting the swap."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    log("=== stage=priority_swap (zig075 first) ===")
    original_priority = tuple(greedy.PRIORITY)
    try:
        greedy.PRIORITY = ("zig075", "h48qual")
        diag_s, ledger_s = o4.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    finally:
        greedy.PRIORITY = original_priority
    no_gate_s, with_gate_s = _metrics_pair(ledger_s, aligned_frame)
    report["priority_swap"] = {"no_gate": no_gate_s, "with_gate": with_gate_s}
    log(f"  swapped  no_gate={no_gate_s['pnl']:.2f}%/{no_gate_s['mdd']:.2f}%/{no_gate_s['trades']}  with_gate={with_gate_s['pnl']:.2f}%/{with_gate_s['mdd']:.2f}%/{with_gate_s['trades']}")

    report["stage_reached"] = "done"
    report["gate_pass"] = True
    _write_report(report)
    log("stage=done")
    return 0


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
