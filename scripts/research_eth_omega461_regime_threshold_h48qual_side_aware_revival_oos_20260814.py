#!/usr/bin/env python3
"""Companion to research_eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814.py --
mandatory dual-window (OOS-Q1+OOS-Q2) single-touch confirmation, per this project's now-standard
multiwindow gate (scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py). The chosen VAL
configuration (h48qual SHORT=found regime map bull/bear=0.30 chop=0.35, h48qual LONG=effectively
disabled thr=1.01, zig075=global flat 0.75 untouched) is locked in and NOT retuned here -- this script
only evaluates it once on oos_q1+oos_q2 together, exactly as the multiwindow gate's pre-registered rule
requires. 2025 Q1/Q2/Q3 context-tier windows are also reported for completeness but are not part of the
pass/fail decision (same as every other candidate judged under this gate tonight).

fresh_forward_bar_by_bar=true. trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false. Does not touch trading_bot.py/omega4_6_1_live.py/runtime_config.py/.env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_regime_threshold_h48qual_side_aware_revival_20260814 as rev  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = rev.OUT_DIR
MDD_SLACK_PP = 3.0  # relaxed-gate MDD tolerance, docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md


def log(msg: str) -> None:
    print(f"[side_aware_revival_oos] {msg}", flush=True)


def evaluate_window(w: dict, thr_short: dict, thr_long: dict, *, oof: bool) -> dict:
    m_h48, no_gate, with_gate = rev.build_portfolio(w, thr_short, thr_long, oof=oof)
    return {"component_h48qual": m_h48, "no_gate": no_gate, "with_gate": with_gate}


def judge(baseline: dict, candidate: dict, *, mdd_slack_pp: float) -> dict:
    orig = (candidate["no_gate"]["pnl"] >= baseline["no_gate"]["pnl"] and candidate["no_gate"]["mdd"] >= baseline["no_gate"]["mdd"] and
            candidate["with_gate"]["pnl"] >= baseline["with_gate"]["pnl"] and candidate["with_gate"]["mdd"] >= baseline["with_gate"]["mdd"])
    relaxed = (candidate["with_gate"]["pnl"] >= baseline["with_gate"]["pnl"] and
               candidate["with_gate"]["mdd"] >= baseline["with_gate"]["mdd"] - mdd_slack_pp)
    return {"pass_original": bool(orig), "pass_relaxed": bool(relaxed)}


def main() -> int:
    prior = json.loads((OUT_DIR / "val_report.json").read_text())
    if not prior["chosen_gate_pass"]:
        raise RuntimeError("VAL gate did not pass -- OOS must not be opened")

    windows = gate.load_all_windows()
    thr_short, thr_long = rev.SHORT_MAP, rev.LONG_SHUTOFF
    baseline_map = {r: 0.50 for r in rev.base.REGIMES}

    log("stage=G0 cross-check against locked VAL numbers from the prior script's own report.json")
    val_recheck = evaluate_window(windows["val"], thr_short, thr_long, oof=True)
    g0_ok = (abs(val_recheck["no_gate"]["pnl"] - prior["chosen_no_gate"]["pnl"]) < 1e-6 and
             abs(val_recheck["with_gate"]["pnl"] - prior["chosen_with_gate"]["pnl"]) < 1e-6)
    log(f"  VAL recheck matches locked report: {g0_ok}")
    if not g0_ok:
        raise RuntimeError("G0 cross-check against locked VAL report failed -- aborting before OOS")

    result: dict = {"g0_ok": True, "short_map": thr_short, "long_map": thr_long, "windows": {}}

    log("stage=context tier (2025 Q1/Q2/Q3) -- reported only, NOT part of the pass/fail decision")
    for wname in ("2025q1", "2025q2", "2025q3"):
        w = windows[wname]
        base_r = evaluate_window(w, baseline_map, baseline_map, oof=w["oof"])
        cand_r = evaluate_window(w, thr_short, thr_long, oof=w["oof"])
        result["windows"][wname] = {"tier": "context", "baseline": base_r, "candidate": cand_r}
        log(f"  {wname} baseline no_gate={base_r['no_gate']['pnl']:+.2f}%/{base_r['no_gate']['mdd']:+.2f}% "
            f"candidate no_gate={cand_r['no_gate']['pnl']:+.2f}%/{cand_r['no_gate']['mdd']:+.2f}%")

    log("stage=OOS-Q1+OOS-Q2 single-touch (mandatory dual-window per this project's multiwindow gate)")
    for wname in ("oos_q1", "oos_q2"):
        w = windows[wname]
        base_r = evaluate_window(w, baseline_map, baseline_map, oof=w["oof"])
        cand_r = evaluate_window(w, thr_short, thr_long, oof=w["oof"])
        j = judge(base_r, cand_r, mdd_slack_pp=MDD_SLACK_PP)
        result["windows"][wname] = {"tier": "oos_confirm", "baseline": base_r, "candidate": cand_r, "judge": j}
        log(f"  {wname} baseline no_gate={base_r['no_gate']['pnl']:+.2f}%/{base_r['no_gate']['mdd']:+.2f}% "
            f"with_gate={base_r['with_gate']['pnl']:+.2f}%/{base_r['with_gate']['mdd']:+.2f}%")
        log(f"  {wname} candidate no_gate={cand_r['no_gate']['pnl']:+.2f}%/{cand_r['no_gate']['mdd']:+.2f}% "
            f"with_gate={cand_r['with_gate']['pnl']:+.2f}%/{cand_r['with_gate']['mdd']:+.2f}%  "
            f"judge={j}")

    both_pass_original = all(result["windows"][w]["judge"]["pass_original"] for w in ("oos_q1", "oos_q2"))
    both_pass_relaxed = all(result["windows"][w]["judge"]["pass_relaxed"] for w in ("oos_q1", "oos_q2"))
    per_window_original = {w: result["windows"][w]["judge"]["pass_original"] for w in ("oos_q1", "oos_q2")}
    per_window_relaxed = {w: result["windows"][w]["judge"]["pass_relaxed"] for w in ("oos_q1", "oos_q2")}

    if both_pass_original:
        final_verdict = "CONFIRMED_ORIGINAL_GATE"
    elif both_pass_relaxed:
        final_verdict = "CONFIRMED_RELAXED_GATE"
    else:
        final_verdict = "REJECTED_SIGN_MISMATCH" if any(per_window_original.values()) or any(per_window_relaxed.values()) else "REJECTED_BOTH_WINDOWS_FAIL"

    result["oos_confirm_per_window_pass_original"] = per_window_original
    result["oos_confirm_per_window_pass_relaxed"] = per_window_relaxed
    result["oos_confirm_all_pass_original"] = bool(both_pass_original)
    result["oos_confirm_all_pass_relaxed"] = bool(both_pass_relaxed)
    result["final_verdict"] = final_verdict

    log(f"stage=OOS_result per_window_original={per_window_original} per_window_relaxed={per_window_relaxed}")
    log(f"FINAL VERDICT: {final_verdict}")

    (OUT_DIR / "oos_report.json").write_text(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
