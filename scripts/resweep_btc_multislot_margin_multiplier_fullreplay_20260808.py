"""Re-sweep the BTC multi-slot MARGIN MULTIPLIER by FULL CAUSAL REPLAY (2026-08-08).

WHY THIS EXISTS -- the provenance bug.
The N=3 shadow's 1.5x margin multiplier was adopted on 2026-08-07 evening by a sweep run
"on the N=3 gated ledgers", i.e. by RESCALING ledger trade returns per multiplier.  That is
invalid for this stack: `margin_fraction` is an INPUT to the exit head (notional / leverage /
exposure sit in pos_values), so changing the multiplier changes the exits and therefore the
ledger itself.  A rescaled ledger answers "what if the same trades were bigger", which is not
the question.  The live loop applies the multiplier BEFORE the exit head, so only a full replay
matches live.

Symptom that exposed it: the recorded adoption figure is OOS gated +19.98% / -10.40%, but a full
causal replay of the identical config gives +25.30% / -10.77% (MDD within 0.37pp, PnL off 5.3pp).
That single point was already re-measured; this script re-measures the WHOLE sweep, because the
*selection* of 1.5x rests on the same broken method as the recorded number, and no script for the
original sweep was ever committed -- there was no reproducible path at all.

SCOPE AND HONESTY.
This is a MEASUREMENT CORRECTION of an already-adopted config, not a new selection and not a new
OOS read in the "spend a fresh window" sense -- OOS for this config has already been seen and
reported.  The original pre-registered rules are re-applied verbatim to the corrected numbers:

  VAL selection : among multipliers with VAL gated MDD >= -8.0%, take the highest VAL gated PnL
  OOS adoption  : gated PnL >= +19.7  AND  gated MDD >= -12.4  AND  worst gated quarter >= -4.0

Two outcomes are possible and BOTH are reported plainly:
  (a) VAL still selects 1.5x and the OOS gates still pass -> the adopted config is unchanged and
      only the recorded numbers were wrong.
  (b) VAL selects a different multiplier -> the adopted config's selection basis was wrong. That
      is escalated, NOT silently swapped: swapping the live shadow's multiplier on the strength of
      a corrected sweep whose OOS has already been read would be re-tuning on a spent window.

The multiplier grid is RECONSTRUCTED ({1.0, 1.25, 1.5, 1.75, 2.0}) because the original ad-hoc
sweep left no script; 1.0 (no multiplier) and 1.5 (the adopted value) are the two anchors that
matter and both are in it.

Built-in equivalence check: the m=1.5 cell must reproduce the already-published full-replay
figures (+25.30% / -10.77%), which also proves the `prepare()` extraction in
eval_btc_multislot_shadow_with_regime_sizing_20260808.py is behaviour-preserving.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
from eval_btc_multislot_shadow_with_regime_sizing_20260808 import (  # noqa: E402
    COST_MULT, EXIT_THRESHOLD, N_SLOTS, prepare,
)
from research_btc_swingtransition_multislot_20260807 import _replay_multislot  # noqa: E402
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import _compound_metrics, _gate  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_multislot_margin_resweep_20260808"
MULTS = [1.0, 1.25, 1.5, 1.75, 2.0]
ADOPTED_MULT = 1.5
VAL_MDD_BAR = -8.0
OOS_GATES = {"pnl": 19.7, "mdd": -12.4, "worst_quarter": -4.0}
FULLREPLAY_EXPECTED_AT_ADOPTED = {"oos_pnl": 25.30, "oos_mdd": -10.77}
RECORDED_BY_LEDGER_RESCALE = {"oos_pnl": 19.98, "oos_mdd": -10.40}
TOL = 0.35


def quarters(gated: pd.DataFrame) -> dict[str, float]:
    if not len(gated):
        return {}
    g = gated.copy()
    g["q"] = pd.to_datetime(g["entry_timestamp"]).dt.to_period("Q")
    return {str(q): round(float(((1 + s["trade_return"]).prod() - 1) * 100), 2)
            for q, s in g.groupby("q")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    device = parent._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # prepared ONCE at multiplier 1.0; the sweep scales the margin vector per cell, which is
    # identical to preparing per multiplier (margin_mult enters as a plain scalar factor) but
    # avoids reloading the stack five times.
    prep = prepare(device, margin_mult=1.0)
    data, loaded, fee, slip = prep["data"], prep["loaded"], prep["fee"], prep["slip"]

    cells: dict[str, dict] = {}
    for m in MULTS:
        cell: dict[str, dict] = {}
        for split in ("validation", "oos"):
            d = data[split]
            led = _replay_multislot(d["raw"], d["x"], d["dec"], loaded, n_slots=N_SLOTS,
                                    risk_margin_fraction=d["margin"] * m, risk_leverage=d["leverage"],
                                    exit_threshold=EXIT_THRESHOLD, fee=fee, slip=slip,
                                    cost_mult=COST_MULT, device=device)
            led.to_csv(OUT_DIR / f"{split}_ledger_n{N_SLOTS}_m{m:g}.csv", index=False)
            g = _gate(led, d["ou"])
            cell[split] = {"ungated": _compound_metrics(led), "gated": _compound_metrics(g)}
            if split == "oos":
                cell["oos_quarters"] = quarters(g)
        cells[f"m{m:g}"] = cell
        print(json.dumps({f"m{m:g}": {"val_gated": cell["validation"]["gated"],
                                      "oos_gated": cell["oos"]["gated"],
                                      "oos_quarters": cell["oos_quarters"]}}), flush=True)

    # ---- equivalence check against the already-published full replay at the adopted multiplier
    adopted = cells[f"m{ADOPTED_MULT:g}"]["oos"]["gated"]
    equiv_ok = bool(abs(adopted["pnl"] - FULLREPLAY_EXPECTED_AT_ADOPTED["oos_pnl"]) <= TOL
                    and abs(adopted["mdd"] - FULLREPLAY_EXPECTED_AT_ADOPTED["oos_mdd"]) <= TOL)

    # ---- re-apply the ORIGINAL pre-registered rules to the corrected numbers
    eligible = {k: v for k, v in cells.items() if v["validation"]["gated"]["mdd"] >= VAL_MDD_BAR}
    val_pick = max(eligible, key=lambda k: eligible[k]["validation"]["gated"]["pnl"]) if eligible else None
    verdict: dict = {"val_eligible": sorted(eligible), "val_selected": val_pick,
                     "adopted_in_live_shadow": f"m{ADOPTED_MULT:g}",
                     "val_selection_unchanged": val_pick == f"m{ADOPTED_MULT:g}"}
    if val_pick is not None:
        og = cells[val_pick]["oos"]["gated"]
        wq = min(cells[val_pick]["oos_quarters"].values()) if cells[val_pick]["oos_quarters"] else None
        verdict["oos_gates_on_val_selection"] = {
            "pnl": {"bar": OOS_GATES["pnl"], "value": round(og["pnl"], 2), "pass": og["pnl"] >= OOS_GATES["pnl"]},
            "mdd": {"bar": OOS_GATES["mdd"], "value": round(og["mdd"], 2), "pass": og["mdd"] >= OOS_GATES["mdd"]},
            "worst_quarter": {"bar": OOS_GATES["worst_quarter"], "value": wq,
                              "pass": wq is not None and wq >= OOS_GATES["worst_quarter"]},
        }
        verdict["oos_gates_all_pass"] = all(g["pass"] for g in verdict["oos_gates_on_val_selection"].values())

    out = {
        "purpose": "provenance correction: re-measure the margin-multiplier sweep by full causal "
                   "replay instead of rescaling ledger returns (margin feeds the exit head)",
        "config": {"n_slots": N_SLOTS, "cost_mult": COST_MULT, "exit_threshold": EXIT_THRESHOLD,
                   "mults": MULTS, "grid_reconstructed": True,
                   "grid_note": "the original ad-hoc sweep left no script; 1.0 and the adopted 1.5 are both included"},
        "rules_reapplied_verbatim": {"val_selection": f"VAL gated MDD >= {VAL_MDD_BAR}, then max VAL gated PnL",
                                     "oos_adoption": OOS_GATES},
        "provenance": {"recorded_by_ledger_rescale": RECORDED_BY_LEDGER_RESCALE,
                       "full_replay_at_adopted_mult": {"pnl": round(adopted["pnl"], 2),
                                                       "mdd": round(adopted["mdd"], 2)},
                       "equivalence_with_published_full_replay_ok": equiv_ok},
        "cells": cells, "verdict": verdict,
        "scope": "measurement correction of an already-adopted config; NOT a new selection. If the "
                 "VAL selection changed, escalate — do not swap the live multiplier on a corrected "
                 "sweep whose OOS window has already been read.",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"provenance": out["provenance"], "verdict": verdict}, indent=2), flush=True)
    print(f"wrote {OUT_DIR / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
