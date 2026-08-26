#!/usr/bin/env python3
"""Does OR-combining funding_oscillator_combo's confirming leg into orthogonal_combo (instead of
keeping them as two separate evidence-signal chips) hold up? User's motivation: funding_oscillator_
combo rarely fires (live: bottom_last_fired_ts/top_last_fired_ts both null across the entire
~5.2-day FETCH_LIMIT window as of 2026-08-26) despite having beaten orthogonal_combo's own lift at
1h in two independent windows (research_eth_funding_crossasset_combo_signal_20260825.py /
research_eth_funding_oscillator_combo_oow_20260825.py) -- so its edge rarely gets to matter to
anyone looking at the dashboard.

This does NOT redefine orthogonal_combo's live formula (that would silently invalidate every
existing backtest/scorecard claim keyed to its exact p_fast/p_slow+delta_z definition -- the same
"reused name, changed logic" trap this repo has been burned by before, e.g. h48qual_label_mismatch_
discovered). Instead it tests a THIRD, brand-new candidate --
"oscillator extreme AND (delta_z beyond +-2 OR funding_z beyond +-2)" -- side by side with the two
existing live signals it would replace/absorb, on the SAME two windows already used to validate
funding_oscillator_combo, for direct comparability:
  - original: VAL 2025-09-01..2025-12-31 + OOS 2026-01-01..2026-02-17 (research_eth_funding_
    crossasset_combo_signal_20260825.build_frame(), reused verbatim/unmodified)
  - OOW replication: 2026-03-01..2026-07-20 (research_eth_funding_oscillator_combo_oow_20260825.
    build_frame(), reused verbatim/unmodified)

Also reports trigger-gap statistics (median/max hours between consecutive fires) per signal per
window -- the direct, systematic version of "hasn't fired in 2 days," instead of relying on one
anecdote.

This is a retrospective lift/event-study diagnostic (same event_study/excess_move methodology as
every sibling script in this family), NOT a cost-gate backtest and NOT a promotion claim -- see
those scripts' own docstrings for why (zigzag-pivot lift, not fresh-forward TP/SL PnL). Matches
this repo's dashboard display bar (IC/statistical informativeness), not the live-trading economic
gate (feedback_dashboard_indicators_ic_bar_not_pnl_bar memory) -- this script's job is only to
check whether the union's IC-level lift survives combining, not to promote anything to live trading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import (  # noqa: E402
    K_HORIZONS,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
import research_eth_funding_crossasset_combo_signal_20260825 as orig_mod  # noqa: E402
import research_eth_funding_oscillator_combo_oow_20260825 as oow_mod  # noqa: E402


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        osc = (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10)
        return {
            "orthogonal_combo_live [delta_z only, current dashboard chip]": osc & (frame["delta_z"] <= -2.0),
            "funding_oscillator_combo_live [funding_z only, current dashboard chip]": osc & (frame["funding_z"] <= -2.0),
            "UNION_candidate (osc AND (delta_z<=-2 OR funding_z<=-2))": osc & ((frame["delta_z"] <= -2.0) | (frame["funding_z"] <= -2.0)),
        }
    osc = (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90)
    return {
        "orthogonal_combo_live [delta_z only, current dashboard chip]": osc & (frame["delta_z"] >= 2.0),
        "funding_oscillator_combo_live [funding_z only, current dashboard chip]": osc & (frame["funding_z"] >= 2.0),
        "UNION_candidate (osc AND (delta_z>=2 OR funding_z>=2))": osc & ((frame["delta_z"] >= 2.0) | (frame["funding_z"] >= 2.0)),
    }


def trigger_gap_stats(trigger_pos: np.ndarray, bar_minutes: int = 5) -> dict:
    """Median/max hours between consecutive fires -- the systematic version of "hasn't fired in N days"."""
    if len(trigger_pos) < 2:
        return {"median_gap_hours": float("nan"), "max_gap_hours": float("nan")}
    gaps_bars = np.diff(np.sort(trigger_pos))
    gaps_hours = gaps_bars * bar_minutes / 60.0
    return {"median_gap_hours": float(np.median(gaps_hours)), "max_gap_hours": float(np.max(gaps_hours))}


def run_window(window_name: str, frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    rows = []
    for side in ("bottom", "top"):
        side_pivots = pivots.loc[pivots["pivot_type"] == side]
        pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
        for sig_name, mask in build_signals(frame, side).items():
            trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
            gaps = trigger_gap_stats(trigger_pos)
            for k_name, K in K_HORIZONS.items():
                stats = event_study(trigger_pos, pivot_pos, all_pos, K)
                move = excess_move(trigger_pos, pivot_pos, close, K)
                rows.append({
                    "window": window_name, "side": side, "signal": sig_name, "horizon": k_name,
                    **stats, "excess_move_mean_pct": move["mean_pct"],
                    "median_gap_hours": gaps["median_gap_hours"], "max_gap_hours": gaps["max_gap_hours"],
                })
    return pd.DataFrame(rows)


def main() -> None:
    pivots = load_zigzag_pivots()

    orig_frame = orig_mod.build_frame()
    orig_ts = orig_frame["timestamp"]
    orig_mask = (((orig_ts >= orig_mod.VAL_START) & (orig_ts <= orig_mod.VAL_END)) |
                 ((orig_ts >= orig_mod.OOS_START) & (orig_ts <= orig_mod.OOS_END))).to_numpy()
    print(f"[original] VAL {orig_mod.VAL_START.date()}..{orig_mod.VAL_END.date()} + "
          f"OOS {orig_mod.OOS_START.date()}..{orig_mod.OOS_END.date()}, {int(orig_mask.sum())} bars")

    oow_frame = oow_mod.build_frame()
    oow_ts = oow_frame["timestamp"]
    oow_mask = ((oow_ts >= oow_mod.OOW_START) & (oow_ts <= oow_mod.OOW_END)).to_numpy()
    print(f"[oow] {oow_mod.OOW_START.date()}..{oow_mod.OOW_END.date()}, {int(oow_mask.sum())} bars")

    result = pd.concat([
        run_window("original_val_oos", orig_frame, orig_mask, pivots),
        run_window("oow_replication", oow_frame, oow_mask, pivots),
    ], ignore_index=True)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 70)
    for window_name in ("original_val_oos", "oow_replication"):
        for side in ("bottom", "top"):
            print(f"\n=== {window_name} / {side.upper()} ===")
            sub = result[(result["window"] == window_name) & (result["side"] == side) & (result["horizon"] == "K12_1h")]
            cols = ["signal", "n_triggers", "precision", "lift", "recall", "median_lead_bars",
                    "excess_move_mean_pct", "median_gap_hours", "max_gap_hours"]
            print(sub[cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_funding_oscillator_union_combo_20260827"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table (all horizons) to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
