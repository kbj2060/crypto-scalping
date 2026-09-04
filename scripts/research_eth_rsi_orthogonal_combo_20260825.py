#!/usr/bin/env python3
"""Follow-up to research_eth_rsi_stochrsi_evidence_signal_20260825.py: user asked "wouldn't RSI
be better combined with another indicator?" This repo already answered the GENERAL version of
that question on 2026-08-14 (eth_reversal_evidence_signal_scorecard_20260814 memory): combining
SAME-family price-position oscillators barely helps (%R+SlowK: 2.28x vs 2.08x/2.09x alone, ~10%
gain), while combining an oscillator with a genuinely different info source (order flow) gave the
single best signal in the whole 22-signal scorecard, orthogonal_combo (2.28x -> 3.51x, +54%),
now live on the dashboard. This script asks the RSI-specific version: given RSI overlaps 84.6%
with the oscillator leg already inside orthogonal_combo (measured in the prior script), does
swapping literal RSI into that exact combo formula change anything, or is it the same information
wearing a different name?

Same methodology (event_study/excess_move/load_zigzag_pivots, VAL+OOS window), reused unmodified.
Not a re-derivation of orthogonal_combo -- reproduces it verbatim as `orthogonal_combo_live` for a
same-run, apples-to-apples baseline, then tests RSI-substituted and RSI-added variants against it.
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
    OOS_END,
    event_study,
    excess_move,
    load_zigzag_pivots,
)
from analyze_eth_creative_reversal_evidence_signals_20260814 import (  # noqa: E402
    add_creative_indicators,
    load_frame_with_orderflow,
)
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    OOS_START,
    VAL_END,
    VAL_START,
    compute_indicators,
)
from research_eth_rsi_stochrsi_evidence_signal_20260825 import RSI_PERIOD, _rsi  # noqa: E402


def build_signals(frame: pd.DataFrame, side: str) -> dict:
    if side == "bottom":
        return {
            "orthogonal_combo_live [reproduced baseline] (p_fast<=.10 AND p_slow<=.10 AND delta_z<=-2)":
                (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
            "rsi20_and_takerdelta (RSI<=20 AND delta_z<=-2) [RSI swapped in for %R/%K leg]":
                (frame["rsi"] <= 20) & (frame["delta_z"] <= -2.0),
            "rsi30_and_takerdelta (RSI<=30 AND delta_z<=-2) [looser RSI, more sample]":
                (frame["rsi"] <= 30) & (frame["delta_z"] <= -2.0),
            "rsi_and_oscillator_and_takerdelta (RSI<=30 AND p_fast<=.10 AND p_slow<=.10 AND delta_z<=-2) [RSI ADDED as 3rd leg]":
                (frame["rsi"] <= 30) & (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
            "rsi30_and_volumewick (RSI<=30 AND vol_z>=2 AND lower_wick>=.5) [RSI + different orthogonal leg]":
                (frame["rsi"] <= 30) & (frame["vol_z"] >= 2.0) & (frame["lower_wick_ratio"] >= 0.5),
        }
    return {
        "orthogonal_combo_live [reproduced baseline] (p_fast>=.90 AND p_slow>=.90 AND delta_z>=2)":
            (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
        "rsi80_and_takerdelta (RSI>=80 AND delta_z>=2) [RSI swapped in for %R/%K leg]":
            (frame["rsi"] >= 80) & (frame["delta_z"] >= 2.0),
        "rsi70_and_takerdelta (RSI>=70 AND delta_z>=2) [looser RSI, more sample]":
            (frame["rsi"] >= 70) & (frame["delta_z"] >= 2.0),
        "rsi_and_oscillator_and_takerdelta (RSI>=70 AND p_fast>=.90 AND p_slow>=.90 AND delta_z>=2) [RSI ADDED as 3rd leg]":
            (frame["rsi"] >= 70) & (frame["p_fast"] >= 0.90) & (frame["p_slow"] >= 0.90) & (frame["delta_z"] >= 2.0),
        "rsi70_and_volumewick (RSI>=70 AND vol_z>=2 AND upper_wick>=.5) [RSI + different orthogonal leg]":
            (frame["rsi"] >= 70) & (frame["vol_z"] >= 2.0) & (frame["upper_wick_ratio"] >= 0.5),
    }


def run_side(frame: pd.DataFrame, window_mask: np.ndarray, pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    side_pivots = pivots.loc[pivots["pivot_type"] == side]
    pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()

    rows = []
    for sig_name, mask in build_signals(frame, side).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"side": side, "signal": sig_name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"], "excess_move_median_pct": move["median_pct"]})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame_with_orderflow()
    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)
    frame["rsi"] = _rsi(frame["close"], RSI_PERIOD)
    frame = frame.reset_index(drop=True)
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = (((ts >= VAL_START) & (ts <= VAL_END)) | ((ts >= OOS_START) & (ts <= OOS_END))).to_numpy()
    print(f"Study window: VAL {VAL_START.date()}..{VAL_END.date()} + OOS {OOS_START.date()}..{OOS_END.date()}, "
          f"{int(window_mask.sum())} bars, {len(pivots)} zigzag pivots")

    all_rows = pd.concat([run_side(frame, window_mask, pivots, "bottom"), run_side(frame, window_mask, pivots, "top")], ignore_index=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 100)
    for side in ("bottom", "top"):
        print(f"\n=== {side.upper()} evidence ===")
        sub = all_rows[all_rows["side"] == side]
        for horizon in K_HORIZONS:
            print(f"\n-- horizon {horizon} --")
            cols = ["signal", "n_triggers", "precision", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
            print(sub[sub["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_rsi_orthogonal_combo_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")


if __name__ == "__main__":
    main()
