#!/usr/bin/env python3
"""Out-of-window replication check for `oscillator_and_funding_low`, the one candidate from
research_eth_funding_crossasset_combo_signal_20260825.py that beat the live orthogonal_combo
baseline at 1h (3.996x vs 3.51x, bottom side only -- top side is untestable, funding rate has a
hard ceiling at 0.0001 in this data). Per this repo's own established discipline (see Round 5 in
eth_reversal_evidence_signal_scorecard_20260814: a single-round number isn't trusted as a
"finding" until it replicates on an independent window), re-tests on 2026-03-01..2026-07-20 --
the SAME out-of-window period Round 5 used, chosen for direct comparability, bounded by the
zigzag pivot label coverage end (tmp/zigzag_action_labels_extended_20260809/
zigzag_action_labels_2026.csv ends 2026-07-20).

Data: data/splits/year_oos/training_features_2026_rebuilt.csv (verified canonical 2026 ETH,
reference_clean_data_locations_20260823) has OHLCV+taker_buy_base needed to reproduce
orthogonal_combo_live exactly, but funding_z is recomputed from the properly-sourced
TOTAL_ETHUSDT_fundingRate_2025_2026.csv (not this file's bundled last_funding_rate column) for
methodological consistency with the original-window run.
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
from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402

DATA_PATH = ROOT / "data" / "splits" / "year_oos" / "training_features_2026_rebuilt.csv"
OOW_START, OOW_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-07-20")


def build_frame() -> pd.DataFrame:
    raw = pd.read_csv(
        DATA_PATH,
        usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"],
        parse_dates=["timestamp"],
    )
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    frame = compute_indicators(raw)
    frame = add_creative_indicators(frame)
    funding = load_funding_z()
    frame = pd.merge_asof(frame.sort_values("timestamp"), funding, left_on="timestamp", right_on="calc_time", direction="backward")
    return frame.reset_index(drop=True)


def build_signals(frame: pd.DataFrame) -> dict:
    return {
        "orthogonal_combo_live [reproduced baseline]": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["delta_z"] <= -2.0),
        "funding_extreme_low [standalone]": frame["funding_z"] <= -2.0,
        "oscillator_and_funding_low [OOW replication target]": (frame["p_fast"] <= 0.10) & (frame["p_slow"] <= 0.10) & (frame["funding_z"] <= -2.0),
    }


def main() -> None:
    frame = build_frame()
    pivots = load_zigzag_pivots()

    ts = frame["timestamp"]
    window_mask = ((ts >= OOW_START) & (ts <= OOW_END)).to_numpy()
    print(f"OOW window: {OOW_START.date()}..{OOW_END.date()}, {int(window_mask.sum())} bars")
    print(f"Funding join coverage: {frame['funding_z'].notna().sum()}/{len(frame)} bars matched")

    close = frame["close"].to_numpy()
    all_pos = np.flatnonzero(window_mask)
    side_pivots = pivots.loc[pivots["pivot_type"] == "bottom"]
    pivot_pos = frame.index[frame["timestamp"].isin(side_pivots["timestamp"])].to_numpy()
    print(f"{len(pivot_pos)} bottom zigzag pivots in the full pivot table (mask restricts to OOW bars)")

    rows = []
    for sig_name, mask in build_signals(frame).items():
        trigger_pos = np.flatnonzero(mask.fillna(False).to_numpy() & window_mask)
        for k_name, K in K_HORIZONS.items():
            stats = event_study(trigger_pos, pivot_pos, all_pos, K)
            move = excess_move(trigger_pos, pivot_pos, close, K)
            rows.append({"signal": sig_name, "horizon": k_name, **stats,
                         "excess_move_mean_pct": move["mean_pct"], "excess_move_median_pct": move["median_pct"]})
    result = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    for horizon in K_HORIZONS:
        print(f"\n-- horizon {horizon} --")
        cols = ["signal", "n_triggers", "precision", "lift", "recall", "median_lead_bars", "excess_move_mean_pct"]
        print(result[result["horizon"] == horizon][cols].to_string(index=False))

    out_dir = ROOT / "tmp" / "eth_funding_oscillator_combo_oow_20260825"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "evidence_table.csv", index=False)
    print(f"\nWrote full table to {out_dir / 'evidence_table.csv'}")
    print("\nOriginal window (2025-09-01..2026-02-17) reference: orthogonal_combo_live 3.51x, "
          "funding_extreme_low 1.42x, oscillator_and_funding_low 4.00x (n=110) @ 1h")


if __name__ == "__main__":
    main()
