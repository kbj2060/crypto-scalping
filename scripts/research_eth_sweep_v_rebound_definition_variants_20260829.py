#!/usr/bin/env python3
"""Compare label-definition variants across the FULL 14,259-event population, not just the
n=10 visual sample, to pick a principled fix for the anchor/confirmation issue the user
spotted (many visually-V-shaped NO_V_REBOUND panels).

Variants:
  A (current):  move = future_high.max() - sweep_bar.low   confirmed = final_close > sweep_bar.close
  B (confirm vs swept level, no vertex search): same move,  confirmed = final_close > sweep_level
  C (free vertex search within the 30min window, no confirmation change): anchor move+confirm
    to wherever the true low/high falls inside the SAME 30-minute window (tests the tautology
    concern -- searching for the extreme and then checking "is it higher after its own minimum"
    is close to circular).
  D (B + C combined)

All four keep the same 1.0x ATR(14) magnitude threshold and 30-minute (6-bar) total window.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LOOKAHEAD_BARS = 6
V_REBOUND_ATR_MULT = 1.0


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_variants_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.add_causal_columns(impl.load_5m(SOURCE))
    lows = frame["low"].to_numpy()
    highs = frame["high"].to_numpy()
    closes = frame["close"].to_numpy()
    n = len(frame)

    rows = []
    for index in range(impl.SWEEP_LOOKBACK_BARS, n - LOOKAHEAD_BARS):
        atr = frame["atr"].iat[index]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sweep_low_level = frame["sweep_level_low"].iat[index]
        sweep_high_level = frame["sweep_level_high"].iat[index]

        if np.isfinite(sweep_low_level) and lows[index] < sweep_low_level and closes[index] > sweep_low_level:
            _emit(rows, "downside", index, atr, sweep_low_level, lows, highs, closes)
        if np.isfinite(sweep_high_level) and highs[index] > sweep_high_level and closes[index] < sweep_high_level:
            _emit(rows, "upside", index, atr, sweep_high_level, lows, highs, closes)

    table = pd.DataFrame(rows)
    print(f"total events: {len(table)}\n")
    for variant in ["A", "B", "C", "D"]:
        rate = table[f"label_{variant}"].mean()
        n1 = int(table[f"label_{variant}"].sum())
        print(f"variant {variant}: V_REBOUND={n1} ({100*rate:.1f}%)  NO_V_REBOUND={len(table)-n1} ({100*(1-rate):.1f}%)")

    print("\nagreement with current (A):")
    for variant in ["B", "C", "D"]:
        agree = (table["label_A"] == table[f"label_{variant}"]).mean()
        flips_to_1 = ((table["label_A"] == 0) & (table[f"label_{variant}"] == 1)).sum()
        flips_to_0 = ((table["label_A"] == 1) & (table[f"label_{variant}"] == 0)).sum()
        print(f"  A vs {variant}: {100*agree:.1f}% same label, 0->1 flips={flips_to_1}, 1->0 flips={flips_to_0}")
    return 0


def _emit(rows, side, index, atr, level, lows, highs, closes, lookahead=LOOKAHEAD_BARS):
    future_low = lows[index + 1: index + lookahead + 1]
    future_high = highs[index + 1: index + lookahead + 1]
    future_close = closes[index + 1: index + lookahead + 1]
    sweep_extreme = lows[index] if side == "downside" else highs[index]
    sweep_close = closes[index]
    final_close = future_close[-1]

    # variant A: current
    if side == "downside":
        move_a = future_high.max() - sweep_extreme
        confirmed_a = final_close > sweep_close
        confirmed_b = final_close > level
    else:
        move_a = sweep_extreme - future_low.min()
        confirmed_a = final_close < sweep_close
        confirmed_b = final_close < level
    label_a = int(move_a >= V_REBOUND_ATR_MULT * atr and confirmed_a)
    label_b = int(move_a >= V_REBOUND_ATR_MULT * atr and confirmed_b)

    # variant C: free vertex search within the same 6-bar window (sweep bar + 6 future = 7 candidates)
    window_low = np.concatenate(([lows[index]], future_low))
    window_high = np.concatenate(([highs[index]], future_high))
    window_close = np.concatenate(([closes[index]], future_close))
    if side == "downside":
        true_offset = int(np.argmin(window_low))
        true_extreme = window_low[true_offset]
        remaining_high = window_high[true_offset + 1:]
        move_c = (remaining_high.max() - true_extreme) if len(remaining_high) else 0.0
        confirmed_c = bool(window_close[-1] > window_close[true_offset]) if len(remaining_high) else False
        confirmed_d = bool(window_close[-1] > level) if len(remaining_high) else False
    else:
        true_offset = int(np.argmax(window_high))
        true_extreme = window_high[true_offset]
        remaining_low = window_low[true_offset + 1:]
        move_c = (true_extreme - remaining_low.min()) if len(remaining_low) else 0.0
        confirmed_c = bool(window_close[-1] < window_close[true_offset]) if len(remaining_low) else False
        confirmed_d = bool(window_close[-1] < level) if len(remaining_low) else False
    label_c = int(move_c >= V_REBOUND_ATR_MULT * atr and confirmed_c)
    label_d = int(move_c >= V_REBOUND_ATR_MULT * atr and confirmed_d)

    rows.append({
        "candidate_index": index, "side": side,
        "label_A": label_a, "label_B": label_b, "label_C": label_c, "label_D": label_d,
    })


if __name__ == "__main__":
    raise SystemExit(main())
