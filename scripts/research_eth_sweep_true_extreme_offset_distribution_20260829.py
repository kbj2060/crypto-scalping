#!/usr/bin/env python3
"""Full-population check: how many bars after the nominal sweep bar does the TRUE
local low/high (the real vertex of the V) actually occur?

Searches a generous 10-bar (50 min) window after each of the 14,259 sweep events and
records the offset of the true extreme, to pick a data-driven grace period for
build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py's anchor fix (not just eyeball
the n=10 sample already inspected visually).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
SEARCH_BARS = 10


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_offsetdist_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    lows = frame["low"].to_numpy()
    highs = frame["high"].to_numpy()
    n = len(frame)

    offsets = np.full(len(labels), -1, dtype=int)
    for i, (idx, side) in enumerate(zip(labels["candidate_index"].to_numpy(), labels["side"].to_numpy())):
        end = min(idx + SEARCH_BARS + 1, n)
        if side == "downside":
            offsets[i] = int(np.argmin(lows[idx:end]))
        else:
            offsets[i] = int(np.argmax(highs[idx:end]))

    labels["true_extreme_offset"] = offsets
    print("overall offset distribution (bars after sweep bar, 0=sweep bar itself is the true extreme):")
    counts = labels["true_extreme_offset"].value_counts().sort_index()
    for offset, count in counts.items():
        print(f"  offset={offset:2d} ({offset*5:3d}min): {count:5d}  ({100*count/len(labels):.1f}%)")
    cum = counts.sort_index().cumsum() / len(labels) * 100
    print("\ncumulative %% at or below each offset:")
    for offset, pct in cum.items():
        print(f"  offset<={offset:2d} ({offset*5:3d}min): {pct:.1f}%")

    print("\nby current label:")
    print(labels.groupby("label")["true_extreme_offset"].describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
