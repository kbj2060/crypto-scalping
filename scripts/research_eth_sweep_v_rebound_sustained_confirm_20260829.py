#!/usr/bin/env python3
"""Check the user's second catch: 3 flagged V_REBOUND panels (2024-02-19, 2026-03-11,
2024-06-04, all upside) show price generally DRIFTING UP through the whole +/-30min window,
just grazing under the swept level at the final bar -- not a real down-reversal. The
"confirmed" check (variant B) only looks at the single last bar's close, so noisy chop that
happens to land under the level at exactly bar+30 passes even with no sustained rejection.

Tests requiring the close to stay beyond the level for the LAST K bars (not just the last
one) across the full population, for K=1 (current), 2, 3, and reports both the aggregate
rate/flip-direction and whether the 3 flagged events specifically flip to 0.
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
    spec = importlib.util.spec_from_file_location("sweep_impl_sustained_20260829", IMPL_SCRIPT)
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

    label_csv = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
    prior_labels = pd.read_csv(label_csv)
    prior_labels["ts"] = pd.to_datetime(prior_labels["timestamp"])
    flagged_kst = ["2024-02-19 13:20", "2026-03-11 01:20", "2024-06-04 22:40"]
    flagged_idx = set()
    for kst_str in flagged_kst:
        target_utc = pd.Timestamp(kst_str, tz="UTC") - pd.Timedelta(hours=9)
        match = prior_labels.loc[(prior_labels["ts"] - target_utc).abs() < pd.Timedelta(minutes=1)]
        if len(match):
            flagged_idx.add(int(match.iloc[0]["candidate_index"]))
    print("resolved flagged candidate_index values:", flagged_idx)

    rows = []
    for index in range(impl.SWEEP_LOOKBACK_BARS, n - LOOKAHEAD_BARS):
        atr = frame["atr"].iat[index]
        if not np.isfinite(atr) or atr <= 0:
            continue
        sweep_low_level = frame["sweep_level_low"].iat[index]
        sweep_high_level = frame["sweep_level_high"].iat[index]
        future_close = closes[index + 1: index + LOOKAHEAD_BARS + 1]
        future_high = highs[index + 1: index + LOOKAHEAD_BARS + 1]
        future_low = lows[index + 1: index + LOOKAHEAD_BARS + 1]

        if np.isfinite(sweep_low_level) and lows[index] < sweep_low_level and closes[index] > sweep_low_level:
            move = future_high.max() - lows[index]
            rows.append(_row(index, "downside", move, atr, future_close, sweep_low_level, above=True))
        if np.isfinite(sweep_high_level) and highs[index] > sweep_high_level and closes[index] < sweep_high_level:
            move = highs[index] - future_low.min()
            rows.append(_row(index, "upside", move, atr, future_close, sweep_high_level, above=False))

    table = pd.DataFrame(rows)
    print(f"\ntotal events: {len(table)}")
    for k in (1, 2, 3):
        col = f"label_k{k}"
        rate = table[col].mean()
        print(f"K={k}: V_REBOUND={int(table[col].sum())} ({100*rate:.1f}%)  NO_V_REBOUND={len(table)-int(table[col].sum())} ({100*(1-rate):.1f}%)")
    for k in (2, 3):
        flips_to_0 = int(((table["label_k1"] == 1) & (table[f"label_k{k}"] == 0)).sum())
        flips_to_1 = int(((table["label_k1"] == 0) & (table[f"label_k{k}"] == 1)).sum())
        print(f"K=1 vs K={k}: 1->0 flips={flips_to_0}, 0->1 flips={flips_to_1}")

    print("\nflagged events (candidate_index, side, K1/K2/K3 labels):")
    flagged_rows = table[table["candidate_index"].isin(flagged_idx)]
    print(flagged_rows[["candidate_index", "side", "move", "atr", "label_k1", "label_k2", "label_k3"]].to_string(index=False))
    return 0


def _row(index, side, move, atr, future_close, level, above: bool):
    row = {"candidate_index": index, "side": side, "move": float(move), "atr": float(atr)}
    for k in (1, 2, 3):
        tail = future_close[-k:]
        confirmed = bool(np.all(tail > level)) if above else bool(np.all(tail < level))
        row[f"label_k{k}"] = int(move >= V_REBOUND_ATR_MULT * atr and confirmed)
    return row


if __name__ == "__main__":
    raise SystemExit(main())
