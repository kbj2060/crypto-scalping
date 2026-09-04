#!/usr/bin/env python3
"""v5 binary redesign (2026-08-30, user request after the 3-class exploration proved too noisy
to train on): fold the lessons from the giveback-ratio research back into a SINGLE, cleaner
binary split instead of 3 classes. Keeps v4's 30min/6bar window, replaces two things:

  (a) the "did it even attempt a real move" gate now requires a CLOSE (not just a wick/high-low
      touch) to reach 1.5x pre-sweep ATR within the first V_REBOUND_FAST_BARS (15min) -- the
      3-class validation chart found wick-only touches let pure noise ("실패-반납" examples that
      never really attempted a reversal, giveback>1.0") pass the old gate.
  (b) the "did it hold" check replaces the v3/v4 swept-level defense (just don't fall back
      through the old level) with giveback_ratio <= GIVEBACK_T over the full 30min window --
      the spectrum chart showed swept-level-only defense let near-failures (giveback 0.8-0.9)
      through as label=1.

Sweeps a few candidate GIVEBACK_T values and reports resulting label rate + flip count vs the
current (v4) label file, so a threshold can be picked before committing. No label file written.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
CURRENT_LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_label_v5_binary_20260830"
LOOKAHEAD_BARS = 6
FAST_BARS = 3
ATR_MULT = 1.5
CANDIDATE_T = [0.40, 0.50, 0.55, 0.60, 0.65]


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v5_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    current = pd.read_csv(CURRENT_LABEL_CSV)

    close_attempted, giveback_ratio = [], []
    for _, event in current.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:FAST_BARS]
        atr = event["atr"]  # pre-sweep, v4
        if event["side"] == "downside":
            sweep_extreme = row["low"]
            fast_close_move = fast_future["close"].max() - sweep_extreme   # v5: close, not high
            peak = future["high"].max()
            end = future["close"].iloc[-1]
            giveback = peak - end
        else:
            sweep_extreme = row["high"]
            fast_close_move = sweep_extreme - fast_future["close"].min()
            peak = future["low"].min()
            end = future["close"].iloc[-1]
            giveback = end - peak
        total_move = abs(peak - sweep_extreme)
        close_attempted.append(bool(fast_close_move >= ATR_MULT * atr))
        giveback_ratio.append(float(giveback / total_move) if total_move > 1e-12 else np.nan)

    current["close_attempted"] = close_attempted
    current["giveback_ratio_v5"] = giveback_ratio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_DIR / "events_with_v5_inputs.csv", index=False)

    n = len(current)
    n_close_attempted = int(current["close_attempted"].sum())
    print(f"total events: {n}")
    print(f"close-confirmed fast attempt (v5 gate): {n_close_attempted} ({n_close_attempted/n:.1%}) "
          f"vs old wick-based v4 gate implied by label!=nan attempts")
    print(f"current v4 label==1 rate: {current['label'].mean():.4f}")

    for t in CANDIDATE_T:
        new_label = (current["close_attempted"] & (current["giveback_ratio_v5"] <= t)).astype(int)
        flips = int((new_label != current["label"]).sum())
        flips_1to0 = int(((current["label"] == 1) & (new_label == 0)).sum())
        flips_0to1 = int(((current["label"] == 0) & (new_label == 1)).sum())
        print(f"T={t:.2f}: v5 label rate={new_label.mean():.4f}  n_label1={int(new_label.sum())}  "
              f"flips vs v4={flips} ({flips/n:.1%})  [1->0: {flips_1to0}, 0->1: {flips_0to1}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
