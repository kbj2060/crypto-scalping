#!/usr/bin/env python3
"""v6 binary redesign (2026-08-30, user: "still not cleanly separated -- split strictly into
V자반등(sustained V) vs 지지/횡보(everything else), and widen the fast-move window 15min->30min").

Two changes from v5:
  (a) fast/speed window widens from 3 bars(15min) to 6 bars(30min); the full observation window
      widens correspondingly from 6 bars(30min) to 12 bars(60min) -- same 50/50 ratio as the
      original v3/v4 design (fast window = exactly half the full window), just scaled 2x.
  (b) the binary cut point moves from the HIGH end of giveback_ratio (v5's T=0.55, separating
      "gave back a lot" from "gave back too much") to the LOW end (T_SUSTAIN, separating "kept
      extending" from everything else) -- this is the boundary the spectrum chart showed was
      actually crisp, unlike the high-end boundary which stayed fuzzy no matter where drawn.

label=1 (V자반등, sustained): close-confirmed 1.5x-ATR move within the first 30min AND
  giveback_ratio (peak-vs-end over the full 60min) <= GIVEBACK_T_SUSTAIN.
label=0 (지지/횡보, everything else): never attempted, OR attempted but giveback_ratio >
  GIVEBACK_T_SUSTAIN -- this merges what used to be "plateau" and "failure" into one class,
  since neither is a clean V by eye.

Sweeps a few candidate T_SUSTAIN values, reports population sizes -- no label file finalized
here, matches this line of work's established chart-before-commit practice.
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
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830"
LOOKAHEAD_BARS = 12   # 60min (was 6/30min)
FAST_BARS = 6         # 30min (was 3/15min) -- exactly half of LOOKAHEAD_BARS, same ratio as v3/v4
ATR_MULT = 1.5
CANDIDATE_T = [0.10, 0.15, 0.20, 0.25, 0.30]


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_v6_20260830", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    current = pd.read_csv(CURRENT_LABEL_CSV)

    n_before = len(current)
    current = current[current["candidate_index"] + LOOKAHEAD_BARS < len(frame)].reset_index(drop=True)
    print(f"events with full 60min of future data: {len(current)}/{n_before}")

    close_attempted, giveback_ratio = [], []
    for _, event in current.iterrows():
        idx = int(event["candidate_index"])
        row = frame.iloc[idx]
        future = frame.iloc[idx + 1: idx + LOOKAHEAD_BARS + 1]
        fast_future = future.iloc[:FAST_BARS]
        atr = event["atr"]  # pre-sweep, v4 -- unaffected by window size, reused as-is
        if event["side"] == "downside":
            sweep_extreme = row["low"]
            fast_close_move = fast_future["close"].max() - sweep_extreme
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
    current["giveback_ratio_v6"] = giveback_ratio

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_DIR / "events_with_v6_inputs.csv", index=False)

    n = len(current)
    n_attempted = int(current["close_attempted"].sum())
    print(f"close-confirmed 30min attempt: {n_attempted} ({n_attempted/n:.1%})")

    for t in CANDIDATE_T:
        new_label = (current["close_attempted"] & (current["giveback_ratio_v6"] <= t)).astype(int)
        print(f"T_sustain={t:.2f}: V자반등(1) rate={new_label.mean():.4f}  n={int(new_label.sum())}  "
              f"지지/횡보(0) n={int((1-new_label).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
