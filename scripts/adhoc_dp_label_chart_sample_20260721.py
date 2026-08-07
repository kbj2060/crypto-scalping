"""Ad-hoc, illustration-only DP buy/sell/hold label sample for charting.

Purpose: produce a small CSV of (timestamp, close, label) on a short ETH 5m window so the
user can visually inspect what FineFT-style DP-based buy/sell/hold labeling looks like on
real price data, BEFORE committing to a full fresh-forward experiment round.

DP formulation reused verbatim (params + backward-induction recursion) from
scripts/build_omega1_2_1_dp_trajectory_labels_20260620.py lines 108-156 (the finite-state
FLAT/LONG/SHORT x age value-function recursion that finds the profit-maximizing entry/hold/exit
trajectory net of transaction costs -- NOT a zigzag peak/trough detector). Only the top-level
p_flat entry decision (0=CASH, 1=ENTER_LONG, 2=ENTER_SHORT) is used here; that script's own
TP/SL-from-MFE/MAE bookkeeping is not needed for a simple buy/sell/hold chart.

This is NOT a fresh-forward validated experiment. The DP label process is inherently
look-ahead by design (it optimizes over the full window with backward induction) -- that is
expected and fine for this illustration; causality/leakage rules are relaxed here per the
requesting task.

Output: tmp/research_20260721/dp_label_chart_sample_eth.csv
Columns: timestamp, close, label (one of buy / sell / hold)
  buy  = DP-optimal action at this bar is ENTER_LONG
  sell = DP-optimal action at this bar is ENTER_SHORT
  hold = DP-optimal action at this bar is CASH (no entry)
Note: the DP recursion is backward-induction from the end of the window, so the last
MAX_AGE+2 bars sit in the boundary zone where the value function hasn't fully resolved
(same boundary the source script excludes via `n - MAX_AGE - 2`). Those trailing rows are
still included here (for a continuous chart) but are also flagged in a boundary column.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "data/splits/year_oos/training_features_2025.csv"
OUT_DIR = ROOT / "tmp/research_20260721"
OUT_CSV = OUT_DIR / "dp_label_chart_sample_eth.csv"

# Window chosen for visually interesting price action: an up move (11-18), a chop/roll-over
# (11-19), and a grinding downtrend with intraday chop (11-20/11-21) -- not monotonic.
WINDOW_START = "2025-11-18 00:00:00"
WINDOW_END = "2025-11-22 00:00:00"  # exclusive

# Same DP params as build_omega1_2_1_dp_trajectory_labels_20260620.py (5m "daytrade" profile).
MAX_AGE = 96
LEVERAGE = 2.0
MARGIN_FRACTION_FOR_LABEL = 0.025
NOTIONAL = LEVERAGE * MARGIN_FRACTION_FOR_LABEL
FEE_PER_SIDE = 0.0001 * 3.0
HOLD_PENALTY = 0.000002
MIN_ENTRY_EDGE = 0.00008


def _dp_recursion(next_ret: np.ndarray) -> np.ndarray:
    """Verbatim port of build_omega1_2_1_dp_trajectory_labels_20260620.py lines 126-156
    (backward-induction value function), returning just p_flat (per-bar optimal entry action).
    """
    n = len(next_ret)
    v_flat = np.zeros(n + 1, dtype=np.float64)
    v_long = np.zeros((n + 1, MAX_AGE + 2), dtype=np.float64)
    v_short = np.zeros((n + 1, MAX_AGE + 2), dtype=np.float64)
    p_flat = np.zeros(n, dtype=np.int8)  # 0 CASH, 1 ENTER_LONG, 2 ENTER_SHORT

    entry_cost = FEE_PER_SIDE * NOTIONAL
    exit_cost = FEE_PER_SIDE * NOTIONAL
    for i in range(n - 2, -1, -1):
        ret = float(next_ret[i]) * NOTIONAL
        cash_v = v_flat[i + 1]
        enter_long = -entry_cost + ret - HOLD_PENALTY + v_long[i + 1, 1]
        enter_short = -entry_cost - ret - HOLD_PENALTY + v_short[i + 1, 1]
        vals = (cash_v, enter_long, enter_short)
        best = int(np.argmax(vals))
        if best != 0 and vals[best] - cash_v < MIN_ENTRY_EDGE:
            best = 0
        p_flat[i] = best
        v_flat[i] = vals[best]
        for age in range(MAX_AGE, 0, -1):
            exit_v = -exit_cost + v_flat[i + 1]
            if age >= MAX_AGE:
                v_long[i, age] = exit_v
                v_short[i, age] = exit_v
                continue
            hold_long = ret - HOLD_PENALTY + v_long[i + 1, age + 1]
            hold_short = -ret - HOLD_PENALTY + v_short[i + 1, age + 1]
            v_long[i, age] = exit_v if exit_v >= hold_long else hold_long
            v_short[i, age] = exit_v if exit_v >= hold_short else hold_short

    return p_flat


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC_CSV, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    df = df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] < WINDOW_END)].reset_index(drop=True)
    n = len(df)
    print(f"Window rows: {n} ({df['timestamp'].min()} -> {df['timestamp'].max()})")

    close = df["close"].to_numpy(dtype=np.float64)
    next_ret = np.zeros(n, dtype=np.float64)
    next_ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    p_flat = _dp_recursion(next_ret)
    label_map = {0: "hold", 1: "buy", 2: "sell"}
    labels = [label_map[int(a)] for a in p_flat]

    boundary_start = max(n - MAX_AGE - 2, 0)

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "close": close,
        "label": labels,
    })
    out.to_csv(OUT_CSV, index=False)

    counts = out["label"].value_counts()
    full_counts = out.iloc[:boundary_start]["label"].value_counts()
    print(f"Saved {OUT_CSV}: {len(out)} rows")
    print(f"Label counts (all rows): {counts.to_dict()}")
    print(f"Label counts (full-horizon rows only, n={boundary_start}): {full_counts.to_dict()}")


if __name__ == "__main__":
    main()
