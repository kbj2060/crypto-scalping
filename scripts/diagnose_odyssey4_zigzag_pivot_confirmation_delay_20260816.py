#!/usr/bin/env python3
"""B2 diagnostic (Odyssey4 layer/parameter improvement proposal 20260816).

Measures the real distribution (median, p95) of "how many bars after a pivot
forms does it take for price to move far enough to confirm/finalize that
pivot" on the live zigzag_action_labels_20260531 label set (used by both
h48qual and zig075). This sizes the purge/embargo gap for C1.

Pivot-confirmation logic is instrumented from
scripts/build_zigzag_action_labels_v2_20260604.py:_zigzag_pivots (lines
71-132) -- that function only returns the final confirmed pivot list
[(bar_idx, price, "H"/"L"), ...], with no record of *when* (which later bar)
each pivot got confirmed. This script re-implements the exact same reversal
loop (same branches, same threshold() rule) but additionally records the bar
index at the moment each pivot is appended to `pivots` (the "confirmation
bar"), so confirm_delay_bars = confirmation_bar_index - pivot_bar_index.

Params: the actual zigzag_action_labels_20260531 audit
(tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_label_audit.json)
was built with an earlier ("v1") revision of this recipe -- min_reversal_pct
=0.01, atr_window=14, atr_multiplier=1.0, no upper max_reversal_pct clamp.
v2 added an upper clamp (max_reversal_pct) on top of the same core reversal
rule; we set it to a large value here (10.0 = never binds) to reproduce the
v1 behavior actually used to build the live label set, while reusing v2's
threshold()/pivot-loop code structure as instructed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_zigzag_action_labels_v2_20260604 as zz  # noqa: E402

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
YEARS = [2024, 2025, 2026]

# Actual params used to build the live zigzag_action_labels_20260531 set
# (from zigzag_action_label_audit.json in LABEL_DIR).
MIN_REVERSAL_PCT = 0.01
MAX_REVERSAL_PCT = 10.0  # effectively unbounded -> reproduces v1 (no upper clamp)
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0


def _zigzag_pivots_with_confirm_delay(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Same control flow as zz._zigzag_pivots, instrumented to also record the
    bar index where each pivot got confirmed (i.e. the current loop index `i`
    at the moment the pivot is appended)."""
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr_pct = zz._atr_pct(frame, ATR_WINDOW)
    n = len(close)
    out: list[dict[str, Any]] = []
    if n == 0:
        return out

    def _threshold(i: int) -> float:
        atr = float(atr_pct[min(max(int(i), 0), n - 1)])
        return float(np.clip(max(MIN_REVERSAL_PCT, atr * ATR_MULTIPLIER), MIN_REVERSAL_PCT, MAX_REVERSAL_PCT))

    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(close[0])

    for i in range(1, n):
        price = float(close[i])
        if not np.isfinite(price):
            continue
        if trend == 0:
            if price < low_price:
                low_idx, low_price = i, price
            if price > high_price:
                high_idx, high_price = i, price
            thr = _threshold(i)
            if high_price / max(low_price, 1e-12) - 1.0 >= thr:
                if low_idx < high_idx:
                    out.append({"pivot_idx": int(low_idx), "confirm_idx": int(i), "type": "L"})
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    out.append({"pivot_idx": int(high_idx), "confirm_idx": int(i), "type": "H"})
                    trend = -1
                    low_idx, low_price = i, price
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            if high_price / max(price, 1e-12) - 1.0 >= _threshold(i):
                out.append({"pivot_idx": int(high_idx), "confirm_idx": int(i), "type": "H"})
                trend = -1
                low_idx, low_price = i, price
        else:
            if price < low_price:
                low_idx, low_price = i, price
            if price / max(low_price, 1e-12) - 1.0 >= _threshold(i):
                out.append({"pivot_idx": int(low_idx), "confirm_idx": int(i), "type": "L"})
                trend = 1
                high_idx, high_price = i, price
    # Note: unlike zz._zigzag_pivots, we deliberately drop the trailing
    # unconfirmed pivot (appended unconditionally after the loop in the
    # original) and skip _filter_alternating -- neither has a well-defined
    # confirm_idx (the trailing one never actually confirms within the
    # series; alternation-merges don't change confirm delay semantics for
    # the surviving pivot). This only affects the last 0-1 pivot per year.
    return out


def _pct(vals: np.ndarray, p: float) -> float:
    return float(np.percentile(vals, p)) if len(vals) else 0.0


def main() -> int:
    all_delays: list[int] = []
    per_year: dict[str, Any] = {}
    for year in YEARS:
        path = LABEL_DIR / f"zigzag_action_labels_{year}.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        pivots = _zigzag_pivots_with_confirm_delay(frame)
        delays = np.array([p["confirm_idx"] - p["pivot_idx"] for p in pivots], dtype=np.int64)
        all_delays.extend(delays.tolist())
        per_year[str(year)] = {
            "rows": int(len(frame)),
            "n_pivots": int(len(pivots)),
            "confirm_delay_bars": {
                "mean": float(delays.mean()) if len(delays) else 0.0,
                "median": _pct(delays, 50),
                "p90": _pct(delays, 90),
                "p95": _pct(delays, 95),
                "p99": _pct(delays, 99),
                "max": int(delays.max()) if len(delays) else 0,
                "min": int(delays.min()) if len(delays) else 0,
            },
        }

    all_delays_arr = np.array(all_delays, dtype=np.int64)
    report = {
        "diagnostic": "odyssey4_zigzag_pivot_confirmation_delay_20260816 (Phase B2)",
        "label_source": str(LABEL_DIR),
        "params_used": {
            "min_reversal_pct": MIN_REVERSAL_PCT,
            "max_reversal_pct_note": "set to 10.0 (unbounded) to reproduce the v1 recipe actually used to build zigzag_action_labels_20260531 (no upper clamp in its audit.json params)",
            "atr_window": ATR_WINDOW,
            "atr_multiplier": ATR_MULTIPLIER,
        },
        "per_year": per_year,
        "combined_all_years": {
            "n_pivots": int(len(all_delays_arr)),
            "confirm_delay_bars": {
                "mean": float(all_delays_arr.mean()) if len(all_delays_arr) else 0.0,
                "median": _pct(all_delays_arr, 50),
                "p90": _pct(all_delays_arr, 90),
                "p95": _pct(all_delays_arr, 95),
                "p99": _pct(all_delays_arr, 99),
                "max": int(all_delays_arr.max()) if len(all_delays_arr) else 0,
            },
        },
        "sizing_note": "C1's purge/embargo gap should be sized off the combined p95 confirm_delay_bars -- "
        "that is the number of forward bars whose price action the labeler consulted before it could "
        "finalize a pivot label near the train/val split boundary.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
