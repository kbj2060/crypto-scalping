#!/usr/bin/env python3
"""Redesign of the liquidity_sweep trigger's "swept level": instead of always using the blind
48-bar (4h) rolling min/max (build_eth_5m_sweep_followthrough_v2_labels_20260829.py::
add_causal_columns), use whichever is MORE RECENT between (a) that same 48-bar rolling extreme
and (b) the most recently CONFIRMED causal zigzag pivot (verbatim algorithm from
build_zigzag_action_labels_v2_20260604.py::_zigzag_pivots -- price must actually reverse by
1.0-1.8% before a pivot confirms, zero future information).

Motivation (research_eth_sweep_zigzag_anchor_comparison_20260829.py, 2026-08-29): comparing the
two level definitions across all 14,258 existing sweep events found neither is uniformly better
-- the zigzag pivot is MORE recent than the 48-bar window only 14% of the time (in strong
sustained trends it can't confirm a new pivot at all until price retraces, so it goes stale,
sometimes by 100+ bars), but when it IS more recent, it can be meaningfully tighter/more relevant
(one example: a sweep tested a 4h-old high while the real recent high, confirmed 40min earlier,
was a full 1.6% closer). Taking whichever is more recent combines both without either failure
mode dominating. Does NOT attempt to fix the OTHER (already-rejected) "fires too early" case --
see eth_liquidity_sweep_frequency_and_orphaned_labeling_wip_20260829.md's Variant C rejection
(circular logic, 203 reverse flips) for why that direction needs future information and isn't
attempted here.

Everything else (trigger shape: low<level & close>level; outcome: 30min/6bar/1.5xATR/all-6-
bars-close-sustained) is UNCHANGED from build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py
-- only the level computation changes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_hybrid_anchor_v_rebound_20260829"
BAR_MINUTES = 5
LOOKAHEAD_BARS = 6
V_REBOUND_ATR_MULT = 1.5
SWEEP_LOOKBACK_BARS = 48

# Verbatim CLI defaults from build_zigzag_action_labels_v2_20260604.py argparse (lines 333-336)
MIN_REVERSAL_PCT = 0.010
MAX_REVERSAL_PCT = 0.018
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_hybrid_20260829", SWEEP_IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Verbatim copy of build_zigzag_action_labels_v2_20260604.py::_atr_pct."""
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = pd.Series(tr).ewm(span=int(window), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


def zigzag_pivots_with_confirmation(frame: pd.DataFrame) -> pd.DataFrame:
    """Same state machine as build_zigzag_action_labels_v2_20260604.py::_zigzag_pivots, verbatim,
    plus `confirmed_at` (the loop index at which .append() fires) -- see research_eth_sweep_
    zigzag_anchor_comparison_20260829.py, identical implementation."""
    close = frame["close"].to_numpy(dtype=np.float64)
    atr_pct = _atr_pct(frame, ATR_WINDOW)
    n = len(close)

    def _threshold(i: int) -> float:
        atr = float(atr_pct[min(max(int(i), 0), n - 1)])
        return float(np.clip(max(MIN_REVERSAL_PCT, atr * ATR_MULTIPLIER), MIN_REVERSAL_PCT, MAX_REVERSAL_PCT))

    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(close[0])
    pivots: list[tuple[int, float, str, int]] = []

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
                    pivots.append((int(low_idx), float(low_price), "L", i))
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    pivots.append((int(high_idx), float(high_price), "H", i))
                    trend = -1
                    low_idx, low_price = i, price
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            if high_price / max(price, 1e-12) - 1.0 >= _threshold(i):
                pivots.append((int(high_idx), float(high_price), "H", i))
                trend = -1
                low_idx, low_price = i, price
        else:
            if price < low_price:
                low_idx, low_price = i, price
            if price / max(low_price, 1e-12) - 1.0 >= _threshold(i):
                pivots.append((int(low_idx), float(low_price), "L", i))
                trend = 1
                high_idx, high_price = i, price
    return pd.DataFrame(pivots, columns=["pivot_idx", "price", "kind", "confirmed_at"])


def rolling_extreme_with_index(values: np.ndarray, window: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Causal rolling min/max over the PRIOR `window` bars (shifted, matching add_causal_columns
    exactly), returning both the extreme value and the ABSOLUTE index where it occurs -- needed
    to compare recency against the zigzag pivot's own index. O(n) monotonic-deque sliding window,
    verified below against pandas' own rolling().min()/.max() for exact agreement."""
    n = len(values)
    out_val = np.full(n, np.nan)
    out_idx = np.full(n, -1, dtype=np.int64)
    dq: list[int] = []  # indices, monotonic (increasing value for min-mode / decreasing for max-mode)
    better = (lambda a, b: a <= b) if mode == "min" else (lambda a, b: a >= b)
    for i in range(n):
        while dq and better(values[i], values[dq[-1]]):
            dq.pop()
        dq.append(i)
        while dq[0] <= i - window:
            dq.pop(0)
        if i >= window - 1:
            out_val[i] = values[dq[0]]
            out_idx[i] = dq[0]
    # shift by 1 so position T reports the extreme over [T-window, T-1], matching .shift(1) elsewhere
    out_val = np.concatenate([[np.nan], out_val[:-1]])
    out_idx = np.concatenate([[-1], out_idx[:-1]])
    return out_val, out_idx


def compute_hybrid_levels(frame: pd.DataFrame, sweep_impl) -> dict[str, np.ndarray]:
    """Returns {"hybrid_low","hybrid_high","atr"} arrays aligned to `frame`. Shared by this
    script and the Tier0 feature builder for the hybrid-anchor variant -- compute once, reuse,
    not reimplemented per caller."""
    n = len(frame)
    low = frame["low"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)

    print("computing 48-bar rolling extremes with index (self-checked against add_causal_columns)...")
    causal_ref = sweep_impl.add_causal_columns(frame)
    roll_low_val, roll_low_idx = rolling_extreme_with_index(low, SWEEP_LOOKBACK_BARS, "min")
    roll_high_val, roll_high_idx = rolling_extreme_with_index(high, SWEEP_LOOKBACK_BARS, "max")
    check_low = np.nanmax(np.abs(roll_low_val - causal_ref["sweep_level_low"].to_numpy())[SWEEP_LOOKBACK_BARS:])
    check_high = np.nanmax(np.abs(roll_high_val - causal_ref["sweep_level_high"].to_numpy())[SWEEP_LOOKBACK_BARS:])
    print(f"  self-check max abs diff vs add_causal_columns: low={check_low:.10f} high={check_high:.10f} (must be ~0)")
    if check_low > 1e-6 or check_high > 1e-6:
        raise RuntimeError("rolling_extreme_with_index does not match add_causal_columns -- aborting")

    print("computing causal zigzag pivots...")
    piv_df = zigzag_pivots_with_confirmation(frame)
    lows_piv = piv_df[piv_df["kind"] == "L"].sort_values("confirmed_at").reset_index(drop=True)
    highs_piv = piv_df[piv_df["kind"] == "H"].sort_values("confirmed_at").reset_index(drop=True)
    lows_confirmed_at = lows_piv["confirmed_at"].to_numpy()
    lows_pivot_idx = lows_piv["pivot_idx"].to_numpy()
    lows_price = lows_piv["price"].to_numpy()
    highs_confirmed_at = highs_piv["confirmed_at"].to_numpy()
    highs_pivot_idx = highs_piv["pivot_idx"].to_numpy()
    highs_price = highs_piv["price"].to_numpy()

    print("building hybrid levels (most-recent-of-two, per bar)...")
    hybrid_low = np.full(n, np.nan)
    hybrid_high = np.full(n, np.nan)
    for t in range(SWEEP_LOOKBACK_BARS, n):
        # most recent CONFIRMED zigzag low as of bar t (confirmed_at < t, same causal cutoff as before)
        pos = np.searchsorted(lows_confirmed_at, t, side="left") - 1
        zz_idx = lows_pivot_idx[pos] if pos >= 0 else -1
        zz_val = lows_price[pos] if pos >= 0 else np.nan
        if zz_idx > roll_low_idx[t]:
            hybrid_low[t] = zz_val
        else:
            hybrid_low[t] = roll_low_val[t]

        pos = np.searchsorted(highs_confirmed_at, t, side="left") - 1
        zz_idx = highs_pivot_idx[pos] if pos >= 0 else -1
        zz_val = highs_price[pos] if pos >= 0 else np.nan
        if zz_idx > roll_high_idx[t]:
            hybrid_high[t] = zz_val
        else:
            hybrid_high[t] = roll_high_val[t]

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    return {"hybrid_low": hybrid_low, "hybrid_high": hybrid_high, "atr": atr}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_impl = load_sweep_impl()
    frame = sweep_impl.load_5m(SOURCE)
    n = len(frame)
    low = frame["low"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)

    levels = compute_hybrid_levels(frame, sweep_impl)
    hybrid_low, hybrid_high, atr = levels["hybrid_low"], levels["hybrid_high"], levels["atr"]

    timestamps = frame["timestamp"].to_numpy()
    rows = []
    for index in range(SWEEP_LOOKBACK_BARS, n - LOOKAHEAD_BARS):
        a = atr[index]
        if not np.isfinite(a) or a <= 0:
            continue
        future_close = close[index + 1: index + LOOKAHEAD_BARS + 1]
        future_high = high[index + 1: index + LOOKAHEAD_BARS + 1]
        future_low = low[index + 1: index + LOOKAHEAD_BARS + 1]

        level = hybrid_low[index]
        if np.isfinite(level) and low[index] < level and close[index] > level:
            move = float(future_high.max() - low[index])
            confirmed = bool((future_close > level).all())
            label = int(move >= V_REBOUND_ATR_MULT * a and confirmed)
            rows.append({"candidate_index": index, "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
                         "side": "downside", "label": label, "sweep_level": float(level), "atr": float(a),
                         "rebound_move": move, "rebound_atr_multiple": move / float(a)})

        level = hybrid_high[index]
        if np.isfinite(level) and high[index] > level and close[index] < level:
            move = float(high[index] - future_low.min())
            confirmed = bool((future_close < level).all())
            label = int(move >= V_REBOUND_ATR_MULT * a and confirmed)
            rows.append({"candidate_index": index, "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
                         "side": "upside", "label": label, "sweep_level": float(level), "atr": float(a),
                         "rebound_move": move, "rebound_atr_multiple": move / float(a)})

    labels = pd.DataFrame(rows)
    label_path = OUT_DIR / "eth_5m_sweep_hybrid_anchor_v_rebound_labels.csv"
    labels.to_csv(label_path, index=False)

    report = {
        "level_definition": "whichever is MORE RECENT of (48-bar rolling extreme) vs (most recently confirmed causal zigzag pivot)",
        "outcome_definition": "unchanged from v3: 30min/6bar sustain, ATR1.5x magnitude",
        "total_events": int(len(labels)),
        "label_rate": float(labels["label"].mean()) if len(labels) else None,
        "by_side": {side: {"n": int((labels["side"] == side).sum()),
                            "label_rate": float(labels.loc[labels["side"] == side, "label"].mean())}
                    for side in ("downside", "upside")},
        "output": str(label_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
