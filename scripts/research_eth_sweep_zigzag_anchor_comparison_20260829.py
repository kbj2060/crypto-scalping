#!/usr/bin/env python3
"""User observation: on a 1h chart, the liquidity_sweep trigger sometimes fires later than the
real structural extreme (the level tested is 'stale' -- a rolling 48-bar min/max, not necessarily
the level a chart-reader would call 'the recent swing'). Explicitly does NOT attempt to fix
"fires too early" (real extreme comes LATER, only knowable in hindsight) -- that's the already-
tested-and-rejected "peak free search" circular-logic trap (see eth_liquidity_sweep_frequency_
and_orphaned_labeling_wip_20260829.md, Variant C: 69.4% but 203 reverse flips). This script only
investigates the "too late" direction using an alternative, fully CAUSAL level definition (a
confirmed zigzag pivot -- price must actually reverse by a threshold before a pivot counts,
requires zero future information beyond what's already happened).

Zigzag state machine copied VERBATIM from build_zigzag_action_labels_v2_20260604.py::
_zigzag_pivots / _atr_pct (the canonical, ~140-file-reused causal zigzag in this repo) -- the
ONLY addition is tracking the confirmation index (the bar `i` at which the pivot.append() call
actually fires) alongside each pivot, so we can correctly ask "what pivots were ALREADY
CONFIRMED as of bar T" without back-dating knowledge to before it actually existed.

Descriptive/comparison only -- does not relabel or retrain anything yet.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP_IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_zigzag_anchor_20260829"

# Verbatim CLI defaults from build_zigzag_action_labels_v2_20260604.py argparse (lines 333-336) -- not re-tuned here
MIN_REVERSAL_PCT = 0.010
MAX_REVERSAL_PCT = 0.018
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0


def load_sweep_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_zz_20260829", SWEEP_IMPL_SCRIPT)
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


def zigzag_pivots_with_confirmation(frame: pd.DataFrame) -> list[tuple[int, float, str, int]]:
    """Same state machine as build_zigzag_action_labels_v2_20260604.py::_zigzag_pivots, verbatim,
    with ONE addition: each returned tuple also carries `confirmed_at` (the loop index `i` at
    which .append() actually fires) -- (pivot_idx, price, "H"/"L", confirmed_at)."""
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
    return pivots


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_impl = load_sweep_impl()
    frame = sweep_impl.add_causal_columns(sweep_impl.load_5m(SOURCE))
    labels = pd.read_csv(LABEL_CSV)

    print("computing causal zigzag pivots (this is a single forward pass over the full series)...")
    pivots = zigzag_pivots_with_confirmation(frame)
    piv_df = pd.DataFrame(pivots, columns=["pivot_idx", "price", "kind", "confirmed_at"])
    print(f"  {len(piv_df)} pivots found ({(piv_df.kind=='L').sum()} lows, {(piv_df.kind=='H').sum()} highs)")

    lows = piv_df[piv_df["kind"] == "L"].sort_values("confirmed_at").reset_index(drop=True)
    highs = piv_df[piv_df["kind"] == "H"].sort_values("confirmed_at").reset_index(drop=True)

    def most_recent_confirmed(pool: pd.DataFrame, as_of_idx: int) -> tuple[float, int] | tuple[None, None]:
        """Most recent pivot with confirmed_at < as_of_idx (strictly before the sweep bar --
        matches causal availability: only pivots ALREADY confirmed before this bar are usable)."""
        eligible = pool[pool["confirmed_at"] < as_of_idx]
        if eligible.empty:
            return None, None
        row = eligible.iloc[-1]
        return float(row["price"]), int(row["pivot_idx"])

    rows = []
    for _, ev in labels.iterrows():
        idx = int(ev["candidate_index"])
        pool = lows if ev["side"] == "downside" else highs
        zz_level, zz_pivot_idx = most_recent_confirmed(pool, idx)
        if zz_level is None:
            continue
        current_level = float(ev["sweep_level"])
        bars_between = idx - zz_pivot_idx  # how many bars ago (from the sweep bar) was the zigzag pivot itself
        rows.append({
            "timestamp": ev["timestamp"], "side": ev["side"], "label": ev["label"],
            "current_level": current_level, "zigzag_level": zz_level,
            "level_diff_pct": (zz_level - current_level) / current_level,
            "zigzag_pivot_bars_ago": bars_between,
        })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(OUT_DIR / "level_comparison.csv", index=False)
    print(f"\ncompared {len(cmp_df)}/{len(labels)} sweep events (rest had no confirmed zigzag pivot yet, early warmup)")
    print(f"level_diff_pct: mean={cmp_df['level_diff_pct'].mean():.5f} "
          f"median={cmp_df['level_diff_pct'].median():.5f} std={cmp_df['level_diff_pct'].std():.5f}")
    print(f"zigzag_pivot_bars_ago: mean={cmp_df['zigzag_pivot_bars_ago'].mean():.1f} "
          f"median={cmp_df['zigzag_pivot_bars_ago'].median():.1f} "
          f"(48 = same as current 4h rolling window; <48 means zigzag found a MORE RECENT structural pivot)")
    print(f"\nfraction where zigzag pivot is more recent than the current 48-bar lookback would even reach "
          f"(bars_ago < 48): {(cmp_df['zigzag_pivot_bars_ago'] < 48).mean():.4f}")
    print(f"fraction where levels differ by >1%: {(cmp_df['level_diff_pct'].abs() > 0.01).mean():.4f}")
    print(f"fraction where levels differ by >3%: {(cmp_df['level_diff_pct'].abs() > 0.03).mean():.4f}")

    # a few concrete disagreement examples for charting -- large level_diff, sorted for inspection
    big_diff = cmp_df.reindex(cmp_df["level_diff_pct"].abs().sort_values(ascending=False).index)
    print("\ntop 5 largest disagreements:")
    print(big_diff.head(5).to_string())
    (OUT_DIR / "report.json").write_text(pd.Series({
        "n_compared": len(cmp_df),
        "level_diff_pct_mean": float(cmp_df["level_diff_pct"].mean()),
        "level_diff_pct_median": float(cmp_df["level_diff_pct"].median()),
        "pivot_bars_ago_median": float(cmp_df["zigzag_pivot_bars_ago"].median()),
        "frac_more_recent_than_48bar": float((cmp_df["zigzag_pivot_bars_ago"] < 48).mean()),
        "frac_diff_gt_1pct": float((cmp_df["level_diff_pct"].abs() > 0.01).mean()),
    }).to_json(indent=2))
    print(f"\nWrote {OUT_DIR / 'level_comparison.csv'} and report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
