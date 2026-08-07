"""BTC 5m candidate #3: multi-scale ATR zigzag, per the AEDL paper's finding that "adaptive
multi-scale thresholding" (not causal-inference machinery) was the actual source of its Sharpe
improvement over fixed-threshold labeling. Reuses the same zigzag pivot algorithm but with the
reversal threshold set from the MAX of several ATR windows (7, 14, 28, 56 bars) instead of a
single 14-bar window -- requires a stronger, more persistent move across multiple timescales
before confirming a pivot, which should produce fewer but higher-conviction waves than the
single-scale version (build_btc_5m_zigzag_and_pivot_labels_20260806.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_wave3_action_labels_20260531 as zigzag  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_multiscale_labels_20260806.parquet"

MIN_REVERSAL_PCT = 0.009
ATR_WINDOWS = (7, 14, 28, 56)
ATR_MULTIPLIER = 1.0
MIN_WAVE_BARS = 6
TRANSITION_BUFFER = 1
MAE_PENALTY = 1.1
SOFTMAX_TEMPERATURE = 1.9
MIN_RISK_FLOOR = 0.001


def _multiscale_atr_pct(frame: pd.DataFrame) -> np.ndarray:
    stacked = np.stack([zigzag._atr_pct(frame, w) for w in ATR_WINDOWS], axis=0)
    return stacked.max(axis=0)  # require the LARGEST of several timescales' ATR to be cleared


def main() -> int:
    frame = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    # monkey-patch-free: reimplement the pivot loop using the multiscale threshold, matching
    # zigzag._zigzag_pivots' logic exactly but with a swapped threshold function
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr_pct = _multiscale_atr_pct(frame)
    n = len(close)

    def _threshold(i: int) -> float:
        return max(MIN_REVERSAL_PCT, float(atr_pct[min(max(i, 0), n - 1)]) * ATR_MULTIPLIER)

    trend = 0
    low_idx = high_idx = 0
    low_price = high_price = float(close[0])
    pivots: list[tuple[int, float, str]] = []
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
                    pivots.append((int(low_idx), float(low_price), "L"))
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    pivots.append((int(high_idx), float(high_price), "H"))
                    trend = -1
                    low_idx, low_price = i, price
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            drop = high_price / max(price, 1e-12) - 1.0
            if drop >= _threshold(i):
                pivots.append((int(high_idx), float(high_price), "H"))
                trend = -1
                low_idx, low_price = i, price
        else:
            if price < low_price:
                low_idx, low_price = i, price
            rise = price / max(low_price, 1e-12) - 1.0
            if rise >= _threshold(i):
                pivots.append((int(low_idx), float(low_price), "L"))
                trend = 1
                high_idx, high_price = i, price
    if trend == 1 and (not pivots or pivots[-1][0] != high_idx):
        pivots.append((int(high_idx), float(high_price), "H"))
    elif trend == -1 and (not pivots or pivots[-1][0] != low_idx):
        pivots.append((int(low_idx), float(low_price), "L"))
    pivots = zigzag._filter_alternating(pivots)

    # reuse build_zigzag_action_labels' segment-fill/soft-label logic by monkeypatching its
    # internal pivot call is not exposed, so replicate the segment-fill portion directly.
    labels = np.zeros(n, dtype=np.int8)
    segment_id = np.full(n, -1, dtype=np.int32)
    wave_ret = np.zeros(n, dtype=np.float32)
    wave_bars = np.zeros(n, dtype=np.int16)
    sid = 0
    for start, end in zip(pivots, pivots[1:]):
        idx_s, val_s, type_s = start
        idx_e, val_e, type_e = end
        if idx_e <= idx_s:
            continue
        bars = int(idx_e - idx_s)
        if bars < MIN_WAVE_BARS:
            continue
        if type_s == "L" and type_e == "H":
            side = 1
        elif type_s == "H" and type_e == "L":
            side = 2
        else:
            continue
        labels[idx_s:idx_e] = side
        segment_id[idx_s:idx_e] = sid
        wave_ret[idx_s:idx_e] = np.float32((val_e / max(val_s, 1e-12) - 1.0) * (1.0 if side == 1 else -1.0))
        wave_bars[idx_s:idx_e] = np.int16(min(bars, np.iinfo(np.int16).max))
        sid += 1

    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = labels
    out["zigzag_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["zigzag_segment_id"] = segment_id
    out["zigzag_wave_return"] = wave_ret
    out["zigzag_wave_bars"] = wave_bars
    out.to_parquet(OUT_PATH, index=False)

    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(out["zigzag_action_name"].value_counts())
    print(f"segments: {sid}")
    print("mean/median wave bars by action:")
    print(out.groupby("zigzag_action_name")["zigzag_wave_bars"].agg(["mean", "median"]))
    print(f"n_pivots={len(pivots)}  (single-scale build had 4069 pivots for comparison)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
