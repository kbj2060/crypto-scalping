"""BTC 1h new-architecture, Layer A: swing-TRANSITION detector, split off from direction/exit
per user instruction ("지그재그 라벨은 스윙 전환되는 순간을 예측하는 레이어의 라벨로 쓰자").

Binary, direction-agnostic label: will a confirmed zigzag pivot (swing turning point, same
detector as build_btc_1h_zigzag_labels_20260805.py) occur within the next K bars. This layer's
only job is "is a turn imminent" -- WHICH direction and WHEN to exit are Layer B's job
(scripts/build_btc_1h_direction_exit_labels_20260805.py, built after this layer is reviewed).

Hindsight label (future pivots looked up to build the target) -- legitimate training-target
convention, not a live feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_wave3_action_labels_20260531 as zigzag  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_1h_pivot_transition_labels_20260805.parquet"

K_BARS = 2  # "transition imminent" horizon
MIN_REVERSAL_PCT = 0.009
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0


def main() -> int:
    frame = pd.read_csv(PANEL_PATH, parse_dates=["timestamp"])
    frame = frame[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp").reset_index(drop=True)

    pivots = zigzag._zigzag_pivots(
        frame, min_reversal_pct=MIN_REVERSAL_PCT, atr_window=ATR_WINDOW, atr_multiplier=ATR_MULTIPLIER,
    )
    n = len(frame)
    is_pivot = np.zeros(n, dtype=np.int8)
    pivot_type = np.full(n, "", dtype=object)
    for idx, price, ptype in pivots:
        is_pivot[idx] = 1
        pivot_type[idx] = ptype

    transition_soon = np.zeros(n, dtype=np.int8)
    for i in range(n):
        lo, hi = i, min(n, i + K_BARS + 1)
        if is_pivot[lo:hi].any():
            transition_soon[i] = 1

    out = frame[["timestamp", "close"]].copy()
    out["is_pivot"] = is_pivot
    out["pivot_type"] = pivot_type
    out["transition_soon"] = transition_soon
    # right-censor: last K_BARS rows can't know the future -> drop from training/eval
    out.loc[out.index[-K_BARS:], "transition_soon"] = np.nan

    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(f"n_pivots={len(pivots)} (H={sum(1 for p in pivots if p[2]=='H')} L={sum(1 for p in pivots if p[2]=='L')})")
    print(f"transition_soon positive rate: {out['transition_soon'].mean():.4f}  (K_BARS={K_BARS})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
