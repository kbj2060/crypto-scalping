"""BTC 5m retry of this session's Layer A (transition detector) + Layer B (zigzag direction)
architecture. All prior attempts this session were on native-1h data; user asked to retry at 5m,
the SAME cadence ETH's live zig075 actually runs on -- so ETH's own canonical zigzag parameters
(build_eth_split_zigzag_labels_20260724.py) are used verbatim, no retuning.

Uses causalfix_final (5m, 114 cols, 2024-01-01..2026-08-01) for OHLC -- NOT a re-run of the closed
causalfix_final quality-classifier lines (docs/btc_new_architecture_session_summary_20260804.md);
only the OHLC/feature columns are reused here, with the entirely new label architecture built this
session (zigzag pivot + transition-detector + binary-direction), which was never tried on 5m before.
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
ZIGZAG_OUT = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
PIVOT_OUT = ROOT / "data/splits/year_oos/btc_5m_pivot_transition_labels_20260806.parquet"

MIN_REVERSAL_PCT = 0.009
MIN_WAVE_BARS = 6
TRANSITION_BUFFER = 1
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0
MAE_PENALTY = 1.1
SOFTMAX_TEMPERATURE = 1.9
MIN_RISK_FLOOR = 0.001

K_BARS = 24  # "transition imminent within next 2h" (24 x 5m bars), same real-time window as the 1h version's K=2h


def main() -> int:
    frame = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    labels = zigzag.build_zigzag_action_labels(
        frame,
        min_reversal_pct=MIN_REVERSAL_PCT,
        min_wave_bars=MIN_WAVE_BARS,
        transition_buffer=TRANSITION_BUFFER,
        atr_window=ATR_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        mae_penalty=MAE_PENALTY,
        softmax_temperature=SOFTMAX_TEMPERATURE,
        min_risk_floor=MIN_RISK_FLOOR,
    )
    labels.to_parquet(ZIGZAG_OUT, index=False)
    print(f"wrote {ZIGZAG_OUT}, shape={labels.shape}")
    print(labels["zigzag_action_name"].value_counts())
    print(f"segments: {labels['zigzag_segment_id'].max() + 1}")
    print("mean/median wave bars by action:")
    print(labels.groupby("zigzag_action_name")["zigzag_wave_bars"].agg(["mean", "median"]))

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

    piv_out = frame[["timestamp", "close"]].copy()
    piv_out["is_pivot"] = is_pivot
    piv_out["pivot_type"] = pivot_type
    piv_out["transition_soon"] = transition_soon.astype(float)
    piv_out.loc[piv_out.index[-K_BARS:], "transition_soon"] = np.nan
    piv_out.to_parquet(PIVOT_OUT, index=False)

    print(f"\nwrote {PIVOT_OUT}, shape={piv_out.shape}")
    print(f"n_pivots={len(pivots)} (H={sum(1 for p in pivots if p[2]=='H')} L={sum(1 for p in pivots if p[2]=='L')})")
    print(f"transition_soon positive rate: {piv_out['transition_soon'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
