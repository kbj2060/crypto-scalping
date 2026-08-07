"""BTC 1h new-architecture, label redesign v3 (per user: vol-regime lacked direction, raw-return
regression had zero rank-correlation same as the closed 5m H_selection result -- try ETH's
zig075 zigzag-swing label instead, on native-1h BTC data, standalone, not combined with any
already-closed BTC component).

Reuses the canonical zigzag-swing labeling algorithm as-is (scripts/build_wave3_action_labels_20260531.py,
same one behind ETH's live zig075 component) with ETH's own default parameters -- the ATR-adaptive
reversal threshold (`atr_multiplier * atr_pct`, floored at `min_reversal_pct`) auto-scales to
1h ATR, so no timeframe-specific retuning is applied on this first pass.

NOTE: BTC zigzag/zig075 already failed on 5m data combined with h48qual
(project-btc-zigzag-dual-component-already-failed-20260802) -- this is a DIFFERENT timeframe
(native 1h, not 5m) and standalone (not a dual-component router), not a re-run of that closed line.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_wave3_action_labels_20260531 as zigzag  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet"


def main() -> int:
    frame = pd.read_csv(PANEL_PATH, parse_dates=["timestamp"])
    frame = frame[["timestamp", "open", "high", "low", "close"]].sort_values("timestamp").reset_index(drop=True)

    labels = zigzag.build_zigzag_action_labels(
        frame,
        min_reversal_pct=0.009,
        min_wave_bars=6,
        transition_buffer=1,
        atr_window=14,
        atr_multiplier=1.0,
        mae_penalty=1.1,
        softmax_temperature=1.9,
        min_risk_floor=0.001,
    )
    labels.to_parquet(OUT_PATH, index=False)

    print(f"wrote {OUT_PATH}, shape={labels.shape}")
    print("action counts:")
    print(labels["zigzag_action_name"].value_counts())
    print(f"segments: {labels['zigzag_segment_id'].max() + 1}")
    print("mean wave return by action:")
    print(labels.groupby("zigzag_action_name")["zigzag_wave_return"].mean())
    print("mean wave bars by action:")
    print(labels.groupby("zigzag_action_name")["zigzag_wave_bars"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
