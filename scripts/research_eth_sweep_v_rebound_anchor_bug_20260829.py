#!/usr/bin/env python3
"""Diagnose why visually V-shaped sweeps end up labeled NO_V_REBOUND.

Hypothesis (from user's visual inspection of the rendered examples): the sweep bar's own
low/high is often NOT the true bottom/top of the move -- price keeps drifting in the sweep
direction for a bar or two AFTER the nominal sweep bar before the real reversal starts. The
current label anchors both the move-magnitude and the confirmation check to the sweep bar's
own low/high/close, so a delayed true extremum could make a real V-rebound compute as small
(measured from the wrong, shallower anchor) or unconfirmed.

Reproduces the exact seed=42 sample of 10 NO_V_REBOUND examples already rendered, and for
each compares:
  - current: move/confirmed anchored to the sweep bar itself (row["low"]/row["high"]/row["close"])
  - alternative: move/confirmed anchored to the TRUE local extremum, wherever it falls within
    an extended look-forward window (60 min, double the label's 30 min) after the sweep bar.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABEL_CSV = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
LABEL_WINDOW_BARS = 6
EXTENDED_WINDOW_BARS = 12
SEED = 42


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_bug_diag_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diagnose(frame: pd.DataFrame, event: pd.Series) -> dict:
    idx = int(event["candidate_index"])
    side = event["side"]
    atr = float(event["atr"])
    sweep_row = frame.iloc[idx]
    extended = frame.iloc[idx: idx + EXTENDED_WINDOW_BARS + 1]  # includes sweep bar at offset 0

    if side == "downside":
        true_offset = int(extended["low"].to_numpy().argmin())
        true_extreme = float(extended["low"].min())
        after = frame.iloc[idx + true_offset + 1: idx + true_offset + 1 + LABEL_WINDOW_BARS]
        alt_move = float(after["high"].max() - true_extreme) if len(after) else float("nan")
        alt_confirmed = bool(after["close"].iloc[-1] > sweep_row["close"]) if len(after) else False
    else:
        true_offset = int(extended["high"].to_numpy().argmax())
        true_extreme = float(extended["high"].max())
        after = frame.iloc[idx + true_offset + 1: idx + true_offset + 1 + LABEL_WINDOW_BARS]
        alt_move = float(true_extreme - after["low"].min()) if len(after) else float("nan")
        alt_confirmed = bool(after["close"].iloc[-1] < sweep_row["close"]) if len(after) else False

    alt_label = int(alt_move >= 1.0 * atr and alt_confirmed) if after is not None and len(after) else None
    return {
        "timestamp": event["timestamp"],
        "side": side,
        "orig_move": float(event["rebound_move"]),
        "orig_reb_atr": float(event["rebound_atr_multiple"]),
        "orig_label": int(event["label"]),
        "true_extreme_offset_bars": true_offset,
        "bars_available_after_true_extreme": len(after),
        "alt_move": alt_move,
        "alt_reb_atr": alt_move / atr if atr else float("nan"),
        "alt_confirmed": alt_confirmed,
        "alt_label": alt_label,
    }


def main() -> int:
    impl = load_impl()
    frame = impl.load_5m(SOURCE)
    labels = pd.read_csv(LABEL_CSV)

    for label_value in (0, 1):
        sample = labels[labels["label"] == label_value].sample(n=10, random_state=SEED)
        rows = [diagnose(frame, event) for _, event in sample.iterrows()]
        table = pd.DataFrame(rows)
        print(f"\n===== label={label_value} ({'NO_V_REBOUND' if label_value == 0 else 'V_REBOUND'}) sample =====")
        print(table.to_string(index=False))
        flips = int(((table["alt_label"] == 1) & (table["orig_label"] == 0)).sum())
        unflips = int(((table["alt_label"] == 0) & (table["orig_label"] == 1)).sum())
        print(f"0->1 flips: {flips}/10   1->0 flips: {unflips}/10")
        print("true-extreme offset distribution:", table["true_extreme_offset_bars"].value_counts().sort_index().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
