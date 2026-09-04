#!/usr/bin/env python3
"""User-proposed alternative label: "within 1 hour, did the high or low touch a sharp (>=1.5x
ATR) move" -- no sustained-close confirmation, window widened 30min->1h. Built to let the user's
hypothesis ("looser label = higher accuracy = better model") get a FAIR test: same TRAIN/VAL/OOS
split and Tier0+rsi features as the current V_REBOUND model, compared via AUC and lift-over-THIS-
LABEL'S-OWN-naive-baseline (not raw accuracy, which is not comparable across labels with
different base rates -- see eth_liquidity_sweep_v_rebound_feature_plan_20260829.md's "loose label
base rate check" section: this definition's own population base rate is ~78%, so raw accuracy
alone would look inflated without reflecting real added skill).

Same sweep trigger population as the current label (import add_causal_columns/load_5m from
build_eth_5m_sweep_followthrough_v2_labels_20260829.py, unmodified) -- only the OUTCOME window
changes: LOOKAHEAD_BARS 6->12 (1h), and the outcome is a single-touch magnitude check
(future["high"].max() - row["low"] >= ATR_MULT*atr, or the upside mirror) with NO confirmed/
sustained-close requirement at all.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
SOURCE = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_sweep_loose_touch_1h_20260829"
BAR_MINUTES = 5
LOOKAHEAD_BARS = 12  # 1h -- user's proposal, vs the current label's 6 (30min)
ATR_MULT = 1.5        # unchanged from the current label -- only the sustain requirement and window change


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_loose_20260829", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_events(frame: pd.DataFrame, impl) -> pd.DataFrame:
    timestamps = frame["timestamp"].to_numpy()
    rows = []
    for index in range(impl.SWEEP_LOOKBACK_BARS, len(frame) - LOOKAHEAD_BARS):
        row = frame.iloc[index]
        atr = row["atr"]
        if not np.isfinite(atr) or atr <= 0:
            continue
        future = frame.iloc[index + 1:index + LOOKAHEAD_BARS + 1]

        level = row["sweep_level_low"]
        if np.isfinite(level) and row["low"] < level and row["close"] > level:
            move = float(future["high"].max() - row["low"])
            label = int(move >= ATR_MULT * atr)
            rows.append({"candidate_index": index, "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
                         "side": "downside", "label": label, "sweep_level": float(level), "atr": float(atr),
                         "move": move, "move_atr_multiple": move / float(atr)})

        level = row["sweep_level_high"]
        if np.isfinite(level) and row["high"] > level and row["close"] < level:
            move = float(row["high"] - future["low"].min())
            label = int(move >= ATR_MULT * atr)
            rows.append({"candidate_index": index, "timestamp": pd.Timestamp(timestamps[index]).isoformat(),
                         "side": "upside", "label": label, "sweep_level": float(level), "atr": float(atr),
                         "move": move, "move_atr_multiple": move / float(atr)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    impl = load_impl()
    frame = impl.add_causal_columns(impl.load_5m(SOURCE))
    labels = label_events(frame, impl)

    label_path = OUT_DIR / "eth_5m_sweep_loose_touch_1h_labels.csv"
    labels.to_csv(label_path, index=False)

    report = {
        "label_definition": (
            f"within {LOOKAHEAD_BARS * BAR_MINUTES} minutes of the sweep bar, high or low touches "
            f">= {ATR_MULT}x ATR(14) move from the sweep extreme -- NO sustained-close confirmation "
            "(unlike the current V_REBOUND label), single-touch only"
        ),
        "total_events": int(len(labels)),
        "label_rate": float(labels["label"].mean()) if len(labels) else None,
        "naive_majority_baseline": max(float(labels["label"].mean()), 1 - float(labels["label"].mean())),
        "by_side": {
            side: {"n": int((labels["side"] == side).sum()),
                   "label_rate": float(labels.loc[labels["side"] == side, "label"].mean())}
            for side in ("downside", "upside")
        },
        "output": str(label_path),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
