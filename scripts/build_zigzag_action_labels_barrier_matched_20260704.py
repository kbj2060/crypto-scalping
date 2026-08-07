#!/usr/bin/env python3
"""Priority-1 test: relabel L2's training target to match the EXACT mechanics the deployed
v2 policy uses to decide wins/losses, instead of the zigzag-swing-pivot label
(scripts/build_wave3_action_labels_20260531.py) that L2 was actually trained on.

Root-cause finding (documented in docs/model_contracts/omega6_synthesis_v1_20260703_contract.md
"Root-cause analysis" section): the existing zigzag_action label asks "is price in a >=1%,
>=8-bar confirmed swing (14-bar ATR reference)" -- a materially different, smaller/faster
condition than what the frozen v2 winner's backtest actually checks: "does a trade opened here
hit a 15x-ATR(192) take-profit before a 5x-ATR(192) stop-loss within a 24h (288-bar) time-stop".
Only 8.9% of the frozen winner's validation trades actually resolve via take-profit -- most
either stop out or time out -- which is consistent with the model not being trained to predict
this specific win condition.

This script simulates, bar-by-bar and side-by-side (LONG vs SHORT), the EXACT trigger formula
scripts/replay_omega6_v2_variants_20260704.py::run_variant() uses (unreal = raw_price_return *
notional, checked against tp_atr_mult*atr / sl_atr_mult*atr -- note this does NOT divide by
notional, i.e. is reproduced bug-for-bug so the label matches what actually executes, not what
the nominal "15x ATR" naming implies -- see contract doc for the units discrepancy this
surfaced). notional/tp_atr_mult/sl_atr_mult/max_hold/atr_window all match the frozen winner's
exact settings. No lookahead violation: this is offline label construction using future bars,
which is the standard and already-established convention for every zigzag_action label version
in this repo (`uses_future_only_for_offline_labeling: true`) -- only used to build a TRAINING
target, never as a live/backtest decision input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_barrier_matched_20260704"

PRICE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}

ATR_WINDOW = 192
TP_ATR_MULT = 15.0
SL_ATR_MULT = 5.0
MAX_HOLD_BARS = 288
NOTIONAL = 0.30 * 2.0  # frozen v2 winner: fixed_margin=0.30 * fixed_leverage=2.0
FEE = 0.00020
SLIP = 0.00050
MIN_UTILITY = 0.0  # require strictly positive net-of-cost utility to label LONG/SHORT


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr = pd.Series(tr).rolling(window=max(int(window), 1), min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1e-12)


@numba.njit(cache=True)
def _simulate(open_: np.ndarray, close: np.ndarray, atr: np.ndarray, n: int, tp_mult: float, sl_mult: float, notional: float, max_hold: int, fee: float, slip: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    long_util = np.zeros(n, dtype=np.float64)
    short_util = np.zeros(n, dtype=np.float64)
    long_hold = np.zeros(n, dtype=np.int32)
    short_hold = np.zeros(n, dtype=np.int32)
    for i in range(n - 1):
        entry_i = i + 1
        entry_atr = atr[i]
        if entry_atr <= 0.0:
            continue
        tp_thr = tp_mult * entry_atr
        sl_thr = sl_mult * entry_atr
        end_i = min(entry_i + max_hold, n)
        for side in (1, -1):
            entry_price = open_[entry_i] * (1.0 + slip) if side > 0 else open_[entry_i] * (1.0 - slip)
            if entry_price <= 0.0:
                continue
            resolved = False
            for j in range(entry_i, end_i):
                px = close[j]
                if side > 0:
                    raw = (px * (1.0 - slip) - entry_price) / entry_price
                else:
                    raw = (entry_price - px * (1.0 + slip)) / entry_price
                unreal = raw * notional
                if unreal >= tp_thr:
                    net = tp_thr - fee * notional * 2.0
                    if side > 0:
                        long_util[i] = net
                        long_hold[i] = j - entry_i
                    else:
                        short_util[i] = net
                        short_hold[i] = j - entry_i
                    resolved = True
                    break
                if unreal <= -sl_thr:
                    net = -sl_thr - fee * notional * 2.0
                    if side > 0:
                        long_util[i] = net
                        long_hold[i] = j - entry_i
                    else:
                        short_util[i] = net
                        short_hold[i] = j - entry_i
                    resolved = True
                    break
            if not resolved and end_i > entry_i:
                j = end_i - 1
                px = close[j]
                if side > 0:
                    raw = (px * (1.0 - slip) - entry_price) / entry_price
                else:
                    raw = (entry_price - px * (1.0 + slip)) / entry_price
                unreal = raw * notional
                net = unreal - fee * notional * 2.0
                if side > 0:
                    long_util[i] = net
                    long_hold[i] = j - entry_i
                else:
                    short_util[i] = net
                    short_hold[i] = j - entry_i
    return long_util, short_util, long_hold, short_hold


def build_barrier_matched_labels(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    open_ = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr = _atr_pct(frame, ATR_WINDOW)
    long_util, short_util, long_hold, short_hold = _simulate(
        open_, close, atr, n, TP_ATR_MULT, SL_ATR_MULT, NOTIONAL, MAX_HOLD_BARS, FEE, SLIP
    )

    labels = np.zeros(n, dtype=np.int8)
    best_util = np.maximum(long_util, short_util)
    is_long_better = long_util >= short_util
    labels[(best_util > MIN_UTILITY) & is_long_better] = 1
    labels[(best_util > MIN_UTILITY) & ~is_long_better] = 2

    hold = np.where(is_long_better, long_hold, short_hold)
    util = np.where(labels == 1, long_util, np.where(labels == 2, short_util, 0.0))

    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = labels
    out["zigzag_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["barrier_matched_long_utility"] = long_util
    out["barrier_matched_short_utility"] = short_util
    out["barrier_matched_hold_bars"] = hold
    out["barrier_matched_utility"] = util
    return out


def _summary(labels: pd.DataFrame) -> dict[str, Any]:
    counts = labels["zigzag_action"].value_counts().sort_index().to_dict()
    total = max(len(labels), 1)
    return {"rows": int(len(labels)), "counts": {str(k): int(v) for k, v in counts.items()}, "ratios": {str(k): float(v) / total for k, v in counts.items()}}


def main() -> int:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "zigzag_3class_barrier_matched_labels",
        "params": {
            "atr_window": ATR_WINDOW,
            "tp_atr_mult": TP_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT,
            "max_hold_bars": MAX_HOLD_BARS,
            "notional": NOTIONAL,
            "fee": FEE,
            "slip": SLIP,
            "trigger_formula": "unreal=raw_price_return*notional compared to tp_atr_mult*atr / sl_atr_mult*atr (matches replay_omega6_v2_variants_20260704.py::run_variant bug-for-bug, not divided by notional)",
        },
        "contract": {
            "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "label_column": "zigzag_action",
            "uses_future_only_for_offline_labeling": True,
            "matches_deployed_v2_frozen_winner_barrier": True,
        },
        "artifacts": {},
        "summaries": {},
    }
    for year, path in PRICE_FILES.items():
        frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        labels = build_barrier_matched_labels(frame)
        out_path = DEFAULT_OUT / f"zigzag_action_labels_{year}.csv"
        labels.to_csv(out_path, index=False)
        audit["artifacts"][str(year)] = str(out_path)
        audit["summaries"][str(year)] = _summary(labels)
        print(f"{year}: {audit['summaries'][str(year)]}", flush=True)
    audit_path = DEFAULT_OUT / "zigzag_action_label_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
