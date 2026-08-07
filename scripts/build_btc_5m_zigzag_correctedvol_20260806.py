"""Re-derive zigzag using the SAME corrected volatility basis fixed for the triple-barrier label
(root cause 3 in [[project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806]]): the original zigzag
pivot-reversal threshold (scripts/build_wave3_action_labels_20260531.py) scales off single-bar
EWMA true-range ATR -- the same "single-bar vol is too tight a scale for a multi-bar structure"
issue diagnosed for triple-barrier's SL. This rebuilds zigzag's pivot detection using the rolling
dispersion of 12-bar (1h) cumulative log returns (288-bar/24h lookback) instead, commensurate with
the same scale the TP/SL backtest itself uses, then keeps everything else about zigzag's algorithm
(alternating-pivot state machine, min_wave_bars filter, transition buffer) identical.

Oracle-validated (per project-btc-oracle-label-selection-protocol-20260806) against the original
ATR-basis zigzag, which already passed at 73-75% win rate / ~4.9x OOS equity through the corrected
TP/SL simulator -- this checks whether using a consistent vol basis for BOTH pivot detection and
TP/SL improves further.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.causal_futures_backtest import simulate_single_position  # noqa: E402

PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_correctedvol_labels_20260806.parquet"

CUMRET_BARS = 12
VOL_LOOKBACK = 288
MIN_REVERSAL_PCT = 0.009  # same floor as the original zigzag construction
VOL_MULTIPLIER = 1.0
MIN_WAVE_BARS = 6
TRANSITION_BUFFER = 1

TP_MULT, SL_MULT, HORIZON_BARS = 2.5, 1.2, 288
MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.0010
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")


@numba.njit(cache=True)
def _zigzag_pivots_correctedvol(close, threshold, min_reversal_pct):
    n = len(close)
    pivots_idx = np.empty(n, dtype=np.int64)
    pivots_price = np.empty(n, dtype=np.float64)
    pivots_type = np.empty(n, dtype=np.int8)  # 1=H, 2=L
    n_piv = 0

    trend = 0
    low_idx, high_idx = 0, 0
    low_price, high_price = close[0], close[0]

    for i in range(1, n):
        price = close[i]
        thr = max(min_reversal_pct, threshold[i])
        if trend == 0:
            if price < low_price:
                low_idx, low_price = i, price
            if price > high_price:
                high_idx, high_price = i, price
            if high_price / max(low_price, 1e-12) - 1.0 >= thr:
                if low_idx < high_idx:
                    pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = low_idx, low_price, 2
                    n_piv += 1
                    trend = 1
                    high_idx, high_price = i, price
                else:
                    pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = high_idx, high_price, 1
                    n_piv += 1
                    trend = -1
                    low_idx, low_price = i, price
        elif trend == 1:
            if price > high_price:
                high_idx, high_price = i, price
            drop = high_price / max(price, 1e-12) - 1.0
            if drop >= thr:
                pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = high_idx, high_price, 1
                n_piv += 1
                trend = -1
                low_idx, low_price = i, price
        else:
            if price < low_price:
                low_idx, low_price = i, price
            rise = price / max(low_price, 1e-12) - 1.0
            if rise >= thr:
                pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = low_idx, low_price, 2
                n_piv += 1
                trend = 1
                high_idx, high_price = i, price

    if trend == 1:
        if n_piv == 0 or pivots_idx[n_piv - 1] != high_idx:
            pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = high_idx, high_price, 1
            n_piv += 1
    elif trend == -1:
        if n_piv == 0 or pivots_idx[n_piv - 1] != low_idx:
            pivots_idx[n_piv], pivots_price[n_piv], pivots_type[n_piv] = low_idx, low_price, 2
            n_piv += 1

    return pivots_idx[:n_piv], pivots_price[:n_piv], pivots_type[:n_piv]


def _filter_alternating(idx, price, ptype):
    if len(idx) == 0:
        return idx, price, ptype
    out_idx, out_price, out_type = [idx[0]], [price[0]], [ptype[0]]
    for i in range(1, len(idx)):
        if ptype[i] == out_type[-1]:
            if ptype[i] == 1 and price[i] > out_price[-1]:
                out_idx[-1], out_price[-1] = idx[i], price[i]
            elif ptype[i] == 2 and price[i] < out_price[-1]:
                out_idx[-1], out_price[-1] = idx[i], price[i]
        else:
            out_idx.append(idx[i])
            out_price.append(price[i])
            out_type.append(ptype[i])
    return np.array(out_idx), np.array(out_price), np.array(out_type)


def _fresh_entry_mask(side_state):
    fresh = np.zeros(len(side_state), dtype=bool)
    fresh[0] = side_state[0] != 0
    fresh[1:] = (side_state[1:] != 0) & (side_state[1:] != side_state[:-1])
    return fresh


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(panel)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    threshold = np.where(np.isfinite(vol), VOL_MULTIPLIER * vol, MIN_REVERSAL_PCT)

    idx, price, ptype = _zigzag_pivots_correctedvol(close, threshold, MIN_REVERSAL_PCT)
    idx, price, ptype = _filter_alternating(idx, price, ptype)

    label = np.zeros(n, dtype=np.int8)
    for k in range(len(idx) - 1):
        idx_s, type_s = idx[k], ptype[k]
        idx_e, type_e = idx[k + 1], ptype[k + 1]
        bars = idx_e - idx_s
        if bars < MIN_WAVE_BARS:
            continue
        if type_s == 2 and type_e == 1:
            side = 1
        elif type_s == 1 and type_e == 2:
            side = 2
        else:
            continue
        label[idx_s:idx_e] = side

    if TRANSITION_BUFFER > 0:
        change = np.flatnonzero(label != np.roll(label, 1))
        change = change[change > 0]
        for c in change:
            lo, hi = max(0, c - TRANSITION_BUFFER), min(n, c + TRANSITION_BUFFER + 1)
            label[lo:hi] = 0

    out = pd.DataFrame({"timestamp": panel["timestamp"], "zigzag_correctedvol_action": label})
    out.to_parquet(OUT_PATH, index=False)

    counts = pd.Series(label).value_counts(normalize=True).sort_index()
    print(json.dumps({
        "rows": int(n), "n_pivots": int(len(idx)),
        "ratios": {"CASH": float(counts.get(0, 0)), "LONG": float(counts.get(1, 0)), "SHORT": float(counts.get(2, 0))},
    }, indent=2))

    tp_moves_all, sl_moves_all = TP_MULT * vol, SL_MULT * vol
    side_state_full = np.where(label == 1, 1, np.where(label == 2, -1, 0))
    ts = panel["timestamp"]
    for split, mask in [
        ("val", (ts >= VAL_START).to_numpy() & (ts <= VAL_END).to_numpy()),
        ("oos", (ts >= OOS_START).to_numpy() & (ts <= OOS_END).to_numpy()),
    ]:
        row_idx = np.flatnonzero(mask)
        side_state = side_state_full[row_idx]
        fresh = _fresh_entry_mask(side_state)
        e_idx, e_side = row_idx[fresh], side_state[fresh]
        tp, sl = tp_moves_all[e_idx], sl_moves_all[e_idx]
        finite = np.isfinite(tp) & np.isfinite(sl)
        e_idx, e_side, tp, sl = e_idx[finite], e_side[finite], tp[finite], sl[finite]
        result = simulate_single_position(
            timestamps=panel["timestamp"], open_px=panel["open"].to_numpy(dtype=np.float64),
            high=panel["high"].to_numpy(dtype=np.float64), low=panel["low"].to_numpy(dtype=np.float64),
            close=close, decision_indices=e_idx, scores=e_side.astype(np.float64), tp_moves=tp, sl_moves=sl,
            upper_threshold=0.0, lower_threshold=0.0, horizon_bars=HORIZON_BARS,
            margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        )
        ledger = result.ledger
        if len(ledger) == 0:
            print(split, "no trades")
            continue
        equity = result.equity
        running_max = np.maximum.accumulate(equity)
        mdd = float(((equity - running_max) / running_max).min() * 100)
        print(split, "n_trades=", len(ledger), "win_rate=%.4f" % (ledger["trade_return"] > 0).mean(),
              "sum_ret=%.2f%%" % (ledger["trade_return"].sum() * 100), "final_equity=%.2fx" % equity[-1], "mdd=%.2f%%" % mdd)

    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
