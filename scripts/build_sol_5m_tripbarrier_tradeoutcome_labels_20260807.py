"""SOL port of scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py.

Same corrected causal triple-barrier trade-outcome label (contract
docs/experiments/sol_dl_rl_architecture_survey_20260807.json): for every bar, simulate opening a
LONG and a SHORT at next bar's open and record which side (if either) hits its own TP before its
own SL within the horizon. TP/SL are sized off the rolling dispersion of 12-bar (1h) cumulative
log returns over a causal 288-bar lookback -- the 2026-08-06 fix for single-bar-noise SL sizing.
Constants (CUMRET 12 / LOOKBACK 288 / TP 2.5 / SL 1.2 / horizon 288) are kept identical to BTC so
the SOL result is comparable architecture-for-architecture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/sol_features_2024_2026.csv"
OUT_PATH = ROOT / "data/splits/year_oos/sol_5m_tripbarrier_tradeoutcome_labels_20260807.parquet"

CUMRET_BARS = 12
VOL_LOOKBACK = 288
TP_MULT = 2.5
SL_MULT = 1.2
HORIZON_BARS = 288
SOFT_TEMPERATURE = 0.35


def _triple_barrier_race(open_, high, low, tp_move, sl_move, horizon):
    """Vectorized port of the numba loop in the BTC 20260806 builder (numba is broken with the
    venv's NumPy 2.3). Per-bar semantics preserved exactly: SL is checked before TP within a bar,
    each side races its own barriers independently, unresolved-by-horizon scores 0."""
    n = len(open_)
    entry = np.full(n, np.nan)
    entry[:-1] = open_[1:]
    valid = np.isfinite(tp_move) & np.isfinite(sl_move) & np.isfinite(entry)
    tp_l = entry * (1.0 + tp_move)
    sl_l = entry * (1.0 - sl_move)
    tp_s = entry * (1.0 - tp_move)
    sl_s = entry * (1.0 + sl_move)

    long_sign = np.zeros(n, dtype=np.int8)
    short_sign = np.zeros(n, dtype=np.int8)
    long_t = np.full(n, horizon, dtype=np.int32)
    short_t = np.full(n, horizon, dtype=np.int32)
    long_done = np.zeros(n, dtype=bool)
    short_done = np.zeros(n, dtype=bool)

    for t in range(1, horizon + 1):
        m = n - t  # decision bars i in [0, m-1] have bar i+t available
        if m <= 0:
            break
        lo = low[t : t + m]
        hi = high[t : t + m]
        upd_l = valid[:m] & ~long_done[:m]
        sl_hit = upd_l & (lo <= sl_l[:m])
        tp_hit = upd_l & ~sl_hit & (hi >= tp_l[:m])
        for hit, sign in ((sl_hit, -1), (tp_hit, 1)):
            long_sign[:m][hit] = sign
            long_t[:m][hit] = t
        long_done[:m] |= sl_hit | tp_hit
        upd_s = valid[:m] & ~short_done[:m]
        sl_hit_s = upd_s & (hi >= sl_s[:m])
        tp_hit_s = upd_s & ~sl_hit_s & (lo <= tp_s[:m])
        for hit, sign in ((sl_hit_s, -1), (tp_hit_s, 1)):
            short_sign[:m][hit] = sign
            short_t[:m][hit] = t
        short_done[:m] |= sl_hit_s | tp_hit_s

    long_score = np.where(long_done, long_sign * (1.0 - long_t / horizon), 0.0)
    short_score = np.where(short_done, short_sign * (1.0 - short_t / horizon), 0.0)
    long_tp = long_done & (long_sign == 1)
    short_tp = short_done & (short_sign == 1)
    label = np.zeros(n, dtype=np.int8)
    label[long_tp & ~short_tp] = 1
    label[short_tp & ~long_tp] = 2
    return label, long_score, short_score


def main() -> int:
    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "open", "high", "low", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    open_ = panel["open"].to_numpy(dtype=np.float64)
    high = panel["high"].to_numpy(dtype=np.float64)
    low = panel["low"].to_numpy(dtype=np.float64)
    close = panel["close"].to_numpy(dtype=np.float64)

    log_ret_1bar = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret_1bar).rolling(CUMRET_BARS).sum().to_numpy()
    vol = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    tp_move = TP_MULT * vol
    sl_move = SL_MULT * vol

    label, long_score, short_score = _triple_barrier_race(open_, high, low, tp_move, sl_move, HORIZON_BARS)

    n = len(label)
    cash_score = np.zeros(n, dtype=np.float64)
    logits = np.stack([cash_score, long_score, short_score], axis=1) / SOFT_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    soft = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    out = pd.DataFrame({
        "timestamp": panel["timestamp"],
        "trade_outcome_action": label,
        "trade_outcome_soft_cash": soft[:, 0],
        "trade_outcome_soft_long": soft[:, 1],
        "trade_outcome_soft_short": soft[:, 2],
        "tp_move": tp_move,
        "sl_move": sl_move,
    })
    out.to_parquet(OUT_PATH, index=False)

    counts = pd.Series(label).value_counts().sort_index()
    summary = {
        "rows": int(n),
        "counts": {"CASH": int(counts.get(0, 0)), "LONG": int(counts.get(1, 0)), "SHORT": int(counts.get(2, 0))},
        "median_tp_move_pct": float(np.nanmedian(tp_move) * 100),
        "median_sl_move_pct": float(np.nanmedian(sl_move) * 100),
        "soft_argmax_matches_hard_label": float((soft.argmax(axis=1) == label).mean()),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
