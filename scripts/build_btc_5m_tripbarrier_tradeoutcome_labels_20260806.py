"""Fundamental fix for the two root causes found in
docs/btc_deepfeat_cnn_transformer_zigzag_soft_label_20260806.md's acc-vs-PnL gap diagnosis
(project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806 memory):

Root cause 1 (structural): the zigzag wave label is retrospective ("which confirmed wave does
this bar sit inside") -- at fresh-entry bars the model hit 76.4% wave-classification accuracy but
only 45-58% on simple forward-return-sign hit rate. Fix: replace it with a genuinely forward-
looking, backtest-consistent CAUSAL TRIPLE-BARRIER label -- for every bar, simulate opening a LONG
and a SHORT right now (entry at next bar's open, matching core/causal_futures_backtest.py's own
convention) and record which side (if either) hits its own TP before its own SL within the
horizon. This is EXACTLY the question a trading classifier needs answered, not a proxy for it.

Root cause 3 (mechanical): SL was sized as `SL_MULT * std(1-bar log returns)` -- a single 5-minute
bar's volatility with only a 1.2x multiplier is far too tight for a multi-bar hold (median SL
distance 0.174% vs median/mean realized 3-bar move of 0.109%/0.167%, i.e. 61.8% of SL exits fired
within 15 minutes of entry: noise stop-outs, not real reversals). Fix: size TP/SL off the rolling
dispersion of 12-BAR (1h) cumulative log returns (still estimated causally over a 288-bar/24h
lookback), which scales correctly with a real multi-bar hold instead of a single bar's noise.
Sanity check: median new SL distance 0.489% comfortably exceeds median/mean 3-bar noise
(0.109%/0.167%) and even 12-bar noise (0.212%/0.328%).

TP_MULT/SL_MULT (2.5/1.2) and horizon (288 bars = 24h) kept identical to the existing backtest so
the label is exactly what the strategy would experience -- no train/live mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_20260806.parquet"

CUMRET_BARS = 12  # 1h -- the multi-bar return-dispersion window TP/SL are sized against
VOL_LOOKBACK = 288  # 24h -- causal rolling window used to estimate that dispersion's typical scale
TP_MULT = 2.5
SL_MULT = 1.2
HORIZON_BARS = 288  # 24h max hold, matches backtest
SOFT_TEMPERATURE = 0.35  # lower = more peaked soft distribution; see _race_conviction docstring


@numba.njit(cache=True)
def _triple_barrier_race(open_, high, low, tp_move, sl_move, horizon):
    """Same hard triple-barrier outcome as before, but also returns a per-side CONVICTION score
    derived from how fast each side's own race resolved: `long_score`/`short_score` in
    [-1, +1] -- sign is which barrier was hit first (+1=TP, -1=SL), magnitude is
    `1 - bars_to_resolution/horizon` (an instant TP/SL hit scores near +-1; a resolution just
    before horizon scores near 0; an unresolved timeout scores exactly 0, i.e. no conviction
    either way). This replaces the old flat epsilon-smoothed one-hot with a genuinely graded soft
    target -- e.g. a LONG that hits TP within 2 bars is a much more confident LONG than one that
    barely hits TP at bar 250 of a 288-bar horizon, even though both get hard label LONG."""
    n = len(open_)
    label = np.zeros(n, dtype=np.int8)  # 0=CASH, 1=LONG, 2=SHORT
    long_score = np.zeros(n, dtype=np.float64)
    short_score = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        entry_i = i + 1
        if not np.isfinite(tp_move[i]) or not np.isfinite(sl_move[i]):
            continue
        entry = open_[entry_i]
        tp_l = entry * (1.0 + tp_move[i])
        sl_l = entry * (1.0 - sl_move[i])
        tp_s = entry * (1.0 - tp_move[i])
        sl_s = entry * (1.0 + sl_move[i])
        long_done = False
        long_sign = 0
        long_t = horizon
        short_done = False
        short_sign = 0
        short_t = horizon
        final_i = entry_i + horizon - 1
        if final_i >= n:
            final_i = n - 1
        for j in range(entry_i, final_i + 1):
            t = j - entry_i + 1
            if not long_done:
                if low[j] <= sl_l:
                    long_done, long_sign, long_t = True, -1, t
                elif high[j] >= tp_l:
                    long_done, long_sign, long_t = True, 1, t
            if not short_done:
                if high[j] >= sl_s:
                    short_done, short_sign, short_t = True, -1, t
                elif low[j] <= tp_s:
                    short_done, short_sign, short_t = True, 1, t
            if long_done and short_done:
                break
        long_score[i] = long_sign * (1.0 - long_t / horizon) if long_done else 0.0
        short_score[i] = short_sign * (1.0 - short_t / horizon) if short_done else 0.0
        long_tp = long_done and long_sign == 1
        short_tp = short_done and short_sign == 1
        if long_tp and not short_tp:
            label[i] = 1
        elif short_tp and not long_tp:
            label[i] = 2
        else:
            label[i] = 0
    return label, long_score, short_score


def main() -> int:
    panel = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
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
    # cash_score is a fixed 0 baseline, not a function of long/short_score: by construction a bar's
    # hard label can only be LONG/SHORT when that side's score is STRICTLY positive (it hit its own
    # TP), and CASH means neither side did (so both scores are <=0, or 0 on timeout) -- a fixed 0
    # baseline is exactly the boundary between "some side won" and "nobody won", so
    # argmax([0, long_score, short_score]) reproduces the hard label almost exactly (the only
    # exception is the rare same-bar double-TP whipsaw, which is CASH by the hard-label tie-break
    # rule but can score positive on both sides).
    cash_score = np.zeros(n, dtype=np.float64)
    logits = np.stack([cash_score, long_score, short_score], axis=1) / SOFT_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    soft = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    argmax_matches_hard = float((soft.argmax(axis=1) == label).mean())
    mean_maxprob_by_class = {
        name: float(soft[label == cls].max(axis=1).mean()) if (label == cls).any() else None
        for cls, name in ((0, "CASH"), (1, "LONG"), (2, "SHORT"))
    }

    out = pd.DataFrame({
        "timestamp": panel["timestamp"],
        "trade_outcome_action": label,
        "trade_outcome_soft_cash": soft[:, 0],
        "trade_outcome_soft_long": soft[:, 1],
        "trade_outcome_soft_short": soft[:, 2],
    })
    out.to_parquet(OUT_PATH, index=False)

    counts = pd.Series(label).value_counts().sort_index()
    summary = {
        "rows": int(n),
        "counts": {"CASH": int(counts.get(0, 0)), "LONG": int(counts.get(1, 0)), "SHORT": int(counts.get(2, 0))},
        "ratios": {k: v / n for k, v in {"CASH": int(counts.get(0, 0)), "LONG": int(counts.get(1, 0)), "SHORT": int(counts.get(2, 0))}.items()},
        "cumret_bars": CUMRET_BARS,
        "vol_lookback": VOL_LOOKBACK,
        "tp_mult": TP_MULT,
        "sl_mult": SL_MULT,
        "horizon_bars": HORIZON_BARS,
        "median_tp_move_pct": float(np.nanmedian(tp_move) * 100),
        "median_sl_move_pct": float(np.nanmedian(sl_move) * 100),
        "soft_temperature": SOFT_TEMPERATURE,
        "soft_argmax_matches_hard_label": argmax_matches_hard,
        "mean_maxprob_by_class": mean_maxprob_by_class,
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
