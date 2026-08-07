"""BTC 5m candidate #1: CUSUM momentum-continuation events + barrier exit.

Unlike zigzag (which needs FUTURE confirmation to know a pivot was real -- hindsight/oracle
label), a CUSUM filter event is known the INSTANT the cumulative sum crosses threshold -- fully
causal, no hindsight required. Direction = momentum continuation (trade the SAME direction as the
move that triggered the event), the opposite philosophy from zigzag's mean-reversion-at-pivot.

This is therefore a real, zero-ML rule-based backtest (not an oracle upper bound) -- reported
honestly as such, distinct from the zigzag/quality-meta numbers which remain hindsight labels.

Standard symmetric CUSUM filter (Lopez de Prado): S+_t = max(0, S+_{t-1} + r_t), reset to 0 on
event; S-_t = min(0, S-_{t-1} + r_t), reset to 0 on event. Event when S+_t >= threshold(t) (UP) or
S-_t <= -threshold(t) (DOWN). Same ATR-adaptive threshold function as zigzag for consistency.
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
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_cusum_momentum_labels_20260806.parquet"

MIN_REVERSAL_PCT = 0.009
ATR_WINDOW = 14
ATR_MULTIPLIER = 1.0
TP_MULT, SL_MULT = 2.5, 1.2
MAX_HOLD = 288  # 24h at 5m
ROUND_TRIP_COST = 0.0010


def main() -> int:
    frame = pd.read_parquet(PANEL_PATH, columns=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    n = len(frame)

    atr_pct = zigzag._atr_pct(frame, ATR_WINDOW)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))

    s_pos, s_neg = 0.0, 0.0
    event = np.zeros(n, dtype=np.int8)  # 0=none, 1=UP, -1=DOWN
    for i in range(1, n):
        thr = max(MIN_REVERSAL_PCT, float(atr_pct[i]) * ATR_MULTIPLIER)
        s_pos = max(0.0, s_pos + log_ret[i])
        s_neg = min(0.0, s_neg + log_ret[i])
        if s_pos >= thr:
            event[i] = 1
            s_pos, s_neg = 0.0, 0.0
        elif s_neg <= -thr:
            event[i] = -1
            s_pos, s_neg = 0.0, 0.0

    # simulate barrier trade at each event (real, causal -- no hindsight)
    net_ret = np.full(n, np.nan)
    for i in range(n - MAX_HOLD - 1):
        d = int(event[i])
        if d == 0:
            continue
        entry_price = close[i]
        tp_price = entry_price * (1 + d * TP_MULT * atr_pct[i])
        sl_price = entry_price * (1 - d * SL_MULT * atr_pct[i])
        exit_price = None
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            hit_tp = high[j] >= tp_price if d == 1 else low[j] <= tp_price
            hit_sl = low[j] <= sl_price if d == 1 else high[j] >= sl_price
            if hit_tp and hit_sl:
                exit_price = sl_price
                break
            if hit_tp:
                exit_price = tp_price
                break
            if hit_sl:
                exit_price = sl_price
                break
        if exit_price is None:
            exit_price = close[j_end]
        net_ret[i] = d * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST

    out = frame[["timestamp", "close"]].copy()
    out["cusum_event"] = event
    out["cusum_net_ret"] = net_ret
    out.to_parquet(OUT_PATH, index=False)

    active = out.dropna(subset=["cusum_net_ret"])
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(f"n_events: {len(active)} (UP={int((active['cusum_event']==1).sum())}, DOWN={int((active['cusum_event']==-1).sum())})")
    print(f"win rate: {(active['cusum_net_ret']>0).mean():.4f}")
    print(f"mean net ret/trade: {active['cusum_net_ret'].mean()*100:.4f}%")
    print(f"total net ret (all events, full sample, 1 trade/event): {active['cusum_net_ret'].sum()*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
