"""
Hybrid BTC label/entry scheme combining three already-vetted-independently
pieces into a new (not previously tried) combination:
  - CUSUM event filter: causal, early detection of "a wave may be starting"
    (same filter used in cusum_filtered_tb from the label comparison).
  - trend-scan (mtf1h_ts_action, already causal, already merged into the
    feature frame): direction confirmation -- only take the CUSUM event if
    the current 1h trend-scan direction agrees.
  - zigzag-style ATR-adaptive trailing exit: instead of h48qual's fixed
    4h barrier or zigzag's hindsight wave-end, walk forward bar-by-bar from
    entry and exit when price pulls back from the running favorable extreme
    by an ATR-adaptive threshold (same threshold formula as zigzag's
    _zigzag_pivots), giving a "one wave = one trade" shape like zigzag but
    computed causally (no peeking at the future pivot).

Every event that fires becomes a trade -- there is no post-hoc quality/edge
filter here (unlike h48qual/zigzag/cusum_filtered_tb), so win rate reported
below is an honest forward-simulated number, not a hindsight tautology.

Diagnostic/dev-score only. Not a Fresh-Forward validated promotion claim.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from compare_btc_label_schemes_20260803 import cusum_events  # noqa: E402

FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_5m1h_mtf_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_cusum_trendscan_zigzag_hybrid_trades_20260803.csv"

TRAIL_MIN_PCT, TRAIL_MAX_PCT, TRAIL_MULT = 0.010, 0.018, 1.0  # same as zigzag_v2 defaults
HARD_SL_MULT, HARD_SL_MIN = 0.8, 0.004  # same floor as h48_conservative
MAX_HOLD_BARS = 288  # 24h cap, generous vs zigzag's ~232-bar avg hold


def trail_threshold(atr_pct_i: float) -> float:
    return float(np.clip(max(TRAIL_MIN_PCT, atr_pct_i * TRAIL_MULT), TRAIL_MIN_PCT, TRAIL_MAX_PCT))


def simulate_trade(side: int, entry: float, atr_pct: np.ndarray, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, start_i: int, sl_move: float) -> tuple[float, str, int]:
    n = len(close)
    if side == 1:
        sl_level = entry * (1.0 - sl_move)
        extreme = entry
        for k in range(1, MAX_HOLD_BARS + 1):
            i = start_i + k
            if i >= n:
                return close[i - 1] / entry - 1.0, "eod", k - 1
            if low[i] <= sl_level:
                return -sl_move, "sl", k
            extreme = max(extreme, high[i])
            trail = trail_threshold(atr_pct[i])
            if close[i] <= extreme * (1.0 - trail):
                return close[i] / entry - 1.0, "trail_exit", k
        return close[start_i + MAX_HOLD_BARS] / entry - 1.0, "timeout", MAX_HOLD_BARS
    else:
        sl_level = entry * (1.0 + sl_move)
        extreme = entry
        for k in range(1, MAX_HOLD_BARS + 1):
            i = start_i + k
            if i >= n:
                return 1.0 - close[i - 1] / entry, "eod", k - 1
            if high[i] >= sl_level:
                return -sl_move, "sl", k
            extreme = min(extreme, low[i])
            trail = trail_threshold(atr_pct[i])
            if close[i] >= extreme * (1.0 + trail):
                return 1.0 - close[i] / entry, "trail_exit", k
        return 1.0 - close[start_i + MAX_HOLD_BARS] / entry, "timeout", MAX_HOLD_BARS


def main():
    frame = pd.read_parquet(FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    ts_action = frame["mtf1h_ts_action"].fillna(0).to_numpy()
    ts = frame["timestamp"]

    atr = _atr_price_move(frame)  # used for CUSUM threshold + hard SL
    atr_window14 = frame["close"].pct_change().rolling(14).std().fillna(atr.mean()).to_numpy()  # cheap proxy for trail ATR window
    # use the same 96-window ATR% for the trailing threshold too (consistent, already causal-shifted)
    trail_atr = atr

    events = cusum_events(frame, atr, mult=2.0)

    rows = []
    last_exit_i = -1
    for ev_i in events:
        entry_i = ev_i + 1
        if entry_i <= last_exit_i or entry_i + 1 >= n:
            continue  # enforce single-position-at-a-time (skip events while a trade is open)
        side_signal = ts_action[ev_i]
        if side_signal not in (1, 2):
            continue
        entry = float(open_px[entry_i])
        vol = float(atr[ev_i])
        sl_move = max(HARD_SL_MIN, HARD_SL_MULT * vol)
        ret, reason, bars = simulate_trade(int(side_signal), entry, trail_atr, high, low, close, entry_i, sl_move)
        exit_i = min(entry_i + bars, n - 1)
        rows.append({
            "entry_ts": ts.iloc[entry_i], "exit_ts": ts.iloc[exit_i], "side": int(side_signal),
            "ret": ret, "reason": reason, "bars": bars,
        })
        last_exit_i = exit_i

    trades = pd.DataFrame(rows)
    trades.to_csv(OUT_CSV, index=False)

    n_trades = len(trades)
    win = (trades["ret"] > 0).sum() if n_trades else 0
    print(f"total CUSUM events: {len(events)}, trend-scan-confirmed+non-overlapping trades: {n_trades}")
    print(f"long/short split: {int((trades['side']==1).sum())}/{int((trades['side']==2).sum())}" if n_trades else "n/a")
    print(f"win rate (honest, forward-simulated): {100*win/max(n_trades,1):.1f}%")
    print(f"mean ret: {100*trades['ret'].mean():.3f}%  median ret: {100*trades['ret'].median():.3f}%")
    print(f"mean hold bars: {trades['bars'].mean():.1f} ({trades['bars'].mean()*5/60:.1f}h)")
    print(f"exit reason counts:\n{trades['reason'].value_counts()}")
    print(f"implied trades/year: {n_trades / (n / (288*365)):.1f}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
