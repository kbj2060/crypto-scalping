"""BTC 1h Layer B v2: quality meta-label on top of the raw zigzag direction call, replicating
ETH's live zig075 architecture (scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py
::_quality_target_barrier_meta_action) instead of our naive hard-argmax 3-class classifier, which
lost -20%/-21% on VAL/OOS with no profitability filter at all.

For every bar where the raw zigzag_action (build_btc_1h_zigzag_labels_20260805.py) is LONG or
SHORT, simulate the SAME TP/SL barrier trade this session has used throughout (ATR/vol-sized TP/SL,
MAX_HOLD bars, ROUND_TRIP_COST) and label quality=1 iff the simulated net return is positive.
Quality=0 (including all CASH bars) is the "don't trade this" class. A quality classifier is then
trained to predict, ex-ante, whether a given raw signal is worth taking -- ETH's zig075 requires
this filter to be profitable; testing whether BTC 1h does too.

Note: BTC's zig075 dual-component already failed decisively on 5m data
(project-btc-zigzag-dual-component-already-failed-20260802) -- this is native-1h + DVOL, untested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIGZAG_LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_zigzag_labels_20260805.parquet"
VOLREGIME_LABEL_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_1h_zigzag_quality_meta_labels_20260806.parquet"

TP_MULT, SL_MULT = 2.5, 1.2
MAX_HOLD = 24
ROUND_TRIP_COST = 0.0010


def main() -> int:
    zz = pd.read_parquet(ZIGZAG_LABEL_PATH, columns=["timestamp", "open", "high", "low", "close", "zigzag_action"])
    vol = pd.read_parquet(VOLREGIME_LABEL_PATH, columns=["timestamp", "trailing_vol_24h"])
    df = zz.merge(vol, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    vol_arr = df["trailing_vol_24h"].to_numpy()
    action = df["zigzag_action"].to_numpy()

    quality = np.full(n, np.nan)
    net_ret = np.full(n, np.nan)

    for i in range(n - MAX_HOLD - 1):
        a = int(action[i])
        if a not in (1, 2):
            continue
        v = vol_arr[i]
        if not np.isfinite(v) or v <= 0:
            continue
        direction = 1 if a == 1 else -1
        entry_price = close[i]
        tp_price = entry_price * (1 + direction * TP_MULT * v)
        sl_price = entry_price * (1 - direction * SL_MULT * v)

        exit_price = None
        j_end = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, j_end + 1):
            hit_tp = high[j] >= tp_price if direction == 1 else low[j] <= tp_price
            hit_sl = low[j] <= sl_price if direction == 1 else high[j] >= sl_price
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

        ret = direction * (exit_price - entry_price) / entry_price - ROUND_TRIP_COST
        net_ret[i] = ret
        quality[i] = 1.0 if ret > 0 else 0.0

    df["net_ret_sim"] = net_ret
    df["quality"] = quality

    out = df[["timestamp", "zigzag_action", "net_ret_sim", "quality"]]
    out.to_parquet(OUT_PATH, index=False)

    active = out.dropna(subset=["quality"])
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(f"active (LONG/SHORT) rows: {len(active)} ({len(active)/n:.1%} of all bars)")
    print(f"quality positive rate (active only): {active['quality'].mean():.4f}")
    print(f"mean net_ret_sim (active only): {active['net_ret_sim'].mean()*100:.4f}%")
    print(f"by action:\n{active.groupby('zigzag_action')[['quality','net_ret_sim']].mean()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
