"""BTC 5m candidate #2: oracle zigzag direction + oracle quality meta-label filter (both
hindsight/oracle -- this measures the CEILING a working meta-label filter could reach, not a
live-testable number, since both zigzag pivots and the barrier-simulated quality outcome require
future price data to compute).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIGZAG_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet"
OUT_PATH = ROOT / "data/splits/year_oos/btc_5m_zigzag_quality_oracle_20260806.parquet"

TP_MULT, SL_MULT = 2.5, 1.2
MAX_HOLD = 288
ROUND_TRIP_COST = 0.0010
TRAIL_VOL_BARS = 288


def main() -> int:
    zz = pd.read_parquet(ZIGZAG_PATH, columns=["timestamp", "open", "high", "low", "close", "zigzag_action"])
    zz = zz.sort_values("timestamp").reset_index(drop=True)
    log_ret = np.log(zz["close"]).diff()
    zz["trailing_vol"] = log_ret.rolling(TRAIL_VOL_BARS, min_periods=TRAIL_VOL_BARS).std()

    n = len(zz)
    close = zz["close"].to_numpy()
    high = zz["high"].to_numpy()
    low = zz["low"].to_numpy()
    vol_arr = zz["trailing_vol"].to_numpy()
    action = zz["zigzag_action"].to_numpy()

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

    zz["net_ret_sim"] = net_ret
    zz["quality"] = quality
    out = zz[["timestamp", "close", "zigzag_action", "net_ret_sim", "quality"]]
    out.to_parquet(OUT_PATH, index=False)

    active = out.dropna(subset=["quality"])
    print(f"wrote {OUT_PATH}, shape={out.shape}")
    print(f"active rows: {len(active)}")
    print(f"quality positive rate: {active['quality'].mean():.4f}")
    print(f"mean net_ret (ALL active bars, unfiltered): {active['net_ret_sim'].mean()*100:.4f}%")
    good = active[active["quality"] == 1]
    print(f"mean net_ret (quality=1 subset only): {good['net_ret_sim'].mean()*100:.4f}%  n={len(good)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
