"""Follow-up to train_eval_btc_simple_squeeze_tradeintensity_20260804.py: that test found the
squeeze_power+trade_intensity composite score consistently negative on VAL (2025-09..12) but
consistently positive and strengthening on OOS (2026-01..03) at all 3 percentile thresholds tested.
Break VAL+OOS down by calendar month (same TRAIN-fit thresholds, same fixed 576-bar horizon, same
conservative cost) to find where/if the sign actually flips, rather than treating VAL and OOS as
two monolithic blocks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"

VAL_START = pd.Timestamp("2025-09-01")
RANGE_END = pd.Timestamp("2026-04-01")  # covers VAL + OOS
HORIZON = 576
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0
ZWINDOW = 8640
ZMIN_PERIODS = 2880
STRIDE = 3
PERCENTILES = [0.80, 0.90, 0.95]


def causal_zscore(s: pd.Series) -> pd.Series:
    mean = s.rolling(ZWINDOW, min_periods=ZMIN_PERIODS).mean()
    std = s.rolling(ZWINDOW, min_periods=ZMIN_PERIODS).std()
    return (s - mean) / std.replace(0, np.nan)


def main() -> int:
    frame = pd.read_parquet(BTC_FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)

    z_squeeze = causal_zscore(frame["squeeze_power"])
    z_intensity = causal_zscore(frame["trade_intensity"])
    score = -(z_squeeze + z_intensity) / 2.0

    open_px = frame["open"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    ts = frame["timestamp"]

    idxs = np.arange(0, n - HORIZON - 2, STRIDE)
    entry_i = idxs + 1
    end_i = entry_i + HORIZON - 1
    valid = end_i < n
    idxs, entry_i, end_i = idxs[valid], entry_i[valid], end_i[valid]

    fwd_ret = close[end_i] / open_px[entry_i] - 1.0
    data = pd.DataFrame({
        "timestamp": ts.iloc[idxs].to_numpy(), "score": score.iloc[idxs].to_numpy(), "fwd_ret": fwd_ret,
    }).dropna(subset=["score"])

    train = data[data["timestamp"] < VAL_START]
    window = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < RANGE_END)].copy()
    window["month"] = window["timestamp"].dt.to_period("M")

    print(f"train={len(train)} (thresholds fit here, never re-fit on VAL/OOS)")

    for pct in PERCENTILES:
        long_thresh = train["score"].quantile(pct)
        short_thresh = train["score"].quantile(1 - pct)
        print(f"\n=== percentile={pct:.2f}  long_thresh={long_thresh:.3f}  short_thresh={short_thresh:.3f} ===")
        rows = []
        for month, grp in window.groupby("month"):
            take_long = grp["score"] >= long_thresh
            take_short = grp["score"] <= short_thresh
            n_trades = int(take_long.sum() + take_short.sum())
            if n_trades == 0:
                continue
            long_net = grp.loc[take_long, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            short_net = -grp.loc[take_short, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            net = np.concatenate([long_net, short_net])
            rows.append({
                "month": str(month), "n_trades": n_trades,
                "n_long": int(take_long.sum()), "n_short": int(take_short.sum()),
                "win_pct": 100 * (net > 0).sum() / n_trades, "mean_net_pct": 100 * net.mean(),
                "sum_net_pct": 100 * net.sum(),
            })
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
