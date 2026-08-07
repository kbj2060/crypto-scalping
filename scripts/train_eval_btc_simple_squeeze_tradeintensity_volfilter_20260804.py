"""Follow-up to diagnose_btc_simple_signal_monthly_20260804.py: the Sep-Oct 2025 losing stretch
for the squeeze_power+trade_intensity composite score lines up with a realized-vol spike
(26.8%->55.3% annualized) and a single flash-crash day (2025-10-10, -7.3%/17.3% range). Test
whether gating entries out of high-trailing-realized-vol regimes recovers/improves the strategy,
using only causally-known information at entry time (trailing realized vol up to the signal bar,
same convention as vol_risk_premium in build_btc_dvol_features_20260804.py).

Vol filter: skip any trade whose entry-bar trailing realized_vol_288 (24h rolling std of 5m log
returns, annualized) exceeds a percentile fit on TRAIN only (never re-fit on VAL/OOS, same
discipline as the score thresholds). Composite score threshold fixed at percentile=0.90 (the
middle of the 3 tested previously) to isolate the effect of the vol filter alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_simple_squeeze_tradeintensity_volfilter_20260804.csv"

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
HORIZON = 576
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0
ZWINDOW = 8640
ZMIN_PERIODS = 2880
STRIDE = 3
SCORE_PCT = 0.90
RVOL_WINDOW = 288  # 24h, same definition as realized_vol_288 used elsewhere this session
VOL_FILTER_PCTS = [None, 0.75, 0.85, 0.90]  # None = no filter (baseline); else skip entries above this TRAIN percentile


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

    log_ret = np.log(frame["close"] / frame["close"].shift(1))
    rvol = log_ret.rolling(RVOL_WINDOW, min_periods=RVOL_WINDOW // 2).std() * np.sqrt(288 * 365)

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
        "i": idxs, "timestamp": ts.iloc[idxs].to_numpy(), "score": score.iloc[idxs].to_numpy(),
        "rvol": rvol.iloc[idxs].to_numpy(), "fwd_ret": fwd_ret,
    }).dropna(subset=["score", "rvol"])

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]

    long_thresh = train["score"].quantile(SCORE_PCT)
    short_thresh = train["score"].quantile(1 - SCORE_PCT)
    print(f"score thresholds (TRAIN-fit): long>={long_thresh:.3f} short<={short_thresh:.3f}")
    print(f"TRAIN rvol distribution: {train['rvol'].describe(percentiles=[0.5, 0.75, 0.9])[['mean', '50%', '75%', '90%', 'max']].to_string()}")

    all_results = []
    for vol_pct in VOL_FILTER_PCTS:
        rvol_cap = train["rvol"].quantile(vol_pct) if vol_pct is not None else np.inf
        label = "no_filter" if vol_pct is None else f"rvol<=p{int(vol_pct*100)}({rvol_cap:.3f})"
        print(f"\n=== {label} ===")

        for split_name, split in [("VAL", val), ("OOS", oos)]:
            take_long = (split["score"] >= long_thresh) & (split["rvol"] <= rvol_cap)
            take_short = (split["score"] <= short_thresh) & (split["rvol"] <= rvol_cap)
            n_trades = int(take_long.sum() + take_short.sum())
            if n_trades == 0:
                print(f"  [{split_name}] no trades")
                continue
            long_net = split.loc[take_long, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            short_net = -split.loc[take_short, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            net = np.concatenate([long_net, short_net])
            win = (net > 0).sum()
            all_results.append({
                "vol_filter": label, "split": split_name, "n_trades": n_trades,
                "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                "sum_net_pct": 100 * net.sum(),
            })
            print(f"  [{split_name}] n={n_trades:5d} win%={100*win/n_trades:5.1f} "
                  f"mean_net={100*net.mean():6.3f}% sum_net={100*net.sum():8.2f}%")

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    # monthly breakdown for the best-looking filter (widest recovery of VAL) -- print all filters' monthly too
    window = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_END)].copy()
    window["month"] = window["timestamp"].dt.to_period("M")
    for vol_pct in VOL_FILTER_PCTS:
        rvol_cap = train["rvol"].quantile(vol_pct) if vol_pct is not None else np.inf
        label = "no_filter" if vol_pct is None else f"rvol<=p{int(vol_pct*100)}"
        print(f"\n--- monthly, {label} ---")
        rows = []
        for month, grp in window.groupby("month"):
            take_long = (grp["score"] >= long_thresh) & (grp["rvol"] <= rvol_cap)
            take_short = (grp["score"] <= short_thresh) & (grp["rvol"] <= rvol_cap)
            n_trades = int(take_long.sum() + take_short.sum())
            if n_trades == 0:
                continue
            long_net = grp.loc[take_long, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            short_net = -grp.loc[take_short, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            net = np.concatenate([long_net, short_net])
            rows.append({"month": str(month), "n_trades": n_trades,
                          "win_pct": 100 * (net > 0).sum() / n_trades, "mean_net_pct": 100 * net.mean()})
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
