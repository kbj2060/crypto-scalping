"""Labeling-paradigm-shift test (per project-btc-feature-return-ic-diagnostic-20260804): the raw
feature-vs-forward-return IC diagnostic found that squeeze_power (OI*funding) and trade_intensity
(trades/volume) both have a real, sign-stable, cross-period Spearman IC against the plain 576-bar
(~2-day) forward return (squeeze_power: -0.057/-0.039/-0.153 TRAIN/VAL/OOS; trade_intensity:
-0.059/-0.123/-0.050), while the triple-barrier LightGBM quality-classifier built on the full
98-column causal frame could not turn any feature set into a working strategy all session.

This script tests whether that raw correlation survives as an actual net-of-cost strategy when
turned into the SIMPLEST possible rule -- no model, no triple-barrier, no 98-column feature
set -- just these two causally-known features combined into one composite score via a causal
rolling z-score (trailing window only, no future data), thresholded at percentiles fit on the
TRAIN period only and held fixed for VAL/OOS. If a two-feature linear rule beats the same
threshold-sweep/cost/split protocol used for every other test this session, that's strong evidence
the earlier "no signal" conclusion was a labeling/model artifact, not an empty feature set. If it
still fails, that's evidence the raw IC, while real, is too weak/noisy to survive costs and
horizon-return realization -- a materially different (and more informative) failure than before.

Entry/exit convention matches every other dense test this session: enter at open[i+1], exit at
close[i+576]. Fixed horizon only (no TP/SL, no early exit) -- deliberately the simplest possible
label, to isolate whether the composite score alone has tradeable directional information.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_simple_squeeze_tradeintensity_20260804.csv"

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
HORIZON = 576
COST_CONSERVATIVE = (0.0005 + 0.0002) * 2.0 * 3.0  # 0.42% round-trip, same conservative assumption used all session
ZWINDOW = 8640  # ~30 days of 5m bars, trailing/causal (pandas .rolling() is trailing by construction)
ZMIN_PERIODS = 2880  # ~10 days
STRIDE = 3
PERCENTILES = [0.80, 0.90, 0.95]  # symmetric long/short percentile cutoffs, swept


def causal_zscore(s: pd.Series) -> pd.Series:
    mean = s.rolling(ZWINDOW, min_periods=ZMIN_PERIODS).mean()
    std = s.rolling(ZWINDOW, min_periods=ZMIN_PERIODS).std()
    return (s - mean) / std.replace(0, np.nan)


def main() -> int:
    frame = pd.read_parquet(BTC_FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    n = len(frame)

    z_squeeze = causal_zscore(frame["squeeze_power"])
    z_intensity = causal_zscore(frame["trade_intensity"])
    # negative IC on both raw features -> negate so higher composite score = higher expected forward return
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
        "i": idxs, "timestamp": ts.iloc[idxs].to_numpy(), "score": score.iloc[idxs].to_numpy(),
        "fwd_ret": fwd_ret,
    }).dropna(subset=["score"])

    train = data[data["timestamp"] < VAL_START]
    val = data[(data["timestamp"] >= VAL_START) & (data["timestamp"] < OOS_START)]
    oos = data[(data["timestamp"] >= OOS_START) & (data["timestamp"] < OOS_END)]
    print(f"train={len(train)} val={len(val)} oos={len(oos)}")

    all_results = []
    for pct in PERCENTILES:
        long_thresh = train["score"].quantile(pct)
        short_thresh = train["score"].quantile(1 - pct)
        print(f"\npercentile={pct:.2f}: long_thresh(TRAIN)={long_thresh:.3f} short_thresh(TRAIN)={short_thresh:.3f}")

        for split_name, split in [("VAL", val), ("OOS", oos)]:
            take_long = split["score"] >= long_thresh
            take_short = split["score"] <= short_thresh
            n_trades = int(take_long.sum() + take_short.sum())
            if n_trades == 0:
                print(f"  [{split_name}] no trades")
                continue
            long_net = split.loc[take_long, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            short_net = -split.loc[take_short, "fwd_ret"].to_numpy() - COST_CONSERVATIVE
            net = np.concatenate([long_net, short_net])
            win = (net > 0).sum()
            all_results.append({
                "percentile": pct, "split": split_name, "n_trades": n_trades,
                "n_long": int(take_long.sum()), "n_short": int(take_short.sum()),
                "win_pct": 100 * win / n_trades, "mean_net_pct": 100 * net.mean(),
                "sum_net_pct": 100 * net.sum(),
            })
            print(f"  [{split_name}] n={n_trades:5d} (long={int(take_long.sum())} short={int(take_short.sum())}) "
                  f"win%={100*win/n_trades:5.1f} mean_net={100*net.mean():6.3f}% sum_net={100*net.sum():8.2f}%")

    out = pd.DataFrame(all_results)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    val_pos = out[(out["split"] == "VAL") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    oos_pos = out[(out["split"] == "OOS") & (out["mean_net_pct"] > 0) & (out["n_trades"] >= 15)]
    both = val_pos.merge(oos_pos, on=["percentile"], suffixes=("_val", "_oos"))
    print(f"\n=== Configs with VAL AND OOS both positive (n>=15 each side): {len(both)}/{len(PERCENTILES)} ===")
    if len(both):
        print(both[["percentile", "n_trades_val", "mean_net_pct_val", "n_trades_oos", "mean_net_pct_oos"]]
              .to_string(index=False))
    else:
        print("(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
