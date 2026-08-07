"""Labeling-paradigm diagnostic, round 2 (per user request 2026-08-04, following up on
project-btc-cusum-architecture-structural-redesign-closed-20260804's H_selection/H_cost checks):
those checks measured predicted-quality vs realized-return correlation, which conflates two
possible causes of "zero signal" -- (a) the causal feature set truly carries no information about
future returns, or (b) triple-barrier quality-regression labeling destroys real information that
IS in the raw features (path-dependent TP/SL first-touch labels are noisy; a continuous quality
target compounds that noise).

This script skips BOTH the model and the triple-barrier label entirely and asks the more primitive
question directly: does any individual causal feature have a non-trivial rank correlation
(information coefficient, Spearman) with the plain forward N-bar return? No model to fail to learn
a mapping, no barrier-crossing path-dependence -- the simplest possible test of "is there
information in this feature set at all" for BTC.

Forward return definition matches the entry/exit convention used throughout this session's dense
tests: enter at open[i+1], measure return to close[i+h] (fwd_ret_h). Evaluated on causalfix_final's
98 cols (post-EXCLUDE_COLS, same set as every classifier test this session) across 5 horizons
spanning the two label geometries already tried (h48qual_shape ~48 bars, longhold_shape ~576 bars),
computed separately on TRAIN/VAL/OOS to check whether any nonzero IC is a real, stable relationship
or overfit noise that flips sign across splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BTC_FRAME_PATH = ROOT / "data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet"
OUT_CSV = ROOT / "tmp/btc_feature_return_ic_20260804.csv"

VAL_START, OOS_START, OOS_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
HORIZONS = [12, 48, 96, 288, 576]

EXCLUDE_COLS = {
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "sum_open_interest_value", "close_btc",
    "volume_btc", "quote_volume_btc",
    "mtf1h_ts_t_value", "mtf1h_ts_opt_L",
}


def main() -> int:
    frame = pd.read_parquet(BTC_FRAME_PATH).sort_values("timestamp").reset_index(drop=True)
    feat_cols = [c for c in frame.columns if c not in EXCLUDE_COLS]
    n = len(frame)
    open_px = frame["open"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)

    train_mask = frame["timestamp"] < VAL_START
    val_mask = (frame["timestamp"] >= VAL_START) & (frame["timestamp"] < OOS_START)
    oos_mask = (frame["timestamp"] >= OOS_START) & (frame["timestamp"] < OOS_END)
    splits = {"TRAIN": train_mask, "VAL": val_mask, "OOS": oos_mask}

    rows = []
    for h in HORIZONS:
        entry_i = np.arange(1, n - h)
        entry_px = open_px[entry_i]
        exit_px = close[entry_i + h - 1]
        fwd_ret = exit_px / entry_px - 1.0
        base_i = entry_i - 1  # feature snapshot is the bar BEFORE entry (causal: known before open[i+1])

        fwd = pd.Series(fwd_ret, index=base_i)
        for split_name, mask in splits.items():
            split_idx = frame.index[mask.to_numpy()]
            idx = np.intersect1d(base_i, split_idx, assume_unique=False)
            if len(idx) < 200:
                continue
            sub_feat = frame.loc[idx, feat_cols]
            sub_fwd = fwd.loc[idx]
            for col in feat_cols:
                x = sub_feat[col].to_numpy(dtype=np.float64)
                y = sub_fwd.to_numpy(dtype=np.float64)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 200:
                    continue
                ic, pval = stats.spearmanr(x[valid], y[valid])
                rows.append({"horizon": h, "split": split_name, "feature": col,
                              "ic": ic, "pval": pval, "n": int(valid.sum())})
        print(f"horizon={h}: done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {len(out)} rows -> {OUT_CSV}")

    # sign-consistency check: for each (horizon, feature), is |IC| > noise floor AND same sign across TRAIN/VAL/OOS?
    piv = out.pivot_table(index=["horizon", "feature"], columns="split", values="ic")
    piv = piv.dropna(subset=["TRAIN", "VAL", "OOS"])
    piv["min_abs_ic"] = piv[["TRAIN", "VAL", "OOS"]].abs().min(axis=1)
    piv["all_same_sign"] = (np.sign(piv["TRAIN"]) == np.sign(piv["VAL"])) & (np.sign(piv["VAL"]) == np.sign(piv["OOS"]))
    NOISE_FLOOR = 0.02  # ~ 2/sqrt(n) for n~2-6k per split, conservative significance threshold
    survivors = piv[(piv["min_abs_ic"] > NOISE_FLOOR) & piv["all_same_sign"]].sort_values("min_abs_ic", ascending=False)

    print(f"\n=== max |IC| per split, across all features/horizons ===")
    print(out.groupby("split")["ic"].apply(lambda s: s.abs().max()).to_string())

    print(f"\n=== features with |IC| > {NOISE_FLOOR} in ALL THREE splits AND consistent sign: {len(survivors)} ===")
    if len(survivors):
        print(survivors[["TRAIN", "VAL", "OOS", "min_abs_ic"]].to_string())
    else:
        print("(none)")

    print(f"\n=== top 15 by |min IC across splits| regardless of sign-consistency (for context) ===")
    piv_sorted = piv.reindex(piv["min_abs_ic"].abs().sort_values(ascending=False).index)
    print(piv_sorted[["TRAIN", "VAL", "OOS", "min_abs_ic", "all_same_sign"]].head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
