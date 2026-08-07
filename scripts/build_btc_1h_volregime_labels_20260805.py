"""BTC 1h new-architecture, Step 1 (label design): volatility-regime-transition label.

Direction-agnostic alternative to triple-barrier/trend-scan (both already closed for BTC --
see docs/btc_new_architecture_session_summary_20260804.md and memory
project-btc-tau1-leg-a-nan-fix-closed-20260805). Target: will realized volatility over the NEXT
24h expand or contract relative to the TRAILING 24h, on BTC's native 1h candles. This is a
hindsight label for training only (forward window used for label construction, never as a
feature) -- not a lookahead bug, same convention as triple-barrier's own forward-looking target.

label_3class: +1 = expansion (top 30% of ratio), -1 = contraction (bottom 30%), 0 = stable (mid 40%).
Quantile cutoffs are computed over the whole sample for this label-design sanity check only;
any live/causal threshold will be re-derived from trailing-only data in a later calibration step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BTC_1H_PATH = ROOT / "data/splits/year_oos/btc_features_1h_full_2024_2026.csv"
DVOL_PATH = ROOT / "data/derivatives/deribit_dvol/BTC_dvol_hourly.csv"
OUT_PATH = ROOT / "data/splits/year_oos/btc_1h_volregime_labels_20260805.parquet"

TRAIL_H = 24
FWD_H = 24
LOW_Q, HIGH_Q = 0.30, 0.70


def main() -> int:
    btc = pd.read_csv(BTC_1H_PATH, usecols=["timestamp", "close"])
    btc["timestamp"] = pd.to_datetime(btc["timestamp"])
    btc = btc.sort_values("timestamp").reset_index(drop=True)

    log_ret = np.log(btc["close"]).diff()
    trailing_vol = log_ret.rolling(TRAIL_H, min_periods=TRAIL_H).std()
    # future_vol at t = std of returns over (t, t+FWD_H], i.e. the window immediately after t
    future_vol = log_ret.rolling(FWD_H, min_periods=FWD_H).std().shift(-FWD_H)

    ratio = future_vol / trailing_vol

    out = pd.DataFrame({
        "timestamp": btc["timestamp"],
        "close": btc["close"],
        "trailing_vol_24h": trailing_vol,
        "future_vol_24h": future_vol,
        "vol_ratio": ratio,
    })

    valid = out["vol_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = valid.quantile(LOW_Q), valid.quantile(HIGH_Q)

    label = pd.Series(0.0, index=out.index)
    label[out["vol_ratio"] <= lo] = -1.0
    label[out["vol_ratio"] >= hi] = 1.0
    label[out["vol_ratio"].isna()] = np.nan
    out["label_3class"] = label

    # attach DVOL (causal, available_at = timestamp+1h) purely for the sanity-check correlation below
    dvol = pd.read_csv(DVOL_PATH)
    dvol["timestamp"] = pd.to_datetime(dvol["timestamp"])
    dvol["available_at"] = dvol["timestamp"] + pd.Timedelta(hours=1)
    dvol = dvol[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"})
    dvol = dvol.sort_values("timestamp")
    out = pd.merge_asof(out, dvol, on="timestamp", direction="backward")

    out.to_parquet(OUT_PATH, index=False)

    n_valid = out["label_3class"].notna().sum()
    print(f"wrote {OUT_PATH}, shape={out.shape}, rows with label={n_valid}")
    print(f"quantile cutoffs (whole-sample): lo={lo:.4f} hi={hi:.4f}")
    print("class balance:")
    print(out["label_3class"].value_counts(dropna=False, normalize=True).sort_index())

    # sanity: does current DVOL level/roc correlate with the forward vol-expansion label at all?
    sane = out.dropna(subset=["label_3class", "dvol_btc"])
    corr = sane["dvol_btc"].corr(sane["vol_ratio"])
    print(f"\ncorr(dvol_btc level, vol_ratio) = {corr:.4f}  (n={len(sane)})")

    dvol_roc_24h = sane["dvol_btc"].pct_change(24)
    corr_roc = dvol_roc_24h.corr(sane["vol_ratio"])
    print(f"corr(dvol_btc 24h RoC, vol_ratio) = {corr_roc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
