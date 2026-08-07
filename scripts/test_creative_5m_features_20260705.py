#!/usr/bin/env python3
"""Design + test CREATIVE new 5m features, measuring their MARGINAL permutation importance when
added to the existing set. Targets the gaps the audit revealed: the existing 125 features have
NO direct trend-vs-chop discriminator (the exact thing that mattered in Sigma6), no trend-age,
and weak/dead funding+orderflow. All new features are causal (OHLCV+funding+OI only).

New features:
  eff_ratio_48 / eff_ratio_12  : Kaufman Efficiency Ratio = |net move| / sum|bar moves|. THE
                                 trendiness-vs-chop measure (0=chop, 1=clean trend). Missing entirely.
  trend_age                    : bars since the smoothed-slope sign last flipped (trend maturity;
                                 trend-followers need young-vs-exhausted). Missing entirely.
  mtf_slope_agree              : sign(ret_12)+sign(ret_48)+sign(ret_144) in {-3..3} -- multi-horizon
                                 trend alignment as one scalar.
  vol_expansion                : realized_vol(12)/realized_vol(96) -- compression->expansion detector.
  dist_hi_atr / dist_lo_atr    : distance to rolling 96-bar high/low in ATR units (range position).
  ret_skew_48                  : rolling skew of log returns (directional asymmetry of the path).
  trend_quality                : eff_ratio_48 * tanh(slope/vol) -- trend strength gated by trendiness.
  cvd_div                      : sign(price mom) != sign(cvd mom) -> weak/divergent move flag.
  accel_vwap                   : 2nd difference of vwap distance (is the move accelerating?).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/causal_regen_20260516/feature_audit_20260705"
NON_FEAT = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "close_btc", "volume_btc", "quote_volume_btc",
            "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
            "last_funding_rate"}
FWD = 12
HOLDOUT_START = pd.Timestamp("2025-07-01")


def add_creative(df: pd.DataFrame) -> list[str]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    logc = np.log(close.clip(lower=1e-9))
    ret = logc.diff()
    absret = ret.abs()

    def eff_ratio(w):
        net = (logc - logc.shift(w)).abs()
        path = absret.rolling(w, min_periods=max(4, w // 4)).sum()
        return (net / path.replace(0, np.nan)).clip(0, 1).fillna(0.0)

    df["eff_ratio_48"] = eff_ratio(48)
    df["eff_ratio_12"] = eff_ratio(12)

    # ATR
    prev = close.shift(1)
    tr = np.maximum.reduce([high - low, (high - prev).abs(), (low - prev).abs()])
    atr = pd.Series(tr, index=df.index).rolling(48, min_periods=8).mean()
    atrp = (atr / close.clip(lower=1e-9)).clip(1e-6, 1)

    # smoothed slope + trend age
    ema = close.ewm(span=24, adjust=False).mean()
    slope = ema.diff()
    ssign = np.sign(slope).fillna(0.0)
    # bars since sign flip
    grp = (ssign != ssign.shift(1)).cumsum()
    df["trend_age"] = (df.groupby(grp).cumcount().clip(0, 288) / 288.0).values * ssign.values

    r12 = (logc - logc.shift(12))
    r48 = (logc - logc.shift(48))
    r144 = (logc - logc.shift(144))
    df["mtf_slope_agree"] = (np.sign(r12).fillna(0) + np.sign(r48).fillna(0) + np.sign(r144).fillna(0)) / 3.0

    rvol12 = ret.rolling(12, min_periods=4).std()
    rvol96 = ret.rolling(96, min_periods=24).std()
    df["vol_expansion"] = (rvol12 / rvol96.replace(0, np.nan)).clip(0, 5).fillna(1.0)

    hh = high.rolling(96, min_periods=24).max()
    ll = low.rolling(96, min_periods=24).min()
    df["dist_hi_atr"] = ((close - hh) / (atr + 1e-9)).clip(-30, 0).fillna(0.0)
    df["dist_lo_atr"] = ((close - ll) / (atr + 1e-9)).clip(0, 30).fillna(0.0)

    df["ret_skew_48"] = ret.rolling(48, min_periods=12).skew().clip(-5, 5).fillna(0.0)

    df["trend_quality"] = (df["eff_ratio_48"] * np.tanh((slope / (atr * close + 1e-12)).clip(-5, 5))).fillna(0.0)

    if "cvd_48" in df.columns:
        cvd_mom = pd.to_numeric(df["cvd_48"], errors="coerce").diff(12)
        df["cvd_div"] = ((np.sign(r12).fillna(0) != np.sign(cvd_mom).fillna(0)).astype(float)).values
    else:
        df["cvd_div"] = 0.0

    if "vwap_dist_96" in df.columns:
        vw = pd.to_numeric(df["vwap_dist_96"], errors="coerce")
        df["accel_vwap"] = vw.diff().diff().clip(-0.05, 0.05).fillna(0.0)
    else:
        df["accel_vwap"] = 0.0

    return ["eff_ratio_48", "eff_ratio_12", "trend_age", "mtf_slope_agree", "vol_expansion",
            "dist_hi_atr", "dist_lo_atr", "ret_skew_48", "trend_quality", "cvd_div", "accel_vwap"]


def main() -> int:
    df = pd.concat([pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", low_memory=False) for y in (2024, 2025)], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    existing = [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]
    new_feats = add_creative(df)
    all_feats = existing + new_feats

    close = df["close"].astype(float)
    fwd = np.log(close.shift(-FWD)) - np.log(close)
    y = (fwd > 0).astype(int)
    X = df[all_feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid = fwd.notna().to_numpy()
    tr = (df["timestamp"] < HOLDOUT_START).to_numpy() & valid
    ho = (df["timestamp"] >= HOLDOUT_START).to_numpy() & valid
    ho_idx = np.flatnonzero(ho)
    rng = np.random.default_rng(0)
    if len(ho_idx) > 20000:
        ho_idx = np.sort(rng.choice(ho_idx, 20000, replace=False))

    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                         l2_regularization=1.0, min_samples_leaf=100, random_state=0)
    clf.fit(X.to_numpy()[tr], y.to_numpy()[tr])
    acc = clf.score(X.to_numpy()[ho_idx], y.to_numpy()[ho_idx])
    print(f"base acc WITH new features: {acc:.4f} (was 0.5151 without)", flush=True)

    perm = permutation_importance(clf, X.to_numpy()[ho_idx], y.to_numpy()[ho_idx], n_repeats=6, random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"feature": all_feats, "imp": perm.importances_mean, "std": perm.importances_std,
                        "is_new": [f in new_feats for f in all_feats]}).sort_values("imp", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT / "with_creative_importance.csv", index=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    print("\n=== NEW features' rank among all 136 ===", flush=True)
    print(imp[imp["is_new"]][["rank", "feature", "imp", "std"]].to_string(index=False), flush=True)
    print("\n=== TOP 15 overall (new marked *) ===", flush=True)
    for _, r in imp.head(15).iterrows():
        print(f"  {r['rank']:>3} {'*' if r['is_new'] else ' '} {r['feature']:<28} {r['imp']:.5f}", flush=True)
    json.dump({"acc_with_new": float(acc)}, open(OUT / "creative_summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
