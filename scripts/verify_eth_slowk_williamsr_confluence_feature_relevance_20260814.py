#!/usr/bin/env python3
"""Feature-relevance screening for the Slow %K x Williams %R persistence-confluence oscillator
(docs/experiments/eth_slowk_williamsr_persistence_confluence_20260814.md), following this repo's
established candidate-feature gate pattern (contamination check + spearman IC + mutual_info),
see scripts/verify_eth_defillama_onchain_direction_relevance_20260812.py.

backtest_eth_slowk_williamsr_persistence_confluence_20260814.py already showed this signal has
NO standalone tradeable edge (every arm lost to max(always_long, always_short) on VAL). This
script asks a DIFFERENT, narrower question: does the underlying oscillator relationship carry
ANY information about forward returns at all, of a size an Omega-style TabM/GBDT model could
plausibly extract as one feature among ~100+ -- independent of whether a hand-written rule on
top of it is profitable standalone.

Exploratory screening only. Forward returns are used here as the SCREENING TARGET being
correlated against (same role labels play in every sibling verify_*_relevance script in this
repo) -- not as trading input, not a promotion claim, no VAL/OOS budget pre-registration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import (  # noqa: E402
    VAL_START,
    compute_indicators,
    load_frame,
)

CONTAMINATION_THRESHOLD = 0.561  # repo convention, see verify_eth_defillama_onchain_direction_relevance_20260812.py
HORIZONS = {"h12_1h": 12, "h48_4h": 48, "h96_8h": 96}


def build_candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    p_fast, p_slow, spread = frame["p_fast"], frame["p_slow"], frame["spread"]
    trend_on = frame["trend_on"].astype(float)

    confluence_score = -(p_fast - 0.5) - (p_slow - 0.5)  # + = both oversold (bullish lean)
    spread_mean = spread.rolling(288, min_periods=288).mean()
    spread_std = spread.rolling(288, min_periods=288).std()
    spread_z = (spread - spread_mean) / spread_std.replace(0.0, np.nan)

    extreme_zone = ((p_fast <= 0.10) | (p_fast >= 0.90)).astype(int)
    zone_id = (extreme_zone.diff() != 0).cumsum()
    persistence_bars = extreme_zone.groupby(zone_id).cumcount() + 1
    persistence_bars = persistence_bars.where(extreme_zone == 1, 0)

    feats = pd.DataFrame(
        {
            "confluence_p_fast": p_fast,
            "confluence_p_slow": p_slow,
            "confluence_spread": spread,
            "confluence_spread_z": spread_z,
            "confluence_score": confluence_score,
            "confluence_persistence_bars": persistence_bars.astype(float),
            "confluence_score_x_fade": confluence_score * (1.0 - trend_on),
            "confluence_score_x_follow": confluence_score * trend_on,
        },
        index=frame.index,
    )
    return feats


def contamination_check(feats: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    n = len(feats)
    time_index = pd.Series(np.arange(n), index=feats.index)
    rows = []
    for col in feats.columns:
        x = feats[col]
        valid = x.notna() & close.notna()
        corr_price = spearmanr(x[valid], close[valid]).statistic if valid.sum() > 10 else np.nan
        corr_time = spearmanr(x[valid], time_index[valid]).statistic if valid.sum() > 10 else np.nan
        rows.append(
            {
                "feature": col,
                "spearman_vs_price": corr_price,
                "spearman_vs_time": corr_time,
                "contaminated": bool(
                    (abs(corr_price) if np.isfinite(corr_price) else 0.0) > CONTAMINATION_THRESHOLD
                    or (abs(corr_time) if np.isfinite(corr_time) else 0.0) > CONTAMINATION_THRESHOLD
                ),
            }
        )
    return pd.DataFrame(rows)


def relevance_screen(feats: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    rows = []
    for horizon_name, h in HORIZONS.items():
        fwd_ret = close.shift(-h) / close - 1.0
        fwd_sign = (fwd_ret > 0).astype(int)
        for col in feats.columns:
            x = feats[col]
            valid = x.notna() & fwd_ret.notna()
            if valid.sum() < 200:
                continue
            xv, yv, sv = x[valid].to_numpy(), fwd_ret[valid].to_numpy(), fwd_sign[valid].to_numpy()
            ic = spearmanr(xv, yv).statistic
            mi = mutual_info_classif(
                xv.reshape(-1, 1), sv, discrete_features=False, random_state=20260814, n_neighbors=5
            )[0]
            rows.append(
                {
                    "horizon": horizon_name,
                    "feature": col,
                    "n": int(valid.sum()),
                    "spearman_ic": float(ic),
                    "mutual_info": float(mi),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_frame()
    frame = compute_indicators(raw)
    frame = frame[frame["timestamp"] >= VAL_START - pd.Timedelta(days=3)].reset_index(drop=True)

    feats = build_candidate_features(frame)
    close = frame["close"]

    print("=== Contamination check (spearman vs price / vs time-index) ===")
    contam = contamination_check(feats, close)
    print(contam.to_string(index=False))

    print("\n=== Relevance screen (spearman IC + mutual_info vs forward-return sign) ===")
    rel = relevance_screen(feats, close)
    pivot_ic = rel.pivot(index="feature", columns="horizon", values="spearman_ic")
    pivot_mi = rel.pivot(index="feature", columns="horizon", values="mutual_info")
    print("\n-- spearman IC --")
    print(pivot_ic.to_string())
    print("\n-- mutual_info (vs sign of forward return) --")
    print(pivot_mi.to_string())

    out_dir = ROOT / "tmp" / "eth_slowk_williamsr_persistence_confluence_20260814"
    out_dir.mkdir(parents=True, exist_ok=True)
    contam.to_csv(out_dir / "feature_contamination.csv", index=False)
    rel.to_csv(out_dir / "feature_relevance.csv", index=False)
    print(f"\nWrote screening tables to {out_dir}")


if __name__ == "__main__":
    main()
