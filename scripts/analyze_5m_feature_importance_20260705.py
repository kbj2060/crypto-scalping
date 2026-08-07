#!/usr/bin/env python3
"""Feature-importance + redundancy audit of the user's 125 engineered 5m features.

Target: sign of forward 12-bar (1h) return (the core directional trading target). Temporal split
(train on 2024+2025H1, score importance on a 2025H2 holdout — no lookahead in the eval). Reports:
  1. Permutation importance (accuracy drop when a feature is shuffled) — the honest "does it help
     OOS" measure, robust to the tree's split-count bias.
  2. Redundancy clusters (features with |corr| >= 0.9 carry the same information).
  3. Category rollups.
Purely diagnostic; no model is promoted.
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
FILES = {y: ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in (2024, 2025)}
FILES[2025] = ROOT / "data/splits/year_oos/training_features_2025.csv"
NON_FEAT = {"timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "close_btc", "volume_btc", "quote_volume_btc",
            "sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio",
            "last_funding_rate"}
FWD = 12  # 1h forward at 5m
HOLDOUT_START = pd.Timestamp("2025-07-01")


def category(c: str) -> str:
    cl = c.lower()
    if "funding" in cl: return "funding"
    if "cvd" in cl or "cvp" in cl: return "cvd/cvp"
    if any(k in cl for k in ("vol", "atr", "garch", "parkinson", "garman", "rogers")): return "volatility"
    if "btc" in cl: return "btc_cross"
    if any(k in cl for k in ("taker", "whale", "smart_money", "ofi", "oi_", "net_taker")): return "orderflow"
    if any(k in cl for k in ("sess", "hour", "minute")): return "time"
    if any(k in cl for k in ("rsi", "macd", "bb_", "hma", "kalman", "hurst")): return "tech_indicator"
    if any(k in cl for k in ("sweep", "breakout", "compression", "squeeze", "fib", "vwap", "wick")): return "price_action"
    return "other"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(FILES[y], low_memory=False) for y in (2024, 2025)], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON_FEAT and pd.api.types.is_numeric_dtype(df[c])]
    close = df["close"].astype(float)
    fwd_ret = np.log(close.shift(-FWD)) - np.log(close)
    y = (fwd_ret > 0).astype(int)
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid = fwd_ret.notna().to_numpy()
    tr = (df["timestamp"] < HOLDOUT_START).to_numpy() & valid
    ho = (df["timestamp"] >= HOLDOUT_START).to_numpy() & valid
    # subsample holdout for speed of permutation importance
    ho_idx = np.flatnonzero(ho)
    rng = np.random.default_rng(0)
    if len(ho_idx) > 20000:
        ho_idx = np.sort(rng.choice(ho_idx, 20000, replace=False))
    print(f"features={len(feats)} train={int(tr.sum())} holdout(eval)={len(ho_idx)}", flush=True)

    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=4,
                                         l2_regularization=1.0, min_samples_leaf=100, random_state=0)
    clf.fit(X.to_numpy()[tr], y.to_numpy()[tr])
    base_acc = clf.score(X.to_numpy()[ho_idx], y.to_numpy()[ho_idx])
    print(f"holdout base accuracy (fwd 1h sign): {base_acc:.4f} (0.5=coin flip)", flush=True)

    perm = permutation_importance(clf, X.to_numpy()[ho_idx], y.to_numpy()[ho_idx],
                                  n_repeats=5, random_state=0, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "perm_importance": perm.importances_mean,
                        "perm_std": perm.importances_std, "category": [category(c) for c in feats]})
    imp = imp.sort_values("perm_importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)

    print("\n=== TOP 25 features by permutation importance ===", flush=True)
    print(imp.head(25).to_string(index=False), flush=True)
    dead = imp[imp["perm_importance"] <= 0.0]
    print(f"\n=== DEAD/NOISE features (perm_importance <= 0): {len(dead)}/{len(feats)} ===", flush=True)
    print(dead["feature"].tolist(), flush=True)

    print("\n=== category rollup (sum of positive perm importance) ===", flush=True)
    roll = imp[imp["perm_importance"] > 0].groupby("category")["perm_importance"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    print(roll.to_string(), flush=True)

    # redundancy: correlation clusters among top-60 informative features
    top = imp[imp["perm_importance"] > 0].head(60)["feature"].tolist()
    corr = X[top].sample(min(40000, int(tr.sum())), random_state=1).corr().abs()
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if corr.iloc[i, j] >= 0.9:
                pairs.append((top[i], top[j], round(float(corr.iloc[i, j]), 3)))
    print(f"\n=== redundant pairs (|corr|>=0.9) among top-60: {len(pairs)} ===", flush=True)
    for a, b, c in pairs[:40]:
        print(f"  {a} ~ {b}  ({c})", flush=True)
    json.dump({"base_acc": float(base_acc), "n_dead": int(len(dead)), "n_redundant_pairs": len(pairs)},
              open(OUT / "summary.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
