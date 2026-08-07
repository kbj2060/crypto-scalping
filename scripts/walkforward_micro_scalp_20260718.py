"""Walk-forward prototype: 1m-decision ETH scalping model on price + live microstructure
features (2026-07-18). This is the design-validation experiment for the microstructure scalping
model -- it answers whether a nonlinear multivariate model clears the cost hurdle that the
single-feature/tail analysis (analyze_microstructure_tails_20260718.py) showed it cannot clear
linearly.

Causality: micro row ts=T usable from decision D = T+2min (see analyze_microstructure_edge).
Price features at decision D use only bars with open time <= D-1min (i.e., closed by D).
Entry at bar-D open (first executable price after decision). No BTC features at all.

Eval: rolling walk-forward (train >= 30d, test 7d). Policy: |pred| >= threshold -> enter,
hold HOLD_MIN minutes, exit at close. Non-overlapping positions (one at a time). Costs
reported at taker-taker (9bps rt), maker-taker (6.5bps rt), maker-maker (4bps rt, optimistic
fill assumption -- upper bound only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import analyze_microstructure_edge_20260718 as base

HOLD_MIN = 5
TRAIN_MIN_DAYS = 30
TEST_DAYS = 7
COSTS_BPS = {"taker_taker": 9.0, "maker_taker": 6.5, "maker_maker": 4.0}
THRESH_Q = [0.90, 0.95, 0.98]  # per-fold signal-strength quantiles from TRAIN predictions


def price_features(kl: pd.DataFrame) -> pd.DataFrame:
    # All computed on closed bars, then shifted so row at decision D only sees bars closed by D:
    # kline row T covers [T, T+1) and closes at T+1 -> value computed on rows ..T is available
    # at D = T+1. Reindexing by T+1min == shift(1) on the same minute grid.
    f = pd.DataFrame(index=kl.index)
    c, h, l, v = kl["close"], kl["high"], kl["low"], kl["volume"]
    for w in [1, 3, 5, 15, 60]:
        f[f"ret_{w}"] = c.pct_change(w)
    for w in [15, 60]:
        r = c.pct_change()
        f[f"vol_{w}"] = r.rolling(w).std()
        f[f"rangepos_{w}"] = (c - l.rolling(w).min()) / (h.rolling(w).max() - l.rolling(w).min() + 1e-12)
    f["vwap_dev_30"] = c / ((kl["close"] * v).rolling(30).sum() / v.rolling(30).sum().replace(0, np.nan)) - 1.0
    f["vol_ratio_5_60"] = v.rolling(5).mean() / v.rolling(60).mean().replace(0, np.nan)
    f["hl_range_5"] = (h.rolling(5).max() - l.rolling(5).min()) / c
    f.index = f.index + pd.Timedelta(minutes=1)  # availability shift
    return f


def build_dataset():
    micro = base.add_derived(base.load_micro())
    micro.index = micro.index + pd.Timedelta(minutes=base.AVAIL_SHIFT_MIN)
    micro = micro.drop(columns=["shadow_regime_tag"])
    kl = pd.read_csv(base.KLINES, parse_dates=["timestamp"],
                     usecols=["timestamp", "open", "high", "low", "close", "volume"])
    kl = kl[kl["timestamp"] >= micro.index.min() - pd.Timedelta("2h")].set_index("timestamp").sort_index()
    pf = price_features(kl)
    df = pd.DataFrame(index=kl.index)
    df["entry_open"] = kl["open"]
    df["target"] = kl["close"].shift(-(HOLD_MIN - 1)) / kl["open"] - 1.0
    df = df.join(pf, how="left").join(micro, how="inner").dropna(subset=["target", "ret_60"])
    feat_cols = [c for c in df.columns if c not in ("entry_open", "target")]
    return df, feat_cols


def simulate(test: pd.DataFrame, preds: np.ndarray, thr: float) -> dict:
    test = test.assign(pred=preds)
    sig = np.where(test["pred"] >= thr, 1, np.where(test["pred"] <= -thr, -1, 0))
    pnl_by_cost = {k: [] for k in COSTS_BPS}
    trades, i, idx = 0, 0, test.index
    while i < len(test):
        if sig[i] == 0:
            i += 1
            continue
        gross = sig[i] * test["target"].iloc[i]
        for k, cbps in COSTS_BPS.items():
            pnl_by_cost[k].append(gross - cbps / 1e4)
        trades += 1
        # skip to first decision at/after exit time
        exit_t = idx[i] + pd.Timedelta(minutes=HOLD_MIN)
        i = int(np.searchsorted(idx.values, np.datetime64(exit_t)))
    out = {"trades": trades}
    for k in COSTS_BPS:
        arr = np.asarray(pnl_by_cost[k])
        out[k] = {"sum_pct": arr.sum() * 100 if trades else 0.0,
                  "mean_bps": arr.mean() * 1e4 if trades else 0.0,
                  "win": (arr > 0).mean() if trades else 0.0}
    return out


def main() -> None:
    df, feat_cols = build_dataset()
    days = pd.Series(df.index.date, index=df.index)
    uniq = sorted(days.unique())
    print(f"dataset: {len(df):,} rows, {len(feat_cols)} features, {len(uniq)} days "
          f"({uniq[0]} -> {uniq[-1]})")

    folds = []
    start = TRAIN_MIN_DAYS
    while start + 1 < len(uniq):
        folds.append((uniq[:start], uniq[start:start + TEST_DAYS]))
        start += TEST_DAYS
    print(f"{len(folds)} folds (expanding train, {TEST_DAYS}d test)")

    agg = {q: {k: [] for k in COSTS_BPS} for q in THRESH_Q}
    agg_trades = {q: 0 for q in THRESH_Q}
    fold_net = {q: [] for q in THRESH_Q}  # per-fold maker_taker net sums
    ics = []
    for fi, (tr_days, te_days) in enumerate(folds):
        tr = df[days.isin(tr_days)]
        te = df[days.isin(te_days)]
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=None,
                                              max_leaf_nodes=31, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=42)
        model.fit(tr[feat_cols], tr["target"])
        tr_pred = model.predict(tr[feat_cols])
        te_pred = model.predict(te[feat_cols])
        from scipy.stats import spearmanr
        ic = spearmanr(te_pred, te["target"]).statistic
        ics.append(ic)
        line = [f"fold{fi} test {te_days[0]}..{te_days[-1]} n={len(te):,} IC={ic:+.3f}"]
        for q in THRESH_Q:
            thr = np.quantile(np.abs(tr_pred), q)
            r = simulate(te, te_pred, thr)
            agg_trades[q] += r["trades"]
            for k in COSTS_BPS:
                agg[q][k].append(r[k]["sum_pct"])
            fold_net[q].append(r["maker_taker"]["sum_pct"])
            line.append(f"q{int(q*100)}: n={r['trades']} mt={r['maker_taker']['sum_pct']:+.2f}% "
                        f"({r['maker_taker']['mean_bps']:+.1f}bps/tr)")
        print("  ".join(line))

    print(f"\nmean test IC: {np.mean(ics):+.3f} (folds pos: {np.mean(np.asarray(ics) > 0):.0%})")
    print("\n=== aggregate over all OOS folds ===")
    for q in THRESH_Q:
        print(f"threshold q{int(q*100)}: total trades={agg_trades[q]}")
        for k in COSTS_BPS:
            tot = np.sum(agg[q][k])
            print(f"   {k:12s}: total={tot:+.2f}%  pos_folds={np.mean(np.asarray(agg[q][k]) > 0):.0%}")


if __name__ == "__main__":
    main()
