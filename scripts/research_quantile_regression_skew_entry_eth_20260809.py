#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #14): quantile regression on the forward
return distribution -- a genuinely different modeling paradigm from every classification-based
idea tried tonight (#1-13 all classified trade_outcome_action). Motivation: a classifier only
sees the median-ish "which barrier gets hit," but a model predicting the FULL conditional
quantile distribution of the forward return could pick up predicted SKEW/asymmetry even when
the median direction call carries no information -- e.g. "this bar's upside tail is fatter than
its downside tail" is a different claim from "this bar will go up."

Target: forward log-return over a fixed horizon (48 bars, matching this repo's usual horizon
convention), NOT the pre-built triple-barrier label (avoids inheriting its baked-in tp:sl bias
outright, though the entry decision is still evaluated through the SAME tp_move/sl_move payoff
for comparability with every other idea tonight).

Two LightGBM quantile regressors (objective="quantile", alpha=0.15 and alpha=0.85) predict the
15th and 85th percentile of the 48-bar-forward return. Trade in the direction of whichever tail
is more extreme (compare |q85| vs |q15|, i.e. predicted upside magnitude vs downside magnitude),
only when the CALL and skew agree; skip when q15 and q85 have the same sign (no clean directional
skew) or are both small. Same excess-over-baseline evaluation as ideas #6+.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
ETH_LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
BTC_PRICE_PATH = ROOT / "data/btc_5m_1year.csv"
WINDOWS = [3, 6, 12, 24, 48]
FORWARD_HORIZON = 48
SKEW_THRESHOLDS = [1.2, 1.5, 2.0, 3.0, 5.0]  # require |bigger tail| > threshold * |smaller tail|
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    eth_price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close", "high", "low", "volume", "taker_buy_base"], parse_dates=["timestamp"])
    btc_price = pd.read_csv(BTC_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "btc_close"})
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = eth_price.merge(btc_price, on="timestamp", how="inner").merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def build_features(df: pd.DataFrame) -> list[str]:
    cols = []
    log_eth, log_btc = np.log(df["close"].to_numpy()), np.log(df["btc_close"].to_numpy())
    bar_ret = log_eth - np.roll(log_eth, 1); bar_ret[0] = 0.0
    true_range = (df["high"].to_numpy() - df["low"].to_numpy()) / df["close"].to_numpy()
    ofi = np.where(df["volume"] > 0, 2 * df["taker_buy_base"] / df["volume"] - 1.0, 0.0)
    for w in WINDOWS:
        eth_ret = log_eth - np.roll(log_eth, w); eth_ret[:w] = np.nan
        btc_ret = log_btc - np.roll(log_btc, w); btc_ret[:w] = np.nan
        ofi_mean = pd.Series(ofi).rolling(w, min_periods=w).mean().to_numpy()
        rvol = pd.Series(bar_ret).rolling(w, min_periods=w).std().to_numpy()
        atr = pd.Series(true_range).rolling(w, min_periods=w).mean().to_numpy()
        for name, arr in [("eth_ret", eth_ret), ("btc_ret", btc_ret), ("ofi", ofi_mean), ("rvol", rvol), ("atr", atr)]:
            col = f"{name}_w{w}"
            df[col] = arr
            cols.append(col)
    return cols


def forward_return(log_close: np.ndarray, horizon: int) -> np.ndarray:
    fwd = np.roll(log_close, -horizon) - log_close
    fwd[-horizon:] = np.nan
    return fwd


def favored_direction(outcome, tp, sl, cost):
    long_net = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0)) - cost
    short_net = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0)) - cost
    return 1 if long_net.sum() >= short_net.sum() else 2


def payoff_for_direction(outcome, tp, sl, cost, direction):
    if direction == 1:
        return np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0)) - cost
    return np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0)) - cost


def _col_sharpe(m):
    mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


def main() -> None:
    merged = load_merged()
    feat_cols = build_features(merged)
    log_close = np.log(merged["close"].to_numpy())
    merged["fwd_ret"] = forward_return(log_close, FORWARD_HORIZON)
    merged = merged.dropna(subset=feat_cols + ["fwd_ret"]).reset_index(drop=True)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    train_outcome, train_tp, train_sl = train_df["trade_outcome_action"].to_numpy(), train_df["tp_move"].to_numpy(), train_df["sl_move"].to_numpy()
    direction = favored_direction(train_outcome, train_tp, train_sl, ROUND_TRIP_COST)
    print(f"Favored direction on TRAIN (for baseline only, not used by the quantile model): {'long' if direction == 1 else 'short'}")

    q_lo = lgb.LGBMRegressor(objective="quantile", alpha=0.15, n_estimators=300, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100, random_state=270705, verbosity=-1)
    q_hi = lgb.LGBMRegressor(objective="quantile", alpha=0.85, n_estimators=300, num_leaves=31,
                             learning_rate=0.05, min_child_samples=100, random_state=270705, verbosity=-1)
    q_lo.fit(train_df[feat_cols], train_df["fwd_ret"])
    q_hi.fit(train_df[feat_cols], train_df["fwd_ret"])

    def skew_calls(split_df, threshold):
        p15 = q_lo.predict(split_df[feat_cols])
        p85 = q_hi.predict(split_df[feat_cols])
        up_mag, down_mag = np.maximum(p85, 0), np.maximum(-p15, 0)
        long_call = up_mag > threshold * np.maximum(down_mag, 1e-9)
        short_call = down_mag > threshold * np.maximum(up_mag, 1e-9)
        cls = np.where(long_call, 1, np.where(short_call, 2, 0))
        return cls

    dev_outcome, dev_tp, dev_sl = dev_df["trade_outcome_action"].to_numpy(), dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()
    dev_baseline_payoff = payoff_for_direction(dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST, direction)
    print(f"\nDEV baseline sum={dev_baseline_payoff.sum():.4f}")

    def bar_payoff(cls, outcome, tp, sl, cost):
        payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
        payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
        realized = np.where(cls == 1, payoff_if_long, np.where(cls == 2, payoff_if_short, 0.0))
        realized = np.where(cls != 0, realized - cost, 0.0)
        return realized

    returns_matrix = np.zeros((len(dev_df), len(SKEW_THRESHOLDS)))
    for j, th in enumerate(SKEW_THRESHOLDS):
        cls = skew_calls(dev_df, th)
        payoff = bar_payoff(cls, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
        returns_matrix[:, j] = payoff
        print(f"  skew_threshold={th:.1f}  n_trades={int((cls != 0).sum()):6d}  sum={payoff.sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_th = SKEW_THRESHOLDS[best_j]
    print(f"\nBest-of-{len(SKEW_THRESHOLDS)} on DEV: skew_threshold={best_th}")

    print("\n=== GATE 1: falsification audit on the DEV skew-threshold search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED. No VAL/OOS spent.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: best skew-model sum ({returns_matrix[:, best_j].sum():.4f}) vs baseline ({dev_baseline_payoff.sum():.4f}) ===")
    if returns_matrix[:, best_j].sum() <= dev_baseline_payoff.sum():
        print("\n[STOP] GATE 2 FAILED -- skew-based entry does not beat always betting the favored side.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_baseline_payoff = payoff_for_direction(val_outcome, val_tp, val_sl, ROUND_TRIP_COST, direction)
    val_cls = skew_calls(val_df, best_th)
    val_payoff = bar_payoff(val_cls, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    print(f"\n=== VAL: baseline sum={val_baseline_payoff.sum():.4f}  skew-model sum={val_payoff.sum():.4f}  "
          f"n_trades={int((val_cls != 0).sum())} ===")
    if val_payoff.sum() <= val_baseline_payoff.sum():
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_baseline_payoff = payoff_for_direction(oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST, direction)
    oos_cls = skew_calls(oos_df, best_th)
    oos_payoff = bar_payoff(oos_cls, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    print(f"\n=== Partial OOS: baseline sum={oos_baseline_payoff.sum():.4f}  skew-model sum={oos_payoff.sum():.4f}  "
          f"n_trades={int((oos_cls != 0).sum())} ===")


if __name__ == "__main__":
    main()
