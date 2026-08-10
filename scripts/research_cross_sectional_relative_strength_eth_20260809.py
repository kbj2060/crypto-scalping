#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #17): cross-sectional relative-strength
skip-filter -- ETH's return RELATIVE to a small BTC+SOL basket, distinct from idea #7's absolute
BTC lead-lag (which tested BTC's own trailing return as a predictor; this tests ETH's SPREAD vs
a basket, a classic relative-strength/pairs construction). Motivated by 2026 cross-sectional
crypto return-prediction literature (multi-relational attention over large coin universes,
ScienceDirect 2026) -- this project only has 3 assets locally, so this is the closest feasible
proxy: does ETH's relative outperformance/underperformance vs BTC+SOL predict which bars are
good for the already-favored direction?
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
SOL_PRICE_PATH = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2024_2026.csv"
WINDOWS = [3, 6, 12, 24, 48, 96]
SKIP_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50]
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    eth_price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    btc_price = pd.read_csv(BTC_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "btc_close"})
    sol_price = pd.read_csv(SOL_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "sol_close"})
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = eth_price.merge(btc_price, on="timestamp", how="inner").merge(sol_price, on="timestamp", how="inner").merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def build_features(df: pd.DataFrame) -> list[str]:
    cols = []
    log_eth = np.log(df["close"].to_numpy())
    log_basket = 0.5 * np.log(df["btc_close"].to_numpy()) + 0.5 * np.log(df["sol_close"].to_numpy())
    for w in WINDOWS:
        eth_ret = log_eth - np.roll(log_eth, w); eth_ret[:w] = np.nan
        basket_ret = log_basket - np.roll(log_basket, w); basket_ret[:w] = np.nan
        relative_strength = eth_ret - basket_ret
        col = f"relstrength_w{w}"
        df[col] = relative_strength
        cols.append(col)
    return cols


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
    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    train_outcome, train_tp, train_sl = train_df["trade_outcome_action"].to_numpy(), train_df["tp_move"].to_numpy(), train_df["sl_move"].to_numpy()
    direction = favored_direction(train_outcome, train_tp, train_sl, ROUND_TRIP_COST)
    print(f"Favored direction on TRAIN: {'long' if direction == 1 else 'short'}")

    train_win = (train_outcome == direction).astype(int)
    model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               min_child_samples=100, random_state=270705, verbosity=-1)
    model.fit(train_df[feat_cols], train_win)

    dev_outcome, dev_tp, dev_sl = dev_df["trade_outcome_action"].to_numpy(), dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()
    dev_baseline_payoff = payoff_for_direction(dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST, direction)
    dev_win_prob = model.predict_proba(dev_df[feat_cols])[:, 1]
    print(f"\nDEV baseline sum={dev_baseline_payoff.sum():.4f}")

    returns_matrix = np.zeros((len(dev_df), len(SKIP_QUANTILES)))
    for j, q in enumerate(SKIP_QUANTILES):
        threshold = np.quantile(dev_win_prob, q)
        keep = dev_win_prob > threshold
        filtered = np.where(keep, dev_baseline_payoff, 0.0)
        returns_matrix[:, j] = filtered
        print(f"  skip_q={q:.2f}  kept={int(keep.sum()):6d}/{len(keep)}  sum={filtered.sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_q = SKIP_QUANTILES[best_j]
    print(f"\nBest-of-{len(SKIP_QUANTILES)} on DEV: skip_q={best_q}")

    print("\n=== GATE 1: falsification audit on the DEV skip-quantile search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED. No VAL/OOS spent.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: best filtered sum ({returns_matrix[:, best_j].sum():.4f}) vs baseline ({dev_baseline_payoff.sum():.4f}) ===")
    if returns_matrix[:, best_j].sum() <= dev_baseline_payoff.sum():
        print("\n[STOP] GATE 2 FAILED -- relative-strength filtering does not beat always betting the favored side.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_baseline_payoff = payoff_for_direction(val_outcome, val_tp, val_sl, ROUND_TRIP_COST, direction)
    val_win_prob = model.predict_proba(val_df[feat_cols])[:, 1]
    val_threshold = np.quantile(dev_win_prob, best_q)
    val_keep = val_win_prob > val_threshold
    val_filtered = np.where(val_keep, val_baseline_payoff, 0.0)
    print(f"\n=== VAL: baseline sum={val_baseline_payoff.sum():.4f}  filtered sum={val_filtered.sum():.4f}  "
          f"kept={int(val_keep.sum())}/{len(val_keep)} ===")
    if val_filtered.sum() <= val_baseline_payoff.sum():
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_baseline_payoff = payoff_for_direction(oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST, direction)
    oos_win_prob = model.predict_proba(oos_df[feat_cols])[:, 1]
    oos_keep = oos_win_prob > val_threshold
    oos_filtered = np.where(oos_keep, oos_baseline_payoff, 0.0)
    print(f"\n=== Partial OOS: baseline sum={oos_baseline_payoff.sum():.4f}  filtered sum={oos_filtered.sum():.4f}  "
          f"kept={int(oos_keep.sum())}/{len(oos_keep)} ===")


if __name__ == "__main__":
    main()
