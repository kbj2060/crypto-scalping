#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #5): do session/time-of-day/day-of-week
calendar features, ALONE (no price data at all), carry ETH direction information?

Motivation: arXiv 2605.04004 (Structural Limits of OHLCV-Based Intraday Signals, systematic
falsification study on 5m MNQ futures) found 14 pure price-pattern signal families ALL fail a
strict deployment bar, and the only 2 survivors were session/liquidity-timing effects (RTH
confluence, London session) -- i.e. session timing may be the one OHLCV-adjacent signal class
that isn't already arbitraged away. Untested in this repo's registry.

Corrected discipline (see project-baseline-must-be-always-long-short-not-zero-20260809, found
earlier tonight): the baseline is max(always_long, always_short) on the SAME period, never zero.
A calendar-only model that predicts "cash" on some bars will forfeit part of the label's built-in
barrier-geometry edge on those bars, so it must be compared against the always-directional
baselines directly, not against a naive one-sample-vs-zero test.

Gates, each a literal `return` in code:
  1. falsification_audit on the DEV feature-set search (raw hour/dow vs full calendar set).
  2. Winning model's DEV sum must beat max(always_long_sum, always_short_sum) on DEV.
  3. VAL sum must beat max(always_long_sum, always_short_sum) on VAL.
  Only then: a (partial) OOS look.
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

PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    price = pd.read_csv(PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = price.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def add_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    ts = df["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek  # 0=Mon .. 6=Sun
    df["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["cal_is_weekend"] = (dow >= 5).astype(np.float64)
    df["cal_is_asia"] = ((hour >= 0) & (hour < 8)).astype(np.float64)
    df["cal_is_europe"] = ((hour >= 8) & (hour < 13)).astype(np.float64)
    df["cal_is_us"] = ((hour >= 13) & (hour < 22)).astype(np.float64)
    df["cal_is_overlap_eu_us"] = ((hour >= 13) & (hour < 17)).astype(np.float64)
    raw_cols = ["cal_hour_sin", "cal_hour_cos", "cal_dow_sin", "cal_dow_cos"]
    full_cols = raw_cols + ["cal_is_weekend", "cal_is_asia", "cal_is_europe", "cal_is_us", "cal_is_overlap_eu_us"]
    return df, raw_cols, full_cols


def bar_payoff(predicted_class, tradeable, outcome, tp, sl, cost=0.0):
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def always_directional_sums(outcome, tp, sl, cost):
    n = len(outcome)
    long_net = bar_payoff(np.ones(n), np.ones(n, bool), outcome, tp, sl, cost)
    short_net = bar_payoff(np.full(n, 2), np.ones(n, bool), outcome, tp, sl, cost)
    return long_net.sum(), short_net.sum()


def train_and_score(train_df, eval_df, feat_cols, label_col="trade_outcome_action"):
    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=200,
        objective="multiclass", num_class=3, random_state=270705, verbosity=-1,
    )
    model.fit(train_df[feat_cols], train_df[label_col])
    return model, model.predict(eval_df[feat_cols])


def _col_sharpe(m):
    mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


def main() -> None:
    merged = load_merged()
    merged, raw_cols, full_cols = add_calendar_features(merged)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    dev_outcome, dev_tp, dev_sl = dev_df["trade_outcome_action"].to_numpy(), dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()
    dev_long_sum, dev_short_sum = always_directional_sums(dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
    dev_baseline = max(dev_long_sum, dev_short_sum)
    print(f"DEV baseline: always_long_sum={dev_long_sum:.4f} always_short_sum={dev_short_sum:.4f} -> baseline={dev_baseline:.4f}")

    configs = {"raw_hour_dow": raw_cols, "full_calendar": full_cols}
    returns_matrix = np.zeros((len(dev_df), len(configs)))
    models = {}
    for j, (name, cols) in enumerate(configs.items()):
        model, pred = train_and_score(train_df, dev_df, cols)
        models[name] = (model, cols)
        tradeable = pred != 0
        returns_matrix[:, j] = bar_payoff(pred, tradeable, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
        print(f"  {name:14s} n_trades={int(tradeable.sum()):6d} sum={returns_matrix[:, j].sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_name = list(configs.keys())[best_j]
    print(f"\nBest-of-{len(configs)} on DEV: {best_name}")

    print("\n=== GATE 1: falsification audit on the DEV feature-set search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: {best_name} DEV sum ({returns_matrix[:, best_j].sum():.4f}) vs baseline ({dev_baseline:.4f}) ===")
    if returns_matrix[:, best_j].sum() <= dev_baseline:
        print("\n[STOP] GATE 2 FAILED -- does not beat the always-directional baseline on DEV.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    model, cols = models[best_name]
    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_long_sum, val_short_sum = always_directional_sums(val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_baseline = max(val_long_sum, val_short_sum)
    val_pred = model.predict(val_df[cols])
    val_net = bar_payoff(val_pred, val_pred != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    print(f"\n=== VAL: {best_name} sum={val_net.sum():.4f}  baseline (always_long={val_long_sum:.4f}, "
          f"always_short={val_short_sum:.4f}) -> {val_baseline:.4f} ===")
    if val_net.sum() <= val_baseline:
        print("\n[STOP] VAL FAILED -- does not beat the always-directional baseline. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_long_sum, oos_short_sum = always_directional_sums(oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_baseline = max(oos_long_sum, oos_short_sum)
    oos_pred = model.predict(oos_df[cols])
    oos_net = bar_payoff(oos_pred, oos_pred != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    print(f"\n=== Partial OOS: {best_name} sum={oos_net.sum():.4f}  baseline -> {oos_baseline:.4f} ===")


if __name__ == "__main__":
    main()
