#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #6): does bar-level taker buy/sell volume
imbalance (order-flow imbalance, OFI -- a cheap proxy for informed-trading pressure, distinct
from idea #3's total-volume-surprise Levy area) carry ETH direction information? Motivated by
Rajendran & Singaravelu 2026 ("Predicting Adverse Selection in High-Frequency Cryptocurrency
Markets Using Gradient Boosting") and the deep-order-flow-imbalance LOB literature (Kolm/Turiel/
Westray) -- both suggest order flow, not raw volume, is where information lives.

Uses this repo's own `taker_buy_base` column (Binance kline field) to build
imbalance_t = 2*taker_buy_base_t/volume_t - 1 in [-1, 1], then rolling means over several
windows as features.

METHODOLOGY UPGRADE from idea #5's finding: raw sum-vs-baseline unfairly penalizes any model
that ever abstains. This script instead computes, bar-by-bar,
    excess_t = model_payoff_t - stronger_always_directional_payoff_t
and gates on whether SUM(excess) is significantly > 0 (one-sample t-test), which credits a
model correctly for smart abstention instead of charging it the full baseline opportunity cost
without any offsetting credit for genuinely avoided losers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402

PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
WINDOWS = [3, 6, 12, 24, 48]
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    price = pd.read_csv(PRICE_PATH, usecols=["timestamp", "volume", "taker_buy_base"], parse_dates=["timestamp"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = price.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def bar_payoff(predicted_class, tradeable, outcome, tp, sl, cost=0.0):
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def stronger_baseline_payoff(outcome, tp, sl, cost) -> tuple[np.ndarray, str]:
    n = len(outcome)
    long_net = bar_payoff(np.ones(n), np.ones(n, bool), outcome, tp, sl, cost)
    short_net = bar_payoff(np.full(n, 2), np.ones(n, bool), outcome, tp, sl, cost)
    if long_net.sum() >= short_net.sum():
        return long_net, "always_long"
    return short_net, "always_short"


def excess_over_baseline(model_payoff: np.ndarray, baseline_payoff: np.ndarray) -> np.ndarray:
    return model_payoff - baseline_payoff


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
    merged["ofi"] = np.where(merged["volume"] > 0, 2 * merged["taker_buy_base"] / merged["volume"] - 1.0, 0.0)
    feat_cols = []
    for w in WINDOWS:
        col = f"ofi_mean_w{w}"
        merged[col] = merged["ofi"].rolling(w, min_periods=w).mean()
        feat_cols.append(col)
    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})  feat_cols={feat_cols}")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    dev_outcome, dev_tp, dev_sl = dev_df["trade_outcome_action"].to_numpy(), dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()
    dev_baseline_payoff, dev_baseline_name = stronger_baseline_payoff(dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
    print(f"DEV baseline: {dev_baseline_name}, sum={dev_baseline_payoff.sum():.4f}")

    # Single config this time (all 5 OFI windows together) -- the "search" is whether OFI
    # carries anything at all, not a per-window architecture hunt (already learned that lesson).
    model, dev_pred = train_and_score(train_df, dev_df, feat_cols)
    dev_model_payoff = bar_payoff(dev_pred, dev_pred != 0, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
    dev_excess = excess_over_baseline(dev_model_payoff, dev_baseline_payoff)
    print(f"DEV: model sum={dev_model_payoff.sum():.4f}  excess sum={dev_excess.sum():.4f}  "
          f"excess mean={dev_excess.mean():.6f}")

    # Falsification audit needs >=2 columns; pair the real excess series against a "null" column
    # built from OFI computed on a randomly-shuffled volume/taker_buy_base pairing (destroys the
    # true buy/sell split while preserving each column's own marginal distribution).
    rng = np.random.default_rng(20260809)
    shuffled = merged.copy()
    shuffled["taker_buy_base"] = rng.permutation(shuffled["taker_buy_base"].to_numpy())
    shuffled["ofi"] = np.where(shuffled["volume"] > 0, 2 * shuffled["taker_buy_base"] / shuffled["volume"] - 1.0, 0.0)
    for w in WINDOWS:
        shuffled[f"ofi_mean_w{w}"] = shuffled["ofi"].rolling(w, min_periods=w).mean()
    shuffled = shuffled.dropna(subset=feat_cols).reset_index(drop=True)
    shuffled_dev = shuffled[(shuffled["timestamp"] >= DEV_START) & (shuffled["timestamp"] <= DEV_END)]
    shuffled_train = shuffled[shuffled["timestamp"] <= TRAIN_END]
    null_model, null_pred = train_and_score(shuffled_train, shuffled_dev, feat_cols)
    null_outcome, null_tp, null_sl = shuffled_dev["trade_outcome_action"].to_numpy(), shuffled_dev["tp_move"].to_numpy(), shuffled_dev["sl_move"].to_numpy()
    null_baseline_payoff, _ = stronger_baseline_payoff(null_outcome, null_tp, null_sl, ROUND_TRIP_COST)
    null_model_payoff = bar_payoff(null_pred, null_pred != 0, null_outcome, null_tp, null_sl, ROUND_TRIP_COST)
    null_excess = excess_over_baseline(null_model_payoff, null_baseline_payoff)

    n = min(len(dev_excess), len(null_excess))
    returns_matrix = np.column_stack([dev_excess[:n], null_excess[:n]])
    print("\n=== GATE 1: falsification audit (real OFI-excess vs shuffled-OFI-excess control) ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: is DEV excess significantly > 0? (real vs shuffled-taker-split control) ===")
    t_real, p_real = stats.ttest_1samp(dev_excess, 0.0)
    t_null, p_null = stats.ttest_1samp(null_excess[:n], 0.0)
    print(f"  real:     mean={dev_excess.mean():.6f}  t={t_real:.4f}  p={p_real:.6f}")
    print(f"  shuffled: mean={null_excess[:n].mean():.6f}  t={t_null:.4f}  p={p_null:.6f}")
    if not (dev_excess.mean() > 0 and p_real < 0.05):
        print("\n[STOP] GATE 2 FAILED -- excess not significantly positive on DEV. No VAL/OOS spent.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_baseline_payoff, val_baseline_name = stronger_baseline_payoff(val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_pred = model.predict(val_df[feat_cols])
    val_model_payoff = bar_payoff(val_pred, val_pred != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_excess = excess_over_baseline(val_model_payoff, val_baseline_payoff)
    t_val, p_val = stats.ttest_1samp(val_excess, 0.0)
    print(f"\n=== VAL: baseline={val_baseline_name}  excess mean={val_excess.mean():.6f}  sum={val_excess.sum():.4f}  "
          f"t={t_val:.4f}  p={p_val:.6f} ===")
    if not (val_excess.mean() > 0 and p_val < 0.05):
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_baseline_payoff, oos_baseline_name = stronger_baseline_payoff(oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_pred = model.predict(oos_df[feat_cols])
    oos_model_payoff = bar_payoff(oos_pred, oos_pred != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_excess = excess_over_baseline(oos_model_payoff, oos_baseline_payoff)
    print(f"\n=== Partial OOS: baseline={oos_baseline_name}  excess mean={oos_excess.mean():.6f}  sum={oos_excess.sum():.4f} ===")


if __name__ == "__main__":
    main()
