#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #7): does BTC's recent trailing return
predict ETH's NEXT direction (cross-asset lead-lag / information-leadership spillover),
beyond what ETH's own trailing return already tells you? Crypto microstructure literature
broadly holds BTC leads altcoins on liquidity/information grounds; this project's own
[[project-btc-sol-fix-transfer-check-20260720]]-style transfer checks test parameter transfer,
not lead-lag information transfer between assets, so this is a genuinely different question.

Feature set: BTC trailing log-returns over {3,6,12,24,48} bars, ALONE (no ETH price feature at
all) predicting ETH's trade_outcome_action -- a clean test of whether BTC's own path alone
carries information about ETH's future, isolated from ETH's own momentum (idea #4/RETRACTED
already showed ETH's own momentum carries nothing beyond the label's structural bias).

Uses the excess-over-baseline methodology (idea #6) and the shuffled-control falsification audit
(pair real BTC-lag features against the SAME features computed on a randomly time-shifted copy
of BTC's price series, which destroys true lead-lag alignment while preserving BTC's own
autocorrelation/vol-clustering fingerprint).
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

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
ETH_LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
BTC_PRICE_PATH = ROOT / "data/btc_5m_1year.csv"
WINDOWS = [3, 6, 12, 24, 48]
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    eth_price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    btc_price = pd.read_csv(BTC_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "btc_close"})
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = eth_price.merge(btc_price, on="timestamp", how="inner").merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def bar_payoff(predicted_class, tradeable, outcome, tp, sl, cost=0.0):
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def stronger_baseline_payoff(outcome, tp, sl, cost):
    n = len(outcome)
    long_net = bar_payoff(np.ones(n), np.ones(n, bool), outcome, tp, sl, cost)
    short_net = bar_payoff(np.full(n, 2), np.ones(n, bool), outcome, tp, sl, cost)
    return (long_net, "always_long") if long_net.sum() >= short_net.sum() else (short_net, "always_short")


def train_and_score(train_df, eval_df, feat_cols, label_col="trade_outcome_action"):
    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=200,
        objective="multiclass", num_class=3, random_state=270705, verbosity=-1,
    )
    model.fit(train_df[feat_cols], train_df[label_col])
    return model, model.predict(eval_df[feat_cols])


def build_btc_lag_features(df: pd.DataFrame, btc_close_col: str) -> list[str]:
    log_btc = np.log(df[btc_close_col].to_numpy())
    cols = []
    for w in WINDOWS:
        ret = log_btc - np.roll(log_btc, w)
        ret[:w] = np.nan
        col = f"btc_ret_w{w}"
        df[col] = ret
        cols.append(col)
    return cols


def main() -> None:
    merged = load_merged()
    feat_cols = build_btc_lag_features(merged, "btc_close")
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

    model, dev_pred = train_and_score(train_df, dev_df, feat_cols)
    dev_model_payoff = bar_payoff(dev_pred, dev_pred != 0, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
    dev_excess = dev_model_payoff - dev_baseline_payoff
    print(f"DEV: model sum={dev_model_payoff.sum():.4f}  excess sum={dev_excess.sum():.4f}  excess mean={dev_excess.mean():.6f}")

    # Falsification control: recompute the same BTC-lag features on a copy of BTC's price series
    # circularly shifted in TIME by a large, fixed offset (destroys true cross-asset alignment,
    # preserves BTC's own autocorrelation/vol-clustering fingerprint).
    shuffled = merged.copy()
    shift = len(shuffled) // 2
    shuffled["btc_close"] = np.roll(shuffled["btc_close"].to_numpy(), shift)
    shuffled_feat_cols = build_btc_lag_features(shuffled, "btc_close")
    shuffled = shuffled.dropna(subset=shuffled_feat_cols).reset_index(drop=True)
    shuffled_train = shuffled[shuffled["timestamp"] <= TRAIN_END]
    shuffled_dev = shuffled[(shuffled["timestamp"] >= DEV_START) & (shuffled["timestamp"] <= DEV_END)]
    null_model, null_pred = train_and_score(shuffled_train, shuffled_dev, shuffled_feat_cols)
    null_outcome, null_tp, null_sl = shuffled_dev["trade_outcome_action"].to_numpy(), shuffled_dev["tp_move"].to_numpy(), shuffled_dev["sl_move"].to_numpy()
    null_baseline_payoff, _ = stronger_baseline_payoff(null_outcome, null_tp, null_sl, ROUND_TRIP_COST)
    null_model_payoff = bar_payoff(null_pred, null_pred != 0, null_outcome, null_tp, null_sl, ROUND_TRIP_COST)
    null_excess = null_model_payoff - null_baseline_payoff

    n = min(len(dev_excess), len(null_excess))
    returns_matrix = np.column_stack([dev_excess[:n], null_excess[:n]])
    print("\n=== GATE 1: falsification audit (real BTC-lead excess vs time-shifted-BTC control) ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED.")
        return
    print("\n[GATE 1 PASSED]")

    t_real, p_real = stats.ttest_1samp(dev_excess, 0.0)
    print(f"\n=== GATE 2: DEV excess significantly > 0? mean={dev_excess.mean():.6f} t={t_real:.4f} p={p_real:.6f} ===")
    if not (dev_excess.mean() > 0 and p_real < 0.05):
        print("\n[STOP] GATE 2 FAILED. No VAL/OOS spent.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_baseline_payoff, val_baseline_name = stronger_baseline_payoff(val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_pred = model.predict(val_df[feat_cols])
    val_model_payoff = bar_payoff(val_pred, val_pred != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_excess = val_model_payoff - val_baseline_payoff
    t_val, p_val = stats.ttest_1samp(val_excess, 0.0)
    print(f"\n=== VAL: baseline={val_baseline_name}  excess mean={val_excess.mean():.6f}  sum={val_excess.sum():.4f}  t={t_val:.4f}  p={p_val:.6f} ===")
    if not (val_excess.mean() > 0 and p_val < 0.05):
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_baseline_payoff, oos_baseline_name = stronger_baseline_payoff(oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_pred = model.predict(oos_df[feat_cols])
    oos_model_payoff = bar_payoff(oos_pred, oos_pred != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_excess = oos_model_payoff - oos_baseline_payoff
    print(f"\n=== Partial OOS: baseline={oos_baseline_name}  excess mean={oos_excess.mean():.6f}  sum={oos_excess.sum():.4f} ===")


if __name__ == "__main__":
    main()
