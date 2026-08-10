#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #13): a genuine two-model priority-router
simulation, testing the REAL h48qual/zig075 mechanism instead of the single-model skip-filter
proxy used in ideas #8-12.

Checked the actual live router code (trading_bot_modules/omega4_6_1_live.py) before building
this: `PRIORITY = ("h48qual", "zig075")`, h48qual quality_threshold=0.50, zig075
quality_threshold=0.75 -- a priority-ordered greedy router where h48qual (lower bar, goes first)
fires liberally and PREEMPTS zig075 (higher bar) on any bar both would have wanted. This is a
genuinely different mechanism from "one model, sometimes skip" (idea #8-12's framing): it needs
TWO independently-different models with their own direction calls and thresholds.

Model H ("h48qual-analog"): fresh 3-class LightGBM on tonight's combined feature set (ETH
momentum + BTC lead-lag + realized-vol/ATR), fires whenever ITS OWN top-class probability clears
a LOW threshold (mimics h48qual's 0.50).
Model Z ("zig075-analog"): the idea #4 naive-momentum(144) rule -- deliberately a DIFFERENT,
much simpler rule family, fires whenever it has a nonzero directional call (mimics zig075's
higher-frequency, always-eager behavior).

Priority rule: on each bar, if Model H clears its threshold, take Model H's direction; else if
Model Z wants to fire, take Model Z's direction; else cash.

Test: does the COMBINED router beat Model Z running ALONE, unimpeded, on every bar? This is the
actual shape of the h48qual/zig075 "blocking, not earning" claim -- H doesn't need genuine
skill on its own trades, it only needs to preempt Z on bars where Z would have done worse than
Z's own average.
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
from pipeline.architecture_workbench import effect_size_report  # noqa: E402

ETH_PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
ETH_LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
BTC_PRICE_PATH = ROOT / "data/btc_5m_1year.csv"
WINDOWS = [3, 6, 12, 24, 48]
Z_MOMENTUM_WINDOW = 144  # frozen from idea #4
H_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60]  # sweep around h48qual's real 0.50
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


def build_h_features(df: pd.DataFrame) -> list[str]:
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


def z_momentum_class(df: pd.DataFrame) -> np.ndarray:
    log_close = np.log(df["close"].to_numpy())
    ret = log_close - np.roll(log_close, Z_MOMENTUM_WINDOW)
    ret[:Z_MOMENTUM_WINDOW] = np.nan
    return np.where(ret > 0, 1, np.where(ret < 0, 2, 0))


def bar_payoff(predicted_class, tradeable, outcome, tp, sl, cost=0.0):
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def _col_sharpe(m):
    mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


def main() -> None:
    merged = load_merged()
    h_feat_cols = build_h_features(merged)
    merged["z_class"] = z_momentum_class(merged)
    merged = merged.dropna(subset=h_feat_cols + ["z_class"]).reset_index(drop=True)
    ts = merged["timestamp"]
    print(f"Merged rows: {len(merged)} ({ts.min()} .. {ts.max()})")

    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial)={len(oos_df)}")

    h_model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                 min_child_samples=100, objective="multiclass", num_class=3,
                                 random_state=270705, verbosity=-1)
    h_model.fit(train_df[h_feat_cols], train_df["trade_outcome_action"])

    def h_calls(split_df):
        proba = h_model.predict_proba(split_df[h_feat_cols])
        top_class = np.argmax(proba, axis=1)
        top_prob = proba[np.arange(len(proba)), top_class]
        return top_class, top_prob

    dev_outcome, dev_tp, dev_sl = dev_df["trade_outcome_action"].to_numpy(), dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()
    dev_z_class = dev_df["z_class"].to_numpy()
    dev_h_class, dev_h_prob = h_calls(dev_df)

    z_alone = bar_payoff(dev_z_class, dev_z_class != 0, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
    print(f"\nZ-alone (naive momentum(144), unimpeded) DEV sum={z_alone.sum():.4f}  n_trades={int((dev_z_class != 0).sum())}")

    returns_matrix = np.zeros((len(dev_df), len(H_THRESHOLDS)))
    for j, th in enumerate(H_THRESHOLDS):
        h_fires = (dev_h_class != 0) & (dev_h_prob >= th)
        final_class = np.where(h_fires, dev_h_class, np.where(dev_z_class != 0, dev_z_class, 0))
        tradeable = final_class != 0
        combined = bar_payoff(final_class, tradeable, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
        returns_matrix[:, j] = combined
        h_share = float(h_fires.sum()) / max(1, int(tradeable.sum()))
        print(f"  H_threshold={th:.2f}  h_fires={int(h_fires.sum()):6d}  combined_n={int(tradeable.sum()):6d}  "
              f"h_share_of_trades={h_share:.3f}  combined_sum={combined.sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_th = H_THRESHOLDS[best_j]
    print(f"\nBest-of-{len(H_THRESHOLDS)} on DEV: H_threshold={best_th}")

    print("\n=== GATE 1: falsification audit on the DEV H-threshold search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED. No VAL/OOS spent.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: best combined sum ({returns_matrix[:, best_j].sum():.4f}) vs Z-alone ({z_alone.sum():.4f}) ===")
    if returns_matrix[:, best_j].sum() <= z_alone.sum():
        print("\n[STOP] GATE 2 FAILED -- the router does not beat Z running alone, unimpeded.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_z_class = val_df["z_class"].to_numpy()
    val_h_class, val_h_prob = h_calls(val_df)
    val_z_alone = bar_payoff(val_z_class, val_z_class != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_h_fires = (val_h_class != 0) & (val_h_prob >= best_th)
    val_final = np.where(val_h_fires, val_h_class, np.where(val_z_class != 0, val_z_class, 0))
    val_combined = bar_payoff(val_final, val_final != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    print(f"\n=== VAL: Z-alone sum={val_z_alone.sum():.4f}  combined sum={val_combined.sum():.4f} ===")
    if val_combined.sum() <= val_z_alone.sum():
        print("\n[STOP] VAL FAILED. No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
    oos_z_class = oos_df["z_class"].to_numpy()
    oos_h_class, oos_h_prob = h_calls(oos_df)
    oos_z_alone = bar_payoff(oos_z_class, oos_z_class != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    oos_h_fires = (oos_h_class != 0) & (oos_h_prob >= best_th)
    oos_final = np.where(oos_h_fires, oos_h_class, np.where(oos_z_class != 0, oos_z_class, 0))
    oos_combined = bar_payoff(oos_final, oos_final != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    print(f"\n=== Partial OOS: Z-alone sum={oos_z_alone.sum():.4f}  combined sum={oos_combined.sum():.4f} ===")


if __name__ == "__main__":
    main()
