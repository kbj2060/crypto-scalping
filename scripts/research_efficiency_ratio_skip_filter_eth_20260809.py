#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #15): Kaufman's Efficiency Ratio (ER) as a
skip-filter feature -- a classic, well-established TA measure distinct from everything tried
tonight. ER = |price_t - price_{t-n}| / sum_{i=t-n+1}^{t} |price_i - price_{i-1}|, in [0,1]:
near 1 means a clean, efficient trend (net movement ~= total path length, little backtracking);
near 0 means choppy/noisy movement that round-trips a lot. This is conceptually distinct from
realized-vol/ATR (idea #9, which measures MAGNITUDE of movement, not its directional efficiency)
and from momentum (which measures net direction, not path quality) -- if favored-direction bets
do better specifically during "clean trend" periods, ER should catch it where ATR/momentum did
not.

Same skip-filter framing (freeze TRAIN-favored direction, binary win-probability model, quantile
sweep) as ideas #8-11, #13, gated the same way.
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
WINDOWS = [6, 12, 24, 48, 96]
SKIP_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50]
ROUND_TRIP_COST = 0.0006
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"


def load_merged() -> pd.DataFrame:
    price = pd.read_csv(ETH_PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    labels = pd.read_parquet(ETH_LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = price.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def efficiency_ratio(close: np.ndarray, window: int) -> np.ndarray:
    net_change = np.abs(close - np.roll(close, window))
    abs_diffs = np.abs(np.diff(close, prepend=close[0]))
    path_length = pd.Series(abs_diffs).rolling(window, min_periods=window).sum().to_numpy()
    er = np.where(path_length > 1e-12, net_change / path_length, np.nan)
    er[:window] = np.nan
    return er


def build_features(df: pd.DataFrame) -> list[str]:
    close = df["close"].to_numpy()
    cols = []
    for w in WINDOWS:
        col = f"er_w{w}"
        df[col] = efficiency_ratio(close, w)
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
    print(f"ER stats: {merged[feat_cols].describe().loc[['mean', 'std', 'min', 'max']].to_string()}")

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
        print("\n[STOP] GATE 2 FAILED -- ER filtering does not beat always betting the favored side.")
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
