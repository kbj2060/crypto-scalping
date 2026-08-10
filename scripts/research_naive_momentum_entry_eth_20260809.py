#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #4): a plain, non-ML momentum-sign rule
(no model, just sign of the trailing window's log return) beat EVERY ML variant tried tonight
(ideas #2 and #3) on the SAME DEV period and label. That was incidental in both prior scripts --
this one gives the momentum rule itself the full, dedicated gate chain it deserves, instead of
leaving it as a footnote baseline, and adds the one thing every prior probe tonight omitted:
a realistic round-trip cost deduction (6bps, the low end of this repo's own historical
microstructure cost-floor estimates -- project-microstructure-1m-edge-study-20260718).

Gates, each a literal `return` in code:
  1. falsification_audit on a DEV sweep across lookback windows.
  2. Winning window's mean payoff, AFTER a 6bps round-trip cost deduction, must be
     significantly > 0 (one-sample t-test) on DEV.
  3. VAL must be positive after the same cost deduction.
  Only then: a (partial, price data ends 2026-02-17) OOS look.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402

PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
WINDOWS = [6, 12, 24, 48, 72, 96, 144, 192, 288]
ROUND_TRIP_COST = 0.0006  # 6bps, low end of this repo's own microstructure cost-floor estimate
TRAIN_END = "2025-06-30 23:55:00"  # kept only for parity with ideas #2/#3; momentum needs no fit
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


def bar_payoff(predicted_class: np.ndarray, tradeable: np.ndarray, outcome: np.ndarray,
               tp: np.ndarray, sl: np.ndarray, cost: float = 0.0) -> np.ndarray:
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def momentum_class(log_close: np.ndarray, window: int) -> np.ndarray:
    ret = log_close - np.roll(log_close, window)
    ret[:window] = np.nan
    return np.where(ret > 0, 1, np.where(ret < 0, 2, 0))


def _col_sharpe(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


def main() -> None:
    merged = load_merged()
    log_close = np.log(merged["close"].to_numpy())
    print(f"Merged rows: {len(merged)} ({merged.timestamp.min()} .. {merged.timestamp.max()})")
    ts = merged["timestamp"]

    dev_mask = (ts >= DEV_START) & (ts <= DEV_END)
    dev_idx = merged.index[dev_mask]
    dev_outcome = merged.loc[dev_idx, "trade_outcome_action"].to_numpy()
    dev_tp, dev_sl = merged.loc[dev_idx, "tp_move"].to_numpy(), merged.loc[dev_idx, "sl_move"].to_numpy()

    returns_matrix_gross = np.zeros((len(dev_idx), len(WINDOWS)), dtype=np.float64)
    returns_matrix_net = np.zeros_like(returns_matrix_gross)
    for j, w in enumerate(WINDOWS):
        cls_full = momentum_class(log_close, w)
        cls_dev = cls_full[dev_idx]
        tradeable = cls_dev != 0
        returns_matrix_gross[:, j] = bar_payoff(cls_dev, tradeable, dev_outcome, dev_tp, dev_sl, cost=0.0)
        returns_matrix_net[:, j] = bar_payoff(cls_dev, tradeable, dev_outcome, dev_tp, dev_sl, cost=ROUND_TRIP_COST)
        n_tr = int(tradeable.sum())
        print(f"  window={w:3d}  n_trades={n_tr:6d}  gross_mean={returns_matrix_gross[:, j].mean():.6f}  "
              f"net_mean(6bps)={returns_matrix_net[:, j].mean():.6f}  net_sum={returns_matrix_net[:, j].sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix_net)))
    best_w = WINDOWS[best_j]
    print(f"\nBest-of-{len(WINDOWS)} on DEV (net of 6bps): window={best_w}")

    print("\n=== GATE 1: falsification audit on the DEV lookback-window search (gross, pre-cost) ===")
    audit = falsification_audit(returns_matrix_gross, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED. No VAL/OOS spent.")
        return
    print("\n[GATE 1 PASSED]")

    print(f"\n=== GATE 2: window={best_w}, net-of-6bps mean significantly > 0 on DEV? ===")
    net = returns_matrix_net[:, best_j]
    traded = net[net != 0]  # note: a bar with tradeable=False contributes an exact 0.0, drop those
    # a bar that trades and nets to exactly 0.0 is possible but vanishingly rare with continuous
    # tp/sl moves; the != 0 filter is a reasonable proxy for "did this bar actually trade"
    t_stat, p_val = stats.ttest_1samp(traded, popmean=0.0)
    print(f"  n_trades={len(traded)}  mean={traded.mean():.6f}  t={t_stat:.4f}  p={p_val:.6f}")
    if not (traded.mean() > 0 and p_val < 0.05):
        print("\n[STOP] GATE 2 FAILED -- not significantly profitable net of a 6bps round-trip cost "
              "on DEV. No VAL/OOS spent.")
        return
    print("\n[GATE 2 PASSED] Proceeding to VAL.")

    val_mask = (ts >= VAL_START) & (ts <= VAL_END)
    val_idx = merged.index[val_mask]
    val_outcome, val_tp, val_sl = merged.loc[val_idx, "trade_outcome_action"].to_numpy(), merged.loc[val_idx, "tp_move"].to_numpy(), merged.loc[val_idx, "sl_move"].to_numpy()
    val_cls = momentum_class(log_close, best_w)[val_idx]
    val_net = bar_payoff(val_cls, val_cls != 0, val_outcome, val_tp, val_sl, cost=ROUND_TRIP_COST)
    val_traded = val_net[val_net != 0]
    t_val, p_val_val = stats.ttest_1samp(val_traded, popmean=0.0) if len(val_traded) >= 3 else (np.nan, np.nan)
    print(f"\n=== VAL check (2025-09-01..2025-12-31), frozen window={best_w}, net of 6bps ===")
    print(f"  n_trades={len(val_traded)}  mean={val_traded.mean():.6f}  sum={val_net.sum():.4f}  "
          f"t={t_val:.4f}  p={p_val_val:.6f}")
    if not (val_net.sum() > 0 and (np.isnan(p_val_val) or p_val_val < 0.05)):
        print("\n[STOP] VAL FAILED (non-positive or not significant net of cost). No OOS spent.")
        return
    print("\n[VAL PASSED] Proceeding to the one, final, partial-OOS look.")

    oos_mask = ts >= OOS_START
    oos_idx = merged.index[oos_mask]
    oos_outcome, oos_tp, oos_sl = merged.loc[oos_idx, "trade_outcome_action"].to_numpy(), merged.loc[oos_idx, "tp_move"].to_numpy(), merged.loc[oos_idx, "sl_move"].to_numpy()
    oos_cls = momentum_class(log_close, best_w)[oos_idx]
    oos_net = bar_payoff(oos_cls, oos_cls != 0, oos_outcome, oos_tp, oos_sl, cost=ROUND_TRIP_COST)
    print(f"\n=== Partial OOS check (2026-01-01..{ts.max().date()}), frozen window={best_w}, net of 6bps ===")
    print(f"  n_trades={int((oos_cls != 0).sum())}  mean={oos_net[oos_net != 0].mean():.6f}  sum={oos_net.sum():.4f}")


if __name__ == "__main__":
    main()
