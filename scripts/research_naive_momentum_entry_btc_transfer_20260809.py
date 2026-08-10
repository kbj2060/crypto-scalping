#!/usr/bin/env python3
"""Robustness check (2026-08-09) for overnight loop idea #4: does the same naive momentum-sign
rule (best window found on ETH DEV = 144 bars, but swept fresh here rather than hardcoded, to
avoid contaminating BTC's own search with an ETH-selected window) survive an identical gate
chain on BTC? This project's own history treats single-asset "wins" as unproven until they
either transfer or are shown to be asset-specific for an understood reason
(project-btc-sol-fix-transfer-check-20260720 and many others). A momentum effect that only
"works" on one asset by chance is exactly what the falsification-audit work tonight exists to
catch -- this is an out-of-search robustness check, not a second bite at the same search.
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

PRICE_PATH = ROOT / "data/btc_5m_1year.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/btc_5m_tripbarrier_tradeoutcome_labels_regimeline_20260808.parquet"
WINDOWS = [6, 12, 24, 48, 72, 96, 144, 192, 288]
ROUND_TRIP_COST = 0.0006
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


def bar_payoff(predicted_class, tradeable, outcome, tp, sl, cost=0.0):
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    realized = np.where(tradeable, realized - cost, 0.0)
    return np.where(tradeable, realized, 0.0)


def momentum_class(log_close, window):
    ret = log_close - np.roll(log_close, window)
    ret[:window] = np.nan
    return np.where(ret > 0, 1, np.where(ret < 0, 2, 0))


def _col_sharpe(m):
    mu, sd = m.mean(axis=0), m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


def main() -> None:
    merged = load_merged()
    log_close = np.log(merged["close"].to_numpy())
    print(f"BTC merged rows: {len(merged)} ({merged.timestamp.min()} .. {merged.timestamp.max()})")
    ts = merged["timestamp"]

    dev_idx = merged.index[(ts >= DEV_START) & (ts <= DEV_END)]
    dev_outcome, dev_tp, dev_sl = merged.loc[dev_idx, "trade_outcome_action"].to_numpy(), merged.loc[dev_idx, "tp_move"].to_numpy(), merged.loc[dev_idx, "sl_move"].to_numpy()

    net_matrix = np.zeros((len(dev_idx), len(WINDOWS)))
    gross_matrix = np.zeros_like(net_matrix)
    for j, w in enumerate(WINDOWS):
        cls_dev = momentum_class(log_close, w)[dev_idx]
        tradeable = cls_dev != 0
        gross_matrix[:, j] = bar_payoff(cls_dev, tradeable, dev_outcome, dev_tp, dev_sl, 0.0)
        net_matrix[:, j] = bar_payoff(cls_dev, tradeable, dev_outcome, dev_tp, dev_sl, ROUND_TRIP_COST)
        print(f"  window={w:3d} n_trades={int(tradeable.sum()):6d} net_mean={net_matrix[:, j].mean():.6f} net_sum={net_matrix[:, j].sum():.4f}")

    best_j = int(np.argmax(_col_sharpe(net_matrix)))
    best_w = WINDOWS[best_j]
    print(f"\nBest-of-{len(WINDOWS)} on BTC DEV (net of 6bps): window={best_w}")

    print("\n=== GATE 1: falsification audit (BTC DEV window search, gross) ===")
    audit = falsification_audit(gross_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")
    if not audit["passes_falsification_audit"]:
        print("\n[STOP] GATE 1 FAILED on BTC.")
        return

    net = net_matrix[:, best_j]
    traded = net[net != 0]
    t_stat, p_val = stats.ttest_1samp(traded, popmean=0.0)
    print(f"\n=== GATE 2: BTC window={best_w}, net-of-6bps mean sig>0 on DEV? n={len(traded)} mean={traded.mean():.6f} t={t_stat:.4f} p={p_val:.6f} ===")
    if not (traded.mean() > 0 and p_val < 0.05):
        print("\n[STOP] GATE 2 FAILED on BTC -- momentum effect does NOT transfer.")
        return
    print("\n[GATE 2 PASSED on BTC] Proceeding to VAL.")

    val_idx = merged.index[(ts >= VAL_START) & (ts <= VAL_END)]
    val_outcome, val_tp, val_sl = merged.loc[val_idx, "trade_outcome_action"].to_numpy(), merged.loc[val_idx, "tp_move"].to_numpy(), merged.loc[val_idx, "sl_move"].to_numpy()
    val_cls = momentum_class(log_close, best_w)[val_idx]
    val_net = bar_payoff(val_cls, val_cls != 0, val_outcome, val_tp, val_sl, ROUND_TRIP_COST)
    val_traded = val_net[val_net != 0]
    t_v, p_v = stats.ttest_1samp(val_traded, 0.0) if len(val_traded) >= 3 else (np.nan, np.nan)
    print(f"\n=== BTC VAL check, window={best_w}: n={len(val_traded)} mean={val_traded.mean():.6f} sum={val_net.sum():.4f} t={t_v:.4f} p={p_v:.6f} ===")
    if not (val_net.sum() > 0 and (np.isnan(p_v) or p_v < 0.05)):
        print("\n[STOP] BTC VAL FAILED.")
        return

    oos_idx = merged.index[ts >= OOS_START]
    oos_outcome, oos_tp, oos_sl = merged.loc[oos_idx, "trade_outcome_action"].to_numpy(), merged.loc[oos_idx, "tp_move"].to_numpy(), merged.loc[oos_idx, "sl_move"].to_numpy()
    oos_cls = momentum_class(log_close, best_w)[oos_idx]
    oos_net = bar_payoff(oos_cls, oos_cls != 0, oos_outcome, oos_tp, oos_sl, ROUND_TRIP_COST)
    print(f"\n=== BTC partial-OOS check (to {ts.max().date()}), window={best_w}: n={int((oos_cls!=0).sum())} mean={oos_net[oos_net!=0].mean():.6f} sum={oos_net.sum():.4f} ===")


if __name__ == "__main__":
    main()
