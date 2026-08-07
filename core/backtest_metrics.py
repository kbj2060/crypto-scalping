#!/usr/bin/env python3
"""Standard backtest performance metrics -- P1-1 of
docs/pipeline_integrity_and_research_redesign_20260730.md.

Every report.json in this repo before 2026-07-28 computed MDD only from the trade-return
compounding curve (`cash *= 1 + trade_return` at each exit), which is blind to intra-trade
drawdown. tmp/research_20260728/three_asset_bar_level_mdd.py found this understates real drawdown
by 2-4pp for every one of ETH/SOL/BTC's VAL and OOS splits once holds run 1-9+ days at notional up
to ~1.8-2.7x equity (see project memory project-3asset-bar-level-mdd-remeasure-20260728): ETH OOS
bar-level MDD -28.3% vs trade-ledger -20.2%, SOL VAL -25.7% vs -23.4%, etc.

`bar_level_performance` promotes that script's `perf()` function here so it's a single, tested,
importable definition instead of being reimplemented per research script. New reports should
record BOTH `mdd_bar_level` and `mdd_trade_ledger` and gate on the former.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bar_level_performance(equity_curve: np.ndarray, ledger: pd.DataFrame) -> dict:
    """`equity_curve` is a mark-to-market equity value at every bar (not just at trade exits):
    for bars with no open position it equals cash; for bars with an open position it equals
    `cash * (1 + unrealized_move * notional)`. `ledger` is the trade-level ledger with at least a
    `trade_return` column (per-trade realized return at exit).

    Returns pnl (%), bar_level_mdd (%, from the bar-by-bar equity curve -- the honest number),
    trade_ledger_mdd (%, from the old trade-return-only curve -- kept for comparison/continuity
    with pre-2026-07-28 reports, never for gating), trades, wr (win rate).
    """
    if len(equity_curve) == 0:
        raise ValueError("equity_curve must be non-empty")
    peak = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peak, 1e-12) - 1.0

    ledger_mdd = None
    if len(ledger):
        curve = np.concatenate([[1.0], np.cumprod(1.0 + ledger["trade_return"].to_numpy())])
        lpeak = np.maximum.accumulate(curve)
        ledger_mdd = float((curve / np.maximum(lpeak, 1e-12) - 1.0).min() * 100.0)

    return {
        "pnl": float((equity_curve[-1] - 1.0) * 100.0),
        "mdd_bar_level": float(dd.min() * 100.0),
        "mdd_trade_ledger": ledger_mdd if ledger_mdd is not None else 0.0,
        "trades": int(len(ledger)),
        "wr": float((ledger["trade_return"] > 0).mean()) if len(ledger) else 0.0,
    }
