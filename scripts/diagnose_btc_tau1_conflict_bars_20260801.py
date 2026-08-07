#!/usr/bin/env python3
"""Diagnose WHY research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py found an
abnormally high conflict-bar rate (~39-40% of all hourly bars, vs ETH Tau1's small trade-pair
count) -- is it structural (both legs are long-hold trend-followers, often simultaneously active)
or is the huge PnL jump mostly explained by regime_side (bull_prob>=bear_prob) being near-constant
over these two specific windows, which would make the "tiebreak" collapse to something closer to
"always favor whichever leg happened to be long/short during this trending stretch" -- a much more
overfit-prone rule than genuine regime-adaptive gating. See
project-btc-tau1-style-leg-combination-first-attempt-20260801.md for full context.

Reuses the joint-portfolio script's own trade loading / leg-state / regime-loading functions
UNCHANGED -- no new mechanism, purely additional printout on the same already-sanity-checked data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    omega_trades_from_ledger, build_leg_equity_path,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    WINDOWS, PFX, load_5m_prices_btc, load_regime_probs, leg_side_series,
)


def diagnose_window(label, start, end, leg_a_path, leg_b_path, prices, regime):
    trades_a = omega_trades_from_ledger(leg_a_path, start, end)
    trades_b = omega_trades_from_ledger(leg_b_path, start, end)
    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)

    eq_a_1h = eq_a.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    n = len(ts)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    both_active = active_a & active_b
    conflict = both_active & (side_a != side_b)
    agree = both_active & (side_a == side_b)

    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)]
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame[f"{PFX}bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame[f"{PFX}bear_prob"].fillna(0.5).to_numpy()
    regime_side = np.where(bull >= bear, 1, -1)

    # regime persistence: how often does regime_side FLIP bar-to-bar vs stay constant
    regime_flips = int(np.sum(regime_side[1:] != regime_side[:-1]))
    regime_bull_frac = float((regime_side == 1).mean())

    # during conflict bars, which side does regime agree with more often
    conflict_regime_agrees_a = np.where(conflict, side_a == regime_side, False).sum()
    conflict_regime_agrees_b = np.where(conflict, side_b == regime_side, False).sum()

    print(f"\n=== DIAGNOSIS {label} {start.date()}..{end.date()} (n_bars={n}) ===")
    print(f"Leg A active: {active_a.sum()} bars ({active_a.mean()*100:.1f}%)   "
          f"Leg B active: {active_b.sum()} bars ({active_b.mean()*100:.1f}%)")
    print(f"Both active : {both_active.sum()} bars ({both_active.mean()*100:.1f}% of all bars)")
    print(f"  -> of both-active bars: agree={agree.sum()} ({agree.sum()/max(both_active.sum(),1)*100:.1f}%), "
          f"conflict={conflict.sum()} ({conflict.sum()/max(both_active.sum(),1)*100:.1f}%)")
    print(f"Regime_side: bull_frac={regime_bull_frac*100:.1f}%  bear_frac={(1-regime_bull_frac)*100:.1f}%  "
          f"flips={regime_flips} (over {n} bars, {regime_flips/max(n,1)*100:.2f}% flip rate)")
    print(f"During conflict bars (n={conflict.sum()}): regime agrees with Leg A side "
          f"{conflict_regime_agrees_a} times ({conflict_regime_agrees_a/max(conflict.sum(),1)*100:.1f}%), "
          f"with Leg B side {conflict_regime_agrees_b} times ({conflict_regime_agrees_b/max(conflict.sum(),1)*100:.1f}%)")
    # trade-level side distribution (not bar-weighted) -- are the legs themselves directionally lopsided?
    a_long = sum(1 for t in trades_a if t["side"] > 0)
    b_long = sum(1 for t in trades_b if t["side"] > 0)
    print(f"Leg A trades: {len(trades_a)} total, {a_long} long / {len(trades_a)-a_long} short")
    print(f"Leg B trades: {len(trades_b)} total, {b_long} long / {len(trades_b)-b_long} short")


def main() -> int:
    prices = load_5m_prices_btc()
    regime = load_regime_probs()
    for label, s, e, la, lb in WINDOWS:
        diagnose_window(label, s, e, la, lb, prices, regime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
