#!/usr/bin/env python3
"""Does a SIMPLE fixed 1x-1x combination of BTC's two independent legs (h48qual Leg A + Sigma9
trend-scan+regime Leg B) generalize across time, or did the earlier regime_tiebreak result
(+58~60% vs ~+19% baseline on one VAL+OOS split) only look good because it leaned on a BTC regime
signal that flips direction ~34-35% of bars and has already failed fresh-forward validation
elsewhere in this project? See project-btc-tau1-style-leg-combination-first-attempt-20260801.md
(CLOSED as a likely VAL-overfit artifact) for the diagnosis that motivated dropping the tiebreak
mechanism entirely here.

This script removes regime_tiebreak completely -- weights are fixed at 1.0/1.0 for both legs in
EVERY window, no per-bar decision of any kind -- and replays that fixed combination, unmodified,
across several overlapping windows spanning the full range where BOTH legs' saved ledgers have
data. Same discipline as research_sigma6_regime_filter_rolling_windows_20260801.py: no re-tuning
per window (there is nothing to tune -- weights are fixed), just checking whether the combination
beats the stronger single leg (Leg A, BTC h48qual) consistently, or only on the windows already
inspected.

Data-range note: Leg A's saved ledgers (docs/model_contracts/btc_omega4_6_1_full_stack_20260708_
contract.md) only cover 2025-10-01..2026-06-25 (~8.8 months) -- narrower than ETH's Sigma3-1h tape
(2025-06-25 on), so only 4 overlapping 4-month windows fit here (vs ETH's 5), a real data
constraint, not a choice to weaken the test.

Combination mechanics (build_leg_equity_path, additive-dollar-PnL independent sleeves) reused
UNCHANGED from the already-audited/sanity-checked joint-portfolio script -- only the WEIGHTS differ
(fixed 1.0/1.0 here vs the dropped tiebreak logic).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    omega_trades_from_ledger, build_leg_equity_path, summarize_equity,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    LEG_A_DIR, LEG_B_DIR, load_5m_prices_btc,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_sigma9regime_fixedweight_rolling_windows"

# Each leg's FULL saved ledger (VAL+OOS concatenated) is used as one continuous trade source, then
# sliced per window by entry_timestamp -- same convention as the joint-portfolio script, just
# spanning both of that script's ledger files at once so windows can cross the VAL/OOS boundary.
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

WINDOWS = [
    ("W1", "2025-10-01", "2026-01-31"),
    ("W2", "2025-12-01", "2026-03-31"),
    ("W3", "2026-02-01", "2026-05-31"),
    ("W4_data_end", "2026-02-25", "2026-06-25"),
]


def load_all_trades(paths: list[Path], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    trades = []
    for p in paths:
        trades.extend(omega_trades_from_ledger(p, start, end))
    return trades


def run_window(label, start_s, end_s, prices):
    start, end = pd.Timestamp(start_s), pd.Timestamp(end_s) + pd.Timedelta("23h59min59s")
    trades_a = load_all_trades(LEG_A_LEDGERS, start, end)
    trades_b = load_all_trades(LEG_B_LEDGERS, start, end)

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)
    eq_ab = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)  # fixed 1.0/1.0 weights, additive dollar-PnL

    sa, sb, sab = summarize_equity(eq_a), summarize_equity(eq_b), summarize_equity(eq_ab)
    beats_leg_a_both_axes = sab["pnl_pct"] > sa["pnl_pct"] and sab["mdd_pct"] > sa["mdd_pct"]
    print(f"\n=== {label} {start_s}..{end_s} ===")
    print(f"Leg A (h48qual) alone      : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"Leg B (sigma9+regime) alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"Fixed 1x-1x combined       : pnl={sab['pnl_pct']:+.2f}% mdd={sab['mdd_pct']:.2f}%  "
          f"beats_leg_a_both_axes={beats_leg_a_both_axes}")
    return {
        "window": label, "start": start_s, "end": end_s,
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"], "leg_a_trades": len(trades_a),
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"], "leg_b_trades": len(trades_b),
        "combined_pnl": sab["pnl_pct"], "combined_mdd": sab["mdd_pct"],
        "beats_leg_a_both_axes": beats_leg_a_both_axes,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    rows = [run_window(label, s, e, prices) for label, s, e in WINDOWS]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "rolling_window_summary.csv", index=False)
    n_wins = int(df["beats_leg_a_both_axes"].sum())
    print(f"\n=== SUMMARY: fixed-weight combo beats Leg A alone (pnl AND mdd) in {n_wins}/{len(df)} windows ===")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
