#!/usr/bin/env python3
"""Re-runs research_eth_sigma6_walkforward_omega461_joint_portfolio_20260801.run_window() but dumps
the bar-level equity curves (Omega alone / Sigma6-filtered alone / combined) and both legs' trade
lists to CSV, for charting VAL_2025Q4 and OOS_2026H1. DIAGNOSTIC/VISUALIZATION ONLY -- reuses that
script's validated equity-path methodology unchanged, does not re-derive any numbers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
from run_sigma6_regime_trend_20260705 import load_tape_with_regime  # noqa: E402
from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    load_5m_prices, omega_trades_from_ledger, build_leg_equity_path,
    sanity_check_omega_reproduction,
)
from research_eth_sigma6_walkforward_omega461_joint_portfolio_20260801 import (  # noqa: E402
    WINNER, sanity_check_sigma6_reproduction, WINDOWS,
)

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_walkforward_omega461_joint_portfolio"


def dump_window(label, start, end, prices, omega_ledger_path):
    raw = load_tape_with_regime()
    tape = v2.apply_quality_threshold(raw, WINNER["thr"])
    s6_trades = sanity_check_sigma6_reproduction(tape, prices, start, end, label)
    om_trades = omega_trades_from_ledger(omega_ledger_path, start, end)
    eq_a = sanity_check_omega_reproduction(om_trades, prices, start, end, label)
    eq_b = build_leg_equity_path(s6_trades, prices, start, end, use_ledger_trade_return=False)
    eq_ab = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)

    curve = pd.DataFrame({"omega_equity": eq_a, "sigma6_equity": eq_b, "combined_equity": eq_ab})
    curve.index.name = "timestamp"
    curve.to_csv(OUT_DIR / f"equity_curve_{label}.csv")

    pd.DataFrame(om_trades).to_csv(OUT_DIR / f"omega_trades_{label}.csv", index=False)
    pd.DataFrame(s6_trades).to_csv(OUT_DIR / f"sigma6_trades_{label}.csv", index=False)
    print(f"[{label}] wrote equity_curve ({len(curve)} bars), omega_trades ({len(om_trades)}), "
          f"sigma6_trades ({len(s6_trades)})")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices()
    for label, start, end, ledger in WINDOWS:
        dump_window(label, start, end, prices, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
