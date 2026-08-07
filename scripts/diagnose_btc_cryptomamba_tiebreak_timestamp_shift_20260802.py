#!/usr/bin/env python3
"""Decisive before/after test for whether BTC's CryptoMamba regime_tiebreak
(scripts/research_btc_tau1_cryptomamba_tiebreak_20260801.py) shares the same-bar look-ahead bug
class already confirmed in the BTC dumb-momentum control
(scripts/diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py) and already ruled out
for ETH's current-HMM regime_tiebreak.

Trace summary (see project note for full detail):
  - CryptoMamba's own regime CSV timestamp is a pass-through of the raw 5m feature row's OWN
    timestamp (train_regime3_cryptomamba_pred_20260531.py::_output(), out["timestamp"] =
    frame["timestamp"]) -- NOT a resampled/left-labeled 1h bin like the dumb-momentum control's
    close_1h was. Its inference-time feature window for row t is x[t-59:t+1] (seq_len=60,
    inclusive of t, nothing later) -- causal.
  - The tiebreak script's merge (research_btc_tau1_cryptomamba_tiebreak_20260801.py::run_window)
    uses the SAME resample("1h").last() + .diff() + merge_asof(direction="backward") pattern for
    the leg-equity delta as the dumb-momentum case. The difference is what's being merged onto it:
    dumb-momentum's regime signal was ITSELF derived from the same resampled 1h close the delta was
    built from (self-referential same-window peek); CryptoMamba's regime signal is a native 5m-bar,
    causal-only forecast from a wholly separate feature/model pipeline, so no shared-window overlap
    is expected a priori.

This script builds a causally-shifted control: shift the CryptoMamba regime CSV's timestamp forward
by +1h (matching the granularity used in both prior investigations) and reruns the identical
regime_tiebreak VAL/OOS + 4 rolling windows, comparing to the unshifted (leaky-by-construction-audit)
baseline. rule_weights/leg_side_series/weighted_pnl/build_leg_equity_path are reused UNCHANGED from
research_btc_tau1_cryptomamba_tiebreak_20260801.py -- only the regime timestamp alignment differs
between the two runs.
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

from research_btc_tau1_cryptomamba_tiebreak_20260801 import (  # noqa: E402
    load_cryptomamba_regime, load_all_trades, run_window,
    VAL_OOS_WINDOWS, ROLLING_WINDOWS, LEG_A_LEDGERS, LEG_B_LEDGERS,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import load_5m_prices_btc  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260802/btc_cryptomamba_lookahead_check"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime_leaky = load_cryptomamba_regime()
    regime_shifted = regime_leaky.copy()
    regime_shifted["timestamp"] = regime_shifted["timestamp"] + pd.Timedelta(hours=1)

    all_windows = VAL_OOS_WINDOWS + ROLLING_WINDOWS

    print("########## LEAKY (unshifted, as originally reported) ##########")
    rows_leaky = [run_window(l, s, e, prices, regime_leaky) for l, s, e in all_windows]
    df_leaky = pd.DataFrame(rows_leaky)
    df_leaky.to_csv(OUT_DIR / "leaky_unshifted_summary.csv", index=False)

    print("\n########## CAUSAL-SHIFTED CONTROL (regime timestamp +1h) ##########")
    rows_shift = [run_window(l, s, e, prices, regime_shifted) for l, s, e in all_windows]
    df_shift = pd.DataFrame(rows_shift)
    df_shift.to_csv(OUT_DIR / "causal_shifted_summary.csv", index=False)

    compare = pd.DataFrame({
        "window": df_leaky["window"],
        "leaky_pnl": df_leaky["cmamba_tiebreak_pnl"],
        "shifted_pnl": df_shift["cmamba_tiebreak_pnl"],
        "pnl_delta": df_shift["cmamba_tiebreak_pnl"] - df_leaky["cmamba_tiebreak_pnl"],
        "leaky_mdd": df_leaky["cmamba_tiebreak_mdd"],
        "shifted_mdd": df_shift["cmamba_tiebreak_mdd"],
        "leaky_beats_leg_a": df_leaky["beats_leg_a_both_axes"],
        "shifted_beats_leg_a": df_shift["beats_leg_a_both_axes"],
    })
    compare.to_csv(OUT_DIR / "leaky_vs_shifted_comparison.csv", index=False)
    print("\n########## COMPARISON ##########")
    print(compare.to_string(index=False))

    n_wins_leaky = int(df_leaky["beats_leg_a_both_axes"].sum())
    n_wins_shift = int(df_shift["beats_leg_a_both_axes"].sum())
    print(f"\n=== SUMMARY: leaky wins {n_wins_leaky}/{len(df_leaky)} windows; "
          f"causal-shifted wins {n_wins_shift}/{len(df_shift)} windows ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
