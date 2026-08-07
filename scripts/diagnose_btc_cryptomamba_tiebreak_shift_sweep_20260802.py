#!/usr/bin/env python3
"""Follow-up to diagnose_btc_cryptomamba_tiebreak_timestamp_shift_20260802.py.

The first +1h-shift test showed a large collapse (5/6 -> 1/6 window wins, ~20-33pp PnL drop, one
sign flip). But unlike the BTC dumb-momentum case -- where the unshifted baseline was traced to be
GENUINELY self-referential (regime signal and gated delta both read the same resample("1h").last()
bar) -- CryptoMamba's own regime CSV timestamp was traced to be a native 5m-bar, causal-only
timestamp (see train_regime3_cryptomamba_pred_20260531.py::_output(), pass-through of the raw
feature row's own timestamp; inference window is x[t-59:t+1], nothing later). A full +1h shift is
a large artificial delay relative to CryptoMamba's own 30-min (h6) forecast horizon, so a collapse
under that shift is also exactly what a genuinely-causal-but-short-horizon signal would do -- it
does not by itself distinguish "was already leaky" from "is just a fast-decaying real signal".

This script disambiguates with a shift SWEEP (not just one +1h point) on the 2 VAL/OOS windows only
(for speed): negative shifts (-60, -30min) give the regime signal MORE future information than the
traced-causal baseline (a deliberately non-causal probe -- if this monotonically improves results,
that is the fingerprint of a leak-sensitive result, matching how the confirmed BTC dumb-momentum bug
behaved). Small positive shifts (+5, +15min) test whether performance is fragile even at a tiny,
near-noise-level delay (suggesting a knife-edge/leak artifact) vs. degrading gradually (suggesting a
genuine but short-lived forecasting edge). Includes 0 (baseline) and +60min (already run) for
reference.
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
    load_cryptomamba_regime, run_window, VAL_OOS_WINDOWS,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import load_5m_prices_btc  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260802/btc_cryptomamba_lookahead_check"
SHIFTS_MIN = [-60, -30, -15, 0, 5, 15, 30, 60, 120]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime_base = load_cryptomamba_regime()

    rows = []
    for shift_min in SHIFTS_MIN:
        regime = regime_base.copy()
        regime["timestamp"] = regime["timestamp"] + pd.Timedelta(minutes=shift_min)
        print(f"\n######## shift = {shift_min:+d} min ########")
        for label, s, e in VAL_OOS_WINDOWS:
            r = run_window(f"{label}_shift{shift_min:+d}m", s, e, prices, regime)
            r["shift_min"] = shift_min
            r["base_window"] = label
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "shift_sweep_val_oos.csv", index=False)
    pivot = df.pivot(index="shift_min", columns="base_window", values="cmamba_tiebreak_pnl")
    print("\n########## SWEEP: PnL% by shift (rows) x window (cols) ##########")
    print(pivot.to_string())
    pivot.to_csv(OUT_DIR / "shift_sweep_pivot_pnl.csv")

    pivot_mdd = df.pivot(index="shift_min", columns="base_window", values="cmamba_tiebreak_mdd")
    print("\n########## SWEEP: MDD% by shift (rows) x window (cols) ##########")
    print(pivot_mdd.to_string())
    pivot_mdd.to_csv(OUT_DIR / "shift_sweep_pivot_mdd.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
