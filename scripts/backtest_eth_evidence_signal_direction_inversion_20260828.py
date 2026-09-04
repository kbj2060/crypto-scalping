#!/usr/bin/env python3
"""User question 2026-08-28: do evidence signals perform BETTER if traded INVERTED (short on a
bottom-signal fire, long on a top-signal fire) than as-designed (long on bottom, short on top)?

Motivated by today's own findings: several signals' as-designed win rate is well below this TP:SL
ratio's breakeven (~38.5%, TP=1.6x:SL=1.0x ATR) when run ungated (e.g. taker_delta_z_climax 33-39%),
and the asymmetric TP:SL (TP set FAR at 1.6x ATR, SL CLOSE at 1.0x ATR) means a low win rate is
usually a losing combination -- naively suggesting the opposite side might do better. This is NOT
guaranteed by simple win/loss inversion, though: flipping the position direction with the SAME
tp_move/sl_move magnitudes changes WHICH price levels count as TP vs SL (the original LONG's near
SL level does not become the flipped SHORT's near level -- the flipped SHORT's SL is instead set on
the OPPOSITE side, same 1.0x-ATR distance, and its TP is the opposite 1.6x-ATR-away level) -- so this
needs an actual backtest, not armchair win-rate arithmetic.

Reuses the exact live engine unchanged (backtest_eth_evidence_signal_chop_gated_costgate_20260827.py
-- _compute_frame/run_window/find_breakeven_bp, same TP:SL/leverage/cost/6-window convention) --
inversion is done purely by swapping which column (bottom_<name> vs top_<name>) is passed as bcol
vs tcol to the SAME run_window() call, no new simulation logic. Tests all 5 signals BOTH-SIDED
(bottom AND top simultaneously, unlike the base script's own CANDIDATES which test each signal
bottom-only OR top-only) since the user's question is specifically about the combined bottom+top
strategy's direction, not a single side.

Ungated (chop_gate=False) only -- matches the user's general question; not re-testing the
chop-gated variant here (today's chop-gate results are the well-established comparison baseline
already documented elsewhere if that's wanted next).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_evidence_signal_chop_gated_costgate_20260827 import (  # noqa: E402
    ROUNDTRIP_COST_RATE, _compute_frame, find_breakeven_bp, run_window,
)

SIGNALS = ["orthogonal_combo", "liquidity_sweep", "smt_divergence", "volume_wick_climax",
           "short_term_return_z", "taker_delta_z_climax"]


def main() -> int:
    print("Building 2025/2026 frames...")
    frames = {"2025": _compute_frame(gate.sweep.BASE_2025), "2026": _compute_frame(gate.sweep.BASE_2026)}

    summary = []
    for name in SIGNALS:
        bcol, tcol = f"bottom_{name}", f"top_{name}"
        print(f"\n=== {name}: normal(bottom=long,top=short) vs inverted(bottom=short,top=long) ===")
        print(f"{'window':<8} {'n(norm)':>8} {'wr(norm)':>9} {'ret(norm)':>10} {'n(inv)':>7} {'wr(inv)':>8} {'ret(inv)':>9} {'a_long':>8} {'a_short':>8}")
        sum_norm, sum_inv = 0.0, 0.0
        for wname, wd in gate.WINDOW_DEFS.items():
            frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
            norm = run_window(frame, bcol, tcol, False, start=wd["start"], end=wd["end"], roundtrip_cost=ROUNDTRIP_COST_RATE)
            inv = run_window(frame, tcol, bcol, False, start=wd["start"], end=wd["end"], roundtrip_cost=ROUNDTRIP_COST_RATE)
            sum_norm += norm["total_return"]
            sum_inv += inv["total_return"]
            print(f"{wname:<8} {norm['n_trades']:>8d} {norm['wr']*100 if np.isfinite(norm['wr']) else float('nan'):>8.1f}% "
                  f"{norm['total_return']*100:>9.2f}% {inv['n_trades']:>7d} "
                  f"{inv['wr']*100 if np.isfinite(inv['wr']) else float('nan'):>7.1f}% {inv['total_return']*100:>8.2f}% "
                  f"{norm['always_long_return']*100:>7.2f}% {norm['always_short_return']*100:>7.2f}%")
        print(f"SUM: normal={sum_norm*100:.2f}%  inverted={sum_inv*100:.2f}%  delta(inv-norm)={(sum_inv-sum_norm)*100:+.2f}%p")
        summary.append({"signal": name, "sum_normal_pct": sum_norm * 100, "sum_inverted_pct": sum_inv * 100,
                         "delta_pct": (sum_inv - sum_norm) * 100})

    print("\n=== Cross-signal summary (sum of total_return across 6 windows) ===")
    print(f"{'signal':<24} {'normal':>10} {'inverted':>10} {'delta':>10}")
    for row in summary:
        print(f"{row['signal']:<24} {row['sum_normal_pct']:>9.2f}% {row['sum_inverted_pct']:>9.2f}% {row['delta_pct']:>+9.2f}%p")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
