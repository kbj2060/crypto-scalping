#!/usr/bin/env python3
"""2026-08-27 ad-hoc follow-up (NOT a saved research script) to backtest_eth_evidence_signal_
chop_gated_costgate_20260827.py: user hypothesis after seeing short_term_return_z's high-win-rate/
negative-return pattern -- "exit on the OPPOSITE signal firing" instead of the fixed TP=1.6xATR/
SL=1.0xATR/48bar-timeout engine. Reuses that script's exact data prep (_compute_frame, CANDIDATES,
same 6 WINDOW_DEFS, same notional/cost/leverage) via direct import -- only the exit resolution
changes, so results are directly comparable to the existing chop_gated report.

Exit rule: once in a position, exit the first bar after the OPPOSITE-side raw signal (same name,
other side, e.g. bottom entry -> top_{name} exit) fires -- filled at that next bar's OPEN, mirroring
how entries themselves fill one bar after their own trigger (causal, no lookahead). NOT chop-gated
on the exit side (a real trend starting against the position is exactly when you want out, chop or
not). Falls back to the same HORIZON_BARS=48 timeout (exit at last bar's close) if no opposite
signal appears in time. No price-level stop-loss at all -- reports max single-trade loss so this
gap is visible, not hidden.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, BacktestResult  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_evidence_signal_chop_gated_costgate_20260827 import (  # noqa: E402
    _compute_frame, CANDIDATES, HORIZON_BARS, LEVERAGE, MARGIN_FRACTION, ROUNDTRIP_COST_RATE,
)

OUT_DIR = ROOT / "tmp/eth_evidence_signal_chop_gated_opposite_signal_exit_20260827"


def _resolve_trade_opposite(*, side: int, entry: float, opens: np.ndarray, closes: np.ndarray,
                             opp_fired: np.ndarray) -> tuple[float, str, int]:
    n = len(closes)
    for k in range(n - 1):  # need k+1 in range to act on bar k's confirmed signal
        if opp_fired[k]:
            exit_price = float(opens[k + 1])
            if side > 0:
                return exit_price / entry - 1.0, "opposite_signal", k + 1
            return 1.0 - exit_price / entry, "opposite_signal", k + 1
    if side > 0:
        return float(closes[-1] / entry - 1.0), "timeout", n - 1
    return float(1.0 - closes[-1] / entry), "timeout", n - 1


def simulate_opposite_signal_exit(*, timestamps, open_px, close, decision_indices, sides,
                                   opp_fired_full, horizon_bars, margin_fraction, leverage,
                                   roundtrip_cost_rate) -> BacktestResult:
    ts = pd.DatetimeIndex(timestamps)
    open_px = np.asarray(open_px, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    opp_fired_full = np.asarray(opp_fired_full, dtype=bool)
    notional = float(margin_fraction * leverage)
    account_cost = float(roundtrip_cost_rate * notional)
    equity = np.ones(len(ts), dtype=np.float64)
    cash = 1.0
    filled_through = -1
    occupied_through = -1
    skipped_while_open = 0
    rows: list[dict] = []

    for decision_i, side in zip(np.asarray(decision_indices, dtype=np.int64), sides):
        if side == 0:
            continue
        entry_i = int(decision_i) + 1
        if entry_i >= len(ts):
            continue
        if entry_i <= occupied_through:
            skipped_while_open += 1
            continue
        final_i = min(entry_i + horizon_bars - 1, len(ts) - 1)
        if final_i < entry_i:
            continue
        if filled_through + 1 < entry_i:
            equity[filled_through + 1: entry_i] = cash

        entry = float(open_px[entry_i])
        price_move, reason, exit_offset = _resolve_trade_opposite(
            side=int(side), entry=entry,
            opens=open_px[entry_i: final_i + 1], closes=close[entry_i: final_i + 1],
            opp_fired=opp_fired_full[entry_i: final_i + 1],
        )
        exit_i = entry_i + exit_offset
        for bar_i in range(entry_i, exit_i + 1):
            unrealized = (close[bar_i] / entry - 1.0) if side > 0 else (1.0 - close[bar_i] / entry)
            equity[bar_i] = cash * (1.0 + unrealized * notional - account_cost)
        trade_return = float(price_move * notional - account_cost)
        cash *= 1.0 + trade_return
        equity[exit_i] = cash
        filled_through = exit_i
        occupied_through = exit_i
        rows.append({
            "decision_timestamp": ts[int(decision_i)], "entry_timestamp": ts[entry_i],
            "exit_timestamp": ts[exit_i], "side": int(side), "reason": reason,
            "bars_held": int(exit_offset + 1), "price_move": float(price_move),
            "trade_return": trade_return,
        })

    if filled_through + 1 < len(equity):
        equity[filled_through + 1:] = cash
    return BacktestResult(equity=equity, ledger=pd.DataFrame(rows), skipped_while_open=skipped_while_open)


def run_window(frame: pd.DataFrame, name: str, side_name: str, chop_gate: bool, *, start, end) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    entry_col = f"bottom_{name}" if side_name == "bottom" else f"top_{name}"
    opp_col = f"top_{name}" if side_name == "bottom" else f"bottom_{name}"
    entry_fired = frame[entry_col].fillna(False).to_numpy()
    if chop_gate:
        chop = (frame["regime_label"] == "chop").to_numpy()
        entry_fired = entry_fired & chop
    opp_fired_full = frame[opp_col].fillna(False).to_numpy()

    mask = eligible & entry_fired
    decision_indices = np.flatnonzero(mask)
    side_const = 1 if side_name == "bottom" else -1
    sides = np.full(len(decision_indices), side_const, dtype=np.int64)

    result = simulate_opposite_signal_exit(
        timestamps=ts, open_px=frame["open"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, sides=sides, opp_fired_full=opp_fired_full,
        horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
    )
    ledger = result.ledger
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    n_trades = int(len(ledger))
    wr = float((ledger["price_move"] * ledger["side"] > 0).mean()) if n_trades else float("nan")
    reason_counts = ledger["reason"].value_counts().to_dict() if n_trades else {}
    avg_bars_held = float(ledger["bars_held"].mean()) if n_trades else float("nan")
    worst_trade_return = float(ledger["trade_return"].min()) if n_trades else float("nan")

    win_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(frame["close"].iloc[win_idx[0]]), float(frame["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "wr": wr, "total_return": total_return,
        "reason_counts": reason_counts, "avg_bars_held": avg_bars_held,
        "worst_trade_return": worst_trade_return,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 2025/2026 frames (reusing chop_gated_costgate script's exact data prep)...")
    frames = {"2025": _compute_frame(gate.sweep.BASE_2025), "2026": _compute_frame(gate.sweep.BASE_2026)}

    report: dict[str, Any] = {"config": {"exit_rule": "opposite_signal_or_48bar_timeout",
                                          "horizon_bars": HORIZON_BARS, "leverage": LEVERAGE,
                                          "margin_fraction": MARGIN_FRACTION,
                                          "roundtrip_cost_rate": ROUNDTRIP_COST_RATE},
                               "results": {}}

    for name, side in CANDIDATES:
        for chop_gate in (False, True):
            variant = "chop_gated" if chop_gate else "ungated"
            key = f"{name}:{side}:{variant}"
            print(f"\n--- {key} ---")
            print(f"{'window':<8} {'n':>5} {'wr':>7} {'return':>9} {'a_long':>8} {'a_short':>8} "
                  f"{'beats_bm':>9} {'avg_hold':>9} {'worst_trade':>12}  reasons")
            windows_out = {}
            for wname, wd in gate.WINDOW_DEFS.items():
                frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                res = run_window(frame, name, side, chop_gate, start=wd["start"], end=wd["end"])
                windows_out[wname] = res
                wr_str = f"{res['wr']*100:.1f}%" if np.isfinite(res["wr"]) else "n/a"
                print(f"{wname:<8} {res['n_trades']:>5d} {wr_str:>7} {res['total_return']*100:>8.2f}% "
                      f"{res['always_long_return']*100:>7.2f}% {res['always_short_return']*100:>7.2f}% "
                      f"{str(res['beats_benchmark']):>9} {res['avg_bars_held']:>8.1f}b "
                      f"{res['worst_trade_return']*100:>10.2f}%  {res['reason_counts']}")
            report["results"][key] = windows_out
            wins = sum(1 for w in windows_out.values() if w["beats_benchmark"])
            print(f"SUMMARY {key}: beats always_long/always_short in {wins}/{len(windows_out)} windows")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
