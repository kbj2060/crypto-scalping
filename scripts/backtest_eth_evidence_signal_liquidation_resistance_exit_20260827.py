#!/usr/bin/env python3
"""New angle on liquidation-map resistance levels: NOT "does resistance improve a TOP/short entry
signal's reliability" (already tested today, no pattern -- research_eth_evidence_signal_liquidation_
confluence_20260827.py Part A) and NOT "does the level itself act as a standalone price barrier"
(tested ~15 ways 2026-08-24/25, all REJECTED/unstable -- eth_liquidation_map_sr_research_2026_rollup
memory). Instead: GIVEN an already-open chop-fade LONG position (entered via a bottom evidence
signal, i.e. exactly the orthogonal_combo:bottom:chop(+confluence) setup that is this whole research
line's one positive result), does taking profit early when price first gets close to a resistance
level above entry beat waiting for the fixed ATR-based TP?

This is a narrower claim than the 15 already-rejected ones: it does not require the resistance level
to have general unconditional predictive power over price (already shown false); it only requires
that, CONDITIONAL on being in one of these specific mean-reversion trades, exiting near resistance
is better than the fixed-TP alternative FOR THOSE SAME TRADES. Flagged upfront: given how uniformly
the 15 broader liquidation-level-reaction tests failed, the prior for this succeeding is low -- this
is tested because it is a genuinely distinct, not-yet-tried framing, not because it is expected to
work; reported honestly either way.

Design: vendors a modified _resolve_trade/simulate_single_position (does not touch core.causal_
futures_backtest.py, same pattern as backtest_eth_evidence_signal_regime_entry_exit_20260827.py's
_resolve_trade_regime_stop) that adds a THIRD exit check, evaluated only after SL/TP both miss for
the bar: if the position is already profitable (close-implied unrealized > 0) AND price is currently
in the near/mid distance tertile of the nearest resistance level above it (same tertile methodology,
same compute_spliced_levels() source, as the already-validated support-side confluence work), exit
now at that bar's close instead of continuing to hold for the fixed TP/timeout. LONG-only (every
candidate here is a bottom/long signal).

Held fixed (isolates the exit rule's effect cleanly): the exact GBM3 chop-gate + support-confluence
entry setup from backtest_eth_evidence_signal_liquidation_confluence_costgate_20260827.py (that
script's report.json is the baseline for both "chop" and "chop_confluence" variants) -- same 6
windows, same TP:SL/leverage/cost, same regime source (GBM2 already found worse for this gate,
see eth_evidence_signal_liquidation_confluence_gbm2gate_rejected_20260827 memory -- no reason to
introduce a second confound here). Primary candidate: orthogonal_combo (the only one with a real
positive result to potentially improve or damage); taker_delta_z_climax included as a robustness
check on a structurally different (non-chop-favoring, see confluence memory) signal.
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

import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_evidence_signal_chop_gated_costgate_20260827 import (  # noqa: E402
    HORIZON_BARS, LEVERAGE, MARGIN_FRACTION, ROUNDTRIP_COST_RATE, SL_ATR_MULT, TP_ATR_MULT,
    _compute_frame,
)
from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from research_eth_evidence_signal_liquidation_confluence_20260827 import (  # noqa: E402
    compute_hourly_levels, resample_1h,
)

OUT_DIR = ROOT / "tmp/eth_evidence_signal_liquidation_resistance_exit_20260827"
CANDIDATES = [("orthogonal_combo", "bottom"), ("taker_delta_z_climax", "bottom")]
GATE_VARIANTS = ("chop", "chop_confluence")


def add_confluence_both_sides(frame: pd.DataFrame) -> pd.DataFrame:
    """Same tertile methodology as backtest_eth_evidence_signal_liquidation_confluence_costgate_
    20260827.py::add_confluence, extended to ALSO tertile the resistance side (that script only
    needed support, for the entry gate; this script needs resistance too, for the new exit rule)."""
    hourly = resample_1h(frame)
    levels_hourly = compute_hourly_levels(hourly)
    merged = pd.merge_asof(frame.sort_values("timestamp").reset_index(drop=True), levels_hourly, on="timestamp", direction="backward")
    sup_dist = merged["support_distance_pct"].abs()
    _, sup_edges = pd.qcut(sup_dist.dropna(), 3, retbins=True, duplicates="drop")
    merged["near_or_mid_support"] = pd.cut(sup_dist, bins=sup_edges, labels=["near", "mid", "far"][: len(sup_edges) - 1], include_lowest=True).isin(["near", "mid"]).to_numpy()
    res_dist = merged["resistance_distance_pct"].abs()
    _, res_edges = pd.qcut(res_dist.dropna(), 3, retbins=True, duplicates="drop")
    res_tertile = pd.cut(res_dist, bins=res_edges, labels=["near", "mid", "far"][: len(res_edges) - 1], include_lowest=True)
    merged["near_or_mid_resistance"] = res_tertile.isin(["near", "mid"]).fillna(False).to_numpy()
    merged["near_only_resistance"] = (res_tertile == "near").fillna(False).to_numpy()
    return merged


def _resolve_trade_resistance_exit(
    *, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    tp_move: float, sl_move: float, near_or_mid_resistance: np.ndarray,
) -> tuple[float, str, int]:
    """LONG-only. Same SL(intrabar)->TP(intrabar)->timeout(close) ladder as core.causal_futures_
    backtest._resolve_trade, plus a resistance_exit check (close-based, after SL/TP miss for the
    bar): if already profitable (close-implied unrealized > 0) AND currently in the near/mid
    distance tertile of the nearest resistance level above price, exit now instead of waiting for
    the fixed TP/timeout."""
    tp_level, sl_level = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
    for offset, (bar_high, bar_low, bar_close) in enumerate(zip(high, low, close)):
        if bar_low <= sl_level:
            return -sl_move, "sl", offset
        if bar_high >= tp_level:
            return tp_move, "tp", offset
        unrealized = bar_close / entry - 1.0
        if near_or_mid_resistance[offset] and unrealized > 0.0:
            return unrealized, "resistance_exit", offset
    return float(close[-1] / entry - 1.0), "timeout", len(close) - 1


def simulate_long_only_resistance_exit(
    *, timestamps, open_px, high, low, close, near_or_mid_resistance, decision_indices,
    tp_moves, sl_moves, horizon_bars, margin_fraction, leverage, roundtrip_cost_rate,
    use_resistance_exit: bool,
):
    ts = pd.DatetimeIndex(timestamps)
    open_values, high_values, low_values, close_values = (np.asarray(a, dtype=np.float64) for a in (open_px, high, low, close))
    res_values = np.asarray(near_or_mid_resistance, dtype=bool)
    idxs = np.asarray(decision_indices, dtype=np.int64)
    tp_values, sl_values = (np.asarray(a, dtype=np.float64) for a in (tp_moves, sl_moves))
    notional = float(margin_fraction * leverage)
    account_cost = float(roundtrip_cost_rate * notional)
    equity = np.ones(len(ts), dtype=np.float64)
    cash = 1.0
    filled_through = -1
    occupied_through = -1
    rows: list[dict] = []

    for decision_i, tp_move, sl_move in zip(idxs, tp_values, sl_values):
        if not np.isfinite(tp_move) or not np.isfinite(sl_move):
            continue
        entry_i = int(decision_i) + 1
        if entry_i >= len(ts) or entry_i <= occupied_through:
            continue
        final_i = min(entry_i + horizon_bars - 1, len(ts) - 1)
        if final_i < entry_i:
            continue
        if filled_through + 1 < entry_i:
            equity[filled_through + 1: entry_i] = cash

        entry = float(open_values[entry_i])
        window_res = res_values[entry_i: final_i + 1] if use_resistance_exit else np.zeros(final_i - entry_i + 1, dtype=bool)
        price_move, reason, exit_offset = _resolve_trade_resistance_exit(
            entry=entry, high=high_values[entry_i: final_i + 1], low=low_values[entry_i: final_i + 1],
            close=close_values[entry_i: final_i + 1], tp_move=float(tp_move), sl_move=float(sl_move),
            near_or_mid_resistance=window_res,
        )
        exit_i = entry_i + exit_offset
        for bar_i in range(entry_i, exit_i + 1):
            unrealized = close_values[bar_i] / entry - 1.0
            equity[bar_i] = cash * (1.0 + unrealized * notional - account_cost)
        trade_return = float(price_move * notional - account_cost)
        cash *= 1.0 + trade_return
        equity[exit_i] = cash
        filled_through = exit_i
        occupied_through = exit_i
        rows.append({"entry_timestamp": ts[entry_i], "exit_timestamp": ts[exit_i], "reason": reason,
                      "bars_held": int(exit_offset + 1), "price_move": float(price_move), "trade_return": trade_return})

    if filled_through + 1 < len(equity):
        equity[filled_through + 1:] = cash
    return equity, pd.DataFrame(rows)


def run_window(frame: pd.DataFrame, bcol: str, gate_variant: str, use_resistance_exit: bool, *, start, end,
               resistance_col: str = "near_or_mid_resistance") -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    bottom = frame[bcol].fillna(False).to_numpy()
    bottom = bottom & (frame["regime_label"] == "chop").to_numpy()
    if gate_variant == "chop_confluence":
        bottom = bottom & frame["near_or_mid_support"].fillna(False).to_numpy()

    has_score = frame["atr_pct"].notna().to_numpy()
    decision_indices = np.flatnonzero(eligible & has_score & bottom)
    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]

    equity, ledger = simulate_long_only_resistance_exit(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        near_or_mid_resistance=frame[resistance_col].to_numpy(),
        decision_indices=decision_indices, tp_moves=tp_moves, sl_moves=sl_moves,
        horizon_bars=HORIZON_BARS, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
        roundtrip_cost_rate=ROUNDTRIP_COST_RATE, use_resistance_exit=use_resistance_exit,
    )
    n_trades = int(len(ledger))
    total_return = float(equity[-1] - 1.0) if len(equity) else float("nan")
    reason_counts = ledger["reason"].value_counts().to_dict() if n_trades else {}
    return {"n_trades": n_trades, "total_return": total_return, "reason_counts": reason_counts}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"results": {}}

    print("Building 2025/2026 frames (GBM3 chop gate + support+resistance confluence)...")
    frames = {}
    for yr, base_csv in (("2025", gate.sweep.BASE_2025), ("2026", gate.sweep.BASE_2026)):
        fr = _compute_frame(base_csv)
        fr = add_confluence_both_sides(fr)
        frames[yr] = fr
        print(f"  {yr}: {len(fr)} rows, near_or_mid_resistance={fr['near_or_mid_resistance'].mean() * 100:.1f}%")

    for name, _side in CANDIDATES:
        bcol = f"bottom_{name}"
        for gate_variant in GATE_VARIANTS:
            print(f"\n=== {name}:{gate_variant} -- baseline (fixed TP) vs resistance_exit ===")
            print(f"{'window':<8} {'n(base)':>8} {'ret(base)':>10} {'n(rexit)':>9} {'ret(rexit)':>11} {'delta':>8}  reason_counts(rexit)")
            windows_out = {}
            sum_base, sum_rexit = 0.0, 0.0
            for wname, wd in gate.WINDOW_DEFS.items():
                frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                base = run_window(frame, bcol, gate_variant, use_resistance_exit=False, start=wd["start"], end=wd["end"])
                rexit = run_window(frame, bcol, gate_variant, use_resistance_exit=True, start=wd["start"], end=wd["end"])
                delta = rexit["total_return"] - base["total_return"]
                sum_base += base["total_return"]
                sum_rexit += rexit["total_return"]
                windows_out[wname] = {"baseline": base, "resistance_exit": rexit, "delta": delta}
                print(f"{wname:<8} {base['n_trades']:>8d} {base['total_return'] * 100:>9.2f}% "
                      f"{rexit['n_trades']:>9d} {rexit['total_return'] * 100:>10.2f}% {delta * 100:>+7.2f}%  {rexit['reason_counts']}")
            report["results"][f"{name}:{gate_variant}"] = windows_out
            print(f"SUM: baseline={sum_base * 100:.2f}%  resistance_exit={sum_rexit * 100:.2f}%  delta={(sum_rexit - sum_base) * 100:+.2f}%p")

    print("\n=== orthogonal_combo:chop_confluence -- stricter trigger (near tertile ONLY, not near_or_mid) ===")
    print(f"{'window':<8} {'n(base)':>8} {'ret(base)':>10} {'n(rexit)':>9} {'ret(rexit)':>11} {'delta':>8}  reason_counts(rexit)")
    sum_base, sum_rexit = 0.0, 0.0
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
        base = run_window(frame, "bottom_orthogonal_combo", "chop_confluence", use_resistance_exit=False, start=wd["start"], end=wd["end"])
        rexit = run_window(frame, "bottom_orthogonal_combo", "chop_confluence", use_resistance_exit=True, start=wd["start"], end=wd["end"], resistance_col="near_only_resistance")
        delta = rexit["total_return"] - base["total_return"]
        sum_base += base["total_return"]
        sum_rexit += rexit["total_return"]
        print(f"{wname:<8} {base['n_trades']:>8d} {base['total_return'] * 100:>9.2f}% "
              f"{rexit['n_trades']:>9d} {rexit['total_return'] * 100:>10.2f}% {delta * 100:>+7.2f}%  {rexit['reason_counts']}")
    print(f"SUM: baseline={sum_base * 100:.2f}%  resistance_exit(near-only)={sum_rexit * 100:.2f}%  delta={(sum_rexit - sum_base) * 100:+.2f}%p")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
