#!/usr/bin/env python3
"""Cost-gate backtest for funding_oscillator_combo (ETH oscillator oversold AND funding_z<=-2 /
ETH oscillator overbought AND funding_z>=2), the 9th evidence signal added to the live dashboard
2026-08-25 (eth_funding_oscillator_combo_candidate_20260825 memory). That work measured
retrospective lift-to-a-future-pivot only ("NOT a trading algorithm") and beat orthogonal_combo's
own lift at 1h in two independent windows -- this script is the first real causal fresh-forward
backtest with entry/TP/SL/cost, following the SAME established convention as the sibling
Dalton-rule-2 cost-gate script (backtest_eth_dalton_rule2_balance_edge_costgate_20260815.py)
rather than inventing new mechanics: engine = core.causal_futures_backtest.simulate_single_
position/purged_decision_mask, constants (TP=1.6xATR, SL=1.0xATR, horizon=48bar, leverage=3x,
margin=30%, roundtrip cost=0.1%) copied VERBATIM from backtest_eth_slowk_williamsr_persistence_
confluence_20260814.py, NOT tuned for this signal. Same 6 pre-registered windows as the Dalton/
top-6 scripts (2025q1/q2/q3 context + val + oos_q1/oos_q2), via
eth_omega461_multiwindow_confirmation_gate_20260814.WINDOW_DEFS.

Signal definition reused verbatim (not redefined) from scripts/live_evidence_signal_dashboard_
20260823.py's funding_oscillator_combo columns: p_fast/p_slow from backtest_eth_slowk_williamsr_
persistence_confluence_20260814.compute_indicators, funding_z from research_eth_funding_
crossasset_combo_signal_20260825.load_funding_z (rolling(90,min30) z-score of
data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv, merge_asof backward -- causal, no lookahead).
score = +1 (LONG) if bottom fires, -1 (SHORT) if top fires, 0 otherwise -- top is already known
to almost never fire in this data (funding rate ceiling at 0.0001, see the candidate memory), so
this is expected to reduce to an effectively long-only test; that expectation is not assumed here,
it's left to the real formula to decide per-window.

Cost gate: reports the strategy at the standard 0.1% roundtrip cost, AND sweeps roundtrip cost
0bp..50bp per pre-registered window to find the breakeven cost (return crosses zero) -- same
"손익분기 비용을 bp로 명시" requirement as every other candidate in this repo's promotion gate.

fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. No training, no GPU.
Does not modify any imported module or live file.
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

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_funding_oscillator_combo_costgate_20260825"

# Copied verbatim from backtest_eth_slowk_williamsr_persistence_confluence_20260814.py -- not tuned here.
TP_ATR_MULT = 1.6
SL_ATR_MULT = 1.0
HORIZON_BARS = 48
LEVERAGE = 3.0
MARGIN_FRACTION = 0.30
ROUNDTRIP_COST_RATE = 0.001
ATR_N = 14
COST_SWEEP_BP = [0, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50]


def log(msg: str) -> None:
    print(f"[funding_osc_combo_costgate] {msg}", flush=True)


def _compute_signal_frame(base_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(base_csv, low_memory=False, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    ind = compute_indicators(raw)  # -> p_fast, p_slow (verbatim, same as orthogonal_combo's leg)
    funding = load_funding_z()
    ind = pd.merge_asof(ind.sort_values("timestamp"), funding, left_on="timestamp", right_on="calc_time", direction="backward")

    bottom = (ind["p_fast"] <= 0.10) & (ind["p_slow"] <= 0.10) & (ind["funding_z"] <= -2.0)
    top = (ind["p_fast"] >= 0.90) & (ind["p_slow"] >= 0.90) & (ind["funding_z"] >= 2.0)
    score = bottom.fillna(False).astype(np.float64) - top.fillna(False).astype(np.float64)
    atr_pct = pd.Series(_atr_pct(raw, ATR_N), index=raw.index)

    return pd.DataFrame({
        "timestamp": raw["timestamp"], "open": raw["open"], "high": raw["high"], "low": raw["low"], "close": raw["close"],
        "score": score, "atr_pct": atr_pct,
    })


def run_window(frame: pd.DataFrame, *, start, end, roundtrip_cost: float) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    has_score = frame["score"].notna().to_numpy() & frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, scores=frame["score"].to_numpy()[decision_indices],
        tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=roundtrip_cost,
    )
    ledger = result.ledger
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    n_trades = int(len(ledger))
    n_long = int((ledger["side"] > 0).sum()) if n_trades else 0
    n_short = int((ledger["side"] < 0).sum()) if n_trades else 0
    wr = float((ledger["price_move"] * ledger["side"] > 0).mean()) if n_trades else float("nan")

    win_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(frame["close"].iloc[win_idx[0]]), float(frame["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "n_long": n_long, "n_short": n_short, "wr": wr, "total_return": total_return,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def find_breakeven_bp(frame: pd.DataFrame, *, start, end) -> float | None:
    """Bisection on roundtrip cost rate for the cost at which total_return crosses zero."""
    lo, hi = 0.0, 0.02  # 0bp .. 200bp
    r_lo = run_window(frame, start=start, end=end, roundtrip_cost=lo)["total_return"]
    r_hi = run_window(frame, start=start, end=end, roundtrip_cost=hi)["total_return"]
    if not np.isfinite(r_lo) or r_lo <= 0:
        return 0.0  # already unprofitable at zero cost
    if r_hi > 0:
        return None  # still profitable at 200bp roundtrip -- report as "> 200bp"
    for _ in range(40):
        mid = (lo + hi) / 2.0
        r_mid = run_window(frame, start=start, end=end, roundtrip_cost=mid)["total_return"]
        if r_mid > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0 * 10000.0)  # -> bp


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {"2025": _compute_signal_frame(gate.sweep.BASE_2025), "2026": _compute_signal_frame(gate.sweep.BASE_2026)}

    report: dict[str, Any] = {
        "signal": "funding_oscillator_combo (p_fast/p_slow<=.10 or >=.90 AND funding_z beyond +-2)",
        "tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT, "horizon_bars": HORIZON_BARS,
        "leverage": LEVERAGE, "margin_fraction": MARGIN_FRACTION, "roundtrip_cost_rate_standard": ROUNDTRIP_COST_RATE,
        "windows": {},
    }

    log(f"{'window':<8} {'n_trades':>8} {'long':>5} {'short':>6} {'wr':>7} {'return':>10} {'a_long':>9} {'a_short':>9} {'beats_bm':>9}  breakeven_bp")
    for wname, wd in gate.WINDOW_DEFS.items():
        frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
        res_std = run_window(frame, start=wd["start"], end=wd["end"], roundtrip_cost=ROUNDTRIP_COST_RATE)
        be = find_breakeven_bp(frame, start=wd["start"], end=wd["end"])
        be_str = f"{be:.1f}bp" if be is not None else ">200bp"

        cost_curve = {}
        for bp in COST_SWEEP_BP:
            r = run_window(frame, start=wd["start"], end=wd["end"], roundtrip_cost=bp / 10000.0)
            cost_curve[bp] = r["total_return"]

        report["windows"][wname] = {**res_std, "breakeven_bp": be, "cost_curve_bp_to_return": cost_curve}
        log(f"{wname:<8} {res_std['n_trades']:>8d} {res_std['n_long']:>5d} {res_std['n_short']:>6d} "
            f"{res_std['wr']*100 if np.isfinite(res_std['wr']) else float('nan'):>6.1f}% "
            f"{res_std['total_return']*100:>9.2f}% {res_std['always_long_return']*100:>8.2f}% "
            f"{res_std['always_short_return']*100:>8.2f}%  {str(res_std['beats_benchmark']):>9}  {be_str}")

    wins = sum(1 for w in report["windows"].values() if w["beats_benchmark"])
    total = len(report["windows"])
    log(f"SUMMARY: beats always_long/always_short in {wins}/{total} windows at standard 10bp roundtrip cost")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
