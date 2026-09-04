#!/usr/bin/env python3
"""Does gating a bottom-side evidence signal to fire ONLY when (regime=chop AND price is in the
near/mid distance tertile from a support level on the live liquidation map) improve on the already-
tested chop-only gate, which today failed the cost-gate backtest 10/10 (backtest_eth_evidence_
signal_chop_gated_costgate_20260827.py)?

Extended 2026-08-27 (same day, user follow-up) from a single-candidate run (orthogonal_combo:bottom
only) to all 5 bottom-side signals that showed the same near>far diagnostic pattern in
research_eth_evidence_signal_liquidation_confluence_20260827.py's Part A: orthogonal_combo,
taker_delta_z_climax, liquidity_sweep, short_term_return_z, smt_divergence. (dalton_rule2_balance_
edge and volume_wick_climax showed a weaker/less monotonic pattern in Part A and are excluded --
fib_extension_exhaustion's chop sample there was already thin, n=17-68 per tertile, excluded too.)
Each candidate now gets all THREE variants computed fresh in this script (ungated / chop-only /
chop+confluence), not spliced in from the earlier chop-gate script's run -- three of these five
candidates (taker_delta_z_climax, short_term_return_z on bottom, and re-running orthogonal_combo/
liquidity_sweep/smt_divergence here) either weren't in that script's candidate list at all or were
tested on the opposite side there (short_term_return_z was TOP there; this repo's own Part A found
the confluence pattern on its BOTTOM side), so a clean apples-to-apples 3-way comparison needs all
three variants from one consistent run rather than stitching two scripts' outputs together.

Engine (TP:SL/6-window/cost convention, _compute_frame) imported unchanged from
backtest_eth_evidence_signal_chop_gated_costgate_20260827.py. Liquidation-level computation
(compute_spliced_levels, causal 24-hourly-bar lookback) imported unchanged from the diagnostic
script.

Expectation stated in advance, not after seeing results (same as the first orthogonal_combo-only
run): shrinking an already-losing-by-benchmark signal to a smaller, higher-quality subset makes
trade count go DOWN, which structurally makes it HARDER, not easier, to out-accumulate a strong-
trend always_long/always_short benchmark over a fixed window -- so chop_confluence is expected to
likely still fail against the benchmark for the same structural reason already diagnosed today
(confirmed for orthogonal_combo: 0/6), even where per-trade quality improves. The point of this run
is the MAGNITUDE of improvement per candidate and whether any candidate behaves differently from
orthogonal_combo's pattern, not fishing for a benchmark pass.
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
    HORIZON_BARS,
    LEVERAGE,
    MARGIN_FRACTION,
    ROUNDTRIP_COST_RATE,
    SL_ATR_MULT,
    TP_ATR_MULT,
    _compute_frame,
)
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_evidence_signal_liquidation_confluence_20260827 import (  # noqa: E402
    compute_hourly_levels,
    resample_1h,
)

OUT_DIR = ROOT / "tmp/eth_evidence_signal_liquidation_confluence_costgate_20260827"
CANDIDATES = [
    ("orthogonal_combo", "bottom"),
    ("taker_delta_z_climax", "bottom"),
    ("liquidity_sweep", "bottom"),
    ("short_term_return_z", "bottom"),
    ("smt_divergence", "bottom"),
]
VARIANTS = ("ungated", "chop", "chop_confluence")


def add_confluence(frame: pd.DataFrame) -> pd.DataFrame:
    hourly = resample_1h(frame)
    levels_hourly = compute_hourly_levels(hourly)
    frame_sorted = frame.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(frame_sorted, levels_hourly, on="timestamp", direction="backward")
    dist = merged["support_distance_pct"].abs()
    _, edges = pd.qcut(dist.dropna(), 3, retbins=True, duplicates="drop")
    tertile = pd.cut(dist, bins=edges, labels=["near", "mid", "far"][: len(edges) - 1], include_lowest=True)
    merged["near_or_mid_support"] = tertile.isin(["near", "mid"]).to_numpy()
    return merged


def run_window_confluence(frame: pd.DataFrame, bcol: str, variant: str, *, start, end,
                           roundtrip_cost: float) -> dict[str, Any]:
    ts = frame["timestamp"]
    eligible = purged_decision_mask(ts, start=pd.Timestamp(start), end=pd.Timestamp(end), horizon_bars=HORIZON_BARS)
    bottom = frame[bcol].fillna(False).to_numpy()
    if variant in ("chop", "chop_confluence"):
        bottom = bottom & (frame["regime_label"] == "chop").to_numpy()
    if variant == "chop_confluence":
        bottom = bottom & frame["near_or_mid_support"].fillna(False).to_numpy()
    top = np.zeros(len(frame), dtype=bool)
    score = bottom.astype(np.float64) - top.astype(np.float64)

    has_score = frame["atr_pct"].notna().to_numpy()
    mask = eligible & has_score
    decision_indices = np.flatnonzero(mask)

    tp_moves = (TP_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]
    sl_moves = (SL_ATR_MULT * frame["atr_pct"].to_numpy())[decision_indices]

    result = simulate_single_position(
        timestamps=ts, open_px=frame["open"].to_numpy(), high=frame["high"].to_numpy(),
        low=frame["low"].to_numpy(), close=frame["close"].to_numpy(),
        decision_indices=decision_indices, scores=score[decision_indices],
        tp_moves=tp_moves, sl_moves=sl_moves,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=HORIZON_BARS,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=roundtrip_cost,
    )
    ledger = result.ledger
    total_return = float(result.equity[-1] - 1.0) if len(result.equity) else float("nan")
    n_trades = int(len(ledger))
    wr = float((ledger["price_move"] * ledger["side"] > 0).mean()) if n_trades else float("nan")

    win_mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    win_idx = np.flatnonzero(win_mask.to_numpy())
    if len(win_idx):
        p0, p1 = float(frame["close"].iloc[win_idx[0]]), float(frame["close"].iloc[win_idx[-1]])
        always_long, always_short = p1 / p0 - 1.0, p0 / p1 - 1.0
    else:
        always_long, always_short = float("nan"), float("nan")

    return {
        "n_trades": n_trades, "wr": wr, "total_return": total_return,
        "always_long_return": always_long, "always_short_return": always_short,
        "beats_benchmark": bool(total_return > max(always_long, always_short))
        if np.isfinite(always_long) and np.isfinite(always_short) else None,
    }


def find_breakeven_bp(frame: pd.DataFrame, bcol: str, variant: str, *, start, end) -> float | None:
    lo, hi = 0.0, 0.02
    r_lo = run_window_confluence(frame, bcol, variant, start=start, end=end, roundtrip_cost=lo)["total_return"]
    r_hi = run_window_confluence(frame, bcol, variant, start=start, end=end, roundtrip_cost=hi)["total_return"]
    if not np.isfinite(r_lo) or r_lo <= 0:
        return 0.0
    if r_hi > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        r_mid = run_window_confluence(frame, bcol, variant, start=start, end=end, roundtrip_cost=mid)["total_return"]
        if r_mid > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0 * 10000.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building 2025/2026 frames (live evidence signals + chop regime + ATR + liquidation confluence)...")
    frames = {}
    for yr, base_csv in (("2025", gate.sweep.BASE_2025), ("2026", gate.sweep.BASE_2026)):
        fr = _compute_frame(base_csv)
        fr = add_confluence(fr)
        frames[yr] = fr
        print(f"  {yr}: {len(fr)} rows, near_or_mid_support={fr['near_or_mid_support'].mean() * 100:.1f}%")

    report: dict[str, Any] = {
        "config": {"tp_atr_mult": TP_ATR_MULT, "sl_atr_mult": SL_ATR_MULT, "horizon_bars": HORIZON_BARS,
                   "leverage": LEVERAGE, "margin_fraction": MARGIN_FRACTION,
                   "roundtrip_cost_rate": ROUNDTRIP_COST_RATE, "candidates": CANDIDATES, "variants": VARIANTS},
        "results": {},
    }

    summary_rows = []
    for name, side in CANDIDATES:
        bcol = f"bottom_{name}"
        for variant in VARIANTS:
            key = f"{name}:{side}:{variant}"
            print(f"\n--- {key} ---")
            print(f"{'window':<8} {'n_trades':>8} {'wr':>7} {'return':>10} {'a_long':>9} {'a_short':>9} {'beats_bm':>9}  breakeven_bp")
            windows_out = {}
            for wname, wd in gate.WINDOW_DEFS.items():
                frame = frames["2025"] if wd["base_csv"] == gate.sweep.BASE_2025 else frames["2026"]
                res_std = run_window_confluence(frame, bcol, variant, start=wd["start"], end=wd["end"],
                                                 roundtrip_cost=ROUNDTRIP_COST_RATE)
                be = find_breakeven_bp(frame, bcol, variant, start=wd["start"], end=wd["end"])
                be_str = f"{be:.1f}bp" if be is not None else ">200bp"
                windows_out[wname] = {**res_std, "breakeven_bp": be}
                print(f"{wname:<8} {res_std['n_trades']:>8d} "
                      f"{res_std['wr'] * 100 if np.isfinite(res_std['wr']) else float('nan'):>6.1f}% "
                      f"{res_std['total_return'] * 100:>9.2f}% {res_std['always_long_return'] * 100:>8.2f}% "
                      f"{res_std['always_short_return'] * 100:>8.2f}%  {str(res_std['beats_benchmark']):>9}  {be_str}")
            report["results"][key] = windows_out
            wins = sum(1 for w in windows_out.values() if w["beats_benchmark"])
            total_ret = sum(w["total_return"] for w in windows_out.values())
            print(f"SUMMARY {key}: beats always_long/always_short in {wins}/{len(windows_out)} windows, "
                  f"sum(total_return)={total_ret * 100:.2f}%")
            summary_rows.append({"signal": name, "variant": variant, "wins": wins,
                                  "sum_total_return_pct": total_ret * 100})

    print("\n=== Cross-candidate summary (sum of total_return across 6 windows) ===")
    print(f"{'signal':<24} {'ungated':>10} {'chop':>10} {'chop_confluence':>16} {'delta(conf-chop)':>18}")
    by_sig: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        by_sig.setdefault(row["signal"], {})[row["variant"]] = row["sum_total_return_pct"]
    for name, _side in CANDIDATES:
        v = by_sig[name]
        delta = v["chop_confluence"] - v["chop"]
        print(f"{name:<24} {v['ungated']:>9.2f}% {v['chop']:>9.2f}% {v['chop_confluence']:>15.2f}% {delta:>+17.2f}%")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
