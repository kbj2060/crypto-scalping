#!/usr/bin/env python3
"""Retry the BTC Tau1-style regime_tiebreak combination (Leg A=h48qual, Leg B=Sigma9
trend-scan+regime), swapping the tiebreak's regime driver from the current-state HMM (bull/bear
flip rate ~34-35% of bars, close to noise, root-caused in
project-btc-tau1-style-leg-combination-first-attempt-20260801.md) for BTC's CryptoMamba
FUTURE-regime prediction model (regime3_cmamba_h6_future_{bull,bear}_prob,
data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721/) -- a genuinely
different, never-before-used-in-a-gate signal (OOS AUC 0.8365, close to ETH's own 0.8438, per
docs/model_contracts/sol_btc_regime_models_retrain_tuning_20260721.md Part 3). Pre-check found this
signal's own bar-to-bar flip rate is 25.3% over the combination window -- better than the
current-HMM's 34-35% but still non-trivial, so this script does NOT stop at one VAL+OOS split (the
mistake made the first time): it runs the SAME frozen mechanism across the 4 rolling windows already
used for the fixed-weight check, to see whether any improvement survives outside the single
already-inspected split.

Leg A/B trade loading, equity-path reconstruction, and the regime_tiebreak weight rule are reused
UNCHANGED from the earlier scripts -- only the regime probability SOURCE changes.
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
    omega_trades_from_ledger, build_leg_equity_path, summarize_equity,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    LEG_A_DIR, LEG_B_DIR, load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_tau1_cryptomamba_tiebreak"
CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_btc_h6_nocurrent_20260721"
CMAMBA_COLS = ["timestamp", "regime3_cmamba_h6_future_bull_prob", "regime3_cmamba_h6_future_bear_prob"]

LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

VAL_OOS_WINDOWS = [
    ("VAL_2025Q4", "2025-10-01", "2025-12-31"),
    ("OOS_2026extended", "2026-01-01", "2026-06-25"),
]
ROLLING_WINDOWS = [
    ("W1", "2025-10-01", "2026-01-31"),
    ("W2", "2025-12-01", "2026-03-31"),
    ("W3", "2026-02-01", "2026-05-31"),
    ("W4_data_end", "2026-02-25", "2026-06-25"),
]


def load_cryptomamba_regime() -> pd.DataFrame:
    dfs = [pd.read_csv(CMAMBA_DIR / f"btc_features_{y}_regime3_cryptomamba_pred_btc_h6_nocurrent_20260721.csv",
                        usecols=CMAMBA_COLS)
           for y in (2025, 2026)]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").rename(columns={
        "regime3_cmamba_h6_future_bull_prob": "bull_prob",
        "regime3_cmamba_h6_future_bear_prob": "bear_prob"})


def load_all_trades(paths, start, end) -> list[dict]:
    trades = []
    for p in paths:
        trades.extend(omega_trades_from_ledger(p, start, end))
    return trades


def run_window(label, start_s, end_s, prices, regime) -> dict:
    """FIX 2026-08-02 (project-btc-run-window-merge-point-fixed-20260802.md): the regime frame's own
    timestamp is shifted forward by +1h before merge_asof, guaranteeing the regime row matched to bar
    t's delta derives from data at least one full hour stale relative to that delta's own window
    (which, under resample("1h").last()'s left-labeled bins, actually spans up to ~t+55min). This is
    the SAME validated fix pattern as build_dumb_momentum_regime()'s 2026-08-01 fix and the
    diagnose_btc_cryptomamba_tiebreak_timestamp_shift_20260802.py control that proved it collapses the
    leaky result (win rate 5/6->1/6). (An earlier attempt to fix this by relabeling eq_a_1h/eq_b_1h's
    resample as right-anchored was tried and reverted -- it inadvertently shifted the EQUITY series
    forward instead of tightening the gap, making results LARGER not smaller. Shifting the regime side
    is the correct, empirically-verified fix.)"""
    start, end = pd.Timestamp(start_s), pd.Timestamp(end_s) + pd.Timedelta("23h59min59s")
    trades_a = load_all_trades(LEG_A_LEDGERS, start, end)
    trades_b = load_all_trades(LEG_B_LEDGERS, start, end)

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)
    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)].copy()
    reg["timestamp"] = reg["timestamp"] + pd.Timedelta(hours=1)
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame["bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame["bear_prob"].fillna(0.5).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)
    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)

    n_conflict = int(conflict.sum())
    beats_leg_a = tiebreak["pnl_pct"] > sa["pnl_pct"] and tiebreak["mdd_pct"] > sa["mdd_pct"]
    print(f"\n=== {label} {start_s}..{end_s} ===")
    print(f"Leg A alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}%   "
          f"Leg B alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}%")
    print(f"Fixed 1x-1x: pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%   "
          f"CryptoMamba tiebreak (n_conflict={n_conflict}): pnl={tiebreak['pnl_pct']:+.2f}% "
          f"mdd={tiebreak['mdd_pct']:.2f}%  beats_leg_a_both_axes={beats_leg_a}")
    return {
        "window": label, "start": start_s, "end": end_s,
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"],
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"],
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "cmamba_tiebreak_pnl": tiebreak["pnl_pct"], "cmamba_tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "beats_leg_a_both_axes": beats_leg_a,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime = load_cryptomamba_regime()

    print("########## VAL/OOS split (comparable to the original current-HMM tiebreak result) ##########")
    rows1 = [run_window(l, s, e, prices, regime) for l, s, e in VAL_OOS_WINDOWS]
    pd.DataFrame(rows1).to_csv(OUT_DIR / "val_oos_summary.csv", index=False)

    print("\n########## Rolling windows (same discipline as the fixed-weight check) ##########")
    rows2 = [run_window(l, s, e, prices, regime) for l, s, e in ROLLING_WINDOWS]
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUT_DIR / "rolling_window_summary.csv", index=False)
    n_wins = int(df2["beats_leg_a_both_axes"].sum())
    print(f"\n=== SUMMARY: CryptoMamba tiebreak beats Leg A alone (pnl AND mdd) in {n_wins}/{len(df2)} rolling windows ===")
    print(df2.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
