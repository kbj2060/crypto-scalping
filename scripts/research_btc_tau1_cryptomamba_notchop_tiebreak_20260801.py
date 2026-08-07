#!/usr/bin/env python3
"""Tests the chop-aware regime_tiebreak variant on BTC's CryptoMamba future-regime signal, mirroring
scripts/eval_sigma6_omega_regime_tiebreak_notchop_20260801.py's ETH test (which FAILED -- notchop
was clearly worse than the plain tiebreak on both pnl and mdd, both windows, see
project-eth-regime-tiebreak-notchop-variant-failed-20260801.md). Run here anyway per explicit user
request, since BTC's regime dynamics (chop fraction, transition behavior) are not guaranteed to
match ETH's -- this project has repeatedly found things that don't transfer between the two assets.

Same VAL/OOS + 4 rolling windows as research_btc_tau1_cryptomamba_tiebreak_20260801.py (the plain,
best-so-far BTC tiebreak). Only the weight rule differs: on conflict bars where
regime3_cmamba_h6_future_chop_prob > max(bull_prob, bear_prob), fall back to baseline 1.0/1.0
instead of forcing a bull/bear pick.
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
    LEG_A_DIR, LEG_B_DIR, load_5m_prices_btc, leg_side_series, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import CMAMBA_DIR, VAL_OOS_WINDOWS, ROLLING_WINDOWS  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260801/btc_tau1_cryptomamba_notchop_tiebreak"
CMAMBA_COLS = ["timestamp", "regime3_cmamba_h6_future_bull_prob", "regime3_cmamba_h6_future_bear_prob",
               "regime3_cmamba_h6_future_chop_prob"]

LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]


def load_cryptomamba_regime() -> pd.DataFrame:
    dfs = [pd.read_csv(CMAMBA_DIR / f"btc_features_{y}_regime3_cryptomamba_pred_btc_h6_nocurrent_20260721.csv",
                        usecols=CMAMBA_COLS)
           for y in (2025, 2026)]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").rename(columns={
        "regime3_cmamba_h6_future_bull_prob": "bull_prob",
        "regime3_cmamba_h6_future_bear_prob": "bear_prob",
        "regime3_cmamba_h6_future_chop_prob": "chop_prob"})


def load_all_trades(paths, start, end) -> list[dict]:
    trades = []
    for p in paths:
        trades.extend(omega_trades_from_ledger(p, start, end))
    return trades


def plain_and_notchop_weights(conflict, side_a, side_b, bull, bear, chop):
    n = len(conflict)
    regime_side = np.where(bull >= bear, 1, -1)
    chop_dominant = chop > np.maximum(bull, bear)

    w_a_plain, w_b_plain = np.ones(n), np.ones(n)
    w_a_plain[conflict] = np.where(side_a[conflict] == regime_side[conflict], 1.0, 0.0)
    w_b_plain[conflict] = np.where(side_b[conflict] == regime_side[conflict], 1.0, 0.0)

    trust = conflict & ~chop_dominant
    w_a_nc, w_b_nc = np.ones(n), np.ones(n)
    w_a_nc[trust] = np.where(side_a[trust] == regime_side[trust], 1.0, 0.0)
    w_b_nc[trust] = np.where(side_b[trust] == regime_side[trust], 1.0, 0.0)

    chop_dominant_frac = float(chop_dominant[conflict].mean()) if conflict.sum() else 0.0
    return (w_a_plain, w_b_plain), (w_a_nc, w_b_nc), chop_dominant_frac


def run_window(label, start_s, end_s, prices, regime) -> dict:
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

    # FIX 2026-08-02 (project-btc-run-window-merge-point-fixed-20260802.md): shift regime timestamp
    # +1h before merge_asof so the matched row is guaranteed at least 1h stale relative to the delta
    # window it gates (closes the same-bar look-ahead in the shared run_window() merge pattern).
    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)].copy()
    reg["timestamp"] = reg["timestamp"] + pd.Timedelta(hours=1)
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame["bull_prob"].fillna(0.34).to_numpy()
    bear = reg_frame["bear_prob"].fillna(0.33).to_numpy()
    chop = reg_frame["chop_prob"].fillna(0.33).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    (w_a_p, w_b_p), (w_a_nc, w_b_nc), chop_frac = plain_and_notchop_weights(conflict, side_a, side_b, bull, bear, chop)
    plain = weighted_pnl(delta_a, delta_b, w_a_p, w_b_p)
    notchop = weighted_pnl(delta_a, delta_b, w_a_nc, w_b_nc)
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)

    n_conflict = int(conflict.sum())
    print(f"\n=== {label} {start_s}..{end_s} (n_conflict={n_conflict}, chop-dominant={chop_frac*100:.1f}%) ===")
    print(f"Leg A alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}%")
    print(f"CryptoMamba tiebreak (plain)  : pnl={plain['pnl_pct']:+.2f}% mdd={plain['mdd_pct']:.2f}%")
    print(f"CryptoMamba tiebreak (notchop): pnl={notchop['pnl_pct']:+.2f}% mdd={notchop['mdd_pct']:.2f}%")
    return {
        "window": label, "start": start_s, "end": end_s, "n_conflict": n_conflict, "chop_dominant_frac": chop_frac,
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"],
        "plain_pnl": plain["pnl_pct"], "plain_mdd": plain["mdd_pct"],
        "notchop_pnl": notchop["pnl_pct"], "notchop_mdd": notchop["mdd_pct"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime = load_cryptomamba_regime()

    print("########## VAL/OOS split ##########")
    rows1 = [run_window(l, s, e, prices, regime) for l, s, e in VAL_OOS_WINDOWS]
    pd.DataFrame(rows1).to_csv(OUT_DIR / "val_oos_summary.csv", index=False)

    print("\n########## Rolling windows ##########")
    rows2 = [run_window(l, s, e, prices, regime) for l, s, e in ROLLING_WINDOWS]
    df2 = pd.DataFrame(rows2)
    df2.to_csv(OUT_DIR / "rolling_window_summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df2.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
