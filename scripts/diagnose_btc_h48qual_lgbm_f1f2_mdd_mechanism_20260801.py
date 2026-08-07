#!/usr/bin/env python3
"""Diagnostic ONLY (per user request 20260801): reconstruct the combined h48qual (Leg A) +
btc_v2_direction_quality_lgbm (Leg B) dumb-momentum-tiebreak (N=3h) equity curve bar-by-bar for
LOWO folds F1 (2025-10-08..2025-11-29) and F2 (2025-11-30..2026-01-20), find the actual point(s) of
max drawdown, and inspect what each leg/weight was doing there. Reuses
scripts/research_btc_h48qual_direction_quality_lgbm_lowo_20260801.py's own primitives/globals
unchanged -- does not modify that script or any artifact. Writes tables to
tmp/research_20260801/btc_h48qual_lgbm_f1f2_mdd_diagnosis/.
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
    build_leg_equity_path, summarize_equity,
)
from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import load_all_trades  # noqa: E402
from diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801 import (  # noqa: E402
    build_dumb_momentum_regime,
)
from research_btc_h48qual_direction_quality_lgbm_lowo_20260801 import (  # noqa: E402
    LEG_A_LEDGERS, LEG_B_LEDGERS,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_lgbm_f1f2_mdd_diagnosis"
N_HOURS = 3  # selected value used for F1/F2 in the LOWO grid

FOLDS = {
    "F1": (pd.Timestamp("2025-10-08"), pd.Timestamp("2025-11-29 23:59:59")),
    "F2": (pd.Timestamp("2025-11-30"), pd.Timestamp("2026-01-20 23:59:59")),
}


def weighted_pnl_series(delta_a, delta_b, w_a, w_b) -> np.ndarray:
    """Like weighted_pnl() but returns the full equity path instead of just the summary."""
    equity = 1.0
    out = np.empty(len(delta_a))
    for i in range(len(delta_a)):
        equity += w_a[i] * delta_a[i] + w_b[i] * delta_b[i]
        out[i] = equity
    return out


def analyze_fold(label: str, start: pd.Timestamp, end: pd.Timestamp, prices, regime) -> None:
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

    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)]
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame["bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame["bear_prob"].fillna(0.5).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    combo_path = weighted_pnl_series(delta_a, delta_b, w_a, w_b)

    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)
    combo_summary = {"pnl_pct": (combo_path[-1] - 1.0) * 100,
                      "mdd_pct": (pd.Series(combo_path) / pd.Series(combo_path).cummax() - 1).min() * 100}

    # Bar-level table for full diagnostic
    df = pd.DataFrame({
        "timestamp": ts.to_numpy(), "eq_a": eq_a_1h.to_numpy(), "eq_b": eq_b_1h.to_numpy(),
        "delta_a": delta_a, "delta_b": delta_b, "side_a": side_a, "side_b": side_b,
        "conflict": conflict, "w_a": w_a, "w_b": w_b, "bull_prob": bull, "bear_prob": bear,
        "combo_equity": combo_path,
    })
    df["combo_peak"] = df["combo_equity"].cummax()
    df["combo_dd"] = df["combo_equity"] / df["combo_peak"] - 1.0
    df["eq_a_peak"] = df["eq_a"].cummax()
    df["eq_a_dd"] = df["eq_a"] / df["eq_a_peak"] - 1.0
    df["eq_b_peak"] = df["eq_b"].cummax()
    df["eq_b_dd"] = df["eq_b"] / df["eq_b_peak"] - 1.0

    combo_mdd_idx = df["combo_dd"].idxmin()
    combo_mdd_row = df.loc[combo_mdd_idx]
    # peak just before this trough
    peak_idx = df.loc[:combo_mdd_idx, "combo_equity"].idxmax()

    print(f"\n########## {label} ({start.date()}..{end.date()}) N={N_HOURS}h ##########")
    print(f"Leg A alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% (own trough at "
          f"{df.loc[df['eq_a_dd'].idxmin(), 'timestamp']}, dd={df['eq_a_dd'].min()*100:.2f}%)")
    print(f"Leg B alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% (own trough at "
          f"{df.loc[df['eq_b_dd'].idxmin(), 'timestamp']}, dd={df['eq_b_dd'].min()*100:.2f}%)")
    print(f"Combined (tiebreak): pnl={combo_summary['pnl_pct']:+.2f}% mdd={combo_summary['mdd_pct']:.2f}%")
    print(f"\nCombined-equity MAX DRAWDOWN trough at {combo_mdd_row['timestamp']} "
          f"(dd={combo_mdd_row['combo_dd']*100:.2f}%), drawdown started from peak at {df.loc[peak_idx, 'timestamp']} "
          f"(peak equity={df.loc[peak_idx, 'combo_equity']:.4f})")

    window = df.loc[peak_idx:combo_mdd_idx]
    n_conf_in_dd = int(window["conflict"].sum())
    both_same_side_losing = window[(window["side_a"] != 0) & (window["side_b"] != 0) &
                                    (window["side_a"] == window["side_b"])]
    print(f"Drawdown window spans {len(window)} bars ({window['timestamp'].iloc[0]} .. {window['timestamp'].iloc[-1]}), "
          f"conflict bars in window={n_conf_in_dd}, both-legs-same-side bars in window={len(both_same_side_losing)}")

    # Was combo dd at trough worse than BOTH legs' own dd measured over the SAME window (not whole fold)?
    eq_a_window = df.loc[peak_idx:combo_mdd_idx, "eq_a"]
    eq_b_window = df.loc[peak_idx:combo_mdd_idx, "eq_b"]
    a_dd_in_window = (eq_a_window.iloc[-1] / eq_a_window.cummax().iloc[-1] - 1) if len(eq_a_window) else np.nan
    a_local_dd = (eq_a_window.min() / eq_a_window.iloc[0] - 1) * 100 if len(eq_a_window) else np.nan
    b_local_dd = (eq_b_window.min() / eq_b_window.iloc[0] - 1) * 100 if len(eq_b_window) else np.nan
    a_move_over_window = (eq_a_window.iloc[-1] - eq_a_window.iloc[0]) * 100
    b_move_over_window = (eq_b_window.iloc[-1] - eq_b_window.iloc[0]) * 100
    print(f"Over the SAME drawdown window: Leg A equity moved {a_move_over_window:+.2f}pp "
          f"(local min dd from window-start {a_local_dd:.2f}%), Leg B equity moved {b_move_over_window:+.2f}pp "
          f"(local min dd from window-start {b_local_dd:.2f}%)")

    # rows with biggest single-bar combined equity drop inside the window, with weights + which leg drove it
    window = window.copy()
    window["combo_delta"] = window["combo_equity"].diff()
    worst_bars = window.sort_values("combo_delta").head(8)
    print("\nWorst single-bar combined-equity drops inside the drawdown window:")
    cols = ["timestamp", "combo_delta", "delta_a", "delta_b", "w_a", "w_b", "side_a", "side_b", "conflict", "bull_prob", "bear_prob"]
    print(worst_bars[cols].to_string(index=False))

    # trades active (by entry/exit span) overlapping the drawdown window, from each leg
    dd_start, dd_end = df.loc[peak_idx, "timestamp"], df.loc[combo_mdd_idx, "timestamp"]

    def trades_overlapping(trades, s, e):
        rows = []
        for tr in trades:
            if tr["entry_timestamp"] <= e and tr["exit_timestamp"] >= s:
                rows.append({"entry": tr["entry_timestamp"], "exit": tr["exit_timestamp"],
                             "side": tr["side"], "trade_return_pct": tr["trade_return"] * 100})
        return pd.DataFrame(rows)

    ta = trades_overlapping(trades_a, dd_start, dd_end)
    tb = trades_overlapping(trades_b, dd_start, dd_end)
    print(f"\nLeg A trades overlapping drawdown window ({len(ta)}):")
    print(ta.to_string(index=False) if len(ta) else "  (none)")
    print(f"\nLeg B trades overlapping drawdown window ({len(tb)}):")
    print(tb.to_string(index=False) if len(tb) else "  (none)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{label}_bar_level.csv", index=False)
    window.to_csv(OUT_DIR / f"{label}_drawdown_window.csv", index=False)
    ta.to_csv(OUT_DIR / f"{label}_leg_a_trades_in_dd.csv", index=False)
    tb.to_csv(OUT_DIR / f"{label}_leg_b_trades_in_dd.csv", index=False)


def main() -> int:
    prices_5m = load_5m_prices_btc()
    close_1h = prices_5m["close"].resample("1h").last().ffill()
    regime = build_dumb_momentum_regime(close_1h, N_HOURS)

    for label, (start, end) in FOLDS.items():
        analyze_fold(label, start, end, prices_5m, regime)

    print(f"\nWrote per-fold tables to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
