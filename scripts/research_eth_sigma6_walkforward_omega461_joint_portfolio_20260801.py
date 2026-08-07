#!/usr/bin/env python3
"""Bar-level combined-portfolio check: does the walk-forward-confirmed Sigma6 regime-filter
candidate (thr=0.60/lev=3.0/sl=1.5/not_chop/rthr=0.50/stab=0.0 -- see
project-sigma6-regime-filter-leave-one-window-out-CANDIDATE-20260801.md) add real value as a SECOND
ETH portfolio slot alongside live Omega4.6.1, on a real bar-level combined equity curve -- not just
the daily-return correlation diagnostic already run
(research_eth_sigma6_walkforward_omega461_correlation_20260801.py), which showed high time-overlap
(84-98% both-in-position days) that a pure correlation number doesn't capture.

Reuses research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py's validated equity-path
methodology UNCHANGED (build_leg_equity_path, omega_trades_from_ledger,
sanity_check_omega_reproduction, summarize_equity) -- that script's docstring documents why this
specific linear-intra-trade / compounding-cross-trade equity model was chosen (matches the Futures
Risk Sizing Contract exactly; an earlier bar-by-bar-compounded version had a reproducibility gap).
Only new code: sigma6_filtered_trades(), which adds the regime-filter entry gate
(reg_mode='not_chop': only enter if chop_prob < reg_thr) on top of that script's sigma3_trades()
mechanics, with a sanity check against run_sigma6_regime_trend_20260705.backtest() for the identical
config before trusting it downstream (same discipline as the original script's sigma3 sanity check).

Both components use independent margin/notional (separate capital sleeves), combined as additive
dollar-PnL on a shared 1.0 reference equity -- same convention as the 3-asset portfolio and the
Sigma3-1h+Omega4.6.1 joint check. DIAGNOSTIC ONLY per CLAUDE.md -- not a promotion decision by
itself; see the memory file above for the full list of promotion conditions still outstanding.
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

import replay_omega6_v2_variants_20260704 as v2  # noqa: E402
from run_sigma6_regime_trend_20260705 import load_tape_with_regime, backtest as orig_backtest  # noqa: E402
from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    load_5m_prices, omega_trades_from_ledger, build_leg_equity_path,
    sanity_check_omega_reproduction, summarize_equity,
)

OUT_DIR = ROOT / "tmp/research_20260801/sigma6_walkforward_omega461_joint_portfolio"
WINNER = dict(thr=0.60, lev=3.0, sl=1.5, mode="not_chop", rthr=0.50, stab=0.0)
BASE_KW = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)
OMEGA_VAL_LEDGER = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv"
OMEGA_OOS_LEDGER = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"
WINDOWS = [
    ("VAL_2025Q4", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59"), OMEGA_VAL_LEDGER),
    ("OOS_2026H1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-25 23:59:59"), OMEGA_OOS_LEDGER),
]


def sigma6_filtered_trades(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold,
                            cooldown, reg_mode, reg_thr, stab_thr, start, end):
    """Same mechanics as the joint-portfolio script's sigma3_trades(), plus the regime-filter entry
    gate from run_sigma6_regime_trend_20260705.backtest() (not_chop / trend_agree / stability)."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
    PFX = "regime3_current_sensitive_wide24_"
    bull = sub[f"{PFX}bull_prob"].to_numpy(np.float64)
    bear = sub[f"{PFX}bear_prob"].to_numpy(np.float64)
    chop = sub[f"{PFX}chop_prob"].to_numpy(np.float64)
    stab = sub["regime3_cmamba_h6_sidecar_stability_score"].fillna(1.0).to_numpy(np.float64)
    ts = sub["timestamp"]
    FEE, SLIP = 0.00020, 0.00050
    notional = margin * leverage
    pos = 0
    entry_price = peak_unreal = entry_atr = 0.0
    hold_start = 0
    trades = []
    cooldown_until = -1
    i = 0
    while i < n - 1:
        if pos == 0:
            if i < cooldown_until or side_arr[i] == 0:
                i += 1
                continue
            side = int(side_arr[i])
            ok = True
            if reg_mode == "trend_agree":
                ok = (side > 0 and bull[i] >= reg_thr) or (side < 0 and bear[i] >= reg_thr)
            elif reg_mode == "not_chop":
                ok = chop[i] < reg_thr
            if ok and stab_thr > 0:
                ok = stab[i] >= stab_thr
            if not ok:
                i += 1
                continue
            entry_price = float(open_[min(i + 1, n - 1)]) * (1 + SLIP if side > 0 else 1 - SLIP)
            pos, hold_start, peak_unreal, entry_atr = side, i, 0.0, max(atr_arr[i], 1e-6)
            i += 1
            continue
        px = close[i]
        raw = (px * (1 - SLIP) - entry_price) / entry_price if pos > 0 else (entry_price - px * (1 + SLIP)) / entry_price
        unreal = raw * notional
        peak_unreal = max(peak_unreal, unreal)
        hold = i - hold_start
        reason = ""
        if unreal <= -sl_atr * entry_atr:
            reason = "stop"
        elif peak_unreal >= min_profit_atr * entry_atr and (peak_unreal - unreal) >= trail_atr * entry_atr:
            reason = "trail"
        elif hold >= max_hold:
            reason = "time"
        if reason:
            exit_price = close[i] * (1 - SLIP if pos > 0 else 1 + SLIP)
            trades.append({"entry_timestamp": ts.iloc[hold_start], "exit_timestamp": ts.iloc[i],
                           "side": pos, "notional": notional, "reason": reason,
                           "entry_price": entry_price, "exit_price": exit_price,
                           "fee_notional": FEE * notional * 2})
            pos = 0
            cooldown_until = i + cooldown
        i += 1
    return trades


def sanity_check_sigma6_reproduction(tape, prices, start, end, label: str) -> list[dict]:
    orig = orig_backtest(tape, leverage=WINNER["lev"], sl_atr=WINNER["sl"], reg_mode=WINNER["mode"],
                          reg_thr=WINNER["rthr"], stab_thr=WINNER["stab"], fee_mult=1.0,
                          start=start, end=end, **BASE_KW)
    trades = sigma6_filtered_trades(tape, leverage=WINNER["lev"], margin=BASE_KW["margin"],
                                     trail_atr=BASE_KW["trail_atr"], sl_atr=WINNER["sl"],
                                     min_profit_atr=BASE_KW["min_profit_atr"], max_hold=BASE_KW["max_hold"],
                                     cooldown=BASE_KW["cooldown"], reg_mode=WINNER["mode"],
                                     reg_thr=WINNER["rthr"], stab_thr=WINNER["stab"], start=start, end=end)
    assert len(trades) == orig["trades"], f"Sigma6-filtered trade count mismatch ({label}): {len(trades)} vs {orig['trades']}"
    eq = build_leg_equity_path(trades, prices, start, end, use_ledger_trade_return=False)
    recon = summarize_equity(eq)
    pnl_diff = recon["pnl_pct"] - orig["pnl"]
    print(f"[sanity check {label}] Sigma6-filtered reimpl trade count matches original backtest(): "
          f"{len(trades)} == {orig['trades']} (orig pnl={orig['pnl']:.2f}% mdd={orig['mdd']:.2f}%)")
    print(f"[sanity check {label}] equity-path reconstruction pnl={recon['pnl_pct']:+.2f}% mdd={recon['mdd_pct']:.2f}% "
          f"(pnl diff vs orig={pnl_diff:+.4f}pp)")
    assert abs(pnl_diff) < 0.5, f"Sigma6-filtered reconstruction does not reproduce original backtest() pnl ({label}): diff={pnl_diff}pp"
    return trades


def run_window(label, start, end, prices, omega_ledger_path):
    raw = load_tape_with_regime()
    tape = v2.apply_quality_threshold(raw, WINNER["thr"])
    s6_trades = sanity_check_sigma6_reproduction(tape, prices, start, end, label)
    om_trades = omega_trades_from_ledger(omega_ledger_path, start, end)
    eq_a = sanity_check_omega_reproduction(om_trades, prices, start, end, label)  # Omega4.6.1 alone
    eq_b = build_leg_equity_path(s6_trades, prices, start, end, use_ledger_trade_return=False)  # Sigma6 alone
    eq_ab = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)  # additive dollar-PnL, independent sleeves

    sa, sb, sab = summarize_equity(eq_a), summarize_equity(eq_b), summarize_equity(eq_ab)
    print(f"\n=== {label} {start.date()}..{end.date()} ===")
    print(f"Omega4.6.1 alone       : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(om_trades)}")
    print(f"Sigma6-filtered alone   : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(s6_trades)}")
    print(f"Combined portfolio      : pnl={sab['pnl_pct']:+.2f}% mdd={sab['mdd_pct']:.2f}%")
    print(f"  MDD change vs Omega alone: {sab['mdd_pct'] - sa['mdd_pct']:+.2f}pp "
          f"({'WORSE' if sab['mdd_pct'] < sa['mdd_pct'] else 'better'})")
    print(f"  PnL change vs Omega alone: {sab['pnl_pct'] - sa['pnl_pct']:+.2f}pp")
    return {"label": label, "omega_pnl": sa["pnl_pct"], "omega_mdd": sa["mdd_pct"],
            "sigma6_pnl": sb["pnl_pct"], "sigma6_mdd": sb["mdd_pct"],
            "combined_pnl": sab["pnl_pct"], "combined_mdd": sab["mdd_pct"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices()
    rows = [run_window(label, start, end, prices, ledger) for label, start, end, ledger in WINDOWS]
    pd.DataFrame(rows).to_csv(OUT_DIR / "joint_portfolio_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
