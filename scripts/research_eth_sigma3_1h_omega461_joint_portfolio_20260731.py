#!/usr/bin/env python3
"""Joint bar-level portfolio check: does the plain (no regime-filter) Sigma3-1h+trailing-stop
baseline -- the one piece of scripts/research_sigma6_freshforward_canonical_20260731.py that showed
VAL/OOS consistency -- add real value as a SECOND ETH portfolio slot alongside the live Omega4.6.1
5m component, on a real bar-level combined equity curve (not the noisy daily-trade-return
correlation check from project-sigma6-omega461-correlation-inconclusive-20260728).

Omega4.6.1 side: real saved VAL/OOS trade ledgers, used diagnostically per CLAUDE.md (established
live baseline, not being re-derived or promoted here -- only combined with the ALREADY
fresh-forward-tested Sigma3-1h candidate to check portfolio-level MDD/trade-count effect, same
pattern as project-portfolio-3asset-design.md's SOL/BTC combination).
Sigma3-1h side: recomputed fresh (no regime filter, thr=0.60, lev=3.0, sl_atr=1.5, trail_atr=5.0,
min_profit_atr=2.0, max_hold=144, cooldown=3 -- the exact winning-baseline config from the prior
script), marked-to-market on REAL 5m ETH prices for bar-level combination with the 5m ledger.

Both components use independent margin/notional (separate capital sleeves), consistent with the
Futures Risk Sizing Contract -- PnL of each is computed independently and summed as % of a shared
1.0 reference equity (same convention as the existing 3-asset portfolio combination).

EQUITY MODEL (fixed after two earlier failed attempts, see project memory): per the Futures Risk
Sizing Contract, "PnL = price_move * notional" is a SIMPLE LINEAR relationship, not a bar-by-bar
compounded one. An earlier version of this script compounded per-bar notional-scaled returns
within a single trade, which decays/inflates leveraged multi-day trades relative to the project's
own convention (~3pp gap against `run_sigma6_regime_trend_20260705.backtest()`'s own aggregate for
an IDENTICAL config -- a reproducibility failure, not just an approximation gap). The fix used here:
within a trade, unrealized PnL is `entry_equity * raw_linear_price_move * notional` (matching the
contract exactly); ACROSS trades, `entry_equity` for trade N+1 is trade N's realized closing
equity (so compounding still happens correctly at trade boundaries, matching how
`cash *= (1 + rex*notional)` sequential replay works everywhere else in this project).
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
from run_sigma6_regime_trend_20260705 import load_tape_with_regime  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260731/sigma3_1h_omega461_joint_portfolio"
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")
OMEGA_VAL_LEDGER = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv"
OMEGA_OOS_LEDGER = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"
PFX = "regime3_current_sensitive_wide24_"


def load_5m_prices() -> pd.Series:
    df = pd.read_csv(ROOT / "data/training_features_5m.csv", usecols=["timestamp", "open", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").set_index("timestamp")


def sigma3_trades(tape, *, leverage, margin, trail_atr, sl_atr, min_profit_atr, max_hold, cooldown,
                   start, end):
    """Same mechanics as run_sigma6_regime_trend_20260705.backtest with reg_mode='none'. Unlike that
    aggregate-only function, this ALSO records each trade's entry/exit timestamp, side, notional,
    and real slip-adjusted entry_price/exit_price so build_leg_equity_path() can mark it to market
    bar-by-bar using the project's linear (non-compounding) PnL=price_move*notional contract."""
    sub = tape[(tape["timestamp"] >= start) & (tape["timestamp"] <= end)].reset_index(drop=True)
    n = len(sub)
    close = sub["close"].to_numpy(np.float64)
    open_ = sub["open"].to_numpy(np.float64)
    side_arr = sub["primary_side"].to_numpy(np.int64)
    atr_arr = sub["atr_pct"].to_numpy(np.float64)
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


OMEGA_FEE_PER_SIDE = 0.0007  # real project fee_eff+slip_eff convention (Futures Risk Sizing
                              # Contract / continuous_weight memory's "turnover_cost"), NOT
                              # Sigma6's own 1h research placeholder (0.00020+0.00050).


def omega_trades_from_ledger(path: Path, start, end) -> list[dict]:
    """Trades with entry_timestamp inside [start,end]. No entry/exit fill price is recorded in the
    saved ledger, so entry_price is approximated as the real 5m close at entry_timestamp (a small,
    acknowledged approximation -- see sanity check) and exit_price is back-solved from the ledger's
    OWN authoritative trade_return so this leg's TOTAL matches the real ledger exactly at each
    trade's close (only the intrabar shape is approximated, not the total)."""
    df = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    df = df[(df["entry_timestamp"] >= start) & (df["entry_timestamp"] <= end)]
    out = []
    for _, r in df.iterrows():
        notional = float(r["notional"])
        out.append({"entry_timestamp": r["entry_timestamp"], "exit_timestamp": r["exit_timestamp"],
                    "side": int(r["side"]), "notional": notional,
                    "trade_return": float(r["trade_return"]), "fee_notional": 0.0})
    return out


def build_leg_equity_path(trades: list[dict], prices: pd.Series, start, end, *,
                           use_ledger_trade_return: bool = False) -> pd.Series:
    """Bar-level equity curve for ONE leg (Omega4.6.1 or Sigma3-1h), starting at 1.0.
    Intra-trade: unrealized = entry_equity * raw_linear_price_move * notional (matches the Futures
    Risk Sizing Contract exactly, no bar-by-bar compounding decay of a leveraged position).
    Cross-trade: entry_equity for trade N+1 is trade N's realized closing equity (real compounding
    across trades, matching every other sequential replay in this project).
    If use_ledger_trade_return, the trade's REAL total return (from the saved ledger) is used to
    back-solve exit_price so the leg's realized total matches the ledger exactly at each close;
    otherwise entry_price/exit_price on the trade dict are used directly (Sigma3-1h side, which has
    real slip-adjusted fill prices already)."""
    idx = prices.loc[start:end].index
    eq = pd.Series(np.nan, index=idx)
    running_eq = 1.0
    for tr in sorted(trades, key=lambda t: t["entry_timestamp"]):
        e = tr["entry_timestamp"]
        x = min(tr["exit_timestamp"], end)
        window = prices.loc[e:x]
        if len(window) < 1:
            continue
        closes = window["close"].to_numpy(np.float64)
        if use_ledger_trade_return:
            entry_price = closes[0]
            if tr["exit_timestamp"] > end:
                # boundary trade still open at window end: partial mark-to-market on real prices,
                # not the eventual full trade_return (would leak post-window information)
                total_move = (closes[-1] - entry_price) / entry_price * tr["side"]
            else:
                total_move = tr["trade_return"] / max(tr["notional"], 1e-9)
            raw_move = (closes - entry_price) / entry_price * tr["side"]
            # Guard against the rescale blowing up when the ledger's back-solved exit price diverges
            # from the raw close at the exit timestamp: raw_move[-1] can be small-but-nonzero (passes
            # the old 1e-12 check) yet still yield an extreme scale factor that amplifies ordinary
            # intrabar noise into a synthetic multi-x equity swing (found 2026-08-01, see
            # project-build-leg-equity-path-rescale-artifact-found-20260801.md: a near-flat -0.085%
            # trade got rescaled ~17x into a fake ~50% swing). Cap the scale factor's magnitude instead
            # of only checking the denominator isn't literally zero.
            MAX_RESCALE_FACTOR = 5.0
            scale = total_move / raw_move[-1] if (len(raw_move) > 1 and abs(raw_move[-1]) > 1e-12) else None
            if scale is not None and abs(scale) <= MAX_RESCALE_FACTOR:
                raw_move = raw_move * scale  # anchor endpoint to real total, preserve shape
            else:
                raw_move = np.full_like(closes, total_move)
        else:
            entry_price = tr["entry_price"]
            raw_move = (closes - entry_price) / entry_price * tr["side"]
            raw_move[-1] = (tr["exit_price"] - entry_price) / entry_price * tr["side"]  # exact exit fill
        sub_equity = running_eq * (1.0 + raw_move * tr["notional"])
        sub_equity[-1] -= running_eq * tr.get("fee_notional", 0.0)
        wi = window.index
        mask = wi.isin(idx)
        eq.loc[wi[mask]] = sub_equity[mask]
        running_eq = float(sub_equity[-1])
    eq = eq.ffill().fillna(1.0)
    return eq


def summarize_equity(eq: pd.Series) -> dict:
    peak = eq.cummax()
    mdd = (eq / peak - 1.0).min()
    return {"pnl_pct": (eq.iloc[-1] - 1.0) * 100, "mdd_pct": mdd * 100}


def sanity_check_sigma3_reproduction(tape, prices, start, end, label: str) -> list[dict]:
    """sigma3_trades() is a reimplementation of run_sigma6_regime_trend_20260705.backtest() (with
    reg_mode='none' baked in) that additionally records entry/exit fill prices -- verify the
    resulting equity path reproduces the ORIGINAL function's own aggregate pnl/mdd/trade-count for
    the identical config before trusting it for anything downstream."""
    from run_sigma6_regime_trend_20260705 import backtest as orig_backtest
    base = dict(margin=0.30, trail_atr=5.0, min_profit_atr=2.0, max_hold=144, cooldown=3)
    orig = orig_backtest(tape, leverage=3.0, sl_atr=1.5, reg_mode="none", reg_thr=0.34, stab_thr=0.0,
                          fee_mult=1.0, start=start, end=end, **base)
    trades = sigma3_trades(tape, leverage=3.0, margin=0.30, trail_atr=5.0, sl_atr=1.5,
                            min_profit_atr=2.0, max_hold=144, cooldown=3, start=start, end=end)
    assert len(trades) == orig["trades"], f"Sigma3-1h trade count mismatch ({label}): {len(trades)} vs {orig['trades']}"
    eq = build_leg_equity_path(trades, prices, start, end, use_ledger_trade_return=False)
    recon = summarize_equity(eq)
    pnl_diff = recon["pnl_pct"] - orig["pnl"]
    print(f"[sanity check {label}] Sigma3-1h reimpl trade count matches original backtest(): {len(trades)} == {orig['trades']} (orig pnl={orig['pnl']:.2f}% mdd={orig['mdd']:.2f}%)")
    print(f"[sanity check {label}] Sigma3-1h equity-path reconstruction pnl={recon['pnl_pct']:+.2f}% mdd={recon['mdd_pct']:.2f}% (pnl diff vs orig={pnl_diff:+.4f}pp)")
    assert abs(pnl_diff) < 0.5, f"Sigma3-1h reconstruction does not reproduce original backtest() pnl ({label}): diff={pnl_diff}pp"
    return trades


def sanity_check_omega_reproduction(om_trades: list[dict], prices: pd.Series, start, end, label: str) -> pd.Series:
    """Ground truth: sequential trade-return compounding straight from the saved ledger (boundary
    trades truncated to `end` on real prices). Compares against build_leg_equity_path()'s
    anchored-shape reconstruction, which should match closely since both use the SAME
    ledger-authoritative trade_return as the compounding driver -- the only difference is that the
    equity-path version also carries an (approximate, real-price-shaped) intrabar path for MDD."""
    cash, peak, mdd = 1.0, 1.0, 0.0
    for tr in sorted(om_trades, key=lambda t: t["entry_timestamp"]):
        target = tr["trade_return"]
        if tr["exit_timestamp"] > end:
            window = prices.loc[tr["entry_timestamp"]:end]
            if len(window) >= 2:
                c = window["close"].to_numpy(np.float64)
                target = (c[-1] - c[0]) / c[0] * tr["side"] * tr["notional"]
        cash *= (1 + target)
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1)
    ledger_pnl, ledger_mdd = (cash - 1) * 100, mdd * 100
    eq = build_leg_equity_path(om_trades, prices, start, end, use_ledger_trade_return=True)
    recon = summarize_equity(eq)
    pnl_diff = recon["pnl_pct"] - ledger_pnl
    print(f"[sanity check {label}] Omega4.6.1 ledger trade-sequential (ground truth): pnl={ledger_pnl:+.2f}% mdd={ledger_mdd:.2f}%")
    print(f"[sanity check {label}] Omega4.6.1 equity-path reconstruction: pnl={recon['pnl_pct']:+.2f}% mdd={recon['mdd_pct']:.2f}% (pnl diff={pnl_diff:+.4f}pp)")
    assert abs(pnl_diff) < 0.05, f"Omega4.6.1 equity-path reconstruction does not reproduce the ledger PnL ({label}): diff={pnl_diff}pp"
    return eq


def run_window(label, start, end, prices, omega_ledger_path):
    raw = load_tape_with_regime()
    tape = v2.apply_quality_threshold(raw, 0.60)
    s3_trades = sanity_check_sigma3_reproduction(tape, prices, start, end, label)
    om_trades = omega_trades_from_ledger(omega_ledger_path, start, end)
    eq_a = sanity_check_omega_reproduction(om_trades, prices, start, end, label)  # Omega4.6.1 alone
    eq_b = build_leg_equity_path(s3_trades, prices, start, end, use_ledger_trade_return=False)  # Sigma3-1h alone
    eq_ab = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)  # additive dollar-PnL combination, independent sleeves

    sa, sb, sab = summarize_equity(eq_a), summarize_equity(eq_b), summarize_equity(eq_ab)
    print(f"\n=== {label} {start.date()}..{end.date()} ===")
    print(f"Omega4.6.1 alone   : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(om_trades)}")
    print(f"Sigma3-1h alone     : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(s3_trades)}")
    print(f"Combined portfolio  : pnl={sab['pnl_pct']:+.2f}% mdd={sab['mdd_pct']:.2f}%")
    print(f"  MDD change vs Omega alone: {sab['mdd_pct'] - sa['mdd_pct']:+.2f}pp "
          f"({'WORSE' if sab['mdd_pct'] < sa['mdd_pct'] else 'better'})")
    print(f"  PnL change vs Omega alone: {sab['pnl_pct'] - sa['pnl_pct']:+.2f}pp")
    return {"label": label, "omega_pnl": sa["pnl_pct"], "omega_mdd": sa["mdd_pct"],
            "sigma3_pnl": sb["pnl_pct"], "sigma3_mdd": sb["mdd_pct"],
            "combined_pnl": sab["pnl_pct"], "combined_mdd": sab["mdd_pct"]}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices()
    rows = [
        run_window("VAL", VAL_START, VAL_END, prices, OMEGA_VAL_LEDGER),
        run_window("OOS", OOS_START, OOS_END, prices, OMEGA_OOS_LEDGER),
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "joint_portfolio_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
