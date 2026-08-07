#!/usr/bin/env python3
"""DIAGNOSTIC ONLY -- does Sigma3-1h (no regime filter) provide real diversification against
Omega4.6.1 on ETH, or do they mostly win/lose on the same days? Same daily-allocated-return +
block-bootstrap methodology as tmp/research_20260728/sigma6_omega461_correlation_check.py (that
check was inconclusive: only 111 days/26 trades of overlap, sign flipped with leverage). This
version uses the FULL canonical VAL+OOS window (6 months) and the trade lists already validated in
scripts/research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py (sanity-checked to reproduce
both legs' own known PnL numbers), so the correlation estimate itself rests on a firmer base.

Per CLAUDE.md this is diagnostic-only: a correlation/overlap check between two already-scored
strategies, not a promotion decision.
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
from research_eth_sigma3_1h_omega461_joint_portfolio_20260731 import (  # noqa: E402
    sigma3_trades, omega_trades_from_ledger,
    VAL_START, VAL_END, OOS_START, OOS_END, OMEGA_VAL_LEDGER, OMEGA_OOS_LEDGER,
)


def daily_allocated_returns(trades: list[dict], day_index: pd.DatetimeIndex, ret_key: str, notional_key: str | None) -> pd.Series:
    daily = pd.Series(0.0, index=day_index)
    for t in trades:
        e, x = pd.Timestamp(t["entry_timestamp"]).floor("D"), pd.Timestamp(t["exit_timestamp"]).floor("D")
        days = pd.date_range(e, x, freq="D")
        days = days[days.isin(day_index)]
        if len(days) == 0:
            continue
        ret = t[ret_key] if notional_key is None else t[ret_key] * t[notional_key]
        daily.loc[days] += ret / len(days)
    return daily


def occupancy_mask(trades: list[dict], day_index: pd.DatetimeIndex) -> pd.Series:
    mask = pd.Series(False, index=day_index)
    for t in trades:
        e, x = pd.Timestamp(t["entry_timestamp"]).floor("D"), pd.Timestamp(t["exit_timestamp"]).floor("D")
        days = pd.date_range(e, x, freq="D")
        mask.loc[mask.index.isin(days)] = True
    return mask


def block_bootstrap_corr_ci(a: pd.Series, b: pd.Series, *, block_days: int = 14, n_boot: int = 5000, seed: int = 20260731) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    a_arr, b_arr = a.to_numpy(), b.to_numpy()
    n_blocks = int(np.ceil(n / block_days))
    boots = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_days) % n for s in starts])[:n]
        aa, bb = a_arr[idx], b_arr[idx]
        boots[k] = np.nan if (aa.std() < 1e-12 or bb.std() < 1e-12) else np.corrcoef(aa, bb)[0, 1]
    boots = boots[~np.isnan(boots)]
    return {"median": float(np.median(boots)), "p05": float(np.quantile(boots, 0.05)),
            "p95": float(np.quantile(boots, 0.95)), "prob_positive": float((boots > 0).mean()),
            "n_boot_valid": int(len(boots))}


def analyze(label, start, end, s3_trades, om_trades):
    """s3_trades / om_trades are already-normalized dicts with entry_timestamp/exit_timestamp/trade_return."""
    day_index = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    s3_daily = daily_allocated_returns(s3_trades, day_index, "trade_return", None)
    om_daily = daily_allocated_returns(om_trades, day_index, "trade_return", None)

    corr = s3_daily.corr(om_daily)
    n_nonzero_both = int(((s3_daily != 0) & (om_daily != 0)).sum())
    ci = block_bootstrap_corr_ci(s3_daily, om_daily)

    s3_occ = occupancy_mask(s3_trades, day_index)
    om_occ = occupancy_mask(om_trades, day_index)
    both_occ = s3_occ & om_occ
    overlap_days = int(both_occ.sum())

    same_sign = float("nan")
    if overlap_days > 0:
        s3_sub, om_sub = s3_daily[both_occ], om_daily[both_occ]
        valid = (s3_sub != 0) & (om_sub != 0)
        if valid.sum() > 0:
            same_sign = float((np.sign(s3_sub[valid]) == np.sign(om_sub[valid])).mean())

    print(f"===== {label} {start.date()}..{end.date()} =====")
    print(f"  calendar days: {len(day_index)}   sigma3-1h trades: {len(s3_trades)}   omega461 trades: {len(om_trades)}")
    print(f"  sigma3-1h days-in-position: {int(s3_occ.sum())}/{len(day_index)} ({100*s3_occ.mean():.1f}%)")
    print(f"  omega461 days-in-position: {int(om_occ.sum())}/{len(day_index)} ({100*om_occ.mean():.1f}%)")
    print(f"  BOTH-in-position days: {overlap_days} ({100*overlap_days/len(day_index):.1f}% of window)")
    print(f"  daily-allocated-return Pearson correlation: {corr:.3f}  (n_days_both_nonzero={n_nonzero_both})")
    print(f"  block-bootstrap (14d blocks, n=5000) 90% CI: [{ci['p05']:.3f}, {ci['p95']:.3f}]  median={ci['median']:.3f}  P(corr>0)={ci['prob_positive']:.2f}")
    print(f"  sign agreement on both-occupied days: {same_sign if same_sign == same_sign else 'n/a'}")
    print()


def main() -> int:
    print("Diagnostic Sigma3-1h vs Omega4.6.1 correlation check -- NOT a promotion claim.\n")
    raw = load_tape_with_regime()
    tape = v2.apply_quality_threshold(raw, 0.60)

    for label, start, end, omega_ledger in [
        ("VAL", VAL_START, VAL_END, OMEGA_VAL_LEDGER),
        ("OOS", OOS_START, OOS_END, OMEGA_OOS_LEDGER),
    ]:
        s3_trades = sigma3_trades(tape, leverage=3.0, margin=0.30, trail_atr=5.0, sl_atr=1.5,
                                   min_profit_atr=2.0, max_hold=144, cooldown=3, start=start, end=end)
        # sigma3_trades() records entry_price/exit_price, not a return -- compute trade_return here
        for t in s3_trades:
            t["notional_ret"] = ((t["exit_price"] - t["entry_price"]) / t["entry_price"] * t["side"] * t["notional"]
                                  - t["fee_notional"])
        om_trades = omega_trades_from_ledger(omega_ledger, start, end)
        s3_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                        "trade_return": t["notional_ret"]} for t in s3_trades]
        om_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                        "trade_return": t["trade_return"]} for t in om_trades]
        analyze(label, start, end, s3_for_corr, om_for_corr)

    # also pooled VAL+OOS (single continuous window) for a longer-sample estimate
    s3_all = sigma3_trades(tape, leverage=3.0, margin=0.30, trail_atr=5.0, sl_atr=1.5,
                            min_profit_atr=2.0, max_hold=144, cooldown=3, start=VAL_START, end=OOS_END)
    for t in s3_all:
        t["notional_ret"] = ((t["exit_price"] - t["entry_price"]) / t["entry_price"] * t["side"] * t["notional"]
                              - t["fee_notional"])
    om_all_val = omega_trades_from_ledger(OMEGA_VAL_LEDGER, VAL_START, VAL_END)
    om_all_oos = omega_trades_from_ledger(OMEGA_OOS_LEDGER, OOS_START, OOS_END)
    om_all = om_all_val + om_all_oos
    s3_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                    "trade_return": t["notional_ret"]} for t in s3_all]
    om_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                    "trade_return": t["trade_return"]} for t in om_all]
    analyze("POOLED VAL+OOS", VAL_START, OOS_END, s3_for_corr, om_for_corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
