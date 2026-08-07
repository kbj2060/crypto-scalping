#!/usr/bin/env python3
"""Follow-up checks for the BTC h48qual + direction_quality_lgbm regime_tiebreak candidate that
survived genuine leave-one-window-out (LOWO) cross-validation at 4/5 held-out folds with a STABLE
selected dumb-momentum lookback of N=3h in every fold
(scripts/research_btc_h48qual_direction_quality_lgbm_lowo_20260801.py,
tmp/research_20260801/btc_h48qual_direction_quality_lgbm_lowo/leave_one_window_out_results.csv).
Mirrors the 3 follow-up checks done for the equivalent ETH candidate
(scripts/research_eth_sigma6_walkforward_omega461_correlation_20260801.py), adapted because BTC has
NO live-wired strategy (grep of trading_bot.py/live config confirms neither h48qual, Sigma9, nor
direction_quality_lgbm is live-wired -- verified at the top of main() below, not just asserted):

1. Standalone continuous backtest, N=3h dumb-momentum tiebreak, over the FULL overlap range
   2025-10-08..2026-06-25 (no fold-slicing) -- Leg A alone, Leg B alone, fixed 1x-1x baseline,
   tiebreak combination.
2. Correlation/diversification: Leg A vs Leg B daily-allocated-return correlation (no live BTC
   strategy exists to compare against, so this is leg-vs-leg instead of leg-vs-live), reusing the
   UNCHANGED block-bootstrap methodology from research_eth_sigma3_1h_omega461_correlation_20260731.py
   (daily_allocated_returns, occupancy_mask, block_bootstrap_corr_ci).
3. Leg-A-as-phantom-live-baseline framing: since Leg A (h48qual) is the only BTC leg ever documented
   as research_positive_signal_not_live_wired (the closest thing to a live-candidate status BTC has,
   see docs/model_contracts/btc_omega4_6_1_full_stack_20260708_contract.md), treat it as the
   "baseline portfolio" and report the tiebreak combination's PnL/MDD delta vs it, same pp-framing
   as the ETH combined-portfolio check.

All leg-loading/equity-path/regime_tiebreak primitives (omega_trades_from_ledger,
build_leg_equity_path, summarize_equity, leg_side_series, rule_weights, weighted_pnl,
build_dumb_momentum_regime, load_all_trades, load_5m_prices_btc) are UNCHANGED reuse of
research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py,
research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py,
research_btc_tau1_cryptomamba_tiebreak_20260801.py, and
diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py -- nothing about the underlying
combination math is re-derived here, matching the LOWO script's own discipline.

DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule: this is a validation-methodology follow-up on
already-saved ledgers, not a Fresh-Forward bar-by-bar walk-forward test. Neither leg is live-wired.
Does not touch trading_bot.py or any live wiring -- this script only reads it (grep) to confirm that.
"""
from __future__ import annotations

import subprocess
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
    LEG_A_DIR, load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import load_all_trades  # noqa: E402
from diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801 import (  # noqa: E402
    build_dumb_momentum_regime,
)
from research_eth_sigma3_1h_omega461_correlation_20260731 import (  # noqa: E402
    daily_allocated_returns, occupancy_mask, block_bootstrap_corr_ci,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_lgbm_standalone_correlation"

LEG_B_DIR = ROOT / "tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

SELECTED_N_HOURS = 3  # LOWO-selected value: same N chosen in every one of 5 folds (see docstring)
FULL_START = pd.Timestamp("2025-10-08")
FULL_END = pd.Timestamp("2026-06-25 23:59:59")


def confirm_btc_not_live_wired() -> None:
    """Grep trading_bot.py (and any live config) for the candidate model ids, to verify the claim
    in the task/docstring rather than take it on faith."""
    needles = ["btc_final_scale_map_20260708", "btc_v2_direction_quality_lgbm_20260714",
               "btc_v2_regime_trendscan_hgb_20260714", "h48qual"]
    hits = []
    for needle in needles:
        r = subprocess.run(["grep", "-rn", needle, str(ROOT / "trading_bot.py")],
                            capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            hits.append((needle, r.stdout.strip()))
    print("### Live-wiring confirmation (grep trading_bot.py) ###")
    if hits:
        print("  UNEXPECTED: found references -- BTC candidate may already be partially wired:")
        for needle, out in hits:
            print(f"    {needle}: {out}")
    else:
        print("  Confirmed: none of h48qual / direction_quality_lgbm / sigma9-regime-trendscan "
              "model ids appear in trading_bot.py. BTC has no live-wired strategy today.")
    print()


def standalone_continuous_backtest(prices_5m, close_1h) -> dict:
    print(f"### 1) Standalone continuous backtest, N={SELECTED_N_HOURS}h, "
          f"{FULL_START.date()}..{FULL_END.date()} (no fold-slicing) ###")
    regime = build_dumb_momentum_regime(close_1h, SELECTED_N_HOURS)

    trades_a = load_all_trades(LEG_A_LEDGERS, FULL_START, FULL_END)
    trades_b = load_all_trades(LEG_B_LEDGERS, FULL_START, FULL_END)

    eq_a = build_leg_equity_path(trades_a, prices_5m, FULL_START, FULL_END, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices_5m, FULL_START, FULL_END, use_ledger_trade_return=True)
    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    reg = regime[(regime["timestamp"] >= FULL_START) & (regime["timestamp"] <= FULL_END)]
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

    print(f"  Leg A (h48qual) alone            : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"  Leg B (direction_quality_lgbm)   : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"  Fixed 1x-1x baseline             : pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"  Dumb-momentum tiebreak (N={SELECTED_N_HOURS}h, n_conflict_bars={n_conflict}): "
          f"pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}%")
    print()

    row = {
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"], "leg_a_trades": len(trades_a),
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"], "leg_b_trades": len(trades_b),
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "tiebreak_pnl": tiebreak["pnl_pct"], "tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict,
    }
    return row, trades_a, trades_b


def correlation_check(trades_a: list[dict], trades_b: list[dict]) -> dict:
    print(f"### 2) Leg A vs Leg B correlation/diversification, {FULL_START.date()}..{FULL_END.date()} ###")
    day_index = pd.date_range(FULL_START.floor("D"), FULL_END.floor("D"), freq="D")

    a_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                   "trade_return": t["trade_return"]} for t in trades_a]
    b_for_corr = [{"entry_timestamp": t["entry_timestamp"], "exit_timestamp": t["exit_timestamp"],
                   "trade_return": t["trade_return"]} for t in trades_b]

    a_daily = daily_allocated_returns(a_for_corr, day_index, "trade_return", None)
    b_daily = daily_allocated_returns(b_for_corr, day_index, "trade_return", None)

    corr = a_daily.corr(b_daily)
    n_nonzero_both = int(((a_daily != 0) & (b_daily != 0)).sum())
    ci = block_bootstrap_corr_ci(a_daily, b_daily)

    a_occ = occupancy_mask(a_for_corr, day_index)
    b_occ = occupancy_mask(b_for_corr, day_index)
    both_occ = a_occ & b_occ
    overlap_days = int(both_occ.sum())

    same_sign = float("nan")
    if overlap_days > 0:
        a_sub, b_sub = a_daily[both_occ], b_daily[both_occ]
        valid = (a_sub != 0) & (b_sub != 0)
        if valid.sum() > 0:
            same_sign = float((np.sign(a_sub[valid]) == np.sign(b_sub[valid])).mean())

    print(f"  calendar days: {len(day_index)}   leg_a trades: {len(trades_a)}   leg_b trades: {len(trades_b)}")
    print(f"  leg_a days-in-position: {int(a_occ.sum())}/{len(day_index)} ({100*a_occ.mean():.1f}%)")
    print(f"  leg_b days-in-position: {int(b_occ.sum())}/{len(day_index)} ({100*b_occ.mean():.1f}%)")
    print(f"  BOTH-in-position days: {overlap_days} ({100*overlap_days/len(day_index):.1f}% of window)")
    print(f"  daily-allocated-return Pearson correlation: {corr:.3f}  (n_days_both_nonzero={n_nonzero_both})")
    print(f"  block-bootstrap (14d blocks, n=5000) 90% CI: [{ci['p05']:.3f}, {ci['p95']:.3f}]  "
          f"median={ci['median']:.3f}  P(corr>0)={ci['prob_positive']:.2f}")
    print(f"  sign agreement on both-occupied days: {same_sign if same_sign == same_sign else 'n/a'}")
    print()
    return {"corr": corr, "ci_p05": ci["p05"], "ci_p95": ci["p95"], "prob_positive": ci["prob_positive"],
            "overlap_days": overlap_days, "n_days_both_nonzero": n_nonzero_both, "sign_agreement": same_sign,
            "leg_a_occupancy_pct": 100 * a_occ.mean(), "leg_b_occupancy_pct": 100 * b_occ.mean()}


def leg_a_phantom_baseline_delta(standalone_row: dict) -> dict:
    print("### 3) Tiebreak combination vs Leg A (h48qual) as phantom-live baseline ###")
    print("  (Leg A is the only BTC leg ever documented research_positive_signal_not_live_wired -- "
          "closest thing to a live-candidate status BTC has; BTC itself has NO live-wired strategy.)")
    pnl_delta = standalone_row["tiebreak_pnl"] - standalone_row["leg_a_pnl"]
    mdd_delta = standalone_row["tiebreak_mdd"] - standalone_row["leg_a_mdd"]
    verdict = "BETTER on both axes" if pnl_delta > 0 and mdd_delta > 0 else (
        "worse on at least one axis")
    print(f"  Leg A alone (phantom baseline) : pnl={standalone_row['leg_a_pnl']:+.2f}% mdd={standalone_row['leg_a_mdd']:.2f}%")
    print(f"  Tiebreak combination           : pnl={standalone_row['tiebreak_pnl']:+.2f}% mdd={standalone_row['tiebreak_mdd']:.2f}%")
    print(f"  PnL delta vs Leg A: {pnl_delta:+.2f}pp   MDD delta vs Leg A: {mdd_delta:+.2f}pp  "
          f"({'better' if mdd_delta > 0 else 'WORSE'} MDD)")
    print(f"  Verdict: {verdict}")
    print()
    return {"pnl_delta_vs_leg_a": pnl_delta, "mdd_delta_vs_leg_a": mdd_delta, "verdict": verdict}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    confirm_btc_not_live_wired()

    prices_5m = load_5m_prices_btc()
    close_1h = prices_5m["close"].resample("1h").last().ffill()

    standalone_row, trades_a, trades_b = standalone_continuous_backtest(prices_5m, close_1h)
    corr_row = correlation_check(trades_a, trades_b)
    delta_row = leg_a_phantom_baseline_delta(standalone_row)

    pd.DataFrame([standalone_row]).to_csv(OUT_DIR / "standalone_continuous_backtest.csv", index=False)
    pd.DataFrame([corr_row]).to_csv(OUT_DIR / "leg_a_vs_leg_b_correlation.csv", index=False)
    pd.DataFrame([delta_row]).to_csv(OUT_DIR / "tiebreak_vs_leg_a_phantom_baseline.csv", index=False)
    print(f"Wrote outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
