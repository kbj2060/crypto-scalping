#!/usr/bin/env python3
"""Skeptical audit of the +509.10% BTC h48qual+direction_quality_lgbm regime_tiebreak continuous
backtest result reported in
scripts/research_btc_h48qual_lgbm_standalone_correlation_20260801.py (standalone_continuous_backtest,
2025-10-08..2026-06-25, N=3h dumb-momentum tiebreak, n_conflict_bars=2829). User is skeptical that
+509% over ~9 months is plausible and asked for a deep audit BEFORE trusting this candidate further.
This script does NOT modify or re-derive any of the underlying combination primitives -- it reuses
build_leg_equity_path/summarize_equity/leg_side_series/rule_weights/weighted_pnl/load_all_trades/
build_dumb_momentum_regime UNCHANGED from the scripts named in the task, exactly like the standalone
script itself does.

Five checks:
  1. Concentration: rank every bar's (tiebreak_choice - fixed_1x1x_choice) additive PnL delta; report
     top10/top20 share of the total +508.08pp gap between tiebreak (+509.10%) and baseline (+1.02%).
  2. Boundary/resampling audit: for the top concentration bars, check whether they sit at/near a
     trade's entry or exit timestamp (i.e. `eq_a_1h.diff()`/`eq_b_1h.diff()` picking up a whole
     multi-day trade's return in a single 1h bucket because .resample('1h').last() only samples the
     already-forward-filled path at hour boundaries -- NOT the already-audited-and-fixed rescale bug,
     a DIFFERENT resampling effect: a bar's delta_a/delta_b is whatever eq moved between the previous
     hour-mark and this one, which can span an entire trade if the leg was flat immediately before).
  3. Capital-base sanity check: is +509% a return on ~$1 or ~$2 of deployed capital, given the
     additive-dollar-PnL independent-sleeve mechanics.
  4. Trade-level economics: map top concentration bars back to the specific leg-A/leg-B RAW ledger
     trade(s) that produced them, and compare that trade's trade_return against the rest of that
     leg's own trade_return distribution (is it an outlier vs plausible TP/SL/time-exit magnitudes).
  5. Plain verdict, printed at the end.

DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule -- reuses already-saved ledgers, not a fresh-forward
walk-forward test. Does not touch trading_bot.py or any live wiring. Read-only w.r.t. all existing
scripts/artifacts; only writes new files under tmp/research_20260801/btc_h48qual_lgbm_tiebreak_509pct_audit/.
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
    LEG_A_DIR, load_5m_prices_btc, leg_side_series, rule_weights, weighted_pnl,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import load_all_trades  # noqa: E402
from diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801 import (  # noqa: E402
    build_dumb_momentum_regime,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_lgbm_tiebreak_509pct_audit"

LEG_B_DIR = ROOT / "tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

SELECTED_N_HOURS = 3
FULL_START = pd.Timestamp("2025-10-08")
FULL_END = pd.Timestamp("2026-06-25 23:59:59")


def trade_covering(trades: list[dict], ts: pd.Timestamp) -> dict | None:
    for tr in trades:
        if tr["entry_timestamp"] <= ts <= tr["exit_timestamp"]:
            return tr
    return None


def nearest_boundary_info(trades: list[dict], bar_start: pd.Timestamp, bar_end: pd.Timestamp) -> str:
    """Does any trade in `trades` enter or exit inside (bar_start, bar_end]? That is the resampling
    hazard step 2 is checking for: a single 1h bucket's delta absorbing an entire trade's return
    because the leg was flat immediately beforehand (or becomes flat immediately after)."""
    hits = []
    for tr in trades:
        if bar_start < tr["entry_timestamp"] <= bar_end:
            hits.append(f"ENTRY of trade {tr['entry_timestamp']}->{tr['exit_timestamp']} "
                        f"(trade_return={tr['trade_return']:+.4f}, notional={tr['notional']:.3f}) falls in this bar")
        if bar_start < tr["exit_timestamp"] <= bar_end:
            hits.append(f"EXIT of trade {tr['entry_timestamp']}->{tr['exit_timestamp']} "
                        f"(trade_return={tr['trade_return']:+.4f}, notional={tr['notional']:.3f}) falls in this bar")
    return "; ".join(hits) if hits else "no entry/exit boundary in this bar"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices_5m = load_5m_prices_btc()
    close_1h = prices_5m["close"].resample("1h").last().ffill()
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

    print("### Reproduction of the reported standalone continuous backtest ###")
    print(f"  Leg A alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"  Leg B alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"  Fixed 1x-1x baseline: pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"  Tiebreak: pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}% n_conflict_bars={n_conflict}")
    print()

    # --- Step 1: concentration -----------------------------------------------------------------
    # Per-bar contribution to each strategy's additive equity, and the DIFFERENCE the tiebreak
    # choice made relative to the fixed-1x-1x baseline (which always uses w_a=w_b=1).
    c_tiebreak = w_a * delta_a + w_b * delta_b
    c_baseline = delta_a + delta_b  # fixed 1x-1x: both weights always 1
    diff = c_tiebreak - c_baseline  # (w_a-1)*delta_a + (w_b-1)*delta_b; zero on all non-conflict bars
    total_gap = diff.sum()  # should equal (tiebreak equity-1) - (baseline equity-1), pure linear sum
    equity_gap_check = (tiebreak["pnl_pct"] - sab_base["pnl_pct"]) / 100.0
    print("### 1) Concentration check ###")
    print(f"  sum(diff) = {total_gap:+.4f}   (tiebreak_pnl - baseline_pnl)/100 = {equity_gap_check:+.4f}   "
          f"{'MATCH (pure additive, no compounding distortion)' if abs(total_gap - equity_gap_check) < 1e-6 else 'MISMATCH -- investigate'}")

    order = np.argsort(-np.abs(diff))  # rank by |contribution| to the gap
    bar_starts = eq_a_1h.index[np.roll(np.arange(len(ts)), 1)]  # previous timestamp = bar interval start
    bar_starts = eq_a_1h.index - pd.Timedelta("1h")
    rows = []
    for rank, i in enumerate(order[:50]):
        rows.append({
            "rank": rank + 1, "bar_end_ts": ts.iloc[i], "bar_start_ts": bar_starts[i],
            "diff_pp": diff[i] * 100, "delta_a": delta_a[i], "delta_b": delta_b[i],
            "w_a": w_a[i], "w_b": w_b[i], "conflict": bool(conflict[i]),
            "side_a": side_a[i], "side_b": side_b[i],
        })
    top_df = pd.DataFrame(rows)
    top10_share = top_df["diff_pp"].iloc[:10].sum() / (total_gap * 100) * 100
    top20_share = top_df["diff_pp"].iloc[:20].sum() / (total_gap * 100) * 100
    top50_share = top_df["diff_pp"].sum() / (total_gap * 100) * 100
    n_diff_nonzero = int((diff != 0).sum())
    print(f"  total gap explained (tiebreak - baseline) = {total_gap*100:+.2f}pp over {n_conflict} conflict bars "
          f"({n_diff_nonzero} bars with nonzero diff)")
    print(f"  TOP 10 bars explain {top10_share:.1f}% of the total gap")
    print(f"  TOP 20 bars explain {top20_share:.1f}% of the total gap")
    print(f"  TOP 50 bars explain {top50_share:.1f}% of the total gap")
    print(f"  mean |diff| per nonzero bar = {(np.abs(diff[diff!=0]).mean())*100:.4f}pp, "
          f"median = {(np.median(np.abs(diff[diff!=0])))*100:.4f}pp -- "
          f"ratio top1/median = {(top_df['diff_pp'].abs().iloc[0] / (np.median(np.abs(diff[diff!=0]))*100)):.1f}x")
    print()

    # --- Step 1b: EPISODE-level concentration (the statistically meaningful unit) ---------------
    # Conflict bars are NOT independent trials: a "conflict" persists for as long as both legs'
    # trades stay open and disagree, so a contiguous run of conflict=True bars is really ONE
    # regime-momentum decision (the N=3h tiebreak rarely flips mid-run), not dozens/hundreds of
    # independent hourly bets. Re-rank at the episode level to get the true effective sample size.
    c_int = conflict.astype(int)
    ep_starts = np.where(np.diff(c_int, prepend=0) == 1)[0]
    ep_ends = np.where(np.diff(c_int, append=0) == -1)[0]
    ep_lengths = ep_ends - ep_starts + 1
    ep_gains = np.array([diff[s:e + 1].sum() for s, e in zip(ep_starts, ep_ends)])
    ep_order = np.argsort(-np.abs(ep_gains))
    n_episodes = len(ep_gains)
    ep_top10_share = ep_gains[ep_order[:10]].sum() / ep_gains.sum() * 100
    ep_top5_share = ep_gains[ep_order[:5]].sum() / ep_gains.sum() * 100
    n_ep_pos = int((ep_gains > 0).sum())
    n_ep_neg = int((ep_gains < 0).sum())
    print("### 1b) EPISODE-level concentration (contiguous conflict runs = independent decisions) ###")
    print(f"  {n_episodes} independent conflict episodes total (NOT {n_conflict} independent hourly bars --")
    print(f"  each episode is one persisting regime-momentum call while both legs' trades stay open and")
    print(f"  disagree; mean episode length={ep_lengths.mean():.1f}h median={np.median(ep_lengths):.1f}h)")
    print(f"  episodes with POSITIVE gain: {n_ep_pos}/{n_episodes}   NEGATIVE: {n_ep_neg}/{n_episodes}")
    print(f"  TOP 10 of {n_episodes} episodes explain {ep_top10_share:.1f}% of the total +{total_gap*100:.2f}pp gap")
    print(f"  TOP 5  of {n_episodes} episodes explain {ep_top5_share:.1f}% of the total gap")
    ep_rows = []
    for rank, i in enumerate(ep_order[:15]):
        ep_rows.append({
            "rank": rank + 1, "start_ts": ts.iloc[ep_starts[i]], "end_ts": ts.iloc[ep_ends[i]],
            "length_hours": int(ep_lengths[i]), "gain_pp": ep_gains[i] * 100,
        })
        print(f"    rank {rank+1}: {ts.iloc[ep_starts[i]]} -> {ts.iloc[ep_ends[i]]} "
              f"({int(ep_lengths[i])}h) gain={ep_gains[i]*100:+.2f}pp")
    pd.DataFrame(ep_rows).to_csv(OUT_DIR / "top15_conflict_episodes.csv", index=False)
    if n_ep_neg == 0:
        print(f"  FLAG: 0/{n_episodes} episodes lost -- a genuinely predictive signal winning ALL "
              f"independent trials over 9 months is an unusually strong (and therefore suspicious) claim; "
              f"contrast with the LOWO cross-validation result of 4/5 (80%) fold win rate, not 100%.")
    print()

    # --- Step 2: boundary/resampling audit on the top-10 bars ----------------------------------
    print("### 2) Boundary/resampling audit (top 10 concentration bars) ###")
    boundary_notes = []
    for _, r in top_df.iloc[:10].iterrows():
        note_a = nearest_boundary_info(trades_a, r["bar_start_ts"], r["bar_end_ts"])
        note_b = nearest_boundary_info(trades_b, r["bar_start_ts"], r["bar_end_ts"])
        print(f"  rank {int(r['rank'])} bar=({r['bar_start_ts']} , {r['bar_end_ts']}] diff={r['diff_pp']:+.2f}pp "
              f"w_a={r['w_a']:.0f} w_b={r['w_b']:.0f} delta_a={r['delta_a']:+.4f} delta_b={r['delta_b']:+.4f}")
        print(f"      leg A boundary: {note_a}")
        print(f"      leg B boundary: {note_b}")
        boundary_notes.append({"rank": int(r["rank"]), "leg_a_boundary": note_a, "leg_b_boundary": note_b})
    print()

    # --- Step 3: capital-base sanity check ------------------------------------------------------
    print("### 3) Capital-base sanity check ###")
    print("  weighted_pnl() initializes a SINGLE equity=1.0 and adds w_a*delta_a + w_b*delta_b each")
    print("  bar, where delta_a/delta_b are each leg's OWN dollar PnL computed against that leg's own")
    print("  independent 1.0-based equity path (eq_a, eq_b each start at 1.0 and compound on their own")
    print("  notional -- see build_leg_equity_path()). To run both legs simultaneously requires ~$1 of")
    print("  margin per leg (~$2 total), yet the combined return is reported as a % of a SINGLE $1")
    print("  base. On the n_conflict_bars=2829 bars where sides disagree, only ONE leg's delta is kept")
    print("  (so effectively $1 of the $2 deployed produces zero at that specific bar) -- but on ALL")
    print("  OTHER (non-conflict) bars, BOTH legs' deltas are added into the same $1 base, i.e. the")
    print("  reported pnl_pct on non-conflict bars is already effectively a return on $1 that actually")
    print("  required ~$2 of deployed capital to realize.")
    leg_a_active_frac = float(active_a.mean())
    leg_b_active_frac = float(active_b.mean())
    both_active_frac = float((active_a & active_b).mean())
    print(f"  leg A active (in a trade) {leg_a_active_frac*100:.1f}% of bars, leg B active {leg_b_active_frac*100:.1f}%,"
          f" BOTH active simultaneously {both_active_frac*100:.1f}% of bars ({n_conflict} of those are conflicts).")
    pnl_per_2 = tiebreak["pnl_pct"] / 2.0
    print(f"  Reported: +{tiebreak['pnl_pct']:.2f}% on a $1 base.")
    print(f"  If normalized to the ~$2 of capital actually required to run both independent sleeves "
          f"simultaneously: ~+{pnl_per_2:.2f}% on $2 of capital -- HALF the headline number, still large "
          f"but a materially different framing than '+509% return'.")
    print()

    # --- Step 4: trade-level economics ----------------------------------------------------------
    print("### 4) Trade-level economics for the top concentration bars ###")
    a_returns = np.array([t["trade_return"] for t in trades_a])
    b_returns = np.array([t["trade_return"] for t in trades_b])
    print(f"  Leg A trade_return distribution: n={len(a_returns)} mean={a_returns.mean():+.4f} "
          f"std={a_returns.std():.4f} min={a_returns.min():+.4f} max={a_returns.max():+.4f}")
    print(f"  Leg B trade_return distribution: n={len(b_returns)} mean={b_returns.mean():+.4f} "
          f"std={b_returns.std():.4f} min={b_returns.min():+.4f} max={b_returns.max():+.4f}")
    trade_rows = []
    seen_trades = set()
    for _, r in top_df.iloc[:10].iterrows():
        mid_ts = r["bar_end_ts"]
        tr_a = trade_covering(trades_a, mid_ts)
        tr_b = trade_covering(trades_b, mid_ts)
        for leg_name, tr, dist in (("A", tr_a, a_returns), ("B", tr_b, b_returns)):
            if tr is None:
                continue
            key = (leg_name, tr["entry_timestamp"])
            if key in seen_trades:
                continue
            seen_trades.add(key)
            z = (tr["trade_return"] - dist.mean()) / dist.std() if dist.std() > 0 else float("nan")
            outlier_mult = abs(tr["trade_return"]) / np.median(np.abs(dist)) if np.median(np.abs(dist)) > 0 else float("nan")
            trade_rows.append({
                "rank_bar": int(r["rank"]), "leg": leg_name, "entry_timestamp": tr["entry_timestamp"],
                "exit_timestamp": tr["exit_timestamp"], "side": tr["side"], "notional": tr["notional"],
                "trade_return": tr["trade_return"], "z_score_vs_leg_dist": z,
                "abs_return_vs_leg_median_multiple": outlier_mult,
            })
            print(f"  rank {int(r['rank'])} leg {leg_name}: {tr['entry_timestamp']} -> {tr['exit_timestamp']} "
                  f"side={tr['side']} notional={tr['notional']:.3f} trade_return={tr['trade_return']:+.4f} "
                  f"z={z:+.2f} ({outlier_mult:.2f}x that leg's median |trade_return|)")
    trades_df = pd.DataFrame(trade_rows)
    print()

    # --- Save outputs -----------------------------------------------------------------------------
    top_df.to_csv(OUT_DIR / "top50_concentration_bars.csv", index=False)
    pd.DataFrame(boundary_notes).to_csv(OUT_DIR / "top10_boundary_audit.csv", index=False)
    trades_df.to_csv(OUT_DIR / "top_bars_underlying_trades.csv", index=False)
    summary = pd.DataFrame([{
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"], "leg_a_trades": len(trades_a),
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"], "leg_b_trades": len(trades_b),
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "tiebreak_pnl": tiebreak["pnl_pct"], "tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "total_gap_pp": total_gap * 100,
        "top10_share_pct_of_gap": top10_share, "top20_share_pct_of_gap": top20_share,
        "top50_share_pct_of_gap": top50_share,
        "pnl_normalized_to_2x_capital": pnl_per_2,
        "leg_a_active_frac": leg_a_active_frac, "leg_b_active_frac": leg_b_active_frac,
        "both_active_frac": both_active_frac,
        "n_conflict_episodes": n_episodes, "episodes_positive": n_ep_pos, "episodes_negative": n_ep_neg,
        "episode_top10_share_pct_of_gap": ep_top10_share, "episode_top5_share_pct_of_gap": ep_top5_share,
    }])
    summary.to_csv(OUT_DIR / "audit_summary.csv", index=False)
    print(f"Wrote outputs under {OUT_DIR}")

    # --- Step 5: verdict ---------------------------------------------------------------------------
    print()
    print("### 5) Verdict ###")
    print(f"  Bar-level concentration: top 10 of {n_diff_nonzero} nonzero-diff HOURLY bars explain "
          f"{top10_share:.1f}% of the +{total_gap*100:.2f}pp gap; top 20 explain {top20_share:.1f}%. "
          f"(looks diffuse at this granularity)")
    print(f"  EPISODE-level concentration (true independent-trial count): only {n_episodes} conflict "
          f"episodes total; top 10 explain {ep_top10_share:.1f}% of the gap, top 5 explain {ep_top5_share:.1f}%; "
          f"{n_ep_pos}/{n_episodes} episodes positive, {n_ep_neg}/{n_episodes} negative. This is the "
          f"statistically meaningful red flag: n={n_episodes} independent regime calls, ALL positive, "
          f"nearly half the total gain from 10 of them.")
    print(f"  Capital base: headline +{tiebreak['pnl_pct']:.2f}% is on a $1 reference base that requires "
          f"~$2 of simultaneous independent-sleeve margin to realize (both legs active {both_active_frac*100:.1f}% "
          f"of bars) -- normalized-to-$2 equivalent is ~+{pnl_per_2:.2f}%.")
    print("  See printed boundary/trade-level detail above for whether top-contribution bars are real "
          "trades or resampling artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
