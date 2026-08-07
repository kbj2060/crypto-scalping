#!/usr/bin/env python3
"""Genuine leave-one-window-out-style walk-forward re-check of the BTC h48qual (Leg A) +
Sigma9-trend-scan-regime-gated (Leg B) regime_tiebreak combination
(research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py), AFTER the 2026-08-02
same-bar-adjacent look-ahead fix to the shared run_window() merge_asof pattern
(project-btc-run-window-merge-point-fixed-20260802.md).

Context: the original joint_portfolio script only tested ONE VAL/OOS split (both windows already
repeatedly inspected in this project's history for each leg separately -- explicitly flagged in its
own docstring as NOT a genuinely blind test and suspected "VAL-overfit artifact"). It was never rerun
after the run_window() bugfix, and no saved evidence of a post-fix run exists locally
(tmp/research_20260801/ does not exist on this machine). A parallel BTC combination attempt with a
DIFFERENT Leg B (direction_quality_lgbm) was re-tested post-fix with genuine 5-fold
leave-one-window-out (LOWO) selection over a hyperparameter grid (dumb-momentum lookback N) and
FAILED (0/5 held-out folds passed) -- see research_btc_h48qual_direction_quality_lgbm_lowo_20260801.py.

Methodology difference from that LGBM LOWO script: the Sigma9 regime_tiebreak rule
(rule_weights() in research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py) has NO
tunable hyperparameter to sweep -- it is a fixed argmax(bull_prob, bear_prob) decision, unlike the
LGBM combination's dumb-momentum lookback-N grid. Per the task instructions ("if there's just one
fixed config, do a straightforward N-fold walk-forward test instead"), this script therefore skips
the config-grid-selection step entirely and just evaluates the ALREADY-FIXED, ALREADY-FROZEN
regime_tiebreak rule independently on K=5 non-overlapping contiguous folds spanning the Leg A/Leg B
ledger overlap range -- each fold is genuinely out-of-sample in the sense that no fold's result
influences any other fold's evaluation (no selection step to leak through), matching this project's
"reproduce before trusting" / Fresh-Forward-adjacent discipline for a single frozen rule.

Overlap range: Leg A ledgers span entry_timestamp 2025-10-01..2026-06-25 (the binding constraint);
Leg B (Sigma9) ledgers span 2025-07-01..2026-06-29 and its regime-probability file spans
2025-07-01..2026-07-12, both comfortably covering Leg A's range. Working range used here is Leg A's
own full range: 2025-10-01..2026-06-25.

Primitives (omega_trades_from_ledger, build_leg_equity_path, summarize_equity, leg_side_series,
rule_weights, weighted_pnl, load_5m_prices_btc, load_regime_probs, PFX, LEG_A_DIR, LEG_B_DIR) are
UNCHANGED reuse of research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py and
research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py -- nothing about the underlying
combination math or the regime-timestamp +1h fix is re-derived here; this script's own run_window()
is copied line-for-line from the (already post-fix) joint_portfolio script, just repointed at
per-fold start/end and combined (validation+oos) ledgers so folds can straddle the VAL/OOS boundary.

DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule: this replays saved ledgers fold-by-fold, it is NOT
a genuinely blind bar-by-bar Fresh-Forward walk-forward test (both legs' ledgers were built and
inspected long before this script existed). Neither leg is live-wired. This checks whether the
regime_tiebreak MECHANISM generalizes across independent sub-periods post-bugfix, not whether BTC
should be promoted.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
    LEG_A_DIR, LEG_B_DIR, PFX, load_5m_prices_btc, load_regime_probs, leg_side_series,
    rule_weights, weighted_pnl,
)

OUT_DIR = ROOT / "tmp/research_20260802/btc_h48qual_sigma9_legb_lowo_verified"

LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

K_FOLDS = 5
THIN_TRADE_THRESHOLD = 5  # min(leg_a_trades, leg_b_trades) below this in a fold => flag as thin

# Leg A's own full ledger range is the binding constraint (Leg B covers this comfortably).
FULL_START = pd.Timestamp("2025-10-01")
FULL_END = pd.Timestamp("2026-06-25 23:59:59")


def load_all_trades(paths, start, end) -> list[dict]:
    trades = []
    for p in paths:
        trades.extend(omega_trades_from_ledger(p, start, end))
    return trades


def build_folds(start: pd.Timestamp, end: pd.Timestamp, k: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """K non-overlapping, roughly-equal contiguous folds spanning [start, end] by calendar day.
    Identical logic to research_btc_h48qual_direction_quality_lgbm_lowo_20260801.build_folds."""
    total_days = (end.normalize() - start.normalize()).days + 1
    base, extra = divmod(total_days, k)
    folds = []
    cursor = start
    for i in range(k):
        n_days = base + (1 if i < extra else 0)
        fold_end = cursor + pd.Timedelta(days=n_days) - pd.Timedelta(seconds=1)
        if i == k - 1:
            fold_end = end
        folds.append((f"F{i + 1}", cursor, fold_end))
        cursor = fold_end + pd.Timedelta(seconds=1)
    return folds


def run_window(label, start, end, prices, regime) -> dict:
    """Line-for-line copy of research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.run_window
    (including its 2026-08-02 regime +1h merge_asof fix), repointed at per-fold start/end and combined
    (validation+oos) ledgers so a fold can straddle the original VAL/OOS boundary."""
    trades_a = load_all_trades(LEG_A_LEDGERS, start, end)
    trades_b = load_all_trades(LEG_B_LEDGERS, start, end)

    eq_a = build_leg_equity_path(trades_a, prices, start, end, use_ledger_trade_return=True)
    eq_b = build_leg_equity_path(trades_b, prices, start, end, use_ledger_trade_return=True)

    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)

    eq_a_1h = eq_a.resample("1h").last().ffill()
    eq_b_1h = eq_b.resample("1h").last().ffill()
    ts = pd.Series(eq_a_1h.index)
    side_a = leg_side_series(trades_a, ts)
    side_b = leg_side_series(trades_b, ts)
    active_a, active_b = side_a != 0, side_b != 0
    conflict = active_a & active_b & (side_a != side_b)

    # FIX 2026-08-02 (project-btc-run-window-merge-point-fixed-20260802.md): shift regime timestamp
    # +1h before merge_asof so the matched row is guaranteed at least 1h stale relative to the delta
    # window it gates -- unchanged copy of the fix already applied in the source script.
    reg = regime[(regime["timestamp"] >= start) & (regime["timestamp"] <= end)].copy()
    reg["timestamp"] = reg["timestamp"] + pd.Timedelta(hours=1)
    reg_frame = pd.merge_asof(pd.DataFrame({"timestamp": ts}), reg, on="timestamp", direction="backward")
    bull = reg_frame[f"{PFX}bull_prob"].fillna(0.5).to_numpy()
    bear = reg_frame[f"{PFX}bear_prob"].fillna(0.5).to_numpy()

    delta_a = eq_a_1h.diff().fillna(0.0).to_numpy()
    delta_b = eq_b_1h.diff().fillna(0.0).to_numpy()

    w_a, w_b = rule_weights(conflict, side_a, side_b, bull, bear)
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)

    n_conflict = int(conflict.sum())
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)
    beats_leg_a = tiebreak["pnl_pct"] > sa["pnl_pct"] and tiebreak["mdd_pct"] > sa["mdd_pct"]
    min_trades = min(len(trades_a), len(trades_b))
    thin = min_trades < THIN_TRADE_THRESHOLD
    print(f"\n=== {label} {start.date()}..{end.date()} ===")
    print(f"Leg A (BTC h48qual) alone       : pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}")
    print(f"Leg B (BTC sigma9+regime) alone : pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"Combined baseline (1x-1x fixed) : pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"Combined regime_tiebreak (n_conflict_bars={n_conflict}): "
          f"pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}%  beats_leg_a_both_axes={beats_leg_a}"
          f"{'  [[THIN FOLD]]' if thin else ''}")
    return {
        "fold": label, "start": start.date(), "end": end.date(),
        "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"], "leg_a_trades": len(trades_a),
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"], "leg_b_trades": len(trades_b),
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "tiebreak_pnl": tiebreak["pnl_pct"], "tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "beats_leg_a_both_axes": beats_leg_a, "thin_fold": thin,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_5m_prices_btc()
    regime = load_regime_probs()

    folds = build_folds(FULL_START, FULL_END, K_FOLDS)
    print("########## Fold definitions (non-overlapping, contiguous) ##########")
    for label, s, e in folds:
        print(f"{label}: {s.date()} .. {e.date()}  ({(e - s).days + 1} days)")

    rows = [run_window(label, s, e, prices, regime) for label, s, e in folds]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "fold_results.csv", index=False)

    n_pass = int(df["beats_leg_a_both_axes"].sum())
    n_thin = int(df["thin_fold"].sum())
    print(f"\n########## SUMMARY: regime_tiebreak beats Leg A alone (pnl AND mdd) in "
          f"{n_pass}/{len(df)} independent folds ##########")
    print(f"({n_thin}/{len(df)} folds flagged THIN -- min(leg_a_trades, leg_b_trades) < "
          f"{THIN_TRADE_THRESHOLD} -- interpret those results with caution.)")
    print(df.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'fold_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
