#!/usr/bin/env python3
"""Genuine leave-one-window-out (LOWO) check for the BTC dumb-momentum regime_tiebreak control
(scripts/diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py), which found that a
purely causal trailing-N-hour-log-return flip-flop, fed through the SAME regime_tiebreak mechanics
used for the CryptoMamba tiebreak, beat Leg A (BTC h48qual) alone on 6/6 windows tested that day.

The unresolved problem: all 6 of those windows (VAL_2025Q4, OOS_2026extended, and rolling W1-W4) are
overlapping SLICES of the SAME single ~9-month BTC downtrend (2025-10-01..2026-06-25, the full extent
where both Leg A and Leg B have ledger data). A config that looks good only within one continuous
market regime is not validated against regime-generality -- only against noise within that one regime
(same failure mode already confirmed for the ETH Sigma6 regime-filter candidate, see
project-sigma6-regime-filter-rolling-window-CONFIRMS-val-overfit-20260801.md).

This script applies the SAME leave-one-window-out discipline used for that ETH check
(research_sigma6_regime_filter_leave_one_window_out_20260801.py): split the full BTC overlap range
into K=5 non-overlapping, roughly-equal contiguous folds (unlike the prior W1-W4 rolling windows,
which heavily overlap and don't count as independent folds for this purpose). The "hyperparameter"
selected via leave-one-out is the dumb-momentum lookback N in {3, 6, 9, 12, 18, 24} hours (wider than
the prior script's pre-committed {6, 12, 24} to give the selection something real to choose from).
For each held-out fold: using ONLY the other 4 folds, pick whichever N beats Leg A alone on BOTH pnl
and mdd in a majority of the selection folds (mirroring the Sigma6 LOWO script's exact selection
rule -- among qualifiers, break ties by best mean pnl margin over Leg A alone on the selection folds);
then evaluate that selected N's dumb-momentum tiebreak on the held-out fold it never saw.

Leg A/B trade loading, equity-path reconstruction, regime_tiebreak weight rule, and the dumb-momentum
signal construction are UNCHANGED reuse of research_btc_h48qual_sigma9regime_tau1_joint_portfolio_
20260801.py and diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py -- nothing about
the underlying math is re-derived here, only the fold/selection harness is new.

DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule: this is a validation-methodology check on already-
saved ledgers, not a Fresh-Forward bar-by-bar walk-forward test, and neither leg is live-wired. Does
not touch trading_bot.py or any live wiring.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801 import (  # noqa: E402
    LEG_A_DIR, LEG_B_DIR, load_5m_prices_btc,
)
from research_btc_tau1_cryptomamba_tiebreak_20260801 import load_all_trades  # noqa: E402
from diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801 import (  # noqa: E402
    build_dumb_momentum_regime, run_window,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_dumbmomentum_leave_one_window_out"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

CANDIDATE_N_HOURS = [3, 6, 9, 12, 18, 24]
K_FOLDS = 5
THIN_TRADE_THRESHOLD = 3  # leg-A trade count below this in a held-out fold => flag as too thin to trust

FULL_START = pd.Timestamp("2025-10-01")
FULL_END = pd.Timestamp("2026-06-25 23:59:59")


def build_folds(start: pd.Timestamp, end: pd.Timestamp, k: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """K non-overlapping, roughly-equal contiguous folds spanning [start, end] by calendar day."""
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


def fold_result(label: str, start: pd.Timestamp, end: pd.Timestamp, prices, regime, n_hours: int) -> dict:
    """Run the dumb-momentum tiebreak (for one candidate N) on one fold, returning full diagnostics
    including Leg A/B alone trade counts so thin folds can be flagged."""
    r = run_window(f"{label}_N{n_hours}h", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), prices, regime)
    trades_a = load_all_trades(LEG_A_LEDGERS, start, end)
    trades_b = load_all_trades(LEG_B_LEDGERS, start, end)
    r["n_hours"] = n_hours
    r["fold"] = label
    r["leg_a_trades"] = len(trades_a)
    r["leg_b_trades"] = len(trades_b)
    return r


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices_5m = load_5m_prices_btc()
    close_1h = prices_5m["close"].resample("1h").last().ffill()

    folds = build_folds(FULL_START, FULL_END, K_FOLDS)
    print("########## Fold definitions (non-overlapping, contiguous) ##########")
    for label, s, e in folds:
        print(f"{label}: {s.date()} .. {e.date()}  ({(e - s).days + 1} days)")

    print("\n########## Precompute: dumb-momentum regime + fold result for every (N, fold) pair ##########")
    regimes = {n: build_dumb_momentum_regime(close_1h, n) for n in CANDIDATE_N_HOURS}
    all_rows = []
    for n in CANDIDATE_N_HOURS:
        for label, s, e in folds:
            row = fold_result(label, s, e, prices_5m, regimes[n], n)
            all_rows.append(row)
    grid_df = pd.DataFrame(all_rows)
    grid_df.to_csv(OUT_DIR / "grid_all_n_all_folds.csv", index=False)

    def get(label: str, n: int) -> dict:
        rows = grid_df[(grid_df["fold"] == label) & (grid_df["n_hours"] == n)]
        return rows.iloc[0].to_dict()

    print("\n########## LOWO selection: for each held-out fold, pick N using ONLY the other folds ##########")
    fold_labels = [label for label, _, _ in folds]
    lowo_rows = []
    selected_ns = []
    for held_idx, (held_label, held_start, held_end) in enumerate(folds):
        selection_labels = [lbl for i, lbl in enumerate(fold_labels) if i != held_idx]
        majority_needed = math.floor(len(selection_labels) / 2) + 1  # majority of selection folds

        candidates = []
        for n in CANDIDATE_N_HOURS:
            wins, margins = 0, []
            for lbl in selection_labels:
                r = get(lbl, n)
                beats = r["dumbmom_tiebreak_pnl"] > r["leg_a_pnl"] and r["dumbmom_tiebreak_mdd"] > r["leg_a_mdd"]
                wins += int(beats)
                margins.append(r["dumbmom_tiebreak_pnl"] - r["leg_a_pnl"])
            if wins >= majority_needed:
                candidates.append((n, wins, sum(margins) / len(margins)))

        if not candidates:
            selected_ns.append(None)
            lowo_rows.append({
                "held_out": held_label, "held_start": held_start.date(), "held_end": held_end.date(),
                "n_candidates": 0, "selected_n_hours": None, "selection_wins": None,
                "held_leg_a_pnl": None, "held_leg_a_mdd": None, "held_leg_a_trades": None,
                "held_leg_b_pnl": None, "held_leg_b_mdd": None, "held_leg_b_trades": None,
                "held_baseline_pnl": None, "held_baseline_mdd": None,
                "held_dumbmom_pnl": None, "held_dumbmom_mdd": None,
                "held_out_beats_leg_a": False, "thin_fold": None,
            })
            print(f"{held_label}: NO N reached majority ({majority_needed}/{len(selection_labels)}) "
                  "on selection folds -- no selection made.")
            continue

        # among qualifiers, break ties by best mean pnl margin over Leg A alone on selection folds
        candidates.sort(key=lambda t: (-t[1], -t[2]))
        best_n, sel_wins, sel_margin = candidates[0]
        selected_ns.append(best_n)

        held = get(held_label, best_n)
        beats_held = held["dumbmom_tiebreak_pnl"] > held["leg_a_pnl"] and held["dumbmom_tiebreak_mdd"] > held["leg_a_mdd"]
        thin = held["leg_a_trades"] < THIN_TRADE_THRESHOLD

        lowo_rows.append({
            "held_out": held_label, "held_start": held_start.date(), "held_end": held_end.date(),
            "n_candidates": len(candidates), "selected_n_hours": best_n,
            "selection_wins": f"{sel_wins}/{len(selection_labels)}",
            "held_leg_a_pnl": held["leg_a_pnl"], "held_leg_a_mdd": held["leg_a_mdd"], "held_leg_a_trades": held["leg_a_trades"],
            "held_leg_b_pnl": held["leg_b_pnl"], "held_leg_b_mdd": held["leg_b_mdd"], "held_leg_b_trades": held["leg_b_trades"],
            "held_baseline_pnl": held["baseline_pnl"], "held_baseline_mdd": held["baseline_mdd"],
            "held_dumbmom_pnl": held["dumbmom_tiebreak_pnl"], "held_dumbmom_mdd": held["dumbmom_tiebreak_mdd"],
            "held_out_beats_leg_a": beats_held, "thin_fold": thin,
        })
        thin_note = "  [[THIN FOLD -- too few trades to be meaningful]]" if thin else ""
        print(f"{held_label}: selected N={best_n}h (selection wins {sel_wins}/{len(selection_labels)}, "
              f"margin={sel_margin:+.2f}pp) | held-out: leg_a_trades={held['leg_a_trades']} "
              f"leg_a pnl={held['leg_a_pnl']:+.2f}%/mdd={held['leg_a_mdd']:.2f}%  "
              f"dumbmom pnl={held['dumbmom_tiebreak_pnl']:+.2f}%/mdd={held['dumbmom_tiebreak_mdd']:.2f}%  "
              f"beats_leg_a={beats_held}{thin_note}")

    lowo_df = pd.DataFrame(lowo_rows)
    lowo_df.to_csv(OUT_DIR / "leave_one_window_out_results.csv", index=False)

    print("\n########## Same-config-vs-varying-selection check ##########")
    resolved = [n for n in selected_ns if n is not None]
    if resolved and len(set(resolved)) == 1:
        print(f"SAME N selected in every fold that reached majority: N={resolved[0]}h "
              f"({len(resolved)}/{K_FOLDS} folds resolved) -- reassuring, not noise-driven cherry-picking.")
    elif resolved:
        print(f"VARYING N selected across folds: {selected_ns} -- RED FLAG for noise-driven cherry-picking, "
              "not a stable generalizing hyperparameter.")
    else:
        print("NO fold reached majority selection at all -- the dumb-momentum tiebreak does not "
              "reliably beat Leg A alone even within-sample across non-overlapping folds.")

    print("\n########## Held-out win rate ##########")
    resolved_df = lowo_df[lowo_df["selected_n_hours"].notna()]
    n_pass = int(resolved_df["held_out_beats_leg_a"].sum()) if len(resolved_df) else 0
    n_thin = int(resolved_df["thin_fold"].sum()) if len(resolved_df) else 0
    print(f"{n_pass}/{len(folds)} held-out folds: LOWO-selected N beats Leg A alone (pnl AND mdd) "
          f"on the fold it never influenced selection for.")
    print(f"({n_thin}/{len(resolved_df) if len(resolved_df) else 0} resolved folds flagged THIN -- "
          f"leg_a_trades < {THIN_TRADE_THRESHOLD} -- interpret those results with caution, not as full evidence.)")
    print(f"\nWrote {OUT_DIR / 'leave_one_window_out_results.csv'} and {OUT_DIR / 'grid_all_n_all_folds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
