#!/usr/bin/env python3
"""Genuine leave-one-window-out (LOWO) check for a NEW BTC Leg B candidate paired with the
already-vetted Leg A (BTC h48qual, tmp/causal_regen_20260516/btc_final_scale_map_20260708/).

Context: this session already closed the h48qual + Sigma9-trend-scan-regime pairing
(btc_v2_regime_trendscan_hgb_20260714) under genuine LOWO -- see
scripts/research_btc_dumbmomentum_tiebreak_leave_one_window_out_20260801.py, which found 2
contiguous non-overlapping folds covering Jan-May 2026 fail on EVERY regime-source/parameter choice
tested, regardless of tiebreak signal -- suggesting the problem may be structural to the Leg A +
Sigma9 pairing in that period specifically, not fixable by a better regime signal. Per the user's
explicit decision after that finding: try a DIFFERENT Leg B this time (different underlying signal
family), keep Leg A = h48qual, and build in genuine LOWO validation from the start instead of testing
on overlapping windows first.

New Leg B: tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714/ (model_id
btc_v2_direction_quality_lgbm_20260714, status research_candidate) -- a side-specific temporal-
environment LightGBM ensemble (direction head = sign of causal 1-day BTC return, quality head =
calibrated probability minus one ensemble std converted to barrier EV, selective entry only when EV
positive). Genuinely different architecture family from both h48qual (TabM) and Sigma9 (HGB
trend-scan) -- good independence properties for a combination leg. Standalone VAL pnl=-7.35%, OOS
pnl=-18.78% (negative standalone, same pattern as Sigma9, which was also negative standalone yet
still useful as a combination leg -- not rejected for this reason alone).

Overlap range: Leg A ledgers span entry_timestamp 2025-10-01..2026-06-25; Leg B ledgers span
2025-10-08..2026-07-05. Working range used here is the intersection: 2025-10-08..2026-06-25.

Methodology: EXACT same LOWO discipline as
research_btc_dumbmomentum_tiebreak_leave_one_window_out_20260801.py (which this script otherwise
mirrors almost line-for-line) -- do NOT test on overlapping rolling windows first. Split the overlap
range into K=5 non-overlapping, roughly-equal contiguous folds. Candidate dumb-momentum lookback N
in {3, 6, 9, 12, 18, 24} hours (purely causal trailing-N-hour-log-return flip-flop regime signal,
cheapest and most model-agnostic tiebreak driver available, already vetted this session as a fair
test of the COMBINATION MECHANISM itself rather than of any one regime model's quality -- reused
unchanged from diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.build_dumb_momentum_
regime). For each held-out fold: using ONLY the other 4 folds, pick whichever N beats Leg A alone on
BOTH pnl and mdd in a majority of the selection folds (ties broken by best mean pnl margin over Leg A
alone on the selection folds); then evaluate that selected N's dumb-momentum tiebreak on the held-out
fold it never saw.

Leg-loading/equity-path/regime_tiebreak-weight primitives (omega_trades_from_ledger,
build_leg_equity_path, summarize_equity, leg_side_series, rule_weights, weighted_pnl,
build_dumb_momentum_regime, load_all_trades, load_5m_prices_btc) are UNCHANGED reuse of
research_eth_sigma3_1h_omega461_joint_portfolio_20260731.py,
research_btc_h48qual_sigma9regime_tau1_joint_portfolio_20260801.py,
research_btc_tau1_cryptomamba_tiebreak_20260801.py, and
diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.py -- nothing about the underlying
combination math is re-derived here. The one thing that CANNOT be imported unchanged is
diagnose_...control's own `run_window`, because that function closes over that file's own
module-level LEG_A_LEDGERS/LEG_B_LEDGERS globals, which point at the OLD Sigma9 Leg B ledgers -- this
script defines its own LEG_A_LEDGERS/LEG_B_LEDGERS (Leg B repointed at
btc_v2_direction_quality_lgbm_20260714) and a `run_window` built from the same imported primitives.

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

OUT_DIR = ROOT / "tmp/research_20260801/btc_h48qual_direction_quality_lgbm_lowo"

LEG_B_DIR = ROOT / "tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS = [LEG_B_DIR / "validation_ledger.csv", LEG_B_DIR / "oos_ledger.csv"]

CANDIDATE_N_HOURS = [3, 6, 9, 12, 18, 24]
K_FOLDS = 5
THIN_TRADE_THRESHOLD = 5  # min(leg_a_trades, leg_b_trades) below this in a held-out fold => flag as thin

# Intersection of Leg A (2025-10-01..2026-06-25) and Leg B (2025-10-08..2026-07-05) ledger coverage.
FULL_START = pd.Timestamp("2025-10-08")
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


def run_window(label, start, end, prices, regime) -> dict:
    """Same mechanics as diagnose_btc_cryptomamba_tiebreak_dumbmomentum_control_20260801.run_window,
    repointed at this script's own LEG_A_LEDGERS/LEG_B_LEDGERS (new Leg B)."""
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
    tiebreak = weighted_pnl(delta_a, delta_b, w_a, w_b)
    eq_ab_baseline = 1.0 + (eq_a - 1.0) + (eq_b - 1.0)
    sab_base = summarize_equity(eq_ab_baseline)
    sa, sb = summarize_equity(eq_a), summarize_equity(eq_b)

    n_conflict = int(conflict.sum())
    beats_leg_a = tiebreak["pnl_pct"] > sa["pnl_pct"] and tiebreak["mdd_pct"] > sa["mdd_pct"]
    print(f"\n=== {label} {start.date()}..{end.date()} ===")
    print(f"Leg A (h48qual) alone: pnl={sa['pnl_pct']:+.2f}% mdd={sa['mdd_pct']:.2f}% trades={len(trades_a)}   "
          f"Leg B (direction_quality_lgbm) alone: pnl={sb['pnl_pct']:+.2f}% mdd={sb['mdd_pct']:.2f}% trades={len(trades_b)}")
    print(f"Fixed 1x-1x: pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%   "
          f"Dumb-momentum tiebreak (n_conflict={n_conflict}): pnl={tiebreak['pnl_pct']:+.2f}% "
          f"mdd={tiebreak['mdd_pct']:.2f}%  beats_leg_a_both_axes={beats_leg_a}")
    return {
        "label": label, "leg_a_pnl": sa["pnl_pct"], "leg_a_mdd": sa["mdd_pct"], "leg_a_trades": len(trades_a),
        "leg_b_pnl": sb["pnl_pct"], "leg_b_mdd": sb["mdd_pct"], "leg_b_trades": len(trades_b),
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "dumbmom_tiebreak_pnl": tiebreak["pnl_pct"], "dumbmom_tiebreak_mdd": tiebreak["mdd_pct"],
        "n_conflict_bars": n_conflict, "beats_leg_a_both_axes": beats_leg_a,
    }


def fold_result(label: str, start: pd.Timestamp, end: pd.Timestamp, prices, regime, n_hours: int) -> dict:
    r = run_window(f"{label}_N{n_hours}h", start, end, prices, regime)
    r["n_hours"] = n_hours
    r["fold"] = label
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
        majority_needed = math.floor(len(selection_labels) / 2) + 1

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
            held = get(held_label, CANDIDATE_N_HOURS[0])  # for trade counts only, no config selected
            min_trades = min(held["leg_a_trades"], held["leg_b_trades"])
            lowo_rows.append({
                "held_out": held_label, "held_start": held_start.date(), "held_end": held_end.date(),
                "n_candidates": 0, "selected_n_hours": None, "selection_wins": None,
                "held_leg_a_pnl": None, "held_leg_a_mdd": None, "held_leg_a_trades": held["leg_a_trades"],
                "held_leg_b_pnl": None, "held_leg_b_mdd": None, "held_leg_b_trades": held["leg_b_trades"],
                "held_baseline_pnl": None, "held_baseline_mdd": None,
                "held_dumbmom_pnl": None, "held_dumbmom_mdd": None,
                "held_out_beats_leg_a": False, "thin_fold": min_trades < THIN_TRADE_THRESHOLD,
            })
            print(f"{held_label}: NO N reached majority ({majority_needed}/{len(selection_labels)}) "
                  "on selection folds -- no selection made. "
                  f"(leg_a_trades={held['leg_a_trades']}, leg_b_trades={held['leg_b_trades']})")
            continue

        # among qualifiers, break ties by best mean pnl margin over Leg A alone on selection folds
        candidates.sort(key=lambda t: (-t[1], -t[2]))
        best_n, sel_wins, sel_margin = candidates[0]
        selected_ns.append(best_n)

        held = get(held_label, best_n)
        beats_held = held["dumbmom_tiebreak_pnl"] > held["leg_a_pnl"] and held["dumbmom_tiebreak_mdd"] > held["leg_a_mdd"]
        min_trades = min(held["leg_a_trades"], held["leg_b_trades"])
        thin = min_trades < THIN_TRADE_THRESHOLD

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
              f"leg_b_trades={held['leg_b_trades']} "
              f"leg_a pnl={held['leg_a_pnl']:+.2f}%/mdd={held['leg_a_mdd']:.2f}%  "
              f"baseline(1x-1x) pnl={held['baseline_pnl']:+.2f}%/mdd={held['baseline_mdd']:.2f}%  "
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
          f"min(leg_a_trades, leg_b_trades) < {THIN_TRADE_THRESHOLD} -- interpret those results with "
          f"caution, not as full evidence.)")
    print(f"\nWrote {OUT_DIR / 'leave_one_window_out_results.csv'} and {OUT_DIR / 'grid_all_n_all_folds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
