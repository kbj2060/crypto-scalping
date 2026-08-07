#!/usr/bin/env python3
"""Bug hunt: does build_dumb_momentum_regime() (diagnose_btc_cryptomamba_tiebreak_dumbmomentum_
control_20260801.py) have a same-bar look-ahead that mechanically inflates the +509% BTC h48qual+
direction_quality_lgbm regime_tiebreak continuous-backtest result (audit_btc_h48qual_lgbm_tiebreak_
509pct_20260801.py), whose 41/41 (100%) conflict-episode win rate is inconsistent with the same
candidate's own genuine LOWO fold win rate of 4/5 (80%)?

Hypothesis: eq_a_1h/eq_b_1h/close_1h are all built via pandas `.resample("1h").last()`, which
LEFT-labels bins -- label t holds the LAST value observed in [t, t+1h), i.e. data through ~t+55min
(verified empirically below with a tiny synthetic example). So:
  - delta_a[t] = eq_a_1h[t] - eq_a_1h[t-1h] represents the leg's equity change accumulated roughly
    over [t, t+1h) -- i.e. it reflects price action that happens DURING bar t, closing at ~t+55min.
  - The dumb-momentum regime's bull_prob at timestamp t is computed from close_1h[t], which is ALSO
    the last close observed in [t, t+1h) -- the SAME window's own closing price.
  - merge_asof(..., direction="backward") assigns ts[i]=t the regime row with timestamp <= t, and
    EQUALITY qualifies -- so the regime "decision" for bar t is built from close_1h[t], the exact
    same end-of-bar price that delta_a[t]/delta_b[t] are also derived from. The tiebreak is, in
    effect, asking "did price go up during bar t" to decide whether to keep bar t's OWN pnl delta.

This script:
  1. Reproduces the resample-label evidence empirically (a fresh synthetic example, not reasoning
     from memory).
  2. Confirms eq_a_1h/eq_b_1h and close_1h share the identical resample convention (both literally
     call `.resample("1h").last()` on data indexed by the same 5m timestamps).
  3. Builds a STRICTLY causal control variant of the regime: shift the ORIGINAL regime rows' own
     timestamp column forward by +1h before merge_asof. Because merge_asof(direction="backward")
     matches ts[i]=t to the largest row timestamp <= t, after the shift that means only rows whose
     UNSHIFTED (original construction) timestamp was <= t-1h can match -- i.e. the assigned regime
     for bar t is now guaranteed to derive from close prices at least one full hour stale relative
     to bar t's own delta. Nothing about rule_weights/weighted_pnl/leg_side_series/conflict/
     build_leg_equity_path is touched -- only the input regime frame passed into the UNCHANGED
     run_window() differs (leaky vs shifted).
  4. Reruns the SAME 41-episode continuous backtest (2025-10-08..2026-06-25, N=3h, Leg
     B=direction_quality_lgbm) with leaky vs causal-shifted regime and reports episode win rate and
     total pnl for both -- the decisive before/after comparison.
  5. Reruns the LOWO harness (research_btc_h48qual_direction_quality_lgbm_lowo_20260801.py -- this
     is the SAME Leg A=h48qual/Leg B=direction_quality_lgbm pairing as the +509% audit, and is the
     script the task background's "4/5 LOWO fold win rate" claim refers to, NOT the older Sigma9-
     regime-trendscan LOWO script) with leaky vs causal-shifted regimes across all 6 candidate N and
     reports the held-out win rate for both, to see whether the LOWO verdict itself was also affected.

Reuses build_leg_equity_path/summarize_equity/leg_side_series/rule_weights/weighted_pnl/
load_all_trades/build_dumb_momentum_regime/run_window UNCHANGED from the existing scripts named
above -- nothing about the combination math is re-derived, only a new regime-input variant is
constructed and fed through the same unmodified pipeline. Does NOT modify any existing script.
Does not touch trading_bot.py or any live wiring. DIAGNOSTIC ONLY per CLAUDE.md Fresh-Forward Rule
(reuses already-saved ledgers, not a fresh-forward walk-forward test).
"""
from __future__ import annotations

import math
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
    build_dumb_momentum_regime, run_window,
)
from research_btc_h48qual_direction_quality_lgbm_lowo_20260801 import (  # noqa: E402
    build_folds, fold_result, CANDIDATE_N_HOURS as LOWO_CANDIDATE_N_HOURS, K_FOLDS,
    FULL_START as LOWO_FULL_START, FULL_END as LOWO_FULL_END,
)

OUT_DIR = ROOT / "tmp/research_20260801/btc_lookahead_bug_hunt"

# Same leg B / window as the +509% audit (direction_quality_lgbm, N=3h, full continuous span).
LEG_B_DIR_509 = ROOT / "tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714"
LEG_A_LEDGERS = [LEG_A_DIR / "validation_ledger.csv", LEG_A_DIR / "oos_ledger.csv"]
LEG_B_LEDGERS_509 = [LEG_B_DIR_509 / "validation_ledger.csv", LEG_B_DIR_509 / "oos_ledger.csv"]
SELECTED_N_HOURS = 3
FULL_START = pd.Timestamp("2025-10-08")
FULL_END = pd.Timestamp("2026-06-25 23:59:59")

# Leg B for the LOWO rerun is direction_quality_lgbm too -- imported fold_result()/run_window()
# above close over research_btc_h48qual_direction_quality_lgbm_lowo_20260801's OWN module-level
# LEG_A_LEDGERS/LEG_B_LEDGERS globals (already pointed at h48qual + direction_quality_lgbm), so no
# separate leg-B ledger paths need to be redeclared here.


def shift_regime_forward_one_bar(regime: pd.DataFrame) -> pd.DataFrame:
    """Strictly-causal control: shift the regime's OWN construction timestamp forward by +1h.
    merge_asof(direction='backward') on ts[i]=t then only admits rows whose ORIGINAL (unshifted)
    timestamp was <= t-1h, guaranteeing the assigned bull/bear_prob derives from close prices at
    least one full hour stale relative to bar t's own delta_a[t]/delta_b[t]."""
    shifted = regime.copy()
    shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta("1h")
    return shifted


def step1_resample_label_evidence() -> None:
    print("########## STEP 1: empirical resample('1h').last() label-convention evidence ##########")
    idx = pd.date_range("2026-01-01 00:00", periods=24, freq="5min")
    s = pd.Series(range(24), index=idx)
    r = s.resample("1h").last()
    print("Synthetic 5m series values 0..23 at :00,:05,...,:55 twice; resample('1h').last():")
    print(r.to_string())
    assert r.iloc[0] == 11 and r.iloc[1] == 23, "resample label convention changed vs assumption"
    print("CONFIRMED: label t=00:00 holds the value from 00:55 (i.e. data through ~t+55min, NOT t).")
    print("This is the SAME convention eq_a_1h/eq_b_1h/close_1h all use (all call "
          "'.resample(\"1h\").last()' on 5m-indexed series/prices).")
    print()


def step2_conflict_episodes(diff: np.ndarray, conflict: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_int = conflict.astype(int)
    ep_starts = np.where(np.diff(c_int, prepend=0) == 1)[0]
    ep_ends = np.where(np.diff(c_int, append=0) == -1)[0]
    ep_gains = np.array([diff[s:e + 1].sum() for s, e in zip(ep_starts, ep_ends)])
    return ep_starts, ep_ends, ep_gains


def run_509_scenario(regime: pd.DataFrame, prices_5m: pd.DataFrame, label: str) -> dict:
    trades_a = load_all_trades(LEG_A_LEDGERS, FULL_START, FULL_END)
    trades_b = load_all_trades(LEG_B_LEDGERS_509, FULL_START, FULL_END)
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

    c_tiebreak = w_a * delta_a + w_b * delta_b
    c_baseline = delta_a + delta_b
    diff = c_tiebreak - c_baseline
    ep_starts, ep_ends, ep_gains = step2_conflict_episodes(diff, conflict)
    n_episodes = len(ep_gains)
    n_pos = int((ep_gains > 0).sum())
    n_neg = int((ep_gains < 0).sum())
    n_zero = n_episodes - n_pos - n_neg
    win_rate = 100.0 * n_pos / n_episodes if n_episodes else float("nan")

    print(f"--- {label} ---")
    print(f"  tiebreak pnl={tiebreak['pnl_pct']:+.2f}% mdd={tiebreak['mdd_pct']:.2f}%   "
          f"baseline pnl={sab_base['pnl_pct']:+.2f}% mdd={sab_base['mdd_pct']:.2f}%")
    print(f"  n_conflict_bars={int(conflict.sum())}  n_episodes={n_episodes}  "
          f"positive={n_pos} negative={n_neg} zero={n_zero}  win_rate={win_rate:.1f}%")
    return {
        "scenario": label, "tiebreak_pnl": tiebreak["pnl_pct"], "tiebreak_mdd": tiebreak["mdd_pct"],
        "baseline_pnl": sab_base["pnl_pct"], "baseline_mdd": sab_base["mdd_pct"],
        "n_conflict_bars": int(conflict.sum()), "n_episodes": n_episodes,
        "episodes_positive": n_pos, "episodes_negative": n_neg, "episodes_zero": n_zero,
        "episode_win_rate_pct": win_rate,
    }


def step4_decisive_509_test(prices_5m: pd.DataFrame, close_1h: pd.Series) -> pd.DataFrame:
    print("\n########## STEP 4: decisive before/after test on the +509% continuous backtest ##########")
    print(f"Window: {FULL_START.date()}..{FULL_END.date()}  Leg A=h48qual  Leg B=direction_quality_lgbm  "
          f"N={SELECTED_N_HOURS}h dumb-momentum tiebreak\n")
    regime_leaky = build_dumb_momentum_regime(close_1h, SELECTED_N_HOURS)
    regime_causal = shift_regime_forward_one_bar(regime_leaky)

    row_leaky = run_509_scenario(regime_leaky, prices_5m, "LEAKY (original, same-bar merge_asof<=)")
    row_causal = run_509_scenario(regime_causal, prices_5m, "CAUSAL (regime timestamp shifted +1h before merge_asof)")

    df = pd.DataFrame([row_leaky, row_causal])
    df.to_csv(OUT_DIR / "step4_509_before_after.csv", index=False)
    return df


def step5_decisive_lowo_test(prices_5m: pd.DataFrame, close_1h: pd.Series) -> pd.DataFrame:
    print("\n########## STEP 5: rerun LOWO harness with leaky vs causal-shifted regime ##########")
    print(f"Leg A=h48qual  Leg B=direction_quality_lgbm (the SAME pairing as the +509% audit; this")
    print(f"is the script the task's '4/5 fold win rate' claim refers to)  K={K_FOLDS} folds  "
          f"N candidates={LOWO_CANDIDATE_N_HOURS}\n")
    folds = build_folds(LOWO_FULL_START, LOWO_FULL_END, K_FOLDS)
    fold_labels = [label for label, _, _ in folds]

    results = {}
    for variant in ("leaky", "causal"):
        regimes = {}
        for n in LOWO_CANDIDATE_N_HOURS:
            reg = build_dumb_momentum_regime(close_1h, n)
            regimes[n] = reg if variant == "leaky" else shift_regime_forward_one_bar(reg)

        all_rows = []
        for n in LOWO_CANDIDATE_N_HOURS:
            for label, s, e in folds:
                row = fold_result(label, s, e, prices_5m, regimes[n], n)
                all_rows.append(row)
        grid_df = pd.DataFrame(all_rows)

        def get(label: str, n: int) -> dict:
            rows = grid_df[(grid_df["fold"] == label) & (grid_df["n_hours"] == n)]
            return rows.iloc[0].to_dict()

        lowo_rows = []
        for held_idx, (held_label, held_start, held_end) in enumerate(folds):
            selection_labels = [lbl for i, lbl in enumerate(fold_labels) if i != held_idx]
            majority_needed = math.floor(len(selection_labels) / 2) + 1
            candidates = []
            for n in LOWO_CANDIDATE_N_HOURS:
                wins, margins = 0, []
                for lbl in selection_labels:
                    r = get(lbl, n)
                    beats = r["dumbmom_tiebreak_pnl"] > r["leg_a_pnl"] and r["dumbmom_tiebreak_mdd"] > r["leg_a_mdd"]
                    wins += int(beats)
                    margins.append(r["dumbmom_tiebreak_pnl"] - r["leg_a_pnl"])
                if wins >= majority_needed:
                    candidates.append((n, wins, sum(margins) / len(margins)))
            if not candidates:
                lowo_rows.append({"held_out": held_label, "selected_n_hours": None, "held_out_beats_leg_a": False})
                continue
            candidates.sort(key=lambda t: (-t[1], -t[2]))
            best_n = candidates[0][0]
            held = get(held_label, best_n)
            beats_held = held["dumbmom_tiebreak_pnl"] > held["leg_a_pnl"] and held["dumbmom_tiebreak_mdd"] > held["leg_a_mdd"]
            lowo_rows.append({
                "held_out": held_label, "selected_n_hours": best_n,
                "held_leg_a_pnl": held["leg_a_pnl"], "held_leg_a_mdd": held["leg_a_mdd"],
                "held_dumbmom_pnl": held["dumbmom_tiebreak_pnl"], "held_dumbmom_mdd": held["dumbmom_tiebreak_mdd"],
                "held_out_beats_leg_a": beats_held,
            })

        lowo_df = pd.DataFrame(lowo_rows)
        lowo_df["variant"] = variant
        n_pass = int(lowo_df["held_out_beats_leg_a"].sum())
        print(f"[{variant}] held-out win rate: {n_pass}/{len(folds)}")
        print(lowo_df.to_string(index=False))
        print()
        lowo_df.to_csv(OUT_DIR / f"step5_lowo_{variant}.csv", index=False)
        results[variant] = lowo_df

    return pd.concat(results.values(), ignore_index=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    step1_resample_label_evidence()

    prices_5m = load_5m_prices_btc()
    close_1h = prices_5m["close"].resample("1h").last().ffill()

    print("########## STEP 2/3: alignment check -- close_1h vs eq_a_1h same convention ##########")
    print("close_1h = prices_5m['close'].resample('1h').last().ffill()  <-- IDENTICAL call pattern to")
    print("eq_a_1h  = eq_a.resample('1h').last().ffill()  used inside run_window(). Both series are")
    print("indexed by the same 5m timestamp grid, so label t means the SAME underlying [t, t+1h) window")
    print("for both close_1h and eq_a_1h/eq_b_1h. merge_asof(direction='backward') on ts[i]=t admits")
    print("regime rows with timestamp <= t, i.e. EXACT equality with t is allowed -- the regime value")
    print("assigned to bar t can (and, since regime.timestamp IS ts's own grid, DOES) come from")
    print(f"close_1h[t] itself, computed from the same bar t's own last close.\n")

    step4_df = step4_decisive_509_test(prices_5m, close_1h)
    step5_df = step5_decisive_lowo_test(prices_5m, close_1h)

    print("\n########## VERDICT ##########")
    leaky = step4_df[step4_df["scenario"].str.startswith("LEAKY")].iloc[0]
    causal = step4_df[step4_df["scenario"].str.startswith("CAUSAL")].iloc[0]
    print(f"+509%-style continuous backtest: LEAKY episode win rate={leaky['episode_win_rate_pct']:.1f}% "
          f"(n={leaky['n_episodes']}) tiebreak_pnl={leaky['tiebreak_pnl']:+.2f}%  ->  "
          f"CAUSAL episode win rate={causal['episode_win_rate_pct']:.1f}% (n={causal['n_episodes']}) "
          f"tiebreak_pnl={causal['tiebreak_pnl']:+.2f}%")
    if causal["episode_win_rate_pct"] < leaky["episode_win_rate_pct"] - 1e-9:
        print("CONFIRMED: fixing the same-bar alignment materially changes the episode win rate/pnl "
              "-> the leaky version's 100% win rate was (at least partly) a look-ahead artifact.")
    else:
        print("NOT CONFIRMED BY THIS METRIC: causal fix did not lower the win rate/pnl -- look-ahead "
              "alignment does not explain the 100% win rate on its own.")
    print(f"\nAll outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
