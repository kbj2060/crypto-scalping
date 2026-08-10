#!/usr/bin/env python3
"""Research probe (2026-08-09): does a split-conformal abstention filter on ETH h48qual's
own dir_p_cash/p_long/p_short beat trading every non-cash raw call?

Motivation: this repo's own memory shows the ONE mechanism that has ever survived in this
project is FILTERING/BLOCKING an existing signal, never predicting direction itself
(project-eth-h48qual-contribution-is-blocking-not-earning-20260808). h48qual's own filter is a
hand-tuned probability threshold. Conformal prediction (Romano/Sesia/Candes 2020, Adaptive
Prediction Sets) gives a coverage-GUARANTEED confidence filter instead of a hand-tuned one --
worth testing as a creative but literature-grounded variant of the one thing that has worked.

Discipline (per this session's falsification-audit work):
  1. Calibrate the conformal quantile on CAL (first 60% of VAL bars) only.
  2. Search over an alpha grid on SEARCH-EVAL (last 40% of VAL bars) only.
  3. Run falsification_audit on the SEARCH-EVAL returns matrix BEFORE trusting any alpha.
  4. Only if it passes: apply the frozen winning alpha to TAIL (2026-07-01..07-12), a window
     never touched by anything above and outside the official OOS window -- report honestly.

This is a bar-level PROXY probe, not a full backtest: payoff = +tp_move on a barrier hit in the
predicted direction, -sl_move on a barrier hit against it, 0 on timeout/cash/abstain. No fees,
funding, slippage, or path-dependent same-bar-touch resolution. Good enough to falsify or not
falsify the idea cheaply before investing in the full causal_futures_backtest harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402
from pipeline.architecture_workbench import effect_size_report  # noqa: E402

PRED_DIR = ROOT / "tmp/eth_h48qual_alone_contract_20260808"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CAL_FRACTION = 0.60


def load_merged(pred_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(pred_path, parse_dates=["timestamp"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = pred.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def aps_calibration_scores(probs: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """Adaptive Prediction Sets nonconformity score per calibration row (Romano et al. 2020)."""
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    rank_of_true = np.argmax(order == true_idx[:, None], axis=1)
    return cum[np.arange(len(probs)), rank_of_true]


def aps_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    n = len(cal_scores)
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(cal_scores, level, method="higher"))


def apply_conformal(probs: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_top_class, is_singleton_and_directional) for every row."""
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_probs, axis=1)
    first_idx_ge = np.argmax(cum >= q_hat, axis=1)
    set_size = first_idx_ge + 1
    top_class = order[:, 0]
    tradeable = (set_size == 1) & (top_class != 0)
    return top_class, tradeable


def bar_payoff(top_class: np.ndarray, tradeable: np.ndarray, outcome: np.ndarray,
               tp: np.ndarray, sl: np.ndarray) -> np.ndarray:
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(top_class == 1, payoff_if_long, np.where(top_class == 2, payoff_if_short, 0.0))
    return np.where(tradeable, realized, 0.0)


def main() -> None:
    val = load_merged(PRED_DIR / "_aligned_val_h48qual.csv")
    tail = load_merged(PRED_DIR / "_aligned_tail_h48qual.csv")
    print(f"VAL bars: {len(val)} ({val.timestamp.min()} .. {val.timestamp.max()})")
    print(f"TAIL bars: {len(tail)} ({tail.timestamp.min()} .. {tail.timestamp.max()})")

    prob_cols = ["omega1_regime3_expertdq_dir_p_cash", "omega1_regime3_expertdq_dir_p_long",
                 "omega1_regime3_expertdq_dir_p_short"]
    val_probs = val[prob_cols].to_numpy()
    val_outcome = val["trade_outcome_action"].to_numpy()
    val_tp = val["tp_move"].to_numpy()
    val_sl = val["sl_move"].to_numpy()
    val_raw_top = val["omega1_regime3_expertdq_dir_action"].to_numpy()

    n_cal = int(len(val) * CAL_FRACTION)
    cal_probs, cal_outcome = val_probs[:n_cal], val_outcome[:n_cal]
    se_probs, se_outcome = val_probs[n_cal:], val_outcome[n_cal:]
    se_tp, se_sl = val_tp[n_cal:], val_sl[n_cal:]
    se_raw_top = val_raw_top[n_cal:]
    print(f"CAL bars: {n_cal}, SEARCH-EVAL bars: {len(se_probs)}")

    cal_scores = aps_calibration_scores(cal_probs, cal_outcome)

    baseline_se = bar_payoff(se_raw_top, se_raw_top != 0, se_outcome, se_tp, se_sl)
    print(f"\nBaseline (trade every non-cash raw call), SEARCH-EVAL split:")
    print(f"  n_trades={int((se_raw_top != 0).sum())}  mean={baseline_se.mean():.6f}  "
          f"sum={baseline_se.sum():.4f}  sharpe={baseline_se.mean() / (baseline_se.std() + 1e-12):.4f}")

    returns_matrix = np.zeros((len(se_probs), len(ALPHA_GRID)), dtype=np.float64)
    q_hats = []
    for j, alpha in enumerate(ALPHA_GRID):
        q_hat = aps_quantile(cal_scores, alpha)
        q_hats.append(q_hat)
        top_class, tradeable = apply_conformal(se_probs, q_hat)
        returns_matrix[:, j] = bar_payoff(top_class, tradeable, se_outcome, se_tp, se_sl)
        n_tr = int(tradeable.sum())
        mean_r = returns_matrix[:, j].mean()
        sharpe = mean_r / (returns_matrix[:, j].std() + 1e-12)
        print(f"  alpha={alpha:.2f} q_hat={q_hat:.4f} n_trades={n_tr:6d} "
              f"mean={mean_r:.6f} sum={returns_matrix[:, j].sum():.4f} sharpe={sharpe:.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_alpha, best_q_hat = ALPHA_GRID[best_j], q_hats[best_j]
    print(f"\nBest-of-{len(ALPHA_GRID)} on SEARCH-EVAL: alpha={best_alpha} (q_hat={best_q_hat:.4f})")

    print("\n=== Falsification audit on the SEARCH-EVAL alpha-grid search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")

    if not audit["passes_falsification_audit"]:
        print("\n[RESULT] FAILS falsification audit -- this alpha-grid search's winner is not "
              "distinguishable from a noise/microstructure-placebo artifact. STOPPING HERE; "
              "the frozen tail window and the h48qual-standalone comparison below are NOT "
              "evidence for adoption, only diagnostic context.")
    else:
        print("\n[RESULT] PASSES falsification audit. Proceeding to the frozen TAIL check.")

    # Effect-size comparison regardless of the falsification-audit verdict, for diagnostic
    # completeness -- but adoption must not rest on this if the audit above failed.
    print(f"\n=== SEARCH-EVAL: confident-conformal (alpha={best_alpha}) vs raw-baseline, per-bar returns ===")
    top_class_se, tradeable_se = apply_conformal(se_probs, best_q_hat)
    confident_bars = returns_matrix[:, best_j][tradeable_se]
    baseline_bars = baseline_se[se_raw_top != 0]
    if len(confident_bars) >= 3 and len(baseline_bars) >= 3:
        report = effect_size_report(confident_bars, baseline_bars, label_a="confident_conformal", label_b="raw_baseline")
        for k, v in report.items():
            print(f"  {k}: {v}")
    else:
        print(f"  too few trades to compare (confident={len(confident_bars)}, baseline={len(baseline_bars)})")

    print(f"\n=== Frozen TAIL check (2026-07-01..07-12, alpha={best_alpha} frozen from SEARCH-EVAL) ===")
    tail_probs = tail[prob_cols].to_numpy()
    tail_outcome = tail["trade_outcome_action"].to_numpy()
    tail_tp, tail_sl = tail["tp_move"].to_numpy(), tail["sl_move"].to_numpy()
    tail_raw_top = tail["omega1_regime3_expertdq_dir_action"].to_numpy()
    top_class_tail, tradeable_tail = apply_conformal(tail_probs, best_q_hat)
    tail_conf_returns = bar_payoff(top_class_tail, tradeable_tail, tail_outcome, tail_tp, tail_sl)
    tail_baseline_returns = bar_payoff(tail_raw_top, tail_raw_top != 0, tail_outcome, tail_tp, tail_sl)
    print(f"  confident: n_trades={int(tradeable_tail.sum())} mean={tail_conf_returns.mean():.6f} "
          f"sum={tail_conf_returns.sum():.4f}")
    print(f"  raw baseline: n_trades={int((tail_raw_top != 0).sum())} mean={tail_baseline_returns.mean():.6f} "
          f"sum={tail_baseline_returns.sum():.4f}")


def _col_sharpe(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


if __name__ == "__main__":
    main()
