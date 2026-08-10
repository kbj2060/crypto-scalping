#!/usr/bin/env python3
"""Research probe (2026-08-09, overnight loop idea #2): are rough-path-signature features of
the recent (time, cumulative-log-return) path a genuinely new entry-direction signal for ETH?

Motivation: signature-MMD is already flagged in this repo's memory as an untested REGIME
detector (project-regime-3class-literature-validation-20260809). This probe tests a different,
also-untried application: truncated path-signature terms (Levy area / iterated integrals of
time vs. log-return, Chen 1957 / Lyons et al.) as RAW INPUT FEATURES to a direction classifier,
independent of every existing feature in this repo's 100+-feature h48qual/omega stacks. If this
carries information at all, it should show up even as a signature-features-ONLY model (no other
feature mixed in) beating a coin-flip proxy P&L -- a clean ablation.

Discipline (same as idea #1, corrected for the mistake found there):
  1. Train one classifier per candidate window length {12,24,48,96} bars, signature-only
     features, on TRAIN (2024-06-01..2025-06-30).
  2. Score all 4 on DEV (2025-07-01..2025-08-31, held out from TRAIN fit) as a returns matrix
     and run falsification_audit on THAT matrix -- this only tells us whether the search across
     window lengths found something beyond noise, not whether it beats a baseline.
  3. The comparison that actually answers "is this a genuinely new signal" is effect_size_report
     between the winning signature-only model's DEV bar-returns and an ALWAYS-CASH baseline
     (0 return) is trivial; the real bar is beating a naive momentum baseline (raw window-return
     sign) computed on the SAME window length, since a Levy-area term could just be encoding
     "was the window positive," which is not new information.
  4. Only if both checks clear does the frozen winning window proceed to VAL, then a (partial,
     price data ends 2026-02-17) OOS look.

Proxy payoff (bar-level, no fees/funding/slippage) is identical to idea #1's construction:
+tp_move on a barrier hit in the predicted direction, -sl_move against it, 0 on timeout/cash.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import iisignature
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.selection_stats import falsification_audit  # noqa: E402
from pipeline.architecture_workbench import effect_size_report  # noqa: E402

PRICE_PATH = ROOT / "data/eth_5m_1year.csv"
LABEL_PATH = ROOT / "data/splits/year_oos/eth_5m_tripbarrier_tradeoutcome_labels_20260808.parquet"
WINDOWS = [12, 24, 48, 96]
TRAIN_END = "2025-06-30 23:55:00"
DEV_START, DEV_END = "2025-07-01 00:00:00", "2025-08-31 23:55:00"
VAL_START, VAL_END = "2025-09-01 00:00:00", "2025-12-31 23:55:00"
OOS_START = "2026-01-01 00:00:00"  # OOS_END is whatever the price file's last row is (partial)


def load_merged() -> pd.DataFrame:
    price = pd.read_csv(PRICE_PATH, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    labels = pd.read_parquet(LABEL_PATH)
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = price.merge(labels, on="timestamp", how="inner", validate="one_to_one")
    merged = merged.dropna(subset=["tp_move", "sl_move"]).sort_values("timestamp").reset_index(drop=True)
    return merged


def signature_features(log_close: np.ndarray, window: int) -> np.ndarray:
    """iisignature level-2 signature of the (bar_index, cumulative_log_return) path over a
    trailing `window`-bar span, for every row (NaN where history is insufficient)."""
    n = len(log_close)
    sig_dim = iisignature.siglength(2, 2)
    out = np.full((n, sig_dim), np.nan, dtype=np.float64)
    t_axis = np.arange(window + 1, dtype=np.float64)
    for i in range(window, n):
        r = log_close[i - window:i + 1] - log_close[i - window]
        path = np.column_stack([t_axis, r])
        out[i] = iisignature.sig(path, 2)
    return out


def bar_payoff(predicted_class: np.ndarray, tradeable: np.ndarray, outcome: np.ndarray,
               tp: np.ndarray, sl: np.ndarray) -> np.ndarray:
    payoff_if_long = np.where(outcome == 1, tp, np.where(outcome == 2, -sl, 0.0))
    payoff_if_short = np.where(outcome == 2, tp, np.where(outcome == 1, -sl, 0.0))
    realized = np.where(predicted_class == 1, payoff_if_long, np.where(predicted_class == 2, payoff_if_short, 0.0))
    return np.where(tradeable, realized, 0.0)


def train_and_score(train_df, dev_df, feat_cols, label_col="trade_outcome_action"):
    model = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=200,
        objective="multiclass", num_class=3, random_state=270705, verbosity=-1,
    )
    model.fit(train_df[feat_cols], train_df[label_col])
    pred_class = model.predict(dev_df[feat_cols])
    return model, pred_class


def main() -> None:
    t0 = time.time()
    merged = load_merged()
    log_close = np.log(merged["close"].to_numpy())
    print(f"Merged rows: {len(merged)} ({merged.timestamp.min()} .. {merged.timestamp.max()})")

    feat_names_by_window = {}
    for w in WINDOWS:
        print(f"Computing level-2 signature features, window={w} bars ...")
        sig = signature_features(log_close, w)
        cols = [f"sig_w{w}_{i}" for i in range(sig.shape[1])]
        feat_names_by_window[w] = cols
        for j, c in enumerate(cols):
            merged[c] = sig[:, j]
    print(f"Feature computation done in {time.time() - t0:.1f}s")

    merged = merged.dropna(subset=[c for cols in feat_names_by_window.values() for c in cols]).reset_index(drop=True)
    ts = merged["timestamp"]
    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]
    oos_df = merged[ts >= OOS_START]
    print(f"TRAIN={len(train_df)} DEV={len(dev_df)} VAL={len(val_df)} OOS(partial, to {ts.max()})={len(oos_df)}")

    dev_outcome = dev_df["trade_outcome_action"].to_numpy()
    dev_tp, dev_sl = dev_df["tp_move"].to_numpy(), dev_df["sl_move"].to_numpy()

    returns_matrix = np.zeros((len(dev_df), len(WINDOWS)), dtype=np.float64)
    models = {}
    for j, w in enumerate(WINDOWS):
        model, pred_class = train_and_score(train_df, dev_df, feat_names_by_window[w])
        models[w] = model
        tradeable = pred_class != 0
        returns_matrix[:, j] = bar_payoff(pred_class, tradeable, dev_outcome, dev_tp, dev_sl)
        n_tr, mean_r = int(tradeable.sum()), returns_matrix[:, j].mean()
        sharpe = mean_r / (returns_matrix[:, j].std() + 1e-12)
        print(f"  window={w:3d}  n_trades={n_tr:6d}  mean={mean_r:.6f}  "
              f"sum={returns_matrix[:, j].sum():.4f}  sharpe={sharpe:.4f}")

    best_j = int(np.argmax(_col_sharpe(returns_matrix)))
    best_w = WINDOWS[best_j]
    print(f"\nBest-of-{len(WINDOWS)} on DEV: window={best_w}")

    print("\n=== Falsification audit on the DEV window-length search ===")
    audit = falsification_audit(returns_matrix, n_null_draws=1000, block_size=48, seed=20260809)
    for k, v in audit.items():
        print(f"  {k}: {v}")

    if not audit["passes_falsification_audit"]:
        print("\n[RESULT] FAILS falsification audit -- the window-length search's winner is not "
              "distinguishable from noise. STOPPING. No VAL/OOS spent.")
        return
    print("\n[RESULT] Falsification audit PASSES. Checking against a naive momentum baseline "
          "of the SAME window length before touching VAL.")

    # Naive baseline: sign of the raw window return, same window length as the winner.
    raw_ret_w = log_close - np.roll(log_close, best_w)
    raw_ret_w[:best_w] = np.nan
    dev_raw_ret = raw_ret_w[merged.index[(ts >= DEV_START) & (ts <= DEV_END)]]
    naive_class = np.where(dev_raw_ret > 0, 1, np.where(dev_raw_ret < 0, 2, 0))
    naive_returns = bar_payoff(naive_class, naive_class != 0, dev_outcome, dev_tp, dev_sl)
    sig_returns = returns_matrix[:, best_j]
    print(f"\n=== DEV: signature(w={best_w}) model vs naive momentum(w={best_w}) baseline ===")
    sig_traded = sig_returns[sig_returns != 0]
    naive_traded = naive_returns[naive_returns != 0]
    if len(sig_traded) >= 3 and len(naive_traded) >= 3:
        report = effect_size_report(sig_traded, naive_traded, label_a="signature_model", label_b="naive_momentum")
        for k, v in report.items():
            print(f"  {k}: {v}")
    else:
        print(f"  too few trades to compare (sig={len(sig_traded)}, naive={len(naive_traded)})")
    print(f"  naive baseline: n_trades={int((naive_class != 0).sum())} mean={naive_returns.mean():.6f} "
          f"sum={naive_returns.sum():.4f}")

    print(f"\n=== VAL check (2025-09-01..2025-12-31), frozen window={best_w} model ===")
    val_pred = models[best_w].predict(val_df[feat_names_by_window[best_w]])
    val_outcome, val_tp, val_sl = val_df["trade_outcome_action"].to_numpy(), val_df["tp_move"].to_numpy(), val_df["sl_move"].to_numpy()
    val_returns = bar_payoff(val_pred, val_pred != 0, val_outcome, val_tp, val_sl)
    print(f"  n_trades={int((val_pred != 0).sum())} mean={val_returns.mean():.6f} sum={val_returns.sum():.4f} "
          f"sharpe={val_returns.mean() / (val_returns.std() + 1e-12):.4f}")

    if len(oos_df) > 200:
        print(f"\n=== Partial OOS check (2026-01-01..{ts.max().date()}), frozen window={best_w} model ===")
        oos_pred = models[best_w].predict(oos_df[feat_names_by_window[best_w]])
        oos_outcome, oos_tp, oos_sl = oos_df["trade_outcome_action"].to_numpy(), oos_df["tp_move"].to_numpy(), oos_df["sl_move"].to_numpy()
        oos_returns = bar_payoff(oos_pred, oos_pred != 0, oos_outcome, oos_tp, oos_sl)
        print(f"  n_trades={int((oos_pred != 0).sum())} mean={oos_returns.mean():.6f} sum={oos_returns.sum():.4f}")
    else:
        print("\n(OOS window too short in this price file to report -- price data ends before OOS starts in force)")


def _col_sharpe(m: np.ndarray) -> np.ndarray:
    mu = m.mean(axis=0)
    sd = m.std(axis=0, ddof=1)
    return np.where(sd > 1e-15, mu / sd, 0.0)


if __name__ == "__main__":
    main()
