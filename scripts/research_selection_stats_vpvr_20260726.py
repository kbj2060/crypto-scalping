#!/usr/bin/env python3
"""Did the formula search find an edge, or just the biggest number in the noise?

Runs an independent (random, not TPE) parameter search over the same space used
for the promotion attempt, keeping every trial's daily return series. That makes
the selection-bias question answerable rather than rhetorical:

  Deflated Sharpe Ratio  is the winner's Sharpe above what a search this wide
                         produces from a strategy with no edge at all?
  PBO (CSCV)             if you select on half the data, how often does the
                         winner land in the bottom half out of sample?

Random search is used deliberately: the DSR trial-count correction assumes
independent draws, which TPE's guided sampling violates. The point of this
script is a calibration number, so the assumptions have to actually hold.

    python scripts/research_selection_stats_vpvr_20260726.py \
        --data data/training_features_5m.csv --trials 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.selection_stats import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    periodic_returns,
    sharpe,
)
from scripts.backtest_vpvr_poc_rsi_vwma_ema_formula import (  # noqa: E402
    _sample_params,
    load_data,
    run_formula_sim,
    split_train_test,
)

BARS_PER_DAY = 288  # 5m bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/training_features_5m.csv")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=10, help="CSCV chunk count (even)")
    ap.add_argument(
        "--out", default="data/ensemble/metrics/vpvr_selection_stats_20260726.json"
    )
    args = ap.parse_args()

    df = load_data(args.data)
    train_df, test_df = split_train_test(df, float(args.train_ratio))
    print(f"rows total={len(df)} train={len(train_df)} test={len(test_df)}", flush=True)

    rng = np.random.default_rng(args.seed)
    trial_params: list[dict] = []
    trial_daily: list[np.ndarray] = []
    trial_pnl: list[float] = []

    for i in range(args.trials):
        p = _sample_params(rng)
        res, curve = run_formula_sim(
            train_df, p, args.fee_bps, args.slip_bps, args.leverage, return_curve=True
        )
        trial_params.append(p)
        trial_daily.append(periodic_returns(curve, BARS_PER_DAY))
        trial_pnl.append(res.pnl_pct)
        if (i + 1) % 25 == 0:
            print(f"  trial {i + 1}/{args.trials}", flush=True)

    # Rectangular matrix (days x configs) for CSCV.
    n_days = min(len(r) for r in trial_daily)
    matrix = np.column_stack([r[:n_days] for r in trial_daily])
    trial_sharpes = np.array([sharpe(r) for r in trial_daily], dtype=np.float64)

    # The winner is chosen exactly the way the promotion attempt chose it: by
    # total PnL on the training split.
    best_by_pnl = int(np.argmax(trial_pnl))
    best_by_sharpe = int(np.argmax(trial_sharpes))

    dsr_pnl = deflated_sharpe_ratio(trial_daily[best_by_pnl], trial_sharpes)
    dsr_sr = deflated_sharpe_ratio(trial_daily[best_by_sharpe], trial_sharpes)
    pbo = pbo_cscv(matrix, n_splits=args.n_splits)

    # What the winner scores on the untouched holdout, for context.
    holdout = run_formula_sim(
        test_df, trial_params[best_by_pnl], args.fee_bps, args.slip_bps, args.leverage
    )
    holdout_sr = run_formula_sim(
        test_df, trial_params[best_by_sharpe], args.fee_bps, args.slip_bps, args.leverage
    )

    result = {
        "meta": {
            "data": args.data,
            "rows_total": len(df),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "trials": int(args.trials),
            "search": "independent random (DSR assumes independent draws)",
            "fee_bps": args.fee_bps,
            "slip_bps": args.slip_bps,
            "leverage": args.leverage,
            "seed": int(args.seed),
            "return_period": "daily (288 x 5m bars)",
            "days_per_trial": int(n_days),
        },
        "trial_distribution": {
            "sharpe_mean": float(np.mean(trial_sharpes)),
            "sharpe_std": float(np.std(trial_sharpes, ddof=1)),
            "sharpe_max": float(np.max(trial_sharpes)),
            "pnl_pct_max": float(np.max(trial_pnl)),
            "pnl_pct_median": float(np.median(trial_pnl)),
            "frac_trials_profitable": float(np.mean(np.asarray(trial_pnl) > 0)),
        },
        "noise_floor": {
            "expected_max_sharpe_daily": expected_max_sharpe(
                int(args.trials), float(np.std(trial_sharpes, ddof=1))
            ),
            "note": (
                "Daily Sharpe a no-edge strategy is expected to reach as the best "
                "of this many independent trials. A winner below this is noise."
            ),
        },
        "winner_by_train_pnl": {
            "train_pnl_pct": float(trial_pnl[best_by_pnl]),
            "deflated_sharpe": dsr_pnl,
            "holdout_pnl_pct": holdout.pnl_pct,
            "holdout_trades": holdout.trades,
        },
        "winner_by_train_sharpe": {
            "train_pnl_pct": float(trial_pnl[best_by_sharpe]),
            "deflated_sharpe": dsr_sr,
            "holdout_pnl_pct": holdout_sr.pnl_pct,
            "holdout_trades": holdout_sr.trades,
        },
        "pbo_cscv": pbo,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
