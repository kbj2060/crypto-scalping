#!/usr/bin/env python3
"""BTC v3 Stage 3: quality classifier on the Stage 1 sparse event dataset, with genuine purged
walk-forward OOF (never same-model-rescores-own-train-rows, the exact problem
docs/model_contracts/btc_v1_deep_analysis_20260714.md found in the old risk sidecar's "OOF").

For each expanding walk-forward fold (quarterly, 1-day embargo, same style as
scripts/btc_v3_walkforward_harness_20260714.py), fits a 5-seed HistGradientBoostingClassifier
ensemble on events strictly before the fold's train_end, then scores ONLY events in that fold's own
test window -- those test-window predictions are genuine out-of-fold. Concatenating every fold's
test-window predictions gives an OOF probability for most of the sparse event history without any
row ever being scored by a model that saw it during training.

Also runs the confidence-quintile diagnostic the deep analysis used to show the OLD model had zero
separation power (28-30% precision across quintiles regardless of confidence) -- this is the direct
test of whether Stage 1+3 actually fixed that problem, not just a description that they "should."

Enforces docs/model_contracts/btc_v3_holdout_policy_20260714.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714/sparse_event_dataset.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_quality_classifier_20260714"
HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
SEEDS = (270705, 270710, 270715, 270720, 270725)

NON_FEATURE = {
    "event_hour_timestamp", "entry_available_timestamp", "side", "ts_t_value",
    "trade_return", "win", "hold_bars_5m", "exit_reason",
}


def _generate_folds(df: pd.DataFrame, *, fold_months: int, embargo_days: int) -> list[tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    ts = df["entry_available_timestamp"]
    start = ts.min().normalize() + pd.DateOffset(months=12)  # need >=1yr warm-start before first fold
    end = ts.max()
    folds = []
    cursor = start
    idx = 0
    while True:
        train_end = cursor
        test_start = train_end + pd.Timedelta(days=embargo_days)
        test_end = test_start + pd.DateOffset(months=fold_months) - pd.Timedelta(minutes=5)
        if test_end > end:
            break
        if test_end >= HOLDOUT_START:
            raise RuntimeError(f"fold test_end={test_end} would cross HOLDOUT_START={HOLDOUT_START}")
        fold_id = chr(ord("A") + idx) if idx < 26 else f"F{idx}"
        folds.append((fold_id, train_end, test_start, test_end))
        cursor = cursor + pd.DateOffset(months=fold_months)
        idx += 1
    return folds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET_PATH).sort_values("entry_available_timestamp").reset_index(drop=True)
    if df["entry_available_timestamp"].max() >= HOLDOUT_START:
        raise RuntimeError("dataset contains rows at/after HOLDOUT_START -- refusing")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    feature_cols = [c for c in feature_cols if c != "timestamp"]
    print(f"dataset rows={len(df)} features={len(feature_cols)}", flush=True)

    folds = _generate_folds(df, fold_months=3, embargo_days=1)
    print(f"folds={len(folds)}", flush=True)

    oof_records = []
    for fold_id, train_end, test_start, test_end in folds:
        train_mask = df["entry_available_timestamp"] <= train_end
        test_mask = (df["entry_available_timestamp"] >= test_start) & (df["entry_available_timestamp"] <= test_end)
        if train_mask.sum() < 100 or test_mask.sum() == 0:
            print(f"fold={fold_id} skipped (train_n={int(train_mask.sum())} test_n={int(test_mask.sum())})", flush=True)
            continue
        x_train = df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
        y_train = df.loc[train_mask, "win"].to_numpy(dtype=np.int64)
        x_test = df.loc[test_mask, feature_cols].to_numpy(dtype=np.float64)

        probs = np.zeros(len(x_test), dtype=np.float64)
        for seed in SEEDS:
            model = HistGradientBoostingClassifier(
                loss="log_loss", learning_rate=0.05, max_iter=200, max_depth=4,
                l2_regularization=1.0, max_leaf_nodes=31, min_samples_leaf=30,
                early_stopping=False, random_state=int(seed), class_weight="balanced",
            )
            model.fit(x_train, y_train)
            probs += model.predict_proba(x_test)[:, 1]
        probs /= len(SEEDS)

        fold_df = df.loc[test_mask, ["entry_available_timestamp", "side", "trade_return", "win"]].copy()
        fold_df["oof_quality_prob"] = probs
        fold_df["fold"] = fold_id
        fold_df["train_n"] = int(train_mask.sum())
        oof_records.append(fold_df)
        print(f"fold={fold_id} train_end={train_end.date()} test=[{test_start.date()}..{test_end.date()}] "
              f"train_n={int(train_mask.sum())} test_n={len(fold_df)}", flush=True)

    oof = pd.concat(oof_records, ignore_index=True)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    print(f"\ntotal OOF-scored events: {len(oof)} / {len(df)} ({len(oof) / len(df):.1%} of dataset)", flush=True)

    # Confidence-quintile diagnostic (the exact test that showed the OLD model had zero separation power)
    oof["quintile"] = pd.qcut(oof["oof_quality_prob"], 5, labels=False, duplicates="drop")
    quintile_stats = oof.groupby("quintile").agg(
        n=("win", "size"), win_rate=("win", "mean"), mean_trade_return=("trade_return", "mean"),
        mean_pred_prob=("oof_quality_prob", "mean"),
    ).reset_index()
    print("\n=== confidence quintile diagnostic (OOF) ===", flush=True)
    print(quintile_stats.to_string(index=False), flush=True)

    # Baseline: take-all vs threshold-filtered aggregate trade_return
    baseline_mean = float(oof["trade_return"].mean())
    baseline_n = len(oof)
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_rows = []
    for th in thresholds:
        sub = oof[oof["oof_quality_prob"] >= th]
        if len(sub) == 0:
            continue
        threshold_rows.append({
            "threshold": th, "n": len(sub), "pct_of_baseline": len(sub) / baseline_n,
            "win_rate": float(sub["win"].mean()), "mean_trade_return": float(sub["trade_return"].mean()),
            "sum_trade_return": float(sub["trade_return"].sum()),
        })
    print("\n=== threshold filter effect (OOF) ===", flush=True)
    print(f"baseline (take all): n={baseline_n} mean_trade_return={baseline_mean * 100:.3f}% sum={oof['trade_return'].sum() * 100:.2f}%", flush=True)
    for row in threshold_rows:
        print(f"  th={row['threshold']:.2f} n={row['n']:4d} ({row['pct_of_baseline']:.1%}) "
              f"win_rate={row['win_rate']:.1%} mean_return={row['mean_trade_return'] * 100:+.3f}% "
              f"sum_return={row['sum_trade_return'] * 100:+.2f}%", flush=True)

    report = {
        "dataset_rows": len(df), "oof_scored_events": len(oof), "n_folds": len(folds),
        "quintile_diagnostic": quintile_stats.to_dict(orient="records"),
        "baseline_mean_trade_return_pct": baseline_mean * 100,
        "threshold_filter_effect": threshold_rows,
        "holdout_start": str(HOLDOUT_START),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nsaved report -> {OUT_DIR / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
