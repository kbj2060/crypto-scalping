#!/usr/bin/env python3
"""Analytical follow-up (2026-08-09) to idea #16: what is the kitchen-sink model's raw AUC on
the binary win/loss target (independent of the $ payoff framing that ideas #8-17 all used)? If
AUC is near 0.50 on DEV/VAL, that's the cleanest possible confirmation that these feature
families carry literally no discriminative information about win/loss, separate from whether
that information would have been profitable to act on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_kitchen_sink_skip_filter_eth_20260809 import (  # noqa: E402
    DEV_END, DEV_START, ROUND_TRIP_COST, TRAIN_END, VAL_END, VAL_START,
    build_features, favored_direction, load_merged,
)


def main() -> None:
    merged = load_merged()
    feat_cols = build_features(merged)
    merged = merged.dropna(subset=feat_cols).reset_index(drop=True)
    ts = merged["timestamp"]
    train_df = merged[ts <= TRAIN_END]
    dev_df = merged[(ts >= DEV_START) & (ts <= DEV_END)]
    val_df = merged[(ts >= VAL_START) & (ts <= VAL_END)]

    train_outcome = train_df["trade_outcome_action"].to_numpy()
    direction = favored_direction(train_outcome, train_df["tp_move"].to_numpy(), train_df["sl_move"].to_numpy(), ROUND_TRIP_COST)
    train_win = (train_outcome == direction).astype(int)

    model = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.03,
                               min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                               random_state=270705, verbosity=-1)
    model.fit(train_df[feat_cols], train_win)

    train_auc = roc_auc_score(train_win, model.predict_proba(train_df[feat_cols])[:, 1])
    print(f"TRAIN (in-sample) AUC: {train_auc:.4f}")
    for name, split in [("DEV", dev_df), ("VAL", val_df)]:
        outcome = split["trade_outcome_action"].to_numpy()
        win = (outcome == direction).astype(int)
        prob = model.predict_proba(split[feat_cols])[:, 1]
        auc = roc_auc_score(win, prob)
        print(f"{name}: AUC={auc:.4f}  win_rate={win.mean():.4f}  n={len(win)}")


if __name__ == "__main__":
    main()
