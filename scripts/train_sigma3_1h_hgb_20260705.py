#!/usr/bin/env python3
"""Sigma3 model: HistGradientBoosting 3-class direction classifier on 1h trend-scanning labels.

Model-family change from all prior attempts (which were neural TabM/GRU and overfit badly):
gradient-boosted trees are far more robust to overfitting on tabular financial features and
cannot memorize sequences the way the GRUs did. Multi-horizon features (returns over 1..24 bars)
already encode temporal context, so a per-bar tree model does not need a sequence encoder.

Trains on 2024-01..2025-06 (18 months), emits a run_variant-compatible decision tape covering
2025-06-25 (context) .. 2026-06-30 per seed, so the existing replay/gate infrastructure scores
it. Two seeds for the pre-registered sign-consistency gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_trendscan_20260705"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/sigma3_1h_hgb_20260705"

TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
TAPE_START = pd.Timestamp("2025-06-25")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}


def load_all() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f"sigma3_1h_{y}.parquet") for y in (2024, 2025, 2026)]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=270705)
    ap.add_argument("--suffix", required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"features: {len(feat_cols)}", flush=True)

    train_mask = df["timestamp"] <= TRAIN_END
    Xtr = df.loc[train_mask, feat_cols].to_numpy(dtype=np.float64)
    ytr = df.loc[train_mask, "ts_action"].to_numpy(dtype=np.int64)
    # sample weight = |t-value| so stronger, cleaner trends dominate training (down-weights weak
    # near-threshold labels). Confidence-weighted training, no lookahead beyond the label itself.
    w = np.clip(np.abs(df.loc[train_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    print(f"train rows: {len(Xtr)}, label dist: {np.bincount(ytr, minlength=3).tolist()}", flush=True)

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.03,
        max_iter=400,
        max_depth=4,
        l2_regularization=1.0,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=int(args.seed),
        class_weight="balanced",
    )
    clf.fit(Xtr, ytr, sample_weight=w)
    print(f"iters: {clf.n_iter_}", flush=True)

    # Predict proba over the full frame; classes_ order may not be [0,1,2] -> remap
    proba_all = clf.predict_proba(df[feat_cols].to_numpy(dtype=np.float64))
    cls = list(clf.classes_)
    col_for = {c: i for i, c in enumerate(cls)}
    p_cash = proba_all[:, col_for[0]] if 0 in col_for else np.zeros(len(df))
    p_long = proba_all[:, col_for[1]] if 1 in col_for else np.zeros(len(df))
    p_short = proba_all[:, col_for[2]] if 2 in col_for else np.zeros(len(df))

    tape_mask = df["timestamp"] >= TAPE_START
    sub = df.loc[tape_mask].reset_index(drop=True)
    pc, pl, ps = p_cash[tape_mask.to_numpy()], p_long[tape_mask.to_numpy()], p_short[tape_mask.to_numpy()]
    probs = np.column_stack([pc, pl, ps])
    dir_action = probs.argmax(axis=1)
    qual = np.where(dir_action > 0, probs[np.arange(len(sub)), dir_action], probs[:, 0])
    DEFAULT_THR = 0.45
    final_action = np.where((dir_action != 0) & (qual >= DEFAULT_THR), dir_action, 0)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0))

    tape = pd.DataFrame({
        "i": np.arange(len(sub)),
        "timestamp": sub["timestamp"],
        "open": sub["open"].astype(float), "high": sub["high"].astype(float),
        "low": sub["low"].astype(float), "close": sub["close"].astype(float),
        "jump_flag": 0.0, "evt_tail_flag": 0.0, "jump_z": 0.0,
        "atr_pct": sub["atr_pct"].astype(float),
        "primary_action": final_action, "primary_side": side,
        "primary_expert": "sigma3", "primary_route_confidence": 1.0, "primary_route_margin": 1.0,
        "primary_dir_p_cash": pc, "primary_dir_p_long": pl, "primary_dir_p_short": ps,
        "primary_quality_p_cash": pc, "primary_quality_p_long": pl, "primary_quality_p_short": ps,
        "primary_quality_score": np.where(final_action != 0, qual, 0.0),
        "primary_confidence": probs.max(axis=1),
        "fallback_action": 0, "fallback_side": 0, "fallback_expert": "none",
        "fallback_route_confidence": 0.0, "fallback_route_margin": 0.0,
        "fallback_dir_p_cash": 1.0, "fallback_dir_p_long": 0.0, "fallback_dir_p_short": 0.0,
        "fallback_quality_p_cash": 1.0, "fallback_quality_p_long": 0.0, "fallback_quality_p_short": 0.0,
        "fallback_quality_score": 0.0, "fallback_confidence": 0.0,
    })
    out_path = OUT_DIR / f"tape_{args.suffix}.parquet"
    tape.to_parquet(out_path, index=False)
    print(f"wrote tape {len(tape)} rows ({tape['timestamp'].min()}..{tape['timestamp'].max()}) -> {out_path}", flush=True)
    print(f"primary_side nonzero pct: {(tape['primary_side'] != 0).mean():.3f}, atr_pct median: {tape['atr_pct'].median():.5f}", flush=True)
    (OUT_DIR / f"report_{args.suffix}.json").write_text(json.dumps({
        "seed": int(args.seed), "n_features": len(feat_cols), "iters": int(clf.n_iter_),
        "train_rows": int(len(Xtr)), "tape_rows": int(len(tape)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
