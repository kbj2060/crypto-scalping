#!/usr/bin/env python3
"""Generate per-fire TabPFN predicted probability (of hit=touched 2xATR within 2h) for every VAL
and OOS fire of taker_delta_z_climax v4 -- needed for R:R scaling by model confidence (2026-08-30
user request, following up on the trailing-stop cost-gate exploration). TRAIN-fit only (never
fits on VAL/OOS themselves), 4 seeds averaged for stability (matches this project's established
practice -- seed variance on this task is already known to be tiny, std ~0.0002-0.001).

HOLDOUT (2026-04-01~) deliberately EXCLUDED here -- Fresh-Forward discipline, stays untouched
until the R:R-scaled design is finalized on VAL/OOS and ready for its single confirmation pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import FEATURE_COLUMNS

FIRES_CSV = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv"
OUT_CSV = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/val_oos_proba_20260830.csv"

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]


def main() -> None:
    from tabpfn import TabPFNClassifier

    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    eval_df = fires.loc[(ts >= VAL_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    print(f"train n={len(train)}, eval(VAL+OOS) n={len(eval_df)}")

    probas = np.zeros(len(eval_df))
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURE_COLUMNS], train["hit"].to_numpy().astype(int))
        p = clf.predict_proba(eval_df[FEATURE_COLUMNS])[:, 1]
        probas += p
        print(f"  seed={seed} done, mean_proba={p.mean():.4f}")
    probas /= len(SEEDS)

    eval_df["model_proba"] = probas
    eval_df[["pos", "timestamp", "side", "hit", "model_proba"]].to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}")
    print(f"proba stats: mean={probas.mean():.4f} std={probas.std():.4f} "
          f"min={probas.min():.4f} max={probas.max():.4f}")
    # quick sanity: does higher proba correlate with actual hit rate (calibration check)?
    eval_df["proba_bucket"] = pd.qcut(eval_df["model_proba"], 5, labels=False, duplicates="drop")
    print("\ncalibration check (proba quintile vs actual hit rate):")
    print(eval_df.groupby("proba_bucket").agg(n=("hit", "size"), mean_proba=("model_proba", "mean"), hit_rate=("hit", "mean")))


if __name__ == "__main__":
    main()
