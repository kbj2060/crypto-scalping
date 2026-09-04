#!/usr/bin/env python3
"""TabPFN confirmation of the persistence-variant GBM screen
(research_eth_liquidity_sweep_topdown_persistence_variant_20260830.py) -- every persistence
variant (giveback-ratio and smoothed-majority) scored WORSE than the deployed touch-only baseline
under GBM, monotonically worse as the persistence requirement got stricter. Confirms with the real
model (4 seeds) on the closest contender (smooth_majority_last3, mildest reduction) vs baseline,
VAL/OOS only -- HOLDOUT stays untouched (already spent on the deployed model, not re-opened for
a purely confirmatory check that isn't going to change the live model either way).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_liquidity_sweep_topdown_persistence_variant_20260830 import build_base_fires  # noqa: E402
from research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830 import load_klines  # noqa: E402
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SEEDS = [20260829, 141592, 271828, 577215]


def log(msg: str) -> None:
    print(f"[liq_sweep_persistence_tabpfn_confirm] {msg}", flush=True)


def tabpfn_panel(train, val, oos, hit_col, tag):
    from tabpfn import TabPFNClassifier
    val_aucs, oos_aucs = [], []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURE_COLUMNS], train[hit_col].to_numpy().astype(int))
        val_auc = roc_auc_score(val[hit_col].to_numpy().astype(int), clf.predict_proba(val[FEATURE_COLUMNS])[:, 1])
        oos_auc = roc_auc_score(oos[hit_col].to_numpy().astype(int), clf.predict_proba(oos[FEATURE_COLUMNS])[:, 1])
        val_aucs.append(val_auc); oos_aucs.append(oos_auc)
        log(f"  [{tag}] seed={seed}: VAL={val_auc:.4f} OOS={oos_auc:.4f}")
    log(f"  [{tag}] VAL {np.mean(val_aucs):.4f}+/-{np.std(val_aucs, ddof=1):.4f}  "
        f"OOS {np.mean(oos_aucs):.4f}+/-{np.std(oos_aucs, ddof=1):.4f}")
    return np.mean(val_aucs), np.mean(oos_aucs)


def main() -> int:
    klines = load_klines()
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    ind = build_indicator_frame(klines)
    fires = build_base_fires(klines, ind, sig)
    fires["hit_smooth3"] = (fires["touched"].astype(bool) & (fires["smooth_frac_3"] > 0.5)).astype(float)

    for hit_col, tag in [("touched", "baseline_touch_only (DEPLOYED)"), ("hit_smooth3", "smooth_majority_last3")]:
        f = fires.dropna(subset=FEATURE_COLUMNS + [hit_col]).reset_index(drop=True)
        ts = f["timestamp"]
        train = f.loc[ts < VAL_START].reset_index(drop=True)
        val = f.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
        oos = f.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
        log(f"\n=== {tag}: n_train={len(train)} n_val={len(val)} n_oos={len(oos)} hit_rate={f[hit_col].mean():.3f} ===")
        tabpfn_panel(train, val, oos, hit_col, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
