#!/usr/bin/env python3
"""Ablation check for demarker_extreme's permutation-importance result: bb_pctb (Bollinger %B)
scored +0.180, ~5x the next feature (dem itself, +0.038) -- research_eth_kalman_demarker_
permutation_importance_20260831.py. Before touching HOLDOUT, confirm whether "DeMarker extreme"'s
apparent edge is actually mostly a bb_pctb proxy (both are "where is price within its recent
range/volatility band" oscillators, plausibly redundant) by comparing: full feature set, full set
minus bb_pctb, dem alone, and bb_pctb alone. Same logic as taker_delta_z_climax's own precedent
(atr_percentile_864 dominated individually but removing the top-3 vol features only cost ~0.01-
0.012 AUC, confirming it wasn't a pure vol-regime proxy) -- this checks the same question for
demarker_extreme's bb_pctb dominance.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) via handoff.sh.
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
from sklearn.metrics import roc_auc_score

from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    OOS_START,
    VAL_START,
    build_fires,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

SEEDS = [20260829, 141592, 271828, 577215]


def log(msg: str) -> None:
    print(f"[demarker_bb_pctb_ablation] {msg}", flush=True)


def run(train, val, y_train, y_val, cols: list[str], tag: str) -> None:
    from tabpfn import TabPFNClassifier
    aucs = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[cols], y_train)
        proba = clf.predict_proba(val[cols])[:, 1]
        aucs.append(roc_auc_score(y_val, proba))
    log(f"{tag}: VAL AUC = {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1):.4f} (n_feat={len(cols)})")


def main() -> int:
    klines = load_klines()
    ind = build_indicator_frame(klines)
    dem = compute_demarker(klines["high"], klines["low"])
    ind2 = ind.copy()
    ind2["dem"] = dem.to_numpy()
    feature_cols = FEATURE_COLUMNS + ["dem"]

    fires = build_fires(klines, ind2, dem >= 0.90, dem <= 0.10, dem.fillna(0.5).to_numpy(),
                        feature_cols, horizon=8, gap=12, K=0.70)
    fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    y_train = train["hit_plain"].to_numpy().astype(int)
    y_val = val["hit_plain"].to_numpy().astype(int)
    log(f"n_train={len(train)} n_val={len(val)}")

    run(train, val, y_train, y_val, feature_cols, "full 24 features")
    run(train, val, y_train, y_val, [c for c in feature_cols if c != "bb_pctb"], "WITHOUT bb_pctb (23 feat)")
    run(train, val, y_train, y_val, ["dem"], "dem ALONE (1 feat)")
    run(train, val, y_train, y_val, ["bb_pctb"], "bb_pctb ALONE (1 feat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
