#!/usr/bin/env python3
"""Follow-up to research_eth_evidence_signal_k_calibration_leakage_audit_20260831.py's finding:
taker_delta_z_climax v5's ATR_HIT_MULT=2.0 was legitimately calibrated for a ~50/50 split on the
v4 (CLUSTER_GAP_MERGE=3) population ("gives a 50.5%/49.5% split" per that script's own docstring),
but was carried over UNCHANGED into v5 (CLUSTER_GAP_MERGE=12) without recalibration -- on the v5
population, K=2.0 now gives a 58.9% hit rate, not 50%. A fresh recalibration (TRAIN-only and
full-period agree exactly, so this is NOT a future-data leakage issue, just a stale hyperparameter
left over from before the v4->v5 clustering change) gives K=2.4.

This is a DIFFERENT category of issue from orthogonal_combo's exclude-middle/K-leakage findings --
not leakage, just an un-refreshed calibration. Checks whether it matters: retrains TabPFN fresh on
a K=2.4-labeled TRAIN set and re-evaluates VAL/OOS/HOLDOUT AUC, comparing to the currently-reported
0.633/0.645/0.667 (K=2.0). Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
"""
from __future__ import annotations

import json
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

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    FEATURE_COLUMNS, build_indicator_frame, load_klines, run_tabpfn_panel,
)
from research_eth_taker_delta_climax_metalabel_v5_gap12_20260830 import build_fires_and_features

OUT_DIR = ROOT / "tmp/eth_taker_delta_climax_metalabel_20260829"
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
K_CORRECTED = 2.4  # from the leakage-audit script's TRAIN-only AND full-period recalibration (both agree)


def log(msg: str) -> None:
    print(f"[taker_k_recalibration_check] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("rebuilding klines + indicator_frame + v5 fires (unchanged fire-building/clustering logic)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    fires = build_fires_and_features(klines, indicator_frame)  # has pred_dir_ret + atr_pct + current (K=2.0) hit
    fires = fires.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"fires: {len(fires)}")

    move_atr_mult = fires["pred_dir_ret"].to_numpy() / fires["atr_pct"].to_numpy()
    fires["hit_corrected"] = (move_atr_mult >= K_CORRECTED).astype(float)
    log(f"hit-rate @ current K=2.0: {fires['hit'].mean():.4f}   hit-rate @ corrected K={K_CORRECTED}: {fires['hit_corrected'].mean():.4f}")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN={len(train)} VAL={len(val)} OOS={len(oos)} HOLDOUT={len(holdout)}")

    # swap in the corrected label for both TRAIN (what the classifier learns) and each eval split
    for df in (train, val, oos, holdout):
        df["hit"] = df["hit_corrected"]

    log("=== VAL (K=2.4, TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL-K2.4")
    log(f"  VAL(K=2.4)     AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  (orig K=2.0: 0.633)")
    log("=== OOS (K=2.4, TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS-K2.4")
    log(f"  OOS(K=2.4)     AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  (orig K=2.0: 0.645)")
    log("=== HOLDOUT (K=2.4, TRAIN-fit, 4 seeds, single touch -- this HOLDOUT was already spent for K=2.0, reusing for a like-for-like relabel comparison only, not a fresh promotion claim) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT-K2.4")
    log(f"  HOLDOUT(K=2.4) AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}  (orig K=2.0: 0.667)")

    report = {
        "k_current": 2.0, "k_corrected": K_CORRECTED,
        "val": val_result, "oos": oos_result, "holdout": holdout_result,
    }
    out_path = OUT_DIR / "k_recalibration_check_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
