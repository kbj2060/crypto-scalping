#!/usr/bin/env python3
"""Deep audit of orthogonal_combo v2 (user flagged VAL/OOS/HOLDOUT AUC 0.684/0.727/0.725 and
91-96% trailing-stop win rates as suspiciously high -- warrants scrutiny beyond the original
screening). Two independent checks, both requiring TabPFN/CUDA (run on the GPU server):

CHECK 1 -- K_center global-calibration leakage. calibrate_k_center() in the original screening
script (research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py::screen_one_combo) computes the
pooled-50%-hit-rate threshold using fires_raw spanning the ENTIRE 2024-01-01..HOLDOUT-end period --
i.e. the label-defining threshold (which in turn decides WHICH rows even survive exclude-middle
filtering, in EVERY split including VAL/OOS/HOLDOUT) was calibrated using future/eval-period data,
not just TRAIN. This is a real methodological concern distinct from classic feature lookahead (no
single row's own future leaks into its own label) but is still a global-population information
leak into the label DEFINITION itself. This check recalibrates K_center using ONLY TRAIN-period
(< 2025-09-01) fires, rebuilds exclude-middle with that causal threshold, retrains TabPFN fresh
(4 seeds, same as the original), and re-evaluates VAL/OOS/HOLDOUT AUC -- if the original 0.684/
0.727/0.725 was substantially inflated by this leakage, the recalibrated numbers should be
noticeably lower.

CHECK 2 -- exclude-middle inflates the headline by construction (evaluates only the "easy" 64% of
fires whose outcome wasn't ambiguous, in EVERY split, not just TRAIN). This check takes the
ORIGINALLY-trained classifier (fit on the original 956-row exclude-middle TRAIN, K_lo=1.786/
K_hi=3.571) and evaluates it on the FULL (unfiltered) VAL/OOS/HOLDOUT populations -- every raw fire,
including the "ambiguous middle" ones the original evaluation excluded -- using a single-threshold
label (move_atr_mult >= K_center) so every row has a well-defined 0/1 outcome. This answers "how
well does this classifier actually discriminate across ALL real fires, not just the easy subset."

Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
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
from sklearn.metrics import roc_auc_score

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_orthogonal_combo_metalabel_tabpfn_20260830 import (
    FEATURE_COLUMNS, HOLDOUT_START, OOS_START, VAL_START,
    apply_exclude_middle, build_raw_fires, calibrate_k_center, load_funding_z, split_train_val_oos,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import build_indicator_frame, load_klines, run_tabpfn_panel

OUT_DIR = ROOT / "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830"
ORIG_TRAIN_CSV = ROOT / "data/labels/eth_5m_orthogonal_combo_metalabel_20260830/tabpfn_train_context_frozen_orthogonal_combo_20260831.csv"
HORIZON, GAP = 24, 12

# CAUGHT BUG (first run of this script, 2026-08-31): FEATURE_COLUMNS imported above is the FULL
# 23-feature Tier0 list (re-exported pass-through from the taker script) -- NOT the 20-feature
# no-session-timing subset that is orthogonal_combo's actual FINAL ADOPTED model (matches the live-
# serving deployment's ORTHOGONAL_COMBO_FEATURE_COLUMNS in live_evidence_signal_metalabel_20260829.py
# exactly). The first run of this audit silently retrained/reevaluated with the WRONG (23-feature,
# pre-ablation) config throughout -- its numbers matched the ablation script's "full_23_features"
# row (0.7230/0.7162/0.7076) instead of the actual deployed "no_session_timing_20_features" row
# (0.6844/0.7274/0.7245). Fixed here; every TabPFN call below now explicitly uses this 20-feature list.
FINAL_FEATURE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ("nyse_open_flag", "hour_utc", "weekday")]


def log(msg: str) -> None:
    print(f"[orthogonal_deep_audit] {msg}", flush=True)


def main() -> int:
    assert len(FINAL_FEATURE_COLUMNS) == 20, f"expected 20 final features, got {len(FINAL_FEATURE_COLUMNS)}"
    log(f"using FINAL_FEATURE_COLUMNS (20, no session-timing): {FINAL_FEATURE_COLUMNS}")
    log("rebuilding klines + indicator_frame + funding_z + compute_signals (must match original exactly)...")
    klines = load_klines()
    indicator_frame = build_indicator_frame(klines)
    funding_df = load_funding_z()
    sig = compute_signals(klines, btc_df=None, funding_df=funding_df).reset_index(drop=True)

    fires_raw = build_raw_fires(indicator_frame, sig, GAP, HORIZON)
    fires_raw = fires_raw.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"fires_raw (all periods, no exclude-middle): {len(fires_raw)}")
    assert len(fires_raw) == 2334, f"expected 2334, got {len(fires_raw)} -- population mismatch, abort"

    train_only_raw = fires_raw.loc[fires_raw["timestamp"] < VAL_START].reset_index(drop=True)
    k_center_full = calibrate_k_center(fires_raw)
    k_center_train_only = calibrate_k_center(train_only_raw)
    log(f"K_center FULL-PERIOD (original, uses VAL/OOS/HOLDOUT data too): {k_center_full}")
    log(f"K_center TRAIN-ONLY (causal, {len(train_only_raw)} fires before {VAL_START.date()}): {k_center_train_only}")

    report = {"k_center_full_period": k_center_full, "k_center_train_only": k_center_train_only}

    # ================= CHECK 1: recalibrate K using TRAIN-only, retrain+reeval fresh =================
    log("\n=== CHECK 1: retrain/reeval with TRAIN-only-calibrated K (leakage test) ===")
    fires_recal, k_lo_recal, k_hi_recal = apply_exclude_middle(fires_raw, k_center_train_only)
    log(f"recalibrated K_lo={k_lo_recal} K_hi={k_hi_recal} (orig: K_lo=1.786 K_hi=3.571) "
        f"-> kept {len(fires_recal)}/{len(fires_raw)} (orig kept 1493)")

    train_r, val_r, oos_r = split_train_val_oos(fires_recal)
    holdout_r = fires_recal.loc[fires_recal["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"recalibrated split sizes: TRAIN={len(train_r)} VAL={len(val_r)} OOS={len(oos_r)} HOLDOUT={len(holdout_r)} "
        f"(orig: TRAIN=956 VAL=181 OOS=128)")

    val_result_recal = run_tabpfn_panel(train_r, val_r, FINAL_FEATURE_COLUMNS, "VAL-recal")
    log(f"  VAL(recal)     AUC {val_result_recal['auc_mean']:.4f}+/-{val_result_recal['auc_std']:.4f}  (orig: 0.6844+/-0.0017)")
    oos_result_recal = run_tabpfn_panel(train_r, oos_r, FINAL_FEATURE_COLUMNS, "OOS-recal")
    log(f"  OOS(recal)     AUC {oos_result_recal['auc_mean']:.4f}+/-{oos_result_recal['auc_std']:.4f}  (orig: 0.7274+/-0.0012)")
    holdout_result_recal = run_tabpfn_panel(train_r, holdout_r, FINAL_FEATURE_COLUMNS, "HOLDOUT-recal") if len(holdout_r) >= 30 else {"note": "too few"}
    if "auc_mean" in holdout_result_recal:
        log(f"  HOLDOUT(recal) AUC {holdout_result_recal['auc_mean']:.4f}+/-{holdout_result_recal['auc_std']:.4f}  (orig: 0.7245+/-0.0018)")

    report["check1_recalibrated"] = {
        "k_lo": k_lo_recal, "k_hi": k_hi_recal,
        "n_train": len(train_r), "n_val": len(val_r), "n_oos": len(oos_r), "n_holdout": len(holdout_r),
        "val": val_result_recal, "oos": oos_result_recal, "holdout": holdout_result_recal,
    }

    # ================= CHECK 2: original classifier evaluated on the FULL (non-exclude-middle) population =================
    log("\n=== CHECK 2: original classifier (fit on original 956-row exclude-middle TRAIN) vs FULL eval population ===")
    orig_train = pd.read_csv(ORIG_TRAIN_CSV, parse_dates=["timestamp"])
    log(f"original TRAIN context: {len(orig_train)} rows (K_lo=1.786/K_hi=3.571)")

    full_labeled = fires_raw.copy()
    full_labeled["hit"] = (full_labeled["move_atr_mult"] >= k_center_full).astype(int)
    full_val = full_labeled.loc[(full_labeled["timestamp"] >= VAL_START) & (full_labeled["timestamp"] < OOS_START)].reset_index(drop=True)
    full_oos = full_labeled.loc[(full_labeled["timestamp"] >= OOS_START) & (full_labeled["timestamp"] < HOLDOUT_START)].reset_index(drop=True)
    full_holdout = full_labeled.loc[full_labeled["timestamp"] >= HOLDOUT_START].reset_index(drop=True)
    log(f"FULL eval population (every raw fire, kept-or-not): VAL={len(full_val)} OOS={len(full_oos)} HOLDOUT={len(full_holdout)} "
        f"(orig kept-only: VAL=181 OOS=128)")

    full_val_result = run_tabpfn_panel(orig_train, full_val, FINAL_FEATURE_COLUMNS, "VAL-fullpop")
    log(f"  VAL(full pop)     AUC {full_val_result['auc_mean']:.4f}+/-{full_val_result['auc_std']:.4f}  (orig kept-only: 0.6844)")
    full_oos_result = run_tabpfn_panel(orig_train, full_oos, FINAL_FEATURE_COLUMNS, "OOS-fullpop")
    log(f"  OOS(full pop)     AUC {full_oos_result['auc_mean']:.4f}+/-{full_oos_result['auc_std']:.4f}  (orig kept-only: 0.7274)")
    full_holdout_result = run_tabpfn_panel(orig_train, full_holdout, FINAL_FEATURE_COLUMNS, "HOLDOUT-fullpop")
    log(f"  HOLDOUT(full pop) AUC {full_holdout_result['auc_mean']:.4f}+/-{full_holdout_result['auc_std']:.4f}  (orig kept-only: 0.7245)")

    report["check2_full_population"] = {
        "label_definition": f"move_atr_mult >= k_center_full({k_center_full})",
        "n_val": len(full_val), "n_oos": len(full_oos), "n_holdout": len(full_holdout),
        "val": full_val_result, "oos": full_oos_result, "holdout": full_holdout_result,
    }

    out_path = OUT_DIR / "deep_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nfull report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
