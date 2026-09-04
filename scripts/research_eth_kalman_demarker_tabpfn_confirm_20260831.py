#!/usr/bin/env python3
"""TabPFN confirmation (4 seeds) for the 2 Homer candidates at their GBM-grid-confirmed
HORIZON/GAP/K (research_eth_kalman_demarker_gridscreen_20260831.py +
research_eth_kalman_demarker_ksweep_20260831.py, see memory eth_kalman_demarker_horizon_gap_k_
screening_20260831 for the full sequential-refinement history):
  demarker_extreme:         HORIZON=8,  GAP=12, K=0.70
  kalman_deviation_meanrev: HORIZON=12, GAP=12, K=2.5
Both use the plain touch-only hit definition (v2 ambiguous-middle exclusion was tried and dropped
-- GBM re-check showed it doesn't beat plain at either signal's optimum, see the memory above).

Same seeds/panel pattern as every other TabPFN confirmation in this lineage (research_eth_
liquidity_sweep_topdown_metalabel_ksweep_tabpfn_confirm_20260830.py). VAL/OOS only -- HOLDOUT stays
untouched (single-exposure-ever policy, this repo's Fresh-Forward convention).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) via handoff.sh.
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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from research_eth_candidate_pool_raw_lift_check_20260831 import (  # noqa: E402
    kalman_level_and_velocity,
    rolling_zscore,
)
from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_eth_kalman_demarker_gridscreen_20260831 import (  # noqa: E402
    HOLDOUT_START,
    OOS_START,
    VAL_START,
    build_fires,
    load_klines,
)
from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (  # noqa: E402
    FEATURE_COLUMNS,
    build_indicator_frame,
)

OUT_DIR = ROOT / "data/labels/eth_5m_kalman_demarker_metalabel_20260831"
REPORT_DIR = ROOT / "tmp/eth_kalman_demarker_tabpfn_confirm_20260831"
SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds used throughout this lineage

SIGNAL_CONFIG = {
    "demarker_extreme": {"horizon": 8, "gap": 12, "K": 0.70},
    "kalman_deviation_meanrev": {"horizon": 12, "gap": 12, "K": 2.5},
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_tabpfn_confirm] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit_plain"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit_plain"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"    [{tag}] seed={seed}: auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
            f"(naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {"n_eval": int(len(eval_df)), "auc_mean": round(float(table["auc"].mean()), 4),
            "auc_std": round(float(table["auc"].std(ddof=1)), 4),
            "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
            "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"], "per_seed": seed_rows}


def confirm_signal(name: str, klines: pd.DataFrame, ind: pd.DataFrame, trigger_top: pd.Series,
                    trigger_bottom: pd.Series, extremeness: np.ndarray, feature_cols: list[str]) -> dict:
    cfg = SIGNAL_CONFIG[name]
    horizon, gap, K = cfg["horizon"], cfg["gap"], cfg["K"]
    log(f"\n=== {name} (H={horizon}, GAP={gap}, K={K}) ===")

    fires = build_fires(klines, ind, trigger_top, trigger_bottom, extremeness, feature_cols, horizon, gap, K)
    fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    log(f"  hit_rate={fires['hit_plain'].mean():.3f}  n_train={len(train)}(pos={int(train['hit_plain'].sum())}) "
        f"n_val={len(val)}(pos={int(val['hit_plain'].sum())}) n_oos={len(oos)}(pos={int(oos['hit_plain'].sum())}) "
        f"(HOLDOUT n={len(fires.loc[ts >= HOLDOUT_START])}, NOT touched)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fires.to_csv(OUT_DIR / f"eth_5m_{name}_metalabel_features_H{horizon}_GAP{gap}_K{K}.csv", index=False)

    val_result = run_tabpfn_panel(train, val, feature_cols, "VAL")
    oos_result = run_tabpfn_panel(train, oos, feature_cols, "OOS")
    val_oos_gap = round(abs(val_result["auc_mean"] - oos_result["auc_mean"]), 4)
    min_val_oos = round(min(val_result["auc_mean"], oos_result["auc_mean"]), 4)
    log(f"  VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"min={min_val_oos:.4f} |gap|={val_oos_gap:.4f}")

    return {"signal": name, "horizon": horizon, "gap": gap, "K": K,
            "hit_rate": round(float(fires["hit_plain"].mean()), 4),
            "n_train": len(train), "n_val": len(val), "n_oos": len(oos),
            "val": val_result, "oos": oos_result, "val_oos_gap": val_oos_gap, "min_val_oos": min_val_oos}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading klines + Tier0 indicator frame...")
    klines = load_klines()
    ind = build_indicator_frame(klines)
    log(f"{len(klines)} bars ready")

    dem = compute_demarker(klines["high"], klines["low"])
    ind_dem = ind.copy()
    ind_dem["dem"] = dem.to_numpy()
    r_dem = confirm_signal("demarker_extreme", klines, ind_dem, dem >= 0.90, dem <= 0.10,
                           dem.fillna(0.5).to_numpy(), FEATURE_COLUMNS + ["dem"])

    levels, _ = kalman_level_and_velocity(klines["close"].to_numpy())
    kalman_dev = pd.Series((klines["close"].to_numpy() - levels) / levels, index=klines.index)
    kalman_dev_z = rolling_zscore(kalman_dev)
    ind_kal = ind.copy()
    ind_kal["kalman_dev_z"] = kalman_dev_z.to_numpy()
    r_kal = confirm_signal("kalman_deviation_meanrev", klines, ind_kal, kalman_dev_z >= 2.0, kalman_dev_z <= -2.0,
                           kalman_dev_z.fillna(0.0).to_numpy(), FEATURE_COLUMNS + ["kalman_dev_z"])

    log("\n=== SUMMARY ===")
    for r in (r_dem, r_kal):
        log(f"  {r['signal']}: H={r['horizon']} GAP={r['gap']} K={r['K']} hit_rate={r['hit_rate']:.3f} "
            f"VAL={r['val']['auc_mean']:.4f}+/-{r['val']['auc_std']:.4f} "
            f"OOS={r['oos']['auc_mean']:.4f}+/-{r['oos']['auc_std']:.4f} min={r['min_val_oos']:.4f}")

    out_path = REPORT_DIR / "tabpfn_confirm_report.json"
    out_path.write_text(json.dumps({"seeds": SEEDS, "results": [r_dem, r_kal]}, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
