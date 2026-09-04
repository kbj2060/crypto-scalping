#!/usr/bin/env python3
"""TabFM confirmation (same 4-seed/panel structure as research_eth_kalman_demarker_tabpfn_
confirm_20260831.py) for the 2 Homer candidates, using the frozen TRAIN-context CSVs that script
already produced (tabpfn_train_context_frozen_<name>_20260831.csv) plus VAL/OOS re-sliced from the
same FIRES CSVs that script wrote -- byte-identical data to the TabPFN run, so this is a fair
head-to-head. HOLDOUT stays untouched (same single-exposure-ever policy).

TabFM (Google Research, released 2026-06-30, https://github.com/google-research/tabfm): in-context
tabular foundation model. TabFMClassifier does its own internal ensembling (n_estimators=32
default) and, per the installed package's real __init__ signature (verified on-server, not from
scraped docs), max_num_rows defaults to None -- no hard row truncation -- so we fit directly on the
full frozen train set per seed, exactly mirroring how TabPFNClassifier is used in the sibling
script, rather than the ~100-row subsample-bagging scheme that secondhand docs implied would be
needed.

License note (user decision 2026-08-31, see AskUserQuestion answer "live 교체까지 검토"): TabFM's
default pretrained weights are distributed under `tabfm-non-commercial-v1.0` -- commercial/
production use is explicitly NOT permitted per the repo's own README. This run is a research-only
VAL/OOS comparison; the report below is flagged license_blocks_live_deployment=true and must NOT be
used to justify a live/production swap without a separate commercial license or a different
(commercially-licensed) model.

Runs on the GPU server in an ISOLATED conda env (tabfm_test), deliberately NOT quant_ai (which
TabPFN depends on) -- TabFM pins its own torch/python>=3.11 and installing it into the shared
live-relevant env risked a version conflict. See tmp/tabfm_confirm_run_20260831.sh.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Hardcoded rather than imported: the natural imports (research_eth_kalman_demarker_gridscreen_
# 20260831 / research_eth_taker_delta_climax_metalabel_tabpfn_20260829) transitively pull in
# core.binance_client (via core/__init__.py) which needs the `binance` package -- not installed in
# this isolated tabfm_test env by design (kept minimal so it can never touch quant_ai). Values below
# are copied verbatim from those modules (VAL_START/OOS_START/HOLDOUT_START at research_eth_kalman_
# demarker_gridscreen_20260831.py:54-56, FEATURE_COLUMNS at research_eth_taker_delta_climax_
# metalabel_tabpfn_20260829.py:155-165 -- the audited Tier0 23-feature list).
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
]

DATA_DIR = ROOT / "data/labels/eth_5m_kalman_demarker_metalabel_20260831"
REPORT_DIR = ROOT / "tmp/eth_kalman_demarker_tabfm_confirm_20260831"
SEEDS = [20260829, 141592, 271828, 577215]  # identical seeds to the TabPFN confirm run, for a fair A/B

SIGNAL_CONFIG = {
    "demarker_extreme": {
        "horizon": 8, "gap": 12, "K": 0.70,
        "fires_csv": DATA_DIR / "eth_5m_demarker_extreme_metalabel_features_H8_GAP12_K0.7.csv",
        "frozen_train_csv": DATA_DIR / "tabpfn_train_context_frozen_demarker_extreme_20260831.csv",
        "feature_cols": FEATURE_COLUMNS + ["dem"],
    },
    "kalman_deviation_meanrev": {
        "horizon": 12, "gap": 12, "K": 2.5,
        "fires_csv": DATA_DIR / "eth_5m_kalman_deviation_meanrev_metalabel_features_H12_GAP12_K2.5.csv",
        "frozen_train_csv": DATA_DIR / "tabpfn_train_context_frozen_kalman_deviation_meanrev_20260831.csv",
        "feature_cols": FEATURE_COLUMNS + ["kalman_dev_z"],
    },
}


def log(msg: str) -> None:
    print(f"[kalman_demarker_tabfm_confirm] {msg}", flush=True)


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabfm_panel(model, train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list, tag: str) -> dict:
    from tabfm import TabFMClassifier
    seed_rows = []
    for seed in SEEDS:
        t0 = time.time()
        clf = TabFMClassifier(model=model, random_state=seed)
        clf.fit(train[feature_cols].to_numpy(dtype=float), train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols].to_numpy(dtype=float))[:, 1]
        elapsed = time.time() - t0
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        r["elapsed_sec"] = round(elapsed, 1)
        seed_rows.append(r)
        log(f"    [{tag}] seed={seed}: auc={r['auc']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
            f"(naive={r['naive_majority_accuracy']:.4f}) elapsed={elapsed:.1f}s")
    table = pd.DataFrame(seed_rows)
    return {"n_eval": int(len(eval_df)), "auc_mean": round(float(table["auc"].mean()), 4),
            "auc_std": round(float(table["auc"].std(ddof=1)), 4),
            "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
            "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"], "per_seed": seed_rows}


def confirm_signal(model, name: str, cfg: dict) -> dict:
    horizon, gap, K = cfg["horizon"], cfg["gap"], cfg["K"]
    feature_cols = cfg["feature_cols"]
    log(f"\n=== {name} (H={horizon}, GAP={gap}, K={K}) ===")

    train = pd.read_csv(cfg["frozen_train_csv"], parse_dates=["timestamp"])
    fires = pd.read_csv(cfg["fires_csv"], parse_dates=["timestamp"])
    fires = fires.dropna(subset=feature_cols + ["hit_plain"]).reset_index(drop=True)
    fires = fires.rename(columns={"hit_plain": "hit"})
    ts = fires["timestamp"]
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    log(f"  n_train={len(train)}(pos={int(train['hit'].sum())}) "
        f"n_val={len(val)}(pos={int(val['hit'].sum())}) n_oos={len(oos)}(pos={int(oos['hit'].sum())}) "
        f"(HOLDOUT n={len(fires.loc[ts >= HOLDOUT_START])}, NOT touched)")

    val_result = run_tabfm_panel(model, train, val, feature_cols, "VAL")
    oos_result = run_tabfm_panel(model, train, oos, feature_cols, "OOS")
    val_oos_gap = round(abs(val_result["auc_mean"] - oos_result["auc_mean"]), 4)
    min_val_oos = round(min(val_result["auc_mean"], oos_result["auc_mean"]), 4)
    log(f"  VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"min={min_val_oos:.4f} |gap|={val_oos_gap:.4f}")

    return {"signal": name, "horizon": horizon, "gap": gap, "K": K,
            "n_train": len(train), "n_val": len(val), "n_oos": len(oos),
            "val": val_result, "oos": oos_result, "val_oos_gap": val_oos_gap, "min_val_oos": min_val_oos}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    from tabfm import tabfm_v1_0_0_pytorch as backend
    log("loading TabFM pretrained weights (tabfm-non-commercial-v1.0 license, research-only use)...")
    t0 = time.time()
    # backend.load()'s device defaults to None (-> CPU) -- the first attempt at this run silently
    # ran on CPU (486% CPU, 0 GPU, stalled >7min on a single fit) instead of the cuda:13.0 GPU this
    # box has (confirmed available in quant_ai; TabPFN's sibling script requests device="cuda"
    # explicitly for the same reason). Force it here too for a fair, GPU-comparable timing.
    model = backend.load(device="cuda")
    log(f"model loaded in {time.time() - t0:.1f}s")

    results = [confirm_signal(model, name, cfg) for name, cfg in SIGNAL_CONFIG.items()]

    log("\n=== SUMMARY ===")
    for r in results:
        log(f"  {r['signal']}: VAL={r['val']['auc_mean']:.4f}+/-{r['val']['auc_std']:.4f} "
            f"OOS={r['oos']['auc_mean']:.4f}+/-{r['oos']['auc_std']:.4f} min={r['min_val_oos']:.4f}")

    out_path = REPORT_DIR / "tabfm_confirm_report.json"
    out_path.write_text(json.dumps({
        "model": "google/tabfm-1.0.0-pytorch",
        "weights_license": "tabfm-non-commercial-v1.0",
        "license_blocks_live_deployment": True,
        "purpose": "research-only VAL/OOS comparison vs TabPFN, NOT a live-promotion input",
        "seeds": SEEDS,
        "results": results,
    }, indent=2, default=str))
    log(f"\nreport saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
